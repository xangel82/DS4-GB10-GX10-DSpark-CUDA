/*
 * GB10/GX10 DSpark CUDA modifications:
 * Copyright (c) 2026 Marco Palaferri. Licensed under the MIT License.
 */

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cub/block/block_radix_sort.cuh>

#if defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define DS4_CUDA_HAS_NVTX 1
#endif
#endif
#ifndef DS4_CUDA_HAS_NVTX
#define DS4_CUDA_HAS_NVTX 0
#endif

#include <stdint.h>
#include <stddef.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "ds4_gpu.h"
#include "cuda/indexer/ds4_indexer_sm121.h"
#include "cuda/indexer/ds4_topk_gvr.h"
#include "cuda/indexer/ds4_topk_radix.h"
#include "cuda/mmq/ds4_mmq.h"
#include "cuda/mmq/ds4_repack.h"

static int cuda_nvtx_requested(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *nvtx = getenv("DS4_CUDA_NVTX");
        const char *capture = getenv("DS4_CUDA_NSYS_PREFILL_START_POS");
        const char *captures = getenv("DS4_CUDA_NSYS_PREFILL_START_POSITIONS");
        const char *decode_capture = getenv("DS4_CUDA_NSYS_CAPTURE_START_POS");
        enabled = (nvtx != NULL && strcmp(nvtx, "1") == 0) ||
                  (capture != NULL && capture[0] != '\0') ||
                  (captures != NULL && captures[0] != '\0') ||
                  (decode_capture != NULL && decode_capture[0] != '\0');
    }
    return enabled;
}

static uint64_t cuda_nvtx_payload(uint32_t first, uint32_t second) {
    return ((uint64_t)first << 32) | second;
}

static void cuda_nvtx_push(const char *name, uint64_t payload) {
#if DS4_CUDA_HAS_NVTX
    if (!cuda_nvtx_requested()) return;
    nvtxEventAttributes_t attr = {};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
    attr.payload.ullValue = payload;
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = name;
    (void)nvtxRangePushEx(&attr);
#else
    (void)name;
    (void)payload;
#endif
}

static void cuda_nvtx_pop(void) {
#if DS4_CUDA_HAS_NVTX
    if (cuda_nvtx_requested()) (void)nvtxRangePop();
#endif
}

class cuda_nvtx_scope {
public:
    cuda_nvtx_scope(const char *name, uint64_t payload, bool enabled = true)
        : active_(enabled && cuda_nvtx_requested()) {
        if (active_) cuda_nvtx_push(name, payload);
    }

    ~cuda_nvtx_scope() {
        if (active_) cuda_nvtx_pop();
    }

    cuda_nvtx_scope(const cuda_nvtx_scope &) = delete;
    cuda_nvtx_scope &operator=(const cuda_nvtx_scope &) = delete;

private:
    bool active_;
};

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_QK_K 256
#define DS4_CUDA_UNUSED __attribute__((unused))

enum {
    /* attention_decode_mixed_kernel stores raw-window scores plus visible
     * compressed scores in shared memory.  The host routes larger unmasked
     * decode calls to the online attention kernel so this fixed buffer never
     * becomes an out-of-bounds write at long context. */
    DS4_CUDA_ATTENTION_SCORE_CAP = 8192u,
    DS4_CUDA_ATTENTION_RAW_SCORE_CAP = 256u,
    /* The dynamic decode path pairs two heads only while the selected raw +
     * compressed working set fits in its fixed two-pass shared score tape. */
    DS4_CUDA_ATTN_HEADS2_SCORE_CAP = 768u,
    DS4_CUDA_ATTN_HEADS2_STAGE_ROWS = 16u,
    DS4_CUDA_TOPK_MERGE_GROUP = 8u,
    DS4_CUDA_STREAM_EXPERT_DEFAULT = 8u * 64u,
    DS4_CUDA_STREAM_EXPERT_MAX = 61u * 384u
};

struct ds4_gpu_tensor {
    void *ptr;
    uint64_t bytes;
    int owner;
};

typedef struct {
    uint8_t scales[CUDA_QK_K / 16];
    uint8_t qs[CUDA_QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} cuda_block_q2_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[CUDA_QK_K / 2];
} cuda_block_q4_K;

typedef struct {
    float d;
    int8_t qs[CUDA_QK_K];
    int16_t bsums[CUDA_QK_K / 16];
} cuda_block_q8_K;

typedef struct {
    uint16_t d;
    uint16_t qs[CUDA_QK_K / 8];
} cuda_block_iq2_xxs;

static_assert(sizeof(cuda_block_iq2_xxs) == 66, "unexpected IQ2_XXS block layout");
static_assert(sizeof(cuda_block_q2_K) == 84, "unexpected Q2_K block layout");
static_assert(offsetof(cuda_block_iq2_xxs, d) == 0 &&
              offsetof(cuda_block_iq2_xxs, qs) == 2,
              "unexpected IQ2_XXS field order");
static_assert(offsetof(cuda_block_q2_K, scales) == 0 &&
              offsetof(cuda_block_q2_K, qs) == 16 &&
              offsetof(cuda_block_q2_K, d) == 80 &&
              offsetof(cuda_block_q2_K, dmin) == 82,
              "unexpected Q2_K field order");

#include "ds4_iq2_tables_cuda.inc"

/* The canonical IQ2 grid lives in constant memory for scalar lookup paths.
 * The GB10 decode and small-batch verifier kernels stage a byte-identical
 * global-memory copy into shared memory with coalesced warp loads. */
__device__ static uint64_t cuda_iq2xxs_grid_global[256];

static const void *g_model_host_base;
static const char *g_model_device_base;
static uint64_t g_model_registered_size;
static int g_model_registered;
static int g_model_device_owned;
static int g_model_range_mapping_supported = 1;
static int g_model_hmm_direct;
static int g_model_fd = -1;
static const void *g_model_fd_host_base;
static int g_model_direct_fd = -1;
static uint64_t g_model_direct_align = 1;
static uint64_t g_model_file_size;
static int g_model_cache_full;
static int g_model_mapping_failure_notice_printed;
static cudaStream_t g_model_prefetch_stream;
static cudaStream_t g_model_upload_stream;
static cublasHandle_t g_cublas;
static int g_cublas_ready;
static int g_mmq_prefill_ready;
static int g_mmq_prefill_notice;
static int g_mmq_prefill_fallback_notice;
struct cuda_moe_aligned_range {
    const void *host_base;
    uint64_t offset;
    uint64_t bytes;
    uint32_t kind;
    const char *device_ptr;
    uint64_t in_dim;
    uint64_t out_dim;
    uint32_t group_count;
};
static std::vector<cuda_moe_aligned_range> g_moe_aligned_ranges;
static const void *g_moe_aligned_host_base;
static uint64_t g_moe_aligned_model_size;
static int g_moe_aligned_ready;
static int g_moe_aligned_notice;
static int g_moe_aligned_small_batch_notice;
static int g_moe_complete_fused_notice;
static constexpr uint32_t DS4_MMQ_PREFILL_MIN_TOKENS = 1024u;
static void *g_cublas_workspace;
static uint64_t g_cublas_workspace_bytes;
static int g_quality_mode;
static int g_ssd_streaming_mode;
static int g_attn_heads2_dense_notice;
static int g_attn_heads2_indexed_notice;

/* The imported Entrpi adapter can consume producer-folded Q8_1 activations
 * in its decode paths. This integration calls MMQ only for target prefill,
 * so no activation is published through that optional registry. */
extern "C" int ds4_cuda_q8_fold_take_q81(
        const void *src, uint64_t in_dim, const void **q81) {
    (void)src;
    (void)in_dim;
    if (q81) *q81 = NULL;
    return 0;
}

/* Three compressor topologies, two alternating executable slots (so the
 * background update never touches the graph currently executing), and two
 * output modes: full logits or device argmax. */
enum { DS4_CUDA_TOKEN_GRAPH_VARIANTS = 3 * 2 * 2 };
static cudaGraphExec_t g_token_graph_exec[DS4_CUDA_TOKEN_GRAPH_VARIANTS];
static uint32_t g_token_graph_prepared_pos[DS4_CUDA_TOKEN_GRAPH_VARIANTS];
static unsigned char g_token_graph_prepared_valid[DS4_CUDA_TOKEN_GRAPH_VARIANTS];
static uint32_t g_token_graph_capture_variant;
static uint32_t g_token_graph_capture_pos;
static int g_token_graph_capturing;
static int g_token_graph_prepare_only;
static int g_token_graph_capture_dynamic_token;
static int g_token_graph_warmed;
static int g_token_graph_disabled;
static int g_token_graph_build_notice;
static int g_token_graph_pipeline_notice;
static uint64_t g_token_graph_launches;
static uint64_t g_token_graph_prepared_launches;
static uint64_t g_token_graph_prepares;
static uint64_t g_token_graph_updates;
static uint64_t g_token_graph_rebuilds;
static uint32_t *g_token_graph_token_device;
static uint32_t *g_token_graph_token_host;
static void cuda_token_graph_release(void);

/* Auxiliary speculative graphs must never replace the normal token graphs.
 * MTP uses two eight-slot families.  DSpark gets five verifier and five
 * drafter eight-slot families, one for each K, so an adaptive K change never
 * forces a graph topology rebuild. */
enum { DS4_CUDA_MTP_GRAPH_VARIANTS = 16 + 10 * 8 };
enum {
    DS4_CUDA_MTP_GRAPH_VERIFY_FAMILY = 1u,
    DS4_CUDA_MTP_GRAPH_DRAFT_FAMILY = 2u,
    DS4_CUDA_DSPARK_GRAPH_K1_FAMILY = 4u,
    DS4_CUDA_DSPARK_DRAFT_GRAPH_K1_FAMILY = 128u,
    DS4_CUDA_MTP_GRAPH_ALL_FAMILIES = 4095u
};
static cudaGraphExec_t g_mtp_graph_exec[DS4_CUDA_MTP_GRAPH_VARIANTS];
static uint32_t g_mtp_graph_capture_variant;
static int g_mtp_graph_capturing;
static uint32_t g_mtp_graph_warm_mask;
/* A verifier-specific capture failure must not discard working draft graphs,
 * and vice versa.  Keep one disable bit per graph family. */
static uint32_t g_mtp_graph_disabled_families;
static int g_mtp_graph_notice;
static uint64_t g_mtp_graph_launches;
static uint64_t g_mtp_graph_draft_launches;
static uint64_t g_mtp_graph_verifier_launches;
static uint64_t g_dspark_graph_verifier_launches;
static uint64_t g_dspark_graph_draft_launches;
static uint64_t g_mtp_graph_updates;
static uint64_t g_mtp_graph_rebuilds;
static void cuda_mtp_graph_release(void);
static uint32_t cuda_mtp_graph_family_bit(uint32_t variant) {
    if (variant < 8u) return DS4_CUDA_MTP_GRAPH_VERIFY_FAMILY;
    if (variant < 16u) return DS4_CUDA_MTP_GRAPH_DRAFT_FAMILY;
    if (variant < 56u) {
        const uint32_t k = (variant - 16u) / 8u;
        return DS4_CUDA_DSPARK_GRAPH_K1_FAMILY << k;
    }
    const uint32_t k = (variant - 56u) / 8u;
    return DS4_CUDA_DSPARK_DRAFT_GRAPH_K1_FAMILY << k;
}
static const char *cuda_mtp_graph_family_name(uint32_t variant) {
    if (variant < 8u) return "mtp-verifier";
    if (variant < 16u) return "mtp-draft";
    static const char *verify_names[5] = {
        "dspark-verifier-k1", "dspark-verifier-k2",
        "dspark-verifier-k3", "dspark-verifier-k4",
        "dspark-verifier-k5"
    };
    static const char *draft_names[5] = {
        "dspark-drafter-k1", "dspark-drafter-k2",
        "dspark-drafter-k3", "dspark-drafter-k4",
        "dspark-drafter-k5"
    };
    if (variant < 56u) {
        uint32_t k = (variant - 16u) / 8u;
        return verify_names[k < 5u ? k : 0u];
    }
    uint32_t k = (variant - 56u) / 8u;
    return draft_names[k < 5u ? k : 0u];
}
static void cuda_nsys_capture_stop(const char *reason);
static void cuda_nsys_prefill_capture_stop(const char *reason);
static int cuda_q8_u16_validate(void);
static int cuda_moe_gb10_validate_signs(void);
static int g_q8_u16_validation = -1;
static int g_moe_gb10_sign_validation = -1;
static int g_moe_tiny_direct_notice;
static int g_batched_argmax_notice;

struct cuda_nsys_capture_state {
    int initialized;
    int enabled;
    int started;
    int stopped;
    uint32_t start_pos;
    uint32_t token_limit;
    uint32_t captured_tokens;
};

static cuda_nsys_capture_state g_nsys_capture;
static int g_nsys_decode_cycle_active;

struct cuda_nsys_prefill_capture_state {
    int initialized;
    int enabled;
    int started;
    int active;
    int stopped;
    uint32_t start_positions[16];
    uint32_t start_count;
    uint32_t start_index;
    uint32_t chunk_pos;
    uint32_t chunk_tokens;
};

static cuda_nsys_prefill_capture_state g_nsys_prefill_capture;

struct cuda_token_graph_timing_current {
    int active;
    uint32_t pos;
    double encode_start_sec;
    double capture_begin_ms;
    double host_encode_ms;
    double capture_end_ms;
    double update_ms;
    double rebuild_ms;
    double bookkeeping_ms;
    double launch_submit_ms;
    uint32_t update_ops;
    uint32_t rebuild_ops;
};

struct cuda_token_graph_timing_aggregate {
    uint64_t tokens;
    uint64_t sampling_samples;
    uint64_t gpu_samples;
    uint64_t update_ops;
    uint64_t rebuild_ops;
    uint32_t first_pos;
    uint32_t last_pos;
    double capture_begin_ms;
    double host_encode_ms;
    double capture_end_ms;
    double update_ms;
    double rebuild_ms;
    double bookkeeping_ms;
    double launch_submit_ms;
    double gpu_execute_ms;
    double read_wait_ms;
    double eval_ms;
    double sampling_ms;
};

struct cuda_token_graph_timing_state {
    cudaEvent_t gpu_start;
    cudaEvent_t gpu_stop;
    int events_ready;
    int events_failed;
    int event_pending;
    int sample_pending;
    int notice_printed;
    cuda_token_graph_timing_current current;
    cuda_token_graph_timing_aggregate aggregate;
};

static cuda_token_graph_timing_state g_token_graph_timing;

struct cuda_model_range {
    const void *host_base;
    uint64_t offset;
    uint64_t bytes;
    char *device_ptr;
    void *registered_base;
    char *registered_device_base;
    uint64_t registered_bytes;
    int host_registered;
    int arena_allocated;
};

struct cuda_model_arena {
    char *device_ptr;
    uint64_t bytes;
    uint64_t used;
};

struct cuda_q8_f16_range {
    const void *host_base;
    uint64_t offset;
    uint64_t weight_bytes;
    uint64_t in_dim;
    uint64_t out_dim;
    __half *device_ptr;
};

struct cuda_q8_f32_range {
    const void *host_base;
    uint64_t offset;
    uint64_t weight_bytes;
    uint64_t in_dim;
    uint64_t out_dim;
    float *device_ptr;
};

struct cuda_stream_selected_cache {
    int valid;
    const void *model_map;
    uint32_t layer;
    uint32_t n_total_expert;
    uint32_t n_selected;
    uint32_t slot_count;
    uint32_t compact_count;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    char *gate_ptr;
    char *up_ptr;
    char *down_ptr;
    uint64_t gate_capacity;
    uint64_t up_capacity;
    uint64_t down_capacity;
    int32_t *slot_selected_ptr;
    uint64_t slot_selected_capacity;
    ds4_gpu_tensor slot_selected_tensor;
};

struct cuda_stream_expert_cache_slot {
    int valid;
    const void *model_map;
    uint64_t model_size;
    uint32_t layer;
    uint32_t n_total_expert;
    uint32_t expert;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    uint64_t age;
};

struct cuda_stream_expert_cache {
    int valid;
    uint32_t capacity;
    uint32_t count;
    uint64_t tick;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    char *gate_ptr;
    char *up_ptr;
    char *down_ptr;
    uint64_t gate_capacity;
    uint64_t up_capacity;
    uint64_t down_capacity;
    std::vector<cuda_stream_expert_cache_slot> slots;
};

static std::vector<cuda_model_range> g_model_ranges;
static std::vector<cuda_model_arena> g_model_arenas;
static std::unordered_map<uint64_t, size_t> g_model_range_by_offset;
static std::vector<cuda_q8_f16_range> g_q8_f16_ranges;
static std::unordered_map<uint64_t, size_t> g_q8_f16_by_offset;
static std::vector<cuda_q8_f32_range> g_q8_f32_ranges;
static std::unordered_map<uint64_t, size_t> g_q8_f32_by_offset;
static cuda_stream_selected_cache g_stream_selected_cache;
static cuda_stream_expert_cache g_stream_expert_cache;
static uint32_t g_stream_expert_budget_override;
static uint32_t g_stream_expert_runtime_cap;
static uint32_t g_stream_expert_memory_cap_notice;
static uint64_t g_stream_expert_runtime_gate_bytes;
static uint64_t g_stream_expert_runtime_down_bytes;
static uint64_t g_model_range_bytes;
static uint64_t g_q8_f16_bytes;
static uint64_t g_q8_f32_bytes;
static int g_q8_f16_disabled_after_oom;
static int g_q8_f16_budget_notice_printed;
static uint64_t g_model_load_progress_next;
static uint64_t g_model_load_progress_last_bytes = UINT64_MAX;
static uint64_t g_model_load_progress_last_cgib = UINT64_MAX;
static double g_model_load_progress_last;
static int g_model_load_progress_started;
static int g_model_load_progress_tty;

/* Tensor offsets are only unique inside one GGUF.  Mix the mmap identity into
 * the lookup key so target and MTP caches stay O(1) even when both files use
 * the same local offsets.  Every hit is still validated against host_base. */
static uint64_t cuda_model_offset_key(const void *model_map, uint64_t offset) {
    uint64_t p = (uint64_t)(uintptr_t)model_map;
    p ^= p >> 30;
    p *= 0xbf58476d1ce4e5b9ull;
    p ^= p >> 27;
    p *= 0x94d049bb133111ebull;
    p ^= p >> 31;
    return offset ^ p;
}

static void cuda_moe_aligned_clear(void) {
    g_moe_aligned_ranges.clear();
    g_moe_aligned_host_base = NULL;
    g_moe_aligned_model_size = 0;
    g_moe_aligned_ready = 0;
    g_moe_aligned_notice = 0;
    g_moe_aligned_small_batch_notice = 0;
    g_moe_complete_fused_notice = 0;
}

static const cuda_moe_aligned_range *cuda_moe_aligned_find(
        const void *model_map, uint64_t offset, uint64_t bytes, uint32_t kind) {
    if (!g_moe_aligned_ready || model_map != g_moe_aligned_host_base) return NULL;
    for (const cuda_moe_aligned_range &r : g_moe_aligned_ranges) {
        if (r.host_base == model_map && r.offset == offset &&
            r.bytes == bytes && r.kind == kind) {
            return &r;
        }
    }
    return NULL;
}
static void *g_cuda_tmp;
static uint64_t g_cuda_tmp_bytes;
static void *g_attention_tokentile_scratch;
static uint64_t g_attention_tokentile_scratch_bytes;
static void *g_mtp_tc_scratch;
static uint64_t g_mtp_tc_scratch_bytes;
static void *g_model_stage_raw[4];
static void *g_model_stage[4];
static cudaEvent_t g_model_stage_event[4];
static uint64_t g_model_stage_bytes;
static void *g_stream_selected_stage_raw[4];
static void *g_stream_selected_stage[4];
static cudaEvent_t g_stream_selected_stage_event[4];
static uint64_t g_stream_selected_stage_bytes;
static cudaStream_t g_stream_selected_upload_stream;

static int cuda_ok(cudaError_t err, const char *what);
static const char *cuda_model_range_ptr_from_fd(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes,
        const char *what);
static const char *cuda_model_direct_fallback_ptr(const void *model_map, uint64_t offset);
static int cuda_model_copy_to_device_streamed(
        char *dst,
        const void *model_map,
        uint64_t model_size,
        uint64_t offset,
        uint64_t bytes,
        const char *what);
static uint64_t cuda_model_cache_limit_bytes(void);
static uint64_t cuda_model_local_model_limit_bytes(void);
static int cuda_model_cache_limit_explicit(void);
__global__ static void dequant_q8_0_to_f16_kernel(
        __half *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks);
__global__ static void dequant_q8_0_to_f32_kernel(
        float *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks);

static void *cuda_tmp_alloc(uint64_t bytes, const char *what) {
    if (bytes == 0) return NULL;
    if (g_cuda_tmp_bytes >= bytes) return g_cuda_tmp;
    /* cudaMalloc/cudaFree invalidate stream capture.  A topology that needs a
     * larger arena is replayed normally by the graph caller; that replay grows
     * the arena and the following request can capture the stable pointer. */
    if (g_token_graph_capturing || g_mtp_graph_capturing) {
        static int capture_growth_notice_printed;
        if (!capture_growth_notice_printed) {
            fprintf(stderr,
                    "ds4: CUDA graph scratch growth deferred for %s "
                    "(need=%.2f MiB have=%.2f MiB)\n",
                    what ? what : "scratch",
                    (double)bytes / 1048576.0,
                    (double)g_cuda_tmp_bytes / 1048576.0);
            capture_growth_notice_printed = 1;
        }
        return NULL;
    }
    if (g_cuda_tmp) {
        /* Captured graph nodes retain raw arena pointers.  Destroy stale
         * executables before replacing the allocation. */
        cuda_token_graph_release();
        (void)cudaFree(g_cuda_tmp);
        g_cuda_tmp = NULL;
        g_cuda_tmp_bytes = 0;
    }
    void *ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, (size_t)bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA temp alloc failed for %s (%.2f MiB): %s\n",
                what ? what : "scratch", (double)bytes / 1048576.0, cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }
    g_cuda_tmp = ptr;
    g_cuda_tmp_bytes = bytes;
    return g_cuda_tmp;
}

/* Wide-prefill attention must not resize the shared arena: captured decode
 * graphs retain pointers into it.  This arena is used only outside capture
 * and grows at most when a new prompt high-water mark is observed. */
static void *cuda_attention_tokentile_scratch_alloc(uint64_t bytes) {
    if (bytes == 0) return NULL;
    if (g_attention_tokentile_scratch_bytes >= bytes) {
        return g_attention_tokentile_scratch;
    }
    if (g_token_graph_capturing || g_mtp_graph_capturing) return NULL;
    if (g_attention_tokentile_scratch) {
        if (!cuda_ok(cudaDeviceSynchronize(), "attention token-tile scratch synchronize")) {
            return NULL;
        }
        (void)cudaFree(g_attention_tokentile_scratch);
        g_attention_tokentile_scratch = NULL;
        g_attention_tokentile_scratch_bytes = 0;
    }
    void *ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, (size_t)bytes);
    if (err != cudaSuccess) {
        static int oom_notice_printed;
        if (!oom_notice_printed) {
            fprintf(stderr,
                    "ds4: CUDA token-tile attention scratch unavailable "
                    "(%.2f MiB): %s; using existing attention path\n",
                    (double)bytes / 1048576.0,
                    cudaGetErrorString(err));
            oom_notice_printed = 1;
        }
        (void)cudaGetLastError();
        return NULL;
    }
    g_attention_tokentile_scratch = ptr;
    g_attention_tokentile_scratch_bytes = bytes;
    return ptr;
}

static inline uint64_t cuda_align256_u64(uint64_t x) {
    return (x + 255ull) & ~255ull;
}

/* MTP runs outside the normal token graph, but the ordinary temporary buffer
 * is referenced by captured graph nodes.  Growing that allocation here would
 * free pointers held by existing graph execs.  Keep a separate persistent
 * arena and start large enough to avoid first-cycle resize synchronizations. */
static void *cuda_mtp_tc_scratch_alloc(uint64_t bytes) {
    if (bytes == 0) return NULL;
    if (g_mtp_tc_scratch_bytes >= bytes) return g_mtp_tc_scratch;
    if (g_mtp_graph_capturing) {
        static int capture_growth_notice_printed;
        if (!capture_growth_notice_printed) {
            fprintf(stderr,
                    "ds4: CUDA MTP Tensor Core scratch growth deferred "
                    "(need=%.2f MiB have=%.2f MiB)\n",
                    (double)bytes / 1048576.0,
                    (double)g_mtp_tc_scratch_bytes / 1048576.0);
            capture_growth_notice_printed = 1;
        }
        return NULL;
    }
    uint64_t alloc_bytes = 64ull * 1048576ull;
    while (alloc_bytes < bytes && alloc_bytes <= UINT64_MAX / 2u) {
        alloc_bytes *= 2u;
    }
    if (alloc_bytes < bytes) alloc_bytes = bytes;
    if (g_mtp_tc_scratch) {
        cuda_mtp_graph_release();
        (void)cudaFree(g_mtp_tc_scratch);
        g_mtp_tc_scratch = NULL;
        g_mtp_tc_scratch_bytes = 0;
    }
    void *ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, (size_t)alloc_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "ds4: CUDA MTP Tensor Core scratch alloc failed (%.2f MiB): %s\n",
                (double)alloc_bytes / 1048576.0,
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }
    g_mtp_tc_scratch = ptr;
    g_mtp_tc_scratch_bytes = alloc_bytes;
    return ptr;
}

static int cuda_attention_score_buffer_fits(uint32_t n_comp) {
    return n_comp <= DS4_CUDA_ATTENTION_SCORE_CAP - DS4_CUDA_ATTENTION_RAW_SCORE_CAP;
}

static const char *cuda_model_ptr(const void *model_map, uint64_t offset) {
    if (model_map == g_model_host_base && g_model_device_base) return g_model_device_base + offset;
    return (const char *)model_map + offset;
}

static const char *cuda_model_range_register_mapped(const void *model_map,
                                                    uint64_t offset,
                                                    uint64_t bytes,
                                                    const char *what) {
    if (!g_model_range_mapping_supported || bytes == 0) return NULL;

    const long page_sz_l = sysconf(_SC_PAGESIZE);
    const uint64_t page_sz = page_sz_l > 0 ? (uint64_t)page_sz_l : 4096u;
    const uintptr_t host_addr = (uintptr_t)((const char *)model_map + offset);
    const uintptr_t reg_addr = host_addr & ~(uintptr_t)(page_sz - 1u);
    const uint64_t reg_delta = (uint64_t)(host_addr - reg_addr);
    uint64_t reg_bytes = (reg_delta + bytes + page_sz - 1u) & ~(page_sz - 1u);
    if (model_map == g_model_host_base &&
        g_model_registered_size >= 88ull * 1073741824ull &&
        g_model_registered_size <= 96ull * 1073741824ull &&
        g_model_range_bytes >= 80ull * 1073741824ull) {
        const uintptr_t model_base = (uintptr_t)model_map;
        const uintptr_t model_end = model_base + (uintptr_t)g_model_registered_size;
        if (model_end > model_base && model_end > reg_addr) {
            const uint64_t tail_bytes = (uint64_t)(model_end - reg_addr);
            reg_bytes = (tail_bytes + page_sz - 1u) & ~(page_sz - 1u);
        }
    }
    void *reg_dev = NULL;

    unsigned int flags = cudaHostRegisterMapped | cudaHostRegisterReadOnly;
    if (getenv("DS4_CUDA_HOST_REGISTER_PLAIN") != NULL) {
        flags = cudaHostRegisterMapped;
    }

    cudaError_t err = cudaHostRegister((void *)reg_addr,
                                       (size_t)reg_bytes,
                                       flags);
    if (err != cudaSuccess &&
        (flags & cudaHostRegisterReadOnly) != 0 &&
        (err == cudaErrorNotSupported || err == cudaErrorInvalidValue)) {
        (void)cudaGetLastError();
        err = cudaHostRegister((void *)reg_addr,
                               (size_t)reg_bytes,
                               cudaHostRegisterMapped);
    }
    if (err == cudaSuccess) {
        err = cudaHostGetDevicePointer(&reg_dev, (void *)reg_addr, 0);
        if (err == cudaSuccess && reg_dev) {
            char *dev_ptr = (char *)reg_dev + reg_delta;
            g_model_ranges.push_back({model_map, offset, bytes, dev_ptr, (void *)reg_addr, (char *)reg_dev, reg_bytes, 1, 0});
            g_model_range_by_offset[cuda_model_offset_key(model_map, offset)] =
                g_model_ranges.size() - 1u;
            if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
                fprintf(stderr, "ds4: CUDA mapped %s %.2f MiB\n",
                        what ? what : "weights",
                        (double)bytes / 1048576.0);
            }
            return dev_ptr;
        }
        fprintf(stderr, "ds4: CUDA model range map pointer failed for %s: %s\n",
                what ? what : "weights", cudaGetErrorString(err));
        (void)cudaHostUnregister((void *)reg_addr);
        (void)cudaGetLastError();
        return NULL;
    }

    if (err == cudaErrorNotSupported || err == cudaErrorInvalidValue) {
        g_model_range_mapping_supported = 0;
    }
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr, "ds4: CUDA model range map skipped for %s: %s\n",
                what ? what : "weights", cudaGetErrorString(err));
    }
    (void)cudaGetLastError();
    return NULL;
}

/* Allocate a device-resident copy of [offset, offset+bytes) from model_map and
 * push it into g_model_ranges so future cuda_model_range_ptr lookups hit it.
 * Returns the device pointer on success, NULL on cudaMalloc/cudaMemcpy failure.
 * Caller is responsible for any policy gating (budget cap, env opt-out, etc.) */
static const char *cuda_model_range_populate_device_copy(const void *model_map,
                                                          uint64_t offset,
                                                          uint64_t bytes,
                                                          const char *what) {
    const uint64_t limit = cuda_model_cache_limit_bytes();
    if (g_model_range_bytes > limit || bytes > limit - g_model_range_bytes) {
        if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
            fprintf(stderr, "ds4: CUDA skipped device copy for %s %.2f MiB (cache budget %.2f GiB exhausted)\n",
                    what ? what : "weights",
                    (double)bytes / 1048576.0,
                    (double)limit / 1073741824.0);
        }
        return NULL;
    }

    void *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)bytes);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: CUDA model range alloc failed for %s (%.2f MiB): %s\n",
                what ? what : "weights", (double)bytes / 1048576.0, cudaGetErrorString(err));
        return NULL;
    }

    const char *src = (const char *)model_map + offset;
    const uint64_t chunk = 64ull * 1024ull * 1024ull;
    for (uint64_t done = 0; done < bytes; done += chunk) {
        uint64_t n = bytes - done < chunk ? bytes - done : chunk;
        err = cudaMemcpy((char *)dev + done, src + done, (size_t)n, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model range copy failed for %s at %.2f/%.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)done / 1048576.0,
                    (double)bytes / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaFree(dev);
            (void)cudaGetLastError();
            return NULL;
        }
    }
    g_model_ranges.push_back({model_map, offset, bytes, (char *)dev, NULL, NULL, 0, 0, 0});
    g_model_range_by_offset[cuda_model_offset_key(model_map, offset)] =
        g_model_ranges.size() - 1u;
    g_model_range_bytes += bytes;
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr, "ds4: CUDA cached %s %.2f MiB (total %.2f GiB)\n",
                what ? what : "weights",
                (double)bytes / 1048576.0,
                (double)g_model_range_bytes / 1073741824.0);
    }
    return (const char *)dev;
}

static const char *cuda_model_range_ptr(const void *model_map, uint64_t offset, uint64_t bytes, const char *what) {
    if (bytes == 0) return cuda_model_ptr(model_map, offset);

    /* Device-resident HBM cache hits win over UVA-mapped registered pointers:
     * direct HBM reads are ~10% faster than mapped reads through host page
     * tables (measured on plain decode at GB10).  Cache lookup runs first; the
     * registered-mapped shortcut below is the cold fallback when an allocation
     * hasn't been pre-populated. */
    const uint64_t end = offset + bytes;
    auto exact = g_model_range_by_offset.find(
            cuda_model_offset_key(model_map, offset));
    if (exact != g_model_range_by_offset.end()) {
        const cuda_model_range &r = g_model_ranges[exact->second];
        if (r.host_base == model_map && end >= offset && bytes <= r.bytes) return r.device_ptr;
    }
    for (const cuda_model_range &r : g_model_ranges) {
        if (r.host_base == model_map && offset >= r.offset && end >= offset && end <= r.offset + r.bytes) {
            return r.device_ptr + (offset - r.offset);
        }
        if (r.host_base == model_map && r.host_registered && r.registered_base && r.registered_device_base) {
            const uintptr_t h0 = (uintptr_t)((const char *)model_map + offset);
            const uintptr_t h1 = h0 + bytes;
            const uintptr_t r0 = (uintptr_t)r.registered_base;
            const uintptr_t r1 = r0 + r.registered_bytes;
            if (h1 >= h0 && h0 >= r0 && h1 <= r1) return r.registered_device_base + (h0 - r0);
        }
    }

    if (g_model_device_owned || g_model_registered) return cuda_model_ptr(model_map, offset);
    if (g_model_hmm_direct &&
        getenv("DS4_CUDA_WEIGHT_CACHE") == NULL &&
        getenv("DS4_CUDA_WEIGHT_PRELOAD") == NULL) {
        return cuda_model_ptr(model_map, offset);
    }
    const char *direct_env = getenv("DS4_CUDA_DIRECT_MODEL");
    if (direct_env && direct_env[0]) return cuda_model_ptr(model_map, offset);

    if (getenv("DS4_CUDA_NO_FD_CACHE") == NULL) {
        const char *fd_ptr = cuda_model_range_ptr_from_fd(model_map, offset, bytes, what);
        if (fd_ptr) return fd_ptr;
    }

    const char *mapped = cuda_model_range_register_mapped(model_map, offset, bytes, what);
    if (mapped) return mapped;

    return cuda_model_range_populate_device_copy(model_map, offset, bytes, what);
}

static int cuda_model_range_is_cached(const void *model_map, uint64_t offset, uint64_t bytes) {
    if (bytes == 0) return 1;
    if (g_model_device_owned || g_model_registered || g_model_hmm_direct) return 1;

    const uint64_t end = offset + bytes;
    if (end < offset) return 0;
    for (const cuda_model_range &r : g_model_ranges) {
        if (r.host_base == model_map &&
            offset >= r.offset &&
            end <= r.offset + r.bytes) {
            return 1;
        }
        if (r.host_base == model_map &&
            r.host_registered &&
            r.registered_base &&
            r.registered_device_base) {
            const uintptr_t h0 = (uintptr_t)((const char *)model_map + offset);
            const uintptr_t h1 = h0 + bytes;
            const uintptr_t r0 = (uintptr_t)r.registered_base;
            const uintptr_t r1 = r0 + r.registered_bytes;
            if (h1 >= h0 && h0 >= r0 && h1 <= r1) return 1;
        }
    }
    return 0;
}

static void cuda_q8_f16_cache_release_all(void) {
    for (const cuda_q8_f16_range &r : g_q8_f16_ranges) {
        (void)cudaFree(r.device_ptr);
    }
    g_q8_f16_ranges.clear();
    g_q8_f16_by_offset.clear();
    g_q8_f16_bytes = 0;
}

static uint64_t cuda_parse_mib_env(const char *name, int *present) {
    const char *env = getenv(name);
    if (present) *present = 0;
    if (!env || !env[0]) return 0;
    char *end = NULL;
    unsigned long long v = strtoull(env, &end, 10);
    if (end == env || *end != '\0') return 0;
    if (present) *present = 1;
    if (v > UINT64_MAX / 1048576ull) return UINT64_MAX;
    return (uint64_t)v * 1048576ull;
}

static uint64_t cuda_q8_f16_cache_limit_bytes(void) {
    int present = 0;
    const uint64_t limit = cuda_parse_mib_env("DS4_CUDA_Q8_F16_CACHE_MB", &present);
    return present ? limit : UINT64_MAX;
}

static uint64_t cuda_q8_f16_cache_reserve_bytes(uint64_t total_bytes) {
    int present = 0;
    const uint64_t reserve = cuda_parse_mib_env("DS4_CUDA_Q8_F16_CACHE_RESERVE_MB", &present);
    if (present) return reserve;

    if (total_bytes >= 112ull * 1024ull * 1024ull * 1024ull) {
        return 512ull * 1048576ull;
    }

    /* The expanded Q8->F16 cache is only an acceleration path.  Keep enough
     * device memory free for cuBLAS workspaces, transient graph buffers, and
     * driver bookkeeping instead of letting optional cached weights consume the
     * last few GiB on 96 GiB cards. */
    const uint64_t min_reserve = 4096ull * 1048576ull;
    const uint64_t pct_reserve = total_bytes / 20u; /* 5% */
    return pct_reserve > min_reserve ? pct_reserve : min_reserve;
}

static void cuda_q8_f16_cache_budget_notice(
        const char *reason,
        uint64_t request_bytes,
        uint64_t free_bytes,
        uint64_t total_bytes,
        uint64_t reserve_bytes,
        uint64_t limit_bytes) {
    if (g_q8_f16_budget_notice_printed && getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE") == NULL) return;
    g_q8_f16_budget_notice_printed = 1;
    if (limit_bytes != UINT64_MAX && free_bytes == 0 && total_bytes == 0 && reserve_bytes == 0) {
        fprintf(stderr,
                "ds4: CUDA q8 fp16 cache %s; using q8 kernels "
                "(request=%.2f MiB cached=%.2f GiB limit=%.2f GiB)\n",
                reason,
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0,
                (double)limit_bytes / 1073741824.0);
    } else if (limit_bytes == UINT64_MAX) {
        fprintf(stderr,
                "ds4: CUDA q8 fp16 cache %s; using q8 kernels "
                "(request=%.2f MiB cached=%.2f GiB free=%.2f GiB reserve=%.2f GiB total=%.2f GiB)\n",
                reason,
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0,
                (double)free_bytes / 1073741824.0,
                (double)reserve_bytes / 1073741824.0,
                (double)total_bytes / 1073741824.0);
    } else {
        fprintf(stderr,
                "ds4: CUDA q8 fp16 cache %s; using q8 kernels "
                "(request=%.2f MiB cached=%.2f GiB limit=%.2f GiB free=%.2f GiB reserve=%.2f GiB total=%.2f GiB)\n",
                reason,
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0,
                (double)limit_bytes / 1073741824.0,
                (double)free_bytes / 1073741824.0,
                (double)reserve_bytes / 1073741824.0,
                (double)total_bytes / 1073741824.0);
    }
}

static int cuda_q8_f16_cache_has_budget(uint64_t request_bytes, const char *label) {
    (void)label;
    uint64_t limit = cuda_q8_f16_cache_limit_bytes();
    if (limit == 0) return 0;
    if (g_q8_f16_bytes > limit || request_bytes > limit - g_q8_f16_bytes) {
        cuda_q8_f16_cache_budget_notice("limit reached", request_bytes, 0, 0, 0, limit);
        return 0;
    }

    size_t free_b = 0;
    size_t total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA q8 fp16 cache memory query failed: %s; using q8 kernels\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    const uint64_t free_bytes = (uint64_t)free_b;
    const uint64_t total_bytes = (uint64_t)total_b;
    /* On 96/128 GB UMA Spark-class systems the expanded Q8->F16 cache can
     * pass a simple free-memory reserve check but still leave too little room
     * for long-prefill cuBLAS execution.  Keep the startup cache useful but
     * bounded unless the caller explicitly sets DS4_CUDA_Q8_F16_CACHE_MB. */
    if (limit == UINT64_MAX &&
        total_bytes <= 128ull * 1073741824ull &&
        (g_model_range_bytes >= 64ull * 1073741824ull ||
         g_model_registered_size >= 64ull * 1073741824ull)) {
        if (g_model_registered_size >= 112ull * 1073741824ull) {
            limit = 4ull * 1073741824ull;
        } else if (g_model_registered_size >= 88ull * 1073741824ull ||
                   g_model_range_bytes >= 88ull * 1073741824ull) {
            limit = 16ull * 1073741824ull;
        } else if (g_model_range_bytes >= 64ull * 1073741824ull) {
            limit = 12ull * 1073741824ull;
        } else {
            limit = 8ull * 1073741824ull;
        }
        if (g_q8_f16_bytes > limit || request_bytes > limit - g_q8_f16_bytes) {
            cuda_q8_f16_cache_budget_notice("limit reached", request_bytes, 0, 0, 0, limit);
            return 0;
        }
    }
    const uint64_t reserve_bytes = cuda_q8_f16_cache_reserve_bytes(total_bytes);
    if (request_bytes > free_bytes ||
        free_bytes - request_bytes < reserve_bytes) {
        cuda_q8_f16_cache_budget_notice("budget exhausted", request_bytes,
                                        free_bytes, total_bytes,
                                        reserve_bytes, limit);
        return 0;
    }
    return 1;
}

static void cuda_q8_f16_cache_disable_after_failure(const char *what, uint64_t request_bytes) {
    if (!g_q8_f16_disabled_after_oom) {
        fprintf(stderr,
                "ds4: CUDA q8 fp16 cache disabled after %s "
                "(request=%.2f MiB cached=%.2f GiB); using q8 kernels\n",
                what ? what : "allocation failure",
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0);
    }
    g_q8_f16_disabled_after_oom = 1;
    if (!g_q8_f16_ranges.empty()) {
        (void)cudaDeviceSynchronize();
        cuda_q8_f16_cache_release_all();
    }
    (void)cudaGetLastError();
}

static int cuda_q8_f16_cache_allowed(const char *label, uint64_t in_dim, uint64_t out_dim) {
    if (g_quality_mode) return 0;
    if (g_q8_f16_disabled_after_oom) return 0;
    if (getenv("DS4_CUDA_NO_Q8_F16_CACHE") != NULL) return 0;
    if (cuda_q8_f16_cache_limit_bytes() == 0) return 0;
    /* MTP verifies a tiny target batch.  Expanding its dense Q8 projections to
     * FP16 lets cuBLAS select Tensor Core kernels for N=2/4.  Keep this opt-in:
     * it deliberately spends the configured Q8->F16 cache budget and is not
     * used by quality mode. */
    if (getenv("DS4_CUDA_MTP_TENSOR_CORES") != NULL) return 1;
    if (getenv("DS4_CUDA_DSPARK_TENSOR_CORES_Q8") != NULL) return 1;
    if (getenv("DS4_CUDA_Q8_F16_ALL") != NULL) return 1;
    if (!label) return 0;
    if (strstr(label, "attn_output_a") != NULL ||
        strstr(label, "attn_output_b") != NULL ||
        strstr(label, "attention_output_a") != NULL ||
        strstr(label, "attention_output_b") != NULL) {
        return getenv("DS4_CUDA_NO_ATTENTION_OUTPUT_F16_CACHE") == NULL;
    }
    if (strstr(label, "attn_q_b") != NULL) {
        return getenv("DS4_CUDA_NO_ATTN_Q_B_F16_CACHE") == NULL;
    }
    if (strstr(label, "ffn_gate_shexp") != NULL ||
        strstr(label, "ffn_up_shexp") != NULL ||
        strstr(label, "ffn_down_shexp") != NULL) {
        return 1;
    }
    return (in_dim == 4096u && out_dim == 2048u) ||
           (in_dim == 2048u && out_dim == 4096u) ||
           (in_dim == 4096u && out_dim == 1024u) ||
           (in_dim == 4096u && out_dim == 512u) ||
           (getenv("DS4_CUDA_NO_ATTN_Q_B_F16_CACHE") == NULL &&
            in_dim == 1024u && out_dim == 32768u);
}

static int cuda_q8_use_dp4a(void) {
    static int mode = -1;
    if (mode < 0) {
        if (getenv("DS4_CUDA_NO_Q8_DP4A") != NULL) {
            mode = 0;
        } else if (getenv("DS4_CUDA_Q8_U16_LOADS") != NULL &&
                   g_q8_u16_validation == 1) {
            mode = 2;
        } else {
            mode = 1;
        }
    }
    return mode;
}

static int cuda_q8_f32_cache_allowed(const char *label, uint64_t in_dim, uint64_t out_dim) {
    if (getenv("DS4_CUDA_NO_Q8_F32_CACHE") != NULL) return 0;
    if (getenv("DS4_CUDA_Q8_F32_ALL") != NULL) return 1;
    if (label && strstr(label, "attn_q_b") != NULL) {
        return getenv("DS4_CUDA_ATTN_Q_B_F32_CACHE") != NULL;
    }
    return getenv("DS4_CUDA_Q8_F32_LARGE") != NULL &&
           in_dim == 1024u && out_dim == 32768u;
}

static const __half *cuda_q8_f16_ptr(
        const void *model_map,
        uint64_t offset,
        uint64_t weight_bytes,
        uint64_t in_dim,
        uint64_t out_dim,
        const char *label) {
    auto exact = g_q8_f16_by_offset.find(
            cuda_model_offset_key(model_map, offset));
    if (exact != g_q8_f16_by_offset.end()) {
        const cuda_q8_f16_range &r = g_q8_f16_ranges[exact->second];
        if (r.host_base == model_map && r.weight_bytes == weight_bytes &&
            r.in_dim == in_dim && r.out_dim == out_dim) {
            return r.device_ptr;
        }
    }
    /* A process can keep both the target model and the optional MTP model
     * resident.  The composite key makes ordinary lookups O(1); retain an
     * identity scan as a collision-safe fallback before considering another
     * large expansion. */
    for (const cuda_q8_f16_range &r : g_q8_f16_ranges) {
        if (r.host_base == model_map && r.offset == offset &&
            r.weight_bytes == weight_bytes && r.in_dim == in_dim &&
            r.out_dim == out_dim) {
            return r.device_ptr;
        }
    }
    if (!cuda_q8_f16_cache_allowed(label, in_dim, out_dim)) return NULL;

    const char *q8 = cuda_model_range_ptr(model_map, offset, weight_bytes, "q8_0");
    if (!q8) return NULL;

    if (in_dim != 0 && out_dim > UINT64_MAX / in_dim / sizeof(__half)) return NULL;
    const uint64_t out_bytes = in_dim * out_dim * sizeof(__half);
    if (!cuda_q8_f16_cache_has_budget(out_bytes, label)) return NULL;

    __half *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)out_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA q8 fp16 cache alloc failed (%.2f MiB): %s\n",
                (double)out_bytes / 1048576.0, cudaGetErrorString(err));
        cuda_q8_f16_cache_disable_after_failure("allocation failure", out_bytes);
        return NULL;
    }
    const uint64_t blocks = (in_dim + 31) / 32;
    const uint64_t n = in_dim * out_dim;
    dequant_q8_0_to_f16_kernel<<<(n + 255) / 256, 256>>>(dev,
                                                          (const unsigned char *)q8,
                                                          in_dim,
                                                          out_dim,
                                                          blocks);
    if (!cuda_ok(cudaGetLastError(), "q8 fp16 dequant launch")) {
        (void)cudaFree(dev);
        cuda_q8_f16_cache_disable_after_failure("dequant launch failure", out_bytes);
        return NULL;
    }
    g_q8_f16_ranges.push_back({model_map, offset, weight_bytes, in_dim, out_dim, dev});
    g_q8_f16_by_offset[cuda_model_offset_key(model_map, offset)] =
        g_q8_f16_ranges.size() - 1u;
    g_q8_f16_bytes += out_bytes;
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr, "ds4: CUDA cached q8 fp16 %.2f MiB (total %.2f GiB)\n",
                (double)out_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0);
    }
    return dev;
}

static float *cuda_q8_f32_ptr(
        const void *model_map,
        uint64_t offset,
        uint64_t weight_bytes,
        uint64_t in_dim,
        uint64_t out_dim,
        const char *label) {
    auto exact = g_q8_f32_by_offset.find(
            cuda_model_offset_key(model_map, offset));
    if (exact != g_q8_f32_by_offset.end()) {
        const cuda_q8_f32_range &r = g_q8_f32_ranges[exact->second];
        if (r.host_base == model_map && r.weight_bytes == weight_bytes &&
            r.in_dim == in_dim && r.out_dim == out_dim) {
            return r.device_ptr;
        }
    }
    for (const cuda_q8_f32_range &r : g_q8_f32_ranges) {
        if (r.host_base == model_map && r.offset == offset &&
            r.weight_bytes == weight_bytes && r.in_dim == in_dim &&
            r.out_dim == out_dim) {
            return r.device_ptr;
        }
    }
    if (!cuda_q8_f32_cache_allowed(label, in_dim, out_dim)) return NULL;

    const char *q8 = cuda_model_range_ptr(model_map, offset, weight_bytes, label ? label : "q8_0");
    if (!q8) return NULL;

    const uint64_t out_bytes = in_dim * out_dim * sizeof(float);
    float *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)out_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA q8 fp32 cache alloc failed (%.2f MiB): %s\n",
                (double)out_bytes / 1048576.0, cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }
    const uint64_t blocks = (in_dim + 31) / 32;
    const uint64_t n = in_dim * out_dim;
    dequant_q8_0_to_f32_kernel<<<(n + 255) / 256, 256>>>(dev,
                                                          (const unsigned char *)q8,
                                                          in_dim,
                                                          out_dim,
                                                          blocks);
    if (!cuda_ok(cudaGetLastError(), "q8 fp32 dequant launch")) {
        (void)cudaFree(dev);
        return NULL;
    }
    g_q8_f32_ranges.push_back({model_map, offset, weight_bytes, in_dim, out_dim, dev});
    g_q8_f32_by_offset[cuda_model_offset_key(model_map, offset)] =
        g_q8_f32_ranges.size() - 1u;
    g_q8_f32_bytes += out_bytes;
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr, "ds4: CUDA cached q8 fp32 %.2f MiB (total %.2f GiB)\n",
                (double)out_bytes / 1048576.0,
                (double)g_q8_f32_bytes / 1073741824.0);
    }
    return dev;
}

static int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    fprintf(stderr, "ds4: CUDA %s failed: %s\n", what, cudaGetErrorString(err));
    return 0;
}

static double cuda_wall_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static int cuda_model_load_progress_enabled(void) {
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE") != NULL) return 0;
    return 1;
}

static int cuda_fused_compressor_update_enabled(void) {
    /* Environment variables are immutable for a running ds4 process.  Cache
     * this one because the decode path reaches it twice per compressed layer. */
    static int enabled = -1;
    if (enabled < 0) {
        enabled = getenv("DS4_CUDA_FUSED_COMPRESSOR_UPDATE") != NULL ? 1 : 0;
        if (enabled) {
            fprintf(stderr, "ds4: CUDA fused ratio4 compressor update enabled\n");
        }
    }
    return enabled;
}

static void cuda_model_load_progress_reset(void) {
    g_model_load_progress_next = 0;
    g_model_load_progress_last_bytes = UINT64_MAX;
    g_model_load_progress_last_cgib = UINT64_MAX;
    g_model_load_progress_last = 0.0;
    g_model_load_progress_started = 0;
    g_model_load_progress_tty = 0;
}

static void cuda_model_load_progress_finish(void) {
    if (!g_model_load_progress_started) return;
    if (g_model_load_progress_tty) {
        fputc('\n', stderr);
        fflush(stderr);
    }
    g_model_load_progress_started = 0;
}

static void cuda_model_load_progress_note(uint64_t cached_bytes) {
    if (!cuda_model_load_progress_enabled()) return;

    const double now = cuda_wall_sec();
    const int tty = isatty(STDERR_FILENO) != 0;
    const uint64_t step = (tty ? 2ull : 16ull) *
                          1024ull * 1024ull * 1024ull;
    const uint64_t gib = 1024ull * 1024ull * 1024ull;
    const uint64_t display_cgib =
        cached_bytes > (UINT64_MAX - gib / 2ull) / 100ull ?
        UINT64_MAX : (cached_bytes * 100ull + gib / 2ull) / gib;
    if (g_model_load_progress_next == 0) {
        g_model_load_progress_next = step;
    }
    if (g_model_load_progress_last != 0.0 &&
        (cached_bytes == g_model_load_progress_last_bytes ||
         display_cgib == g_model_load_progress_last_cgib)) {
        return;
    }
    if (g_model_load_progress_last != 0.0 &&
        cached_bytes < g_model_load_progress_next &&
        now - g_model_load_progress_last < (tty ? 2.0 : 10.0)) {
        return;
    }

    g_model_load_progress_started = 1;
    g_model_load_progress_tty = tty;
    if (g_model_load_progress_tty) {
        fprintf(stderr, "\r\033[Kds4: CUDA loading model tensors into device cache: %.2f GiB",
                (double)cached_bytes / 1073741824.0);
    } else {
        if (g_model_load_progress_last == 0.0) {
            fprintf(stderr, "ds4: CUDA loading model tensors into device cache\n");
        } else {
            fprintf(stderr, "ds4: CUDA loading model tensors %.2f GiB cached\n",
                    (double)cached_bytes / 1073741824.0);
        }
    }
    fflush(stderr);
    g_model_load_progress_last_bytes = cached_bytes;
    g_model_load_progress_last_cgib = display_cgib;
    g_model_load_progress_last = now;
    while (g_model_load_progress_next <= cached_bytes) {
        g_model_load_progress_next += step;
    }
}

static int cuda_model_prefetch_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size) {
    if (!model_map || map_size == 0 || map_offset > model_size || map_size > model_size - map_offset) return 0;
    if (getenv("DS4_CUDA_NO_MODEL_PREFETCH") != NULL ||
        getenv("DS4_CUDA_COPY_MODEL") != NULL ||
        getenv("DS4_CUDA_WEIGHT_CACHE") != NULL ||
        getenv("DS4_CUDA_WEIGHT_PRELOAD") != NULL) {
        return 0;
    }

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }

    int pageable = 0;
    cudaError_t err = cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, device);
    if (err != cudaSuccess || !pageable) {
        (void)cudaGetLastError();
        return 0;
    }
    cudaMemLocation loc;
    memset(&loc, 0, sizeof(loc));
    loc.type = cudaMemLocationTypeDevice;
    loc.id = device;

    const long page_sz_l = sysconf(_SC_PAGESIZE);
    const uint64_t page_sz = page_sz_l > 0 ? (uint64_t)page_sz_l : 4096u;
    const uintptr_t host_addr = (uintptr_t)((const char *)model_map + map_offset);
    const uintptr_t pre_addr = host_addr & ~(uintptr_t)(page_sz - 1u);
    const uint64_t pre_delta = (uint64_t)(host_addr - pre_addr);
    const uint64_t pre_bytes = (pre_delta + map_size + page_sz - 1u) & ~(page_sz - 1u);
    void *pre_ptr = (void *)pre_addr;

    const double t0 = cuda_wall_sec();
    err = cudaMemAdvise(pre_ptr, (size_t)pre_bytes, cudaMemAdviseSetReadMostly, loc);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model read-mostly advise skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    err = cudaMemAdvise(pre_ptr, (size_t)pre_bytes, cudaMemAdviseSetPreferredLocation, loc);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model preferred-location advise skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    if (!g_model_prefetch_stream) {
        err = cudaStreamCreateWithFlags(&g_model_prefetch_stream, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model prefetch stream creation skipped: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }

    err = cudaMemPrefetchAsync(pre_ptr, (size_t)pre_bytes, loc, 0, g_model_prefetch_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model prefetch skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    if (getenv("DS4_CUDA_MODEL_PREFETCH_SYNC") != NULL) {
        err = cudaStreamSynchronize(g_model_prefetch_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model prefetch sync failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    const double t1 = cuda_wall_sec();
    fprintf(stderr,
            "ds4: CUDA ATS/HMM prefetch queued %.2f GiB of model tensors in %.3fs\n",
            (double)map_size / 1073741824.0,
            t1 - t0);
    g_model_hmm_direct = 1;
    return 1;
}

static uint64_t cuda_model_copy_chunk_bytes(void) {
    uint64_t mb = 64;
    const char *env = getenv("DS4_CUDA_MODEL_COPY_CHUNK_MB");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env && v > 0) mb = (uint64_t)v;
    }
    if (mb < 16) mb = 16;
    if (mb > 4096) mb = 4096;
    return mb * 1048576ull;
}

static void cuda_model_discard_source_pages(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes) {
#if defined(POSIX_MADV_DONTNEED)
    if (getenv("DS4_CUDA_KEEP_MODEL_PAGES") != NULL || !model_map || bytes == 0 || offset > model_size) return;
    if (bytes > model_size - offset) bytes = model_size - offset;
    const long page_sz_l = sysconf(_SC_PAGESIZE);
    const uint64_t page_sz = page_sz_l > 0 ? (uint64_t)page_sz_l : 4096u;
    const uintptr_t h0 = (uintptr_t)((const char *)model_map + offset);
    const uintptr_t h1 = h0 + bytes;
    const uintptr_t p0 = h0 & ~(uintptr_t)(page_sz - 1u);
    const uintptr_t p1 = (h1 + page_sz - 1u) & ~(uintptr_t)(page_sz - 1u);
    if (p1 > p0) (void)posix_madvise((void *)p0, (size_t)(p1 - p0), POSIX_MADV_DONTNEED);
#else
    (void)model_map;
    (void)model_size;
    (void)offset;
    (void)bytes;
#endif
}

static void cuda_model_drop_file_pages_for_map(const void *model_map, uint64_t offset, uint64_t bytes) {
#if defined(POSIX_FADV_DONTNEED)
    if (g_model_fd < 0 || getenv("DS4_CUDA_KEEP_MODEL_PAGES") != NULL || bytes == 0) return;
    if (g_model_fd_host_base != NULL && model_map != NULL && model_map != g_model_fd_host_base) return;
    (void)posix_fadvise(g_model_fd, (off_t)offset, (off_t)bytes, POSIX_FADV_DONTNEED);
#else
    (void)model_map;
    (void)offset;
    (void)bytes;
#endif
}

static void cuda_model_drop_file_pages(uint64_t offset, uint64_t bytes) {
    cuda_model_drop_file_pages_for_map(g_model_fd_host_base, offset, bytes);
}

static void cuda_model_drop_copied_source_pages(
        const void *model_map,
        uint64_t model_size,
        const char *label) {
    if (!model_map || model_size == 0 ||
        getenv("DS4_CUDA_KEEP_MODEL_PAGES") != NULL) {
        return;
    }
    cuda_model_drop_file_pages_for_map(model_map, 0, model_size);
    cuda_model_discard_source_pages(model_map, model_size, 0, model_size);
    if (getenv("DS4_CUDA_MODEL_COPY_VERBOSE") != NULL) {
        fprintf(stderr,
                "ds4: CUDA dropped %.2f GiB source pages after %s device copy\n",
                (double)model_size / 1073741824.0,
                label ? label : "model");
    }
}

static uint64_t cuda_round_down(uint64_t v, uint64_t align) {
    if (align <= 1) return v;
    return (v / align) * align;
}

static uint64_t cuda_round_up(uint64_t v, uint64_t align) {
    if (align <= 1) return v;
    const uint64_t rem = v % align;
    return rem == 0 ? v : v + (align - rem);
}

static void *cuda_align_ptr(void *ptr, uint64_t align) {
    if (align <= 1) return ptr;
    uintptr_t p = (uintptr_t)ptr;
    uintptr_t a = (uintptr_t)align;
    return (void *)(((p + a - 1u) / a) * a);
}

static void cuda_model_stage_pool_release(void) {
    if (g_model_upload_stream) {
        (void)cudaStreamSynchronize(g_model_upload_stream);
    }
    for (size_t i = 0; i < 4; i++) {
        if (g_model_stage_event[i]) {
            (void)cudaEventDestroy(g_model_stage_event[i]);
            g_model_stage_event[i] = NULL;
        }
        if (g_model_stage_raw[i]) {
            (void)cudaFreeHost(g_model_stage_raw[i]);
            g_model_stage_raw[i] = NULL;
            g_model_stage[i] = NULL;
        }
    }
    g_model_stage_bytes = 0;
    if (g_model_upload_stream) {
        (void)cudaStreamDestroy(g_model_upload_stream);
        g_model_upload_stream = NULL;
    }
}

static int cuda_model_stage_pool_alloc(uint64_t bytes) {
    if (g_model_stage_bytes >= bytes) return 1;
    cuda_model_stage_pool_release();
    const uint64_t align_slack = g_model_direct_align > 1 ? g_model_direct_align : 1;
    if (bytes > (uint64_t)SIZE_MAX - align_slack) return 0;
    const size_t alloc_bytes = (size_t)(bytes + align_slack);
    if (!g_model_upload_stream) {
        cudaError_t err = cudaStreamCreateWithFlags(&g_model_upload_stream, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model upload stream creation failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    for (size_t i = 0; i < 4; i++) {
        cudaError_t err = cudaMallocHost(&g_model_stage_raw[i], alloc_bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA pinned model staging allocation failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            cuda_model_stage_pool_release();
            return 0;
        }
        g_model_stage[i] = cuda_align_ptr(g_model_stage_raw[i], g_model_direct_align);
        err = cudaEventCreateWithFlags(&g_model_stage_event[i], cudaEventDisableTiming);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model staging event creation failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            cuda_model_stage_pool_release();
            return 0;
        }
    }
    g_model_stage_bytes = bytes;
    return 1;
}

static int cuda_pread_full(int fd, void *buf, uint64_t bytes, uint64_t offset) {
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n_req = (bytes - done > (uint64_t)SSIZE_MAX) ? (size_t)SSIZE_MAX : (size_t)(bytes - done);
        ssize_t n = pread(fd, (char *)buf + done, n_req, (off_t)(offset + done));
        if (n < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        if (n == 0) return 0;
        done += (uint64_t)n;
    }
    return 1;
}

static int cuda_model_stage_read(void *stage, uint64_t stage_bytes,
                                 uint64_t offset, uint64_t bytes,
                                 const char **payload) {
    *payload = (const char *)stage;
#if defined(__linux__) && defined(O_DIRECT)
    if (g_model_direct_fd >= 0 && g_model_direct_align > 1 && g_model_file_size != 0) {
        const uint64_t aligned_off = cuda_round_down(offset, g_model_direct_align);
        const uint64_t delta = offset - aligned_off;
        uint64_t read_size = cuda_round_up(delta + bytes, g_model_direct_align);
        if (aligned_off <= g_model_file_size &&
            read_size <= stage_bytes &&
            read_size <= g_model_file_size - aligned_off) {
            const int saved_errno = errno;
            errno = 0;
            if (cuda_pread_full(g_model_direct_fd, stage, read_size, aligned_off)) {
                *payload = (const char *)stage + delta;
                errno = saved_errno;
                return 1;
            }
            const int direct_errno = errno;
            if (direct_errno == EINVAL || direct_errno == EFAULT || direct_errno == ENOTSUP || direct_errno == EOPNOTSUPP) {
                if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
                    fprintf(stderr, "ds4: CUDA direct model read disabled: %s\n", strerror(direct_errno));
                }
                (void)close(g_model_direct_fd);
                g_model_direct_fd = -1;
                g_model_direct_align = 1;
            }
            errno = direct_errno;
        }
    }
#else
    (void)stage_bytes;
#endif
    return cuda_pread_full(g_model_fd, stage, bytes, offset);
}

static uint64_t cuda_model_cache_limit_bytes(void) {
    uint64_t gb = 0;
    const char *env = getenv("DS4_CUDA_WEIGHT_CACHE_LIMIT_GB");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env) gb = (uint64_t)v;
        return gb * 1073741824ull;
    }
    /* One Spark can run the IQ2 model (~81 GiB) and the mixed q2/q4 model
     * (~91 GiB) via the old startup tensor cache.  Keep enough headroom for
     * scratch, KV, and optional Q8->F16 buffers, and make the full-Q4 model
     * use distributed layer loading unless the operator opts into a larger
     * cache budget explicitly. */
    return 96ull * 1073741824ull;
}

static uint64_t cuda_model_local_model_limit_bytes(void) {
    const uint64_t default_limit = 96ull * 1073741824ull;
    if (!cuda_model_cache_limit_explicit()) return default_limit;
    const uint64_t explicit_limit = cuda_model_cache_limit_bytes();
    return explicit_limit > default_limit ? explicit_limit : default_limit;
}

static int cuda_model_cache_limit_explicit(void) {
    const char *env = getenv("DS4_CUDA_WEIGHT_CACHE_LIMIT_GB");
    return env && env[0];
}

static uint64_t cuda_model_arena_chunk_bytes(uint64_t need) {
    uint64_t mb = 1792;
    const char *env = getenv("DS4_CUDA_WEIGHT_ARENA_CHUNK_MB");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env && v > 0) mb = (uint64_t)v;
    }
    if (mb < 256) mb = 256;
    if (mb > 8192) mb = 8192;
    uint64_t bytes = mb * 1048576ull;
    if (need > bytes / 2u) {
        const uint64_t align = 64ull * 1048576ull;
        return (need + align - 1u) & ~(align - 1u);
    }
    if (bytes < need) {
        const uint64_t align = 256ull * 1048576ull;
        bytes = (need + align - 1u) & ~(align - 1u);
    }
    return bytes;
}

static char *cuda_model_arena_alloc(uint64_t bytes, const char *what) {
    if (bytes == 0) return NULL;
    if (g_model_cache_full) return NULL;
    const uint64_t align = 256u;
    const uint64_t aligned = (bytes + align - 1u) & ~(align - 1u);

    for (cuda_model_arena &a : g_model_arenas) {
        const uint64_t used = (a.used + align - 1u) & ~(align - 1u);
        if (used <= a.bytes && aligned <= a.bytes - used) {
            char *ptr = a.device_ptr + used;
            a.used = used + aligned;
            return ptr;
        }
    }

    const uint64_t limit = cuda_model_cache_limit_bytes();
    if (g_model_range_bytes > limit || aligned > limit - g_model_range_bytes) return NULL;

    const uint64_t chunk = cuda_model_arena_chunk_bytes(aligned);
    void *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)chunk);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model arena alloc failed for %s (%.2f MiB chunk): %s\n",
                what ? what : "weights",
                (double)chunk / 1048576.0,
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        g_model_cache_full = 1;
        return NULL;
    }
    g_model_arenas.push_back({(char *)dev, chunk, aligned});
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        uint64_t arena_bytes = 0;
        for (const cuda_model_arena &a : g_model_arenas) arena_bytes += a.bytes;
        fprintf(stderr, "ds4: CUDA model arena allocated %.2f MiB (arenas %.2f GiB)\n",
                (double)chunk / 1048576.0,
                (double)arena_bytes / 1073741824.0);
    }
    return (char *)dev;
}

/* A raw host pointer is safe for kernels only after CUDA owns, registered, or
 * HMM-prefetched the mapping.  Otherwise let the caller try per-range mapping
 * or a device copy instead of surfacing an async illegal access later. */
static const char *cuda_model_direct_fallback_ptr(const void *model_map, uint64_t offset) {
    if (g_model_device_owned || g_model_registered || g_model_hmm_direct ||
        getenv("DS4_CUDA_DIRECT_MODEL") != NULL) {
        return cuda_model_ptr(model_map, offset);
    }
    return NULL;
}

static const char *cuda_model_range_ptr_from_fd(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes,
        const char *what) {
    if (g_model_fd < 0 || bytes == 0) return NULL;
    if (g_model_fd_host_base != NULL && model_map != g_model_fd_host_base) return NULL;
    const uint64_t limit = cuda_model_cache_limit_bytes();
    if (g_model_range_bytes > limit || bytes > limit - g_model_range_bytes) {
        if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
            fprintf(stderr, "ds4: CUDA direct %s %.2f MiB (cache budget %.2f GiB exhausted)\n",
                    what ? what : "weights",
                    (double)bytes / 1048576.0,
                    (double)limit / 1073741824.0);
        }
        return cuda_model_direct_fallback_ptr(model_map, offset);
    }

    char *dev = cuda_model_arena_alloc(bytes, what);
    if (!dev) {
        if (getenv("DS4_CUDA_STRICT_WEIGHT_CACHE") != NULL) return NULL;
        return cuda_model_direct_fallback_ptr(model_map, offset);
    }
    cudaError_t err = cudaSuccess;

    const uint64_t chunk = cuda_model_copy_chunk_bytes();
    const uint64_t stage_bytes = chunk + (g_model_direct_align > 1 ? g_model_direct_align : 1);
    if (!cuda_model_stage_pool_alloc(stage_bytes)) return NULL;

    uint64_t copied = 0;
    uint64_t chunk_idx = 0;
    while (copied < bytes) {
        const uint64_t n = (bytes - copied < chunk) ? (bytes - copied) : chunk;
        const uint64_t bi = chunk_idx % 4u;
        if (chunk_idx >= 4u) {
            err = cudaEventSynchronize(g_model_stage_event[bi]);
            if (err != cudaSuccess) {
                fprintf(stderr, "ds4: CUDA model staging wait failed for %s: %s\n",
                        what ? what : "weights", cudaGetErrorString(err));
                (void)cudaGetLastError();
                return NULL;
            }
        }
        const char *payload = NULL;
        if (!cuda_model_stage_read(g_model_stage[bi], g_model_stage_bytes,
                                   offset + copied, n, &payload)) {
            fprintf(stderr, "ds4: CUDA model range read failed for %s at %.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)copied / 1048576.0,
                    strerror(errno));
            return NULL;
        }
        err = cudaMemcpyAsync(dev + copied, payload, (size_t)n,
                              cudaMemcpyHostToDevice, g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model range copy failed for %s at %.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)copied / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return NULL;
        }
        err = cudaEventRecord(g_model_stage_event[bi], g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model staging record failed for %s: %s\n",
                    what ? what : "weights", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return NULL;
        }
        cuda_model_drop_file_pages(offset + copied, n);
        cuda_model_discard_source_pages(model_map, g_model_registered_size, offset + copied, n);
        copied += n;
        cuda_model_load_progress_note(g_model_range_bytes + copied);
        chunk_idx++;
    }
    err = cudaStreamSynchronize(g_model_upload_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model range upload sync failed for %s: %s\n",
                what ? what : "weights", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }

    g_model_ranges.push_back({model_map, offset, bytes, dev, NULL, NULL, 0, 0, 1});
    g_model_range_by_offset[cuda_model_offset_key(model_map, offset)] =
        g_model_ranges.size() - 1u;
    g_model_range_bytes += bytes;
    cuda_model_load_progress_note(g_model_range_bytes);
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr, "ds4: CUDA fd-cached %s %.2f MiB (total %.2f GiB)\n",
                what ? what : "weights",
                (double)bytes / 1048576.0,
                (double)g_model_range_bytes / 1073741824.0);
    }
    return (const char *)dev;
}

static int cuda_model_copy_chunked(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size) {
    if (!model_map || model_size == 0 || map_offset > model_size || map_size > model_size - map_offset) return 0;
    if (getenv("DS4_CUDA_NO_MODEL_COPY") != NULL ||
        getenv("DS4_CUDA_DIRECT_MODEL") != NULL ||
        getenv("DS4_CUDA_WEIGHT_CACHE") != NULL ||
        getenv("DS4_CUDA_WEIGHT_PRELOAD") != NULL) {
        return 0;
    }
    if (g_model_device_owned || g_model_registered) return 1;

    void *dev = NULL;
    const double t0 = cuda_wall_sec();
    cudaError_t err = cudaMalloc(&dev, (size_t)model_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model allocation skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    const uint64_t chunk = cuda_model_copy_chunk_bytes();
    const uint64_t align_slack = g_model_direct_align > 1 ? g_model_direct_align : 1;
    if (chunk > UINT64_MAX - align_slack ||
        !cuda_model_stage_pool_alloc(chunk + align_slack)) {
        (void)cudaFree(dev);
        return 0;
    }

    const uint64_t copy_bytes = map_offset + map_size;
    fprintf(stderr,
            "ds4: CUDA pipelined model copy %.2f GiB (chunk=%llu MiB, stages=4, direct-io=%d)\n",
            (double)copy_bytes / 1073741824.0,
            (unsigned long long)(chunk / 1048576ull),
            g_model_direct_fd >= 0 ? 1 : 0);

    uint64_t copied = 0;
    uint64_t chunk_idx = 0;
    double last_report = t0;
    while (copied < copy_bytes) {
        const uint64_t n = (copy_bytes - copied < chunk) ? (copy_bytes - copied) : chunk;
        const uint64_t bi = chunk_idx % 4u;
        if (chunk_idx >= 4u) {
            err = cudaEventSynchronize(g_model_stage_event[bi]);
            if (err != cudaSuccess) {
                fprintf(stderr, "ds4: CUDA model staging wait failed at %.2f GiB: %s\n",
                        (double)copied / 1073741824.0, cudaGetErrorString(err));
                (void)cudaGetLastError();
                cuda_model_stage_pool_release();
                (void)cudaFree(dev);
                return 0;
            }
        }

        const char *payload = NULL;
        if (g_model_fd >= 0) {
            if (!cuda_model_stage_read(g_model_stage[bi], g_model_stage_bytes,
                                       copied, n, &payload)) {
                fprintf(stderr, "ds4: CUDA model read failed at %.2f GiB: %s\n",
                        (double)copied / 1073741824.0, strerror(errno));
                cuda_model_stage_pool_release();
                (void)cudaFree(dev);
                return 0;
            }
        } else {
            memcpy(g_model_stage[bi], (const char *)model_map + copied, (size_t)n);
            payload = (const char *)g_model_stage[bi];
        }

        err = cudaMemcpyAsync((char *)dev + copied, payload, (size_t)n,
                              cudaMemcpyHostToDevice, g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model async copy failed at %.2f GiB: %s\n",
                    (double)copied / 1073741824.0, cudaGetErrorString(err));
            (void)cudaGetLastError();
            cuda_model_stage_pool_release();
            (void)cudaFree(dev);
            return 0;
        }
        err = cudaEventRecord(g_model_stage_event[bi], g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model staging record failed at %.2f GiB: %s\n",
                    (double)copied / 1073741824.0, cudaGetErrorString(err));
            (void)cudaGetLastError();
            cuda_model_stage_pool_release();
            (void)cudaFree(dev);
            return 0;
        }
        cuda_model_drop_file_pages_for_map(model_map, copied, n);
        cuda_model_discard_source_pages(model_map, model_size, copied, n);
        copied += n;
        chunk_idx++;
        const double now = cuda_wall_sec();
        if (getenv("DS4_CUDA_MODEL_COPY_VERBOSE") != NULL && now - last_report >= 2.0) {
            fprintf(stderr, "ds4: CUDA pipelined model copy %.2f/%.2f GiB\n",
                    (double)copied / 1073741824.0,
                    (double)copy_bytes / 1073741824.0);
            last_report = now;
        }
    }

    err = cudaStreamSynchronize(g_model_upload_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model upload sync failed: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        cuda_model_stage_pool_release();
        (void)cudaFree(dev);
        return 0;
    }
    cuda_model_stage_pool_release();
    g_model_device_base = (const char *)dev;
    g_model_device_owned = 1;
    g_model_hmm_direct = 0;
    const double t1 = cuda_wall_sec();
    const double elapsed = t1 - t0;
    fprintf(stderr,
            "ds4: CUDA pipelined model copy complete in %.3fs (%.2f GiB, %.2f GiB/s)\n",
            elapsed,
            (double)copy_bytes / 1073741824.0,
            elapsed > 0.0 ? ((double)copy_bytes / 1073741824.0) / elapsed : 0.0);
    return 1;
}

static void cuda_model_range_release_all(void) {
    cuda_model_load_progress_finish();
    for (const cuda_model_range &r : g_model_ranges) {
        if (r.host_registered && r.registered_base) {
            (void)cudaHostUnregister(r.registered_base);
        } else if (r.device_ptr && !r.arena_allocated) {
            (void)cudaFree(r.device_ptr);
        }
    }
    for (const cuda_model_arena &a : g_model_arenas) {
        if (a.device_ptr) (void)cudaFree(a.device_ptr);
    }
    g_model_arenas.clear();
    g_model_ranges.clear();
    g_model_range_by_offset.clear();
    g_model_range_bytes = 0;
}

static void cuda_stream_selected_cache_invalidate(void) {
    g_stream_selected_cache.valid = 0;
}

static void cuda_stream_selected_cache_release(void) {
    if (g_stream_selected_cache.gate_ptr) {
        (void)cudaFree(g_stream_selected_cache.gate_ptr);
    }
    if (g_stream_selected_cache.up_ptr) {
        (void)cudaFree(g_stream_selected_cache.up_ptr);
    }
    if (g_stream_selected_cache.down_ptr) {
        (void)cudaFree(g_stream_selected_cache.down_ptr);
    }
    if (g_stream_selected_cache.slot_selected_ptr) {
        (void)cudaFree(g_stream_selected_cache.slot_selected_ptr);
    }
    memset(&g_stream_selected_cache, 0, sizeof(g_stream_selected_cache));
}

static void cuda_stream_expert_cache_release_all(void) {
    if (g_stream_expert_cache.gate_ptr) {
        (void)cudaFree(g_stream_expert_cache.gate_ptr);
    }
    if (g_stream_expert_cache.up_ptr) {
        (void)cudaFree(g_stream_expert_cache.up_ptr);
    }
    if (g_stream_expert_cache.down_ptr) {
        (void)cudaFree(g_stream_expert_cache.down_ptr);
    }
    g_stream_expert_cache.slots.clear();
    memset(&g_stream_expert_cache, 0, sizeof(g_stream_expert_cache));
}

static void cuda_stream_expert_cache_invalidate(void) {
    for (cuda_stream_expert_cache_slot &slot : g_stream_expert_cache.slots) {
        slot.valid = 0;
    }
    g_stream_expert_cache.valid = 0;
    g_stream_expert_cache.count = 0;
    g_stream_expert_cache.tick = 0;
}

static uint32_t cuda_stream_expert_cache_requested_budget(void) {
    uint32_t cap = g_stream_expert_budget_override != 0 ?
        g_stream_expert_budget_override : DS4_CUDA_STREAM_EXPERT_DEFAULT;
    const char *env = getenv("DS4_CUDA_STREAMING_EXPERT_CACHE_N");
    if (env && env[0]) {
        char *end = NULL;
        errno = 0;
        unsigned long v = strtoul(env, &end, 10);
        while (end && (*end == ' ' || *end == '\t')) end++;
        if (end != env && errno == 0 && end && *end == '\0') {
            cap = v > DS4_CUDA_STREAM_EXPERT_MAX ?
                DS4_CUDA_STREAM_EXPERT_MAX : (uint32_t)v;
        }
    }
    if (cap > DS4_CUDA_STREAM_EXPERT_MAX) {
        cap = DS4_CUDA_STREAM_EXPERT_MAX;
    }
    return cap;
}

static uint32_t cuda_stream_expert_cache_configured_budget(void) {
    uint32_t cap = cuda_stream_expert_cache_requested_budget();
    if (g_stream_expert_runtime_cap != 0 && cap > g_stream_expert_runtime_cap) {
        cap = g_stream_expert_runtime_cap;
    }
    return cap;
}

static int cuda_stream_expert_cache_budget_visible_to_shared(void) {
    if (!g_ssd_streaming_mode) return 0;
    if (g_stream_expert_budget_override != 0) return 1;
    const char *env = getenv("DS4_CUDA_STREAMING_EXPERT_CACHE_N");
    if (env && env[0]) return 1;
    env = getenv("DS4_CUDA_ENABLE_STREAMING_EXPERT_HOTLIST");
    if (!env || !env[0]) {
        env = getenv("DS4_CUDA_STREAMING_EXPERT_HOTLIST");
    }
    return env && env[0] && strcmp(env, "0") != 0;
}

static uint64_t cuda_stream_expert_cache_reserve_bytes(void) {
    uint64_t gb = 16;
    const char *env = getenv("DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB");
    if (env && env[0]) {
        char *end = NULL;
        errno = 0;
        unsigned long long v = strtoull(env, &end, 10);
        while (end && (*end == ' ' || *end == '\t')) end++;
        if (end != env && errno == 0 && end && *end == '\0') {
            gb = (uint64_t)v;
        }
    }
    if (gb > UINT64_MAX / 1073741824ull) return UINT64_MAX;
    return gb * 1073741824ull;
}

static uint32_t cuda_stream_expert_cache_live_budget(
        uint32_t requested,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        uint64_t reclaim_bytes,
        int report) {
    if (requested == 0 ||
        gate_expert_bytes == 0 ||
        down_expert_bytes == 0 ||
        gate_expert_bytes > (UINT64_MAX - down_expert_bytes) / 2ull) {
        return 0;
    }
    const uint64_t per_expert_bytes =
        gate_expert_bytes * 2ull + down_expert_bytes;
    if (per_expert_bytes == 0) return 0;

    size_t free_b = 0;
    size_t total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "ds4: CUDA streaming expert cache memory query failed: %s; "
                "using direct selected loads\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    uint64_t free_bytes = (uint64_t)free_b;
    if (reclaim_bytes > UINT64_MAX - free_bytes) {
        free_bytes = UINT64_MAX;
    } else {
        free_bytes += reclaim_bytes;
    }
    uint64_t reserve = cuda_stream_expert_cache_reserve_bytes();
    const uint64_t total_bytes = (uint64_t)total_b;
    if (total_bytes != 0 && reserve > total_bytes / 2ull) {
        reserve = total_bytes / 2ull;
    }
    if (free_bytes <= reserve) {
        if (report && g_stream_expert_memory_cap_notice != requested) {
            cuda_model_load_progress_finish();
            fprintf(stderr,
                    "ds4: CUDA streaming expert cache disabled: available %.2f GiB <= reserve %.2f GiB\n",
                    (double)free_bytes / 1073741824.0,
                    (double)reserve / 1073741824.0);
            g_stream_expert_memory_cap_notice = requested;
        }
        return 0;
    }

    uint64_t usable = free_bytes - reserve;
    uint64_t max_slots64 = usable / per_expert_bytes;
    if (max_slots64 > UINT32_MAX) max_slots64 = UINT32_MAX;
    uint32_t capped = requested;
    if ((uint64_t)capped > max_slots64) capped = (uint32_t)max_slots64;
    if (report && capped != requested && g_stream_expert_memory_cap_notice != capped) {
        cuda_model_load_progress_finish();
        fprintf(stderr,
                "ds4: CUDA streaming expert cache capped from %u to %u experts "
                "(available %.2f GiB, reserve %.2f GiB, %.2f MiB/expert)\n",
                requested,
                capped,
                (double)free_bytes / 1073741824.0,
                (double)reserve / 1073741824.0,
                (double)per_expert_bytes / 1048576.0);
        g_stream_expert_memory_cap_notice = capped;
    }
    return capped;
}

static uint64_t cuda_stream_expert_cache_expert_bytes(
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (gate_expert_bytes == 0 ||
        down_expert_bytes == 0 ||
        gate_expert_bytes > (UINT64_MAX - down_expert_bytes) / 2ull) {
        return 0;
    }
    return gate_expert_bytes * 2ull + down_expert_bytes;
}

static void cuda_stream_expert_cache_note_size(
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (g_stream_expert_runtime_gate_bytes == gate_expert_bytes &&
        g_stream_expert_runtime_down_bytes == down_expert_bytes) {
        return;
    }
    g_stream_expert_runtime_gate_bytes = gate_expert_bytes;
    g_stream_expert_runtime_down_bytes = down_expert_bytes;
    g_stream_expert_runtime_cap = 0;
    g_stream_expert_memory_cap_notice = 0;
}

static uint32_t cuda_stream_expert_cache_shrunken_cap(uint32_t cap) {
    if (cap == 0) return 0;
    const uint32_t release = (cap + 9u) / 10u;
    return cap > release ? cap - release : 0;
}

static void cuda_stream_expert_cache_note_oom_cap(
        uint32_t failed_cap,
        uint32_t new_cap,
        uint64_t expert_bytes,
        const char *errstr) {
    if (g_stream_expert_runtime_cap != 0 &&
        g_stream_expert_runtime_cap <= new_cap) {
        return;
    }
    g_stream_expert_runtime_cap = new_cap;
    const uint32_t released =
        failed_cap > new_cap ? failed_cap - new_cap : 0;
    cuda_model_load_progress_finish();
    fprintf(stderr,
            "ds4: CUDA streaming expert cache allocation failed at %u experts "
            "/ %.2f GiB%s%s\n",
            failed_cap,
            expert_bytes != 0 ?
                (double)((uint64_t)failed_cap * expert_bytes) / 1073741824.0 :
                0.0,
            errstr && errstr[0] ? ": " : "",
            errstr && errstr[0] ? errstr : "");
    if (new_cap != 0) {
        fprintf(stderr,
                "ds4:   shrinking resident cache margin by %u experts / %.2f GiB; "
                "runtime cache cap now %u experts\n",
                released,
                expert_bytes != 0 ?
                    (double)((uint64_t)released * expert_bytes) / 1073741824.0 :
                    0.0,
                new_cap);
    } else {
        fprintf(stderr,
                "ds4:   disabling resident expert cache after OOM; using direct selected loads\n");
    }
}

static int cuda_stream_expert_cache_try_alloc(
        uint32_t cap,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        char **gate_ptr,
        char **up_ptr,
        char **down_ptr,
        const char **errstr) {
    *gate_ptr = NULL;
    *up_ptr = NULL;
    *down_ptr = NULL;
    if (errstr) *errstr = NULL;
    if (cap == 0 ||
        (uint64_t)cap > UINT64_MAX / gate_expert_bytes ||
        (uint64_t)cap > UINT64_MAX / down_expert_bytes) {
        return 0;
    }
    const uint64_t gate_bytes = (uint64_t)cap * gate_expert_bytes;
    const uint64_t down_bytes = (uint64_t)cap * down_expert_bytes;

    void *gate = NULL;
    void *up = NULL;
    void *down = NULL;
    cudaError_t err = cudaMalloc(&gate, (size_t)gate_bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&up, (size_t)gate_bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&down, (size_t)down_bytes);
    if (err != cudaSuccess) goto fail;

    *gate_ptr = (char *)gate;
    *up_ptr = (char *)up;
    *down_ptr = (char *)down;
    return 1;

fail:
    if (errstr) *errstr = cudaGetErrorString(err);
    (void)cudaGetLastError();
    if (gate) (void)cudaFree(gate);
    if (up) (void)cudaFree(up);
    if (down) (void)cudaFree(down);
    return 0;
}

static void cuda_stream_selected_stage_release(void) {
    for (size_t i = 0; i < 4; i++) {
        if (g_stream_selected_stage_event[i]) {
            (void)cudaEventDestroy(g_stream_selected_stage_event[i]);
            g_stream_selected_stage_event[i] = NULL;
        }
        if (g_stream_selected_stage_raw[i]) {
            (void)cudaFreeHost(g_stream_selected_stage_raw[i]);
            g_stream_selected_stage_raw[i] = NULL;
            g_stream_selected_stage[i] = NULL;
        }
    }
    g_stream_selected_stage_bytes = 0;
    if (g_stream_selected_upload_stream) {
        (void)cudaStreamDestroy(g_stream_selected_upload_stream);
        g_stream_selected_upload_stream = NULL;
    }
}

static int cuda_stream_selected_stage_pool_alloc(uint64_t bytes) {
    if (g_stream_selected_stage_bytes >= bytes) return 1;
    for (size_t i = 0; i < 4; i++) {
        if (g_stream_selected_stage_event[i]) {
            (void)cudaEventDestroy(g_stream_selected_stage_event[i]);
            g_stream_selected_stage_event[i] = NULL;
        }
        if (g_stream_selected_stage_raw[i]) {
            (void)cudaFreeHost(g_stream_selected_stage_raw[i]);
            g_stream_selected_stage_raw[i] = NULL;
            g_stream_selected_stage[i] = NULL;
        }
    }
    g_stream_selected_stage_bytes = 0;
    if (!g_stream_selected_upload_stream) {
        cudaError_t err = cudaStreamCreateWithFlags(&g_stream_selected_upload_stream,
                                                    cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected upload stream creation failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    for (size_t i = 0; i < 4; i++) {
        cudaError_t err = cudaMallocHost(&g_stream_selected_stage_raw[i],
                                         (size_t)bytes);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected staging allocation failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        g_stream_selected_stage[i] =
            cuda_align_ptr(g_stream_selected_stage_raw[i],
                           g_model_direct_align);
        err = cudaEventCreateWithFlags(&g_stream_selected_stage_event[i],
                                       cudaEventDisableTiming);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected staging event creation failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    g_stream_selected_stage_bytes = bytes;
    return 1;
}

static int cuda_stream_selected_ensure_bytes(
        char **ptr,
        uint64_t *capacity,
        uint64_t bytes,
        const char *what) {
    if (bytes == 0) return 1;
    if (*ptr && *capacity >= bytes) return 1;
    if (*ptr) {
        (void)cudaFree(*ptr);
        *ptr = NULL;
        *capacity = 0;
    }
    void *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)bytes);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "ds4: CUDA streaming selected cache allocation failed for %s (%.2f MiB): %s\n",
                what ? what : "experts",
                (double)bytes / 1048576.0,
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    *ptr = (char *)dev;
    *capacity = bytes;
    return 1;
}

static int cuda_stream_selected_ensure_i32(
        int32_t **ptr,
        uint64_t *capacity,
        uint64_t count,
        const char *what) {
    if (count == 0 || count > UINT64_MAX / sizeof(int32_t)) return 0;
    const uint64_t bytes = count * sizeof(int32_t);
    if (*ptr && *capacity >= bytes) return 1;
    if (*ptr) {
        (void)cudaFree(*ptr);
        *ptr = NULL;
        *capacity = 0;
    }
    void *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)bytes);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "ds4: CUDA streaming selected cache allocation failed for %s (%u entries): %s\n",
                what ? what : "selected slots",
                (unsigned)count,
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    *ptr = (int32_t *)dev;
    *capacity = bytes;
    return 1;
}

static cuda_stream_expert_cache *cuda_stream_expert_cache_prepare(
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        uint32_t target_cap) {
    const uint64_t expert_bytes =
        cuda_stream_expert_cache_expert_bytes(gate_expert_bytes,
                                              down_expert_bytes);
    if (expert_bytes == 0) return NULL;
    cuda_stream_expert_cache_note_size(gate_expert_bytes, down_expert_bytes);

    const uint32_t requested_cap = cuda_stream_expert_cache_configured_budget();
    if (requested_cap == 0) return NULL;
    if (target_cap == 0 || target_cap > requested_cap) target_cap = requested_cap;
    if (target_cap == 0) return NULL;
    const int same_dims =
        g_stream_expert_cache.valid &&
        g_stream_expert_cache.gate_expert_bytes == gate_expert_bytes &&
        g_stream_expert_cache.down_expert_bytes == down_expert_bytes;
    if (!same_dims && g_stream_expert_cache.valid) {
        cuda_stream_expert_cache_release_all();
    }
    if (same_dims &&
        g_stream_expert_cache.capacity != 0 &&
        g_stream_expert_cache.capacity >= target_cap &&
        g_stream_expert_cache.slots.size() == g_stream_expert_cache.capacity) {
        return &g_stream_expert_cache;
    }

    uint64_t reclaim_bytes = 0;
    if (same_dims &&
        g_stream_expert_cache.capacity != 0 &&
        (uint64_t)g_stream_expert_cache.capacity <= UINT64_MAX / expert_bytes) {
        reclaim_bytes = (uint64_t)g_stream_expert_cache.capacity * expert_bytes;
    }
    uint32_t cap =
        cuda_stream_expert_cache_live_budget(target_cap,
                                             gate_expert_bytes,
                                             down_expert_bytes,
                                             reclaim_bytes,
                                             reclaim_bytes == 0);
    if (cap == 0) return NULL;
    if (same_dims &&
        g_stream_expert_cache.capacity != 0 &&
        g_stream_expert_cache.capacity >= cap &&
        g_stream_expert_cache.slots.size() == g_stream_expert_cache.capacity) {
        return &g_stream_expert_cache;
    }

    cuda_stream_expert_cache_release_all();
    while (cap != 0) {
        if ((uint64_t)cap > UINT64_MAX / gate_expert_bytes ||
            (uint64_t)cap > UINT64_MAX / down_expert_bytes) {
            fprintf(stderr, "ds4: CUDA streaming expert cache size overflow\n");
            return NULL;
        }

        char *gate_ptr = NULL;
        char *up_ptr = NULL;
        char *down_ptr = NULL;
        const char *alloc_error = NULL;
        if (!cuda_stream_expert_cache_try_alloc(cap,
                                                gate_expert_bytes,
                                                down_expert_bytes,
                                                &gate_ptr,
                                                &up_ptr,
                                                &down_ptr,
                                                &alloc_error)) {
            const uint32_t new_cap =
                cuda_stream_expert_cache_shrunken_cap(cap);
            cuda_stream_expert_cache_note_oom_cap(cap,
                                                  new_cap,
                                                  expert_bytes,
                                                  alloc_error);
            cap = new_cap;
            if (cap != 0) {
                cap = cuda_stream_expert_cache_live_budget(cap,
                                                           gate_expert_bytes,
                                                           down_expert_bytes,
                                                           0,
                                                           1);
            }
            continue;
        }

        try {
            g_stream_expert_cache.slots.resize(cap);
        } catch (...) {
            fprintf(stderr, "ds4: CUDA streaming expert cache metadata allocation failed\n");
            (void)cudaFree(gate_ptr);
            (void)cudaFree(up_ptr);
            (void)cudaFree(down_ptr);
            cuda_stream_expert_cache_release_all();
            return NULL;
        }

        g_stream_expert_cache.valid = 1;
        g_stream_expert_cache.capacity = cap;
        g_stream_expert_cache.count = 0;
        g_stream_expert_cache.tick = 0;
        g_stream_expert_cache.gate_expert_bytes = gate_expert_bytes;
        g_stream_expert_cache.down_expert_bytes = down_expert_bytes;
        g_stream_expert_cache.gate_ptr = gate_ptr;
        g_stream_expert_cache.up_ptr = up_ptr;
        g_stream_expert_cache.down_ptr = down_ptr;
        g_stream_expert_cache.gate_capacity =
            (uint64_t)cap * gate_expert_bytes;
        g_stream_expert_cache.up_capacity =
            (uint64_t)cap * gate_expert_bytes;
        g_stream_expert_cache.down_capacity =
            (uint64_t)cap * down_expert_bytes;
        return &g_stream_expert_cache;
    }
    return NULL;
}

static int cuda_stream_expert_cache_find(
        cuda_stream_expert_cache *cache,
        const void *model_map,
        uint64_t model_size,
        uint32_t layer,
        uint32_t n_total_expert,
        uint32_t expert,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (!cache || !cache->valid) return -1;
    for (uint32_t i = 0; i < cache->capacity; i++) {
        const cuda_stream_expert_cache_slot &slot = cache->slots[i];
        if (slot.valid &&
            slot.model_map == model_map &&
            slot.model_size == model_size &&
            slot.layer == layer &&
            slot.n_total_expert == n_total_expert &&
            slot.expert == expert &&
            slot.gate_offset == gate_offset &&
            slot.up_offset == up_offset &&
            slot.down_offset == down_offset &&
            slot.gate_expert_bytes == gate_expert_bytes &&
            slot.down_expert_bytes == down_expert_bytes) {
            return (int)i;
        }
    }
    return -1;
}

static uint32_t cuda_stream_expert_cache_lru_slot(
        cuda_stream_expert_cache *cache) {
    for (uint32_t i = 0; i < cache->capacity; i++) {
        if (!cache->slots[i].valid) return i;
    }
    uint32_t slot = 0;
    uint64_t best_age = cache->slots[0].age;
    for (uint32_t i = 1; i < cache->capacity; i++) {
        if (cache->slots[i].age < best_age) {
            best_age = cache->slots[i].age;
            slot = i;
        }
    }
    return slot;
}

static int cuda_stream_expert_cache_copy_to_compact(
        cuda_stream_expert_cache *cache,
        uint32_t cache_slot,
        uint32_t compact_slot,
        char *compact_gate,
        char *compact_up,
        char *compact_down) {
    const uint64_t gate_src = (uint64_t)cache_slot * cache->gate_expert_bytes;
    const uint64_t down_src = (uint64_t)cache_slot * cache->down_expert_bytes;
    const uint64_t gate_dst = (uint64_t)compact_slot * cache->gate_expert_bytes;
    const uint64_t down_dst = (uint64_t)compact_slot * cache->down_expert_bytes;
    return cuda_ok(cudaMemcpy(compact_gate + gate_dst,
                              cache->gate_ptr + gate_src,
                              (size_t)cache->gate_expert_bytes,
                              cudaMemcpyDeviceToDevice),
                   "streaming selected gate cache copy") &&
           cuda_ok(cudaMemcpy(compact_up + gate_dst,
                              cache->up_ptr + gate_src,
                              (size_t)cache->gate_expert_bytes,
                              cudaMemcpyDeviceToDevice),
                   "streaming selected up cache copy") &&
           cuda_ok(cudaMemcpy(compact_down + down_dst,
                              cache->down_ptr + down_src,
                              (size_t)cache->down_expert_bytes,
                              cudaMemcpyDeviceToDevice),
                   "streaming selected down cache copy");
}

static int cuda_stream_expert_cache_load_slot(
        cuda_stream_expert_cache *cache,
        const void *model_map,
        uint64_t model_size,
        uint32_t slot,
        uint32_t layer,
        uint32_t n_total_expert,
        uint32_t expert,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    const uint64_t gate_src =
        gate_offset + (uint64_t)expert * gate_expert_bytes;
    const uint64_t up_src =
        up_offset + (uint64_t)expert * gate_expert_bytes;
    const uint64_t down_src =
        down_offset + (uint64_t)expert * down_expert_bytes;
    const uint64_t gate_dst = (uint64_t)slot * gate_expert_bytes;
    const uint64_t down_dst = (uint64_t)slot * down_expert_bytes;
    if (!cuda_model_copy_to_device_streamed(cache->gate_ptr + gate_dst,
                                            model_map,
                                            model_size,
                                            gate_src,
                                            gate_expert_bytes,
                                            "cached moe_gate") ||
        !cuda_model_copy_to_device_streamed(cache->up_ptr + gate_dst,
                                            model_map,
                                            model_size,
                                            up_src,
                                            gate_expert_bytes,
                                            "cached moe_up") ||
        !cuda_model_copy_to_device_streamed(cache->down_ptr + down_dst,
                                            model_map,
                                            model_size,
                                            down_src,
                                            down_expert_bytes,
                                            "cached moe_down")) {
        return 0;
    }
    cuda_stream_expert_cache_slot &entry = cache->slots[slot];
    entry.valid = 1;
    entry.model_map = model_map;
    entry.model_size = model_size;
    entry.layer = layer;
    entry.n_total_expert = n_total_expert;
    entry.expert = expert;
    entry.gate_offset = gate_offset;
    entry.up_offset = up_offset;
    entry.down_offset = down_offset;
    entry.gate_expert_bytes = gate_expert_bytes;
    entry.down_expert_bytes = down_expert_bytes;
    entry.age = ++cache->tick;
    return 1;
}

static int cuda_stream_expert_cache_seed_one(
        cuda_stream_expert_cache *cache,
        const void *model_map,
        uint64_t model_size,
        uint32_t layer,
        uint32_t n_total_expert,
        uint32_t expert,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    int cache_slot = cuda_stream_expert_cache_find(cache,
                                                   model_map,
                                                   model_size,
                                                   layer,
                                                   n_total_expert,
                                                   expert,
                                                   gate_offset,
                                                   up_offset,
                                                   down_offset,
                                                   gate_expert_bytes,
                                                   down_expert_bytes);
    if (cache_slot >= 0) {
        cache->slots[(uint32_t)cache_slot].age = ++cache->tick;
        return 1;
    }

    const uint32_t load_slot = cuda_stream_expert_cache_lru_slot(cache);
    const int append = !cache->slots[load_slot].valid;
    if (!cuda_stream_expert_cache_load_slot(cache,
                                            model_map,
                                            model_size,
                                            load_slot,
                                            layer,
                                            n_total_expert,
                                            expert,
                                            gate_offset,
                                            up_offset,
                                            down_offset,
                                            gate_expert_bytes,
                                            down_expert_bytes)) {
        return 0;
    }
    if (append && cache->count < cache->capacity) cache->count++;
    return 1;
}

static int cuda_stream_layer_expert_ranges_valid(
        uint64_t model_size,
        uint32_t n_total_expert,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        const char *what) {
    if (n_total_expert == 0 ||
        gate_expert_bytes == 0 ||
        down_expert_bytes == 0 ||
        (uint64_t)n_total_expert > UINT64_MAX / gate_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / down_expert_bytes) {
        fprintf(stderr,
                "ds4: CUDA streaming %s expert size overflow\n",
                what ? what : "selected");
        return 0;
    }
    const uint64_t full_gate_bytes =
        (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t full_down_bytes =
        (uint64_t)n_total_expert * down_expert_bytes;
    if (gate_offset > model_size || up_offset > model_size ||
        down_offset > model_size ||
        full_gate_bytes > model_size - gate_offset ||
        full_gate_bytes > model_size - up_offset ||
        full_down_bytes > model_size - down_offset) {
        fprintf(stderr,
                "ds4: CUDA streaming %s expert range outside model map\n",
                what ? what : "selected");
        return 0;
    }
    return 1;
}

static int cuda_model_copy_to_device_streamed(
        char *dst,
        const void *model_map,
        uint64_t model_size,
        uint64_t offset,
        uint64_t bytes,
        const char *what) {
    if (!dst || !model_map || offset > model_size || bytes > model_size - offset) {
        return 0;
    }
    if (bytes == 0) return 1;
    if (g_model_fd < 0 ||
        (g_model_fd_host_base != NULL && model_map != g_model_fd_host_base)) {
        return cuda_ok(cudaMemcpy(dst,
                                  (const char *)model_map + offset,
                                  (size_t)bytes,
                                  cudaMemcpyHostToDevice),
                       what ? what : "stream selected expert copy");
    }

    const uint64_t chunk = cuda_model_copy_chunk_bytes();
    const uint64_t stage_bytes = chunk + (g_model_direct_align > 1 ? g_model_direct_align : 1);
    if (!cuda_stream_selected_stage_pool_alloc(stage_bytes)) return 0;

    cudaError_t err = cudaSuccess;
    uint64_t copied = 0;
    uint64_t chunk_idx = 0;
    while (copied < bytes) {
        const uint64_t n = (bytes - copied < chunk) ? (bytes - copied) : chunk;
        const uint64_t bi = chunk_idx % 4u;
        if (chunk_idx >= 4u) {
            err = cudaEventSynchronize(g_stream_selected_stage_event[bi]);
            if (err != cudaSuccess) {
                fprintf(stderr,
                        "ds4: CUDA streaming selected staging wait failed for %s: %s\n",
                        what ? what : "expert",
                        cudaGetErrorString(err));
                (void)cudaGetLastError();
                return 0;
            }
        }

        const char *payload = NULL;
        if (!cuda_model_stage_read(g_stream_selected_stage[bi],
                                   g_stream_selected_stage_bytes,
                                   offset + copied,
                                   n,
                                   &payload)) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected read failed for %s at %.2f MiB: %s\n",
                    what ? what : "expert",
                    (double)copied / 1048576.0,
                    strerror(errno));
            return 0;
        }
        err = cudaMemcpyAsync(dst + copied,
                              payload,
                              (size_t)n,
                              cudaMemcpyHostToDevice,
                              g_stream_selected_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected copy failed for %s at %.2f MiB: %s\n",
                    what ? what : "expert",
                    (double)copied / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        err = cudaEventRecord(g_stream_selected_stage_event[bi],
                              g_stream_selected_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected staging record failed for %s: %s\n",
                    what ? what : "expert",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        cuda_model_drop_file_pages(offset + copied, n);
        cuda_model_discard_source_pages(model_map, model_size, offset + copied, n);
        copied += n;
        chunk_idx++;
    }

    err = cudaStreamSynchronize(g_stream_selected_upload_stream);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "ds4: CUDA streaming selected upload sync failed for %s: %s\n",
                what ? what : "expert",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}

static int cublas_ok(cublasStatus_t st, const char *what) {
    if (st == CUBLAS_STATUS_SUCCESS) return 1;
    fprintf(stderr, "ds4: cuBLAS %s failed: status %d\n", what, (int)st);
    return 0;
}

extern "C" int ds4_gpu_init(void) {
    int dev = 0;
    if (!cuda_ok(cudaSetDevice(dev), "set device")) return 0;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
        fprintf(stderr, "ds4: CUDA backend initialized on %s (sm_%d%d)\n",
                prop.name, prop.major, prop.minor);
    }
    if (!g_mmq_prefill_ready) {
        g_mmq_prefill_ready = ds4_mmq_init(dev) == 0;
    }
    if (!g_cublas_ready) {
        if (!cublas_ok(cublasCreate(&g_cublas), "create handle")) return 0;
#ifdef DS4_CUDA_TOKEN_GRAPH_BUILD
        if (!cublas_ok(cublasSetStream(g_cublas, cudaStreamPerThread),
                       "set per-thread default stream")) {
            (void)cublasDestroy(g_cublas);
            g_cublas = NULL;
            return 0;
        }
#endif
        const cublasMath_t math_mode =
            (g_quality_mode || getenv("DS4_CUDA_NO_TF32") != NULL)
                ? CUBLAS_DEFAULT_MATH
                : CUBLAS_TF32_TENSOR_OP_MATH;
        (void)cublasSetMathMode(g_cublas, math_mode);
        const int tiny_tc_enabled =
            getenv("DS4_CUDA_MTP_TENSOR_CORES") != NULL ||
            getenv("DS4_CUDA_DSPARK_TENSOR_CORES") != NULL ||
            getenv("DS4_CUDA_TINY_TENSOR_CORES") != NULL;
        if (tiny_tc_enabled) {
            int workspace_present = 0;
            uint64_t workspace_bytes = cuda_parse_mib_env(
                    "DS4_CUDA_DSPARK_TC_WORKSPACE_MB", &workspace_present);
            if (!workspace_present) {
                workspace_bytes = cuda_parse_mib_env(
                        "DS4_CUDA_MTP_TC_WORKSPACE_MB", &workspace_present);
            }
            if (!workspace_present) {
                workspace_bytes = cuda_parse_mib_env(
                        "DS4_CUDA_TINY_TC_WORKSPACE_MB", &workspace_present);
            }
            if (!workspace_present) workspace_bytes = 64ull * 1048576ull;
            if (workspace_bytes != 0 && workspace_bytes <= (uint64_t)SIZE_MAX) {
                void *workspace = NULL;
                cudaError_t workspace_err = cudaMalloc(
                        &workspace, (size_t)workspace_bytes);
                if (workspace_err == cudaSuccess &&
                    cublasSetWorkspace(g_cublas, workspace,
                                       (size_t)workspace_bytes) ==
                        CUBLAS_STATUS_SUCCESS) {
                    g_cublas_workspace = workspace;
                    g_cublas_workspace_bytes = workspace_bytes;
                    fprintf(stderr,
                            "ds4: CUDA tiny-batch Tensor Core cuBLAS workspace %.2f MiB\n",
                            (double)workspace_bytes / 1048576.0);
                } else {
                    if (workspace) (void)cudaFree(workspace);
                    (void)cudaGetLastError();
                    fprintf(stderr,
                            "ds4: CUDA tiny-batch Tensor Core workspace unavailable; "
                            "using cuBLAS default workspace\n");
                }
            }
        }
        g_cublas_ready = 1;
    }
    if (getenv("DS4_CUDA_Q8_U16_LOADS") != NULL) {
        (void)cuda_q8_u16_validate();
    }
    /* The aligned verifier path is the default target MoE path for DSpark
     * K+1 batches, so prepare its byte-identical IQ2 lookup table before any
     * CUDA graph capture starts. */
    (void)cuda_moe_gb10_validate_signs();
    return 1;
}

extern "C" void ds4_gpu_cleanup(void) {
    (void)cudaDeviceSynchronize();
    g_mmq_prefill_ready = 0;
    g_mmq_prefill_notice = 0;
    g_mmq_prefill_fallback_notice = 0;
    cuda_moe_aligned_clear();
    cuda_token_graph_release();
    if (g_cublas_ready) {
        (void)cublasDestroy(g_cublas);
        g_cublas_ready = 0;
        g_cublas = NULL;
    }
    if (g_cublas_workspace) {
        (void)cudaFree(g_cublas_workspace);
        g_cublas_workspace = NULL;
        g_cublas_workspace_bytes = 0;
    }
    cuda_stream_selected_cache_release();
    cuda_stream_expert_cache_release_all();
    cuda_stream_selected_stage_release();
    cuda_model_range_release_all();
    cuda_model_load_progress_reset();
    cuda_q8_f16_cache_release_all();
    g_q8_f16_disabled_after_oom = 0;
    g_q8_f16_budget_notice_printed = 0;
    for (const cuda_q8_f32_range &r : g_q8_f32_ranges) {
        (void)cudaFree(r.device_ptr);
    }
    g_q8_f32_ranges.clear();
    g_q8_f32_by_offset.clear();
    g_q8_f32_bytes = 0;
    if (g_cuda_tmp) {
        (void)cudaFree(g_cuda_tmp);
        g_cuda_tmp = NULL;
        g_cuda_tmp_bytes = 0;
    }
    if (g_attention_tokentile_scratch) {
        (void)cudaFree(g_attention_tokentile_scratch);
        g_attention_tokentile_scratch = NULL;
        g_attention_tokentile_scratch_bytes = 0;
    }
    if (g_mtp_tc_scratch) {
        (void)cudaFree(g_mtp_tc_scratch);
        g_mtp_tc_scratch = NULL;
        g_mtp_tc_scratch_bytes = 0;
    }
    cuda_model_stage_pool_release();
    if (g_model_device_owned && g_model_device_base) {
        (void)cudaFree((void *)g_model_device_base);
    }
    if (g_model_registered && g_model_host_base) {
        (void)cudaHostUnregister((void *)g_model_host_base);
    }
    g_model_host_base = NULL;
    g_model_device_base = NULL;
    g_model_registered_size = 0;
    g_model_registered = 0;
    g_model_device_owned = 0;
    g_model_range_mapping_supported = 1;
    g_model_hmm_direct = 0;
    g_model_fd = -1;
    if (g_model_direct_fd >= 0) {
        (void)close(g_model_direct_fd);
        g_model_direct_fd = -1;
    }
    g_model_direct_align = 1;
    g_model_file_size = 0;
    g_model_cache_full = 0;
    g_model_mapping_failure_notice_printed = 0;
    g_ssd_streaming_mode = 0;
    if (g_model_prefetch_stream) {
        (void)cudaStreamDestroy(g_model_prefetch_stream);
        g_model_prefetch_stream = NULL;
    }
}

__global__ static void fill_f32_kernel(float *x, uint64_t n, float v);

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    if (bytes == 0) bytes = 1;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    if (!cuda_ok(cudaMalloc(&t->ptr, (size_t)bytes), "tensor alloc")) {
        free(t);
        return NULL;
    }
    t->bytes = bytes;
    t->owner = 1;
    return t;
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t bytes) {
    if (bytes == 0) bytes = 1;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    if (!cuda_ok(cudaMallocManaged(&t->ptr, (size_t)bytes), "managed tensor alloc")) {
        free(t);
        return NULL;
    }
    t->bytes = bytes;
    t->owner = 1;
    return t;
}

static uint64_t cuda_managed_kv_reserve_bytes(uint64_t total_bytes) {
    const uint64_t min_reserve = 8ull * 1073741824ull;
    const uint64_t max_reserve = 40ull * 1073741824ull;
    uint64_t reserve = total_bytes / 4u;
    if (reserve < min_reserve) reserve = min_reserve;
    if (reserve > max_reserve) reserve = max_reserve;
    return reserve;
}

extern "C" int ds4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes) {
    if (kv_cache_bytes == 0) return 0;

    /* Very large KV caches are where device-only cudaMalloc() can make a
     * unified-memory machine unresponsive.  Managed memory restores the old
     * demand-paged behavior for this one long-lived allocation class only. */
    const uint64_t huge_kv = 8ull * 1073741824ull;
    if (kv_cache_bytes >= huge_kv) return 1;

    const uint64_t large_context = 8ull * 1073741824ull;
    if (context_bytes < large_context) return 0;

    size_t free_b = 0;
    size_t total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }

    const uint64_t free_bytes = (uint64_t)free_b;
    const uint64_t total_bytes = (uint64_t)total_b;
    const uint64_t reserve_bytes = cuda_managed_kv_reserve_bytes(total_bytes);
    if (context_bytes > free_bytes) return 1;
    return free_bytes - context_bytes < reserve_bytes;
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes) {
    if (!base || offset > base->bytes || bytes > base->bytes - offset) return NULL;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->ptr = (char *)base->ptr + offset;
    t->bytes = bytes;
    t->owner = 0;
    return t;
}

extern "C" void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor) {
    if (!tensor) return;
    if (tensor->owner && tensor->ptr) (void)cudaFree(tensor->ptr);
    free(tensor);
}

extern "C" uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor) {
    return tensor ? tensor->bytes : 0;
}

extern "C" void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor) {
    if (!tensor) return NULL;
    (void)cudaDeviceSynchronize();
    return tensor->ptr;
}

extern "C" int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value, uint64_t count) {
    if (!tensor || count > tensor->bytes / sizeof(float)) return 0;
    if (count == 0) return 1;
    fill_f32_kernel<<<(count + 255u) / 256u, 256>>>((float *)tensor->ptr, count, value);
    return cuda_ok(cudaGetLastError(), "tensor fill f32 launch");
}

extern "C" int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes) {
    if (!tensor || !data || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    return cuda_ok(cudaMemcpy((char *)tensor->ptr + offset, data, (size_t)bytes, cudaMemcpyHostToDevice), "tensor write");
}

extern "C" int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes) {
    if (!tensor || !data || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    return cuda_ok(cudaMemcpy(data, (const char *)tensor->ptr + offset, (size_t)bytes, cudaMemcpyDeviceToHost), "tensor read");
}

extern "C" int ds4_gpu_tensor_read_after_selected_event(const ds4_gpu_tensor *tensor,
                                                         uint64_t offset,
                                                         void *data,
                                                         uint64_t bytes,
                                                         uint64_t event_value,
                                                         const char *label) {
    (void)event_value;
    (void)label;
    return ds4_gpu_tensor_read(tensor, offset, data, bytes);
}

extern "C" int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                                     const ds4_gpu_tensor *src, uint64_t src_offset,
                                     uint64_t bytes) {
    if (!dst || !src || dst_offset > dst->bytes || src_offset > src->bytes ||
        bytes > dst->bytes - dst_offset || bytes > src->bytes - src_offset) {
        return 0;
    }
    if (bytes == 0) return 1;
    char *dst_ptr = (char *)dst->ptr + dst_offset;
    const char *src_ptr = (const char *)src->ptr + src_offset;
    /* cudaMemcpy() is not permitted while a stream is being captured.  The
     * verifier uses this helper to preserve the prefix-1 compressor frontier,
     * so record an ordered D2D memcpy node in either CUDA graph.  Keep the
     * existing synchronous semantics everywhere else. */
    if (g_token_graph_capturing || g_mtp_graph_capturing) {
        return cuda_ok(cudaMemcpyAsync(dst_ptr,
                                       src_ptr,
                                       (size_t)bytes,
                                       cudaMemcpyDeviceToDevice,
                                       cudaStreamPerThread),
                       "tensor copy async");
    }
    return cuda_ok(cudaMemcpy(dst_ptr,
                              src_ptr,
                              (size_t)bytes,
                              cudaMemcpyDeviceToDevice),
                   "tensor copy");
}

extern "C" int ds4_gpu_begin_commands(void) { return 1; }
extern "C" int ds4_gpu_flush_commands(void) { return cuda_ok(cudaDeviceSynchronize(), "flush"); }

static int cuda_token_graph_timing_enabled(void) {
#ifdef DS4_CUDA_TOKEN_GRAPH_BUILD
    static int enabled = -1;
    if (enabled < 0) {
        enabled = getenv("DS4_CUDA_TOKEN_GRAPH_TIMING") != NULL ? 1 : 0;
    }
    return enabled;
#else
    return 0;
#endif
}

static int cuda_token_graph_pipeline_requested(void) {
#ifdef DS4_CUDA_TOKEN_GRAPH_BUILD
    return getenv("DS4_CUDA_TOKEN_GRAPH_PIPELINE") != NULL;
#else
    return 0;
#endif
}

/* Nsight range capture and the detailed graph timer both own global capture
 * bookkeeping.  Keep the background prepare path disabled in those diagnostic
 * modes; the ordinary graph remains fully available for profiling. */
static int cuda_token_graph_pipeline_allowed(void) {
    if (!cuda_token_graph_pipeline_requested()) return 0;
    if (cuda_token_graph_timing_enabled()) return 0;
    if (getenv("DS4_CUDA_NSYS_CAPTURE_START_POS") != NULL) return 0;
    const char *prefill_capture = getenv("DS4_CUDA_NSYS_PREFILL_START_POS");
    if (prefill_capture != NULL && prefill_capture[0] != '\0') return 0;
    prefill_capture = getenv("DS4_CUDA_NSYS_PREFILL_START_POSITIONS");
    if (prefill_capture != NULL && prefill_capture[0] != '\0') return 0;
    return 1;
}

static int cuda_token_graph_pipeline_ensure_token(void) {
#ifndef DS4_CUDA_TOKEN_GRAPH_BUILD
    return 0;
#else
    if (!cuda_token_graph_pipeline_allowed()) return 0;
    if (g_token_graph_token_device && g_token_graph_token_host) return 1;

    uint32_t *device_ptr = NULL;
    uint32_t *host_ptr = NULL;
    cudaError_t err = cudaMalloc((void **)&device_ptr, sizeof(*device_ptr));
    if (err == cudaSuccess) {
        err = cudaMallocHost((void **)&host_ptr, sizeof(*host_ptr));
    }
    if (err != cudaSuccess) {
        if (device_ptr) (void)cudaFree(device_ptr);
        if (host_ptr) (void)cudaFreeHost(host_ptr);
        fprintf(stderr,
                "ds4: CUDA token graph pipeline token state allocation failed: %s; "
                "using capture/update on the request thread\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    g_token_graph_token_device = device_ptr;
    g_token_graph_token_host = host_ptr;
    return 1;
#endif
}

static int cuda_token_graph_upload_token(uint32_t token) {
#ifndef DS4_CUDA_TOKEN_GRAPH_BUILD
    (void)token;
    return 0;
#else
    if (!g_token_graph_token_device || !g_token_graph_token_host) return 0;
    *g_token_graph_token_host = token;
    return cuda_ok(cudaMemcpyAsync(g_token_graph_token_device,
                                   g_token_graph_token_host,
                                   sizeof(*g_token_graph_token_host),
                                   cudaMemcpyHostToDevice,
                                   cudaStreamPerThread),
                   "token graph dynamic token upload");
#endif
}

static void cuda_nsys_capture_init(void) {
#ifdef DS4_CUDA_TOKEN_GRAPH_BUILD
    if (g_nsys_capture.initialized) return;
    g_nsys_capture.initialized = 1;

    const char *start_env = getenv("DS4_CUDA_NSYS_CAPTURE_START_POS");
    const char *tokens_env = getenv("DS4_CUDA_NSYS_CAPTURE_TOKENS");
    if (!start_env || !start_env[0] || !tokens_env || !tokens_env[0]) return;

    char *start_end = NULL;
    char *tokens_end = NULL;
    unsigned long long start = strtoull(start_env, &start_end, 10);
    unsigned long long tokens = strtoull(tokens_env, &tokens_end, 10);
    if (start_end == start_env || *start_end != '\0' ||
        tokens_end == tokens_env || *tokens_end != '\0' ||
        start > UINT32_MAX || tokens == 0 || tokens > UINT32_MAX) {
        fprintf(stderr,
                "ds4: invalid Nsight capture window start=%s tokens=%s; disabled\n",
                start_env, tokens_env);
        return;
    }

    g_nsys_capture.start_pos = (uint32_t)start;
    g_nsys_capture.token_limit = (uint32_t)tokens;
    g_nsys_capture.enabled = 1;
#endif
}

static void cuda_nsys_capture_maybe_start(uint32_t pos) {
#ifdef DS4_CUDA_TOKEN_GRAPH_BUILD
    cuda_nsys_capture_init();
    if (!g_nsys_capture.enabled || g_nsys_capture.started ||
        g_nsys_capture.stopped || pos < g_nsys_capture.start_pos) {
        return;
    }

    cudaError_t err = cudaProfilerStart();
    if (err != cudaSuccess) {
        fprintf(stderr,
                "ds4: CUDA Nsight capture start failed at pos=%u: %s; disabled\n",
                pos, cudaGetErrorString(err));
        g_nsys_capture.stopped = 1;
        (void)cudaGetLastError();
        return;
    }

    g_nsys_capture.started = 1;
    fprintf(stderr,
            "ds4: CUDA Nsight capture started pos=%u tokens=%u\n",
            pos, g_nsys_capture.token_limit);
#else
    (void)pos;
#endif
}

static void cuda_nsys_capture_stop(const char *reason) {
#ifdef DS4_CUDA_TOKEN_GRAPH_BUILD
    if (!g_nsys_capture.started || g_nsys_capture.stopped) return;

    cudaError_t err = cudaProfilerStop();
    g_nsys_capture.stopped = 1;
    if (err == cudaSuccess) {
        fprintf(stderr,
                "ds4: CUDA Nsight capture stopped after %u tokens reason=%s\n",
                g_nsys_capture.captured_tokens,
                reason ? reason : "requested");
    } else {
        fprintf(stderr,
                "ds4: CUDA Nsight capture stop failed after %u tokens: %s\n",
                g_nsys_capture.captured_tokens, cudaGetErrorString(err));
        (void)cudaGetLastError();
    }
#else
    (void)reason;
#endif
}

static void cuda_nsys_capture_note_tokens(uint32_t tokens) {
#ifdef DS4_CUDA_TOKEN_GRAPH_BUILD
    if (!g_nsys_capture.started || g_nsys_capture.stopped || tokens == 0u) return;
    const uint64_t total = (uint64_t)g_nsys_capture.captured_tokens + tokens;
    g_nsys_capture.captured_tokens = total > UINT32_MAX
        ? UINT32_MAX : (uint32_t)total;
    if (g_nsys_capture.captured_tokens < g_nsys_capture.token_limit) return;
    cuda_nsys_capture_stop("window-complete");
#else
    (void)tokens;
#endif
}

static void cuda_nsys_capture_note_readback(void) {
    if (!g_nsys_decode_cycle_active) {
        cuda_nsys_capture_note_tokens(1u);
    }
}

static void cuda_nsys_prefill_capture_init(void) {
    if (g_nsys_prefill_capture.initialized) return;
    g_nsys_prefill_capture.initialized = 1;

    const char *start_env = getenv("DS4_CUDA_NSYS_PREFILL_START_POSITIONS");
    if (!start_env || !start_env[0]) {
        start_env = getenv("DS4_CUDA_NSYS_PREFILL_START_POS");
    }
    if (!start_env || !start_env[0]) return;
    if (getenv("DS4_CUDA_NSYS_CAPTURE_START_POS") != NULL) {
        fprintf(stderr,
                "ds4: CUDA Nsight prefill capture disabled: decode and prefill "
                "capture windows cannot be combined\n");
        g_nsys_prefill_capture.stopped = 1;
        return;
    }

    const char *cursor = start_env;
    uint32_t previous = 0;
    while (*cursor) {
        if (g_nsys_prefill_capture.start_count ==
            sizeof(g_nsys_prefill_capture.start_positions) /
                sizeof(g_nsys_prefill_capture.start_positions[0])) {
            fprintf(stderr,
                    "ds4: too many Nsight prefill capture positions in %s; disabled\n",
                    start_env);
            g_nsys_prefill_capture.stopped = 1;
            return;
        }
        errno = 0;
        char *end = NULL;
        const unsigned long long start = strtoull(cursor, &end, 10);
        if (errno == ERANGE || end == cursor || start > UINT32_MAX ||
            (*end != '\0' && *end != ',') ||
            (g_nsys_prefill_capture.start_count != 0u && start <= previous) ||
            (*end == ',' && end[1] == '\0')) {
            fprintf(stderr,
                    "ds4: invalid Nsight prefill capture positions=%s; "
                    "expected a strictly increasing comma-separated list\n",
                    start_env);
            g_nsys_prefill_capture.stopped = 1;
            return;
        }
        g_nsys_prefill_capture.start_positions[
            g_nsys_prefill_capture.start_count++] = (uint32_t)start;
        previous = (uint32_t)start;
        if (*end == '\0') break;
        cursor = end + 1;
    }
    if (g_nsys_prefill_capture.start_count == 0u) {
        fprintf(stderr,
                "ds4: invalid Nsight prefill capture positions=%s; disabled\n",
                start_env);
        g_nsys_prefill_capture.stopped = 1;
        return;
    }

    g_nsys_prefill_capture.enabled = 1;
#if !DS4_CUDA_HAS_NVTX
    fprintf(stderr,
            "ds4: CUDA Toolkit NVTX3 headers were unavailable at build time; "
            "prefill capture will contain CUDA kernels without DS4 ranges\n");
#endif
}

static void cuda_nsys_prefill_capture_maybe_start(
        uint32_t pos0,
        uint32_t n_tokens) {
    cuda_nsys_prefill_capture_init();
    if (!g_nsys_prefill_capture.enabled ||
        g_nsys_prefill_capture.started ||
        g_nsys_prefill_capture.stopped ||
        g_nsys_prefill_capture.start_index >=
            g_nsys_prefill_capture.start_count ||
        pos0 < g_nsys_prefill_capture.start_positions[
            g_nsys_prefill_capture.start_index]) {
        return;
    }

    const cudaError_t err = cudaProfilerStart();
    if (err != cudaSuccess) {
        fprintf(stderr,
                "ds4: CUDA Nsight prefill capture start failed pos=%u tokens=%u: %s; disabled\n",
                pos0, n_tokens, cudaGetErrorString(err));
        g_nsys_prefill_capture.stopped = 1;
        (void)cudaGetLastError();
        return;
    }

    g_nsys_prefill_capture.started = 1;
    g_nsys_prefill_capture.active = 1;
    g_nsys_prefill_capture.chunk_pos = pos0;
    g_nsys_prefill_capture.chunk_tokens = n_tokens;
    fprintf(stderr,
            "ds4: CUDA Nsight prefill capture started window=%u/%u pos=%u tokens=%u\n",
            g_nsys_prefill_capture.start_index + 1u,
            g_nsys_prefill_capture.start_count,
            pos0,
            n_tokens);
}

static void cuda_nsys_prefill_capture_stop(const char *reason) {
    if (!g_nsys_prefill_capture.active || g_nsys_prefill_capture.stopped) return;

    const cudaError_t err = cudaProfilerStop();
    g_nsys_prefill_capture.active = 0;
    if (err == cudaSuccess) {
        fprintf(stderr,
                "ds4: CUDA Nsight prefill capture stopped window=%u/%u "
                "pos=%u tokens=%u reason=%s\n",
                g_nsys_prefill_capture.start_index + 1u,
                g_nsys_prefill_capture.start_count,
                g_nsys_prefill_capture.chunk_pos,
                g_nsys_prefill_capture.chunk_tokens,
                reason ? reason : "requested");
        if (reason && strcmp(reason, "chunk-complete") == 0) {
            g_nsys_prefill_capture.start_index++;
            g_nsys_prefill_capture.started = 0;
            if (g_nsys_prefill_capture.start_index >=
                g_nsys_prefill_capture.start_count) {
                g_nsys_prefill_capture.stopped = 1;
            }
        } else {
            g_nsys_prefill_capture.stopped = 1;
        }
    } else {
        g_nsys_prefill_capture.stopped = 1;
        fprintf(stderr,
                "ds4: CUDA Nsight prefill capture stop failed pos=%u tokens=%u: %s\n",
                g_nsys_prefill_capture.chunk_pos,
                g_nsys_prefill_capture.chunk_tokens,
                cudaGetErrorString(err));
        (void)cudaGetLastError();
    }
}

extern "C" void ds4_gpu_nvtx_range_push(const char *name, uint64_t payload) {
    cuda_nvtx_push(name, payload);
}

extern "C" void ds4_gpu_nvtx_range_pop(void) {
    cuda_nvtx_pop();
}

extern "C" void ds4_gpu_nsys_decode_cycle_begin(uint32_t pos) {
    cuda_nsys_capture_maybe_start(pos);
    g_nsys_decode_cycle_active = 1;
    cuda_nvtx_push("ds4/decode/dspark/cycle", cuda_nvtx_payload(pos, 0u));
}

extern "C" void ds4_gpu_nsys_decode_cycle_end(uint32_t emitted_tokens) {
    cuda_nvtx_pop();
    g_nsys_decode_cycle_active = 0;
    cuda_nsys_capture_note_tokens(emitted_tokens);
}

extern "C" void ds4_gpu_prefill_trace_begin(uint32_t pos0, uint32_t n_tokens) {
    cuda_nsys_prefill_capture_maybe_start(pos0, n_tokens);
    cuda_nvtx_push("ds4/prefill/chunk", cuda_nvtx_payload(pos0, n_tokens));
}

extern "C" void ds4_gpu_prefill_trace_end(
        uint32_t pos0,
        uint32_t n_tokens,
        bool success) {
    (void)pos0;
    (void)n_tokens;
    cuda_nvtx_pop();
    cuda_nsys_prefill_capture_stop(success ? "chunk-complete" : "chunk-failed");
}

static uint64_t cuda_token_graph_timing_interval(void) {
    static uint64_t interval = 0;
    if (interval != 0) return interval;
    interval = 500;
    const char *env = getenv("DS4_CUDA_TOKEN_GRAPH_TIMING_EVERY");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long long value = strtoull(env, &end, 10);
        if (end != env && value >= 10u && value <= 100000u) {
            interval = (uint64_t)value;
        }
    }
    return interval;
}

static void cuda_token_graph_timing_reset_aggregate(void) {
    memset(&g_token_graph_timing.aggregate, 0,
           sizeof(g_token_graph_timing.aggregate));
}

static void cuda_token_graph_timing_report(const char *reason) {
    cuda_token_graph_timing_aggregate *a = &g_token_graph_timing.aggregate;
    if (a->tokens == 0) return;
    const double inv = 1.0 / (double)a->tokens;
    const double gpu_ms = a->gpu_samples != 0
        ? a->gpu_execute_ms / (double)a->gpu_samples
        : 0.0;
    const double sampling_ms = a->sampling_samples != 0
        ? a->sampling_ms / (double)a->sampling_samples
        : 0.0;
    const double read_wait_ms = a->read_wait_ms * inv;
    const double read_tail_ms = read_wait_ms > gpu_ms
        ? read_wait_ms - gpu_ms
        : 0.0;
    const double eval_ms = a->eval_ms * inv;
    fprintf(stderr,
            "ds4: CUDA token timing reason=%s tokens=%llu pos=%u..%u "
            "begin=%.3fms encode=%.3fms end_capture=%.3fms "
            "update=%.3fms rebuild=%.3fms bookkeeping=%.3fms "
            "launch=%.3fms gpu=%.3fms read_wait=%.3fms "
            "read_tail_est=%.3fms eval=%.3fms sampling=%.3fms "
            "total=%.3fms updates=%llu rebuilds=%llu samples=%llu\n",
            reason ? reason : "interval",
            (unsigned long long)a->tokens,
            a->first_pos,
            a->last_pos,
            a->capture_begin_ms * inv,
            a->host_encode_ms * inv,
            a->capture_end_ms * inv,
            a->update_ms * inv,
            a->rebuild_ms * inv,
            a->bookkeeping_ms * inv,
            a->launch_submit_ms * inv,
            gpu_ms,
            read_wait_ms,
            read_tail_ms,
            eval_ms,
            sampling_ms,
            eval_ms + sampling_ms,
            (unsigned long long)a->update_ops,
            (unsigned long long)a->rebuild_ops,
            (unsigned long long)a->sampling_samples);
    cuda_token_graph_timing_reset_aggregate();
}

static void cuda_token_graph_timing_maybe_report(const char *reason) {
    if (g_token_graph_timing.aggregate.tokens >=
        cuda_token_graph_timing_interval()) {
        cuda_token_graph_timing_report(reason);
    }
}

static void cuda_token_graph_timing_prepare(uint32_t pos) {
    if (!cuda_token_graph_timing_enabled()) return;
    if (g_token_graph_timing.sample_pending) {
        /* Some benchmark paths consume logits without calling the public
         * sampler.  Keep their GPU/eval timings and report sampling only for
         * tokens for which ds4_session_sample() was actually observed. */
        g_token_graph_timing.sample_pending = 0;
        cuda_token_graph_timing_maybe_report("interval");
    }
    cuda_token_graph_timing_aggregate *a = &g_token_graph_timing.aggregate;
    if (a->tokens != 0 && pos != a->last_pos + 1u) {
        cuda_token_graph_timing_report("context-jump");
    }
    if (!g_token_graph_timing.notice_printed) {
        fprintf(stderr,
                "ds4: CUDA token graph timing enabled (interval=%llu; "
                "read_wait overlaps gpu, read_tail_est is approximate)\n",
                (unsigned long long)cuda_token_graph_timing_interval());
        g_token_graph_timing.notice_printed = 1;
    }
    if (!g_token_graph_timing.events_ready &&
        !g_token_graph_timing.events_failed) {
        cudaError_t start_err = cudaEventCreate(&g_token_graph_timing.gpu_start);
        cudaError_t stop_err = start_err == cudaSuccess
            ? cudaEventCreate(&g_token_graph_timing.gpu_stop)
            : start_err;
        if (start_err == cudaSuccess && stop_err == cudaSuccess) {
            g_token_graph_timing.events_ready = 1;
        } else {
            g_token_graph_timing.events_failed = 1;
            if (g_token_graph_timing.gpu_start) {
                (void)cudaEventDestroy(g_token_graph_timing.gpu_start);
                g_token_graph_timing.gpu_start = NULL;
            }
            if (g_token_graph_timing.gpu_stop) {
                (void)cudaEventDestroy(g_token_graph_timing.gpu_stop);
                g_token_graph_timing.gpu_stop = NULL;
            }
            fprintf(stderr,
                    "ds4: CUDA token graph timing events unavailable: %s; "
                    "continuing with host timings\n",
                    cudaGetErrorString(start_err != cudaSuccess ? start_err : stop_err));
            (void)cudaGetLastError();
        }
    }
}

static void cuda_mtp_graph_release_family(uint32_t family) {
#ifdef DS4_CUDA_TOKEN_GRAPH_BUILD
    for (uint32_t i = 0; i < DS4_CUDA_MTP_GRAPH_VARIANTS; i++) {
        if (cuda_mtp_graph_family_bit(i) != family) continue;
        if (g_mtp_graph_exec[i]) {
            (void)cudaGraphExecDestroy(g_mtp_graph_exec[i]);
            g_mtp_graph_exec[i] = NULL;
        }
    }
#else
    (void)family;
#endif
}

static void cuda_mtp_graph_release(void) {
#ifdef DS4_CUDA_TOKEN_GRAPH_BUILD
    if (g_mtp_graph_capturing) {
        cudaGraph_t graph = NULL;
        (void)cudaStreamEndCapture(cudaStreamPerThread, &graph);
        if (graph) (void)cudaGraphDestroy(graph);
        g_mtp_graph_capturing = 0;
    }
    for (uint32_t i = 0; i < DS4_CUDA_MTP_GRAPH_VARIANTS; i++) {
        if (g_mtp_graph_exec[i]) {
            (void)cudaGraphExecDestroy(g_mtp_graph_exec[i]);
            g_mtp_graph_exec[i] = NULL;
        }
    }
#endif
    g_mtp_graph_capture_variant = 0;
    g_mtp_graph_warm_mask = 0;
    g_mtp_graph_disabled_families = 0;
    g_mtp_graph_notice = 0;
    g_mtp_graph_launches = 0;
    g_mtp_graph_draft_launches = 0;
    g_mtp_graph_verifier_launches = 0;
    g_dspark_graph_verifier_launches = 0;
    g_dspark_graph_draft_launches = 0;
    g_mtp_graph_updates = 0;
    g_mtp_graph_rebuilds = 0;
}

static void cuda_token_graph_release(void) {
    cuda_mtp_graph_release();
#ifdef DS4_CUDA_TOKEN_GRAPH_BUILD
    cuda_nsys_capture_stop("graph-release");
#endif
#ifdef DS4_CUDA_TOKEN_GRAPH_BUILD
    if (cuda_token_graph_timing_enabled()) {
        g_token_graph_timing.sample_pending = 0;
        cuda_token_graph_timing_report("release");
    }
    if (g_token_graph_timing.gpu_start) {
        (void)cudaEventDestroy(g_token_graph_timing.gpu_start);
    }
    if (g_token_graph_timing.gpu_stop) {
        (void)cudaEventDestroy(g_token_graph_timing.gpu_stop);
    }
    memset(&g_token_graph_timing, 0, sizeof(g_token_graph_timing));
#endif
#ifdef DS4_CUDA_TOKEN_GRAPH_BUILD
    if (g_token_graph_capturing) {
        cudaGraph_t graph = NULL;
        (void)cudaStreamEndCapture(cudaStreamPerThread, &graph);
        if (graph) (void)cudaGraphDestroy(graph);
        g_token_graph_capturing = 0;
    }
    for (uint32_t i = 0; i < DS4_CUDA_TOKEN_GRAPH_VARIANTS; i++) {
        if (g_token_graph_exec[i]) {
            (void)cudaGraphExecDestroy(g_token_graph_exec[i]);
            g_token_graph_exec[i] = NULL;
        }
    }
    if (g_token_graph_token_device) {
        (void)cudaFree(g_token_graph_token_device);
        g_token_graph_token_device = NULL;
    }
    if (g_token_graph_token_host) {
        (void)cudaFreeHost(g_token_graph_token_host);
        g_token_graph_token_host = NULL;
    }
#endif
    memset(g_token_graph_prepared_valid, 0,
           sizeof(g_token_graph_prepared_valid));
    memset(g_token_graph_prepared_pos, 0,
           sizeof(g_token_graph_prepared_pos));
    g_token_graph_capture_variant = 0;
    g_token_graph_capture_pos = 0;
    g_token_graph_warmed = 0;
    g_token_graph_disabled = 0;
    g_token_graph_prepare_only = 0;
    g_token_graph_capture_dynamic_token = 0;
    g_token_graph_launches = 0;
    g_token_graph_prepared_launches = 0;
    g_token_graph_prepares = 0;
    g_token_graph_updates = 0;
    g_token_graph_rebuilds = 0;
}

extern "C" int ds4_gpu_token_graph_begin(uint32_t variant, uint32_t pos) {
    if (getenv("DS4_CUDA_TOKEN_GRAPH") == NULL) return 0;
#ifndef DS4_CUDA_TOKEN_GRAPH_BUILD
    if (!g_token_graph_build_notice) {
        fprintf(stderr,
                "ds4: CUDA token graph requested but this binary was not built "
                "with make cuda-spark-graph; using normal launches\n");
        g_token_graph_build_notice = 1;
    }
    (void)variant;
    (void)pos;
    return 0;
#else
    if (g_token_graph_disabled || g_token_graph_capturing) return 0;
    if (g_ssd_streaming_mode || !g_model_device_owned) {
        if (!g_token_graph_build_notice) {
            fprintf(stderr,
                    "ds4: CUDA token graph requires an HBM-resident model "
                    "(DS4_CUDA_COPY_MODEL=1); using normal launches\n");
            g_token_graph_build_notice = 1;
        }
        g_token_graph_disabled = 1;
        return 0;
    }
    /* The first real decode token warms temporary arenas and lazy CUDA state;
     * cudaMalloc/cudaFree are deliberately forbidden inside stream capture. */
    if (!g_token_graph_warmed) {
        g_token_graph_warmed = 1;
        g_token_graph_prepare_only = 0;
        return 0;
    }
    if (variant >= DS4_CUDA_TOKEN_GRAPH_VARIANTS) variant = 0;
    const int pipeline_ready = cuda_token_graph_pipeline_ensure_token();
    if (g_token_graph_prepare_only && !pipeline_ready) {
        g_token_graph_prepare_only = 0;
        return 0;
    }
    if (!g_token_graph_prepare_only) {
        g_token_graph_prepared_valid[variant] = 0;
        cuda_nsys_capture_maybe_start(pos);
        cuda_token_graph_timing_prepare(pos);
    }
    const double begin_t0 = cuda_token_graph_timing_enabled()
        ? cuda_wall_sec()
        : 0.0;
    cudaError_t err = cudaStreamBeginCapture(cudaStreamPerThread,
                                              cudaStreamCaptureModeThreadLocal);
    const double begin_t1 = cuda_token_graph_timing_enabled()
        ? cuda_wall_sec()
        : 0.0;
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA token graph capture start failed: %s; disabling graph path\n",
                cudaGetErrorString(err));
        cuda_nsys_capture_stop("graph-capture-start-failed");
        (void)cudaGetLastError();
        g_token_graph_disabled = 1;
        g_token_graph_prepare_only = 0;
        return 0;
    }
    g_token_graph_capture_variant = variant;
    g_token_graph_capture_pos = pos;
    g_token_graph_capturing = 1;
    g_token_graph_capture_dynamic_token = pipeline_ready;
    if (!g_token_graph_prepare_only && cuda_token_graph_timing_enabled()) {
        memset(&g_token_graph_timing.current, 0,
               sizeof(g_token_graph_timing.current));
        g_token_graph_timing.current.active = 1;
        g_token_graph_timing.current.pos = pos;
        g_token_graph_timing.current.capture_begin_ms =
            (begin_t1 - begin_t0) * 1000.0;
        g_token_graph_timing.current.encode_start_sec = begin_t1;
    }
    if (!g_token_graph_build_notice) {
        fprintf(stderr,
                "ds4: CUDA token graph capture enabled (per-thread default stream, %u executable variants)\n",
                (unsigned)DS4_CUDA_TOKEN_GRAPH_VARIANTS);
        g_token_graph_build_notice = 1;
    }
    if (pipeline_ready && !g_token_graph_pipeline_notice) {
        fprintf(stderr,
                "ds4: CUDA token graph look-ahead pipeline enabled "
                "(late-bound token, exact position/KV state)\n");
        g_token_graph_pipeline_notice = 1;
    }
    return 1;
#endif
}

extern "C" int ds4_gpu_token_graph_prepare_begin(uint32_t variant,
                                                   uint32_t pos) {
#ifndef DS4_CUDA_TOKEN_GRAPH_BUILD
    (void)variant;
    (void)pos;
    return 0;
#else
    if (!cuda_token_graph_pipeline_allowed() || g_token_graph_capturing) {
        return 0;
    }
    g_token_graph_prepare_only = 1;
    const int rc = ds4_gpu_token_graph_begin(variant, pos);
    if (rc == 0) g_token_graph_prepare_only = 0;
    return rc;
#endif
}

static int cuda_aux_graph_begin(uint32_t variant, uint32_t pos,
                                const char *enable_env) {
    if (!enable_env || getenv(enable_env) == NULL) return 0;
#ifndef DS4_CUDA_TOKEN_GRAPH_BUILD
    if (!g_mtp_graph_notice) {
        fprintf(stderr,
                "ds4: CUDA speculative graph requested but this binary was "
                "not built with token-graph support; using normal launches\n");
        g_mtp_graph_notice = 1;
    }
    (void)variant;
    (void)pos;
    return 0;
#else
    if (variant >= DS4_CUDA_MTP_GRAPH_VARIANTS) variant = 0;
    const uint32_t family = cuda_mtp_graph_family_bit(variant);
    if ((g_mtp_graph_disabled_families & family) != 0u ||
        g_mtp_graph_capturing ||
        g_token_graph_capturing) {
        return 0;
    }
    if (g_ssd_streaming_mode || !g_model_device_owned) {
        if (!g_mtp_graph_notice) {
            fprintf(stderr,
                    "ds4: CUDA MTP graph requires an HBM-resident target "
                    "model; using normal MTP launches\n");
            g_mtp_graph_notice = 1;
        }
        g_mtp_graph_disabled_families =
            DS4_CUDA_MTP_GRAPH_ALL_FAMILIES;
        return 0;
    }
    /* Every MTP role and every DSpark K has a distinct family.  Warm each
     * topology once outside capture so lazy CUDA/cuBLAS setup cannot poison
     * stream capture. */
    if ((g_mtp_graph_warm_mask & family) == 0u) {
        g_mtp_graph_warm_mask |= family;
        return 0;
    }
    cudaError_t err = cudaStreamBeginCapture(cudaStreamPerThread,
                                              cudaStreamCaptureModeThreadLocal);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "ds4: CUDA MTP graph capture start failed family=%s "
                "at pos=%u: %s; disabling that family\n",
                cuda_mtp_graph_family_name(variant),
                pos,
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        g_mtp_graph_disabled_families |= family;
        cuda_mtp_graph_release_family(family);
        return 0;
    }
    g_mtp_graph_capture_variant = variant;
    g_mtp_graph_capturing = 1;
    if (!g_mtp_graph_notice) {
        fprintf(stderr,
                "ds4: CUDA speculative auxiliary graphs enabled "
                "(per-thread stream, %u topology variants)\n",
                (unsigned)DS4_CUDA_MTP_GRAPH_VARIANTS);
        g_mtp_graph_notice = 1;
    }
    return 1;
#endif
}

extern "C" int ds4_gpu_mtp_graph_begin(uint32_t variant, uint32_t pos) {
    return cuda_aux_graph_begin(variant, pos, "DS4_CUDA_MTP_GRAPH");
}

extern "C" int ds4_gpu_dspark_graph_begin(uint32_t n_tokens,
                                            uint32_t position_variant,
                                            uint32_t pos) {
    /* Official DSpark verifies the pending current token together with K
     * drafts.  The five graph families therefore represent row counts 2..6. */
    if (n_tokens < 2u || n_tokens > 6u) return 0;
    const uint32_t variant = 16u + (n_tokens - 2u) * 8u +
                             (position_variant & 7u);
    return cuda_aux_graph_begin(variant, pos, "DS4_CUDA_DSPARK_GRAPH");
}

extern "C" int ds4_gpu_dspark_draft_graph_begin(
        uint32_t n_tokens,
        uint32_t position_variant,
        uint32_t pos) {
    if (n_tokens == 0 || n_tokens > 5u) return 0;
    const uint32_t variant = 56u + (n_tokens - 1u) * 8u +
                             (position_variant & 7u);
    return cuda_aux_graph_begin(variant, pos, "DS4_CUDA_DSPARK_GRAPH");
}

extern "C" void ds4_gpu_mtp_graph_abort(void) {
#ifdef DS4_CUDA_TOKEN_GRAPH_BUILD
    if (!g_mtp_graph_capturing) return;
    const uint32_t variant = g_mtp_graph_capture_variant;
    cudaGraph_t graph = NULL;
    cudaError_t err = cudaStreamEndCapture(cudaStreamPerThread, &graph);
    if (graph) (void)cudaGraphDestroy(graph);
    g_mtp_graph_capturing = 0;
    g_mtp_graph_disabled_families |=
        cuda_mtp_graph_family_bit(variant);
    cuda_mtp_graph_release_family(
        cuda_mtp_graph_family_bit(variant));
    fprintf(stderr,
            "ds4: CUDA MTP graph capture aborted family=%s (%s); "
            "using normal launches for that family\n",
            cuda_mtp_graph_family_name(variant),
            err == cudaSuccess ? "encode failure" : cudaGetErrorString(err));
    (void)cudaGetLastError();
#endif
}

extern "C" void ds4_gpu_token_graph_abort(void) {
#ifdef DS4_CUDA_TOKEN_GRAPH_BUILD
    if (!g_token_graph_capturing) return;
    const int prepare_only = g_token_graph_prepare_only;
    const uint32_t variant = g_token_graph_capture_variant;
    cudaGraph_t graph = NULL;
    cudaError_t err = cudaStreamEndCapture(cudaStreamPerThread, &graph);
    if (graph) (void)cudaGraphDestroy(graph);
    g_token_graph_capturing = 0;
    g_token_graph_prepare_only = 0;
    g_token_graph_capture_dynamic_token = 0;
    g_token_graph_timing.current.active = 0;
    g_token_graph_timing.event_pending = 0;
    if (prepare_only) {
        if (variant < DS4_CUDA_TOKEN_GRAPH_VARIANTS) {
            g_token_graph_prepared_valid[variant] = 0;
        }
        fprintf(stderr,
                "ds4: CUDA token graph look-ahead prepare aborted (%s); "
                "next token will use synchronous capture/update\n",
                err == cudaSuccess ? "encode failure" : cudaGetErrorString(err));
    } else {
        g_token_graph_disabled = 1;
        fprintf(stderr,
                "ds4: CUDA token graph capture aborted (%s); falling back to normal launches\n",
                err == cudaSuccess ? "encode failure" : cudaGetErrorString(err));
        cuda_nsys_capture_stop("graph-capture-aborted");
    }
    (void)cudaGetLastError();
#endif
}

extern "C" void ds4_gpu_token_graph_reset(void) {
#ifdef DS4_CUDA_TOKEN_GRAPH_BUILD
    if (!g_token_graph_capturing) {
        (void)cudaStreamSynchronize(cudaStreamPerThread);
    }
#endif
    cuda_token_graph_release();
}

static cudaError_t cuda_token_graph_instantiate(cudaGraphExec_t *exec,
                                                cudaGraph_t graph) {
#if CUDART_VERSION >= 13000
    return cudaGraphInstantiate(exec, graph, 0);
#else
    return cudaGraphInstantiate(exec, graph, NULL, NULL, 0);
#endif
}

extern "C" int ds4_gpu_mtp_graph_end(void) {
#ifndef DS4_CUDA_TOKEN_GRAPH_BUILD
    return 0;
#else
    if (!g_mtp_graph_capturing) return 0;
    const uint32_t variant = g_mtp_graph_capture_variant;
    const uint32_t family = cuda_mtp_graph_family_bit(variant);
    cudaGraph_t graph = NULL;
    cudaError_t err = cudaStreamEndCapture(cudaStreamPerThread, &graph);
    g_mtp_graph_capturing = 0;
    if (err != cudaSuccess || !graph) {
        fprintf(stderr,
                "ds4: CUDA MTP graph capture end failed family=%s: %s; "
                "disabling that family\n",
                cuda_mtp_graph_family_name(variant),
                cudaGetErrorString(err));
        if (graph) (void)cudaGraphDestroy(graph);
        (void)cudaGetLastError();
        g_mtp_graph_disabled_families |= family;
        cuda_mtp_graph_release_family(family);
        return 0;
    }

    size_t node_count = 0;
    (void)cudaGraphGetNodes(graph, NULL, &node_count);
    cudaGraphExec_t *exec = &g_mtp_graph_exec[variant];
    if (*exec) {
#if CUDART_VERSION >= 13000
        cudaGraphExecUpdateResultInfo info = {};
        err = cudaGraphExecUpdate(*exec, graph, &info);
        const bool updated = err == cudaSuccess &&
                             info.result == cudaGraphExecUpdateSuccess;
#else
        cudaGraphNode_t error_node = NULL;
        cudaGraphExecUpdateResult update_result = cudaGraphExecUpdateError;
        err = cudaGraphExecUpdate(*exec, graph, &error_node, &update_result);
        const bool updated = err == cudaSuccess &&
                             update_result == cudaGraphExecUpdateSuccess;
#endif
        if (updated) {
            g_mtp_graph_updates++;
        } else {
            (void)cudaGetLastError();
            (void)cudaGraphExecDestroy(*exec);
            *exec = NULL;
        }
    }
    if (!*exec) {
        err = cuda_token_graph_instantiate(exec, graph);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4: CUDA MTP graph instantiate failed for variant %u: "
                    "%s; disabling family=%s\n",
                    variant,
                    cudaGetErrorString(err),
                    cuda_mtp_graph_family_name(variant));
            (void)cudaGraphDestroy(graph);
            (void)cudaGetLastError();
            g_mtp_graph_disabled_families |= family;
            cuda_mtp_graph_release_family(family);
            return 0;
        }
        g_mtp_graph_rebuilds++;
        fprintf(stderr,
                "ds4: CUDA MTP graph variant=%u nodes=%zu %s\n",
                variant,
                node_count,
                g_mtp_graph_rebuilds <= DS4_CUDA_MTP_GRAPH_VARIANTS
                    ? "instantiated"
                    : "rebuilt after topology change");
    }
    (void)cudaGraphDestroy(graph);

    err = cudaGraphLaunch(*exec, cudaStreamPerThread);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "ds4: CUDA MTP graph launch failed family=%s: %s; "
                "disabling that family\n",
                cuda_mtp_graph_family_name(variant),
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        g_mtp_graph_disabled_families |= family;
        cuda_mtp_graph_release_family(family);
        return -1;
    }
    g_mtp_graph_launches++;
    if (family == DS4_CUDA_MTP_GRAPH_DRAFT_FAMILY) {
        g_mtp_graph_draft_launches++;
    } else if (family == DS4_CUDA_MTP_GRAPH_VERIFY_FAMILY) {
        g_mtp_graph_verifier_launches++;
    } else if (family >= DS4_CUDA_DSPARK_DRAFT_GRAPH_K1_FAMILY) {
        g_dspark_graph_draft_launches++;
    } else {
        g_dspark_graph_verifier_launches++;
    }
    if ((getenv("DS4_CUDA_MTP_GRAPH_VERBOSE") != NULL ||
         getenv("DS4_CUDA_DSPARK_GRAPH_VERBOSE") != NULL) &&
        (g_mtp_graph_launches <= 4u ||
         (g_mtp_graph_launches % 500u) == 0u)) {
        fprintf(stderr,
                "ds4: CUDA speculative graph launches=%llu mtp_draft=%llu "
                "mtp_verify=%llu dspark_draft=%llu dspark_verify=%llu "
                "updates=%llu rebuilds=%llu\n",
                (unsigned long long)g_mtp_graph_launches,
                (unsigned long long)g_mtp_graph_draft_launches,
                (unsigned long long)g_mtp_graph_verifier_launches,
                (unsigned long long)g_dspark_graph_draft_launches,
                (unsigned long long)g_dspark_graph_verifier_launches,
                (unsigned long long)g_mtp_graph_updates,
                (unsigned long long)g_mtp_graph_rebuilds);
    }
    return 1;
#endif
}

extern "C" int ds4_gpu_token_graph_end_token(uint32_t token) {
#ifndef DS4_CUDA_TOKEN_GRAPH_BUILD
    (void)token;
    return 0;
#else
    if (!g_token_graph_capturing) return 0;
    const uint32_t variant = g_token_graph_capture_variant;
    const int prepare_only = g_token_graph_prepare_only;
    const int dynamic_token = g_token_graph_capture_dynamic_token;
    const int timing = !prepare_only && cuda_token_graph_timing_enabled() &&
                       g_token_graph_timing.current.active;
    const double end_t0 = timing ? cuda_wall_sec() : 0.0;
    if (timing) {
        g_token_graph_timing.current.host_encode_ms =
            (end_t0 - g_token_graph_timing.current.encode_start_sec) * 1000.0;
    }
    cudaGraph_t graph = NULL;
    cudaError_t err = cudaStreamEndCapture(cudaStreamPerThread, &graph);
    const double end_t1 = timing ? cuda_wall_sec() : 0.0;
    if (timing) {
        g_token_graph_timing.current.capture_end_ms =
            (end_t1 - end_t0) * 1000.0;
    }
    g_token_graph_capturing = 0;
    g_token_graph_prepare_only = 0;
    g_token_graph_capture_dynamic_token = 0;
    if (err != cudaSuccess || !graph) {
        fprintf(stderr,
                "ds4: CUDA token graph capture end failed%s: %s; %s\n",
                prepare_only ? " during look-ahead prepare" : "",
                cudaGetErrorString(err),
                prepare_only ? "using synchronous capture on the next token"
                             : "disabling graph path");
        if (graph) (void)cudaGraphDestroy(graph);
        if (!prepare_only) cuda_nsys_capture_stop("graph-capture-end-failed");
        (void)cudaGetLastError();
        g_token_graph_timing.current.active = 0;
        g_token_graph_timing.event_pending = 0;
        if (prepare_only) {
            g_token_graph_prepared_valid[variant] = 0;
        } else {
            g_token_graph_disabled = 1;
        }
        return 0;
    }

    size_t node_count = 0;
    (void)cudaGraphGetNodes(graph, NULL, &node_count);
    cudaGraphExec_t *exec = &g_token_graph_exec[variant];
    if (*exec) {
        const double update_t0 = timing ? cuda_wall_sec() : 0.0;
#if CUDART_VERSION >= 13000
        cudaGraphExecUpdateResultInfo info = {};
        err = cudaGraphExecUpdate(*exec, graph, &info);
        const bool updated = err == cudaSuccess &&
                             info.result == cudaGraphExecUpdateSuccess;
#else
        cudaGraphNode_t error_node = NULL;
        cudaGraphExecUpdateResult update_result = cudaGraphExecUpdateError;
        err = cudaGraphExecUpdate(*exec, graph, &error_node, &update_result);
        const bool updated = err == cudaSuccess &&
                             update_result == cudaGraphExecUpdateSuccess;
#endif
        if (timing) {
            g_token_graph_timing.current.update_ms +=
                (cuda_wall_sec() - update_t0) * 1000.0;
            g_token_graph_timing.current.update_ops++;
        }
        if (updated) {
            g_token_graph_updates++;
        } else {
            (void)cudaGetLastError();
            (void)cudaGraphExecDestroy(*exec);
            *exec = NULL;
        }
    }

    if (!*exec) {
        const double rebuild_t0 = timing ? cuda_wall_sec() : 0.0;
        err = cuda_token_graph_instantiate(exec, graph);
        if (timing) {
            g_token_graph_timing.current.rebuild_ms +=
                (cuda_wall_sec() - rebuild_t0) * 1000.0;
            g_token_graph_timing.current.rebuild_ops++;
        }
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4: CUDA token graph instantiate failed for variant %u%s: %s; %s\n",
                    variant,
                    prepare_only ? " during look-ahead prepare" : "",
                    cudaGetErrorString(err),
                    prepare_only ? "using synchronous capture on the next token"
                                 : "disabling graph path");
            (void)cudaGraphDestroy(graph);
            if (!prepare_only) cuda_nsys_capture_stop("graph-instantiate-failed");
            (void)cudaGetLastError();
            g_token_graph_timing.current.active = 0;
            g_token_graph_timing.event_pending = 0;
            if (prepare_only) {
                g_token_graph_prepared_valid[variant] = 0;
            } else {
                g_token_graph_disabled = 1;
            }
            return 0;
        }
        g_token_graph_rebuilds++;
        fprintf(stderr,
                "ds4: CUDA token graph variant=%u nodes=%zu %s\n",
                variant, node_count,
                g_token_graph_rebuilds <= DS4_CUDA_TOKEN_GRAPH_VARIANTS
                    ? "instantiated" : "rebuilt after topology change");
    }
    (void)cudaGraphDestroy(graph);

    if (prepare_only) {
        g_token_graph_prepared_pos[variant] = g_token_graph_capture_pos;
        g_token_graph_prepared_valid[variant] = 1;
        g_token_graph_prepares++;
        if (getenv("DS4_CUDA_TOKEN_GRAPH_PIPELINE_VERBOSE") != NULL &&
            (g_token_graph_prepares <= 4u ||
             (g_token_graph_prepares % 1000u) == 0u)) {
            fprintf(stderr,
                    "ds4: CUDA token graph prepared=%llu pos=%u variant=%u "
                    "updates=%llu rebuilds=%llu\n",
                    (unsigned long long)g_token_graph_prepares,
                    g_token_graph_capture_pos,
                    variant,
                    (unsigned long long)g_token_graph_updates,
                    (unsigned long long)g_token_graph_rebuilds);
        }
        return 1;
    }

    if (dynamic_token && !cuda_token_graph_upload_token(token)) {
        fprintf(stderr,
                "ds4: CUDA token graph dynamic token upload failed; disabling graph path\n");
        cuda_nsys_capture_stop("graph-token-upload-failed");
        g_token_graph_timing.current.active = 0;
        g_token_graph_timing.event_pending = 0;
        g_token_graph_disabled = 1;
        return -1;
    }

    if (timing) {
        const double post_capture_ms = (cuda_wall_sec() - end_t1) * 1000.0;
        double bookkeeping_ms = post_capture_ms -
                                g_token_graph_timing.current.update_ms -
                                g_token_graph_timing.current.rebuild_ms;
        if (bookkeeping_ms < 0.0) bookkeeping_ms = 0.0;
        g_token_graph_timing.current.bookkeeping_ms = bookkeeping_ms;
        g_token_graph_timing.event_pending = 0;
        if (g_token_graph_timing.events_ready) {
            cudaError_t event_err = cudaEventRecord(
                    g_token_graph_timing.gpu_start, cudaStreamPerThread);
            if (event_err == cudaSuccess) {
                g_token_graph_timing.event_pending = 1;
            } else {
                (void)cudaGetLastError();
            }
        }
    }

    const double launch_t0 = timing ? cuda_wall_sec() : 0.0;
    err = cudaGraphLaunch(*exec, cudaStreamPerThread);
    if (timing) {
        g_token_graph_timing.current.launch_submit_ms =
            (cuda_wall_sec() - launch_t0) * 1000.0;
    }
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA token graph launch failed: %s\n",
                cudaGetErrorString(err));
        cuda_nsys_capture_stop("graph-launch-failed");
        (void)cudaGetLastError();
        g_token_graph_timing.current.active = 0;
        g_token_graph_timing.event_pending = 0;
        g_token_graph_disabled = 1;
        return -1;
    }
    if (timing && g_token_graph_timing.event_pending) {
        cudaError_t event_err = cudaEventRecord(
                g_token_graph_timing.gpu_stop, cudaStreamPerThread);
        if (event_err != cudaSuccess) {
            g_token_graph_timing.event_pending = 0;
            (void)cudaGetLastError();
        }
    }
    g_token_graph_launches++;
    if (getenv("DS4_CUDA_TOKEN_GRAPH_VERBOSE") != NULL &&
        (g_token_graph_launches <= 4u || (g_token_graph_launches % 1000u) == 0u)) {
        fprintf(stderr,
                "ds4: CUDA token graph launches=%llu direct=%llu prepares=%llu "
                "updates=%llu rebuilds=%llu\n",
                (unsigned long long)g_token_graph_launches,
                (unsigned long long)g_token_graph_prepared_launches,
                (unsigned long long)g_token_graph_prepares,
                (unsigned long long)g_token_graph_updates,
                (unsigned long long)g_token_graph_rebuilds);
    }
    return 1;
#endif
}

extern "C" int ds4_gpu_token_graph_prepare_end(void) {
#ifndef DS4_CUDA_TOKEN_GRAPH_BUILD
    return 0;
#else
    if (!g_token_graph_capturing || !g_token_graph_prepare_only) return 0;
    return ds4_gpu_token_graph_end_token(0);
#endif
}

extern "C" int ds4_gpu_token_graph_launch_prepared(uint32_t variant,
                                                     uint32_t pos,
                                                     uint32_t token) {
#ifndef DS4_CUDA_TOKEN_GRAPH_BUILD
    (void)variant;
    (void)pos;
    (void)token;
    return 0;
#else
    if (!cuda_token_graph_pipeline_allowed() ||
        g_token_graph_disabled ||
        g_token_graph_capturing ||
        variant >= DS4_CUDA_TOKEN_GRAPH_VARIANTS ||
        !g_token_graph_exec[variant] ||
        !g_token_graph_prepared_valid[variant] ||
        g_token_graph_prepared_pos[variant] != pos) {
        return 0;
    }
    if (!cuda_token_graph_upload_token(token)) return -1;
    cudaError_t err = cudaGraphLaunch(g_token_graph_exec[variant],
                                      cudaStreamPerThread);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "ds4: CUDA prepared token graph launch failed pos=%u variant=%u: %s\n",
                pos,
                variant,
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        g_token_graph_disabled = 1;
        return -1;
    }
    g_token_graph_prepared_valid[variant] = 0;
    g_token_graph_launches++;
    g_token_graph_prepared_launches++;
    if (getenv("DS4_CUDA_TOKEN_GRAPH_PIPELINE_VERBOSE") != NULL &&
        (g_token_graph_prepared_launches <= 4u ||
         (g_token_graph_prepared_launches % 1000u) == 0u)) {
        fprintf(stderr,
                "ds4: CUDA token graph direct launches=%llu total=%llu "
                "prepares=%llu updates=%llu rebuilds=%llu\n",
                (unsigned long long)g_token_graph_prepared_launches,
                (unsigned long long)g_token_graph_launches,
                (unsigned long long)g_token_graph_prepares,
                (unsigned long long)g_token_graph_updates,
                (unsigned long long)g_token_graph_rebuilds);
    }
    return 1;
#endif
}

extern "C" void ds4_gpu_token_graph_note_readback(double readback_ms,
                                                    double eval_ms) {
#ifdef DS4_CUDA_TOKEN_GRAPH_BUILD
    cuda_nsys_capture_note_readback();
    if (!cuda_token_graph_timing_enabled() ||
        !g_token_graph_timing.current.active) {
        return;
    }
    cuda_token_graph_timing_current *c = &g_token_graph_timing.current;
    cuda_token_graph_timing_aggregate *a = &g_token_graph_timing.aggregate;
    float gpu_ms = 0.0f;
    int have_gpu_ms = 0;
    if (g_token_graph_timing.event_pending) {
        cudaError_t event_err = cudaEventElapsedTime(
                &gpu_ms,
                g_token_graph_timing.gpu_start,
                g_token_graph_timing.gpu_stop);
        if (event_err == cudaSuccess) {
            have_gpu_ms = 1;
        } else {
            (void)cudaGetLastError();
        }
    }

    if (a->tokens == 0) a->first_pos = c->pos;
    a->last_pos = c->pos;
    a->tokens++;
    a->capture_begin_ms += c->capture_begin_ms;
    a->host_encode_ms += c->host_encode_ms;
    a->capture_end_ms += c->capture_end_ms;
    a->update_ms += c->update_ms;
    a->rebuild_ms += c->rebuild_ms;
    a->bookkeeping_ms += c->bookkeeping_ms;
    a->launch_submit_ms += c->launch_submit_ms;
    a->read_wait_ms += readback_ms;
    a->eval_ms += eval_ms;
    a->update_ops += c->update_ops;
    a->rebuild_ops += c->rebuild_ops;
    if (have_gpu_ms) {
        a->gpu_execute_ms += (double)gpu_ms;
        a->gpu_samples++;
    }

    c->active = 0;
    g_token_graph_timing.event_pending = 0;
    g_token_graph_timing.sample_pending = 1;
#else
    (void)readback_ms;
    (void)eval_ms;
#endif
}

extern "C" void ds4_gpu_token_graph_note_sampling(double sampling_ms) {
#ifdef DS4_CUDA_TOKEN_GRAPH_BUILD
    if (!cuda_token_graph_timing_enabled() ||
        !g_token_graph_timing.sample_pending) {
        return;
    }
    g_token_graph_timing.aggregate.sampling_ms += sampling_ms;
    g_token_graph_timing.aggregate.sampling_samples++;
    g_token_graph_timing.sample_pending = 0;
    cuda_token_graph_timing_maybe_report("interval");
#else
    (void)sampling_ms;
#endif
}

extern "C" int ds4_gpu_signal_selected_readback_ready(uint64_t *event_value) {
    if (event_value) *event_value = 1;
    return cuda_ok(cudaDeviceSynchronize(), "selected readback signal");
}
extern "C" int ds4_gpu_commit_and_wait_selected_readback(uint64_t event_value, const char *label) {
    (void)event_value;
    return cuda_ok(cudaDeviceSynchronize(), label ? label : "selected readback wait");
}
extern "C" int ds4_gpu_wait_selected_readback_ready(uint64_t event_value, const char *label) {
    (void)event_value;
    (void)label;
    return cuda_ok(cudaDeviceSynchronize(), "selected readback wait");
}
extern "C" int ds4_gpu_end_commands(void) {
    cuda_model_load_progress_finish();
    return cuda_ok(cudaDeviceSynchronize(), "end commands");
}
extern "C" int ds4_gpu_synchronize(void) {
    cuda_model_load_progress_finish();
    return cuda_ok(cudaDeviceSynchronize(), "synchronize");
}

struct cuda_moe_repack_alloc_ctx {
    const char *device_base;
    uint64_t model_size;
};

static bool cuda_moe_repack_alloc(void *opaque, ds4_repack_artifact *art) {
    cuda_moe_repack_alloc_ctx *ctx = (cuda_moe_repack_alloc_ctx *)opaque;
    if (!ctx || !art || !art->t || !ctx->device_base ||
        art->bytes != art->t->bytes ||
        art->t->off > ctx->model_size ||
        art->bytes > ctx->model_size - art->t->off) {
        if (art && art->t) {
            fprintf(stderr,
                    "ds4: CUDA in-place MoE repack rejected %s "
                    "(raw=%llu aligned=%llu)\n",
                    art->t->name.c_str(),
                    (unsigned long long)art->t->bytes,
                    (unsigned long long)art->bytes);
        }
        return false;
    }
    art->dev = (void *)(ctx->device_base + art->t->off);
    return true;
}

static void cuda_moe_repack_noop_free(void *, ds4_repack_artifact *art) {
    if (art) art->dev = NULL;
}

static bool cuda_moe_tensor_suffix(
        const std::string &name, const char *suffix, std::string *prefix) {
    const size_t suffix_len = strlen(suffix);
    if (name.size() <= suffix_len ||
        name.compare(name.size() - suffix_len, suffix_len, suffix) != 0) {
        return false;
    }
    if (prefix) *prefix = name.substr(0, name.size() - suffix_len);
    return true;
}

/* Convert the resident target routed-MoE tensors in place. IQ2_XXS and Q2_K
 * aligned layouts are byte-neutral for the Flash geometry, so the operation
 * changes no allocation and leaves the Q4 DSpark sidecar untouched. */
static int cuda_model_prepare_moe_aligned(
        const void *model_map, uint64_t model_size) {
    if (g_moe_aligned_ready &&
        g_moe_aligned_host_base == model_map &&
        g_moe_aligned_model_size == model_size) {
        return 1;
    }
    if (!g_model_device_owned || !g_model_device_base ||
        model_map != g_model_host_base) {
        return 1;
    }
    if (g_model_fd < 0 ||
        (g_model_fd_host_base && g_model_fd_host_base != model_map)) {
        fprintf(stderr,
                "ds4: CUDA in-place MoE repack skipped: target GGUF fd unavailable\n");
        return 1;
    }

    int dev = 0;
    cudaDeviceProp prop = {};
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaGetDeviceProperties(&prop, dev) != cudaSuccess ||
        prop.major < 12) {
        (void)cudaGetLastError();
        return 1;
    }

    char path[64];
    snprintf(path, sizeof(path), "/proc/self/fd/%d", g_model_fd);
    ds4_repack_file mapped;
    if (!ds4_repack_map_file("ds4", path, mapped)) return 0;
    std::vector<ds4_repack_tensor> records;
    const bool catalog_ok = ds4_repack_collect_catalog(
            "ds4", mapped, NULL, &records);
    ds4_repack_unmap_file(mapped);
    if (!catalog_ok) return 0;

    static const char gate_suffix[] = ".ffn_gate_exps.weight";
    static const char up_suffix[] = ".ffn_up_exps.weight";
    static const char down_suffix[] = ".ffn_down_exps.weight";
    std::unordered_map<std::string, uint32_t> layer_parts;
    uint32_t expected_iq2 = 0;
    uint32_t expected_q2 = 0;
    for (const ds4_repack_tensor &t : records) {
        std::string prefix;
        uint32_t part = 0;
        bool eligible = true;
        if (cuda_moe_tensor_suffix(t.name, gate_suffix, &prefix)) {
            part = 1u;
            eligible = ds4_repack_iq2_candidate(t);
            expected_iq2++;
        } else if (cuda_moe_tensor_suffix(t.name, up_suffix, &prefix)) {
            part = 2u;
            eligible = ds4_repack_iq2_candidate(t);
            expected_iq2++;
        } else if (cuda_moe_tensor_suffix(t.name, down_suffix, &prefix)) {
            part = 4u;
            eligible = ds4_repack_q2k_candidate(t);
            expected_q2++;
        }
        if (!part) continue;
        if (!eligible) {
            fprintf(stderr,
                    "ds4: CUDA in-place MoE repack unsupported tensor %s\n",
                    t.name.c_str());
            return 0;
        }
        layer_parts[prefix] |= part;
    }
    if (layer_parts.empty() || expected_iq2 != 2u * expected_q2) {
        fprintf(stderr,
                "ds4: CUDA in-place MoE repack catalog is incomplete "
                "(iq2=%u q2=%u layers=%zu)\n",
                expected_iq2, expected_q2, layer_parts.size());
        return 0;
    }
    for (const auto &entry : layer_parts) {
        if (entry.second != 7u) {
            fprintf(stderr,
                    "ds4: CUDA in-place MoE repack missing gate/up/down for %s\n",
                    entry.first.c_str());
            return 0;
        }
    }

    cuda_moe_repack_alloc_ctx alloc_ctx = {
        g_model_device_base,
        model_size,
    };
    ds4_repack_build_args args;
    args.log_prefix = "ds4";
    args.model_id = "target-in-place";
    args.path = path;
    args.records = &records;
    args.device = dev;
    args.copy_chunk_bytes = 64ull * 1048576ull;
    args.alloc_fn = cuda_moe_repack_alloc;
    args.free_fn = cuda_moe_repack_noop_free;
    args.alloc_ctx = &alloc_ctx;

    std::vector<ds4_repack_artifact> q2_artifacts;
    std::vector<ds4_repack_artifact> iq2_artifacts;
    uint64_t q2_bytes = 0;
    uint64_t iq2_bytes = 0;
    if (!ds4_repack_build_q2k_aligned(args, q2_artifacts, &q2_bytes) ||
        !ds4_repack_build_iq2_aligned(args, iq2_artifacts, &iq2_bytes)) {
        fprintf(stderr,
                "ds4: CUDA in-place MoE repack failed after target replacement; "
                "aborting startup\n");
        cuda_moe_aligned_clear();
        return 0;
    }
    if (q2_artifacts.size() != expected_q2 ||
        iq2_artifacts.size() != expected_iq2) {
        fprintf(stderr,
                "ds4: CUDA in-place MoE repack count mismatch "
                "(iq2=%zu/%u q2=%zu/%u)\n",
                iq2_artifacts.size(), expected_iq2,
                q2_artifacts.size(), expected_q2);
        cuda_moe_aligned_clear();
        return 0;
    }

    g_moe_aligned_ranges.reserve(q2_artifacts.size() + iq2_artifacts.size());
    auto remember = [model_map](const ds4_repack_artifact &art) {
        g_moe_aligned_ranges.push_back({
            model_map,
            art.t->off,
            art.bytes,
            art.kind,
            (const char *)art.dev,
            art.in_dim,
            art.out_dim,
            art.group_count,
        });
    };
    for (const ds4_repack_artifact &art : q2_artifacts) remember(art);
    for (const ds4_repack_artifact &art : iq2_artifacts) remember(art);
    g_moe_aligned_host_base = model_map;
    g_moe_aligned_model_size = model_size;
    g_moe_aligned_ready = 1;
    fprintf(stderr,
            "ds4: CUDA target MoE replaced in place: %zu tensors %.2f GiB "
            "(IQ2 gate/up + Q2 down, zero permanent duplication)\n",
            g_moe_aligned_ranges.size(),
            (double)(iq2_bytes + q2_bytes) / 1073741824.0);
    return 1;
}

static int cuda_model_set_host_map(const void *model_map, uint64_t model_size) {
    if (!model_map || model_size == 0) return 0;
    cuda_token_graph_release();
    const int same_backing_model =
        g_model_host_base == model_map &&
        g_model_registered_size == model_size;
    cuda_stream_selected_cache_invalidate();
    if (!same_backing_model) {
        cuda_stream_expert_cache_release_all();
    }
    cuda_model_range_release_all();
    if (!same_backing_model) {
        cuda_moe_aligned_clear();
        cuda_model_load_progress_reset();
    }
    cuda_q8_f16_cache_release_all();
    g_q8_f16_disabled_after_oom = 0;
    g_q8_f16_budget_notice_printed = 0;
    for (const cuda_q8_f32_range &r : g_q8_f32_ranges) {
        (void)cudaFree(r.device_ptr);
    }
    g_q8_f32_ranges.clear();
    g_q8_f32_by_offset.clear();
    g_q8_f32_bytes = 0;
    if (!same_backing_model) {
        if (g_model_device_owned && g_model_device_base) {
            (void)cudaFree((void *)g_model_device_base);
            g_model_device_owned = 0;
        }
        if (g_model_registered && g_model_host_base) {
            (void)cudaHostUnregister((void *)g_model_host_base);
            g_model_registered = 0;
        }
        g_model_host_base = model_map;
        g_model_device_base = (const char *)model_map;
        g_model_registered_size = model_size;
    } else if (!g_model_device_owned && !g_model_registered) {
        g_model_device_base = (const char *)model_map;
    }
    g_model_range_mapping_supported = 1;
    g_model_hmm_direct = 0;
    g_model_cache_full = 0;
    g_model_mapping_failure_notice_printed = 0;
    if (g_model_fd >= 0 && g_model_fd_host_base == NULL) {
        g_model_fd_host_base = model_map;
    }
    return 1;
}

/* Keep an optional support model resident without replacing the primary model
 * mapping.  The old single-map path called cuda_model_set_host_map() for the
 * MTP GGUF, which released an 80+ GiB target-model device copy when
 * DS4_CUDA_COPY_MODEL=1.  Secondary mappings live in g_model_ranges, whose
 * lookup already keys the slow path by host_base as well as offset. */
static int cuda_model_add_secondary_map(const void *model_map, uint64_t model_size) {
    if (!model_map || model_size == 0) return 0;
    for (const cuda_model_range &r : g_model_ranges) {
        if (r.host_base == model_map && r.offset == 0 && r.bytes >= model_size) {
            return 1;
        }
    }

    const char *copy_secondary_env = getenv("DS4_CUDA_COPY_SECONDARY_MODEL");
    const bool copy_secondary =
        !(copy_secondary_env && copy_secondary_env[0] &&
          strcmp(copy_secondary_env, "0") == 0);
    const char *copy_env = getenv("DS4_CUDA_COPY_MODEL");
    if (copy_secondary && copy_env && copy_env[0]) {
        void *dev = NULL;
        const double t0 = cuda_wall_sec();
        cudaError_t err = cudaMalloc(&dev, (size_t)model_size);
        if (err == cudaSuccess) {
            fprintf(stderr,
                    "ds4: CUDA copying %.2f GiB secondary model to device memory\n",
                    (double)model_size / 1073741824.0);
            err = cudaMemcpy(dev, model_map, (size_t)model_size,
                             cudaMemcpyHostToDevice);
            if (err == cudaSuccess) {
                g_model_ranges.push_back({model_map, 0, model_size,
                                          (char *)dev, NULL, NULL, 0, 0, 0});
                g_model_range_by_offset[cuda_model_offset_key(model_map, 0)] =
                    g_model_ranges.size() - 1u;
                g_model_range_bytes += model_size;
                cuda_model_drop_copied_source_pages(model_map,
                                                     model_size,
                                                     "secondary model");
                fprintf(stderr,
                        "ds4: CUDA secondary model copy complete in %.3fs\n",
                        cuda_wall_sec() - t0);
                return 1;
            }
            fprintf(stderr, "ds4: CUDA secondary model copy failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaFree(dev);
            (void)cudaGetLastError();
        } else {
            fprintf(stderr,
                    "ds4: CUDA secondary model allocation skipped: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
        }
    } else if (copy_env && copy_env[0]) {
        fprintf(stderr,
                "ds4: CUDA secondary model device copy disabled; using mapped host backing\n");
    }

    unsigned int flags = cudaHostRegisterMapped | cudaHostRegisterReadOnly;
    if (getenv("DS4_CUDA_HOST_REGISTER_PLAIN") != NULL) {
        flags = cudaHostRegisterMapped;
    }
    cudaError_t err = cudaHostRegister((void *)model_map, (size_t)model_size,
                                       flags);
    if (err != cudaSuccess &&
        (flags & cudaHostRegisterReadOnly) != 0 &&
        (err == cudaErrorNotSupported || err == cudaErrorInvalidValue)) {
        (void)cudaGetLastError();
        err = cudaHostRegister((void *)model_map, (size_t)model_size,
                               cudaHostRegisterMapped);
    }
    if (err == cudaSuccess) {
        void *dev = NULL;
        err = cudaHostGetDevicePointer(&dev, (void *)model_map, 0);
        if (err == cudaSuccess && dev) {
            g_model_ranges.push_back({model_map, 0, model_size,
                                      (char *)dev, (void *)model_map,
                                      (char *)dev, model_size, 1, 0});
            g_model_range_by_offset[cuda_model_offset_key(model_map, 0)] =
                g_model_ranges.size() - 1u;
            fprintf(stderr,
                    "ds4: CUDA registered %.2f GiB secondary model mapping\n",
                    (double)model_size / 1073741824.0);
            return 1;
        }
        (void)cudaHostUnregister((void *)model_map);
        (void)cudaGetLastError();
    } else {
        (void)cudaGetLastError();
    }

    /* Last-resort device range copy.  It remains independently owned and is
     * released by cuda_model_range_release_all(). */
    return cuda_model_range_populate_device_copy(model_map, 0, model_size,
                                                   "secondary model") != NULL;
}

extern "C" int ds4_gpu_set_model_map(const void *model_map, uint64_t model_size) {
    if (g_model_host_base && model_map != g_model_host_base) {
        return cuda_model_add_secondary_map(model_map, model_size);
    }
    if (!cuda_model_set_host_map(model_map, model_size)) return 0;

    const char *copy_env = getenv("DS4_CUDA_COPY_MODEL");
    if (copy_env && copy_env[0]) {
        if (cuda_model_copy_chunked(model_map, model_size, 0, model_size)) {
            return 1;
        }
        fprintf(stderr, "ds4: CUDA pipelined model copy unavailable; falling back to monolithic copy\n");
        void *dev = NULL;
        const double t0 = cuda_wall_sec();
        cudaError_t err = cudaMalloc(&dev, (size_t)model_size);
        if (err == cudaSuccess) {
            fprintf(stderr, "ds4: CUDA copying %.2f GiB model to device memory\n",
                    (double)model_size / 1073741824.0);
            err = cudaMemcpy(dev, model_map, (size_t)model_size, cudaMemcpyHostToDevice);
            if (err == cudaSuccess) {
                g_model_device_base = (const char *)dev;
                g_model_device_owned = 1;
                cuda_model_drop_copied_source_pages(model_map,
                                                     model_size,
                                                     "model");
                const double t1 = cuda_wall_sec();
                fprintf(stderr, "ds4: CUDA model copy complete in %.3fs\n", t1 - t0);
                return 1;
            }
            fprintf(stderr, "ds4: CUDA model copy failed: %s\n", cudaGetErrorString(err));
            (void)cudaFree(dev);
            (void)cudaGetLastError();
        } else {
            fprintf(stderr, "ds4: CUDA model allocation skipped: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
        }
    }

    unsigned int flags = cudaHostRegisterMapped | cudaHostRegisterReadOnly;
    if (getenv("DS4_CUDA_HOST_REGISTER_PLAIN") != NULL) {
        flags = cudaHostRegisterMapped;
    }
    cudaError_t err = cudaHostRegister((void *)model_map, (size_t)model_size,
                                       flags);
    if (err == cudaSuccess) {
        void *dev = NULL;
        err = cudaHostGetDevicePointer(&dev, (void *)model_map, 0);
        if (err == cudaSuccess && dev) {
            g_model_device_base = (const char *)dev;
            g_model_registered = 1;
            fprintf(stderr, "ds4: CUDA registered %.2f GiB model mapping for device access\n",
                    (double)model_size / 1073741824.0);
        } else {
            fprintf(stderr, "ds4: CUDA host registration pointer lookup failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
        }
    } else {
        fprintf(stderr, "ds4: CUDA host registration skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        const uint64_t limit = cuda_model_local_model_limit_bytes();
        if (!cuda_model_cache_limit_explicit() && model_size > limit) {
            fprintf(stderr,
                    "ds4: CUDA model %.2f GiB exceeds the default single-GPU "
                    "startup cache budget %.2f GiB; use distributed layer "
                    "loading or set DS4_CUDA_WEIGHT_CACHE_LIMIT_GB explicitly\n",
                    (double)model_size / 1073741824.0,
                    (double)limit / 1073741824.0);
            return 0;
        }
    }
    return 1;
}

extern "C" int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes) {
    (void)max_tensor_bytes;
    const int secondary = g_model_host_base && model_map != g_model_host_base;
    if (!ds4_gpu_set_model_map(model_map, model_size)) return 0;
    if (secondary) return 1;
    if (!cuda_model_prepare_moe_aligned(model_map, model_size)) return 0;
    if (getenv("DS4_CUDA_COPY_MODEL_CHUNKED") != NULL &&
        !cuda_model_copy_chunked(model_map, model_size, map_offset, map_size)) {
        (void)cuda_model_prefetch_range(model_map, model_size, map_offset, map_size);
    }
    return 1;
}

extern "C" int ds4_gpu_pro_q4_expert_table_auto_available(void) {
    return 0;
}

extern "C" int ds4_gpu_preload_q4_expert_tables(const void *model_map, uint64_t model_size,
                                                uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
                                                uint64_t gate_expert_bytes, uint64_t down_expert_bytes,
                                                uint32_t n_total_expert) {
    (void)model_map;
    (void)model_size;
    (void)gate_offset;
    (void)up_offset;
    (void)down_offset;
    (void)gate_expert_bytes;
    (void)down_expert_bytes;
    (void)n_total_expert;
    return 1;
}

extern "C" int ds4_gpu_set_model_map_spans(
        const void *model_map,
        uint64_t model_size,
        const uint64_t *offsets,
        const uint64_t *sizes,
        uint32_t count,
        uint64_t max_tensor_bytes) {
    (void)max_tensor_bytes;
    if (!model_map || model_size == 0 || !offsets || !sizes || count == 0) return 0;
    for (uint32_t i = 0; i < count; i++) {
        if (offsets[i] > model_size ||
            sizes[i] == 0 ||
            sizes[i] > model_size - offsets[i]) {
            return 0;
        }
    }
    if (!cuda_model_set_host_map(model_map, model_size)) return 0;

    if (getenv("DS4_CUDA_COPY_MODEL_CHUNKED") != NULL) {
        for (uint32_t i = 0; i < count; i++) {
            (void)cuda_model_prefetch_range(model_map, model_size, offsets[i], sizes[i]);
        }
    }
    return 1;
}

extern "C" int ds4_gpu_set_model_fd_for_map(int fd, const void *model_map) {
    g_model_fd = fd;
    g_model_fd_host_base = model_map;
    g_model_file_size = 0;
    if (g_model_direct_fd >= 0) {
        (void)close(g_model_direct_fd);
        g_model_direct_fd = -1;
    }
    g_model_direct_align = 1;
    if (fd >= 0) {
        struct stat st;
        if (fstat(fd, &st) == 0 && st.st_size > 0) {
            g_model_file_size = (uint64_t)st.st_size;
            if (st.st_blksize > 1) g_model_direct_align = (uint64_t)st.st_blksize;
        }
#if defined(__linux__) && defined(O_DIRECT)
        if (getenv("DS4_CUDA_NO_DIRECT_IO") == NULL) {
            char proc_path[64];
            snprintf(proc_path, sizeof(proc_path), "/proc/self/fd/%d", fd);
            int direct_fd = open(proc_path, O_RDONLY | O_DIRECT);
            if (direct_fd >= 0) {
                g_model_direct_fd = direct_fd;
                if (g_model_direct_align < 512) g_model_direct_align = 512;
                if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
                    fprintf(stderr, "ds4: CUDA model direct I/O enabled (align=%llu)\n",
                            (unsigned long long)g_model_direct_align);
                }
            } else if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
                fprintf(stderr, "ds4: CUDA model direct I/O unavailable: %s\n", strerror(errno));
            }
        }
#endif
    }
    return 1;
}

extern "C" int ds4_gpu_set_model_fd(int fd) {
    return ds4_gpu_set_model_fd_for_map(fd, g_model_host_base);
}

extern "C" int ds4_gpu_cache_model_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, const char *label) {
    if (!model_map || bytes == 0) return 1;
    if (offset > model_size || bytes > model_size - offset) return 0;
    if (cuda_model_range_is_cached(model_map, offset, bytes)) return 1;

    const char *ptr = cuda_model_range_ptr(model_map, offset, bytes, label ? label : "model_tensor");
    if (!ptr || !cuda_model_range_is_cached(model_map, offset, bytes)) {
        if (!g_model_mapping_failure_notice_printed) {
            fprintf(stderr,
                    "ds4: CUDA failed to prepare model tensor spans for device access\n");
            g_model_mapping_failure_notice_printed = 1;
        }
        return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_cache_q8_f16_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, uint64_t in_dim, uint64_t out_dim, const char *label) {
    if (!model_map || bytes == 0) return 1;
    if (offset > model_size || bytes > model_size - offset) return 0;
    const char *cache_label = label ? label : "q8_0";
    if (getenv("DS4_CUDA_Q8_F32_PRELOAD") != NULL &&
        cuda_q8_f32_cache_allowed(cache_label, in_dim, out_dim)) {
        if (cuda_q8_f32_ptr(model_map, offset, bytes, in_dim, out_dim, cache_label)) return 1;
        return 1;
    }
    if (!cuda_q8_f16_cache_allowed(cache_label, in_dim, out_dim)) return 1;
    if (cuda_q8_f16_ptr(model_map, offset, bytes, in_dim, out_dim, cache_label)) return 1;
    return 1;
}

extern "C" void ds4_gpu_print_memory_report(const char *label) {
    size_t free_b = 0, total_b = 0;
    (void)cudaMemGetInfo(&free_b, &total_b);
    fprintf(stderr,
            "ds4: CUDA memory report %s: free %.2f MiB total %.2f MiB "
            "secondary/ranges %.2f GiB q8-f16 %.2f GiB q8-f32 %.2f GiB "
            "cublas-workspace %.2f MiB\n",
            label ? label : "",
            (double)free_b / 1048576.0,
            (double)total_b / 1048576.0,
            (double)g_model_range_bytes / 1073741824.0,
            (double)g_q8_f16_bytes / 1073741824.0,
            (double)g_q8_f32_bytes / 1073741824.0,
            (double)g_cublas_workspace_bytes / 1048576.0);
}

extern "C" void ds4_gpu_set_quality(bool quality) {
    g_quality_mode = quality ? 1 : 0;
    if (g_cublas_ready) {
        const cublasMath_t math_mode =
            (g_quality_mode || getenv("DS4_CUDA_NO_TF32") != NULL)
                ? CUBLAS_DEFAULT_MATH
                : CUBLAS_TF32_TENSOR_OP_MATH;
        (void)cublasSetMathMode(g_cublas, math_mode);
    }
}

extern "C" void ds4_gpu_set_ssd_streaming(bool enabled) {
    g_ssd_streaming_mode = enabled ? 1 : 0;
    g_stream_expert_runtime_cap = 0;
    g_stream_expert_runtime_gate_bytes = 0;
    g_stream_expert_runtime_down_bytes = 0;
    g_stream_expert_memory_cap_notice = 0;
    if (!g_ssd_streaming_mode) {
        cuda_stream_selected_cache_release();
        cuda_stream_expert_cache_release_all();
    }
}

extern "C" void ds4_gpu_set_streaming_expert_cache_budget(uint32_t experts) {
    g_stream_expert_budget_override = experts;
    g_stream_expert_runtime_cap = 0;
    g_stream_expert_runtime_gate_bytes = 0;
    g_stream_expert_runtime_down_bytes = 0;
    g_stream_expert_memory_cap_notice = 0;
    cuda_stream_selected_cache_invalidate();
    cuda_stream_expert_cache_release_all();
}

extern "C" void ds4_gpu_set_streaming_expert_cache_expert_bytes(uint64_t bytes) {
    (void)bytes;
}

extern "C" uint64_t ds4_gpu_recommended_working_set_size(void) {
    return 0;
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_configured_count(void) {
    if (!cuda_stream_expert_cache_budget_visible_to_shared()) return 0;
    return cuda_stream_expert_cache_configured_budget();
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_current_count(void) {
    return g_stream_expert_cache.count;
}

extern "C" void ds4_gpu_stream_expert_cache_reset_route_hotness(void) {
}

extern "C" void ds4_gpu_stream_expert_cache_release_resident(void) {
    cuda_stream_expert_cache_release_all();
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_budget_for_expert_size(
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (!cuda_stream_expert_cache_budget_visible_to_shared() ||
        cuda_stream_expert_cache_expert_bytes(gate_expert_bytes,
                                              down_expert_bytes) == 0) {
        return 0;
    }
    cuda_stream_expert_cache_note_size(gate_expert_bytes, down_expert_bytes);
    return cuda_stream_expert_cache_configured_budget();
}

extern "C" int ds4_gpu_stream_expert_cache_seed_selected(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected) {
    if (!g_ssd_streaming_mode) return 1;
    if (!table) return 0;
    const void *model_map = table->model_map;
    const uint64_t model_size = table->model_size;
    const uint32_t layer = table->layer;
    const uint32_t n_total_expert = table->n_total_expert;
    const uint64_t gate_offset = table->gate_offset;
    const uint64_t up_offset = table->up_offset;
    const uint64_t down_offset = table->down_offset;
    const uint64_t gate_expert_bytes = table->gate_expert_bytes;
    const uint64_t down_expert_bytes = table->down_expert_bytes;
    if (!model_map || !selected_ids || n_selected == 0 ||
        n_selected > n_total_expert ||
        !cuda_stream_layer_expert_ranges_valid(model_size,
                                               n_total_expert,
                                               gate_offset,
                                               up_offset,
                                               down_offset,
                                               gate_expert_bytes,
                                               down_expert_bytes,
                                               "seed selected")) {
        return 0;
    }

    cuda_stream_expert_cache *cache =
        cuda_stream_expert_cache_prepare(gate_expert_bytes,
                                         down_expert_bytes,
                                         n_selected);
    if (!cache) return 1;
    for (uint32_t i = 0; i < n_selected; i++) {
        if (selected_ids[i] < 0 || (uint32_t)selected_ids[i] >= n_total_expert) {
            fprintf(stderr,
                    "ds4: CUDA streaming seed selected expert id %d is outside 0..%u at layer %u\n",
                    selected_ids[i],
                    n_total_expert,
                    layer);
            return 0;
        }
        if (!cuda_stream_expert_cache_seed_one(cache,
                                               model_map,
                                               model_size,
                                               layer,
                                               n_total_expert,
                                               (uint32_t)selected_ids[i],
                                               gate_offset,
                                               up_offset,
                                               down_offset,
                                               gate_expert_bytes,
                                               down_expert_bytes)) {
            cuda_stream_expert_cache_invalidate();
            return 1;
        }
    }
    return 1;
}

static int cuda_stream_selected_cache_begin_compact_load(
        const void    *model_map,
        uint64_t       model_size,
        uint32_t       layer,
        const int32_t *compact_ids,
        const int32_t *slot_ids,
        uint32_t       n_total_expert,
        uint32_t       compact_count,
        uint32_t       slot_count,
        uint64_t       gate_offset,
        uint64_t       up_offset,
        uint64_t       down_offset,
        uint64_t       gate_expert_bytes,
        uint64_t       down_expert_bytes,
        int            strict_failure,
        int            allow_global_cache) {
    cuda_stream_selected_cache_invalidate();
    cuda_model_load_progress_finish();

    if (!g_ssd_streaming_mode) return 1;
    if (!model_map || !compact_ids || !slot_ids ||
        n_total_expert == 0 ||
        compact_count == 0 || compact_count > n_total_expert ||
        slot_count == 0 ||
        gate_expert_bytes == 0 || down_expert_bytes == 0) {
        return 0;
    }
    if ((uint64_t)n_total_expert > UINT64_MAX / gate_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / down_expert_bytes ||
        (uint64_t)compact_count > UINT64_MAX / gate_expert_bytes ||
        (uint64_t)compact_count > UINT64_MAX / down_expert_bytes) {
        fprintf(stderr, "ds4: CUDA streaming selected expert size overflow\n");
        return 0;
    }

    const uint64_t full_gate_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t full_down_bytes = (uint64_t)n_total_expert * down_expert_bytes;
    const uint64_t compact_gate_bytes = (uint64_t)compact_count * gate_expert_bytes;
    const uint64_t compact_down_bytes = (uint64_t)compact_count * down_expert_bytes;
    if (gate_offset > model_size || up_offset > model_size || down_offset > model_size ||
        full_gate_bytes > model_size - gate_offset ||
        full_gate_bytes > model_size - up_offset ||
        full_down_bytes > model_size - down_offset) {
        fprintf(stderr, "ds4: CUDA streaming selected expert range outside model map\n");
        return 0;
    }

    if (!allow_global_cache) {
        cuda_stream_expert_cache_release_all();
    }

    if (!cuda_stream_selected_ensure_bytes(&g_stream_selected_cache.gate_ptr,
                                           &g_stream_selected_cache.gate_capacity,
                                           compact_gate_bytes,
                                           "selected gate experts") ||
        !cuda_stream_selected_ensure_bytes(&g_stream_selected_cache.up_ptr,
                                           &g_stream_selected_cache.up_capacity,
                                           compact_gate_bytes,
                                           "selected up experts") ||
        !cuda_stream_selected_ensure_bytes(&g_stream_selected_cache.down_ptr,
                                           &g_stream_selected_cache.down_capacity,
                                           compact_down_bytes,
                                           "selected down experts") ||
        !cuda_stream_selected_ensure_i32(&g_stream_selected_cache.slot_selected_ptr,
                                         &g_stream_selected_cache.slot_selected_capacity,
                                         slot_count,
                                         "selected expert slots")) {
        return strict_failure ? 0 : 1;
    }

    if (allow_global_cache) {
        cuda_stream_expert_cache_note_size(gate_expert_bytes,
                                           down_expert_bytes);
    }
    const uint32_t configured_cache_budget =
        cuda_stream_expert_cache_configured_budget();
    const int use_global_cache =
        allow_global_cache &&
        configured_cache_budget != 0;
    cuda_stream_expert_cache *expert_cache = use_global_cache ?
        cuda_stream_expert_cache_prepare(gate_expert_bytes,
                                         down_expert_bytes,
                                         configured_cache_budget) :
        NULL;
    int expert_cache_disabled = expert_cache == NULL;
    const uint32_t cache_count_before =
        expert_cache && expert_cache->valid ? expert_cache->count : 0;
    uint32_t cache_hits = 0;
    uint32_t cache_misses = 0;
    uint32_t direct_loads = 0;

    for (uint32_t i = 0; i < compact_count; i++) {
        if (compact_ids[i] < 0 || (uint32_t)compact_ids[i] >= n_total_expert) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected expert id %d is outside 0..%u at layer %u\n",
                    compact_ids[i],
                    n_total_expert,
                    layer);
            return 0;
        }

        const uint64_t expert = (uint64_t)(uint32_t)compact_ids[i];
        const uint64_t gate_dst = (uint64_t)i * gate_expert_bytes;
        const uint64_t down_dst = (uint64_t)i * down_expert_bytes;
        int copied_from_global_cache = 0;

        if (!expert_cache_disabled) {
            int cache_slot =
                cuda_stream_expert_cache_find(expert_cache,
                                              model_map,
                                              model_size,
                                              layer,
                                              n_total_expert,
                                              (uint32_t)expert,
                                              gate_offset,
                                              up_offset,
                                              down_offset,
                                              gate_expert_bytes,
                                              down_expert_bytes);
            if (cache_slot >= 0) {
                cache_hits++;
                expert_cache->slots[(uint32_t)cache_slot].age =
                    ++expert_cache->tick;
            } else {
                cache_misses++;
                const uint32_t load_slot =
                    cuda_stream_expert_cache_lru_slot(expert_cache);
                const int append = !expert_cache->slots[load_slot].valid;
                if (cuda_stream_expert_cache_load_slot(expert_cache,
                                                       model_map,
                                                       model_size,
                                                       load_slot,
                                                       layer,
                                                       n_total_expert,
                                                       (uint32_t)expert,
                                                       gate_offset,
                                                       up_offset,
                                                       down_offset,
                                                       gate_expert_bytes,
                                                       down_expert_bytes)) {
                    if (append && expert_cache->count < expert_cache->capacity) {
                        expert_cache->count++;
                    }
                    cache_slot = (int)load_slot;
                } else {
                    cuda_stream_expert_cache_invalidate();
                    expert_cache_disabled = 1;
                    cache_slot = -1;
                }
            }

            if (cache_slot >= 0) {
                copied_from_global_cache =
                    cuda_stream_expert_cache_copy_to_compact(
                            expert_cache,
                            (uint32_t)cache_slot,
                            i,
                            g_stream_selected_cache.gate_ptr,
                            g_stream_selected_cache.up_ptr,
                            g_stream_selected_cache.down_ptr);
                if (!copied_from_global_cache) {
                    cuda_stream_expert_cache_invalidate();
                    expert_cache_disabled = 1;
                }
            }
        }

        if (!copied_from_global_cache) {
            const uint64_t gate_src = gate_offset + expert * gate_expert_bytes;
            const uint64_t up_src = up_offset + expert * gate_expert_bytes;
            const uint64_t down_src = down_offset + expert * down_expert_bytes;
            direct_loads++;
            if (!cuda_model_copy_to_device_streamed(g_stream_selected_cache.gate_ptr + gate_dst,
                                                    model_map,
                                                    model_size,
                                                    gate_src,
                                                    gate_expert_bytes,
                                                    "selected moe_gate") ||
                !cuda_model_copy_to_device_streamed(g_stream_selected_cache.up_ptr + gate_dst,
                                                    model_map,
                                                    model_size,
                                                    up_src,
                                                    gate_expert_bytes,
                                                    "selected moe_up") ||
                !cuda_model_copy_to_device_streamed(g_stream_selected_cache.down_ptr + down_dst,
                                                    model_map,
                                                    model_size,
                                                    down_src,
                                                    down_expert_bytes,
                                                    "selected moe_down")) {
                cuda_stream_selected_cache_invalidate();
                return strict_failure ? 0 : 1;
            }
        }
    }

    if (!cuda_ok(cudaMemcpy(g_stream_selected_cache.slot_selected_ptr,
                            slot_ids,
                            (size_t)slot_count * sizeof(slot_ids[0]),
                            cudaMemcpyHostToDevice),
                 "streaming selected slot upload")) {
        cuda_stream_selected_cache_invalidate();
        return strict_failure ? 0 : 1;
    }

    g_stream_selected_cache.model_map = model_map;
    g_stream_selected_cache.layer = layer;
    g_stream_selected_cache.n_total_expert = n_total_expert;
    g_stream_selected_cache.n_selected = slot_count;
    g_stream_selected_cache.slot_count = slot_count;
    g_stream_selected_cache.compact_count = compact_count;
    g_stream_selected_cache.gate_offset = gate_offset;
    g_stream_selected_cache.up_offset = up_offset;
    g_stream_selected_cache.down_offset = down_offset;
    g_stream_selected_cache.gate_expert_bytes = gate_expert_bytes;
    g_stream_selected_cache.down_expert_bytes = down_expert_bytes;
    g_stream_selected_cache.slot_selected_tensor.ptr =
        g_stream_selected_cache.slot_selected_ptr;
    g_stream_selected_cache.slot_selected_tensor.bytes =
        (uint64_t)slot_count * sizeof(int32_t);
    g_stream_selected_cache.slot_selected_tensor.owner = 0;
    g_stream_selected_cache.valid = 1;

    if (getenv("DS4_CUDA_STREAMING_EXPERT_CACHE_VERBOSE")) {
        cuda_model_load_progress_finish();
        fprintf(stderr,
                "ds4: CUDA streaming selected layer=%u slots=%u compact=%u global_budget=%u before=%u after=%u hits=%u misses=%u direct=%u gate/up %.2f MiB down %.2f MiB\n",
                layer,
                slot_count,
                compact_count,
                expert_cache && expert_cache->valid ? expert_cache->capacity : 0,
                cache_count_before,
                expert_cache && expert_cache->valid ? expert_cache->count : 0,
                cache_hits,
                cache_misses,
                direct_loads,
                (double)compact_gate_bytes / 1048576.0,
                (double)compact_down_bytes / 1048576.0);
    }
    return 1;
}

extern "C" int ds4_gpu_stream_expert_cache_begin_selected_load(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected) {
    if (!g_ssd_streaming_mode) return 1;
    if (!table || !selected_ids || n_selected == 0) return 0;
    const void *model_map = table->model_map;
    const uint64_t model_size = table->model_size;
    const uint32_t layer = table->layer;
    const uint32_t n_total_expert = table->n_total_expert;
    const uint64_t gate_offset = table->gate_offset;
    const uint64_t up_offset = table->up_offset;
    const uint64_t down_offset = table->down_offset;
    const uint64_t gate_expert_bytes = table->gate_expert_bytes;
    const uint64_t down_expert_bytes = table->down_expert_bytes;

    std::vector<int32_t> expert_to_slot(n_total_expert, -1);
    std::vector<int32_t> compact_ids;
    std::vector<int32_t> slot_ids(n_selected);
    compact_ids.reserve(n_selected);
    for (uint32_t i = 0; i < n_selected; i++) {
        const int32_t expert_i = selected_ids[i];
        if (expert_i < 0 || (uint32_t)expert_i >= n_total_expert) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected expert id %d is outside 0..%u at layer %u\n",
                    expert_i,
                    n_total_expert,
                    layer);
            return 0;
        }
        expert_to_slot[(uint32_t)expert_i] = -2;
    }
    for (uint32_t e = 0; e < n_total_expert; e++) {
        if (expert_to_slot[e] != -2) continue;
        expert_to_slot[e] = (int32_t)compact_ids.size();
        compact_ids.push_back((int32_t)e);
    }
    for (uint32_t i = 0; i < n_selected; i++) {
        slot_ids[i] = expert_to_slot[(uint32_t)selected_ids[i]];
    }
    if (compact_ids.empty() || compact_ids.size() > UINT32_MAX) return 0;
    return cuda_stream_selected_cache_begin_compact_load(
            model_map,
            model_size,
            layer,
            compact_ids.data(),
            slot_ids.data(),
            n_total_expert,
            (uint32_t)compact_ids.size(),
            n_selected,
            gate_offset,
            up_offset,
            down_offset,
            gate_expert_bytes,
            down_expert_bytes,
            0,
            1);
}

extern "C" int ds4_gpu_stream_expert_cache_prepare_selected_batch(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_tokens,
        uint32_t                           n_selected) {
    if (!g_ssd_streaming_mode) return 1;
    if (!table ||
        !selected_ids ||
        table->n_total_expert == 0 ||
        n_selected == 0 ||
        n_tokens == 0 ||
        (uint64_t)n_tokens > UINT32_MAX / (uint64_t)n_selected) {
        return 0;
    }
    const void *model_map = table->model_map;
    const uint64_t model_size = table->model_size;
    const uint32_t layer = table->layer;
    const uint32_t n_total_expert = table->n_total_expert;
    const uint64_t gate_offset = table->gate_offset;
    const uint64_t up_offset = table->up_offset;
    const uint64_t down_offset = table->down_offset;
    const uint64_t gate_expert_bytes = table->gate_expert_bytes;
    const uint64_t down_expert_bytes = table->down_expert_bytes;

    std::vector<int32_t> expert_to_slot(n_total_expert, -1);
    std::vector<int32_t> compact_ids;
    const uint32_t slot_count = n_tokens * n_selected;
    std::vector<int32_t> slot_ids(slot_count);
    compact_ids.reserve(slot_count < n_total_expert ? slot_count : n_total_expert);

    for (uint32_t i = 0; i < slot_count; i++) {
        const int32_t expert_i = selected_ids[i];
        if (expert_i < 0 || (uint32_t)expert_i >= n_total_expert) {
            fprintf(stderr,
                    "ds4: CUDA streaming batch selected expert id %d is outside 0..%u at layer %u\n",
                    expert_i,
                    n_total_expert,
                    layer);
            return 0;
        }
        expert_to_slot[(uint32_t)expert_i] = -2;
    }
    for (uint32_t e = 0; e < n_total_expert; e++) {
        if (expert_to_slot[e] != -2) continue;
        expert_to_slot[e] = (int32_t)compact_ids.size();
        compact_ids.push_back((int32_t)e);
    }
    for (uint32_t i = 0; i < slot_count; i++) {
        slot_ids[i] = expert_to_slot[(uint32_t)selected_ids[i]];
    }

    if (compact_ids.empty() || compact_ids.size() > UINT32_MAX) return 0;
    return cuda_stream_selected_cache_begin_compact_load(
            model_map,
            model_size,
            layer,
            compact_ids.data(),
            slot_ids.data(),
            n_total_expert,
            (uint32_t)compact_ids.size(),
            slot_count,
            gate_offset,
            up_offset,
            down_offset,
            gate_expert_bytes,
            down_expert_bytes,
            1,
            0);
}

extern "C" int ds4_gpu_stream_expert_cache_seed_experts(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *expert_ids,
        const uint32_t                    *expert_priorities,
        uint32_t                           n_experts) {
    if (!g_ssd_streaming_mode) return 1;
    if (!table) return 0;
    const void *model_map = table->model_map;
    const uint64_t model_size = table->model_size;
    const uint32_t layer = table->layer;
    const uint32_t n_total_expert = table->n_total_expert;
    const uint64_t gate_offset = table->gate_offset;
    const uint64_t up_offset = table->up_offset;
    const uint64_t down_offset = table->down_offset;
    const uint64_t gate_expert_bytes = table->gate_expert_bytes;
    const uint64_t down_expert_bytes = table->down_expert_bytes;
    if (!model_map || !expert_ids || n_experts == 0 ||
        !cuda_stream_layer_expert_ranges_valid(model_size,
                                               n_total_expert,
                                               gate_offset,
                                               up_offset,
                                               down_offset,
                                               gate_expert_bytes,
                                               down_expert_bytes,
                                               "seed hotlist")) {
        return 0;
    }

    cuda_stream_expert_cache *cache =
        cuda_stream_expert_cache_prepare(gate_expert_bytes,
                                         down_expert_bytes,
                                         n_experts);
    if (!cache || cache->capacity == 0) return 1;

    const uint32_t layer_seed_cap =
        n_experts < cache->capacity ? n_experts : cache->capacity;
    std::vector<uint32_t> chosen;
    try {
        chosen.reserve(layer_seed_cap);
    } catch (...) {
        return 1;
    }

    for (uint32_t i = 0; i < n_experts; i++) {
        const int32_t expert = expert_ids[i];
        if (expert < 0 || (uint32_t)expert >= n_total_expert) {
            fprintf(stderr,
                    "ds4: CUDA streaming hotlist seed expert id %d is outside 0..%u at layer %u\n",
                    expert,
                    n_total_expert,
                    layer);
            return 0;
        }
        const uint32_t priority =
            expert_priorities ? expert_priorities[i] : (n_experts - i);
        uint32_t pos = 0;
        while (pos < chosen.size()) {
            const uint32_t other = chosen[pos];
            const uint32_t other_priority =
                expert_priorities ? expert_priorities[other] :
                                    (n_experts - other);
            if (priority > other_priority) break;
            pos++;
        }
        if (chosen.size() < layer_seed_cap) {
            chosen.insert(chosen.begin() + pos, i);
        } else if (pos < chosen.size()) {
            chosen.insert(chosen.begin() + pos, i);
            chosen.pop_back();
        }
    }

    const uint32_t n = (uint32_t)chosen.size();
    for (uint32_t ri = 0; ri < n; ri++) {
        const uint32_t i = chosen[n - 1u - ri];
        if (!cuda_stream_expert_cache_seed_one(cache,
                                               model_map,
                                               model_size,
                                               layer,
                                               n_total_expert,
                                               (uint32_t)expert_ids[i],
                                               gate_offset,
                                               up_offset,
                                               down_offset,
                                               gate_expert_bytes,
                                               down_expert_bytes)) {
            cuda_stream_expert_cache_invalidate();
            return 1;
        }
    }
    if (getenv("DS4_CUDA_STREAMING_EXPERT_CACHE_VERBOSE")) {
        fprintf(stderr,
                "ds4: CUDA streaming hotlist seeded layer=%u requested=%u cached=%u cap=%u\n",
                layer,
                n_experts,
                n,
                cache->capacity);
    }
    return 1;
}

__global__ static void embed_token_hc_kernel(float *out, const unsigned short *w, uint32_t token, uint32_t n_embd, uint32_t n_hc) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = n_embd * n_hc;
    if (i >= n) return;
    uint32_t e = i % n_embd;
    out[i] = __half2float(reinterpret_cast<const __half *>(w)[(uint64_t)token * n_embd + e]);
}

/* The pipelined graph for position N+1 is captured while position N is still
 * executing, before the sampler knows the next token.  Only that scalar is
 * late-bound: all mathematical work and every positional/KV argument remain
 * the exact values captured for N+1. */
__global__ static void embed_token_hc_dynamic_kernel(
        float                *out,
        const unsigned short *w,
        const uint32_t       *token_ptr,
        uint32_t              n_embd,
        uint32_t              n_hc) {
    const uint32_t token = *token_ptr;
    const uint32_t i = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t n = n_embd * n_hc;
    if (i >= n) return;
    const uint32_t hc = i / n_embd;
    const uint32_t d = i - hc * n_embd;
    out[i] = __half2float(((const __half *)w)[(uint64_t)token * n_embd + d]);
}

__global__ static void embed_tokens_hc_kernel(
        float *out,
        const int32_t *tokens,
        const __half *w,
        uint32_t n_vocab,
        uint32_t n_tokens,
        uint32_t n_embd,
        uint32_t n_hc) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n) return;
    uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    uint32_t t = tmp / n_hc;
    int32_t tok_i = tokens[t];
    uint32_t tok = tok_i < 0 ? 0u : (uint32_t)tok_i;
    if (tok >= n_vocab) tok = 0;
    out[gid] = __half2float(w[(uint64_t)tok * n_embd + d]);
}

__global__ static void matmul_f16_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    float sum = 0.0f;
    const __half *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += __half2float(wr[i]) * xr[i];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

__global__ static void matmul_f16_serial_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok || threadIdx.x != 0) return;

    float sum = 0.0f;
    const __half *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = 0; i < in_dim; i++) {
        sum += __half2float(wr[i]) * xr[i];
    }
    out[tok * out_dim + row] = sum;
}

__global__ static void matmul_f16_ordered_chunks_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    __shared__ float partial[32];
    const uint32_t tid = threadIdx.x;
    float sum = 0.0f;
    const uint64_t chunk = (in_dim + 31u) / 32u;
    const uint64_t k0 = (uint64_t)tid * chunk;
    uint64_t k1 = k0 + chunk;
    if (k1 > in_dim) k1 = in_dim;
    const __half *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = k0; i < k1; i++) {
        sum += __half2float(wr[i]) * xr[i];
    }
    partial[tid] = sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (uint32_t i = 0; i < 32u; i++) total += partial[i];
        out[tok * out_dim + row] = total;
    }
}

__global__ static void matmul_f16_pair_ordered_chunks_kernel(
        float *out0,
        float *out1,
        const __half *w0,
        const __half *w1,
        const float *x,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim) {
    uint64_t row = (uint64_t)blockIdx.x;
    if (row >= out0_dim && row >= out1_dim) return;

    __shared__ float partial0[32];
    __shared__ float partial1[32];
    const uint32_t tid = threadIdx.x;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    const uint64_t chunk = (in_dim + 31u) / 32u;
    const uint64_t k0 = (uint64_t)tid * chunk;
    uint64_t k1 = k0 + chunk;
    if (k1 > in_dim) k1 = in_dim;
    const __half *wr0 = row < out0_dim ? w0 + row * in_dim : w0;
    const __half *wr1 = row < out1_dim ? w1 + row * in_dim : w1;
    for (uint64_t i = k0; i < k1; i++) {
        const float xv = x[i];
        if (row < out0_dim) sum0 += __half2float(wr0[i]) * xv;
        if (row < out1_dim) sum1 += __half2float(wr1[i]) * xv;
    }
    partial0[tid] = sum0;
    partial1[tid] = sum1;
    __syncthreads();
    if (tid == 0) {
        float total0 = 0.0f;
        float total1 = 0.0f;
        for (uint32_t i = 0; i < 32u; i++) {
            total0 += partial0[i];
            total1 += partial1[i];
        }
        if (row < out0_dim) out0[row] = total0;
        if (row < out1_dim) out1[row] = total1;
    }
}

__global__ static void matmul_f32_kernel(
        float *out,
        const float *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    float sum = 0.0f;
    const float *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += wr[i] * xr[i];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

__global__ static void repeat_hc_kernel(float *out, const float *row, uint32_t n_embd, uint32_t n_hc) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_embd * n_hc;
    if (i >= n) return;
    out[i] = row[i % n_embd];
}

__global__ static void f32_to_f16_kernel(__half *out, const float *x, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(x[i]);
}

extern "C" int ds4_gpu_tensor_copy_f32_to_f16(
        ds4_gpu_tensor       *dst,
        uint64_t              dst_offset,
        const ds4_gpu_tensor *src,
        uint64_t              src_offset,
        uint64_t              count) {
    if (!dst || !src || (dst_offset % sizeof(__half)) != 0u ||
        (src_offset % sizeof(float)) != 0u ||
        dst_offset > dst->bytes || src_offset > src->bytes ||
        count > (dst->bytes - dst_offset) / sizeof(__half) ||
        count > (src->bytes - src_offset) / sizeof(float)) {
        return 0;
    }
    if (count == 0u) return 1;
    __half *out = (__half *)((char *)dst->ptr + dst_offset);
    const float *in = (const float *)((const char *)src->ptr + src_offset);
    f32_to_f16_kernel<<<(count + 255u) / 256u, 256>>>(out, in, count);
    return cuda_ok(cudaGetLastError(), "tensor copy f32 to f16 launch");
}

__device__ static float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

/* Decode GEMV path for F16 weights.  Each warp owns one output row and all
 * lanes advance together through adjacent input elements.  This preserves the
 * fused F16-pair path while replacing the strided-across-lanes access pattern
 * of ordered_chunks with coalesced warp loads. */
__global__ static void matmul_f16_coalesced_warp8_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint64_t row = (uint64_t)blockIdx.x * 8u + warp;
    const uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    const __half *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    float sum = 0.0f;
    for (uint64_t i = lane; i < in_dim; i += 32u) {
        sum += __half2float(wr[i]) * xr[i];
    }
    sum = warp_sum_f32(sum);
    if (lane == 0) out[tok * out_dim + row] = sum;
}

__global__ static void matmul_f16_pair_coalesced_warp8_kernel(
        float *out0,
        float *out1,
        const __half *w0,
        const __half *w1,
        const float *x,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint64_t row = (uint64_t)blockIdx.x * 8u + warp;
    if (row >= out0_dim && row >= out1_dim) return;

    const __half *wr0 = row < out0_dim ? w0 + row * in_dim : NULL;
    const __half *wr1 = row < out1_dim ? w1 + row * in_dim : NULL;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (uint64_t i = lane; i < in_dim; i += 32u) {
        const float xv = x[i];
        if (wr0) sum0 += __half2float(wr0[i]) * xv;
        if (wr1) sum1 += __half2float(wr1[i]) * xv;
    }
    sum0 = warp_sum_f32(sum0);
    sum1 = warp_sum_f32(sum1);
    if (lane == 0) {
        if (wr0) out0[row] = sum0;
        if (wr1) out1[row] = sum1;
    }
}

__device__ static float warp_max_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ static float dot4_f32(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ __forceinline__ static int32_t load_i8x4_i32_aligned(const int8_t *p) {
    return *(const int32_t *)p;
}

__device__ __forceinline__ static int32_t load_i8x4_i32_unaligned(const int8_t *p) {
    const uint8_t *u = (const uint8_t *)p;
    return (int32_t)((uint32_t)u[0] |
                     ((uint32_t)u[1] << 8) |
                     ((uint32_t)u[2] << 16) |
                     ((uint32_t)u[3] << 24));
}

__device__ __forceinline__ static int32_t load_i8x4_i32_u16(const int8_t *p) {
    /* Q8_0 payloads start after a two-byte scale inside a 34-byte block, so
     * every four-byte group is naturally aligned to uint16_t even when it is
     * not aligned to uint32_t.  Two exact half-word loads avoid both an
     * unaligned dword access and four byte-load/shift operations. */
    const uint16_t *u = reinterpret_cast<const uint16_t *>(p);
    return (int32_t)((uint32_t)u[0] | ((uint32_t)u[1] << 16));
}

__device__ __forceinline__ static int32_t dot_i8x32_dp4a(const int8_t *a, const int8_t *b) {
    int32_t dot = 0;
#pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        dot = __dp4a(load_i8x4_i32_unaligned(a + i), load_i8x4_i32_aligned(b + i), dot);
    }
    return dot;
}

__device__ __forceinline__ static int32_t dot_i8x32_dp4a_u16(
        const int8_t *a,
        const int8_t *b) {
    int32_t dot = 0;
#pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        dot = __dp4a(load_i8x4_i32_u16(a + i),
                     load_i8x4_i32_aligned(b + i), dot);
    }
    return dot;
}

__device__ __forceinline__ static int32_t dot_i8_block(const int8_t *a, const int8_t *b, uint64_t n, int use_dp4a) {
    if (use_dp4a && n == 32u) {
        return use_dp4a > 1
            ? dot_i8x32_dp4a_u16(a, b)
            : dot_i8x32_dp4a(a, b);
    }
    int32_t dot = 0;
    for (uint64_t i = 0; i < n; i++) dot += (int32_t)a[i] * (int32_t)b[i];
    return dot;
}

__global__ static void q8_u16_load_validate_kernel(
        const int8_t *data,
        uint32_t *mismatches) {
    const uint32_t offset = threadIdx.x * 2u;
    const int32_t old_v = load_i8x4_i32_unaligned(data + offset);
    const int32_t new_v = load_i8x4_i32_u16(data + offset);
    if (old_v != new_v) atomicAdd(mismatches, 1u);
}

static int cuda_q8_u16_validate(void) {
    if (g_q8_u16_validation != -1) return g_q8_u16_validation == 1;
    g_q8_u16_validation = -2;

    uint8_t host_data[66];
    for (uint32_t i = 0; i < sizeof(host_data); i++) {
        host_data[i] = (uint8_t)(11u + i * 37u);
    }
    int8_t *device_data = NULL;
    uint32_t *device_mismatches = NULL;
    uint32_t mismatches = 0;
    const char *failed_at = NULL;
    cudaError_t err = cudaMalloc((void **)&device_data, sizeof(host_data));
    if (err != cudaSuccess) failed_at = "allocate sample";
    if (!failed_at) {
        err = cudaMalloc((void **)&device_mismatches, sizeof(uint32_t));
        if (err != cudaSuccess) failed_at = "allocate result";
    }
    if (!failed_at) {
        err = cudaMemcpy(device_data, host_data, sizeof(host_data),
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) failed_at = "upload sample";
    }
    if (!failed_at) {
        err = cudaMemset(device_mismatches, 0, sizeof(uint32_t));
        if (err != cudaSuccess) failed_at = "clear result";
    }
    if (!failed_at) {
        q8_u16_load_validate_kernel<<<1, 32>>>(device_data,
                                               device_mismatches);
        err = cudaGetLastError();
        if (err != cudaSuccess) failed_at = "launch validation";
    }
    if (!failed_at) {
        err = cudaMemcpy(&mismatches, device_mismatches, sizeof(mismatches),
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) failed_at = "read validation";
    }
    if (device_mismatches) (void)cudaFree(device_mismatches);
    if (device_data) (void)cudaFree(device_data);

    if (failed_at || mismatches != 0) {
        fprintf(stderr,
                "ds4: CUDA Q8 U16 load validation failed (%s%s%u); "
                "using byte loads\n",
                failed_at ? failed_at : "mismatches=",
                failed_at ? ": " : "",
                failed_at ? (unsigned)err : mismatches);
        (void)cudaGetLastError();
        return 0;
    }

    g_q8_u16_validation = 1;
    fprintf(stderr,
            "ds4: CUDA Q8 U16 DP4A loads enabled (32/32 bit-exact samples)\n");
    return 1;
}

static int cuda_moe_gb10_validate_signs(void) {
    if (g_moe_gb10_sign_validation != -1) {
        return g_moe_gb10_sign_validation == 1;
    }
    g_moe_gb10_sign_validation = -2;

    uint8_t host_signs[128];
    uint64_t host_grid[256];
    cudaError_t err = cudaMemcpyFromSymbol(host_signs,
                                           cuda_ksigns_iq2xs,
                                           sizeof(host_signs));
    uint32_t mismatch = UINT32_MAX;
    if (err == cudaSuccess) {
        mismatch = 128u;
        for (uint32_t i = 0; i < 128u; i++) {
            const uint8_t expected = (uint8_t)(
                i ^ (((uint32_t)__builtin_popcount(i) & 1u) << 7u));
            if (host_signs[i] != expected) {
                mismatch = i;
                break;
            }
        }
    }

    if (err == cudaSuccess && mismatch == 128u) {
        err = cudaMemcpyFromSymbol(host_grid,
                                   cuda_iq2xxs_grid,
                                   sizeof(host_grid));
    }
    if (err == cudaSuccess && mismatch == 128u) {
        err = cudaMemcpyToSymbol(cuda_iq2xxs_grid_global,
                                 host_grid,
                                 sizeof(host_grid));
    }

    if (err != cudaSuccess || mismatch != 128u) {
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4: CUDA GB10 MoE setup failed (%s); "
                    "using baseline decode kernel\n",
                    cudaGetErrorString(err));
        } else {
            fprintf(stderr,
                    "ds4: CUDA GB10 MoE sign validation failed (index=%u); "
                    "using baseline decode kernel\n",
                    mismatch);
        }
        (void)cudaGetLastError();
        g_moe_gb10_sign_validation = 0;
        return 0;
    }

    g_moe_gb10_sign_validation = 1;
    fprintf(stderr,
            "ds4: CUDA GB10 IQ2 computed-sign path ready "
            "(global grid, 128/128 exact)\n");
    return 1;
}

__global__ static DS4_CUDA_UNUSED void matmul_q8_0_kernel(
        float *out,
        const unsigned char *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;
    const uint64_t blocks = (in_dim + 31) / 32;
    const unsigned char *wr = w + row * blocks * 34;
    const float *xr = x + tok * in_dim;
    float acc = 0.0f;

    for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        float amax = 0.0f;
        for (uint64_t i = 0; i < bn; i++) amax = fmaxf(amax, fabsf(xr[i0 + i]));
        float d = amax / 127.0f;
        float id = d != 0.0f ? 1.0f / d : 0.0f;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        int dot = 0;
        for (uint64_t i = 0; i < bn; i++) {
            int q = (int)lrintf(xr[i0 + i] * id);
            q = q > 127 ? 127 : (q < -128 ? -128 : q);
            dot += (int)qs[i] * q;
        }
        acc += __half2float(*scale_h) * d * (float)dot;
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

__global__ static void quantize_q8_0_f32_kernel(
        int8_t *xq,
        float *xscale,
        const float *x,
        uint64_t in_dim,
        uint64_t blocks) {
    uint64_t b = blockIdx.x;
    uint64_t tok = blockIdx.y;
    if (b >= blocks) return;
    uint64_t i0 = b * 32;
    uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
    const float *xr = x + tok * in_dim + i0;

    float a = 0.0f;
    if (threadIdx.x < bn) a = fabsf(xr[threadIdx.x]);
    a = warp_max_f32(a);
    const float d = __shfl_sync(0xffffffffu, a, 0) / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    if (threadIdx.x == 0) xscale[tok * blocks + b] = d;
    int8_t *dst = xq + (tok * blocks + b) * 32;
    if (threadIdx.x < bn) {
        int v = (int)lrintf(xr[threadIdx.x] * id);
        v = v > 127 ? 127 : (v < -128 ? -128 : v);
        dst[threadIdx.x] = (int8_t)v;
    } else {
        dst[threadIdx.x] = 0;
    }
}

static int launch_quantize_q8_0_f32_rows(
        int8_t *xq,
        float *xscale,
        const float *x,
        uint64_t rows,
        uint64_t in_dim,
        uint64_t blocks,
        const char *what) {
    if (!xq || !xscale || !x || rows == 0 || in_dim == 0 || blocks == 0) return 0;
    const uint64_t max_grid_y = 32768u;
    for (uint64_t row0 = 0; row0 < rows; ) {
        uint64_t chunk = rows - row0;
        if (chunk > max_grid_y) chunk = max_grid_y;
        dim3 qgrid((unsigned)blocks, (unsigned)chunk, 1);
        quantize_q8_0_f32_kernel<<<qgrid, 32>>>(
                xq + row0 * blocks * 32u,
                xscale + row0 * blocks,
                x + row0 * in_dim,
                in_dim,
                blocks);
        if (!cuda_ok(cudaGetLastError(), what)) return 0;
        row0 += chunk;
    }
    return 1;
}

__global__ static void matmul_q8_0_preq_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;
    const unsigned char *wr = w + row * blocks * 34;
    const int8_t *xqr = xq + tok * blocks * 32;
    const float *xsr = xscale + tok * blocks;
    float acc = 0.0f;
    for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xqr + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

__global__ static void matmul_q8_0_preq_warp8_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks,
        int use_dp4a) {
    uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const unsigned char *wr = w + row * blocks * 34;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xq + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xscale[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) out[row] = acc;
}

__global__ static void matmul_q8_0_pair_preq_warp8_kernel(
        float *out0,
        float *out1,
        const unsigned char *w0,
        const unsigned char *w1,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        uint64_t blocks,
        int use_dp4a) {
    uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    if (row >= out0_dim && row >= out1_dim) return;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const unsigned char *wr0 = row < out0_dim ? w0 + row * blocks * 34 : NULL;
    const unsigned char *wr1 = row < out1_dim ? w1 + row * blocks * 34 : NULL;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const int8_t *xqb = xq + b * 32;
        const float xs = xscale[b];
        if (wr0) {
            const __half *scale_h = (const __half *)(wr0 + b * 34);
            const int8_t *qs = (const int8_t *)(wr0 + b * 34 + 2);
            int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
            acc0 += __half2float(*scale_h) * xs * (float)dot;
        }
        if (wr1) {
            const __half *scale_h = (const __half *)(wr1 + b * 34);
            const int8_t *qs = (const int8_t *)(wr1 + b * 34 + 2);
            int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
            acc1 += __half2float(*scale_h) * xs * (float)dot;
        }
    }
    acc0 = warp_sum_f32(acc0);
    acc1 = warp_sum_f32(acc1);
    if (lane == 0) {
        if (row < out0_dim) out0[row] = acc0;
        if (row < out1_dim) out1[row] = acc1;
    }
}

__global__ static void matmul_q8_0_hc_expand_preq_warp8_kernel(
        float *out_hc,
        float *block_out,
        const float *block_add,
        const float *residual_hc,
        const float *split,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t n_embd,
        uint32_t n_hc,
        uint64_t blocks,
        int has_add,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const unsigned char *wr = w + row * blocks * 34;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xq + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xscale[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) {
        const uint32_t d = (uint32_t)row;
        block_out[d] = acc;
        float block_v = acc;
        if (has_add) block_v += block_add[d];
        const float *post = split + n_hc;
        const float *comb = split + 2u * n_hc;
        for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
            float hc_acc = block_v * post[dst_hc];
            for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
                const float comb_v = comb[dst_hc + (uint64_t)src_hc * n_hc];
                const float res_v = residual_hc[(uint64_t)src_hc * n_embd + d];
                hc_acc += comb_v * res_v;
            }
            out_hc[(uint64_t)dst_hc * n_embd + d] = hc_acc;
        }
    }
}

__global__ static void matmul_q8_0_preq_batch_warp8_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim || tok >= n_tok) return;

    const unsigned char *wr = w + row * blocks * 34;
    const int8_t *xqr = xq + tok * blocks * 32;
    const float *xsr = xscale + tok * blocks;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xqr + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) out[tok * out_dim + row] = acc;
}

/* Tiny verifier batches repeatedly multiply the same Q8 row by 2..6 token
 * activations.  The generic grid uses a separate CUDA block for every token,
 * re-reading the full weight row N times.  GB10 has ample parallelism across
 * output rows, so keep one block per row and reuse each packed Q8 block across
 * all token accumulators.  Per-token block assignment and the 256-lane tree
 * reduction are unchanged, preserving the native Q8 numerical order. */
__global__ static void matmul_q8_0_preq_batch_reuse_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t n_tok,
        uint64_t blocks,
        int use_dp4a) {
    enum { MAX_TOKENS = 6, THREADS = 256 };
    const uint64_t row = (uint64_t)blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (row >= out_dim || n_tok == 0 || n_tok > MAX_TOKENS) return;

    float acc[MAX_TOKENS];
#pragma unroll
    for (uint32_t t = 0; t < MAX_TOKENS; t++) acc[t] = 0.0f;
    const unsigned char *wr = w + row * blocks * 34u;

    for (uint64_t b = tid; b < blocks; b += THREADS) {
        const uint64_t i0 = b * 32u;
        const uint64_t bn = in_dim - i0 < 32u ? in_dim - i0 : 32u;
        const __half *scale_h = (const __half *)(wr + b * 34u);
        const int8_t *qs = (const int8_t *)(wr + b * 34u + 2u);
        const float ws = __half2float(*scale_h);

        if (use_dp4a && bn == 32u) {
            int32_t qw[8];
#pragma unroll
            for (uint32_t j = 0; j < 8u; j++) {
                qw[j] = use_dp4a > 1
                    ? load_i8x4_i32_u16(qs + j * 4u)
                    : load_i8x4_i32_unaligned(qs + j * 4u);
            }
            for (uint32_t t = 0; t < n_tok; t++) {
                const int8_t *xqb = xq + ((uint64_t)t * blocks + b) * 32u;
                int32_t dot = 0;
#pragma unroll
                for (uint32_t j = 0; j < 8u; j++) {
                    dot = __dp4a(qw[j],
                                  load_i8x4_i32_aligned(xqb + j * 4u),
                                  dot);
                }
                acc[t] += ws * xscale[(uint64_t)t * blocks + b] * (float)dot;
            }
        } else {
            for (uint32_t t = 0; t < n_tok; t++) {
                const int8_t *xqb = xq + ((uint64_t)t * blocks + b) * 32u;
                const int32_t dot = dot_i8_block(qs, xqb, bn, use_dp4a);
                acc[t] += ws * xscale[(uint64_t)t * blocks + b] * (float)dot;
            }
        }
    }

    __shared__ float partial[MAX_TOKENS][THREADS];
    for (uint32_t t = 0; t < n_tok; t++) partial[t][tid] = acc[t];
    __syncthreads();
    for (uint32_t stride = THREADS >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            for (uint32_t t = 0; t < n_tok; t++) {
                partial[t][tid] += partial[t][tid + stride];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        for (uint32_t t = 0; t < n_tok; t++) {
            out[(uint64_t)t * out_dim + row] = partial[t][0];
        }
    }
}

__global__ static void dequant_q8_0_to_f16_kernel(
        __half *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = in_dim * out_dim;
    if (gid >= n) return;
    uint64_t row = gid / in_dim;
    uint64_t i = gid - row * in_dim;
    uint64_t b = i / 32;
    uint64_t j = i - b * 32;
    const unsigned char *blk = w + (row * blocks + b) * 34;
    const __half scale = *(const __half *)blk;
    const int8_t q = *(const int8_t *)(blk + 2 + j);
    out[gid] = __hmul(scale, __float2half((float)q));
}

__global__ static void dequant_q8_0_to_f32_kernel(
        float *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = in_dim * out_dim;
    if (gid >= n) return;
    uint64_t row = gid / in_dim;
    uint64_t i = gid - row * in_dim;
    uint64_t b = i / 32;
    uint64_t j = i - b * 32;
    const unsigned char *blk = w + (row * blocks + b) * 34;
    const float scale = __half2float(*(const __half *)blk);
    const int8_t q = *(const int8_t *)(blk + 2 + j);
    out[gid] = scale * (float)q;
}

__global__ static void grouped_q8_0_a_preq_warp8_kernel(
        float *low,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        uint32_t n_tokens,
        uint64_t blocks,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    if (row >= low_dim || tok >= n_tokens) return;

    const uint64_t group = row / rank;
    const uint64_t row_in_group = row - group * rank;
    const unsigned char *wr = w + (group * rank + row_in_group) * blocks * 34;
    const uint64_t xrow = tok * (uint64_t)n_groups + group;
    const int8_t *xqr = xq + xrow * blocks * 32;
    const float *xsr = xscale + xrow * blocks;
    float acc = 0.0f;

    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = group_dim - i0 < 32 ? group_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xqr + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) low[tok * low_dim + row] = acc;
}

__global__ static void rms_norm_plain_kernel(float *out, const float *x, uint32_t n, uint32_t rows, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = x + (uint64_t)row * n;
    float *orow = out + (uint64_t)row * n;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        orow[i] = xr[i] * scale;
    }
}

__global__ static void rms_norm_plain_f16_kernel(
        __half *out,
        const float *x,
        uint32_t n,
        uint32_t rows,
        float eps) {
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = x + (uint64_t)row * n;
    __half *orow = out + (uint64_t)row * n;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        orow[i] = __float2half_rn(xr[i] * scale);
    }
}

__global__ static void rms_norm_weight_kernel(float *out, const float *x, const float *w, uint32_t n, uint32_t rows, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = x + (uint64_t)row * n;
    float *orow = out + (uint64_t)row * n;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        orow[i] = xr[i] * scale * w[i];
    }
}

__global__ static void dsv4_qkv_rms_norm_rows_kernel(
        float *q_out,
        const float *q,
        const float *q_w,
        uint32_t q_n,
        float *kv_out,
        const float *kv,
        const float *kv_w,
        uint32_t kv_n,
        uint32_t rows,
        float eps) {
    const uint32_t row = blockIdx.x;
    const uint32_t which = blockIdx.y;
    if (row >= rows || which > 1u) return;
    const uint32_t n = which == 0u ? q_n : kv_n;
    const float *xr = (which == 0u ? q : kv) + (uint64_t)row * n;
    float *orow = (which == 0u ? q_out : kv_out) + (uint64_t)row * n;
    const float *w = which == 0u ? q_w : kv_w;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        orow[i] = xr[i] * scale * w[i];
    }
}

__global__ static void head_rms_norm_kernel(float *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= n_tok * n_head) return;
    float *xr = x + (uint64_t)row * head_dim;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)head_dim + eps);
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) xr[i] *= scale;
}

__device__ static float rope_yarn_ramp_dev(float low, float high, int i0);

__global__ static void head_rms_norm_rope_tail_kernel(
        float *x,
        uint32_t n_tok,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t pos0,
        uint32_t n_ctx_orig,
        int inverse,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        float eps) {
    uint32_t row = blockIdx.x;
    if (row >= n_tok * n_head) return;
    uint32_t t = row / n_head;
    float *xr = x + (uint64_t)row * head_dim;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)head_dim + eps);
    const uint32_t n_nope = head_dim - n_rot;
    for (uint32_t i = threadIdx.x; i < n_nope; i += blockDim.x) {
        xr[i] *= scale;
    }

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
        corr1 = ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1), corr1);
    }
    for (uint32_t pair = threadIdx.x; pair < n_rot / 2; pair += blockDim.x) {
        uint32_t i = pair * 2u;
        float theta_extrap = (float)(pos0 + t) * powf(freq_base, -((float)i) / (float)n_rot);
        float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            float ramp_mix = rope_yarn_ramp_dev(corr0, corr1, (int)i) * ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        float c = cosf(theta) * mscale;
        float s = sinf(theta) * mscale;
        if (inverse) s = -s;
        float *tail = xr + n_nope;
        float x0 = tail[i] * scale;
        float x1 = tail[i + 1] * scale;
        tail[i] = x0 * c - x1 * s;
        tail[i + 1] = x0 * s + x1 * c;
    }
}

__device__ static float rope_yarn_ramp_dev(float low, float high, int i0) {
    float y = ((float)(i0 / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

__global__ static void rope_tail_kernel(
        float *x,
        uint32_t n_tok,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t pos0,
        uint32_t pos_stride,
        uint32_t n_ctx_orig,
        int inverse,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pairs = n_tok * n_head * (n_rot / 2);
    if (gid >= pairs) return;
    uint32_t pair = gid % (n_rot / 2);
    uint32_t tmp = gid / (n_rot / 2);
    uint32_t h = tmp % n_head;
    uint32_t t = tmp / n_head;
    uint32_t n_nope = head_dim - n_rot;
    uint32_t i = pair * 2;

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
        corr1 = ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1), corr1);
    }

    float theta_extrap = (float)(pos0 + t * pos_stride) * powf(freq_base, -((float)i) / (float)n_rot);
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = attn_factor;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp_dev(corr0, corr1, (int)i) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    float c = cosf(theta) * mscale;
    float s = sinf(theta) * mscale;
    if (inverse) s = -s;

    float *tail = x + ((uint64_t)t * n_head + h) * head_dim + n_nope;
    float x0 = tail[i];
    float x1 = tail[i + 1];
    tail[i] = x0 * c - x1 * s;
    tail[i + 1] = x0 * s + x1 * c;
}

__device__ static float dsv4_e4m3fn_value_dev(int i) {
    int exp = (i >> 3) & 15;
    int mant = i & 7;
    if (exp == 0) return (float)mant * 0.001953125f;
    return (1.0f + (float)mant * 0.125f) * exp2f((float)exp - 7.0f);
}

__device__ static float dsv4_e4m3fn_dequant_dev(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = fminf(fabsf(x), 448.0f);
    int lo = 0, hi = 126;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (dsv4_e4m3fn_value_dev(mid) <= ax) lo = mid;
        else hi = mid - 1;
    }
    int best = lo;
    if (best < 126) {
        float bd = fabsf(ax - dsv4_e4m3fn_value_dev(best));
        float nd = fabsf(ax - dsv4_e4m3fn_value_dev(best + 1));
        if (nd < bd || (nd == bd && (((best + 1) & 1) == 0) && ((best & 1) != 0))) best++;
    }
    return sign * dsv4_e4m3fn_value_dev(best);
}

__device__ static float dsv4_e2m1fn_value_dev(int i) {
    switch (i & 7) {
    case 0: return 0.0f;
    case 1: return 0.5f;
    case 2: return 1.0f;
    case 3: return 1.5f;
    case 4: return 2.0f;
    case 5: return 3.0f;
    case 6: return 4.0f;
    default: return 6.0f;
    }
}

__device__ static float dsv4_e2m1fn_dequant_dev(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = fminf(fabsf(x), 6.0f);
    int best = 0;
    float best_diff = fabsf(ax - dsv4_e2m1fn_value_dev(0));
    for (int i = 1; i < 8; i++) {
        float diff = fabsf(ax - dsv4_e2m1fn_value_dev(i));
        if (diff < best_diff || (diff == best_diff && ((i & 1) == 0) && ((best & 1) != 0))) {
            best = i;
            best_diff = diff;
        }
    }
    return sign * dsv4_e2m1fn_value_dev(best);
}

__device__ static float model_scalar_dev(const void *base, uint64_t offset, uint32_t type, uint64_t idx) {
    const char *p = (const char *)base + offset;
    if (type == 1u) return __half2float(((const __half *)p)[idx]);
    return ((const float *)p)[idx];
}

__device__ static float rope_yarn_ramp_cpu_equiv_dev(float low, float high, int i0) {
    float y = ((float)(i0 / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

__device__ static DS4_CUDA_UNUSED void rope_tail_one_dev(float *x, uint32_t head_dim, uint32_t n_rot, uint32_t pos, uint32_t n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    uint32_t n_nope = head_dim - n_rot;
    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        float denom = 2.0f * logf(freq_base);
        corr0 = fmaxf(0.0f, floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom));
        corr1 = fminf((float)(n_rot - 1), ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom));
    }
    for (uint32_t i = 0; i < n_rot; i += 2) {
        float theta_extrap = (float)pos * powf(freq_base, -((float)i) / (float)n_rot);
        float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            float mix = rope_yarn_ramp_cpu_equiv_dev(corr0, corr1, (int)i) * ext_factor;
            theta = theta_interp * (1.0f - mix) + theta_extrap * mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        float c = cosf(theta) * mscale;
        float s = sinf(theta) * mscale;
        float x0 = x[n_nope + i];
        float x1 = x[n_nope + i + 1];
        x[n_nope + i] = x0 * c - x1 * s;
        x[n_nope + i + 1] = x0 * s + x1 * c;
    }
}

__global__ static void fp8_kv_quantize_kernel(float *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t n_nope = head_dim - n_rot;
    float *xr = x + (uint64_t)row * head_dim;
    __shared__ float scratch[64];
    for (uint32_t off = 0; off < n_nope; off += 64) {
        float v = 0.0f;
        if (off + tid < n_nope) v = xr[off + tid];
        scratch[tid] = off + tid < n_nope ? fabsf(v) : 0.0f;
        __syncthreads();
        for (uint32_t stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            __syncthreads();
        }
        float scale = exp2f(ceilf(log2f(fmaxf(scratch[0], 1.0e-4f) / 448.0f)));
        if (off + tid < n_nope) {
            float q = dsv4_e4m3fn_dequant_dev(fminf(448.0f, fmaxf(-448.0f, v / scale))) * scale;
            xr[off + tid] = q;
        }
        __syncthreads();
    }
}

__global__ static void indexer_hadamard_fp4_kernel(float *x, uint32_t n_rows, uint32_t head_dim) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (row >= n_rows || head_dim != 128u || tid >= 128u) return;

    __shared__ float vals[128];
    __shared__ float absbuf[128];
    float *xr = x + (uint64_t)row * head_dim;
    vals[tid] = xr[tid];
    __syncthreads();

    for (uint32_t stride = 1u; stride < 128u; stride <<= 1u) {
        if ((tid & stride) == 0u) {
            uint32_t base = (tid & ~(2u * stride - 1u)) + (tid & (stride - 1u));
            float a = vals[base];
            float b = vals[base + stride];
            vals[base] = a + b;
            vals[base + stride] = a - b;
        }
        __syncthreads();
    }

    float v = vals[tid] * 0.08838834764831845f;
    uint32_t fp4_block = tid >> 5u;
    uint32_t lane = tid & 31u;
    uint32_t block_base = fp4_block * 32u;
    absbuf[tid] = fabsf(v);
    __syncthreads();

    for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            absbuf[block_base + lane] = fmaxf(absbuf[block_base + lane],
                                              absbuf[block_base + lane + stride]);
        }
        __syncthreads();
    }

    float amax = fmaxf(absbuf[block_base], 7.052966104933725e-38f);
    float scale = exp2f(ceilf(log2f(amax / 6.0f)));
    xr[tid] = dsv4_e2m1fn_dequant_dev(fminf(6.0f, fmaxf(-6.0f, v / scale))) * scale;
}

__global__ static void store_raw_kv_batch_kernel(float *raw, const float *kv, uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens, uint32_t head_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * head_dim;
    if (gid >= n) return;
    uint32_t d = gid % head_dim;
    uint32_t t = gid / head_dim;
    uint32_t row = (pos0 + t) % raw_cap;
    raw[(uint64_t)row * head_dim + d] = __half2float(__float2half(kv[(uint64_t)t * head_dim + d]));
}

__global__ static void ring_rows_save_kernel(
        float       *backup,
        const float *ring,
        uint32_t     ring_cap,
        uint32_t     pos0,
        uint32_t     n_rows,
        uint32_t     width) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_rows * width;
    if (gid >= n) return;
    const uint32_t row = (uint32_t)(gid / width);
    const uint32_t d = (uint32_t)(gid % width);
    backup[gid] = ring[(uint64_t)((pos0 + row) % ring_cap) * width + d];
}

__global__ static void ring_rows_restore_kernel(
        float       *ring,
        const float *backup,
        uint32_t     ring_cap,
        uint32_t     pos0,
        uint32_t     restore_from,
        uint32_t     n_rows,
        uint32_t     width) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t restore_rows = n_rows - restore_from;
    const uint64_t n = (uint64_t)restore_rows * width;
    if (gid >= n) return;
    const uint32_t rel = (uint32_t)(gid / width);
    const uint32_t row = restore_from + rel;
    const uint32_t d = (uint32_t)(gid % width);
    ring[(uint64_t)((pos0 + row) % ring_cap) * width + d] =
        backup[(uint64_t)row * width + d];
}

__global__ static void interleave3_rows_kernel(
        float *out,
        const float *a,
        const float *b,
        const float *c,
        uint32_t n_rows,
        uint32_t width) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_rows * 3u * width;
    if (gid >= n) return;
    uint32_t d = gid % width;
    uint64_t q = gid / width;
    uint32_t plane = q % 3u;
    uint32_t row = q / 3u;
    const float *src = plane == 0u ? a : (plane == 1u ? b : c);
    out[gid] = src[(uint64_t)row * width + d];
}

__global__ static void dspark_gather_kv_kernel(
        float *dst,
        const float *main_kv,
        const float *draft_kv,
        uint32_t n_main,
        uint32_t main_cap,
        uint32_t main_start,
        uint32_t n_draft,
        uint32_t head_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)(n_main + n_draft) * head_dim;
    if (gid >= n) return;
    uint32_t d = gid % head_dim;
    uint32_t row = gid / head_dim;
    if (row < n_main) {
        uint32_t src_row = (main_start + row) % main_cap;
        dst[gid] = main_kv[(uint64_t)src_row * head_dim + d];
    } else {
        dst[gid] = draft_kv[(uint64_t)(row - n_main) * head_dim + d];
    }
}

__global__ static void dspark_attention_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *kv,
        uint32_t n_tokens,
        uint32_t n_keys,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head || n_keys > 256u) return;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    __shared__ float scores[256];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    const float scale = rsqrtf((float)head_dim);
    float local_max = sinks[h];
    for (uint32_t r = threadIdx.x; r < n_keys; r += blockDim.x) {
        const float *kr = kv + (uint64_t)r * head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kr[d];
        const float s = dot * scale;
        scores[r] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] = fmaxf(partial[threadIdx.x],
                                          partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t r = threadIdx.x; r < n_keys; r += blockDim.x) {
        const float p = expf(scores[r] - max_s);
        scores[r] = p;
        den_local += p;
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < n_keys; r++) {
            acc += kv[(uint64_t)r * head_dim + d] * scores[r];
        }
        oh[d] = acc / denom;
    }
}

__global__ static void dspark_confidence_kernel(
        float        *out,
        const float  *hidden,
        const float  *markov,
        const __half *weight,
        uint32_t      hidden_dim,
        uint32_t      markov_dim) {
    const uint32_t row = blockIdx.x;
    float acc = 0.0f;
    for (uint32_t d = threadIdx.x; d < hidden_dim; d += blockDim.x) {
        acc += hidden[(uint64_t)row * hidden_dim + d] *
               __half2float(weight[d]);
    }
    for (uint32_t d = threadIdx.x; d < markov_dim; d += blockDim.x) {
        acc += markov[(uint64_t)row * markov_dim + d] *
               __half2float(weight[hidden_dim + d]);
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = partial[0];
}

__global__ static void attention_prefill_raw_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        uint32_t n_tokens,
        uint32_t window,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    uint32_t raw_count = t + 1 < window ? t + 1 : window;
    uint32_t raw_start = t + 1 - raw_count;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    __shared__ float scores[256];
    __shared__ float partial[128];
    __shared__ float max_s;
    __shared__ float denom;
    float scale = rsqrtf((float)head_dim);
    float local_max = sinks[h];
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        const float *kv = raw_kv + (uint64_t)(raw_start + r) * head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kv[d];
        scores[r] = dot * scale;
        local_max = fmaxf(local_max, scores[r]);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    if (threadIdx.x == 0) {
        float den = expf(sinks[h] - max_s);
        for (uint32_t r = 0; r < raw_count; r++) {
            scores[r] = expf(scores[r] - max_s);
            den += scores[r];
        }
        denom = den;
    }
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            acc += raw_kv[(uint64_t)(raw_start + r) * head_dim + d] * scores[r];
        }
        oh[d] = acc / denom;
    }
}

template <bool COMP_F16>
__device__ __forceinline__ float attention_comp_load(
        const void *comp_kv,
        uint64_t idx) {
    if constexpr (COMP_F16) {
        return __half2float(((const half *)comp_kv)[idx]);
    } else {
        return ((const float *)comp_kv)[idx];
    }
}

template <bool COMP_F16>
__device__ __forceinline__ float4 attention_comp_load4(
        const void *comp_kv,
        uint64_t idx) {
    if constexpr (COMP_F16) {
        const half2 *src = (const half2 *)((const half *)comp_kv + idx);
        const float2 a = __half22float2(src[0]);
        const float2 b = __half22float2(src[1]);
        return make_float4(a.x, a.y, b.x, b.y);
    } else {
        return *(const float4 *)((const float *)comp_kv + idx);
    }
}

template <bool COMP_F16>
__global__ static void attention_prefill_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const void *comp_kv,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    uint32_t raw_start = (window != 0 && t + 1u > window) ? t + 1u - window : 0u;
    uint32_t raw_count = t + 1u - raw_start;
    uint32_t visible_comp = (t + 1u) / ratio;
    if (visible_comp > n_comp) visible_comp = n_comp;
    __shared__ float scores[512];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float scale = rsqrtf((float)head_dim);
    float local_max = sinks[h];
    uint32_t n_score = raw_count + visible_comp;

    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        const float *kvrow = raw_kv + (uint64_t)(raw_start + r) * head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
        scores[r] = dot * scale;
        local_max = fmaxf(local_max, scores[r]);
    }
    for (uint32_t c = threadIdx.x; c < visible_comp; c += blockDim.x) {
        float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
        float s = -INFINITY;
        if (add > -1.0e20f) {
            float dot = 0.0f;
            const uint64_t row_off = (uint64_t)c * head_dim;
            for (uint32_t d = 0; d < head_dim; d++) {
                dot += qh[d] * attention_comp_load<COMP_F16>(comp_kv, row_off + d);
            }
            s = dot * scale + add;
        }
        scores[raw_count + c] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv[(uint64_t)(raw_start + r) * head_dim + d] * scores[r];
        for (uint32_t c = 0; c < visible_comp; c++) {
            acc += attention_comp_load<COMP_F16>(
                       comp_kv, (uint64_t)c * head_dim + d) *
                   scores[raw_count + c];
        }
        oh[d] = acc / denom;
    }
}

__global__ static void attention_prefill_raw_softmax_kernel(
        float *scores,
        const float *sinks,
        uint32_t n_tokens,
        uint32_t window,
        uint32_t n_keys) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens) return;
    float *row = scores + ((uint64_t)h * n_tokens + t) * n_keys;
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float local_max = sinks[h];
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        bool valid = k <= t && (window == 0 || t - k < window);
        float s = valid ? row[k] : -INFINITY;
        row[k] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float p = isfinite(row[k]) ? expf(row[k] - max_s) : 0.0f;
        row[k] = p;
        den_local += p;
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) row[k] /= denom;
}

__global__ static void attention_prefill_mixed_softmax_kernel(
        float *scores,
        const float *sinks,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_keys) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || ratio == 0) return;
    float *row = scores + ((uint64_t)h * n_tokens + t) * n_keys;
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float local_max = sinks[h];
    const uint32_t visible_comp = (t + 1u) / ratio;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float s = -INFINITY;
        if (k < n_tokens) {
            if (k <= t && (window == 0 || t - k < window)) s = row[k];
        } else {
            uint32_t c = k - n_tokens;
            if (c < n_comp && c < visible_comp) {
                float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
                if (add > -1.0e20f) s = row[k] + add;
            }
        }
        row[k] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float p = isfinite(row[k]) ? expf(row[k] - max_s) : 0.0f;
        row[k] = p;
        den_local += p;
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) row[k] /= denom;
}

template <bool COMP_F16>
__global__ static void attention_prefill_pack_mixed_kv_kernel(
        float *dst,
        const float *raw_kv,
        const void *comp_kv,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t head_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)(n_tokens + n_comp) * head_dim;
    if (gid >= n) return;
    uint32_t d = gid % head_dim;
    uint32_t r = gid / head_dim;
    dst[gid] = r < n_tokens
        ? raw_kv[(uint64_t)r * head_dim + d]
        : attention_comp_load<COMP_F16>(
              comp_kv, (uint64_t)(r - n_tokens) * head_dim + d);
}

__global__ static void attention_prefill_unpack_heads_kernel(
        float *heads,
        const float *tmp,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t head_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
    if (gid >= n) return;
    uint32_t d = gid % head_dim;
    uint64_t q = gid / head_dim;
    uint32_t h = q % n_head;
    uint32_t t = q / n_head;
    heads[gid] = tmp[((uint64_t)h * n_tokens + t) * head_dim + d];
}

__global__ static void attention_pack_group_heads_f16_kernel(
        __half *dst,
        const float *heads,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t group_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t pair_dim = group_dim / 2u;
    uint64_t n = (uint64_t)n_groups * n_tokens * pair_dim;
    if (gid >= n || (group_dim & 1u) != 0u) return;
    uint32_t d2 = gid % pair_dim;
    uint64_t q = gid / pair_dim;
    uint32_t t = q % n_tokens;
    uint32_t g = q / n_tokens;
    const uint64_t src = ((uint64_t)t * n_groups + g) * group_dim + 2u * d2;
    const uint64_t dst_off = ((uint64_t)g * n_tokens + t) * group_dim + 2u * d2;
    const float2 v = *reinterpret_cast<const float2 *>(heads + src);
    *reinterpret_cast<__half2 *>(dst + dst_off) = __floats2half2_rn(v.x, v.y);
}

__global__ static void attention_inverse_rope_pack_group_heads_f16_kernel(
        __half *dst,
        const float *heads,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t group_dim,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t pos0,
        uint32_t n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t pair_dim = group_dim / 2u;
    const uint64_t n = (uint64_t)n_groups * n_tokens * pair_dim;
    if (gid >= n || (group_dim & 1u) != 0u ||
        (head_dim & 1u) != 0u || n_rot > head_dim || (n_rot & 1u) != 0u) {
        return;
    }
    const uint32_t d2 = gid % pair_dim;
    const uint64_t q = gid / pair_dim;
    const uint32_t t = q % n_tokens;
    const uint32_t g = q / n_tokens;
    const uint32_t d = 2u * d2;
    const uint32_t head_d = d % head_dim;
    const uint64_t src = ((uint64_t)t * n_groups + g) * group_dim + d;
    const uint64_t dst_off = ((uint64_t)g * n_tokens + t) * group_dim + d;
    const float2 v = *reinterpret_cast<const float2 *>(heads + src);
    float x0 = v.x;
    float x1 = v.y;
    const uint32_t n_nope = head_dim - n_rot;
    if (head_d >= n_nope) {
        const uint32_t i = head_d - n_nope;
        float corr0 = 0.0f;
        float corr1 = 0.0f;
        if (ext_factor != 0.0f) {
            const float denom = 2.0f * logf(freq_base);
            corr0 = floorf((float)n_rot *
                    logf((float)n_ctx_orig /
                         (beta_fast * 2.0f * (float)M_PI)) / denom);
            corr1 = ceilf((float)n_rot *
                    logf((float)n_ctx_orig /
                         (beta_slow * 2.0f * (float)M_PI)) / denom);
            corr0 = fmaxf(0.0f, corr0);
            corr1 = fminf((float)(n_rot - 1u), corr1);
        }
        const float theta_extrap = (float)(pos0 + t) *
            powf(freq_base, -((float)i) / (float)n_rot);
        const float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            const float ramp_mix =
                rope_yarn_ramp_dev(corr0, corr1, (int)i) * ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        const float c = cosf(theta) * mscale;
        const float s = -sinf(theta) * mscale;
        const float r0 = x0 * c - x1 * s;
        const float r1 = x0 * s + x1 * c;
        x0 = r0;
        x1 = r1;
    }
    *reinterpret_cast<__half2 *>(dst + dst_off) = __floats2half2_rn(x0, x1);
}

__global__ static void attention_unpack_group_low_kernel(
        float *low,
        const float *tmp,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t rank) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t vec_rank = rank / 4u;
    uint64_t n = (uint64_t)n_groups * n_tokens * vec_rank;
    if (gid >= n || (rank & 3u) != 0u) return;
    uint32_t r4 = gid % vec_rank;
    uint64_t q = gid / vec_rank;
    uint32_t t = q % n_tokens;
    uint32_t g = q / n_tokens;
    uint32_t low_dim = n_groups * rank;
    const uint64_t src = ((uint64_t)g * n_tokens + t) * rank + 4u * r4;
    const uint64_t dst = (uint64_t)t * low_dim + (uint64_t)g * rank + 4u * r4;
    *reinterpret_cast<float4 *>(low + dst) =
        *reinterpret_cast<const float4 *>(tmp + src);
}

template <bool COMP_F16>
__global__ static void attention_decode_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const void *comp_kv,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    const bool single_all = (n_tokens == 1u && ratio == 0u);
    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = single_all ? n_comp : (n_comp ? (qpos + 1u) / ratio : 0u);
    if (visible_comp > n_comp) visible_comp = n_comp;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    __shared__ float scores[DS4_CUDA_ATTENTION_SCORE_CAP];
    __shared__ uint32_t raw_rows[256];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    float scale = rsqrtf((float)head_dim);
    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (single_all) {
                raw_count = n_raw > 256u ? 256u : n_raw;
            } else if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    __syncthreads();
    uint32_t n_score = raw_count + visible_comp;
    float local_max = sinks[h];
    if (visible_comp == 0 || n_tokens == 1u) {
        for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
            const float *kvrow = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
            scores[r] = dot * scale;
            local_max = fmaxf(local_max, scores[r]);
        }
        for (uint32_t c = threadIdx.x; c < visible_comp; c += blockDim.x) {
            float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
            float s = -INFINITY;
            if (add > -1.0e20f) {
                float dot = 0.0f;
                const uint64_t row_off = (uint64_t)c * head_dim;
                for (uint32_t d = 0; d < head_dim; d++) {
                    dot += qh[d] * attention_comp_load<COMP_F16>(comp_kv, row_off + d);
                }
                s = dot * scale + add;
            }
            scores[raw_count + c] = s;
            local_max = fmaxf(local_max, s);
        }
    } else {
        uint32_t qlane = threadIdx.x & 7u;
        uint32_t qgroup = threadIdx.x >> 3u;
        for (uint32_t row0 = 0; row0 < n_score; row0 += 32u) {
            uint32_t row = row0 + qgroup;
            if (row < n_score) {
                float add = 0.0f;
                const float *raw_row = NULL;
                uint32_t comp_row = 0u;
                bool use_comp = false;
                if (row < raw_count) {
                    raw_row = raw_kv + (uint64_t)raw_rows[row] * head_dim;
                } else {
                    uint32_t c = row - raw_count;
                    add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
                    if (add > -1.0e20f) {
                        comp_row = c;
                        use_comp = true;
                    }
                }
                float s = -INFINITY;
                if (raw_row || use_comp) {
                    float dot = 0.0f;
                    const uint64_t comp_off = (uint64_t)comp_row * head_dim;
                    for (uint32_t d = qlane; d < head_dim; d += 8u) {
                        const float kv = raw_row
                            ? raw_row[d]
                            : attention_comp_load<COMP_F16>(comp_kv, comp_off + d);
                        dot += qh[d] * kv;
                    }
                    const uint32_t mask = 0xffu << (threadIdx.x & 24u);
                    for (uint32_t off = 4u; off > 0u; off >>= 1u) {
                        dot += __shfl_down_sync(mask, dot, off, 8);
                    }
                    s = dot * scale + add;
                }
                if (qlane == 0) scores[row] = s;
            }
        }
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
            local_max = fmaxf(local_max, scores[i]);
        }
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    if (head_dim == 512u && blockDim.x == 256u) {
        uint32_t d0 = threadIdx.x;
        uint32_t d1 = d0 + 256u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            float s = scores[r];
            const float *kv = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        for (uint32_t c = 0; c < visible_comp; c++) {
            float s = scores[raw_count + c];
            const uint64_t row_off = (uint64_t)c * head_dim;
            acc0 += attention_comp_load<COMP_F16>(comp_kv, row_off + d0) * s;
            acc1 += attention_comp_load<COMP_F16>(comp_kv, row_off + d1) * s;
        }
        oh[d0] = acc0 / denom;
        oh[d1] = acc1 / denom;
    } else {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv[(uint64_t)raw_rows[r] * head_dim + d] * scores[r];
            for (uint32_t c = 0; c < visible_comp; c++) {
                acc += attention_comp_load<COMP_F16>(
                           comp_kv, (uint64_t)c * head_dim + d) *
                       scores[raw_count + c];
            }
            oh[d] = acc / denom;
        }
    }
}

template <bool COMP_F16>
__global__ static void attention_indexed_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const void *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = n_comp;
    if (ratio != 0) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    __shared__ float scores[768];
    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t comp_rows[512];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    __shared__ uint32_t comp_count;
    float scale = rsqrtf((float)head_dim);
    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        comp_count = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    for (uint32_t i = threadIdx.x; i < top_k; i += blockDim.x) {
        int32_t c = topk[(uint64_t)t * top_k + i];
        if (c >= 0 && (uint32_t)c < visible_comp) {
            uint32_t slot = atomicAdd(&comp_count, 1u);
            if (slot < 512u) comp_rows[slot] = (uint32_t)c;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        if (comp_count > 512u) comp_count = 512u;
    }
    __syncthreads();
    uint32_t n_score = raw_count + comp_count;
    float local_max = sinks[h];
    if (comp_count == 0) {
        for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
            const float *kvrow = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
            scores[r] = dot * scale;
            local_max = fmaxf(local_max, scores[r]);
        }
    } else {
        uint32_t qlane = threadIdx.x & 7u;
        uint32_t qgroup = threadIdx.x >> 3u;
        for (uint32_t row0 = 0; row0 < n_score; row0 += 32u) {
            uint32_t row = row0 + qgroup;
            if (row < n_score) {
                float dot = 0.0f;
                const bool use_comp = row >= raw_count;
                const float *raw_row = use_comp
                    ? NULL
                    : raw_kv + (uint64_t)raw_rows[row] * head_dim;
                const uint64_t comp_off = use_comp
                    ? (uint64_t)comp_rows[row - raw_count] * head_dim
                    : 0u;
                for (uint32_t d = qlane; d < head_dim; d += 8u) {
                    const float kv = use_comp
                        ? attention_comp_load<COMP_F16>(comp_kv, comp_off + d)
                        : raw_row[d];
                    dot += qh[d] * kv;
                }
                const uint32_t mask = 0xffu << (threadIdx.x & 24u);
                for (uint32_t off = 4u; off > 0u; off >>= 1u) {
                    dot += __shfl_down_sync(mask, dot, off, 8);
                }
                if (qlane == 0) scores[row] = dot * scale;
            }
        }
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
            local_max = fmaxf(local_max, scores[i]);
        }
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    if (head_dim == 512u && blockDim.x == 256u) {
        uint32_t d0 = threadIdx.x;
        uint32_t d1 = d0 + 256u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            float s = scores[r];
            const float *kv = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        for (uint32_t c = 0; c < comp_count; c++) {
            float s = scores[raw_count + c];
            const uint64_t row_off = (uint64_t)comp_rows[c] * head_dim;
            acc0 += attention_comp_load<COMP_F16>(comp_kv, row_off + d0) * s;
            acc1 += attention_comp_load<COMP_F16>(comp_kv, row_off + d1) * s;
        }
        oh[d0] = acc0 / denom;
        oh[d1] = acc1 / denom;
    } else {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv[(uint64_t)raw_rows[r] * head_dim + d] * scores[r];
            for (uint32_t s = 0; s < comp_count; s++) {
                acc += attention_comp_load<COMP_F16>(
                           comp_kv, (uint64_t)comp_rows[s] * head_dim + d) *
                       scores[raw_count + s];
            }
            oh[d] = acc / denom;
        }
    }
}

/* Decode on Flash uses one shared 512-wide latent K/V row for every attention
 * head.  The generic one-head/block kernels maximize grid parallelism, but at
 * long context they also issue the same K/V loads once per head.  This GB10
 * experiment pairs two heads in a 256-thread block: all threads cooperatively
 * stage 16 rows, then two 128-thread teams consume the rows with different Q
 * vectors.  Arithmetic still covers every head and every selected row; only
 * the redundant global loads are removed.
 *
 * The two-pass softmax tape deliberately emulates the generic 256-lane
 * reduction with two virtual lanes per physical thread.  Indexed attention
 * keeps the full model top-k (up to 512); no context rows are discarded. */
template <bool INDEXED, bool COMP_F16>
__global__ static void attention_decode_mixed_heads2_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const void *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    const uint32_t t = blockIdx.x;
    const uint32_t head_slot = threadIdx.x >> 7u;
    const uint32_t local = threadIdx.x & 127u;
    const uint32_t head = blockIdx.y * 2u + head_slot;
    const bool valid_head = t < n_tokens && head < n_head;
    if (t >= n_tokens || head_dim != 512u || blockDim.x != 256u) return;

    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t comp_rows[512];
    __shared__ uint32_t raw_count_s;
    __shared__ uint32_t raw_first_idx_s;
    __shared__ uint32_t comp_count_s;
    __shared__ float kv_shared[
        DS4_CUDA_ATTN_HEADS2_STAGE_ROWS * 512u];
    __shared__ float scores[2u * DS4_CUDA_ATTN_HEADS2_SCORE_CAP];
    __shared__ float partial[2u * 256u];
    __shared__ float max_score[2];
    __shared__ float denom[2];

    const uint32_t qpos = pos0 + t;
    const bool single_all = n_tokens == 1u && ratio == 0u;
    const uint32_t first_raw_pos = single_all ? 0u : pos0 + n_tokens - n_raw;
    uint32_t visible_comp = n_comp;
    if (!single_all && ratio != 0u) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }

    if (threadIdx.x == 0) {
        uint32_t raw_count = 0;
        uint32_t raw_first_idx = 0;
        if (n_raw != 0u) {
            if (single_all) {
                raw_count = n_raw > 256u ? 256u : n_raw;
            } else {
                const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
                if (qpos >= first_raw_pos) {
                    uint32_t lo = first_raw_pos;
                    if (window != 0u && qpos + 1u > window) {
                        const uint32_t wlo = qpos + 1u - window;
                        if (wlo > lo) lo = wlo;
                    }
                    const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                    if (hi >= lo) {
                        raw_first_idx = lo - first_raw_pos;
                        raw_count = hi - lo + 1u;
                        if (raw_count > 256u) raw_count = 256u;
                    }
                }
            }
        }
        raw_count_s = raw_count;
        raw_first_idx_s = raw_first_idx;
        uint32_t comp_count = INDEXED ? top_k : visible_comp;
        if (comp_count > visible_comp) comp_count = visible_comp;
        if (INDEXED && comp_count > 512u) comp_count = 512u;
        const uint32_t room = DS4_CUDA_ATTN_HEADS2_SCORE_CAP - raw_count;
        if (comp_count > room) comp_count = room;
        comp_count_s = comp_count;
    }
    __syncthreads();

    const uint32_t raw_count = raw_count_s;
    const uint32_t comp_count = comp_count_s;
    const uint32_t n_score = raw_count + comp_count;
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx_s + r) % raw_cap;
    }
    if (INDEXED) {
        for (uint32_t i = threadIdx.x; i < comp_count; i += blockDim.x) {
            int32_t c = topk[(uint64_t)t * top_k + i];
            if (c < 0 || (uint32_t)c >= visible_comp) c = 0;
            comp_rows[i] = (uint32_t)c;
        }
    }
    __syncthreads();

    const float *qh = valid_head
        ? q + ((uint64_t)t * n_head + head) * head_dim
        : NULL;
    const float scale = rsqrtf((float)head_dim);
    const uint32_t qlane = local & 7u;
    const uint32_t qgroup = local >> 3u;
    float *score_row = scores + head_slot * DS4_CUDA_ATTN_HEADS2_SCORE_CAP;

    /* QK pass.  Sixteen rows are fetched once and consumed by both heads. */
    for (uint32_t row0 = 0; row0 < n_score;
         row0 += DS4_CUDA_ATTN_HEADS2_STAGE_ROWS) {
        const uint32_t nr = n_score - row0 < DS4_CUDA_ATTN_HEADS2_STAGE_ROWS
            ? n_score - row0
            : DS4_CUDA_ATTN_HEADS2_STAGE_ROWS;
        const uint32_t stage_elems = nr * 512u;
        for (uint32_t off = threadIdx.x; off < stage_elems; off += blockDim.x) {
            const uint32_t rr = off / 512u;
            const uint32_t d = off - rr * 512u;
            const uint32_t sr = row0 + rr;
            if (sr < raw_count) {
                const float *src = raw_kv + (uint64_t)raw_rows[sr] * head_dim;
                kv_shared[off] = src[d];
            } else {
                const uint32_t ci = sr - raw_count;
                const uint32_t cr = INDEXED ? comp_rows[ci] : ci;
                kv_shared[off] = attention_comp_load<COMP_F16>(
                    comp_kv, (uint64_t)cr * head_dim + d);
            }
        }
        __syncthreads();
        if (valid_head && qgroup < nr) {
            float dot = 0.0f;
            const float *kvrow = kv_shared + qgroup * 512u;
            for (uint32_t d = qlane; d < 512u; d += 8u) {
                dot += qh[d] * kvrow[d];
            }
            const uint32_t mask = 0xffu << (threadIdx.x & 24u);
            for (uint32_t off = 4u; off > 0u; off >>= 1u) {
                dot += __shfl_down_sync(mask, dot, off, 8);
            }
            if (qlane == 0) score_row[row0 + qgroup] = dot * scale;
        }
        __syncthreads();
    }

    /* Reproduce the old 256-lane max/sum tree with two virtual lanes per
     * physical thread in each 128-thread head team. */
    const uint32_t pbase = head_slot * 256u;
    float vmax0 = valid_head ? sinks[head] : -INFINITY;
    float vmax1 = vmax0;
    if (valid_head) {
        for (uint32_t i = local; i < n_score; i += 256u) {
            vmax0 = fmaxf(vmax0, score_row[i]);
        }
        for (uint32_t i = local + 128u; i < n_score; i += 256u) {
            vmax1 = fmaxf(vmax1, score_row[i]);
        }
    }
    partial[pbase + local] = vmax0;
    partial[pbase + local + 128u] = vmax1;
    __syncthreads();
    for (uint32_t stride = 128u; stride > 0u; stride >>= 1u) {
        if (local < stride) {
            partial[pbase + local] = fmaxf(partial[pbase + local],
                                             partial[pbase + local + stride]);
        }
        __syncthreads();
    }
    if (local == 0) max_score[head_slot] = partial[pbase];
    __syncthreads();

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    if (valid_head) {
        const float m = max_score[head_slot];
        for (uint32_t i = local; i < n_score; i += 256u) {
            const float p = expf(score_row[i] - m);
            score_row[i] = p;
            sum0 += p;
        }
        for (uint32_t i = local + 128u; i < n_score; i += 256u) {
            const float p = expf(score_row[i] - m);
            score_row[i] = p;
            sum1 += p;
        }
    }
    partial[pbase + local] = sum0;
    partial[pbase + local + 128u] = sum1;
    __syncthreads();
    for (uint32_t stride = 128u; stride > 0u; stride >>= 1u) {
        if (local < stride) {
            partial[pbase + local] += partial[pbase + local + stride];
        }
        __syncthreads();
    }
    if (local == 0) {
        denom[head_slot] = valid_head
            ? partial[pbase] + expf(sinks[head] - max_score[head_slot])
            : 1.0f;
    }
    __syncthreads();

    /* PV pass.  Four output dimensions per physical thread represent the two
     * virtual lanes used by the original 256-thread value accumulation. */
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (uint32_t row0 = 0; row0 < n_score;
         row0 += DS4_CUDA_ATTN_HEADS2_STAGE_ROWS) {
        const uint32_t nr = n_score - row0 < DS4_CUDA_ATTN_HEADS2_STAGE_ROWS
            ? n_score - row0
            : DS4_CUDA_ATTN_HEADS2_STAGE_ROWS;
        const uint32_t stage_elems = nr * 512u;
        for (uint32_t off = threadIdx.x; off < stage_elems; off += blockDim.x) {
            const uint32_t rr = off / 512u;
            const uint32_t d = off - rr * 512u;
            const uint32_t sr = row0 + rr;
            if (sr < raw_count) {
                const float *src = raw_kv + (uint64_t)raw_rows[sr] * head_dim;
                kv_shared[off] = src[d];
            } else {
                const uint32_t ci = sr - raw_count;
                const uint32_t cr = INDEXED ? comp_rows[ci] : ci;
                kv_shared[off] = attention_comp_load<COMP_F16>(
                    comp_kv, (uint64_t)cr * head_dim + d);
            }
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float p = score_row[row0 + rr];
                const float *kvrow = kv_shared + rr * 512u;
                acc0 += kvrow[local] * p;
                acc1 += kvrow[local + 256u] * p;
                acc2 += kvrow[local + 128u] * p;
                acc3 += kvrow[local + 384u] * p;
            }
        }
        __syncthreads();
    }
    if (valid_head) {
        float *out = heads + ((uint64_t)t * n_head + head) * head_dim;
        const float den = denom[head_slot];
        out[local] = acc0 / den;
        out[local + 256u] = acc1 / den;
        out[local + 128u] = acc2 / den;
        out[local + 384u] = acc3 / den;
    }
}

static uint32_t cuda_attn_heads2_min_rows(void) {
    static int initialized;
    static uint32_t min_rows = UINT_MAX;
    if (!initialized) {
        initialized = 1;
        if (getenv("DS4_CUDA_ATTN_HEADS2") != NULL) {
            min_rows = 384u;
            const char *env = getenv("DS4_CUDA_ATTN_HEADS2_MIN_ROWS");
            if (env && env[0]) {
                char *end = NULL;
                unsigned long v = strtoul(env, &end, 10);
                if (end != env) {
                    if (v < 1ul) v = 1ul;
                    if (v > DS4_CUDA_ATTN_HEADS2_SCORE_CAP) {
                        v = DS4_CUDA_ATTN_HEADS2_SCORE_CAP;
                    }
                    min_rows = (uint32_t)v;
                }
            }
            fprintf(stderr,
                    "ds4: CUDA dynamic decode attention heads2 enabled "
                    "(min_rows=%u, score_cap=%u)\n",
                    min_rows,
                    (unsigned)DS4_CUDA_ATTN_HEADS2_SCORE_CAP);
        }
    }
    return min_rows;
}

static int cuda_attn_heads2_use(uint32_t rows,
                                uint32_t n_tokens,
                                uint32_t n_head,
                                uint32_t head_dim) {
    const uint32_t min_rows = cuda_attn_heads2_min_rows();
    return n_tokens == 1u && n_head >= 2u && head_dim == 512u &&
           rows >= min_rows && rows <= DS4_CUDA_ATTN_HEADS2_SCORE_CAP;
}

template <bool COMP_F16>
__global__ static void attention_indexed_mixed_heads8_rb4_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const void *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * 8u + warp;
    const bool valid_head = head < n_head;

    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t comp_rows[512];
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    __shared__ uint32_t comp_count;
    __shared__ float4 kv_shared[4 * 128];
    __shared__ float scores[8 * 768];

    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = n_comp;
    if (ratio != 0) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }

    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        comp_count = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    if (threadIdx.x == 0) {
        for (uint32_t i = 0; i < top_k && comp_count < 512u; i++) {
            int32_t c = topk[(uint64_t)t * top_k + i];
            if (c >= 0 && (uint32_t)c < visible_comp) comp_rows[comp_count++] = (uint32_t)c;
        }
    }
    __syncthreads();

    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            if (sr < raw_count) {
                const float4 *src = (const float4 *)(
                    raw_kv + (uint64_t)raw_rows[sr] * head_dim);
                kv_shared[off] = src[c4];
            } else {
                kv_shared[off] = attention_comp_load4<COMP_F16>(
                    comp_kv,
                    (uint64_t)comp_rows[sr - raw_count] * head_dim +
                        (uint64_t)c4 * 4u);
            }
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float dot = dot4_f32(q0, kv4[lane +  0u]) +
                            dot4_f32(q1, kv4[lane + 32u]) +
                            dot4_f32(q2, kv4[lane + 64u]) +
                            dot4_f32(q3, kv4[lane + 96u]);
                dot = warp_sum_f32(dot);
                if (lane == 0) scores[warp * 768u + row0 + rr] = dot * scale;
            }
        }
        __syncthreads();
    }

    float max_s = valid_head ? sinks[head] : -INFINITY;
    if (valid_head) {
        const float *score_row = scores + warp * 768u;
        for (uint32_t i = lane; i < n_score; i += 32u) max_s = fmaxf(max_s, score_row[i]);
        max_s = warp_max_f32(max_s);
        max_s = __shfl_sync(0xffffffffu, max_s, 0);
    }
    float den = 0.0f;
    if (valid_head) {
        float *score_row = scores + warp * 768u;
        for (uint32_t i = lane; i < n_score; i += 32u) {
            float p = expf(score_row[i] - max_s);
            score_row[i] = p;
            den += p;
        }
        den = warp_sum_f32(den);
        den += expf(sinks[head] - max_s);
        den = __shfl_sync(0xffffffffu, den, 0);
    }

    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;
    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            if (sr < raw_count) {
                const float4 *src = (const float4 *)(
                    raw_kv + (uint64_t)raw_rows[sr] * head_dim);
                kv_shared[off] = src[c4];
            } else {
                kv_shared[off] = attention_comp_load4<COMP_F16>(
                    comp_kv,
                    (uint64_t)comp_rows[sr - raw_count] * head_dim +
                        (uint64_t)c4 * 4u);
            }
        }
        __syncthreads();
        if (valid_head) {
            const float *score_row = scores + warp * 768u;
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float p = den == 0.0f ? 0.0f : score_row[row0 + rr] / den;
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                o0.x += k0.x * p; o0.y += k0.y * p; o0.z += k0.z * p; o0.w += k0.w * p;
                o1.x += k1.x * p; o1.y += k1.y * p; o1.z += k1.z * p; o1.w += k1.w * p;
                o2.x += k2.x * p; o2.y += k2.y * p; o2.z += k2.z * p; o2.w += k2.w * p;
                o3.x += k3.x * p; o3.y += k3.y * p; o3.z += k3.z * p; o3.w += k3.w * p;
            }
        }
        __syncthreads();
    }
    if (valid_head) {
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

template <uint32_t ROWS_PER_STAGE, uint32_t HEADS_PER_GROUP, bool COMP_F16>
__global__ static void attention_indexed_mixed_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const void *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * HEADS_PER_GROUP + warp;
    const bool valid_head = head < n_head;

    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    __shared__ float4 kv_shared[ROWS_PER_STAGE * 128];

    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = n_comp;
    if (ratio != 0) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }

    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    __syncthreads();

    uint32_t comp_count = top_k < visible_comp ? top_k : visible_comp;
    if (comp_count > 512u) comp_count = 512u;
    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    float max_s = -INFINITY;
    float sum_s = 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t row0 = 0; row0 < n_score; row0 += ROWS_PER_STAGE) {
        const uint32_t nr = n_score - row0 < ROWS_PER_STAGE ? n_score - row0 : ROWS_PER_STAGE;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const uint32_t comp_idx = sr < raw_count
                ? 0u
                : (uint32_t)topk[(uint64_t)t * top_k + (sr - raw_count)];
            if (sr < raw_count) {
                const float4 *src = (const float4 *)(
                    raw_kv + (uint64_t)raw_rows[sr] * head_dim);
                kv_shared[off] = src[c4];
            } else {
                kv_shared[off] = attention_comp_load4<COMP_F16>(
                    comp_kv,
                    (uint64_t)comp_idx * head_dim + (uint64_t)c4 * 4u);
            }
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) +
                              dot4_f32(q1, k1) +
                              dot4_f32(q2, k2) +
                              dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(0xffffffffu, score, 0);

                const float new_m = fmaxf(max_s, score);
                const float old_scale = new_m == max_s ? 1.0f : expf(max_s - new_m);
                const float row_scale = score == new_m ? 1.0f : expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + k0.x * row_scale;
                o0.y = o0.y * old_scale + k0.y * row_scale;
                o0.z = o0.z * old_scale + k0.z * row_scale;
                o0.w = o0.w * old_scale + k0.w * row_scale;
                o1.x = o1.x * old_scale + k1.x * row_scale;
                o1.y = o1.y * old_scale + k1.y * row_scale;
                o1.z = o1.z * old_scale + k1.z * row_scale;
                o1.w = o1.w * old_scale + k1.w * row_scale;
                o2.x = o2.x * old_scale + k2.x * row_scale;
                o2.y = o2.y * old_scale + k2.y * row_scale;
                o2.z = o2.z * old_scale + k2.z * row_scale;
                o2.w = o2.w * old_scale + k2.w * row_scale;
                o3.x = o3.x * old_scale + k3.x * row_scale;
                o3.y = o3.y * old_scale + k3.y * row_scale;
                o3.z = o3.z * old_scale + k3.z * row_scale;
                o3.w = o3.w * old_scale + k3.w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }

    if (valid_head) {
        const float sink = sinks[head];
        const float new_m = fmaxf(max_s, sink);
        const float old_scale = new_m == max_s ? 1.0f : expf(max_s - new_m);
        const float sink_scale = expf(sink - new_m);
        sum_s = sum_s * old_scale + sink_scale;
        o0.x *= old_scale; o0.y *= old_scale; o0.z *= old_scale; o0.w *= old_scale;
        o1.x *= old_scale; o1.y *= old_scale; o1.z *= old_scale; o1.w *= old_scale;
        o2.x *= old_scale; o2.y *= old_scale; o2.z *= old_scale; o2.w *= old_scale;
        o3.x *= old_scale; o3.y *= old_scale; o3.z *= old_scale; o3.w *= old_scale;

        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

template <bool COMP_F16>
__global__ static void attention_static_mixed_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const void *comp_kv,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * 8u + warp;
    const bool valid_head = head < n_head;

    __shared__ float4 kv_shared[4 * 128];

    const uint32_t raw_count = window != 0u && t + 1u > window ? window : t + 1u;
    const uint32_t raw_start = t + 1u - raw_count;
    uint32_t comp_count = 0;
    if (n_comp != 0u && ratio != 0u) {
        comp_count = (t + 1u) / ratio;
        if (comp_count > n_comp) comp_count = n_comp;
    }
    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    float max_s = -INFINITY;
    float sum_s = 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            if (sr < raw_count) {
                const float4 *src = (const float4 *)(
                    raw_kv + (uint64_t)(raw_start + sr) * head_dim);
                kv_shared[off] = src[c4];
            } else {
                kv_shared[off] = attention_comp_load4<COMP_F16>(
                    comp_kv,
                    (uint64_t)(sr - raw_count) * head_dim +
                        (uint64_t)c4 * 4u);
            }
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) +
                              dot4_f32(q1, k1) +
                              dot4_f32(q2, k2) +
                              dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(0xffffffffu, score, 0);

                const float new_m = fmaxf(max_s, score);
                const float old_scale = new_m == max_s ? 1.0f : expf(max_s - new_m);
                const float row_scale = score == new_m ? 1.0f : expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + k0.x * row_scale;
                o0.y = o0.y * old_scale + k0.y * row_scale;
                o0.z = o0.z * old_scale + k0.z * row_scale;
                o0.w = o0.w * old_scale + k0.w * row_scale;
                o1.x = o1.x * old_scale + k1.x * row_scale;
                o1.y = o1.y * old_scale + k1.y * row_scale;
                o1.z = o1.z * old_scale + k1.z * row_scale;
                o1.w = o1.w * old_scale + k1.w * row_scale;
                o2.x = o2.x * old_scale + k2.x * row_scale;
                o2.y = o2.y * old_scale + k2.y * row_scale;
                o2.z = o2.z * old_scale + k2.z * row_scale;
                o2.w = o2.w * old_scale + k2.w * row_scale;
                o3.x = o3.x * old_scale + k3.x * row_scale;
                o3.y = o3.y * old_scale + k3.y * row_scale;
                o3.z = o3.z * old_scale + k3.z * row_scale;
                o3.w = o3.w * old_scale + k3.w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }

    if (valid_head) {
        const float sink = sinks[head];
        const float new_m = fmaxf(max_s, sink);
        const float old_scale = new_m == max_s ? 1.0f : expf(max_s - new_m);
        const float sink_scale = expf(sink - new_m);
        sum_s = sum_s * old_scale + sink_scale;
        o0.x *= old_scale; o0.y *= old_scale; o0.z *= old_scale; o0.w *= old_scale;
        o1.x *= old_scale; o1.y *= old_scale; o1.z *= old_scale; o1.w *= old_scale;
        o2.x *= old_scale; o2.y *= old_scale; o2.z *= old_scale; o2.w *= old_scale;
        o3.x *= old_scale; o3.y *= old_scale; o3.z *= old_scale; o3.w *= old_scale;

        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

/* -------------------------------------------------------------------------
 * Token-tile HMMA attention for wide single-session prefill batches.
 * STAGE_ROWS stays 32; dense uses T16/G2 and indexed sparse uses T1/G32.
 * Decode and verifier shapes are ineligible.
 * Adapted from Entrpi/ds4 commits 47438d7 and 9de3044 (MIT).
 * Sparse head-major mapping follows deepseek-ai/FlashMLA commit 9241ae3 (MIT).
 */

static constexpr uint32_t kTTTileTokens = 16u;
static constexpr uint32_t kTTG = 2u;
static constexpr uint32_t kTTM = 32u;
/* FlashMLA sparse prefill keeps one query token in a CTA and amortizes each
 * selected KV row across many heads.  M remains 32, so this mapping reuses
 * the established SM121 warp-MMA fragments and shared-memory layout. */
static constexpr uint32_t kTTSparseTileTokens = 1u;
static constexpr uint32_t kTTSparseG = 32u;
static constexpr uint32_t kTTStageRows = 32u;
static constexpr uint32_t kTTRawWindow = 128u;
static constexpr uint32_t kTTHeadDim = 512u;
static constexpr uint32_t kTTWarps = 16u;
static constexpr uint32_t kTTThreads = 512u;
static constexpr uint32_t kTTScoreKQuarters = 4u;
static constexpr uint32_t kTTScoreKSliceDim = kTTHeadDim / kTTScoreKQuarters;
static constexpr uint32_t kTTScoreKStepsPerQuarter = kTTScoreKSliceDim / 16u;
static constexpr uint32_t kTTRecordRingPlanes = 4u;
static constexpr uint32_t kTTProbStride = 40u;
static constexpr uint32_t kTTRingChunkBytes = 16u;
static constexpr uint32_t kTTRingChunksPerRow =
    (kTTHeadDim * sizeof(half)) / kTTRingChunkBytes;
static constexpr size_t kTTSmemHardCap = 90ull * 1024ull;

static_assert(kTTM == kTTTileTokens * kTTG, "token-tile M must be 16 tokens x G2");
static_assert(kTTM == kTTSparseTileTokens * kTTSparseG,
              "sparse FlashMLA mapping must be one token x 32 heads");
static_assert(kTTProbStride == kTTStageRows + 8u, "token-tile prob stride changed");
static_assert(kTTScoreKSliceDim % 16u == 0, "token-tile score K split changed");
static_assert(kTTRingChunksPerRow == 64u, "token-tile KV ring expects 64 chunks");
static_assert(sizeof(int2) == 8u, "token-tile union record must remain 8 bytes");

template <uint32_t TT_STAGE_ROWS>
struct tt_TokentileLayout {
    static constexpr uint32_t prob_stride = TT_STAGE_ROWS + 8u;
    static constexpr uint32_t ring_plane_bytes =
        TT_STAGE_ROWS * kTTRingChunksPerRow * kTTRingChunkBytes;
    static constexpr uint32_t ring_plane_elems = ring_plane_bytes / sizeof(half);
};

__device__ static float tt_warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ static float tt_warp_max_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ __forceinline__ uint32_t tt_lane_id(void) {
    return threadIdx.x & 31u;
}

__device__ __forceinline__ uint32_t tt_warp_id(void) {
    return threadIdx.x >> 5u;
}

__device__ __forceinline__ int tt_mma_c_i(uint32_t lane, int l) {
    return ((l >> 1) << 3) + (int)(lane >> 2);
}

__device__ __forceinline__ int tt_mma_c_j(uint32_t lane, int l) {
    return (int)((lane & 3u) << 1) + (l & 1);
}

__device__ __forceinline__ unsigned tt_smem_addr(const void *p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ uint32_t tt_ring_off_bytes(uint32_t row, uint32_t c) {
    return (row * kTTRingChunksPerRow + (c ^ (row & 7u))) * kTTRingChunkBytes;
}

__device__ __forceinline__ void tt_ldmatrix_x4_addr(uint32_t (&r)[4], unsigned a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
#else
    (void)a;
    r[0] = r[1] = r[2] = r[3] = 0;
#endif
}

__device__ __forceinline__ void tt_ldmatrix_x2_addr(uint32_t (&r)[2], unsigned a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(a));
#else
    (void)a;
    r[0] = r[1] = 0;
#endif
}

__device__ __forceinline__ void tt_ldmatrix_x2_trans_addr(uint32_t (&r)[2], unsigned a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.b16 {%0, %1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(a));
#else
    (void)a;
    r[0] = r[1] = 0;
#endif
}

__device__ __forceinline__ void tt_ldmatrix_x4(uint32_t (&r)[4], const void *p) {
    tt_ldmatrix_x4_addr(r, tt_smem_addr(p));
}

__device__ __forceinline__ void tt_ldmatrix_x2(uint32_t (&r)[2], const void *p) {
    tt_ldmatrix_x2_addr(r, tt_smem_addr(p));
}

__device__ __forceinline__ void tt_ldmatrix_x2_trans(uint32_t (&r)[2], const void *p) {
    tt_ldmatrix_x2_trans_addr(r, tt_smem_addr(p));
}

__device__ __forceinline__ void tt_mma_m16n8k16_f16_f32(
        float *d,
        const uint32_t (&a)[4],
        const uint32_t (&b)[2]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint32_t d0 = __float_as_uint(d[0]);
    uint32_t d1 = __float_as_uint(d[1]);
    uint32_t d2 = __float_as_uint(d[2]);
    uint32_t d3 = __float_as_uint(d[3]);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    d[0] = __uint_as_float(d0);
    d[1] = __uint_as_float(d1);
    d[2] = __uint_as_float(d2);
    d[3] = __uint_as_float(d3);
#else
    (void)a;
    (void)b;
#endif
}

__device__ __forceinline__ unsigned char *tt_align16(unsigned char *p) {
    uintptr_t x = reinterpret_cast<uintptr_t>(p);
    x = (x + 15u) & ~uintptr_t(15u);
    return reinterpret_cast<unsigned char *>(x);
}

__device__ __forceinline__ void tt_zero_16B(void *dst) {
    *reinterpret_cast<int4 *>(dst) = make_int4(0, 0, 0, 0);
}

__device__ __forceinline__ void tt_zero_8B(void *dst) {
    *reinterpret_cast<int2 *>(dst) = make_int2(0, 0);
}

__device__ __forceinline__ void tt_cp_async_16B(void *dst, const void *src, bool pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (pred) {
        const unsigned smem = tt_smem_addr(dst);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                     :: "r"(smem), "l"(src));
    } else {
        tt_zero_16B(dst);
    }
#else
    (void)src;
    (void)pred;
    tt_zero_16B(dst);
#endif
}

__device__ __forceinline__ void tt_cp_async_8B(void *dst, const void *src, bool pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (pred) {
        const unsigned smem = tt_smem_addr(dst);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;"
                     :: "r"(smem), "l"(src));
    } else {
        tt_zero_8B(dst);
    }
#else
    (void)src;
    (void)pred;
    tt_zero_8B(dst);
#endif
}

__device__ __forceinline__ void tt_cp_async_commit(void) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
#endif
}

template <int KeepGroups>
__device__ __forceinline__ void tt_cp_async_wait_group(void) {
    static_assert(KeepGroups >= 0 && KeepGroups <= 7, "bad cp.async wait_group depth");
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;" :: "n"(KeepGroups));
#endif
}

__device__ __forceinline__ uint32_t tt_score_partial_slot(
        uint32_t kq,
        uint32_t m,
        uint32_t r) {
    return (kq + r + m) & 3u;
}

template <uint32_t TT_STAGE_ROWS>
__device__ __forceinline__ void tt_store_score_partial(
        float4 * __restrict__ partials,
        uint32_t kq,
        uint32_t m,
        uint32_t r,
        float v) {
    float *dst = &partials[m * TT_STAGE_ROWS + r].x;
    dst[tt_score_partial_slot(kq, m, r)] = v;
}

__device__ __forceinline__ float4 tt_load_score_partial_record(
        const float4 * __restrict__ p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    float x, y, z, w;
    asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(x), "=f"(y), "=f"(z), "=f"(w)
                 : "r"(tt_smem_addr(p)));
    return make_float4(x, y, z, w);
#else
    return *p;
#endif
}

__device__ __forceinline__ void tt_issue_cp_async_row(
        half * __restrict__ dst,
        uint32_t rr,
        uint32_t lane16,
        const half * __restrict__ src,
        bool live) {
    const char *src_b = reinterpret_cast<const char *>(src);
    char *dst_b = reinterpret_cast<char *>(dst);

#pragma unroll
    for (uint32_t i = 0; i < 4u; ++i) {
        const uint32_t chunk = lane16 + i * 16u;
        char *db = dst_b + tt_ring_off_bytes(rr, chunk);
        tt_cp_async_16B(db, live ? static_cast<const void *>(src_b + chunk * 16u)
                                 : static_cast<const void *>(db),
                        live);
    }
}

__device__ __forceinline__ uint32_t tt_stage_raw_rows(
        uint32_t row0,
        uint32_t nr,
        uint32_t raw_union_count) {
    uint32_t raw_rows = 0;
    if (row0 < raw_union_count) {
        const uint32_t raw_left = raw_union_count - row0;
        raw_rows = raw_left < nr ? raw_left : nr;
    }
    return raw_rows;
}

__global__ static void __launch_bounds__(256, 1) attention_tokentile_raw_mirror_kernel(
        half *dst,
        const float *raw_kv,
        const int32_t *seq_id,
        uint32_t tt_run_pos0,
        uint32_t n_tokens,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t first_raw_pos,
        uint32_t raw_row_min,
        uint32_t head_dim) {
    const uint32_t row = blockIdx.x;
    const uint32_t d0 = threadIdx.x << 1u;
    if (d0 >= head_dim) return;
    half *dst_row = dst + (uint64_t)row * head_dim;
    if (row < raw_row_min) {
        dst_row[d0] = __float2half(0.0f);
        if (d0 + 1u < head_dim) dst_row[d0 + 1u] = __float2half(0.0f);
        return;
    }

    const int64_t p = (int64_t)tt_run_pos0 - (int64_t)(kTTRawWindow - 1u) + (int64_t)row;
    uint32_t slot = 0u;
    if (seq_id) {
        slot = (uint32_t)seq_id[0] * raw_cap + (uint32_t)((uint64_t)p % raw_cap);
    } else {
        const uint32_t rel = (uint32_t)(p - (int64_t)first_raw_pos);
        slot = (raw_start + rel) % raw_cap;
    }
    const float *src = raw_kv + (uint64_t)slot * head_dim;
    dst_row[d0] = __float2half(src[d0]);
    if (d0 + 1u < head_dim) dst_row[d0 + 1u] = __float2half(src[d0 + 1u]);
}

__global__ static void __launch_bounds__(256, 1) attention_tokentile_comp_mirror_kernel(
        half *dst,
        const float *comp_kv,
        uint32_t n_comp,
        uint32_t head_dim) {
    const uint32_t row = blockIdx.x;
    const uint32_t c4 = threadIdx.x;
    if (row >= n_comp || c4 >= (head_dim >> 2u)) return;
    const float4 v = ((const float4 *)(comp_kv + (uint64_t)row * head_dim))[c4];
    half *out = dst + (uint64_t)row * head_dim + (c4 << 2u);
    out[0] = __float2half(v.x);
    out[1] = __float2half(v.y);
    out[2] = __float2half(v.z);
    out[3] = __float2half(v.w);
}

/* Decode-mixed port: the non-indexed layers see comp rows as the causal
 * RANGE [0, visible(tok)) -- floor(qpos/ratio) per-seq, floor((qpos+1)/ratio)
 * single-seq, clamped to n_comp -- not a topk set, so the per-tile union is
 * [0, max visible) and the per-token masks are arithmetic.  Emits the same
 * ascending-id interleaved int2 {comp_id, mask} records the hmma kernel's
 * record ring consumes; no bitmap, no scan. */
__global__ static void __launch_bounds__(512, 1) attention_tokentile_dense_build_kernel(
        int2 *records,
        uint32_t *counts,
        const int32_t *positions,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t ratio,
        uint32_t n_comp,
        uint32_t rec_stride) {
    __shared__ uint32_t visible_s[kTTTileTokens];
    const uint32_t tid = threadIdx.x;
    const uint32_t tile_base = blockIdx.x * kTTTileTokens;
    if (tile_base >= n_tokens) {
        if (tid == 0u) counts[blockIdx.x] = 0u;
        return;
    }
    const uint32_t tile_count =
        n_tokens - tile_base < kTTTileTokens ? n_tokens - tile_base : kTTTileTokens;
    if (tid < kTTTileTokens) {
        uint32_t visible = 0u;
        if (tid < tile_count && n_comp != 0u && ratio != 0u) {
            const uint32_t t = tile_base + tid;
            const uint32_t qpos = positions ? (uint32_t)positions[t] : pos0 + t;
            visible = positions ? qpos / ratio : (qpos + 1u) / ratio;
            if (visible > n_comp) visible = n_comp;
        }
        visible_s[tid] = visible;
    }
    __syncthreads();
    uint32_t vmax = 0u;
    for (uint32_t k = 0u; k < tile_count; k++) {
        if (visible_s[k] > vmax) vmax = visible_s[k];
    }
    for (uint32_t c = tid; c < vmax; c += blockDim.x) {
        uint32_t mask = 0u;
        for (uint32_t k = 0u; k < tile_count; k++) {
            if (c < visible_s[k]) mask |= 1u << k;
        }
        records[(uint64_t)blockIdx.x * rec_stride + c] = make_int2((int)c, (int)mask);
    }
    if (tid == 0u) counts[blockIdx.x] = vmax;
}

template <uint32_t TT_STAGE_ROWS>
__device__ __forceinline__ void tt_issue_record_stage_cp_async(
        int2 * __restrict__ rec_plane,
        uint32_t row0,
        uint32_t nr,
        uint32_t raw_union_count,
        const int2 * __restrict__ union_records_tile) {
    static_assert(TT_STAGE_ROWS == 32u, "token-tile record issue is fixed at R32");
    const uint32_t raw_rows = tt_stage_raw_rows(row0, nr, raw_union_count);
    const uint32_t lane = tt_lane_id();
    const uint32_t warp = tt_warp_id();
    const uint32_t lane16 = lane & 15u;
    const uint32_t rr = warp * 2u + (lane >> 4u);
    const bool active = rr < TT_STAGE_ROWS;
    const bool comp_live = active && rr >= raw_rows && rr < nr;
    if (comp_live && lane16 == 0u) {
        int2 *dst = rec_plane + rr;
        const uint32_t ci = row0 + rr - raw_union_count;
        tt_cp_async_8B(dst, union_records_tile + ci, true);
    }
}

/* FlashMLA-style direct sparse metadata stage.  Only one lane writes each
 * record; the following CTA synchronization publishes it before the KV stage
 * consumes the record.  Invalid indices remain in the stage with mask zero,
 * preserving exact sparse semantics without a global compacted copy. */
template <uint32_t TT_STAGE_ROWS>
__device__ __forceinline__ void tt_issue_sparse_record_stage(
        int2 * __restrict__ rec_plane,
        uint32_t row0,
        uint32_t nr,
        uint32_t raw_union_count,
        const int32_t * __restrict__ sparse_indices,
        uint32_t sparse_visible) {
    static_assert(TT_STAGE_ROWS == 32u, "sparse record stage is fixed at R32");
    const uint32_t raw_rows = tt_stage_raw_rows(row0, nr, raw_union_count);
    const uint32_t lane = tt_lane_id();
    const uint32_t warp = tt_warp_id();
    const uint32_t lane16 = lane & 15u;
    const uint32_t rr = warp * 2u + (lane >> 4u);
    const bool comp_slot = rr < nr && rr >= raw_rows;
    if (comp_slot && lane16 == 0u) {
        const uint32_t ci = row0 + rr - raw_union_count;
        const int32_t c = __ldg(sparse_indices + ci);
        const int valid = c >= 0 && (uint32_t)c < sparse_visible;
        rec_plane[rr] = make_int2(valid ? c : 0, valid);
    }
}

template <uint32_t TT_STAGE_ROWS, bool USE_SMEM_RECORDS>
__device__ __forceinline__ void tt_issue_kv_stage_cp_async(
        half * __restrict__ dst,
        const int2 * __restrict__ rec_plane,
        uint32_t row0,
        uint32_t nr,
        uint32_t raw_union_count,
        const int2 * __restrict__ union_records_tile,
        uint32_t tile_base,
        const half * __restrict__ raw_kv,
        const half * __restrict__ comp_kv,
        uint32_t tid) {
    constexpr uint32_t kCp16PerRow = (kTTHeadDim * sizeof(half)) / 16u;
    static_assert(kCp16PerRow == 64u, "expected 64 cp.async chunks per f16 KV row");
    static_assert(TT_STAGE_ROWS == 32u, "token-tile KV issue is fixed at R32");
    (void)tid;
    const uint32_t raw_rows = tt_stage_raw_rows(row0, nr, raw_union_count);
    const uint32_t lane = tt_lane_id();
    const uint32_t warp = tt_warp_id();
    const uint32_t lane16 = lane & 15u;
    const uint32_t rr = warp * 2u + (lane >> 4u);
    const bool active = rr < TT_STAGE_ROWS;

    if (active && rr < raw_rows) {
        const uint32_t sr = row0 + rr;
        const half *src = raw_kv + (uint64_t)(tile_base + sr) * kTTHeadDim;
        tt_issue_cp_async_row(dst, rr, lane16, src, true);
    }

    if (active && rr >= raw_rows && rr < nr) {
        uint32_t comp_id = 0u;
        bool comp_live = true;
        if (USE_SMEM_RECORDS) {
            const int2 rec = rec_plane[rr];
            comp_id = (uint32_t)rec.x;
            comp_live = rec.y != 0;
        } else {
            const uint32_t ci = row0 + rr - raw_union_count;
            comp_id = (uint32_t)union_records_tile[ci].x;
        }
        const half *src = comp_live
            ? comp_kv + (uint64_t)comp_id * kTTHeadDim
            : NULL;
        tt_issue_cp_async_row(dst, rr, lane16, src, comp_live);
    }

    if (active && rr >= nr) {
        tt_issue_cp_async_row(dst, rr, lane16, NULL, false);
    }
}

template <uint32_t TT_TILE_TOKENS, uint32_t TT_G>
__device__ __forceinline__ void tt_load_score_q_frag(
        uint32_t (&q_frag)[kTTScoreKStepsPerQuarter][4],
        const float * __restrict__ q,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t tile_base,
        uint32_t head_base) {
    static_assert(TT_TILE_TOKENS * TT_G == kTTM,
                  "token/head factorization must preserve M32");
    constexpr uint32_t kMtiles = kTTM / 16u;
    constexpr uint32_t kScoreWarps = kMtiles * kTTScoreKQuarters;
    const uint32_t warp = tt_warp_id();
    if (warp >= kScoreWarps) {
        return;
    }

    const uint32_t mtile = warp >> 2u;
    const uint32_t kq = warp & 3u;
    const uint32_t k_base = kq * kTTScoreKSliceDim;
    const uint32_t lane = tt_lane_id();
    const uint32_t a_group = lane >> 2u;
    const uint32_t a_col_pair = (lane & 3u) << 1u;
#pragma unroll
    for (uint32_t kt = 0; kt < kTTScoreKStepsPerQuarter; ++kt) {
        const uint32_t k0 = k_base + kt * 16u;
#pragma unroll
        for (uint32_t r = 0; r < 4u; ++r) {
            const uint32_t m = mtile * 16u + a_group + ((r & 1u) ? 8u : 0u);
            const uint32_t tok = m / TT_G;
            const uint32_t h = m - tok * TT_G;
            const uint32_t gt = tile_base + tok;
            const uint32_t gh = head_base + h;
            const uint32_t d = k0 + a_col_pair + ((r & 2u) ? 8u : 0u);
            float x0 = 0.0f;
            float x1 = 0.0f;
            if (gt < n_tokens && gh < n_head) {
                const float *q_row = q + ((uint64_t)gt * n_head + gh) * kTTHeadDim;
                x0 = q_row[d];
                x1 = q_row[d + 1u];
            }
            const half2 packed = __floats2half2_rn(x0, x1);
            q_frag[kt][r] =
                (uint32_t)__half_as_ushort(__low2half(packed)) |
                ((uint32_t)__half_as_ushort(__high2half(packed)) << 16);
        }
    }
}

template <uint32_t TT_STAGE_ROWS>
__device__ __forceinline__ void tt_hmma_score_stage(
        float4 * __restrict__ partial_scores,
        const uint32_t (&q_frag)[kTTScoreKStepsPerQuarter][4],
        const half * __restrict__ kv_cur,
        uint32_t nr,
        float score_scale) {
    constexpr uint32_t kMtiles = kTTM / 16u;
    constexpr uint32_t kScoreWarps = kMtiles * kTTScoreKQuarters;
    constexpr uint32_t kNtiles = TT_STAGE_ROWS / 8u;
    const uint32_t warp = tt_warp_id();
    if (warp >= kScoreWarps) {
        return;
    }

    const uint32_t mtile = warp >> 2u;
    const uint32_t kq = warp & 3u;
    const uint32_t lane = tt_lane_id();
    const unsigned kv_smem = tt_smem_addr(kv_cur);
    const uint32_t score_row_lane = lane & 7u;
    const uint32_t score_chunk_lane = (lane >> 3u) & 1u;
    const uint32_t score_chunk_base = kq * (kTTScoreKSliceDim / 8u) + score_chunk_lane;
#pragma unroll
    for (uint32_t ntile = 0; ntile < kNtiles; ++ntile) {
        const uint32_t row_base = ntile * 8u;
        if (row_base < nr) {
            float s_frag[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const uint32_t score_row = row_base + score_row_lane;
#pragma unroll
            for (uint32_t kt = 0; kt < kTTScoreKStepsPerQuarter; ++kt) {
                uint32_t b[2];
                tt_ldmatrix_x2_addr(
                    b,
                    kv_smem + tt_ring_off_bytes(score_row, score_chunk_base + kt * 2u));
                tt_mma_m16n8k16_f16_f32(s_frag, q_frag[kt], b);
            }
            const uint32_t m0 = mtile * 16u + (lane >> 2u);
            const uint32_t r0 = row_base + ((lane & 3u) << 1u);
            const uint32_t r1 = r0 + 1u;
            const uint32_t m1 = m0 + 8u;
            if (r0 < nr) {
                tt_store_score_partial<TT_STAGE_ROWS>(
                    partial_scores, kq, m0, r0, s_frag[0] * score_scale);
                tt_store_score_partial<TT_STAGE_ROWS>(
                    partial_scores, kq, m1, r0, s_frag[2] * score_scale);
            }
            if (r1 < nr) {
                tt_store_score_partial<TT_STAGE_ROWS>(
                    partial_scores, kq, m0, r1, s_frag[1] * score_scale);
                tt_store_score_partial<TT_STAGE_ROWS>(
                    partial_scores, kq, m1, r1, s_frag[3] * score_scale);
            }
        }
    }
}

template <uint32_t TT_STAGE_ROWS, uint32_t TT_G>
__device__ __forceinline__ void tt_softmax_stage(
        half * __restrict__ probs,
        float * __restrict__ stage_rescale,
        float * __restrict__ max_s,
        float * __restrict__ sum_s,
        const float4 * __restrict__ scores,
        const int2 * __restrict__ records,
        uint32_t row0,
        uint32_t nr,
        uint32_t raw_union_count,
        uint32_t tile_count,
        uint32_t tile_base,
        uint32_t raw_row_min) {
    constexpr uint32_t kMPerWarp = kTTM / kTTWarps;
    constexpr uint32_t kProbStride = tt_TokentileLayout<TT_STAGE_ROWS>::prob_stride;
    static_assert(kTTM % kTTWarps == 0u, "softmax maps integral m rows per warp");
    const uint32_t lane = tt_lane_id();
    const uint32_t warp = tt_warp_id();
    const uint32_t sr = row0 + lane;
    const bool lane_live = lane < TT_STAGE_ROWS && lane < nr;
    const bool raw_slot = lane_live && sr < raw_union_count;
    uint16_t comp_mask = 0u;
    if (lane_live && !raw_slot) {
        comp_mask = (uint16_t)records[lane].y;
    }
    const uint32_t prob_lane_base = warp * kProbStride + lane;

#pragma unroll
    for (uint32_t mi = 0; mi < kMPerWarp; ++mi) {
        const uint32_t m = warp + mi * kTTWarps;
        const uint32_t tok = m / TT_G;
        const bool valid_token = tok < tile_count;
        const uint32_t score_idx = m * TT_STAGE_ROWS + lane;
        const uint32_t prob_idx = prob_lane_base + mi * kTTWarps * kProbStride;
        float score = -INFINITY;
        if (lane_live && valid_token) {
            const bool selected = raw_slot
                ? (((uint32_t)(sr - tok) < kTTRawWindow) && (tile_base + sr >= raw_row_min))
                : ((comp_mask & (uint16_t)(1u << tok)) != 0u);
            if (selected) {
                const float4 parts = tt_load_score_partial_record(scores + score_idx);
                const float s01 = parts.x + parts.y;
                const float s23 = parts.z + parts.w;
                score = s01 + s23;
            }
        }

        const float stage_m = tt_warp_max_f32(score);
        const float old_m = max_s[m];
        const float new_m = fmaxf(old_m, stage_m);
        float old_scale = 1.0f;
        if (new_m != -INFINITY) {
            old_scale = old_m == -INFINITY
                ? 0.0f
                : (old_m == new_m ? 1.0f : expf(old_m - new_m));
        }
        const float row_scale = (score == -INFINITY || new_m == -INFINITY)
            ? 0.0f
            : (score == new_m ? 1.0f : expf(score - new_m));
        const float stage_sum = tt_warp_sum_f32(row_scale);

        if (lane < TT_STAGE_ROWS) {
            probs[prob_idx] = __float2half(lane < nr ? row_scale : 0.0f);
        }
        if (lane == 0u) {
            max_s[m] = new_m;
            sum_s[m] = sum_s[m] * old_scale + stage_sum;
            stage_rescale[m] = old_scale;
        }
    }
}

template <uint32_t TT_STAGE_ROWS>
__device__ __forceinline__ void tt_pv_mma_stage(
        float (&o_acc)[2u * kTTTileTokens * kTTG],
        const half * __restrict__ probs,
        const float * __restrict__ stage_rescale,
        const half * __restrict__ kv_cur) {
    constexpr uint32_t kMtiles = kTTM / 16u;
    constexpr uint32_t kPvWarpBase = 8u;
    constexpr uint32_t kPvNTiles = 8u;
    const uint32_t lane = tt_lane_id();
    const uint32_t warp = tt_warp_id();
    if (warp < kPvWarpBase) {
        return;
    }
    const uint32_t pv_warp = warp - kPvWarpBase;

#pragma unroll
    for (uint32_t mtile = 0; mtile < kMtiles; ++mtile) {
        const float *scale0 = stage_rescale + mtile * 16u + (lane >> 2u);
        const float *scale1 = scale0 + 8u;
        const float rs0 = *scale0;
        const float rs1 = *scale1;
#pragma unroll
        for (uint32_t ntile = 0; ntile < kPvNTiles; ++ntile) {
            const uint32_t idx = ((mtile * kPvNTiles + ntile) << 2);
            if (rs0 != 1.0f) {
                o_acc[idx + 0u] *= rs0;
                o_acc[idx + 1u] *= rs0;
            }
            if (rs1 != 1.0f) {
                o_acc[idx + 2u] *= rs1;
                o_acc[idx + 3u] *= rs1;
            }
        }
    }

    constexpr uint32_t kProbStride = tt_TokentileLayout<TT_STAGE_ROWS>::prob_stride;
    constexpr unsigned kPvAStepBytes = 16u * sizeof(half);
    constexpr unsigned kPvMtileBytes = 16u * kProbStride * sizeof(half);
    const unsigned probs_lane_base =
        tt_smem_addr(probs) +
        (unsigned)(((lane & 15u) * (kProbStride / 2u) +
                    (lane >> 4u) * 4u) * sizeof(uint32_t));
    const unsigned kv_smem = tt_smem_addr(kv_cur);
    const uint32_t pv_row_lane = lane & 15u;
    const uint32_t pv_chunk_base = pv_warp * kPvNTiles;
#pragma unroll
    for (uint32_t kt = 0; kt < TT_STAGE_ROWS / 16u; ++kt) {
        const unsigned probs_kt_base = probs_lane_base + (unsigned)(kt * kPvAStepBytes);
        const uint32_t pv_row = kt * 16u + pv_row_lane;
#pragma unroll
        for (uint32_t mtile = 0; mtile < kMtiles; ++mtile) {
            uint32_t a[4];
            tt_ldmatrix_x4_addr(a, probs_kt_base + (unsigned)(mtile * kPvMtileBytes));
#pragma unroll
            for (uint32_t ntile = 0; ntile < kPvNTiles; ++ntile) {
                uint32_t b[2];
                const uint32_t idx = ((mtile * kPvNTiles + ntile) << 2);
                tt_ldmatrix_x2_trans_addr(
                    b,
                    kv_smem + tt_ring_off_bytes(pv_row, pv_chunk_base + ntile));
                tt_mma_m16n8k16_f16_f32(o_acc + idx, a, b);
            }
        }
    }
}

template <uint32_t TT_G>
__device__ __forceinline__ void tt_pv_mma_epilogue(
        const float (&o_acc)[2u * kTTTileTokens * kTTG],
        const float * __restrict__ final_scale,
        float * __restrict__ heads,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t tile_base,
        uint32_t head_base) {
    constexpr uint32_t kMtiles = kTTM / 16u;
    constexpr uint32_t kPvWarpBase = 8u;
    constexpr uint32_t kPvNTiles = 8u;
    const uint32_t lane = tt_lane_id();
    const uint32_t warp = tt_warp_id();
    if (warp < kPvWarpBase) {
        return;
    }
    const uint32_t pv_warp = warp - kPvWarpBase;

#pragma unroll
    for (uint32_t mtile = 0; mtile < kMtiles; ++mtile) {
#pragma unroll
        for (uint32_t ntile = 0; ntile < kPvNTiles; ++ntile) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const uint32_t idx = ((mtile * kPvNTiles + ntile) << 2) + (uint32_t)l;
                const uint32_t m = mtile * 16u + (uint32_t)tt_mma_c_i(lane, l);
                const uint32_t tok = m / TT_G;
                const uint32_t h = m - tok * TT_G;
                const uint32_t gt = tile_base + tok;
                const uint32_t gh = head_base + h;
                const uint32_t d =
                    pv_warp * 64u + ntile * 8u + (uint32_t)tt_mma_c_j(lane, l);
                if (gt < n_tokens && gh < n_head) {
                    heads[((uint64_t)gt * n_head + gh) * kTTHeadDim + d] =
                        o_acc[idx] * final_scale[m];
                }
            }
        }
    }
}

template <uint32_t TT_TILE_TOKENS, uint32_t TT_G, bool TT_DIRECT_SPARSE>
__global__ static void __launch_bounds__(512, 1) attention_tokentile_hmma_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const half *raw_kv,
        const half *comp_kv,
        const int2 *union_records,
        const uint32_t *union_counts,
        const int32_t *sparse_indices,
        uint32_t sparse_pos0,
        uint32_t sparse_ratio,
        uint32_t sparse_n_comp,
        uint32_t sparse_topk,
        uint32_t rec_stride,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t raw_row_min) {
    static_assert(TT_TILE_TOKENS * TT_G == kTTM,
                  "token/head factorization must preserve M32");
    static_assert(!TT_DIRECT_SPARSE || TT_TILE_TOKENS == 1u,
                  "direct sparse mapping owns exactly one query token");
    static_assert(!TT_DIRECT_SPARSE || kTTRawWindow % kTTStageRows == 0u,
                  "direct sparse records must begin on a stage boundary");
    constexpr uint32_t kKvElems = tt_TokentileLayout<kTTStageRows>::ring_plane_elems;
    constexpr uint32_t kProbStride = tt_TokentileLayout<kTTStageRows>::prob_stride;
    const uint32_t tid = threadIdx.x;
    const uint32_t tile_idx = TT_DIRECT_SPARSE ? blockIdx.y : blockIdx.x;
    const uint32_t head_group = TT_DIRECT_SPARSE ? blockIdx.x : blockIdx.y;
    const uint32_t tile_base = tile_idx * TT_TILE_TOKENS;
    const uint32_t head_base = head_group * TT_G;
    if (tile_base >= n_tokens) {
        return;
    }
    const uint32_t tile_count =
        n_tokens - tile_base < TT_TILE_TOKENS ? n_tokens - tile_base : TT_TILE_TOKENS;
    const uint32_t raw_union_count = tile_count + kTTRawWindow - 1u;
    uint32_t sparse_visible = sparse_n_comp;
    if constexpr (TT_DIRECT_SPARSE) {
        sparse_visible = sparse_ratio
            ? (sparse_pos0 + tile_base + 1u) / sparse_ratio
            : sparse_n_comp;
        if (sparse_visible > sparse_n_comp) sparse_visible = sparse_n_comp;
    }
    const uint32_t comp_union_count = TT_DIRECT_SPARSE
        ? (sparse_topk < sparse_visible ? sparse_topk : sparse_visible)
        : union_counts[tile_idx];
    const uint32_t n_score = raw_union_count + comp_union_count;
    const uint64_t union_tile_off = (uint64_t)tile_idx * rec_stride;
    const int2 * __restrict__ union_records_tile = TT_DIRECT_SPARSE
        ? NULL : union_records + union_tile_off;
    const int32_t * __restrict__ sparse_indices_tile = TT_DIRECT_SPARSE
        ? sparse_indices + union_tile_off : NULL;
    const float score_scale = rsqrtf((float)kTTHeadDim);

    extern __shared__ unsigned char smem[];
    unsigned char *p = tt_align16(smem);
    half *kv_h = reinterpret_cast<half *>(p);
    p = tt_align16(p + 2u * tt_TokentileLayout<kTTStageRows>::ring_plane_bytes);
    half *probs = reinterpret_cast<half *>(p);
    p = tt_align16(p + 2u * kTTM * kProbStride * sizeof(half));
    float4 *score_scratch = reinterpret_cast<float4 *>(p);
    p = tt_align16(p + kTTM * kTTStageRows * sizeof(float4));
    float *max_s = reinterpret_cast<float *>(p);
    p = tt_align16(p + kTTM * sizeof(float));
    float *sum_s = reinterpret_cast<float *>(p);
    p = tt_align16(p + kTTM * sizeof(float));
    float *stage_rescale = reinterpret_cast<float *>(p);
    p = tt_align16(p + kTTM * sizeof(float));
    float *final_scale = reinterpret_cast<float *>(p);
    p = tt_align16(p + kTTM * sizeof(float));
    int2 *rec_ring = reinterpret_cast<int2 *>(p);
    p = tt_align16(p + kTTRecordRingPlanes * kTTStageRows * sizeof(int2));

    for (uint32_t m = tid; m < kTTM; m += blockDim.x) {
        max_s[m] = -INFINITY;
        sum_s[m] = 0.0f;
        stage_rescale[m] = 1.0f;
        final_scale[m] = 0.0f;
    }
    __syncthreads();

    union tt_TokentileRoleRegs {
        uint32_t score_q_frag[kTTScoreKStepsPerQuarter][4];
        float o_acc[2u * kTTTileTokens * kTTG];
    };
    tt_TokentileRoleRegs role_regs;
    tt_load_score_q_frag<TT_TILE_TOKENS, TT_G>(
        role_regs.score_q_frag, q, n_tokens, n_head, tile_base, head_base);
    if (tt_warp_id() >= 8u) {
#pragma unroll
        for (uint32_t i = 0; i < 2u * kTTM; ++i) {
            role_regs.o_acc[i] = 0.0f;
        }
    }

    uint32_t cur = 0u;
    uint32_t free = 1u;
    if (n_score != 0u) {
        const uint32_t nr0 = n_score < kTTStageRows ? n_score : kTTStageRows;
        if constexpr (TT_DIRECT_SPARSE) {
            tt_issue_kv_stage_cp_async<kTTStageRows, true>(
                kv_h + cur * kKvElems,
                rec_ring,
                0u,
                nr0,
                raw_union_count,
                NULL,
                tile_base,
                raw_kv,
                comp_kv,
                tid);
            tt_issue_sparse_record_stage<kTTStageRows>(
                rec_ring,
                0u,
                nr0,
                raw_union_count,
                sparse_indices_tile,
                sparse_visible);
        } else {
            tt_issue_kv_stage_cp_async<kTTStageRows, false>(
                kv_h + cur * kKvElems,
                rec_ring,
                0u,
                nr0,
                raw_union_count,
                union_records_tile,
                tile_base,
                raw_kv,
                comp_kv,
                tid);
            tt_issue_record_stage_cp_async<kTTStageRows>(
                rec_ring,
                0u,
                nr0,
                raw_union_count,
                union_records_tile);
        }
        if (kTTStageRows < n_score) {
            const uint32_t nr1 =
                n_score - kTTStageRows < kTTStageRows ? n_score - kTTStageRows : kTTStageRows;
            if constexpr (TT_DIRECT_SPARSE) {
                tt_issue_sparse_record_stage<kTTStageRows>(
                    rec_ring + kTTStageRows,
                    kTTStageRows,
                    nr1,
                    raw_union_count,
                    sparse_indices_tile,
                    sparse_visible);
            } else {
                tt_issue_record_stage_cp_async<kTTStageRows>(
                    rec_ring + kTTStageRows,
                    kTTStageRows,
                    nr1,
                    raw_union_count,
                    union_records_tile);
            }
        }
        tt_cp_async_commit();
        tt_cp_async_wait_group<0>();
    }
    __syncthreads();

    uint32_t prob_cur = 0u;
    for (uint32_t row0 = 0u; row0 < n_score; row0 += kTTStageRows) {
        const uint32_t nr = n_score - row0 < kTTStageRows ? n_score - row0 : kTTStageRows;
        half *kv_cur = kv_h + cur * kKvElems;
        half *kv_free = kv_h + free * kKvElems;

        tt_hmma_score_stage<kTTStageRows>(
            score_scratch, role_regs.score_q_frag, kv_cur, nr, score_scale);
        if (row0 != 0u) {
            const half *kv_prev = kv_h + (cur ^ 1u) * kKvElems;
            const half *probs_prev = probs + (prob_cur ^ 1u) * kTTM * kProbStride;
            tt_pv_mma_stage<kTTStageRows>(
                role_regs.o_acc, probs_prev, stage_rescale, kv_prev);
        }
        __syncthreads();

        const uint32_t next_row0 = row0 + kTTStageRows;
        const bool has_next = next_row0 < n_score;
        if (has_next) {
            const uint32_t next_nr =
                n_score - next_row0 < kTTStageRows ? n_score - next_row0 : kTTStageRows;
            tt_issue_kv_stage_cp_async<kTTStageRows, true>(
                kv_free,
                rec_ring + (((row0 / kTTStageRows) + 1u) & 3u) * kTTStageRows,
                next_row0,
                next_nr,
                raw_union_count,
                TT_DIRECT_SPARSE ? NULL : union_records_tile,
                tile_base,
                raw_kv,
                comp_kv,
                tid);
            const uint32_t prefetch_row0 = next_row0 + kTTStageRows;
            if (prefetch_row0 < n_score) {
                const uint32_t prefetch_nr =
                    n_score - prefetch_row0 < kTTStageRows
                        ? n_score - prefetch_row0
                        : kTTStageRows;
                if constexpr (TT_DIRECT_SPARSE) {
                    tt_issue_sparse_record_stage<kTTStageRows>(
                        rec_ring + (((row0 / kTTStageRows) + 2u) & 3u) * kTTStageRows,
                        prefetch_row0,
                        prefetch_nr,
                        raw_union_count,
                        sparse_indices_tile,
                        sparse_visible);
                } else {
                    tt_issue_record_stage_cp_async<kTTStageRows>(
                        rec_ring + (((row0 / kTTStageRows) + 2u) & 3u) * kTTStageRows,
                        prefetch_row0,
                        prefetch_nr,
                        raw_union_count,
                        union_records_tile);
                }
            }
            tt_cp_async_commit();
        }

        half *probs_cur = probs + prob_cur * kTTM * kProbStride;
        tt_softmax_stage<kTTStageRows, TT_G>(
            probs_cur,
            stage_rescale,
            max_s,
            sum_s,
            score_scratch,
            rec_ring + ((row0 / kTTStageRows) & 3u) * kTTStageRows,
            row0,
            nr,
            raw_union_count,
            tile_count,
            tile_base,
            raw_row_min);
        tt_cp_async_wait_group<0>();
        __syncthreads();
        cur ^= 1u;
        free ^= 1u;
        prob_cur ^= 1u;
    }

    if (n_score != 0u) {
        const half *kv_prev = kv_h + (cur ^ 1u) * kKvElems;
        const half *probs_prev = probs + (prob_cur ^ 1u) * kTTM * kProbStride;
        tt_pv_mma_stage<kTTStageRows>(
            role_regs.o_acc, probs_prev, stage_rescale, kv_prev);
    }
    __syncthreads();

    for (uint32_t m = tid; m < kTTM; m += blockDim.x) {
        const uint32_t tok = m / TT_G;
        const uint32_t h = m - tok * TT_G;
        const uint32_t gt = tile_base + tok;
        const uint32_t gh = head_base + h;
        if (gt < n_tokens && gh < n_head) {
            const float sink = sinks[gh];
            const float old_m = max_s[m];
            const float new_m = fmaxf(old_m, sink);
            const float old_scale = old_m == -INFINITY
                ? 0.0f
                : (old_m == new_m ? 1.0f : expf(old_m - new_m));
            const float sink_scale = expf(sink - new_m);
            const float den = sum_s[m] * old_scale + sink_scale;
            final_scale[m] = den == 0.0f ? 0.0f : old_scale / den;
        } else {
            final_scale[m] = 0.0f;
        }
    }
    __syncthreads();

    tt_pv_mma_epilogue<TT_G>(
        role_regs.o_acc, final_scale, heads, n_tokens, n_head, tile_base, head_base);
}

constexpr size_t tt_align16_const(size_t x) {
    return (x + 15u) & ~size_t(15u);
}

template <uint32_t TT_STAGE_ROWS, uint32_t TT_TILE_TOKENS, uint32_t TT_G>
struct tt_TokentileSmemBudget {
    static constexpr uint32_t M = TT_TILE_TOKENS * TT_G;
    static constexpr uint32_t prob_stride = tt_TokentileLayout<TT_STAGE_ROWS>::prob_stride;
    static constexpr size_t q_bytes = 0;
    static constexpr size_t ring_bytes = 2ull * tt_TokentileLayout<TT_STAGE_ROWS>::ring_plane_bytes;
    static constexpr size_t p_bytes = 2ull * M * prob_stride * sizeof(half);
    static constexpr size_t partial_records = (size_t)M * TT_STAGE_ROWS;
    static constexpr size_t partial_bytes = partial_records * sizeof(float4);
    static constexpr size_t stats_bytes = 4ull * M * sizeof(float);
    static constexpr size_t record_bytes =
        (size_t)kTTRecordRingPlanes * TT_STAGE_ROWS * sizeof(int2);
    static constexpr size_t total =
        tt_align16_const(
        tt_align16_const(
        tt_align16_const(
        tt_align16_const(
        tt_align16_const(
        tt_align16_const(q_bytes) + ring_bytes) + p_bytes) +
        partial_bytes) + stats_bytes) + record_bytes);
};

static_assert(tt_TokentileSmemBudget<32, 16, 2>::p_bytes == 5120ull,
              "M32/R32 P double-buffer must use R+8 stride");
static_assert(sizeof(float4) == 16u, "score partial records must stay 16 bytes");
static_assert(tt_TokentileSmemBudget<32, 16, 2>::partial_bytes == 16ull * 1024ull,
              "M32/R32 score partial records must be M*R float4");
static_assert(tt_TokentileSmemBudget<32, 16, 2>::record_bytes == 1024ull,
              "M32/R32 record ring must be four R-row int2 planes");
static_assert(tt_TokentileSmemBudget<32, 16, 2>::total == 88576ull,
              "M32/R32 total dynamic shared memory changed unexpectedly");
static_assert(tt_TokentileSmemBudget<kTTStageRows,
                                    kTTTileTokens,
                                    kTTG>::total <= kTTSmemHardCap,
              "token-tile dynamic shared memory must stay under the 90 KiB pass gate");
static_assert(tt_TokentileSmemBudget<kTTStageRows,
                                    kTTSparseTileTokens,
                                    kTTSparseG>::total ==
                  tt_TokentileSmemBudget<kTTStageRows,
                                         kTTTileTokens,
                                         kTTG>::total,
              "sparse head-major mapping must not increase shared memory");
static_assert(2ull * tt_TokentileLayout<64>::ring_plane_bytes > kTTSmemHardCap,
              "a double-buffered 64-row KV ring cannot fit in GB10 shared memory");

static int cuda_attention_tokentile_dense_launch(
        ds4_gpu_tensor       *heads,
        const float          *sinks,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t              comp_kv_f16,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              n_raw,
        uint32_t              raw_cap,
        uint32_t              raw_start,
        uint32_t              n_comp,
        uint32_t              window,
        uint32_t              ratio,
        uint32_t              n_head,
        uint32_t              head_dim);

template <bool COMP_F16>
__global__ static void attention_decode_mixed_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const void *comp_kv,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * 8u + warp;
    const bool valid_head = head < n_head;

    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t raw_count_s;
    __shared__ uint32_t raw_first_idx_s;
    __shared__ float4 kv_shared[4 * 128];

    const uint32_t qpos = pos0 + t;
    const uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t comp_count = 0;
    if (n_comp != 0u) {
        if (n_tokens == 1u && ratio == 0u) {
            comp_count = n_comp;
        } else if (ratio != 0u) {
            comp_count = (qpos + 1u) / ratio;
            if (comp_count > n_comp) comp_count = n_comp;
        }
    }
    if (threadIdx.x == 0) {
        uint32_t raw_count = 0;
        uint32_t raw_first_idx = 0;
        if (n_raw != 0u) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0u && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
        raw_count_s = raw_count;
        raw_first_idx_s = raw_first_idx;
    }
    __syncthreads();
    const uint32_t raw_count = raw_count_s;
    const uint32_t raw_first_idx = raw_first_idx_s;
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    __syncthreads();

    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    float max_s = -INFINITY;
    float sum_s = 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            if (sr < raw_count) {
                const float4 *src = (const float4 *)(
                    raw_kv + (uint64_t)raw_rows[sr] * head_dim);
                kv_shared[off] = src[c4];
            } else {
                kv_shared[off] = attention_comp_load4<COMP_F16>(
                    comp_kv,
                    (uint64_t)(sr - raw_count) * head_dim +
                        (uint64_t)c4 * 4u);
            }
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) +
                              dot4_f32(q1, k1) +
                              dot4_f32(q2, k2) +
                              dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(0xffffffffu, score, 0);

                const float new_m = fmaxf(max_s, score);
                const float old_scale = new_m == max_s ? 1.0f : expf(max_s - new_m);
                const float row_scale = score == new_m ? 1.0f : expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + k0.x * row_scale;
                o0.y = o0.y * old_scale + k0.y * row_scale;
                o0.z = o0.z * old_scale + k0.z * row_scale;
                o0.w = o0.w * old_scale + k0.w * row_scale;
                o1.x = o1.x * old_scale + k1.x * row_scale;
                o1.y = o1.y * old_scale + k1.y * row_scale;
                o1.z = o1.z * old_scale + k1.z * row_scale;
                o1.w = o1.w * old_scale + k1.w * row_scale;
                o2.x = o2.x * old_scale + k2.x * row_scale;
                o2.y = o2.y * old_scale + k2.y * row_scale;
                o2.z = o2.z * old_scale + k2.z * row_scale;
                o2.w = o2.w * old_scale + k2.w * row_scale;
                o3.x = o3.x * old_scale + k3.x * row_scale;
                o3.y = o3.y * old_scale + k3.y * row_scale;
                o3.z = o3.z * old_scale + k3.z * row_scale;
                o3.w = o3.w * old_scale + k3.w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }

    if (valid_head) {
        const float sink = sinks[head];
        const float new_m = fmaxf(max_s, sink);
        const float old_scale = new_m == max_s ? 1.0f : expf(max_s - new_m);
        const float sink_scale = expf(sink - new_m);
        sum_s = sum_s * old_scale + sink_scale;
        o0.x *= old_scale; o0.y *= old_scale; o0.z *= old_scale; o0.w *= old_scale;
        o1.x *= old_scale; o1.y *= old_scale; o1.z *= old_scale; o1.w *= old_scale;
        o2.x *= old_scale; o2.y *= old_scale; o2.z *= old_scale; o2.w *= old_scale;
        o3.x *= old_scale; o3.y *= old_scale; o3.z *= old_scale; o3.w *= old_scale;

        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

__device__ static void hc4_split_one(float *out, const float *mix, const float *scale, const float *base, uint32_t sinkhorn_iters, float epsv) {
    const float pre_scale = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    for (int i = 0; i < 4; i++) {
        float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + epsv;
    }
    for (int i = 0; i < 4; i++) {
        float z = mix[4 + i] * post_scale + base[4 + i];
        out[4 + i] = 2.0f / (1.0f + expf(-z));
    }
    float c[16];
    for (int r = 0; r < 4; r++) {
        float m = -INFINITY;
        for (int col = 0; col < 4; col++) {
            float v = mix[8 + r * 4 + col] * comb_scale + base[8 + r * 4 + col];
            c[r * 4 + col] = v;
            m = fmaxf(m, v);
        }
        float s = 0.0f;
        for (int col = 0; col < 4; col++) {
            float v = expf(c[r * 4 + col] - m);
            c[r * 4 + col] = v;
            s += v;
        }
        for (int col = 0; col < 4; col++) c[r * 4 + col] = c[r * 4 + col] / s + epsv;
    }
    for (int col = 0; col < 4; col++) {
        float s = epsv;
        for (int r = 0; r < 4; r++) s += c[r * 4 + col];
        for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
    }
    for (uint32_t iter = 1; iter < sinkhorn_iters; iter++) {
        for (int r = 0; r < 4; r++) {
            float s = epsv;
            for (int col = 0; col < 4; col++) s += c[r * 4 + col];
            for (int col = 0; col < 4; col++) c[r * 4 + col] /= s;
        }
        for (int col = 0; col < 4; col++) {
            float s = epsv;
            for (int r = 0; r < 4; r++) s += c[r * 4 + col];
            for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
        }
    }
    for (int i = 0; i < 16; i++) out[8 + i] = c[i];
}

__global__ static void hc_split_sinkhorn_kernel(float *out, const float *mix, const float *scale, const float *base, uint32_t n_rows, uint32_t sinkhorn_iters, float epsv) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    hc4_split_one(out + (uint64_t)row * 24, mix + (uint64_t)row * 24, scale, base, sinkhorn_iters, epsv);
}

__global__ static void hc_weighted_sum_kernel(float *out, const float *x, const float *w, uint32_t n_embd, uint32_t n_hc, uint32_t n_tokens, uint32_t weight_stride_f32) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_embd * n_tokens;
    if (gid >= n) return;
    uint32_t d = gid % n_embd;
    uint32_t t = gid / n_embd;
    float acc = 0.0f;
    for (uint32_t h = 0; h < n_hc; h++) {
        acc += x[(uint64_t)t * n_hc * n_embd + (uint64_t)h * n_embd + d] *
               w[(uint64_t)t * weight_stride_f32 + h];
    }
    out[(uint64_t)t * n_embd + d] = acc;
}

__global__ static void hc_expand_kernel(
        float *out_hc,
        const float *block_out,
        const float *block_add,
        const float *residual_hc,
        const float *post,
        const float *comb,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_tokens,
        uint32_t post_stride,
        uint32_t comb_stride,
        int has_add) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n_elem) return;
    uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    uint32_t dst_hc = tmp % n_hc;
    uint32_t t = tmp / n_hc;

    float block_v = block_out[(uint64_t)t * n_embd + d];
    if (has_add) block_v += block_add[(uint64_t)t * n_embd + d];
    float acc = block_v * post[(uint64_t)t * post_stride + dst_hc];
    for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
        float comb_v = comb[(uint64_t)t * comb_stride + dst_hc + (uint64_t)src_hc * n_hc];
        float res_v = residual_hc[(uint64_t)t * n_hc * n_embd + (uint64_t)src_hc * n_embd + d];
        acc += comb_v * res_v;
    }
    out_hc[(uint64_t)t * n_hc * n_embd + (uint64_t)dst_hc * n_embd + d] = acc;
}

/* Wide prefill consumes an expanded HC row immediately through one F16
 * projection.  A token-owned CTA reuses the four residual values and the
 * split coefficients across all four destinations, then performs the same
 * 256-lane RMS reduction used by rms_norm_plain_kernel and emits F16
 * directly.  The persistent HC state remains FP32. */
__global__ static void hc_expand_norm_f16_kernel(
        float *out_hc,
        __half *norm_h,
        const float *block_out,
        const float *block_add,
        const float *residual_hc,
        const float *split,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_tokens,
        int has_add,
        float eps) {
    const uint32_t t = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (t >= n_tokens || n_hc != 4u) return;

    __shared__ float coeff[20];
    __shared__ float partial[256];
    const float *sp = split + (uint64_t)t * 24u;
    if (lane < 20u) coeff[lane] = sp[4u + lane];
    __syncthreads();

    float *dst = out_hc + (uint64_t)t * 4u * n_embd;
    const float *res = residual_hc + (uint64_t)t * 4u * n_embd;
    const float *bo = block_out + (uint64_t)t * n_embd;
    const float *ba = has_add ? block_add + (uint64_t)t * n_embd : NULL;
    for (uint32_t d = lane; d < n_embd; d += blockDim.x) {
        float block_v = bo[d];
        if (has_add) block_v += ba[d];
        const float r0 = res[d];
        const float r1 = res[n_embd + d];
        const float r2 = res[2u * n_embd + d];
        const float r3 = res[3u * n_embd + d];
#pragma unroll
        for (uint32_t h = 0; h < 4u; ++h) {
            float acc = block_v * coeff[h];
            acc += coeff[4u + h] * r0;
            acc += coeff[8u + h] * r1;
            acc += coeff[12u + h] * r2;
            acc += coeff[16u + h] * r3;
            dst[(uint64_t)h * n_embd + d] = acc;
        }
    }
    __syncthreads();

    const uint32_t width = 4u * n_embd;
    float sum = 0.0f;
    for (uint32_t i = lane; i < width; i += blockDim.x) {
        const float v = dst[i];
        sum += v * v;
    }
    partial[lane] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    const float norm_scale = rsqrtf(partial[0] / (float)width + eps);
    __half *norm_row = norm_h + (uint64_t)t * width;
    for (uint32_t i = lane; i < width; i += blockDim.x) {
        norm_row[i] = __float2half_rn(dst[i] * norm_scale);
    }
}

__global__ static void moe_down_hc_expand_norm_f16_kernel(
        float *out_hc,
        __half *norm_h,
        const float *routed_down,
        const float *block_add,
        const float *residual_hc,
        const float *split,
        uint32_t n_embd,
        uint32_t n_expert,
        uint32_t n_tokens,
        float eps) {
    const uint32_t t = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (t >= n_tokens || n_expert != 6u) return;

    __shared__ float coeff[20];
    __shared__ float partial[256];
    const float *sp = split + (uint64_t)t * 24u;
    if (lane < 20u) coeff[lane] = sp[4u + lane];
    __syncthreads();

    float *dst = out_hc + (uint64_t)t * 4u * n_embd;
    const float *res = residual_hc + (uint64_t)t * 4u * n_embd;
    const float *shared = block_add + (uint64_t)t * n_embd;
    const float *down = routed_down + (uint64_t)t * n_expert * n_embd;
    for (uint32_t d = lane; d < n_embd; d += blockDim.x) {
        float routed = 0.0f;
#pragma unroll
        for (uint32_t e = 0; e < 6u; ++e) {
            const float v = down[(uint64_t)e * n_embd + d];
            if (isfinite(v)) routed += v;
        }
        const float block_v = routed + shared[d];
        const float r0 = res[d];
        const float r1 = res[n_embd + d];
        const float r2 = res[2u * n_embd + d];
        const float r3 = res[3u * n_embd + d];
#pragma unroll
        for (uint32_t h = 0; h < 4u; ++h) {
            float acc = block_v * coeff[h];
            acc += coeff[4u + h] * r0;
            acc += coeff[8u + h] * r1;
            acc += coeff[12u + h] * r2;
            acc += coeff[16u + h] * r3;
            dst[(uint64_t)h * n_embd + d] = acc;
        }
    }
    __syncthreads();

    const uint32_t width = 4u * n_embd;
    float sum = 0.0f;
    for (uint32_t i = lane; i < width; i += blockDim.x) {
        const float v = dst[i];
        sum += v * v;
    }
    partial[lane] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    const float norm_scale = rsqrtf(partial[0] / (float)width + eps);
    __half *norm_row = norm_h + (uint64_t)t * width;
    for (uint32_t i = lane; i < width; i += blockDim.x) {
        norm_row[i] = __float2half_rn(dst[i] * norm_scale);
    }
}

__global__ static void hc_split_weighted_sum_fused_kernel(
        float *out,
        float *split,
        const float *mix,
        const float *residual_hc,
        const float *scale,
        const float *base,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_rows,
        uint32_t sinkhorn_iters,
        float epsv) {
    uint32_t t = blockIdx.x;
    uint32_t d = threadIdx.x;
    if (t >= n_rows || n_hc != 4) return;
    const uint32_t mix_hc = 24;
    float *sp = split + (uint64_t)t * mix_hc;
    if (d == 0) hc4_split_one(sp, mix + (uint64_t)t * mix_hc, scale, base, sinkhorn_iters, epsv);
    __syncthreads();
    for (uint32_t col = d; col < n_embd; col += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < 4; h++) {
            acc += residual_hc[(uint64_t)t * 4u * n_embd + (uint64_t)h * n_embd + col] * sp[h];
        }
        out[(uint64_t)t * n_embd + col] = acc;
    }
}

__global__ static void hc_split_weighted_sum_norm_fused_kernel(
        float *out,
        float *norm_out,
        float *split,
        const float *mix,
        const float *residual_hc,
        const float *scale,
        const float *base,
        const float *norm_w,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_rows,
        uint32_t sinkhorn_iters,
        float epsv,
        float norm_eps) {
    const uint32_t t = blockIdx.x;
    const uint32_t d = threadIdx.x;
    if (t >= n_rows || n_hc != 4) return;
    const uint32_t mix_hc = 24;
    float *sp = split + (uint64_t)t * mix_hc;
    if (d == 0) hc4_split_one(sp, mix + (uint64_t)t * mix_hc, scale, base, sinkhorn_iters, epsv);
    __syncthreads();

    float sum = 0.0f;
    for (uint32_t col = d; col < n_embd; col += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < 4; h++) {
            acc += residual_hc[(uint64_t)t * 4u * n_embd + (uint64_t)h * n_embd + col] * sp[h];
        }
        out[(uint64_t)t * n_embd + col] = acc;
        sum += acc * acc;
    }

    __shared__ float partial[256];
    partial[d] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (d < stride) partial[d] += partial[d + stride];
        __syncthreads();
    }
    const float norm_scale = rsqrtf(partial[0] / (float)n_embd + norm_eps);
    for (uint32_t col = d; col < n_embd; col += blockDim.x) {
        const float v = out[(uint64_t)t * n_embd + col];
        norm_out[(uint64_t)t * n_embd + col] = v * norm_scale * norm_w[col];
    }
}

__global__ static void output_hc_weights_kernel(
        float *out,
        const float *pre,
        const float *scale,
        const float *base,
        uint32_t n_hc,
        uint32_t n_tokens,
        float epsv) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = n_tokens * n_hc;
    if (gid >= n) return;
    uint32_t h = gid % n_hc;
    float z = pre[gid] * scale[0] + base[h];
    out[gid] = 1.0f / (1.0f + expf(-z)) + epsv;
}

__global__ static void fill_f32_kernel(float *x, uint64_t n, float v) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v;
}

__global__ static void compressor_store_kernel(
        const float *kv,
        const float *sc,
        float *state_kv,
        float *state_score,
        const void *model_map,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint32_t head_dim,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t n_tokens) {
    uint32_t coff = ratio == 4u ? 2u : 1u;
    uint32_t width = coff * head_dim;
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * width;
    if (gid >= n) return;
    uint32_t t = gid / width;
    uint32_t j = gid - (uint64_t)t * width;
    uint32_t pos_mod = (pos0 + t) % ratio;
    uint32_t dst_row = ratio == 4u ? ratio + pos_mod : pos_mod;
    state_kv[(uint64_t)dst_row * width + j] = kv[(uint64_t)t * width + j];
    state_score[(uint64_t)dst_row * width + j] =
        sc[(uint64_t)t * width + j] + model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)pos_mod * width + j);
}

__global__ static void compressor_set_rows_kernel(
        float *state_kv,
        float *state_score,
        const float *kv,
        const float *sc,
        const void *model_map,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint32_t width,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t src0,
        uint32_t dst0,
        uint32_t rows) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)rows * width;
    if (gid >= n) return;
    uint32_t r = gid / width;
    uint32_t j = gid - (uint64_t)r * width;
    uint32_t src = src0 + r;
    uint32_t dst = dst0 + r;
    uint32_t phase = (pos0 + src) % ratio;
    state_kv[(uint64_t)dst * width + j] = kv[(uint64_t)src * width + j];
    state_score[(uint64_t)dst * width + j] =
        sc[(uint64_t)src * width + j] + model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)phase * width + j);
}

__global__ static void compressor_prefill_pool_kernel(
        float *comp,
        const float *kv,
        const float *sc,
        const float *state_kv,
        const float *state_score,
        const void *model_map,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint32_t head_dim,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t n_comp,
        uint32_t replay) {
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t c = blockIdx.y;
    if (d >= head_dim || c >= n_comp) return;
    uint32_t coff = ratio == 4u ? 2u : 1u;
    uint32_t width = coff * head_dim;
    float vals[128];
    float scores[128];
    float max_s = -INFINITY;
    uint32_t n_cand = 0;
    if (ratio == 4u) {
        if (replay && c == 0) {
            for (uint32_t r = 0; r < 4; r++) {
                vals[n_cand] = state_kv[(uint64_t)r * width + d];
                scores[n_cand] = state_score[(uint64_t)r * width + d];
                max_s = fmaxf(max_s, scores[n_cand++]);
            }
        } else if (c > 0) {
            uint32_t base = (c - 1u) * ratio;
            for (uint32_t r = 0; r < 4; r++) {
                uint32_t t = base + r;
                float ape = model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)((pos0 + t) % ratio) * width + d);
                vals[n_cand] = kv[(uint64_t)t * width + d];
                scores[n_cand] = sc[(uint64_t)t * width + d] + ape;
                max_s = fmaxf(max_s, scores[n_cand++]);
            }
        }
        uint32_t base = c * ratio;
        for (uint32_t r = 0; r < 4; r++) {
            uint32_t t = base + r;
            float ape = model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)((pos0 + t) % ratio) * width + head_dim + d);
            vals[n_cand] = kv[(uint64_t)t * width + head_dim + d];
            scores[n_cand] = sc[(uint64_t)t * width + head_dim + d] + ape;
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    } else {
        uint32_t base = c * ratio;
        for (uint32_t r = 0; r < ratio; r++) {
            uint32_t t = base + r;
            float ape = model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)((pos0 + t) % ratio) * width + d);
            vals[n_cand] = kv[(uint64_t)t * width + d];
            scores[n_cand] = sc[(uint64_t)t * width + d] + ape;
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    }
    float den = 0.0f, acc = 0.0f;
    for (uint32_t i = 0; i < n_cand; i++) {
        float w = expf(scores[i] - max_s);
        den += w;
        acc += vals[i] * w;
    }
    comp[(uint64_t)c * head_dim + d] = den != 0.0f ? acc / den : 0.0f;
}

__global__ static void compressor_update_pool_kernel(
        float *row,
        const float *state_kv,
        const float *state_score,
        uint32_t head_dim,
        uint32_t ratio) {
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= head_dim) return;
    uint32_t coff = ratio == 4u ? 2u : 1u;
    uint32_t width = coff * head_dim;
    float vals[128];
    float scores[128];
    float max_s = -INFINITY;
    uint32_t n_cand = 0;
    if (ratio == 4u) {
        for (uint32_t r = 0; r < 4; r++) {
            vals[n_cand] = state_kv[(uint64_t)r * width + d];
            scores[n_cand] = state_score[(uint64_t)r * width + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
        for (uint32_t r = 0; r < 4; r++) {
            vals[n_cand] = state_kv[(uint64_t)(ratio + r) * width + head_dim + d];
            scores[n_cand] = state_score[(uint64_t)(ratio + r) * width + head_dim + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    } else {
        for (uint32_t r = 0; r < ratio; r++) {
            vals[n_cand] = state_kv[(uint64_t)r * width + d];
            scores[n_cand] = state_score[(uint64_t)r * width + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    }
    float den = 0.0f, acc = 0.0f;
    for (uint32_t i = 0; i < n_cand; i++) {
        float w = expf(scores[i] - max_s);
        den += w;
        acc += vals[i] * w;
    }
    row[d] = den != 0.0f ? acc / den : 0.0f;
}

__global__ static void compressor_shift_ratio4_kernel(float *state_kv, float *state_score, uint32_t width) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t half = 4ull * width;
    if (i >= half) return;
    float v = state_kv[half + i];
    float s = state_score[half + i];
    state_kv[i] = v;
    state_score[i] = s;
    state_kv[half + i] = v;
    state_score[half + i] = s;
}

/* One-token, ratio-4 compressor fast path.  The generic path needs a store
 * launch for every token and, on an emitting token, separate pool, RMS norm,
 * RoPE and state-shift launches.  All of those operations cover at most a few
 * KiB and have a strict producer/consumer dependency, so keeping them in one
 * CTA removes launch latency and intermediate scheduling without changing the
 * model math. */
__global__ static void compressor_update_ratio4_decode_kernel(
        const float *kv,
        const float *sc,
        float *state_kv,
        float *state_score,
        float *row,
        const void *ape,
        uint32_t ape_type,
        const float *norm_w,
        uint32_t head_dim,
        uint32_t pos,
        uint32_t n_rot,
        uint32_t n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        float rms_eps) {
    const uint32_t tid = threadIdx.x;
    const uint32_t ratio = 4u;
    const uint32_t width = 2u * head_dim;
    const uint32_t pos_mod = pos % ratio;
    const uint32_t dst_row = ratio + pos_mod;

    for (uint32_t j = tid; j < width; j += blockDim.x) {
        state_kv[(uint64_t)dst_row * width + j] = kv[j];
        state_score[(uint64_t)dst_row * width + j] =
            sc[j] + model_scalar_dev(ape, 0, ape_type, (uint64_t)pos_mod * width + j);
    }
    __syncthreads();

    /* Non-emitting tokens only update one state row, just like the generic
     * compressor_store_kernel path. */
    if (pos_mod != ratio - 1u) return;

    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        float vals[8];
        float scores[8];
        float max_s = -INFINITY;
        uint32_t n_cand = 0;
        for (uint32_t r = 0; r < ratio; r++) {
            vals[n_cand] = state_kv[(uint64_t)r * width + d];
            scores[n_cand] = state_score[(uint64_t)r * width + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
        for (uint32_t r = 0; r < ratio; r++) {
            vals[n_cand] = state_kv[(uint64_t)(ratio + r) * width + head_dim + d];
            scores[n_cand] = state_score[(uint64_t)(ratio + r) * width + head_dim + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
        float den = 0.0f;
        float acc = 0.0f;
        for (uint32_t i = 0; i < n_cand; i++) {
            const float w = expf(scores[i] - max_s);
            den += w;
            acc += vals[i] * w;
        }
        row[d] = den != 0.0f ? acc / den : 0.0f;
    }
    __syncthreads();

    /* Match rms_norm_weight_kernel's 256-thread reduction order. */
    float sum = 0.0f;
    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        const float v = row[d];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[tid] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float norm_scale = rsqrtf(partial[0] / (float)head_dim + rms_eps);
    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        row[d] = row[d] * norm_scale * norm_w[d];
    }
    __syncthreads();

    const uint32_t n_nope = head_dim - n_rot;
    float corr0 = 0.0f;
    float corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        const float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot * logf((float)n_ctx_orig /
                      (beta_fast * 2.0f * (float)M_PI)) / denom);
        corr1 = ceilf((float)n_rot * logf((float)n_ctx_orig /
                     (beta_slow * 2.0f * (float)M_PI)) / denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1u), corr1);
    }
    const uint32_t rope_pos = pos + 1u - ratio;
    for (uint32_t pair = tid; pair < n_rot / 2u; pair += blockDim.x) {
        const uint32_t i = pair * 2u;
        const float theta_extrap = (float)rope_pos *
            powf(freq_base, -((float)i) / (float)n_rot);
        const float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            const float ramp_mix = rope_yarn_ramp_dev(corr0, corr1, (int)i) * ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        const float c = cosf(theta) * mscale;
        const float s = sinf(theta) * mscale;
        const float x0 = row[n_nope + i];
        const float x1 = row[n_nope + i + 1u];
        row[n_nope + i] = x0 * c - x1 * s;
        row[n_nope + i + 1u] = x0 * s + x1 * c;
    }
    __syncthreads();

    /* Rows 4..7 are already the desired next state; only copy them over the
     * previous rows.  Rewriting the source half, as the generic shift kernel
     * does, would store each value back unchanged. */
    const uint64_t half = 4ull * width;
    for (uint64_t i = tid; i < half; i += blockDim.x) {
        state_kv[i] = state_kv[half + i];
        state_score[i] = state_score[half + i];
    }
}

__device__ static float softplus_dev(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

__global__ static void router_select_kernel(
        int32_t *selected,
        float *weights,
        float *probs,
        const float *bias,
        const int32_t *hash,
        const float *logits,
        const int32_t *tokens,
        int32_t token_scalar,
        uint32_t hash_rows,
        uint32_t n_tokens,
        int has_bias,
        int hash_mode) {
    uint32_t t = blockIdx.x;
    if (t >= n_tokens || threadIdx.x != 0) return;
    const float *log = logits + (uint64_t)t * 256;
    float *prob = probs + (uint64_t)t * 256;
    int32_t *sel = selected + (uint64_t)t * 6;
    float *w = weights + (uint64_t)t * 6;

    for (int i = 0; i < 256; i++) prob[i] = sqrtf(softplus_dev(log[i]));

    if (hash_mode) {
        int32_t tok = tokens ? tokens[t] : token_scalar;
        if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;
        const int32_t *row = hash + (uint64_t)tok * 6;
        for (int i = 0; i < 6; i++) sel[i] = row[i];
    } else {
        for (int i = 0; i < 6; i++) sel[i] = -1;
        for (int i = 0; i < 256; i++) {
            float score = prob[i] + (has_bias ? bias[i] : 0.0f);
            for (int j = 0; j < 6; j++) {
                if (sel[j] < 0 || score > prob[sel[j]] + (has_bias ? bias[sel[j]] : 0.0f)) {
                    for (int k = 5; k > j; k--) sel[k] = sel[k - 1];
                    sel[j] = i;
                    break;
                }
            }
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < 6; i++) {
        int e = sel[i];
        float v = (e >= 0 && e < 256) ? prob[e] : 0.0f;
        w[i] = v;
        sum += v;
    }
    sum = fmaxf(sum, 6.103515625e-5f);
    for (int i = 0; i < 6; i++) w[i] = w[i] / sum * 1.5f;
}

__global__ static void router_select_parallel_kernel(
        int32_t *selected,
        float *weights,
        float *probs,
        const float *bias,
        const int32_t *hash,
        const float *logits,
        const int32_t *tokens,
        int32_t token_scalar,
        uint32_t hash_rows,
        uint32_t n_tokens,
        int has_bias,
        int hash_mode) {
    uint32_t t = blockIdx.x;
    uint32_t i = threadIdx.x;
    if (t >= n_tokens || i >= 256u) return;
    const float *log = logits + (uint64_t)t * 256;
    float *prob = probs + (uint64_t)t * 256;
    int32_t *sel = selected + (uint64_t)t * 6;
    float *w = weights + (uint64_t)t * 6;
    __shared__ float sprob[256];

    const float p = sqrtf(softplus_dev(log[i]));
    sprob[i] = p;
    prob[i] = p;
    __syncthreads();

    if (i != 0) return;
    if (hash_mode) {
        int32_t tok = tokens ? tokens[t] : token_scalar;
        if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;
        const int32_t *row = hash + (uint64_t)tok * 6;
        for (int j = 0; j < 6; j++) sel[j] = row[j];
    } else {
        for (int j = 0; j < 6; j++) sel[j] = -1;
        for (int e = 0; e < 256; e++) {
            float score = sprob[e] + (has_bias ? bias[e] : 0.0f);
            for (int j = 0; j < 6; j++) {
                if (sel[j] < 0 || score > sprob[sel[j]] + (has_bias ? bias[sel[j]] : 0.0f)) {
                    for (int k = 5; k > j; k--) sel[k] = sel[k - 1];
                    sel[j] = e;
                    break;
                }
            }
        }
    }

    float sum = 0.0f;
    for (int j = 0; j < 6; j++) {
        int e = sel[j];
        float v = (e >= 0 && e < 256) ? sprob[e] : 0.0f;
        w[j] = v;
        sum += v;
    }
    sum = fmaxf(sum, 6.103515625e-5f);
    for (int j = 0; j < 6; j++) w[j] = w[j] / sum * 1.5f;
}

__device__ __forceinline__ static bool router_score_better(float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

__global__ static void router_select_warp_topk_kernel(
        int32_t *selected,
        float *weights,
        float *probs,
        const float *bias,
        const int32_t *hash,
        const float *logits,
        const int32_t *tokens,
        int32_t token_scalar,
        uint32_t hash_rows,
        uint32_t n_tokens,
        int has_bias,
        int hash_mode) {
    const uint32_t lane = threadIdx.x;
    const uint32_t row_in_block = threadIdx.y;
    const uint32_t t = blockIdx.x * blockDim.y + row_in_block;
    if (t >= n_tokens || lane >= 32u) return;

    const float *log = logits + (uint64_t)t * 256u;
    float *prob = probs + (uint64_t)t * 256u;
    int32_t *sel = selected + (uint64_t)t * 6u;
    float *w = weights + (uint64_t)t * 6u;
    __shared__ float sprob[4][256];
    float local_prob[8];
    float local_score[8];

    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        const uint32_t e = lane + j * 32u;
        const float p = sqrtf(softplus_dev(log[e]));
        local_prob[j] = p;
        local_score[j] = p + (has_bias ? bias[e] : 0.0f);
        sprob[row_in_block][e] = p;
        prob[e] = p;
    }
    __syncwarp();

    if (hash_mode) {
        if (lane == 0) {
            int32_t tok = tokens ? tokens[t] : token_scalar;
            if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;
            const int32_t *row = hash + (uint64_t)tok * 6u;
            float sum = 0.0f;
            #pragma unroll
            for (uint32_t j = 0; j < 6u; j++) {
                const int32_t e = row[j];
                sel[j] = e;
                const float v = (e >= 0 && e < 256) ? sprob[row_in_block][(uint32_t)e] : 0.0f;
                w[j] = v;
                sum += v;
            }
            sum = fmaxf(sum, 6.103515625e-5f);
            #pragma unroll
            for (uint32_t j = 0; j < 6u; j++) w[j] = w[j] / sum * 1.5f;
        }
        return;
    }

    float out_prob[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t out_idx[6] = {0, 0, 0, 0, 0, 0};
    #pragma unroll
    for (uint32_t k = 0; k < 6u; k++) {
        float best_score = -INFINITY;
        float best_prob = 0.0f;
        uint32_t best_idx = UINT32_MAX;
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t e = lane + j * 32u;
            const float s = local_score[j];
            if (router_score_better(s, e, best_score, best_idx)) {
                best_score = s;
                best_prob = local_prob[j];
                best_idx = e;
            }
        }
        #pragma unroll
        for (uint32_t mask = 16u; mask > 0u; mask >>= 1u) {
            const float other_score = __shfl_xor_sync(0xffffffffu, best_score, mask);
            const float other_prob = __shfl_xor_sync(0xffffffffu, best_prob, mask);
            const uint32_t other_idx = __shfl_xor_sync(0xffffffffu, best_idx, mask);
            if (router_score_better(other_score, other_idx, best_score, best_idx)) {
                best_score = other_score;
                best_prob = other_prob;
                best_idx = other_idx;
            }
        }
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t e = lane + j * 32u;
            if (e == best_idx) local_score[j] = -INFINITY;
        }
        if (lane == 0) {
            out_idx[k] = best_idx;
            out_prob[k] = best_prob;
        }
    }

    if (lane == 0) {
        float sum = 0.0f;
        #pragma unroll
        for (uint32_t j = 0; j < 6u; j++) {
            sel[j] = (int32_t)out_idx[j];
            w[j] = out_prob[j];
            sum += out_prob[j];
        }
        sum = fmaxf(sum, 6.103515625e-5f);
        #pragma unroll
        for (uint32_t j = 0; j < 6u; j++) w[j] = w[j] / sum * 1.5f;
    }
}

__global__ static void swiglu_kernel(float *out, const float *gate, const float *up, uint32_t n, float clamp, float weight) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float u = up[i];
    if (clamp > 1.0e-6f) {
        g = fminf(g, clamp);
        u = fminf(fmaxf(u, -clamp), clamp);
    }
    float s = g / (1.0f + expf(-g));
    out[i] = s * u * weight;
}

__global__ static void add_kernel(float *out, const float *a, const float *b, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] + b[i];
}

__global__ static void directional_steering_project_kernel(
        float       *x,
        const float *directions,
        uint32_t     layer,
        uint32_t     width,
        uint32_t     rows,
        float        scale) {
    const uint32_t row = blockIdx.x;
    if (row >= rows || width == 0) return;

    float *xr = x + (uint64_t)row * width;
    const float *dir = directions + (uint64_t)layer * width;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
        sum += xr[i] * dir[i];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }

    const float coeff = scale * partial[0];
    for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
        xr[i] -= coeff * dir[i];
    }
}

__global__ static void zero_kernel(float *out, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 0.0f;
}

__global__ static void indexer_scores_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        int causal) {
    uint32_t c = blockIdx.x;
    uint32_t t = blockIdx.y;
    if (c >= n_comp || t >= n_tokens) return;
    if (causal) {
        uint32_t n_visible = (pos0 + t + 1u) / ratio;
        if (c >= n_visible) {
            if (threadIdx.x == 0) scores[(uint64_t)t * n_comp + c] = -INFINITY;
            return;
        }
    }
    float total = 0.0f;
    for (uint32_t h = 0; h < n_head; h++) {
        const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
        const float *kh = index_comp + (uint64_t)c * head_dim;
        float dot = 0.0f;
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) dot += qh[d] * kh[d];
        __shared__ float partial[256];
        partial[threadIdx.x] = dot;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        total += fmaxf(partial[0], 0.0f) * weights[(uint64_t)t * n_head + h];
        __syncthreads();
    }
    if (threadIdx.x == 0) scores[(uint64_t)t * n_comp + c] = total * scale;
}

__global__ static void indexer_score_one_direct_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t pos0,
        uint32_t ratio,
        float scale,
        int causal) {
    const uint32_t c = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (c >= n_comp || tid >= 128u) return;
    if (causal) {
        const uint32_t visible = ratio ? (pos0 + 1u) / ratio : n_comp;
        if (c >= visible) {
            if (tid == 0) scores[c] = -INFINITY;
            return;
        }
    }

    __shared__ float krow[128];
    __shared__ float partial[4];
    if (tid < 128u) krow[tid] = index_comp[(uint64_t)c * 128u + tid];
    __syncthreads();

    float total = 0.0f;
    for (uint32_t h0 = 0; h0 < 64u; h0 += 4u) {
        const uint32_t h = h0 + warp;
        const float4 qv = ((const float4 *)(q + (uint64_t)h * 128u))[lane];
        const float4 kv = ((const float4 *)krow)[lane];
        float dot = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
        dot = warp_sum_f32(dot);
        if (lane == 0) partial[warp] = fmaxf(dot, 0.0f) * weights[h] * scale;
        __syncthreads();
        if (tid == 0) total += partial[0] + partial[1] + partial[2] + partial[3];
        __syncthreads();
    }
    if (tid == 0) scores[c] = total;
}

__global__ static void indexer_scores_wmma_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        int causal) {
#if __CUDA_ARCH__ >= 700
    namespace wmma = nvcuda::wmma;
    const uint32_t tile_c = blockIdx.x * 16u;
    const uint32_t tile_t = blockIdx.y * 16u;
    const uint32_t tid = threadIdx.x;
    if (tid >= 32u || head_dim != 128u) return;

    if (causal) {
        const uint32_t last_token = min(tile_t + 16u, n_tokens);
        const uint32_t max_visible = last_token > tile_t
            ? min((pos0 + last_token) / ratio, n_comp)
            : 0u;
        if (tile_c >= max_visible) {
            for (uint32_t i = tid; i < 16u * 16u; i += 32u) {
                const uint32_t r = i >> 4u;
                const uint32_t c = i & 15u;
                const uint32_t token = tile_t + r;
                const uint32_t comp = tile_c + c;
                if (token < n_tokens && comp < n_comp) {
                    scores[(uint64_t)token * n_comp + comp] = -INFINITY;
                }
            }
            return;
        }
    }

    __shared__ __half a_sh[16 * 128];
    __shared__ __half b_sh[16 * 128];
    __shared__ float c_sh[16 * 16];
    __shared__ float acc_sh[16 * 16];

    for (uint32_t i = tid; i < 16u * 16u; i += 32u) acc_sh[i] = 0.0f;
    for (uint32_t i = tid; i < 16u * 128u; i += 32u) {
        const uint32_t c = i >> 7u;
        const uint32_t d = i & 127u;
        const uint32_t comp = tile_c + c;
        float v = 0.0f;
        if (comp < n_comp) v = index_comp[(uint64_t)comp * head_dim + d];
        b_sh[d + c * 128u] = __float2half(v);
    }
    __syncthreads();

    for (uint32_t h = 0; h < n_head; h++) {
        for (uint32_t i = tid; i < 16u * 128u; i += 32u) {
            const uint32_t r = i >> 7u;
            const uint32_t d = i & 127u;
            const uint32_t token = tile_t + r;
            float v = 0.0f;
            if (token < n_tokens) {
                v = q[((uint64_t)token * n_head + h) * head_dim + d];
            }
            a_sh[i] = __float2half(v);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        for (uint32_t k0 = 0; k0 < 128u; k0 += 16u) {
            wmma::load_matrix_sync(a_frag, a_sh + k0, 128);
            wmma::load_matrix_sync(b_frag, b_sh + k0, 128);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(c_sh, c_frag, 16, wmma::mem_row_major);
        __syncthreads();

        for (uint32_t i = tid; i < 16u * 16u; i += 32u) {
            const uint32_t r = i >> 4u;
            const uint32_t token = tile_t + r;
            if (token < n_tokens) {
                const float w = weights[(uint64_t)token * n_head + h];
                acc_sh[i] += fmaxf(c_sh[i], 0.0f) * w;
            }
        }
        __syncthreads();
    }

    for (uint32_t i = tid; i < 16u * 16u; i += 32u) {
        const uint32_t r = i >> 4u;
        const uint32_t c = i & 15u;
        const uint32_t token = tile_t + r;
        const uint32_t comp = tile_c + c;
        if (token < n_tokens && comp < n_comp) {
            float out = acc_sh[i] * scale;
            if (causal) {
                const uint32_t visible = (pos0 + token + 1u) / ratio;
                if (comp >= visible) out = -INFINITY;
            }
            scores[(uint64_t)token * n_comp + comp] = out;
        }
    }
#endif
}

__global__ static void indexer_scores_wmma32_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        int causal) {
#if __CUDA_ARCH__ >= 700
    namespace wmma = nvcuda::wmma;
    const uint32_t tile_c = blockIdx.x * 32u;
    const uint32_t tile_t = blockIdx.y * 16u;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    if (tid >= 64u || head_dim != 128u) return;

    if (causal) {
        const uint32_t last_token = min(tile_t + 16u, n_tokens);
        const uint32_t max_visible = last_token > tile_t
            ? min((pos0 + last_token) / ratio, n_comp)
            : 0u;
        if (tile_c >= max_visible) {
            for (uint32_t i = tid; i < 16u * 32u; i += 64u) {
                const uint32_t r = i >> 5u;
                const uint32_t c = i & 31u;
                const uint32_t token = tile_t + r;
                const uint32_t comp = tile_c + c;
                if (token < n_tokens && comp < n_comp) {
                    scores[(uint64_t)token * n_comp + comp] = -INFINITY;
                }
            }
            return;
        }
    }

    __shared__ __half a_sh[16 * 128];
    __shared__ __half b_sh[32 * 128];
    __shared__ float c_sh[2 * 16 * 16];
    __shared__ float acc_sh[2 * 16 * 16];

    for (uint32_t i = tid; i < 2u * 16u * 16u; i += 64u) acc_sh[i] = 0.0f;
    for (uint32_t i = tid; i < 32u * 128u; i += 64u) {
        const uint32_t c = i >> 7u;
        const uint32_t d = i & 127u;
        const uint32_t comp = tile_c + c;
        float v = 0.0f;
        if (comp < n_comp) v = index_comp[(uint64_t)comp * head_dim + d];
        b_sh[d + c * 128u] = __float2half(v);
    }
    __syncthreads();

    for (uint32_t h = 0; h < n_head; h++) {
        for (uint32_t i = tid; i < 16u * 128u; i += 64u) {
            const uint32_t r = i >> 7u;
            const uint32_t d = i & 127u;
            const uint32_t token = tile_t + r;
            float v = 0.0f;
            if (token < n_tokens) {
                v = q[((uint64_t)token * n_head + h) * head_dim + d];
            }
            a_sh[i] = __float2half(v);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        const uint32_t col0 = warp * 16u;
        for (uint32_t k0 = 0; k0 < 128u; k0 += 16u) {
            wmma::load_matrix_sync(a_frag, a_sh + k0, 128);
            wmma::load_matrix_sync(b_frag, b_sh + col0 * 128u + k0, 128);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(c_sh + warp * 16u * 16u, c_frag, 16, wmma::mem_row_major);
        __syncthreads();

        for (uint32_t i = tid; i < 2u * 16u * 16u; i += 64u) {
            const uint32_t wtile = i >> 8u;
            const uint32_t local = i & 255u;
            const uint32_t r = local >> 4u;
            const uint32_t c = local & 15u;
            const uint32_t token = tile_t + r;
            const uint32_t comp = tile_c + wtile * 16u + c;
            if (token < n_tokens && comp < n_comp) {
                const float w = weights[(uint64_t)token * n_head + h];
                acc_sh[i] += fmaxf(c_sh[i], 0.0f) * w;
            }
        }
        __syncthreads();
    }

    for (uint32_t i = tid; i < 2u * 16u * 16u; i += 64u) {
        const uint32_t wtile = i >> 8u;
        const uint32_t local = i & 255u;
        const uint32_t r = local >> 4u;
        const uint32_t c = local & 15u;
        const uint32_t token = tile_t + r;
        const uint32_t comp = tile_c + wtile * 16u + c;
        if (token < n_tokens && comp < n_comp) {
            float out = acc_sh[i] * scale;
            if (causal) {
                const uint32_t visible = (pos0 + token + 1u) / ratio;
                if (comp >= visible) out = -INFINITY;
            }
            scores[(uint64_t)token * n_comp + comp] = out;
        }
    }
#endif
}

__global__ static void indexer_scores_wmma64_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        int causal) {
#if __CUDA_ARCH__ >= 700
    namespace wmma = nvcuda::wmma;
    const uint32_t tile_c = blockIdx.x * 64u;
    const uint32_t tile_t = blockIdx.y * 16u;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    if (tid >= 128u || head_dim != 128u) return;

    if (causal) {
        const uint32_t last_token = min(tile_t + 16u, n_tokens);
        const uint32_t max_visible = last_token > tile_t
            ? min((pos0 + last_token) / ratio, n_comp)
            : 0u;
        if (tile_c >= max_visible) {
            for (uint32_t i = tid; i < 16u * 64u; i += 128u) {
                const uint32_t r = i >> 6u;
                const uint32_t c = i & 63u;
                const uint32_t token = tile_t + r;
                const uint32_t comp = tile_c + c;
                if (token < n_tokens && comp < n_comp) {
                    scores[(uint64_t)token * n_comp + comp] = -INFINITY;
                }
            }
            return;
        }
    }

    __shared__ __half a_sh[16 * 128];
    __shared__ __half b_sh[64 * 128];
    __shared__ float c_sh[4 * 16 * 16];
    __shared__ float acc_sh[4 * 16 * 16];

    for (uint32_t i = tid; i < 4u * 16u * 16u; i += 128u) acc_sh[i] = 0.0f;
    for (uint32_t i = tid; i < 64u * 128u; i += 128u) {
        const uint32_t c = i >> 7u;
        const uint32_t d = i & 127u;
        const uint32_t comp = tile_c + c;
        float v = 0.0f;
        if (comp < n_comp) v = index_comp[(uint64_t)comp * head_dim + d];
        b_sh[d + c * 128u] = __float2half(v);
    }
    __syncthreads();

    for (uint32_t h = 0; h < n_head; h++) {
        for (uint32_t i = tid; i < 16u * 128u; i += 128u) {
            const uint32_t r = i >> 7u;
            const uint32_t d = i & 127u;
            const uint32_t token = tile_t + r;
            float v = 0.0f;
            if (token < n_tokens) {
                v = q[((uint64_t)token * n_head + h) * head_dim + d];
            }
            a_sh[i] = __float2half(v);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        const uint32_t col0 = warp * 16u;
        for (uint32_t k0 = 0; k0 < 128u; k0 += 16u) {
            wmma::load_matrix_sync(a_frag, a_sh + k0, 128);
            wmma::load_matrix_sync(b_frag, b_sh + col0 * 128u + k0, 128);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(c_sh + warp * 16u * 16u, c_frag, 16, wmma::mem_row_major);
        __syncthreads();

        for (uint32_t i = tid; i < 4u * 16u * 16u; i += 128u) {
            const uint32_t wtile = i >> 8u;
            const uint32_t local = i & 255u;
            const uint32_t r = local >> 4u;
            const uint32_t c = local & 15u;
            const uint32_t token = tile_t + r;
            const uint32_t comp = tile_c + wtile * 16u + c;
            if (token < n_tokens && comp < n_comp) {
                const float w = weights[(uint64_t)token * n_head + h];
                acc_sh[i] += fmaxf(c_sh[i], 0.0f) * w;
            }
        }
        __syncthreads();
    }

    for (uint32_t i = tid; i < 4u * 16u * 16u; i += 128u) {
        const uint32_t wtile = i >> 8u;
        const uint32_t local = i & 255u;
        const uint32_t r = local >> 4u;
        const uint32_t c = local & 15u;
        const uint32_t token = tile_t + r;
        const uint32_t comp = tile_c + wtile * 16u + c;
        if (token < n_tokens && comp < n_comp) {
            float out = acc_sh[i] * scale;
            if (causal) {
                const uint32_t visible = (pos0 + token + 1u) / ratio;
                if (comp >= visible) out = -INFINITY;
            }
            scores[(uint64_t)token * n_comp + comp] = out;
        }
    }
#endif
}

__global__ static void indexer_scores_wmma128_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        int causal) {
#if __CUDA_ARCH__ >= 700
    namespace wmma = nvcuda::wmma;
    const uint32_t tile_c = blockIdx.x * 128u;
    const uint32_t tile_t = blockIdx.y * 16u;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    if (tid >= 256u || head_dim != 128u) return;

    if (causal) {
        const uint32_t last_token = min(tile_t + 16u, n_tokens);
        const uint32_t max_visible = last_token > tile_t
            ? min((pos0 + last_token) / ratio, n_comp)
            : 0u;
        if (tile_c >= max_visible) {
            for (uint32_t i = tid; i < 16u * 128u; i += 256u) {
                const uint32_t r = i >> 7u;
                const uint32_t c = i & 127u;
                const uint32_t token = tile_t + r;
                const uint32_t comp = tile_c + c;
                if (token < n_tokens && comp < n_comp) {
                    scores[(uint64_t)token * n_comp + comp] = -INFINITY;
                }
            }
            return;
        }
    }

    __shared__ __half a_sh[16 * 128];
    __shared__ __half b_sh[128 * 128];
    __shared__ float c_sh[8 * 16 * 16];

    float acc[8];
#pragma unroll
    for (uint32_t i = 0; i < 8u; i++) acc[i] = 0.0f;

    for (uint32_t i = tid; i < 128u * 128u; i += 256u) {
        const uint32_t c = i >> 7u;
        const uint32_t d = i & 127u;
        const uint32_t comp = tile_c + c;
        float v = 0.0f;
        if (comp < n_comp) v = index_comp[(uint64_t)comp * head_dim + d];
        b_sh[d + c * 128u] = __float2half(v);
    }
    __syncthreads();

    for (uint32_t h = 0; h < n_head; h++) {
        for (uint32_t i = tid; i < 16u * 128u; i += 256u) {
            const uint32_t r = i >> 7u;
            const uint32_t d = i & 127u;
            const uint32_t token = tile_t + r;
            float v = 0.0f;
            if (token < n_tokens) {
                v = q[((uint64_t)token * n_head + h) * head_dim + d];
            }
            a_sh[i] = __float2half(v);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        const uint32_t col0 = warp * 16u;
        for (uint32_t k0 = 0; k0 < 128u; k0 += 16u) {
            wmma::load_matrix_sync(a_frag, a_sh + k0, 128);
            wmma::load_matrix_sync(b_frag, b_sh + col0 * 128u + k0, 128);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(c_sh + warp * 16u * 16u, c_frag, 16, wmma::mem_row_major);
        __syncthreads();

        const uint32_t local0 = tid & 255u;
        const uint32_t token0 = tile_t + (local0 >> 4u);
        const float w0 = token0 < n_tokens ? weights[(uint64_t)token0 * n_head + h] : 0.0f;
        uint32_t slot = 0;
        for (uint32_t i = tid; i < 8u * 16u * 16u; i += 256u, slot++) {
            const uint32_t wtile = i >> 8u;
            const uint32_t local = i & 255u;
            const uint32_t r = local >> 4u;
            const uint32_t c = local & 15u;
            const uint32_t token = tile_t + r;
            const uint32_t comp = tile_c + wtile * 16u + c;
            if (token < n_tokens && comp < n_comp) {
                acc[slot] += fmaxf(c_sh[i], 0.0f) * w0;
            }
        }
        __syncthreads();
    }

    uint32_t slot = 0;
    for (uint32_t i = tid; i < 8u * 16u * 16u; i += 256u, slot++) {
        const uint32_t wtile = i >> 8u;
        const uint32_t local = i & 255u;
        const uint32_t r = local >> 4u;
        const uint32_t c = local & 15u;
        const uint32_t token = tile_t + r;
        const uint32_t comp = tile_c + wtile * 16u + c;
        if (token < n_tokens && comp < n_comp) {
            float out = acc[slot] * scale;
            if (causal) {
                const uint32_t visible = (pos0 + token + 1u) / ratio;
                if (comp >= visible) out = -INFINITY;
            }
            scores[(uint64_t)token * n_comp + comp] = out;
        }
    }
#endif
}

/* One-block-per-row argmax over n_vocab F32 logits. 1024 threads cooperate on
 * each row, tracking a (best_v, best_idx) pair per thread and then reducing in
 * shared memory with value-keyed comparison.
 *
 * Tie-breaking: lower index wins, matching the host sample_argmax used by
 * the CPU reference path. Replaces the indexer-as-argmax workaround used
 * in the MTP top-id sites, which fell through to the legacy single-thread
 * indexer_topk_kernel at top_k=1, costing ~17.5 ms per call on n_vocab=129280. */
__global__ static void argmax_kernel(int32_t *out_idx, const float *logits, uint32_t n_vocab) {
    enum { THREADS = 1024 };
    __shared__ float sm_val[THREADS];
    __shared__ int32_t sm_idx[THREADS];

    const uint32_t tid = threadIdx.x;
    const uint32_t row = blockIdx.x;
    logits += (uint64_t)row * n_vocab;
    float local_v = -INFINITY;
    int32_t local_i = 0;
    for (uint32_t i = tid; i < n_vocab; i += THREADS) {
        const float v = logits[i];
        if (v > local_v) {
            local_v = v;
            local_i = (int32_t)i;
        }
    }
    sm_val[tid] = local_v;
    sm_idx[tid] = local_i;
    __syncthreads();

    for (uint32_t s = THREADS / 2u; s > 0u; s >>= 1) {
        if (tid < s) {
            const float vr = sm_val[tid + s];
            const int32_t ir = sm_idx[tid + s];
            const float vl = sm_val[tid];
            const int32_t il = sm_idx[tid];
            /* Larger value wins; on exact ties prefer the lower index. */
            const bool take_right = (vr > vl) || (vr == vl && ir < il);
            if (take_right) {
                sm_val[tid] = vr;
                sm_idx[tid] = ir;
            }
        }
        __syncthreads();
    }

    if (tid == 0) out_idx[row] = sm_idx[0];
}

/* Full-vocabulary sampler used by DSpark's causal Markov head.  The server's
 * default stochastic policy is top_k=0, top_p=1 and min_p=0.05, so no sort is
 * required: one block computes max, the filtered partition function, the
 * normalized q vector and a vocabulary-order inverse-CDF draw.  Contiguous
 * per-thread ranges make the final draw deterministic without a global scan.
 * Keeping q on device lets the target verifier later apply exact p/q rejection
 * sampling instead of the former argmax/exact-match approximation. */
__global__ static void sample_min_p_kernel(
        int32_t *out_idx,
        float *out_probs,
        const float *logits,
        const float *uniform,
        uint32_t n_vocab,
        float temperature,
        float min_p) {
    enum { THREADS = 1024 };
    __shared__ float scratch[THREADS];
    __shared__ float row_max;
    __shared__ float row_sum;
    __shared__ uint32_t selected_thread;
    __shared__ float selected_offset;

    const uint32_t tid = threadIdx.x;
    const uint32_t chunk = (n_vocab + THREADS - 1u) / THREADS;
    const uint32_t begin = tid * chunk;
    const uint32_t end = min(begin + chunk, n_vocab);

    float local_max = -INFINITY;
    for (uint32_t i = begin; i < end; i++) {
        const float v = logits[i];
        if (isfinite(v) && v > local_max) local_max = v;
    }
    scratch[tid] = local_max;
    __syncthreads();
    for (uint32_t step = THREADS / 2u; step > 0u; step >>= 1u) {
        if (tid < step) scratch[tid] = fmaxf(scratch[tid], scratch[tid + step]);
        __syncthreads();
    }
    if (tid == 0) row_max = scratch[0];
    __syncthreads();

    float local_sum = 0.0f;
    if (isfinite(row_max) && temperature > 0.0f) {
        for (uint32_t i = begin; i < end; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            const float w = expf((v - row_max) / temperature);
            if (w >= min_p) local_sum += w;
        }
    }
    scratch[tid] = local_sum;
    __syncthreads();
    for (uint32_t step = THREADS / 2u; step > 0u; step >>= 1u) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        __syncthreads();
    }
    if (tid == 0) row_sum = scratch[0];
    __syncthreads();

    const bool valid = row_sum > 0.0f && isfinite(row_sum);
    for (uint32_t i = begin; i < end; i++) {
        float probability = 0.0f;
        if (valid && isfinite(logits[i])) {
            const float w = expf((logits[i] - row_max) / temperature);
            if (w >= min_p) probability = w / row_sum;
        }
        out_probs[i] = probability;
    }

    /* Preserve each contiguous range sum for a cheap ordered prefix search.
     * Thread zero only scans 1024 values; the expensive vocabulary work stays
     * parallel. */
    scratch[tid] = valid ? local_sum / row_sum : 0.0f;
    __syncthreads();
    if (tid == 0) {
        float target = valid ? fminf(fmaxf(uniform[0], 0.0f),
                                     0.9999999403953552f) : 0.0f;
        float prefix = 0.0f;
        selected_thread = 0;
        selected_offset = 0.0f;
        for (uint32_t t = 0; t < THREADS; t++) {
            const float mass = scratch[t];
            if (mass > 0.0f) {
                selected_thread = t;
                selected_offset = mass;
            }
            if (mass > 0.0f && target <= prefix + mass) {
                selected_thread = t;
                selected_offset = target - prefix;
                break;
            }
            prefix += mass;
        }
    }
    __syncthreads();

    if (tid == selected_thread) {
        float remaining = selected_offset;
        int32_t selected = 0;
        bool found = false;
        for (uint32_t i = begin; i < end; i++) {
            const float probability = out_probs[i];
            if (probability <= 0.0f) continue;
            selected = (int32_t)i;
            remaining -= probability;
            if (remaining <= 0.0f) {
                found = true;
                break;
            }
        }
        if (!found && end > begin) {
            for (uint32_t i = end; i > begin; i--) {
                if (out_probs[i - 1u] > 0.0f) {
                    selected = (int32_t)(i - 1u);
                    break;
                }
            }
        }
        out_idx[0] = selected;
    }
}

__global__ static void dspark_rejection_verify_kernel(
        int32_t *out_tokens,
        int32_t *out_accept,
        const float *target_logits,
        const float *draft_probs,
        const int32_t *dspark_tokens,
        const float *accept_uniforms,
        const float *residual_uniforms,
        uint32_t n_rows,
        uint32_t n_vocab,
        float temperature,
        float min_p) {
    enum { THREADS = 1024 };
    __shared__ float scratch[THREADS];
    __shared__ float row_max;
    __shared__ float row_sum;
    __shared__ int32_t draft_token_shared;
    __shared__ int32_t accepted_shared;
    __shared__ uint32_t selected_thread;
    __shared__ float selected_offset;

    const uint32_t row = blockIdx.x;
    if (row >= n_rows) return;
    const uint32_t tid = threadIdx.x;
    const uint32_t chunk = (n_vocab + THREADS - 1u) / THREADS;
    const uint32_t begin = tid * chunk;
    const uint32_t end = min(begin + chunk, n_vocab);
    const float *logits = target_logits + (uint64_t)row * n_vocab;
    const float *q = draft_probs + (uint64_t)row * n_vocab;

    if (tid == 0) {
        const int32_t tok = dspark_tokens[row + 1u];
        draft_token_shared = (tok >= 0 && (uint32_t)tok < n_vocab) ? tok : 0;
        accepted_shared = 0;
        out_accept[row] = 0;
        out_tokens[row] = draft_token_shared;
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (uint32_t i = begin; i < end; i++) {
        const float v = logits[i];
        if (isfinite(v) && v > local_max) local_max = v;
    }
    scratch[tid] = local_max;
    __syncthreads();
    for (uint32_t step = THREADS / 2u; step > 0u; step >>= 1u) {
        if (tid < step) scratch[tid] = fmaxf(scratch[tid], scratch[tid + step]);
        __syncthreads();
    }
    if (tid == 0) row_max = scratch[0];
    __syncthreads();

    float local_sum = 0.0f;
    if (isfinite(row_max) && temperature > 0.0f) {
        for (uint32_t i = begin; i < end; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            const float w = expf((v - row_max) / temperature);
            if (w >= min_p) local_sum += w;
        }
    }
    scratch[tid] = local_sum;
    __syncthreads();
    for (uint32_t step = THREADS / 2u; step > 0u; step >>= 1u) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        __syncthreads();
    }
    if (tid == 0) {
        row_sum = scratch[0];
        float p_tok = 0.0f;
        if (row_sum > 0.0f && isfinite(row_sum)) {
            const float v = logits[draft_token_shared];
            if (isfinite(v)) {
                const float w = expf((v - row_max) / temperature);
                if (w >= min_p) p_tok = w / row_sum;
            }
        }
        const float q_tok = q[draft_token_shared];
        float ap = q_tok > 0.0f ? p_tok / q_tok : 0.0f;
        if (ap > 1.0f) ap = 1.0f;
        if (ap < 0.0f || !isfinite(ap)) ap = 0.0f;
        const float u = fminf(fmaxf(accept_uniforms[row], 0.0f),
                              0.9999999403953552f);
        if (u <= ap) {
            accepted_shared = 1;
            out_accept[row] = 1;
            out_tokens[row] = draft_token_shared;
        }
    }
    __syncthreads();
    if (accepted_shared) return;

    float residual_sum_local = 0.0f;
    if (row_sum > 0.0f && isfinite(row_sum)) {
        for (uint32_t i = begin; i < end; i++) {
            const float v = logits[i];
            float p = 0.0f;
            if (isfinite(v)) {
                const float w = expf((v - row_max) / temperature);
                if (w >= min_p) p = w / row_sum;
            }
            const float residual = p - q[i];
            if (residual > 0.0f) residual_sum_local += residual;
        }
    }
    scratch[tid] = residual_sum_local;
    __syncthreads();
    for (uint32_t step = THREADS / 2u; step > 0u; step >>= 1u) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        __syncthreads();
    }
    const float residual_sum = scratch[0];

    float range_mass = 0.0f;
    if (residual_sum > 0.0f && isfinite(residual_sum) &&
        row_sum > 0.0f && isfinite(row_sum)) {
        for (uint32_t i = begin; i < end; i++) {
            const float v = logits[i];
            float p = 0.0f;
            if (isfinite(v)) {
                const float w = expf((v - row_max) / temperature);
                if (w >= min_p) p = w / row_sum;
            }
            const float residual = p - q[i];
            if (residual > 0.0f) range_mass += residual / residual_sum;
        }
    } else if (row_sum > 0.0f && isfinite(row_sum)) {
        for (uint32_t i = begin; i < end; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            const float w = expf((v - row_max) / temperature);
            if (w >= min_p) range_mass += w / row_sum;
        }
    }
    scratch[tid] = range_mass;
    __syncthreads();
    if (tid == 0) {
        float target = fminf(fmaxf(residual_uniforms[row], 0.0f),
                             0.9999999403953552f);
        float prefix = 0.0f;
        selected_thread = 0;
        selected_offset = 0.0f;
        for (uint32_t t = 0; t < THREADS; t++) {
            const float mass = scratch[t];
            if (mass > 0.0f) {
                selected_thread = t;
                selected_offset = mass;
            }
            if (mass > 0.0f && target <= prefix + mass) {
                selected_thread = t;
                selected_offset = target - prefix;
                break;
            }
            prefix += mass;
        }
    }
    __syncthreads();

    if (tid == selected_thread) {
        float remaining = selected_offset;
        int32_t selected = 0;
        bool found = false;
        for (uint32_t i = begin; i < end; i++) {
            const float v = logits[i];
            float mass = 0.0f;
            if (isfinite(v) && row_sum > 0.0f && isfinite(row_sum)) {
                const float w = expf((v - row_max) / temperature);
                if (w >= min_p) {
                    const float p = w / row_sum;
                    const float residual = p - q[i];
                    if (residual_sum > 0.0f && isfinite(residual_sum)) {
                        if (residual > 0.0f) mass = residual / residual_sum;
                    } else {
                        mass = p;
                    }
                }
            }
            if (mass <= 0.0f) continue;
            selected = (int32_t)i;
            remaining -= mass;
            if (remaining <= 0.0f) {
                found = true;
                break;
            }
        }
        if (!found && end > begin) {
            for (uint32_t i = end; i > begin; i--) {
                const float v = logits[i - 1u];
                if (!isfinite(v)) continue;
                const float w = expf((v - row_max) / temperature);
                if (w >= min_p) {
                    selected = (int32_t)(i - 1u);
                    break;
                }
            }
        }
        out_tokens[row] = selected;
    }
}

__global__ static void indexer_topk_kernel(uint32_t *selected, const float *scores, uint32_t n_comp, uint32_t n_tokens, uint32_t top_k) {
    uint32_t t = blockIdx.x;
    if (t >= n_tokens || threadIdx.x != 0) return;
    const float *row = scores + (uint64_t)t * n_comp;
    uint32_t *sel = selected + (uint64_t)t * top_k;
    for (uint32_t k = 0; k < top_k; k++) sel[k] = 0;
    for (uint32_t c = 0; c < n_comp; c++) {
        float v = row[c];
        for (uint32_t k = 0; k < top_k; k++) {
            if ((k >= c) || v > row[sel[k]]) {
                for (uint32_t j = top_k - 1; j > k; j--) sel[j] = sel[j - 1];
                sel[k] = c;
                break;
            }
        }
    }
}

__device__ __forceinline__ static bool topk_score_better(float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

__device__ __forceinline__ static uint32_t topk_float_ordered_key(float v) {
    const uint32_t u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

__device__ __forceinline__ static uint64_t topk_pack_key(float v, uint32_t idx) {
    return ((uint64_t)topk_float_ordered_key(v) << 32u) | (uint64_t)(0xffffffffu - idx);
}

__global__ static void indexer_topk_8192_cub_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    constexpr uint32_t BLOCK_THREADS = 512u;
    constexpr uint32_t ITEMS_PER_THREAD = 16u;
    using BlockSort = cub::BlockRadixSort<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>;
    extern __shared__ __align__(16) unsigned char sort_smem[];
    typename BlockSort::TempStorage &sort_storage =
        *reinterpret_cast<typename BlockSort::TempStorage *>(sort_smem);

    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= BLOCK_THREADS) return;

    const float *row = scores + (uint64_t)t * n_comp;
    uint64_t keys[ITEMS_PER_THREAD];
#pragma unroll
    for (uint32_t item = 0; item < ITEMS_PER_THREAD; item++) {
        const uint32_t i = tid * ITEMS_PER_THREAD + item;
        if (i < n_comp) {
            keys[item] = topk_pack_key(row[i], i);
        } else {
            keys[item] = topk_pack_key(-INFINITY, UINT32_MAX);
        }
    }

    BlockSort(sort_storage).SortDescending(keys);

#pragma unroll
    for (uint32_t item = 0; item < ITEMS_PER_THREAD; item++) {
        const uint32_t i = tid * ITEMS_PER_THREAD + item;
        if (i < top_k) {
            selected[(uint64_t)t * top_k + i] = 0xffffffffu - (uint32_t)keys[item];
        }
    }
}

__global__ static void indexer_topk_1024_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= 1024u) return;
    __shared__ float vals[1024];
    __shared__ uint32_t idxs[1024];

    const float *row = scores + (uint64_t)t * n_comp;
    if (tid < n_comp) {
        vals[tid] = row[tid];
        idxs[tid] = tid;
    } else {
        vals[tid] = -INFINITY;
        idxs[tid] = UINT32_MAX;
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= 1024u; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            uint32_t other = tid ^ j;
            if (other > tid && other < 1024u) {
                const float av = vals[tid];
                const float bv = vals[other];
                const uint32_t ai = idxs[tid];
                const uint32_t bi = idxs[other];
                const bool desc_half = (tid & k) == 0u;
                const bool swap = desc_half
                    ? topk_score_better(bv, bi, av, ai)
                    : topk_score_better(av, ai, bv, bi);
                if (swap) {
                    vals[tid] = bv;
                    idxs[tid] = bi;
                    vals[other] = av;
                    idxs[other] = ai;
                }
            }
            __syncthreads();
        }
    }

    if (tid < top_k) selected[(uint64_t)t * top_k + tid] = idxs[tid];
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_pow2_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        if (i < n_comp) {
            vals[i] = row[i];
            idxs[i] = i;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = UINT32_MAX;
        }
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        selected[(uint64_t)t * top_k + i] = idxs[i];
    }
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_pow2_u16_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    __shared__ float vals[SORT_N];
    __shared__ uint16_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        if (i < n_comp) {
            vals[i] = row[i];
            idxs[i] = (uint16_t)i;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = UINT16_MAX;
        }
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = (uint16_t)bi;
                        vals[other] = av;
                        idxs[other] = (uint16_t)ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        selected[(uint64_t)t * top_k + i] = idxs[i];
    }
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_chunk_pow2_kernel(
        uint32_t *candidates,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t candidate_stride) {
    uint32_t t = blockIdx.x;
    uint32_t chunk = blockIdx.y;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;

    const uint32_t chunk_start = chunk * SORT_N;
    if (chunk_start >= n_comp) return;
    const uint32_t chunk_n = n_comp - chunk_start < SORT_N ? n_comp - chunk_start : SORT_N;
    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        if (i < chunk_n) {
            vals[i] = row[chunk_start + i];
            idxs[i] = chunk_start + i;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = UINT32_MAX;
        }
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    uint32_t *out = candidates + (uint64_t)t * candidate_stride + chunk * top_k;
    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        out[i] = idxs[i];
    }
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_merge_pow2_kernel(
        uint32_t *selected,
        const uint32_t *candidates,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t candidate_count,
        uint32_t candidate_stride) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    const uint32_t *cand = candidates + (uint64_t)t * candidate_stride;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        uint32_t idx = UINT32_MAX;
        float v = -INFINITY;
        if (i < candidate_count) {
            idx = cand[i];
            if (idx < n_comp) v = row[idx];
        }
        vals[i] = v;
        idxs[i] = idx;
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        selected[(uint64_t)t * top_k + i] = idxs[i];
    }
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_tree_merge_pow2_kernel(
        uint32_t *out,
        const uint32_t *candidates,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t n_sets,
        uint32_t merge_group,
        uint32_t candidate_stride,
        uint32_t out_stride) {
    uint32_t t = blockIdx.x;
    uint32_t group = blockIdx.y;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;

    const uint32_t set0 = group * merge_group;
    if (set0 >= n_sets) return;
    uint32_t set_count = n_sets - set0;
    if (set_count > merge_group) set_count = merge_group;
    const uint32_t candidate_count = set_count * top_k;

    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    const uint32_t *cand = candidates + (uint64_t)t * candidate_stride + set0 * top_k;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        uint32_t idx = UINT32_MAX;
        float v = -INFINITY;
        if (i < candidate_count) {
            idx = cand[i];
            if (idx < n_comp) v = row[idx];
        }
        vals[i] = v;
        idxs[i] = idx;
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    uint32_t *dst = out + (uint64_t)t * out_stride + group * top_k;
    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        dst[i] = idxs[i];
    }
}

__global__ static void indexed_topk_sort_512_asc_kernel(
        int32_t *dst,
        const int32_t *src,
        uint32_t n_tokens) {
    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= 512u) return;
    __shared__ int32_t rows[512];

    const int32_t *src_row = src + (uint64_t)t * 512u;
    int32_t *dst_row = dst + (uint64_t)t * 512u;
    rows[tid] = src_row[tid];
    __syncthreads();

    for (uint32_t k = 2u; k <= 512u; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            const uint32_t other = tid ^ j;
            if (other > tid && other < 512u) {
                const int32_t a = rows[tid];
                const int32_t b = rows[other];
                const bool up = (tid & k) == 0u;
                if ((up && a > b) || (!up && a < b)) {
                    rows[tid] = b;
                    rows[other] = a;
                }
            }
            __syncthreads();
        }
    }

    dst_row[tid] = rows[tid];
}

__global__ static void topk_mask_kernel(float *mask, const uint32_t *topk, uint32_t n_comp, uint32_t n_tokens, uint32_t top_k) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_comp;
    if (gid >= n) return;
    uint32_t t = gid / n_comp;
    uint32_t c = gid - (uint64_t)t * n_comp;
    float v = -INFINITY;
    for (uint32_t k = 0; k < top_k; k++) {
        if (topk[(uint64_t)t * top_k + k] == c) {
            v = 0.0f;
            break;
        }
    }
    mask[gid] = v;
}

extern "C" int ds4_gpu_embed_token_hc_tensor(ds4_gpu_tensor *out_hc, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n_vocab, uint32_t token, uint32_t n_embd, uint32_t n_hc) {
    (void)n_vocab;
    if (!out_hc || !model_map || weight_offset >= model_size) return 0;
    uint64_t weight_bytes = (uint64_t)n_vocab * n_embd * sizeof(uint16_t);
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "token_embd");
    if (!wptr) return 0;
    uint32_t n = n_embd * n_hc;
    if (g_token_graph_capturing &&
        g_token_graph_capture_dynamic_token &&
        g_token_graph_token_device) {
        embed_token_hc_dynamic_kernel<<<(n + 255) / 256, 256>>>(
                (float *)out_hc->ptr,
                (const unsigned short *)wptr,
                g_token_graph_token_device,
                n_embd,
                n_hc);
    } else {
        embed_token_hc_kernel<<<(n + 255) / 256, 256>>>(
                (float *)out_hc->ptr,
                (const unsigned short *)wptr,
                token,
                n_embd,
                n_hc);
    }
    return cuda_ok(cudaGetLastError(), "embed token launch");
}

extern "C" int ds4_gpu_embed_tokens_hc_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *tokens_t,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!out_hc || !tokens_t || !model_map ||
        weight_offset > model_size ||
        (uint64_t)n_vocab * n_embd * sizeof(uint16_t) > model_size - weight_offset ||
        tokens_t->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
        out_hc->bytes < (uint64_t)n_tokens * n_hc * n_embd * sizeof(float)) {
        return 0;
    }
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset,
                                            (uint64_t)n_vocab * n_embd * sizeof(uint16_t),
                                            "token_embd");
    if (!wptr) return 0;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    embed_tokens_hc_kernel<<<(n + 255) / 256, 256>>>(
        (float *)out_hc->ptr,
        (const int32_t *)tokens_t->ptr,
        (const __half *)wptr,
        n_vocab, n_tokens, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "embed tokens launch");
}

static int indexer_scores_launch(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale,
        uint32_t                causal) {
    if (!scores || !q || !weights || !index_comp ||
        n_comp == 0 || n_tokens == 0 || n_head == 0 || head_dim == 0 ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        weights->bytes < (uint64_t)n_tokens * n_head * sizeof(float) ||
        index_comp->bytes < (uint64_t)n_comp * head_dim * sizeof(float) ||
        scores->bytes < (uint64_t)n_tokens * n_comp * sizeof(float)) {
        return 0;
    }
    if (causal && ratio == 0) return 0;
    if (n_tokens == 1u && head_dim == 128u && n_head == 64u &&
        getenv("DS4_CUDA_NO_INDEXER_DIRECT_ONE") == NULL) {
        indexer_score_one_direct_kernel<<<n_comp, 128>>>((float *)scores->ptr,
                                                         (const float *)q->ptr,
                                                         (const float *)weights->ptr,
                                                         (const float *)index_comp->ptr,
                                                         n_comp, pos0, ratio,
                                                         scale, causal ? 1 : 0);
        return cuda_ok(cudaGetLastError(), "indexer score one direct launch");
    }
    if (!g_quality_mode && head_dim == 128u && n_head == 64u &&
        getenv("DS4_CUDA_NO_INDEXER_WMMA") == NULL) {
        if (getenv("DS4_CUDA_NO_INDEXER_WMMA128") == NULL) {
            dim3 grid((n_comp + 127u) / 128u, (n_tokens + 15u) / 16u, 1);
            indexer_scores_wmma128_kernel<<<grid, 256>>>((float *)scores->ptr,
                                                         (const float *)q->ptr,
                                                         (const float *)weights->ptr,
                                                         (const float *)index_comp->ptr,
                                                         n_comp, n_tokens, pos0, n_head,
                                                         head_dim, ratio, scale, causal ? 1 : 0);
            return cuda_ok(cudaGetLastError(), "indexer scores wmma128 launch");
        } else if (getenv("DS4_CUDA_NO_INDEXER_WMMA64") == NULL) {
            dim3 grid((n_comp + 63u) / 64u, (n_tokens + 15u) / 16u, 1);
            indexer_scores_wmma64_kernel<<<grid, 128>>>((float *)scores->ptr,
                                                        (const float *)q->ptr,
                                                        (const float *)weights->ptr,
                                                        (const float *)index_comp->ptr,
                                                        n_comp, n_tokens, pos0, n_head,
                                                        head_dim, ratio, scale, causal ? 1 : 0);
            return cuda_ok(cudaGetLastError(), "indexer scores wmma64 launch");
        } else if (getenv("DS4_CUDA_NO_INDEXER_WMMA32") == NULL) {
            dim3 grid((n_comp + 31u) / 32u, (n_tokens + 15u) / 16u, 1);
            indexer_scores_wmma32_kernel<<<grid, 64>>>((float *)scores->ptr,
                                                       (const float *)q->ptr,
                                                       (const float *)weights->ptr,
                                                       (const float *)index_comp->ptr,
                                                       n_comp, n_tokens, pos0, n_head,
                                                       head_dim, ratio, scale, causal ? 1 : 0);
            return cuda_ok(cudaGetLastError(), "indexer scores wmma32 launch");
        } else {
            dim3 grid((n_comp + 15u) / 16u, (n_tokens + 15u) / 16u, 1);
            indexer_scores_wmma_kernel<<<grid, 32>>>((float *)scores->ptr,
                                                     (const float *)q->ptr,
                                                     (const float *)weights->ptr,
                                                     (const float *)index_comp->ptr,
                                                     n_comp, n_tokens, pos0, n_head,
                                                     head_dim, ratio, scale, causal ? 1 : 0);
            return cuda_ok(cudaGetLastError(), "indexer scores wmma launch");
        }
    }
    dim3 grid(n_comp, n_tokens, 1);
    indexer_scores_kernel<<<grid, 256>>>((float *)scores->ptr,
                                         (const float *)q->ptr,
                                         (const float *)weights->ptr,
                                         (const float *)index_comp->ptr,
                                         n_comp, n_tokens, pos0, n_head,
                                         head_dim, ratio, scale, causal ? 1 : 0);
    return cuda_ok(cudaGetLastError(), "indexer scores launch");
}

extern "C" int ds4_gpu_indexer_score_one_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_head,
        uint32_t                head_dim,
        float                   scale) {
    return indexer_scores_launch(scores, q, weights, index_comp, n_comp, 1, 0,
                                 n_head, head_dim, 1, scale, 0);
}

extern "C" int ds4_gpu_indexer_scores_prefill_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    cuda_nvtx_scope scope("ds4/prefill/indexer/score",
                          cuda_nvtx_payload(n_comp, n_tokens),
                          n_tokens >= 128u);
    return indexer_scores_launch(scores, q, weights, index_comp, n_comp, n_tokens, 0,
                                 n_head, head_dim, ratio, scale, 1);
}

extern "C" int ds4_gpu_indexer_scores_decode_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    return indexer_scores_launch(scores, q, weights, index_comp, n_comp, n_tokens, pos0,
                                 n_head, head_dim, ratio, scale, 1);
}

extern "C" int ds4_gpu_indexer_topk_tensor(
        ds4_gpu_tensor       *selected,
        const ds4_gpu_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k) {
    if (!selected || !scores || n_comp == 0 || n_tokens == 0 || top_k == 0 ||
        top_k > n_comp ||
        scores->bytes < (uint64_t)n_tokens * n_comp * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * top_k * sizeof(uint32_t)) {
        return 0;
    }
    cuda_nvtx_scope scope("ds4/prefill/indexer/topk",
                          cuda_nvtx_payload(n_comp, n_tokens),
                          n_tokens >= 128u && top_k == 512u);
    if (top_k == 1u && getenv("DS4_CUDA_NO_BATCHED_ARGMAX") == NULL) {
        /* DSpark K3 verifies three draft positions at once.  Falling through
         * to indexer_topk_kernel used one CUDA thread per 129280-wide row and
         * serialized a material part of the verifier.  The same exact argmax
         * reduction used by K1 naturally scales to one block per row. */
        if (!g_batched_argmax_notice) {
            g_batched_argmax_notice = 1;
            fprintf(stderr,
                    "ds4: CUDA batched vocabulary argmax enabled "
                    "(one 1024-thread block per row)\n");
        }
        argmax_kernel<<<n_tokens, 1024>>>((int32_t *)selected->ptr,
                                          (const float *)scores->ptr,
                                          n_comp);
        return cuda_ok(cudaGetLastError(), "batched argmax launch");
    }
    if (top_k == 512u && n_comp <= 1024u &&
        getenv("DS4_CUDA_NO_TOPK1024") == NULL) {
        indexer_topk_1024_kernel<<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                     (const float *)scores->ptr,
                                                     n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 1024 launch");
    }
    if (top_k == 512u && n_comp <= 2048u &&
        getenv("DS4_CUDA_NO_TOPK2048") == NULL) {
        indexer_topk_pow2_kernel<2048><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                           (const float *)scores->ptr,
                                                           n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 2048 launch");
    }
    if (top_k == 512u && n_comp <= 4096u &&
        getenv("DS4_CUDA_NO_TOPK2048") == NULL) {
        if (n_comp == 4096u) {
            using TopkCubSort = cub::BlockRadixSort<uint64_t, 512, 16>;
            const int smem = (int)sizeof(typename TopkCubSort::TempStorage);
            int dev = 0;
            int max_optin_smem = 0;
            cudaError_t attr_err = cudaGetDevice(&dev);
            if (attr_err == cudaSuccess) {
                attr_err = cudaDeviceGetAttribute(&max_optin_smem,
                                                  cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                                  dev);
            }
            if (attr_err == cudaSuccess && max_optin_smem >= smem) {
                attr_err = cudaFuncSetAttribute(indexer_topk_8192_cub_kernel,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                smem);
                if (attr_err == cudaSuccess) {
                    indexer_topk_8192_cub_kernel<<<n_tokens, 512, (size_t)smem>>>((uint32_t *)selected->ptr,
                                                                                 (const float *)scores->ptr,
                                                                                 n_comp, n_tokens, top_k);
                    return cuda_ok(cudaGetLastError(), "indexer topk 4096 cub launch");
                }
            }
        }
        indexer_topk_pow2_kernel<4096><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                           (const float *)scores->ptr,
                                                           n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 4096 launch");
    }
    if (top_k == 512u && n_comp <= 8192u &&
        getenv("DS4_CUDA_NO_TOPK2048") == NULL &&
        getenv("DS4_CUDA_NO_TOPK8192") == NULL) {
        if (n_comp > 4096u) {
            using TopkCubSort = cub::BlockRadixSort<uint64_t, 512, 16>;
            const int smem = (int)sizeof(typename TopkCubSort::TempStorage);
            int dev = 0;
            int max_optin_smem = 0;
            cudaError_t attr_err = cudaGetDevice(&dev);
            if (attr_err == cudaSuccess) {
                attr_err = cudaDeviceGetAttribute(&max_optin_smem,
                                                  cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                                  dev);
            }
            if (attr_err == cudaSuccess && max_optin_smem >= smem) {
                attr_err = cudaFuncSetAttribute(indexer_topk_8192_cub_kernel,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                smem);
                if (attr_err == cudaSuccess) {
                    indexer_topk_8192_cub_kernel<<<n_tokens, 512, (size_t)smem>>>((uint32_t *)selected->ptr,
                                                                                 (const float *)scores->ptr,
                                                                                 n_comp, n_tokens, top_k);
                    return cuda_ok(cudaGetLastError(), "indexer topk 8192 cub launch");
                }
            }
        }
        indexer_topk_pow2_u16_kernel<8192><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                               (const float *)scores->ptr,
                                                               n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 8192 launch");
    }
    /* Radix keeps memory bounded and scales well across the thousands of rows
     * in a prefill chunk.  A decode/verifier batch has only 1..6 rows, where
     * one Radix CTA per row leaves most SMs idle.  Feed those rows to the
     * existing exact chunk tree instead: its independent 4096-column chunks
     * expose enough parallel work while preserving the same low-index tie
     * break and allocating only a small transient candidate list. */
    if (top_k == 512u && n_comp > 8192u && n_tokens > 16u) {
        static int radix_notice = 0;
        if (!radix_notice) {
            radix_notice = 1;
            fprintf(stderr,
                    "ds4: CUDA exact radix Top-512 enabled "
                    "(bounded shared scratch, low-index tie break)\n");
        }
        return cuda_ok(ds4_topk_radix_exact_512(
                           (uint32_t *)selected->ptr,
                           (const float *)scores->ptr,
                           n_comp,
                           n_tokens,
                           0),
                       "indexer exact radix topk launch");
    }
    if (top_k == 512u && getenv("DS4_CUDA_NO_TOPK2048") == NULL &&
        getenv("DS4_CUDA_NO_TOPK_CHUNKED") == NULL) {
        static int small_batch_notice = 0;
        if (n_comp > 8192u && n_tokens <= 16u && !small_batch_notice) {
            small_batch_notice = 1;
            fprintf(stderr,
                    "ds4: CUDA exact parallel Top-512 enabled for small batches "
                    "(4096-column chunk tree, low-index tie break)\n");
        }
        const uint32_t chunk_n = 4096u;
        const uint32_t n_chunks = (n_comp + chunk_n - 1u) / chunk_n;
        const uint32_t candidate_stride = n_chunks * top_k;
        uint32_t n_sets = n_chunks;
        uint64_t scratch_u32_per_token = candidate_stride;
        while (n_sets > DS4_CUDA_TOPK_MERGE_GROUP) {
            n_sets = (n_sets + DS4_CUDA_TOPK_MERGE_GROUP - 1u) / DS4_CUDA_TOPK_MERGE_GROUP;
            scratch_u32_per_token += (uint64_t)n_sets * top_k;
        }
        if (scratch_u32_per_token > UINT64_MAX / n_tokens / sizeof(uint32_t)) return 0;
        const uint64_t tmp_bytes = (uint64_t)n_tokens * scratch_u32_per_token * sizeof(uint32_t);
        uint32_t *scratch = (uint32_t *)cuda_tmp_alloc(tmp_bytes, "indexer topk tree");
        if (!scratch) return 0;

        uint32_t *cur = scratch;
        n_sets = n_chunks;
        uint32_t cur_stride = candidate_stride;
        dim3 grid_chunks(n_tokens, n_chunks, 1);
        indexer_topk_chunk_pow2_kernel<4096><<<grid_chunks, 1024>>>(cur,
                                                                    (const float *)scores->ptr,
                                                                    n_comp,
                                                                    n_tokens,
                                                                    top_k,
                                                                    candidate_stride);
        if (!cuda_ok(cudaGetLastError(), "indexer topk chunk launch")) return 0;

        while (n_sets > DS4_CUDA_TOPK_MERGE_GROUP) {
            const uint32_t next_sets = (n_sets + DS4_CUDA_TOPK_MERGE_GROUP - 1u) / DS4_CUDA_TOPK_MERGE_GROUP;
            const uint32_t next_stride = next_sets * top_k;
            uint32_t *next = cur + (uint64_t)n_tokens * cur_stride;
            dim3 grid_merge(n_tokens, next_sets, 1);
            indexer_topk_tree_merge_pow2_kernel<4096><<<grid_merge, 1024>>>(
                    next,
                    cur,
                    (const float *)scores->ptr,
                    n_comp,
                    n_tokens,
                    top_k,
                    n_sets,
                    DS4_CUDA_TOPK_MERGE_GROUP,
                    cur_stride,
                    next_stride);
            if (!cuda_ok(cudaGetLastError(), "indexer topk tree merge launch")) return 0;
            cur = next;
            n_sets = next_sets;
            cur_stride = next_stride;
        }

        indexer_topk_merge_pow2_kernel<4096><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                                 cur,
                                                                 (const float *)scores->ptr,
                                                                 n_comp,
                                                                 n_tokens,
                                                                 top_k,
                                                                 n_sets * top_k,
                                                                 cur_stride);
        return cuda_ok(cudaGetLastError(), "indexer topk tree final launch");
    }
    indexer_topk_kernel<<<n_tokens, 1>>>((uint32_t *)selected->ptr,
                                         (const float *)scores->ptr,
                                         n_comp, n_tokens, top_k);
    return cuda_ok(cudaGetLastError(), "indexer topk launch");
}

extern "C" int ds4_gpu_indexer_topk_gvr_tensor(
        ds4_gpu_tensor       *selected,
        const ds4_gpu_tensor *scores,
        const ds4_gpu_tensor *previous,
        ds4_gpu_tensor       *fallback_mask,
        uint32_t                n_comp,
        uint32_t                n_tokens) {
    if (!selected || !scores || !previous || !fallback_mask ||
        n_comp < 512u || n_tokens == 0u ||
        scores->bytes < (uint64_t)n_tokens * n_comp * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * 512u * sizeof(uint32_t) ||
        previous->bytes < (uint64_t)n_tokens * 512u * sizeof(uint32_t) ||
        fallback_mask->bytes < n_tokens) {
        return 0;
    }
    cuda_nvtx_scope scope("ds4/decode/indexer/gvr_topk",
                          cuda_nvtx_payload(n_comp, n_tokens));
    static int notice;
    if (!notice) {
        notice = 1;
        fprintf(stderr,
                "ds4: CUDA Blackwell exact GVR Top-512 enabled "
                "(temporal hint, ballot-free collect, radix fallback)\n");
    }
    return cuda_ok(ds4_topk_gvr_exact_512(
                       (uint32_t *)selected->ptr,
                       (const float *)scores->ptr,
                       (const uint32_t *)previous->ptr,
                       (uint8_t *)fallback_mask->ptr,
                       n_comp,
                       n_tokens,
                       0),
                   "indexer exact GVR topk launch");
}

extern "C" int ds4_gpu_argmax_tensor(
        ds4_gpu_tensor       *out_idx,
        const ds4_gpu_tensor *logits,
        uint32_t                n_vocab) {
    if (!out_idx || !logits || n_vocab == 0 ||
        out_idx->bytes < sizeof(int32_t) ||
        logits->bytes < (uint64_t)n_vocab * sizeof(float)) {
        return 0;
    }
    argmax_kernel<<<1, 1024>>>((int32_t *)out_idx->ptr,
                               (const float *)logits->ptr,
                               n_vocab);
    return cuda_ok(cudaGetLastError(), "argmax launch");
}

extern "C" int ds4_gpu_sample_min_p_tensor(
        ds4_gpu_tensor       *out_idx,
        ds4_gpu_tensor       *out_probs,
        const ds4_gpu_tensor *logits,
        const ds4_gpu_tensor *uniform,
        uint32_t                n_vocab,
        float                   temperature,
        float                   min_p) {
    if (!out_idx || !out_probs || !logits || !uniform || n_vocab == 0 ||
        temperature <= 0.0f || min_p < 0.0f || min_p > 1.0f ||
        out_idx->bytes < sizeof(int32_t) ||
        out_probs->bytes < (uint64_t)n_vocab * sizeof(float) ||
        logits->bytes < (uint64_t)n_vocab * sizeof(float) ||
        uniform->bytes < sizeof(float)) {
        return 0;
    }
    sample_min_p_kernel<<<1, 1024>>>((int32_t *)out_idx->ptr,
                                     (float *)out_probs->ptr,
                                     (const float *)logits->ptr,
                                     (const float *)uniform->ptr,
                                     n_vocab,
                                     temperature,
                                     min_p);
    return cuda_ok(cudaGetLastError(), "DSpark min-p sampler launch");
}

extern "C" int ds4_gpu_dspark_rejection_verify_tensor(
        ds4_gpu_tensor       *out_tokens,
        ds4_gpu_tensor       *out_accept,
        const ds4_gpu_tensor *spec_logits,
        const ds4_gpu_tensor *draft_probs,
        const ds4_gpu_tensor *dspark_tokens,
        const ds4_gpu_tensor *accept_uniforms,
        const ds4_gpu_tensor *residual_uniforms,
        uint32_t                n_rows,
        uint32_t                n_vocab,
        float                   temperature,
        float                   min_p) {
    if (!out_tokens || !out_accept || !spec_logits || !draft_probs ||
        !dspark_tokens || !accept_uniforms || !residual_uniforms ||
        n_rows == 0 || n_rows > 5u || n_vocab == 0 ||
        temperature <= 0.0f || min_p < 0.0f || min_p > 1.0f ||
        out_tokens->bytes < (uint64_t)n_rows * sizeof(int32_t) ||
        out_accept->bytes < (uint64_t)n_rows * sizeof(int32_t) ||
        spec_logits->bytes < (uint64_t)(n_rows + 1u) * n_vocab * sizeof(float) ||
        draft_probs->bytes < (uint64_t)n_rows * n_vocab * sizeof(float) ||
        dspark_tokens->bytes < (uint64_t)(n_rows + 1u) * sizeof(int32_t) ||
        accept_uniforms->bytes < (uint64_t)n_rows * sizeof(float) ||
        residual_uniforms->bytes < (uint64_t)n_rows * sizeof(float)) {
        return 0;
    }
    dspark_rejection_verify_kernel<<<n_rows, 1024>>>(
            (int32_t *)out_tokens->ptr,
            (int32_t *)out_accept->ptr,
            (const float *)spec_logits->ptr,
            (const float *)draft_probs->ptr,
            (const int32_t *)dspark_tokens->ptr,
            (const float *)accept_uniforms->ptr,
            (const float *)residual_uniforms->ptr,
            n_rows,
            n_vocab,
            temperature,
            min_p);
    return cuda_ok(cudaGetLastError(), "DSpark rejection verify launch");
}

extern "C" int ds4_gpu_dsv4_topk_mask_tensor(
        ds4_gpu_tensor       *mask,
        const ds4_gpu_tensor *topk,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k) {
    if (!mask || !topk || n_comp == 0 || n_tokens == 0 || top_k == 0 ||
        mask->bytes < (uint64_t)n_tokens * n_comp * sizeof(float) ||
        topk->bytes < (uint64_t)n_tokens * top_k * sizeof(uint32_t)) {
        return 0;
    }
    uint64_t n = (uint64_t)n_tokens * n_comp;
    uint64_t nk = (uint64_t)n_tokens * top_k;
    uint64_t blocks = ((n > nk ? n : nk) + 255) / 256;
    topk_mask_kernel<<<blocks, 256>>>((float *)mask->ptr,
                                      (const uint32_t *)topk->ptr,
                                      n_comp, n_tokens, top_k);
    return cuda_ok(cudaGetLastError(), "topk mask launch");
}

struct cuda_mtp_tc_counters {
    uint64_t q8_gemms;
    uint64_t f16_gemms;
    uint64_t padded_gemms;
    uint64_t autotune_gemms;
    uint64_t fallbacks;
    uint64_t last_report_gemms;
    int notice_printed;
};

static cuda_mtp_tc_counters g_mtp_tc_counters;

static int cuda_tiny_tc_requested(void) {
    return getenv("DS4_CUDA_MTP_TENSOR_CORES") != NULL ||
           getenv("DS4_CUDA_DSPARK_TENSOR_CORES") != NULL ||
           getenv("DS4_CUDA_TINY_TENSOR_CORES") != NULL;
}

static int cuda_mtp_tc_enabled(uint64_t n_tok) {
    return !g_quality_mode && g_cublas_ready && n_tok > 1u && n_tok <= 16u &&
           cuda_tiny_tc_requested();
}

static int cuda_q8_tiny_tc_enabled(uint64_t n_tok) {
    if (!cuda_mtp_tc_enabled(n_tok)) return 0;
    if (getenv("DS4_CUDA_MTP_TENSOR_CORES") != NULL ||
        getenv("DS4_CUDA_TINY_TENSOR_CORES") != NULL ||
        getenv("DS4_CUDA_DSPARK_TENSOR_CORES_Q8") != NULL) {
        return 1;
    }
    /* On GB10 DSpark K4, the native Q8 tiny-batch reuse kernel consistently
     * beats the generic cuBLAS path: Tensor Core would first expand/cache Q8
     * weights and convert F32 activations to F16 for every GEMM.  Keep Q8 TC
     * as an explicit experiment instead of hijacking the measured fast path. */
    return 0;
}

static uint32_t cuda_mtp_tc_pad_tokens(uint64_t n_tok) {
    uint64_t pad = 8u;
    const char *env = getenv("DS4_CUDA_DSPARK_TC_PAD_N");
    if (!env || !env[0]) env = getenv("DS4_CUDA_MTP_TC_PAD_N");
    if (!env || !env[0]) env = getenv("DS4_CUDA_TINY_TC_PAD_N");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env && *end == '\0') pad = (uint64_t)v;
    }
    if (pad == 0 || pad < n_tok) pad = n_tok;
    if (pad > 32u) pad = 32u;
    return (uint32_t)pad;
}

static int cuda_mtp_tc_autotune_enabled(void) {
#if defined(CUBLAS_VER_MAJOR) && CUBLAS_VER_MAJOR >= 13
    /* cuBLAS autotune is useful during the uncaptured warm-up, but algorithm
     * search itself is not a stable CUDA Graph capture operation. */
    return !g_mtp_graph_capturing &&
           (getenv("DS4_CUDA_MTP_TC_AUTOTUNE") != NULL ||
            getenv("DS4_CUDA_DSPARK_TC_AUTOTUNE") != NULL ||
            getenv("DS4_CUDA_TINY_TC_AUTOTUNE") != NULL);
#else
    return 0;
#endif
}

static void cuda_mtp_tc_notice(uint32_t pad_n, int autotune) {
    if (g_mtp_tc_counters.notice_printed) return;
    g_mtp_tc_counters.notice_printed = 1;
    fprintf(stderr,
            "ds4: CUDA tiny-batch Tensor Core enabled "
            "(fp16 inputs, fp32 accumulate, pad_n=%u, autotune=%s)\n",
            pad_n,
            autotune ? "yes" : "no");
}

static void cuda_mtp_tc_maybe_report(void) {
    if (getenv("DS4_CUDA_MTP_TC_VERBOSE") == NULL &&
        getenv("DS4_CUDA_DSPARK_TC_VERBOSE") == NULL &&
        getenv("DS4_CUDA_TINY_TC_VERBOSE") == NULL) return;
    const uint64_t total = g_mtp_tc_counters.q8_gemms +
                           g_mtp_tc_counters.f16_gemms;
    if (total == 0) return;
    if (g_mtp_tc_counters.last_report_gemms != 0 &&
        total / 1000u == g_mtp_tc_counters.last_report_gemms / 1000u) {
        return;
    }
    g_mtp_tc_counters.last_report_gemms = total;
    fprintf(stderr,
            "ds4: CUDA tiny-batch Tensor Core gemms=%llu q8=%llu f16=%llu "
            "padded=%llu autotune=%llu fallbacks=%llu\n",
            (unsigned long long)total,
            (unsigned long long)g_mtp_tc_counters.q8_gemms,
            (unsigned long long)g_mtp_tc_counters.f16_gemms,
            (unsigned long long)g_mtp_tc_counters.padded_gemms,
            (unsigned long long)g_mtp_tc_counters.autotune_gemms,
            (unsigned long long)g_mtp_tc_counters.fallbacks);
}

static cublasStatus_t cuda_mtp_tc_gemm_launch(
        float *out,
        const __half *w,
        const __half *xh,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t gemm_n,
        int autotune) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
#if defined(CUBLAS_VER_MAJOR) && CUBLAS_VER_MAJOR >= 13
    if (autotune) algo = CUBLAS_GEMM_AUTOTUNE;
#else
    (void)autotune;
#endif
    cublasStatus_t st = cublasGemmEx(g_cublas,
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     (int)out_dim,
                                     (int)gemm_n,
                                     (int)in_dim,
                                     &alpha,
                                     w,
                                     CUDA_R_16F,
                                     (int)in_dim,
                                     xh,
                                     CUDA_R_16F,
                                     (int)in_dim,
                                     &beta,
                                     out,
                                     CUDA_R_32F,
                                     (int)out_dim,
                                     CUDA_R_32F,
                                     algo);
#if defined(CUBLAS_VER_MAJOR) && CUBLAS_VER_MAJOR >= 13
    if (st != CUBLAS_STATUS_SUCCESS && autotune) {
        st = cublasGemmEx(g_cublas,
                          CUBLAS_OP_T,
                          CUBLAS_OP_N,
                          (int)out_dim,
                          (int)gemm_n,
                          (int)in_dim,
                          &alpha,
                          w,
                          CUDA_R_16F,
                          (int)in_dim,
                          xh,
                          CUDA_R_16F,
                          (int)in_dim,
                          &beta,
                          out,
                          CUDA_R_32F,
                          (int)out_dim,
                          CUDA_R_32F,
                          CUBLAS_GEMM_DEFAULT);
    }
#endif
    return st;
}

/* Tiny speculative batches have an awkward N=2/4 shape for Tensor Cores.
 * Optionally pad N with zero activation rows, let cuBLAS choose a mixed-FP16
 * kernel, and copy only the real result columns back.  Each output column is
 * independent, so zero padding cannot affect the requested rows. */
static int cuda_mtp_tc_f16_gemm(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        int q8_source) {
    if (!out || !w || !x || !cuda_mtp_tc_enabled(n_tok)) return 0;

    const uint32_t gemm_n = cuda_mtp_tc_pad_tokens(n_tok);
    if (gemm_n < n_tok || in_dim > (uint64_t)INT_MAX ||
        out_dim > (uint64_t)INT_MAX) {
        return 0;
    }
    if (in_dim > UINT64_MAX / gemm_n / sizeof(__half) ||
        out_dim > UINT64_MAX / gemm_n / sizeof(float)) {
        return 0;
    }

    const uint64_t xh_bytes = (uint64_t)gemm_n * in_dim * sizeof(__half);
    const uint64_t out_offset = (xh_bytes + 255u) & ~255ull;
    const uint64_t padded_out_bytes = gemm_n > n_tok ?
        (uint64_t)gemm_n * out_dim * sizeof(float) : 0u;
    if (out_offset > UINT64_MAX - padded_out_bytes) return 0;
    unsigned char *scratch = (unsigned char *)cuda_mtp_tc_scratch_alloc(
            out_offset + padded_out_bytes);
    if (!scratch) return 0;
    __half *xh = (__half *)scratch;
    float *gemm_out = gemm_n > n_tok ? (float *)(scratch + out_offset) : out;

    cudaStream_t stream = NULL;
    if (cublasGetStream(g_cublas, &stream) != CUBLAS_STATUS_SUCCESS) return 0;
    if (gemm_n > n_tok) {
        if (!cuda_ok(cudaMemsetAsync(xh, 0, (size_t)xh_bytes, stream),
                     "MTP Tensor Core activation pad")) {
            return 0;
        }
    }
    const uint64_t x_count = n_tok * in_dim;
    f32_to_f16_kernel<<<(x_count + 255u) / 256u, 256, 0, stream>>>(
            xh, x, x_count);
    if (!cuda_ok(cudaGetLastError(), "MTP Tensor Core activation convert")) return 0;

    const int autotune = cuda_mtp_tc_autotune_enabled();
    const cublasStatus_t st = cuda_mtp_tc_gemm_launch(gemm_out,
                                                       w,
                                                       xh,
                                                       in_dim,
                                                       out_dim,
                                                       gemm_n,
                                                       autotune);
    if (st != CUBLAS_STATUS_SUCCESS) return 0;

    if (gemm_n > n_tok) {
        const uint64_t real_bytes = n_tok * out_dim * sizeof(float);
        if (!cuda_ok(cudaMemcpyAsync(out, gemm_out, (size_t)real_bytes,
                                     cudaMemcpyDeviceToDevice, stream),
                     "MTP Tensor Core result trim")) {
            return 0;
        }
        g_mtp_tc_counters.padded_gemms++;
    }
    if (q8_source) g_mtp_tc_counters.q8_gemms++;
    else g_mtp_tc_counters.f16_gemms++;
    if (autotune) g_mtp_tc_counters.autotune_gemms++;
    cuda_mtp_tc_notice(gemm_n, autotune);
    cuda_mtp_tc_maybe_report();
    return 1;
}

/* Gate/up, q/kv and compressor pairs consume the same activation matrix.
 * Convert it once and issue both Tensor Core GEMMs from the same FP16 tile.
 * This removes one conversion kernel (and one padded memset) per pair while
 * retaining separate FP32 outputs and the ordinary pair path as fallback. */
static int cuda_mtp_tc_f16_gemm_pair(
        float *out0,
        float *out1,
        const __half *w0,
        const __half *w1,
        const float *x,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        uint64_t n_tok,
        int q8_source) {
    if (!out0 || !out1 || !w0 || !w1 || !x ||
        !cuda_mtp_tc_enabled(n_tok)) {
        return 0;
    }

    const uint32_t gemm_n = cuda_mtp_tc_pad_tokens(n_tok);
    if (gemm_n < n_tok || in_dim > (uint64_t)INT_MAX ||
        out0_dim > (uint64_t)INT_MAX || out1_dim > (uint64_t)INT_MAX) {
        return 0;
    }
    if (in_dim > UINT64_MAX / gemm_n / sizeof(__half) ||
        out0_dim > UINT64_MAX / gemm_n / sizeof(float) ||
        out1_dim > UINT64_MAX / gemm_n / sizeof(float)) {
        return 0;
    }

    const uint64_t xh_bytes = (uint64_t)gemm_n * in_dim * sizeof(__half);
    const bool padded = gemm_n > n_tok;
    const uint64_t out0_bytes = padded ?
        (uint64_t)gemm_n * out0_dim * sizeof(float) : 0u;
    const uint64_t out1_bytes = padded ?
        (uint64_t)gemm_n * out1_dim * sizeof(float) : 0u;
    const uint64_t out0_offset = (xh_bytes + 255u) & ~255ull;
    if (out0_offset > UINT64_MAX - out0_bytes) return 0;
    const uint64_t out0_end = out0_offset + out0_bytes;
    if (out0_end > UINT64_MAX - 255u) return 0;
    const uint64_t out1_offset =
        (out0_end + 255u) & ~255ull;
    if (out1_offset > UINT64_MAX - out1_bytes) return 0;
    const uint64_t scratch_bytes = padded ?
        out1_offset + out1_bytes : xh_bytes;
    unsigned char *scratch = (unsigned char *)cuda_mtp_tc_scratch_alloc(
            scratch_bytes);
    if (!scratch) return 0;
    __half *xh = (__half *)scratch;
    float *gemm_out0 = padded ? (float *)(scratch + out0_offset) : out0;
    float *gemm_out1 = padded ? (float *)(scratch + out1_offset) : out1;

    cudaStream_t stream = NULL;
    if (cublasGetStream(g_cublas, &stream) != CUBLAS_STATUS_SUCCESS) return 0;
    if (padded &&
        !cuda_ok(cudaMemsetAsync(xh, 0, (size_t)xh_bytes, stream),
                 "MTP Tensor Core paired activation pad")) {
        return 0;
    }
    const uint64_t x_count = n_tok * in_dim;
    f32_to_f16_kernel<<<(x_count + 255u) / 256u, 256, 0, stream>>>(
            xh, x, x_count);
    if (!cuda_ok(cudaGetLastError(),
                 "MTP Tensor Core paired activation convert")) {
        return 0;
    }

    const int autotune = cuda_mtp_tc_autotune_enabled();
    if (cuda_mtp_tc_gemm_launch(gemm_out0, w0, xh, in_dim, out0_dim,
                                gemm_n, autotune) != CUBLAS_STATUS_SUCCESS ||
        cuda_mtp_tc_gemm_launch(gemm_out1, w1, xh, in_dim, out1_dim,
                                gemm_n, autotune) != CUBLAS_STATUS_SUCCESS) {
        return 0;
    }

    if (padded) {
        const uint64_t real0_bytes = n_tok * out0_dim * sizeof(float);
        const uint64_t real1_bytes = n_tok * out1_dim * sizeof(float);
        if (!cuda_ok(cudaMemcpyAsync(out0, gemm_out0, (size_t)real0_bytes,
                                     cudaMemcpyDeviceToDevice, stream),
                     "MTP Tensor Core paired result0 trim") ||
            !cuda_ok(cudaMemcpyAsync(out1, gemm_out1, (size_t)real1_bytes,
                                     cudaMemcpyDeviceToDevice, stream),
                     "MTP Tensor Core paired result1 trim")) {
            return 0;
        }
        g_mtp_tc_counters.padded_gemms += 2u;
    }
    if (q8_source) g_mtp_tc_counters.q8_gemms += 2u;
    else g_mtp_tc_counters.f16_gemms += 2u;
    if (autotune) g_mtp_tc_counters.autotune_gemms += 2u;
    cuda_mtp_tc_notice(gemm_n, autotune);
    cuda_mtp_tc_maybe_report();
    return 1;
}

static int cuda_q8_batch_reuse_enabled(void) {
    static int enabled = -1;
    static int notice = 0;
    if (enabled < 0) {
        enabled = getenv("DS4_CUDA_Q8_BATCH_REUSE") != NULL ? 1 : 0;
    }
    if (enabled && !notice) {
        notice = 1;
        fprintf(stderr,
                "ds4: CUDA Q8 tiny-batch weight reuse enabled (2..6 rows)\n");
    }
    return enabled;
}

static int cuda_matmul_q8_0_tensor_labeled(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok, const char *label) {
    if (!out || !x || !model_map) return 0;
    uint64_t blocks = (in_dim + 31) / 32;
    if (weight_offset > model_size || out_dim > UINT64_MAX / (blocks * 34)) return 0;
    uint64_t weight_bytes = out_dim * blocks * 34;
    if (weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "q8_0");
    if (!wptr) return 0;
    if (g_cublas_ready && n_tok > 1) {
        const float *w_f32 = cuda_q8_f32_ptr(model_map, weight_offset, weight_bytes, in_dim, out_dim, label);
        if (w_f32) {
            const float alpha = 1.0f;
            const float beta = 0.0f;
            cublasStatus_t st = cublasSgemm(g_cublas,
                                            CUBLAS_OP_T,
                                            CUBLAS_OP_N,
                                            (int)out_dim,
                                            (int)n_tok,
                                            (int)in_dim,
                                            &alpha,
                                            w_f32,
                                            (int)in_dim,
                                            (const float *)x->ptr,
                                            (int)in_dim,
                                            &beta,
                                            (float *)out->ptr,
                                            (int)out_dim);
            return cublas_ok(st, "q8 fp32 matmul");
        }
        const __half *w_f16 = cuda_q8_f16_ptr(model_map, weight_offset, weight_bytes, in_dim, out_dim, label);
        if (w_f16) {
            if (cuda_q8_tiny_tc_enabled(n_tok)) {
                if (cuda_mtp_tc_f16_gemm((float *)out->ptr,
                                          w_f16,
                                          (const float *)x->ptr,
                                          in_dim,
                                          out_dim,
                                          n_tok,
                                          1)) {
                    return 1;
                }
                g_mtp_tc_counters.fallbacks++;
                cuda_mtp_tc_maybe_report();
            }
            const uint64_t xh_count = n_tok * in_dim;
            __half *xh = (__half *)cuda_tmp_alloc(xh_count * sizeof(__half), "q8 f16 gemm activations");
            if (!xh) return 0;
            f32_to_f16_kernel<<<(xh_count + 255) / 256, 256>>>(xh, (const float *)x->ptr, xh_count);
            if (!cuda_ok(cudaGetLastError(), "q8 f16 activation convert launch")) return 0;
            const float alpha = 1.0f;
            const float beta = 0.0f;
            cublasStatus_t st = cublasGemmEx(g_cublas,
                                             CUBLAS_OP_T,
                                             CUBLAS_OP_N,
                                             (int)out_dim,
                                             (int)n_tok,
                                             (int)in_dim,
                                             &alpha,
                                             w_f16,
                                             CUDA_R_16F,
                                             (int)in_dim,
                                             xh,
                                             CUDA_R_16F,
                                             (int)in_dim,
                                             &beta,
                                             out->ptr,
                                             CUDA_R_32F,
                                             (int)out_dim,
                                             CUDA_R_32F,
                                             CUBLAS_GEMM_DEFAULT);
            if (st == CUBLAS_STATUS_SUCCESS) return 1;
            fprintf(stderr, "ds4: cuBLAS q8 f16 matmul failed: status %d\n", (int)st);
            cuda_q8_f16_cache_disable_after_failure("cuBLAS f16 matmul failure",
                                                    in_dim * out_dim * sizeof(__half));
            /* The F16 expansion cache is only an optimization.  If cuBLAS
             * rejects the cached path under memory pressure, retry the same
             * operation through the native Q8 kernels below. */
        } else if (cuda_q8_tiny_tc_enabled(n_tok)) {
            g_mtp_tc_counters.fallbacks++;
            cuda_mtp_tc_maybe_report();
        }
    }
    const uint64_t xq_bytes = n_tok * blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + n_tok * blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "q8_0 prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    if (!launch_quantize_q8_0_f32_rows(xq,
                                       xscale,
                                       (const float *)x->ptr,
                                       n_tok,
                                       in_dim,
                                       blocks,
                                       "matmul_q8_0 quantize launch")) return 0;
    if (n_tok == 1) {
        matmul_q8_0_preq_warp8_kernel<<<((unsigned)out_dim + 7u) / 8u, 256>>>(
                (float *)out->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                xq,
                xscale,
                in_dim,
                out_dim,
                blocks,
                use_dp4a);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0 warp launch");
    }
    if (getenv("DS4_CUDA_NO_Q8_BATCH_WARP") == NULL && blocks <= 32u) {
        dim3 bgrid(((unsigned)out_dim + 7u) / 8u, (unsigned)n_tok, 1);
        matmul_q8_0_preq_batch_warp8_kernel<<<bgrid, 256>>>(
                (float *)out->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                xq,
                xscale,
                in_dim,
                out_dim,
                n_tok,
                blocks,
                use_dp4a);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0 batch warp launch");
    }
    if (n_tok <= 6u && blocks > 32u && cuda_q8_batch_reuse_enabled()) {
        matmul_q8_0_preq_batch_reuse_kernel<<<(unsigned)out_dim, 256>>>(
                (float *)out->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                xq,
                xscale,
                in_dim,
                out_dim,
                (uint32_t)n_tok,
                blocks,
                use_dp4a);
        return cuda_ok(cudaGetLastError(),
                       "matmul_q8_0 tiny-batch reuse launch");
    }
    dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    matmul_q8_0_preq_kernel<<<grid, 256>>>((float *)out->ptr,
                                           reinterpret_cast<const unsigned char *>(wptr),
                                           xq,
                                           xscale,
                                           in_dim, out_dim, n_tok, blocks,
                                           use_dp4a);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0 launch");
}

extern "C" int ds4_gpu_matmul_q8_0_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    return cuda_matmul_q8_0_tensor_labeled(out, model_map, model_size, weight_offset,
                                           in_dim, out_dim, x, n_tok, "q8_0");
}

extern "C" int ds4_gpu_matmul_q8_0_pair_tensor(
        ds4_gpu_tensor *out0,
        ds4_gpu_tensor *out1,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight0_offset,
        uint64_t weight1_offset,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    if (!out0 || !out1 || !x || !model_map || in_dim == 0 || out0_dim == 0 || out1_dim == 0 || n_tok == 0) {
        return 0;
    }
    const uint64_t blocks = (in_dim + 31) / 32;
    if (weight0_offset > model_size || weight1_offset > model_size ||
        out0_dim > UINT64_MAX / (blocks * 34) ||
        out1_dim > UINT64_MAX / (blocks * 34)) {
        return 0;
    }
    const uint64_t weight0_bytes = out0_dim * blocks * 34;
    const uint64_t weight1_bytes = out1_dim * blocks * 34;
    if (weight0_bytes > model_size - weight0_offset ||
        weight1_bytes > model_size - weight1_offset ||
        n_tok > UINT64_MAX / in_dim / sizeof(float) ||
        n_tok > UINT64_MAX / out0_dim / sizeof(float) ||
        n_tok > UINT64_MAX / out1_dim / sizeof(float) ||
        x->bytes < n_tok * in_dim * sizeof(float) ||
        out0->bytes < n_tok * out0_dim * sizeof(float) ||
        out1->bytes < n_tok * out1_dim * sizeof(float)) {
        return 0;
    }

    if (n_tok != 1) {
        if (cuda_q8_tiny_tc_enabled(n_tok)) {
            const __half *w0_f16 = cuda_q8_f16_ptr(
                    model_map, weight0_offset, weight0_bytes,
                    in_dim, out0_dim, "q8_0_pair0");
            const __half *w1_f16 = cuda_q8_f16_ptr(
                    model_map, weight1_offset, weight1_bytes,
                    in_dim, out1_dim, "q8_0_pair1");
            if (w0_f16 && w1_f16 &&
                cuda_mtp_tc_f16_gemm_pair((float *)out0->ptr,
                                           (float *)out1->ptr,
                                           w0_f16,
                                           w1_f16,
                                           (const float *)x->ptr,
                                           in_dim,
                                           out0_dim,
                                           out1_dim,
                                           n_tok,
                                           1)) {
                return 1;
            }
            g_mtp_tc_counters.fallbacks++;
            cuda_mtp_tc_maybe_report();
        }
        return cuda_matmul_q8_0_tensor_labeled(out0, model_map, model_size,
                                               weight0_offset, in_dim,
                                               out0_dim, x, n_tok,
                                               "q8_0_pair0") &&
               cuda_matmul_q8_0_tensor_labeled(out1, model_map, model_size,
                                               weight1_offset, in_dim,
                                               out1_dim, x, n_tok,
                                               "q8_0_pair1");
    }
    const char *w0 = cuda_model_range_ptr(model_map, weight0_offset, weight0_bytes, "q8_0_pair0");
    const char *w1 = cuda_model_range_ptr(model_map, weight1_offset, weight1_bytes, "q8_0_pair1");
    if (!w0 || !w1) return 0;

    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "q8_0 pair prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    if (!launch_quantize_q8_0_f32_rows(xq,
                                       xscale,
                                       (const float *)x->ptr,
                                       1,
                                       in_dim,
                                       blocks,
                                       "matmul_q8_0 pair quantize launch")) return 0;
    const uint64_t max_out = out0_dim > out1_dim ? out0_dim : out1_dim;
    matmul_q8_0_pair_preq_warp8_kernel<<<((unsigned)max_out + 7u) / 8u, 256>>>(
            (float *)out0->ptr,
            (float *)out1->ptr,
            reinterpret_cast<const unsigned char *>(w0),
            reinterpret_cast<const unsigned char *>(w1),
            xq,
            xscale,
            in_dim,
            out0_dim,
            out1_dim,
            blocks,
            use_dp4a);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0 pair warp launch");
}

extern "C" int ds4_gpu_matmul_q8_0_f16_out_tensor(
        ds4_gpu_tensor *out_h,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    (void)out_h; (void)model_map; (void)model_size; (void)weight_offset;
    (void)in_dim; (void)out_dim; (void)x; (void)n_tok;
    return 0;
}

static int cuda_matmul_q8_0_hc_expand_tensor_labeled(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc,
        const char             *label) {
    if (!out_hc || !block_out || !x || !residual_hc || !split || !model_map ||
        in_dim == 0 || out_dim == 0 || n_embd == 0 || n_hc == 0 ||
        out_dim != (uint64_t)n_embd) {
        return 0;
    }
    const uint64_t blocks = (in_dim + 31) / 32;
    if (weight_offset > model_size || out_dim > UINT64_MAX / (blocks * 34)) return 0;
    const uint64_t weight_bytes = out_dim * blocks * 34;
    const uint64_t hc_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    const uint64_t split_bytes = (uint64_t)(2u * n_hc + n_hc * n_hc) * sizeof(float);
    if (weight_bytes > model_size - weight_offset ||
        x->bytes < in_dim * sizeof(float) ||
        block_out->bytes < out_dim * sizeof(float) ||
        residual_hc->bytes < hc_bytes ||
        split->bytes < split_bytes ||
        out_hc->bytes < hc_bytes ||
        (block_add && block_add->bytes < out_dim * sizeof(float))) {
        return 0;
    }
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, label ? label : "q8_0_hc_expand");
    if (!wptr) return 0;

    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "q8_0 hc expand prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    if (!launch_quantize_q8_0_f32_rows(xq,
                                       xscale,
                                       (const float *)x->ptr,
                                       1,
                                       in_dim,
                                       blocks,
                                       "matmul_q8_0_hc_expand quantize launch")) return 0;
    matmul_q8_0_hc_expand_preq_warp8_kernel<<<((unsigned)out_dim + 7u) / 8u, 256>>>(
            (float *)out_hc->ptr,
            (float *)block_out->ptr,
            block_add ? (const float *)block_add->ptr : (const float *)block_out->ptr,
            (const float *)residual_hc->ptr,
            (const float *)split->ptr,
            reinterpret_cast<const unsigned char *>(wptr),
            xq,
            xscale,
            in_dim,
            out_dim,
            n_embd,
            n_hc,
            blocks,
            block_add ? 1 : 0,
            use_dp4a);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0_hc_expand launch");
}

static int cuda_coalesced_f16_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        enabled = getenv("DS4_CUDA_COALESCED_F16_MATMUL") != NULL ? 1 : 0;
        if (enabled) {
            fprintf(stderr,
                    "ds4: CUDA coalesced warp8 F16 decode matmul enabled\n");
        }
    }
    return enabled;
}

extern "C" int ds4_gpu_matmul_f16_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !model_map) return 0;
    if (weight_offset > model_size || out_dim > UINT64_MAX / in_dim) return 0;
    uint64_t weight_bytes = out_dim * in_dim * sizeof(uint16_t);
    if (weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "f16");
    if (!wptr) return 0;
    const __half *w = (const __half *)wptr;
    const int serial_f16 = getenv("DS4_CUDA_SERIAL_F16_MATMUL") != NULL;
    const int router_shape = in_dim == 4096u && out_dim == 256u && n_tok == 1u;
    const int serial_router =
        !serial_f16 &&
        router_shape &&
        getenv("DS4_CUDA_SERIAL_ROUTER") != NULL;
    const int coalesced_f16 =
        !serial_f16 &&
        !serial_router &&
        n_tok == 1u &&
        cuda_coalesced_f16_enabled();
    const int ordered_router =
        !serial_f16 &&
        !serial_router &&
        !coalesced_f16 &&
        n_tok == 1u &&
        getenv("DS4_CUDA_NO_ORDERED_F16_MATMUL") == NULL;
    if (!serial_f16 && g_cublas_ready && n_tok > 1) {
        if (cuda_mtp_tc_enabled(n_tok)) {
            if (cuda_mtp_tc_f16_gemm((float *)out->ptr,
                                      w,
                                      (const float *)x->ptr,
                                      in_dim,
                                      out_dim,
                                      n_tok,
                                      0)) {
                return 1;
            }
            g_mtp_tc_counters.fallbacks++;
            cuda_mtp_tc_maybe_report();
        }
        const uint64_t xh_count = n_tok * in_dim;
        __half *xh = (__half *)cuda_tmp_alloc(xh_count * sizeof(__half), "f16 gemm activations");
        if (!xh) return 0;
        f32_to_f16_kernel<<<(xh_count + 255) / 256, 256>>>(xh, (const float *)x->ptr, xh_count);
        if (!cuda_ok(cudaGetLastError(), "f16 activation convert launch")) return 0;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t st = cublasGemmEx(g_cublas,
                                         CUBLAS_OP_T,
                                         CUBLAS_OP_N,
                                         (int)out_dim,
                                         (int)n_tok,
                                         (int)in_dim,
                                         &alpha,
                                         w,
                                         CUDA_R_16F,
                                         (int)in_dim,
                                         xh,
                                         CUDA_R_16F,
                                         (int)in_dim,
                                         &beta,
                                         out->ptr,
                                         CUDA_R_32F,
                                         (int)out_dim,
                                         CUDA_R_32F,
                                         CUBLAS_GEMM_DEFAULT);
        return cublas_ok(st, "f16 matmul");
    }
    dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    if (serial_f16 || serial_router) {
        matmul_f16_serial_kernel<<<grid, 1>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
        return cuda_ok(cudaGetLastError(), serial_router ? "matmul_f16_router_serial launch" : "matmul_f16_serial launch");
    }
    if (coalesced_f16) {
        dim3 cgrid(((unsigned)out_dim + 7u) / 8u, (unsigned)n_tok, 1);
        matmul_f16_coalesced_warp8_kernel<<<cgrid, 256>>>(
                (float *)out->ptr, w, (const float *)x->ptr,
                in_dim, out_dim, n_tok);
        return cuda_ok(cudaGetLastError(),
                       "matmul_f16_coalesced_warp8 launch");
    }
    if (ordered_router) {
        matmul_f16_ordered_chunks_kernel<<<grid, 32>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
        return cuda_ok(cudaGetLastError(), "matmul_f16_ordered_chunks launch");
    }
    matmul_f16_kernel<<<grid, 256>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
    return cuda_ok(cudaGetLastError(), "matmul_f16 launch");
}

extern "C" int ds4_gpu_matmul_f16_f16_input_tensor(
        ds4_gpu_tensor *out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x_h,
        uint64_t n_tok) {
    if (!out || !x_h || !model_map || !g_cublas_ready || n_tok < 128u ||
        in_dim == 0u || out_dim == 0u ||
        in_dim > (uint64_t)INT_MAX || out_dim > (uint64_t)INT_MAX ||
        n_tok > (uint64_t)UINT32_MAX ||
        weight_offset > model_size || out_dim > UINT64_MAX / in_dim ||
        n_tok > UINT64_MAX / in_dim / sizeof(__half) ||
        n_tok > UINT64_MAX / out_dim / sizeof(float)) {
        return 0;
    }
    const uint64_t weight_bytes = out_dim * in_dim * sizeof(__half);
    if (weight_bytes > model_size - weight_offset ||
        x_h->bytes < n_tok * in_dim * sizeof(__half) ||
        out->bytes < n_tok * out_dim * sizeof(float)) {
        return 0;
    }
    const __half *w = (const __half *)cuda_model_range_ptr(
            model_map, weight_offset, weight_bytes, "f16_half_input");
    if (!w) return 0;
    return cuda_mtp_tc_gemm_launch((float *)out->ptr,
                                    w,
                                    (const __half *)x_h->ptr,
                                    in_dim,
                                    out_dim,
                                    (uint32_t)n_tok,
                                    0) == CUBLAS_STATUS_SUCCESS;
}

extern "C" int ds4_gpu_matmul_f16_pair_tensor(
        ds4_gpu_tensor *out0,
        ds4_gpu_tensor *out1,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight0_offset,
        uint64_t weight1_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    if (!out0 || !out1 || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0) {
        return 0;
    }
    const bool pair_disabled =
        getenv("DS4_CUDA_NO_F16_PAIR_MATMUL") != NULL ||
        getenv("DS4_CUDA_SERIAL_F16_MATMUL") != NULL ||
        getenv("DS4_CUDA_SERIAL_ROUTER") != NULL ||
        getenv("DS4_CUDA_NO_ORDERED_F16_MATMUL") != NULL;
    if (weight0_offset > model_size || weight1_offset > model_size ||
        out_dim > UINT64_MAX / in_dim) {
        return 0;
    }
    const uint64_t weight_bytes = out_dim * in_dim * sizeof(uint16_t);
    if (weight_bytes > model_size - weight0_offset ||
        weight_bytes > model_size - weight1_offset ||
        n_tok > UINT64_MAX / in_dim / sizeof(float) ||
        n_tok > UINT64_MAX / out_dim / sizeof(float) ||
        x->bytes < n_tok * in_dim * sizeof(float) ||
        out0->bytes < n_tok * out_dim * sizeof(float) ||
        out1->bytes < n_tok * out_dim * sizeof(float)) {
        return 0;
    }

    if (n_tok != 1) {
        if (!pair_disabled && cuda_mtp_tc_enabled(n_tok)) {
            const __half *w0 = (const __half *)cuda_model_range_ptr(
                    model_map, weight0_offset, weight_bytes, "f16_pair0");
            const __half *w1 = (const __half *)cuda_model_range_ptr(
                    model_map, weight1_offset, weight_bytes, "f16_pair1");
            if (w0 && w1 &&
                cuda_mtp_tc_f16_gemm_pair((float *)out0->ptr,
                                           (float *)out1->ptr,
                                           w0,
                                           w1,
                                           (const float *)x->ptr,
                                           in_dim,
                                           out_dim,
                                           out_dim,
                                           n_tok,
                                           0)) {
                return 1;
            }
            g_mtp_tc_counters.fallbacks++;
            cuda_mtp_tc_maybe_report();
        }
        return ds4_gpu_matmul_f16_tensor(out0, model_map, model_size,
                                         weight0_offset, in_dim, out_dim,
                                         x, n_tok) &&
               ds4_gpu_matmul_f16_tensor(out1, model_map, model_size,
                                         weight1_offset, in_dim, out_dim,
                                         x, n_tok);
    }
    if (pair_disabled) {
        return ds4_gpu_matmul_f16_tensor(out0, model_map, model_size,
                                         weight0_offset, in_dim, out_dim,
                                         x, n_tok) &&
               ds4_gpu_matmul_f16_tensor(out1, model_map, model_size,
                                         weight1_offset, in_dim, out_dim,
                                         x, n_tok);
    }
    const __half *w0 = (const __half *)cuda_model_range_ptr(model_map, weight0_offset, weight_bytes, "f16_pair0");
    const __half *w1 = (const __half *)cuda_model_range_ptr(model_map, weight1_offset, weight_bytes, "f16_pair1");
    if (!w0 || !w1) return 0;
    if (cuda_coalesced_f16_enabled()) {
        matmul_f16_pair_coalesced_warp8_kernel<<<
                ((unsigned)out_dim + 7u) / 8u, 256>>>(
                (float *)out0->ptr,
                (float *)out1->ptr,
                w0,
                w1,
                (const float *)x->ptr,
                in_dim,
                out_dim,
                out_dim);
        return cuda_ok(cudaGetLastError(),
                       "matmul_f16_pair_coalesced_warp8 launch");
    }
    matmul_f16_pair_ordered_chunks_kernel<<<(unsigned)out_dim, 32>>>(
        (float *)out0->ptr,
        (float *)out1->ptr,
        w0,
        w1,
        (const float *)x->ptr,
        in_dim,
        out_dim,
        out_dim);
    return cuda_ok(cudaGetLastError(), "matmul_f16_pair_ordered_chunks launch");
}

extern "C" int ds4_gpu_matmul_f32_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (weight_offset > model_size || out_dim > UINT64_MAX / in_dim) return 0;
    uint64_t weight_elems = out_dim * in_dim;
    if (weight_elems > UINT64_MAX / sizeof(float)) return 0;
    uint64_t weight_bytes = weight_elems * sizeof(float);
    if (weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "f32");
    if (!wptr) return 0;
    const float *w = (const float *)wptr;
    if (g_cublas_ready && n_tok > 1) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t st = cublasSgemm(g_cublas,
                                        CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        (int)out_dim,
                                        (int)n_tok,
                                        (int)in_dim,
                                        &alpha,
                                        w,
                                        (int)in_dim,
                                        (const float *)x->ptr,
                                        (int)in_dim,
                                        &beta,
                                        (float *)out->ptr,
                                        (int)out_dim);
        return cublas_ok(st, "f32 matmul");
    }
    dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    matmul_f32_kernel<<<grid, 256>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
    return cuda_ok(cudaGetLastError(), "matmul_f32 launch");
}

extern "C" int ds4_gpu_repeat_hc_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *row, uint32_t n_embd, uint32_t n_hc) {
    if (!out || !row || n_embd == 0 || n_hc == 0 ||
        row->bytes < (uint64_t)n_embd * sizeof(float) ||
        out->bytes < (uint64_t)n_embd * n_hc * sizeof(float)) {
        return 0;
    }
    uint64_t n = (uint64_t)n_embd * n_hc;
    repeat_hc_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)row->ptr, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "repeat_hc launch");
}

extern "C" int ds4_gpu_rms_norm_plain_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, uint32_t n, float eps) {
    if (!out || !x || out->bytes < (uint64_t)n * sizeof(float) ||
        x->bytes < (uint64_t)n * sizeof(float)) return 0;
    rms_norm_plain_kernel<<<1, 256>>>((float *)out->ptr, (const float *)x->ptr, n, 1, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_plain launch");
}
extern "C" int ds4_gpu_rms_norm_plain_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, uint32_t n, uint32_t rows, float eps) {
    if (!out || !x || out->bytes < (uint64_t)n * rows * sizeof(float) ||
        x->bytes < (uint64_t)n * rows * sizeof(float)) return 0;
    rms_norm_plain_kernel<<<rows, 256>>>((float *)out->ptr, (const float *)x->ptr, n, rows, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_plain launch");
}
extern "C" int ds4_gpu_rms_norm_plain_rows_f16_tensor(
        ds4_gpu_tensor *out_h,
        const ds4_gpu_tensor *x,
        uint32_t n,
        uint32_t rows,
        float eps) {
    if (!out_h || !x || rows < 128u ||
        out_h->bytes < (uint64_t)n * rows * sizeof(__half) ||
        x->bytes < (uint64_t)n * rows * sizeof(float)) {
        return 0;
    }
    rms_norm_plain_f16_kernel<<<rows, 256>>>(
            (__half *)out_h->ptr, (const float *)x->ptr, n, rows, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_plain_f16 launch");
}
extern "C" int ds4_gpu_rms_norm_weight_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n, float eps) {
    if (!out || !x || !model_map || weight_offset > model_size ||
        model_size - weight_offset < (uint64_t)n * sizeof(float) ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        x->bytes < (uint64_t)n * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, (uint64_t)n * sizeof(float), "rms_weight");
    if (!wptr) return 0;
    const float *w = (const float *)wptr;
    rms_norm_weight_kernel<<<1, 256>>>((float *)out->ptr, (const float *)x->ptr, w, n, 1, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_weight launch");
}
extern "C" int ds4_gpu_rms_norm_weight_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n, uint32_t rows, float eps) {
    if (!out || !x || !model_map || weight_offset > model_size ||
        model_size - weight_offset < (uint64_t)n * sizeof(float) ||
        out->bytes < (uint64_t)n * rows * sizeof(float) ||
        x->bytes < (uint64_t)n * rows * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, (uint64_t)n * sizeof(float), "rms_weight");
    if (!wptr) return 0;
    const float *w = (const float *)wptr;
    rms_norm_weight_kernel<<<rows, 256>>>((float *)out->ptr, (const float *)x->ptr, w, n, rows, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_weight launch");
}
extern "C" int ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        float                   eps) {
    if (getenv("DS4_CUDA_DISABLE_QKV_RMS_FUSED") == NULL) {
        if (!q_out || !q || !kv_out || !kv || !model_map ||
            q_weight_offset > model_size ||
            kv_weight_offset > model_size ||
            model_size - q_weight_offset < (uint64_t)q_n * sizeof(float) ||
            model_size - kv_weight_offset < (uint64_t)kv_n * sizeof(float) ||
            q_out->bytes < (uint64_t)q_n * rows * sizeof(float) ||
            q->bytes < (uint64_t)q_n * rows * sizeof(float) ||
            kv_out->bytes < (uint64_t)kv_n * rows * sizeof(float) ||
            kv->bytes < (uint64_t)kv_n * rows * sizeof(float)) {
            return 0;
        }
        const float *q_w = (const float *)cuda_model_range_ptr(model_map,
                q_weight_offset, (uint64_t)q_n * sizeof(float), "q_rms_weight");
        const float *kv_w = (const float *)cuda_model_range_ptr(model_map,
                kv_weight_offset, (uint64_t)kv_n * sizeof(float), "kv_rms_weight");
        if (!q_w || !kv_w) return 0;
        dim3 grid(rows, 2u, 1u);
        dsv4_qkv_rms_norm_rows_kernel<<<grid, 256>>>(
                (float *)q_out->ptr,
                (const float *)q->ptr,
                q_w,
                q_n,
                (float *)kv_out->ptr,
                (const float *)kv->ptr,
                kv_w,
                kv_n,
                rows,
                eps);
        return cuda_ok(cudaGetLastError(), "dsv4 qkv rms norm rows launch");
    }
    return ds4_gpu_rms_norm_weight_rows_tensor(q_out, q, model_map, model_size,
                                                 q_weight_offset, q_n, rows, eps) &&
           ds4_gpu_rms_norm_weight_rows_tensor(kv_out, kv, model_map, model_size,
                                                 kv_weight_offset, kv_n, rows, eps);
}
extern "C" int ds4_gpu_head_rms_norm_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, float eps) {
    if (!x || x->bytes < (uint64_t)n_tok * n_head * head_dim * sizeof(float)) return 0;
    head_rms_norm_kernel<<<n_tok * n_head, 256>>>((float *)x->ptr, n_tok, n_head, head_dim, eps);
    return cuda_ok(cudaGetLastError(), "head_rms_norm launch");
}
extern "C" int ds4_gpu_head_rms_norm_rope_tail_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow, float eps) {
    if (!x || n_rot > head_dim || (n_rot & 1u) ||
        x->bytes < (uint64_t)n_tok * n_head * head_dim * sizeof(float)) return 0;
    head_rms_norm_rope_tail_kernel<<<n_tok * n_head, 256>>>((float *)x->ptr, n_tok, n_head, head_dim, n_rot, pos0, n_ctx_orig, inverse ? 1 : 0, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, eps);
    return cuda_ok(cudaGetLastError(), "head_rms_norm_rope_tail launch");
}

extern "C" int ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *q_half,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint32_t n_tok,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t pos0,
        uint32_t n_ctx_orig,
        bool inverse,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        float eps) {
    (void)out; (void)q_half; (void)model_map; (void)model_size;
    (void)weight_offset; (void)in_dim; (void)out_dim; (void)x;
    (void)n_tok; (void)n_head; (void)head_dim; (void)n_rot; (void)pos0;
    (void)n_ctx_orig; (void)inverse; (void)freq_base; (void)freq_scale;
    (void)ext_factor; (void)attn_factor; (void)beta_fast; (void)beta_slow;
    (void)eps;
    return 0;
}

extern "C" int ds4_gpu_dsv4_fp8_kv_quantize_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot) {
    if (!x || n_rot > head_dim || x->bytes < (uint64_t)n_tok * head_dim * sizeof(float)) return 0;
    fp8_kv_quantize_kernel<<<n_tok, 64>>>((float *)x->ptr, n_tok, head_dim, n_rot);
    return cuda_ok(cudaGetLastError(), "fp8_kv_quantize launch");
}
extern "C" int ds4_gpu_dsv4_indexer_qat_tensor(ds4_gpu_tensor *x, uint32_t n_rows, uint32_t head_dim) {
    if (!x || n_rows == 0 || head_dim != 128u ||
        x->bytes < (uint64_t)n_rows * head_dim * sizeof(float)) {
        return 0;
    }
    indexer_hadamard_fp4_kernel<<<n_rows, 128>>>((float *)x->ptr, n_rows, head_dim);
    return cuda_ok(cudaGetLastError(), "indexer_hadamard_fp4 launch");
}
extern "C" int ds4_gpu_dsv4_indexer_pack_tensor(
        ds4_gpu_tensor       *packed,
        const ds4_gpu_tensor *src,
        uint32_t                n_rows) {
    if (!packed || !src || n_rows == 0u ||
        packed->bytes < (uint64_t)n_rows * DS4_INDEXER_FP4_ROW_BYTES ||
        src->bytes < (uint64_t)n_rows * DS4_INDEXER_DIM * sizeof(float)) {
        return 0;
    }
    return cuda_ok(ds4_indexer_sm121_pack(packed->ptr,
                                           (const float *)src->ptr,
                                           n_rows,
                                           0),
                   "indexer MXFP4 pack launch");
}
extern "C" int ds4_gpu_dsv4_indexer_unpack_tensor(
        ds4_gpu_tensor       *dst,
        const ds4_gpu_tensor *packed,
        uint32_t                n_rows) {
    if (!dst || !packed || n_rows == 0u ||
        dst->bytes < (uint64_t)n_rows * DS4_INDEXER_DIM * sizeof(float) ||
        packed->bytes < (uint64_t)n_rows * DS4_INDEXER_FP4_ROW_BYTES) {
        return 0;
    }
    return cuda_ok(ds4_indexer_sm121_unpack((float *)dst->ptr,
                                             packed->ptr,
                                             n_rows,
                                             0),
                   "indexer MXFP4 unpack launch");
}
extern "C" int ds4_gpu_indexer_scores_packed_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q_packed,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp_packed,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                ratio,
        float                   scale,
        uint32_t                causal) {
    if (!scores || !q_packed || !weights || !index_comp_packed ||
        n_comp == 0u || n_tokens == 0u || n_head == 0u ||
        (causal && ratio == 0u) ||
        scores->bytes < (uint64_t)n_tokens * n_comp * sizeof(float) ||
        q_packed->bytes < (uint64_t)n_tokens * n_head * DS4_INDEXER_FP4_ROW_BYTES ||
        weights->bytes < (uint64_t)n_tokens * n_head * sizeof(float) ||
        index_comp_packed->bytes < (uint64_t)n_comp * DS4_INDEXER_FP4_ROW_BYTES) {
        return 0;
    }
    cuda_nvtx_scope scope("ds4/prefill/indexer/score-mxfp4",
                          cuda_nvtx_payload(n_comp, n_tokens),
                          n_tokens >= 128u);
    static int notice;
    if (!notice) {
        notice = 1;
        if (ds4_indexer_sm121_has_native_mxfp4()) {
            fprintf(stderr,
                    "ds4: CUDA packed MXFP4 indexer scorer enabled "
                    "(68-byte rows, native block-scaled MMA; token-tile prefill "
                    "+ head-tile verifier on sm_121a)\n");
        } else {
            fprintf(stderr,
                    "ds4: CUDA packed MXFP4 indexer scorer enabled "
                    "(68-byte rows, scalar fallback; build sm_121a for native MMA)\n");
        }
    }
    return cuda_ok(ds4_indexer_sm121_scores(
                       (float *)scores->ptr,
                       q_packed->ptr,
                       (const float *)weights->ptr,
                       index_comp_packed->ptr,
                       n_comp, n_tokens, pos0, n_head, ratio,
                       scale, causal ? 1 : 0, 0),
                   "indexer MXFP4 scores launch");
}
extern "C" int ds4_gpu_rope_tail_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    if (!x || n_rot > head_dim || (n_rot & 1) || x->bytes < (uint64_t)n_tok * n_head * head_dim * sizeof(float)) return 0;
    uint32_t pairs = n_tok * n_head * (n_rot / 2);
    rope_tail_kernel<<<(pairs + 255) / 256, 256>>>((float *)x->ptr, n_tok, n_head, head_dim, n_rot, pos0, 1, n_ctx_orig, inverse ? 1 : 0, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    return cuda_ok(cudaGetLastError(), "rope_tail launch");
}
extern "C" int ds4_gpu_store_raw_kv_tensor(ds4_gpu_tensor *raw_cache, const ds4_gpu_tensor *kv, uint32_t raw_cap, uint32_t row, uint32_t head_dim);
extern "C" int ds4_gpu_kv_fp8_store_raw_tensor(
        ds4_gpu_tensor *kv,
        ds4_gpu_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          raw_row,
        uint32_t          head_dim,
        uint32_t          n_rot) {
    return ds4_gpu_dsv4_fp8_kv_quantize_tensor(kv, 1, head_dim, n_rot) &&
           ds4_gpu_store_raw_kv_tensor(raw_cache, kv, raw_cap, raw_row, head_dim);
}
extern "C" int ds4_gpu_store_raw_kv_tensor(ds4_gpu_tensor *raw_cache, const ds4_gpu_tensor *kv, uint32_t raw_cap, uint32_t row, uint32_t head_dim) {
    if (!raw_cache || !kv || raw_cap == 0 ||
        raw_cache->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        kv->bytes < (uint64_t)head_dim * sizeof(float)) return 0;
    store_raw_kv_batch_kernel<<<(head_dim + 255) / 256, 256>>>((float *)raw_cache->ptr, (const float *)kv->ptr, raw_cap, row, 1, head_dim);
    return cuda_ok(cudaGetLastError(), "store_raw_kv launch");
}
extern "C" int ds4_gpu_store_raw_kv_batch_tensor(ds4_gpu_tensor *raw_cache, const ds4_gpu_tensor *kv, uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens, uint32_t head_dim) {
    if (!raw_cache || !kv || raw_cap == 0 ||
        raw_cache->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float)) return 0;
    uint64_t n = (uint64_t)n_tokens * head_dim;
    store_raw_kv_batch_kernel<<<(n + 255) / 256, 256>>>((float *)raw_cache->ptr, (const float *)kv->ptr, raw_cap, pos0, n_tokens, head_dim);
    return cuda_ok(cudaGetLastError(), "store_raw_kv_batch launch");
}

extern "C" int ds4_gpu_ring_rows_save_tensor(
        ds4_gpu_tensor       *backup,
        const ds4_gpu_tensor *ring,
        uint32_t              ring_cap,
        uint32_t              pos0,
        uint32_t              n_rows,
        uint32_t              width) {
    const uint64_t bytes = (uint64_t)n_rows * width * sizeof(float);
    if (!backup || !ring || ring_cap == 0 || n_rows == 0 || n_rows > ring_cap ||
        width == 0 || backup->bytes < bytes ||
        ring->bytes < (uint64_t)ring_cap * width * sizeof(float)) return 0;
    const uint64_t n = (uint64_t)n_rows * width;
    ring_rows_save_kernel<<<(n + 255u) / 256u, 256>>>(
            (float *)backup->ptr, (const float *)ring->ptr,
            ring_cap, pos0, n_rows, width);
    return cuda_ok(cudaGetLastError(), "ring rows save launch");
}

extern "C" int ds4_gpu_ring_rows_restore_tensor(
        ds4_gpu_tensor       *ring,
        const ds4_gpu_tensor *backup,
        uint32_t              ring_cap,
        uint32_t              pos0,
        uint32_t              restore_from,
        uint32_t              n_rows,
        uint32_t              width) {
    const uint64_t bytes = (uint64_t)n_rows * width * sizeof(float);
    if (!ring || !backup || ring_cap == 0 || n_rows == 0 || n_rows > ring_cap ||
        restore_from > n_rows || width == 0 || backup->bytes < bytes ||
        ring->bytes < (uint64_t)ring_cap * width * sizeof(float)) return 0;
    if (restore_from == n_rows) return 1;
    const uint64_t n = (uint64_t)(n_rows - restore_from) * width;
    ring_rows_restore_kernel<<<(n + 255u) / 256u, 256>>>(
            (float *)ring->ptr, (const float *)backup->ptr,
            ring_cap, pos0, restore_from, n_rows, width);
    return cuda_ok(cudaGetLastError(), "ring rows restore launch");
}
extern "C" int ds4_gpu_compressor_store_batch_tensor(
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    if (!kv || !sc || !state_kv || !state_score || !model_map ||
        head_dim == 0 || ratio == 0 || n_tokens == 0 ||
        (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes) {
        return 0;
    }
    const char *ape = cuda_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
    if (!ape) return 0;
    uint64_t n = (uint64_t)n_tokens * width;
    compressor_store_kernel<<<(n + 255) / 256, 256>>>(
            (const float *)kv->ptr,
            (const float *)sc->ptr,
            (float *)state_kv->ptr,
            (float *)state_score->ptr,
            ape,
            0,
            ape_type,
            head_dim,
            ratio,
            pos0,
            n_tokens);
    return cuda_ok(cudaGetLastError(), "compressor store launch");
}

extern "C" int ds4_gpu_compressor_update_tensor(
        const ds4_gpu_tensor *kv_cur,
        const ds4_gpu_tensor *sc_cur,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        ds4_gpu_tensor       *comp_cache,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos,
        uint32_t                comp_row,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!kv_cur || !sc_cur || !state_kv || !state_score || !comp_cache ||
        !model_map || head_dim == 0 || ratio == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) || norm_type != 0u) {
        return 0;
    }
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t emit = ((pos + 1u) % ratio) == 0u ? 1u : 0u;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)(comp_row + (emit ? 1u : 0u)) * head_dim * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || norm_bytes > model_size - norm_offset ||
        kv_cur->bytes < kv_bytes || sc_cur->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        (emit && comp_cache->bytes < comp_bytes)) {
        return 0;
    }
    if (ratio == 4u && cuda_fused_compressor_update_enabled()) {
        const char *ape = cuda_model_range_ptr(model_map, ape_offset, ape_bytes,
                                                "compressor_ape_fused");
        const float *norm_w = (const float *)cuda_model_range_ptr(
                model_map, norm_offset, norm_bytes, "compressor_norm_fused");
        if (!ape || !norm_w) return 0;
        float *row = emit
            ? (float *)((char *)comp_cache->ptr +
                        (uint64_t)comp_row * head_dim * sizeof(float))
            : (float *)comp_cache->ptr;
        compressor_update_ratio4_decode_kernel<<<1, 256>>>(
                (const float *)kv_cur->ptr,
                (const float *)sc_cur->ptr,
                (float *)state_kv->ptr,
                (float *)state_score->ptr,
                row,
                ape,
                ape_type,
                norm_w,
                head_dim,
                pos,
                n_rot,
                n_ctx_orig,
                freq_base,
                freq_scale,
                ext_factor,
                attn_factor,
                beta_fast,
                beta_slow,
                rms_eps);
        return cuda_ok(cudaGetLastError(), "fused compressor ratio4 update launch");
    }
    if (!ds4_gpu_compressor_store_batch_tensor(kv_cur, sc_cur, state_kv, state_score,
                                                 model_map, model_size, ape_offset, ape_type,
                                                 head_dim, ratio, pos, 1)) {
        return 0;
    }
    if (!emit) return 1;
    ds4_gpu_tensor *comp_row_view = ds4_gpu_tensor_view(
            comp_cache,
            (uint64_t)comp_row * head_dim * sizeof(float),
            (uint64_t)head_dim * sizeof(float));
    if (!comp_row_view) return 0;
    compressor_update_pool_kernel<<<(head_dim + 255) / 256, 256>>>(
            (float *)comp_row_view->ptr,
            (const float *)state_kv->ptr,
            (const float *)state_score->ptr,
            head_dim,
            ratio);
    int ok = cuda_ok(cudaGetLastError(), "compressor update pool launch");
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(comp_row_view, comp_row_view,
                                                       model_map, model_size, norm_offset,
                                                       head_dim, 1, rms_eps);
    if (ok) ok = ds4_gpu_rope_tail_tensor(comp_row_view, 1, 1, head_dim, n_rot,
                                            pos + 1u - ratio, n_ctx_orig, false,
                                            freq_base, freq_scale, ext_factor, attn_factor,
                                            beta_fast, beta_slow);
    ds4_gpu_tensor_free(comp_row_view);
    if (ok && ratio == 4u) {
        uint64_t half = 4ull * width;
        compressor_shift_ratio4_kernel<<<(half + 255) / 256, 256>>>(
                (float *)state_kv->ptr, (float *)state_score->ptr, width);
        ok = cuda_ok(cudaGetLastError(), "compressor ratio4 shift launch");
    }
    return ok;
}
extern "C" int ds4_gpu_compressor_prefill_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        head_dim == 0 || ratio == 0 || n_tokens == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) || norm_type != 0u) {
        return 0;
    }

    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t n_comp = n_tokens / ratio;
    const uint32_t cutoff = n_comp * ratio;
    const uint32_t rem = n_tokens - cutoff;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);

    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || norm_bytes > model_size - norm_offset ||
        kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        (n_comp && comp_cache->bytes < comp_bytes)) {
        return 0;
    }
    const char *ape = cuda_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
    if (!ape) return 0;

    uint64_t state_n = (uint64_t)state_rows * width;
    if (!cuda_ok(cudaMemsetAsync(state_kv->ptr, 0, (size_t)(state_n * sizeof(float))),
                 "compressor state kv zero")) return 0;
    fill_f32_kernel<<<(state_n + 255) / 256, 256>>>((float *)state_score->ptr, state_n, -INFINITY);
    if (!cuda_ok(cudaGetLastError(), "compressor state score fill launch")) return 0;

    if (ratio == 4u) {
        if (cutoff >= ratio) {
            uint32_t prev_start = cutoff - ratio;
            uint64_t n = (uint64_t)ratio * width;
            compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
                    (float *)state_kv->ptr, (float *)state_score->ptr,
                    (const float *)kv->ptr, (const float *)sc->ptr,
                    ape, 0, ape_type, width, ratio, pos0,
                    prev_start, 0, ratio);
            if (!cuda_ok(cudaGetLastError(), "compressor prefill prev state launch")) return 0;
        }
        if (rem != 0) {
            uint64_t n = (uint64_t)rem * width;
            compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
                    (float *)state_kv->ptr, (float *)state_score->ptr,
                    (const float *)kv->ptr, (const float *)sc->ptr,
                    ape, 0, ape_type, width, ratio, pos0,
                    cutoff, ratio, rem);
            if (!cuda_ok(cudaGetLastError(), "compressor prefill rem state launch")) return 0;
        }
    } else if (rem != 0) {
        uint64_t n = (uint64_t)rem * width;
        compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
                (float *)state_kv->ptr, (float *)state_score->ptr,
                (const float *)kv->ptr, (const float *)sc->ptr,
                ape, 0, ape_type, width, ratio, pos0,
                cutoff, 0, rem);
        if (!cuda_ok(cudaGetLastError(), "compressor prefill rem state launch")) return 0;
    }
    if (n_comp != 0) {
        dim3 grid((head_dim + 255) / 256, n_comp, 1);
        compressor_prefill_pool_kernel<<<grid, 256>>>(
                (float *)comp_cache->ptr,
                (const float *)kv->ptr,
                (const float *)sc->ptr,
                (const float *)state_kv->ptr,
                (const float *)state_score->ptr,
                ape, 0, ape_type, head_dim, ratio, pos0, n_comp, 0);
        if (!cuda_ok(cudaGetLastError(), "compressor prefill pool launch")) return 0;
        if (!ds4_gpu_rms_norm_weight_rows_tensor(comp_cache, comp_cache,
                                                   model_map, model_size, norm_offset,
                                                   head_dim, n_comp, rms_eps)) return 0;
        if (n_rot != 0) {
            const uint32_t pairs = n_comp * (n_rot / 2u);
            rope_tail_kernel<<<(pairs + 255) / 256, 256>>>(
                    (float *)comp_cache->ptr, n_comp, 1, head_dim, n_rot,
                    pos0, ratio, n_ctx_orig, 0, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            if (!cuda_ok(cudaGetLastError(), "compressor prefill rope launch")) return 0;
        }
        if (quantize_fp8 && !ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim, n_rot)) return 0;
    }
    return 1;
}
extern "C" int ds4_gpu_compressor_prefill_ratio4_replay_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        head_dim == 0 || n_tokens == 0 || (n_tokens & 3u) != 0 || (pos0 & 3u) != 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) || norm_type != 0u) {
        return 0;
    }

    const uint32_t ratio = 4u;
    const uint32_t width = 2u * head_dim;
    const uint32_t state_rows = 8u;
    const uint32_t n_comp = n_tokens / ratio;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || norm_bytes > model_size - norm_offset ||
        kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        comp_cache->bytes < comp_bytes) {
        return 0;
    }
    const char *ape = cuda_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
    if (!ape) return 0;
    dim3 grid((head_dim + 255) / 256, n_comp, 1);
    compressor_prefill_pool_kernel<<<grid, 256>>>(
            (float *)comp_cache->ptr,
            (const float *)kv->ptr,
            (const float *)sc->ptr,
            (const float *)state_kv->ptr,
            (const float *)state_score->ptr,
            ape, 0, ape_type, head_dim, ratio, pos0, n_comp, 1);
    if (!cuda_ok(cudaGetLastError(), "compressor replay pool launch")) return 0;
    if (!ds4_gpu_rms_norm_weight_rows_tensor(comp_cache, comp_cache,
                                               model_map, model_size, norm_offset,
                                               head_dim, n_comp, rms_eps)) return 0;
    if (n_rot != 0) {
        const uint32_t pairs = n_comp * (n_rot / 2u);
        rope_tail_kernel<<<(pairs + 255) / 256, 256>>>(
                (float *)comp_cache->ptr, n_comp, 1, head_dim, n_rot,
                pos0, ratio, n_ctx_orig, 0, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        if (!cuda_ok(cudaGetLastError(), "compressor replay rope launch")) return 0;
    }
    if (quantize_fp8 && !ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim, n_rot)) return 0;

    uint64_t state_n = (uint64_t)state_rows * width;
    if (!cuda_ok(cudaMemsetAsync(state_kv->ptr, 0, (size_t)(state_n * sizeof(float))),
                 "compressor replay state kv zero")) return 0;
    fill_f32_kernel<<<(state_n + 255) / 256, 256>>>((float *)state_score->ptr, state_n, -INFINITY);
    if (!cuda_ok(cudaGetLastError(), "compressor replay state score fill launch")) return 0;
    uint32_t prev_start = n_tokens - ratio;
    uint64_t n = (uint64_t)ratio * width;
    compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
            (float *)state_kv->ptr, (float *)state_score->ptr,
            (const float *)kv->ptr, (const float *)sc->ptr,
            ape, 0, ape_type, width, ratio, pos0,
            prev_start, 0, ratio);
    return cuda_ok(cudaGetLastError(), "compressor replay state launch");
}
extern "C" int ds4_gpu_compressor_prefill_state_ratio4_tensor(
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv_tail,
        const ds4_gpu_tensor *sc_tail,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                pos0) {
    if (!state_kv || !state_score || !kv_tail || !sc_tail || !model_map ||
        head_dim == 0 || (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }
    const uint32_t ratio = 4u;
    const uint32_t width = 2u * head_dim;
    const uint32_t state_rows = 8u;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t tail_bytes = (uint64_t)ratio * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)ratio * width * elem_ape;
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        kv_tail->bytes < tail_bytes || sc_tail->bytes < tail_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes) {
        return 0;
    }
    const char *ape = cuda_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
    if (!ape) return 0;
    uint64_t state_n = (uint64_t)state_rows * width;
    if (!cuda_ok(cudaMemsetAsync(state_kv->ptr, 0, (size_t)(state_n * sizeof(float))),
                 "compressor state kv zero")) return 0;
    fill_f32_kernel<<<(state_n + 255) / 256, 256>>>((float *)state_score->ptr, state_n, -INFINITY);
    if (!cuda_ok(cudaGetLastError(), "compressor state score fill launch")) return 0;
    uint64_t n = (uint64_t)ratio * width;
    compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
            (float *)state_kv->ptr, (float *)state_score->ptr,
            (const float *)kv_tail->ptr, (const float *)sc_tail->ptr,
            ape, 0, ape_type, width, ratio, pos0,
            0, 0, ratio);
    return cuda_ok(cudaGetLastError(), "compressor state set launch");
}
extern "C" int ds4_gpu_attention_decode_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_comp,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_mask,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (comp_kv_f16 > 1u || !heads || !q || !raw_kv || !model_map ||
        n_raw == 0 || raw_cap < n_raw ||
        raw_start >= raw_cap || (n_comp != 0 && !comp_kv) || (use_mask && !comp_mask) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp * head_dim *
             (comp_kv_f16 ? sizeof(half) : sizeof(float))) ||
        (use_mask && comp_mask->bytes < (uint64_t)n_comp * sizeof(float))) {
        return 0;
    }
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    const uint32_t heads2_raw = n_raw > 256u ? 256u : n_raw;
    const uint32_t heads2_rows = heads2_raw + n_comp;
    if (!use_mask &&
        cuda_attn_heads2_use(heads2_rows, 1u, n_head, head_dim)) {
        if (!g_attn_heads2_dense_notice) {
            g_attn_heads2_dense_notice = 1;
            fprintf(stderr,
                    "ds4: CUDA dynamic attention selected path=dense "
                    "rows=%u blocks=%u heads=%u\n",
                    heads2_rows,
                    (n_head + 1u) / 2u,
                    n_head);
        }
        dim3 heads2_grid(1u, (n_head + 1u) / 2u, 1u);
        if (comp_kv_f16) {
            attention_decode_mixed_heads2_kernel<false, true><<<heads2_grid, 256>>>(
                (float *)heads->ptr,
                sinks,
                (const float *)q->ptr,
                (const float *)raw_kv->ptr,
                n_comp ? comp_kv->ptr : raw_kv->ptr,
                NULL,
                1u,
                0u,
                n_raw,
                raw_cap,
                raw_start,
                n_comp,
                0u,
                0u,
                0u,
                n_head,
                head_dim);
        } else {
            attention_decode_mixed_heads2_kernel<false, false><<<heads2_grid, 256>>>(
                (float *)heads->ptr, sinks, (const float *)q->ptr,
                (const float *)raw_kv->ptr,
                n_comp ? comp_kv->ptr : raw_kv->ptr, NULL,
                1u, 0u, n_raw, raw_cap, raw_start, n_comp,
                0u, 0u, 0u, n_head, head_dim);
        }
        return cuda_ok(cudaGetLastError(),
                       "attention decode dynamic heads2 launch");
    }
    if (!cuda_attention_score_buffer_fits(n_comp)) {
        if (!use_mask && head_dim == 512u &&
            getenv("DS4_CUDA_NO_WINDOW_ATTENTION") == NULL) {
            dim3 online_grid(1, (n_head + 7u) / 8u, 1);
            if (comp_kv_f16) {
                attention_decode_mixed_heads8_online_kernel<true><<<online_grid, 256>>>((float *)heads->ptr,
                                                                              sinks,
                                                                              (const float *)q->ptr,
                                                                              (const float *)raw_kv->ptr,
                                                                              n_comp ? comp_kv->ptr : raw_kv->ptr,
                                                                              1,
                                                                              0,
                                                                              n_raw,
                                                                              raw_cap,
                                                                              raw_start,
                                                                              n_comp,
                                                                              0,
                                                                              0,
                                                                              n_head,
                                                                              head_dim);
            } else {
                attention_decode_mixed_heads8_online_kernel<false><<<online_grid, 256>>>(
                    (float *)heads->ptr, sinks, (const float *)q->ptr,
                    (const float *)raw_kv->ptr,
                    n_comp ? comp_kv->ptr : raw_kv->ptr,
                    1u, 0u, n_raw, raw_cap, raw_start, n_comp,
                    0u, 0u, n_head, head_dim);
            }
            return cuda_ok(cudaGetLastError(), "attention decode online launch");
        }
        fprintf(stderr, "ds4: CUDA attention score buffer too small for %u compressed rows\n", n_comp);
        return 0;
    }
    dim3 grid(1, n_head, 1);
    if (comp_kv_f16) {
        attention_decode_mixed_kernel<true><<<grid, 256>>>((float *)heads->ptr,
                                                 sinks,
                                                 (const float *)q->ptr,
                                                 (const float *)raw_kv->ptr,
                                                 n_comp ? comp_kv->ptr : raw_kv->ptr,
                                                 use_mask ? (const float *)comp_mask->ptr : NULL,
                                                 use_mask,
                                                 1, 0, n_raw, raw_cap, raw_start, n_comp,
                                                 0, 0, n_head, head_dim);
    } else {
        attention_decode_mixed_kernel<false><<<grid, 256>>>(
            (float *)heads->ptr, sinks, (const float *)q->ptr,
            (const float *)raw_kv->ptr,
            n_comp ? comp_kv->ptr : raw_kv->ptr,
            use_mask ? (const float *)comp_mask->ptr : NULL,
            use_mask, 1u, 0u, n_raw, raw_cap, raw_start, n_comp,
            0u, 0u, n_head, head_dim);
    }
    return cuda_ok(cudaGetLastError(), "attention decode launch");
}
extern "C" int ds4_gpu_attention_prefill_raw_heads_tensor(ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size, uint64_t sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv, uint32_t n_tokens, uint32_t window, uint32_t n_head, uint32_t head_dim) {
    if (!heads || !q || !raw_kv || !model_map || sinks_offset > model_size ||
        model_size - sinks_offset < (uint64_t)n_head * sizeof(float) ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float) ||
        window > 256) return 0;
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    if (n_tokens >= 128u && getenv("DS4_CUDA_NO_WINDOW_ATTENTION") == NULL &&
        (getenv("DS4_CUDA_WINDOW_ATTENTION") != NULL || !g_quality_mode)) {
        const int tokentile = cuda_attention_tokentile_dense_launch(
                heads, sinks, q, raw_kv, NULL, 0u,
                n_tokens, 0, n_tokens, n_tokens, 0, 0,
                window, 1u, n_head, head_dim);
        if (tokentile != 0) return tokentile > 0;
    }
    if (n_tokens > 1 && head_dim == 512 &&
        getenv("DS4_CUDA_NO_WINDOW_ATTENTION") == NULL &&
        (getenv("DS4_CUDA_WINDOW_ATTENTION") != NULL || (!g_quality_mode && n_tokens >= 128u))) {
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
        attention_static_mixed_heads8_online_kernel<false><<<grid, 256>>>((float *)heads->ptr,
                                                                   sinks,
                                                                   (const float *)q->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   n_tokens,
                                                                   0,
                                                                   window,
                                                                   1,
                                                                   n_head,
                                                                   head_dim);
        return cuda_ok(cudaGetLastError(), "attention raw window launch");
    }
    if (g_cublas_ready && n_tokens > 1 && head_dim == 512 &&
        getenv("DS4_CUDA_NO_CUBLAS_ATTENTION") == NULL) {
        const uint32_t n_keys = n_tokens;
        const uint64_t score_count = (uint64_t)n_head * n_tokens * n_keys;
        const uint64_t out_count = (uint64_t)n_head * n_tokens * head_dim;
        const uint64_t score_bytes = score_count * sizeof(float);
        const uint64_t out_offset = (score_bytes + 255u) & ~255ull;
        const uint64_t tmp_bytes = out_offset + out_count * sizeof(float);
        float *tmp = (float *)cuda_tmp_alloc(tmp_bytes, "attention raw cublas");
        if (!tmp) return 0;
        float *scores = tmp;
        float *out_tmp = (float *)((char *)tmp + out_offset);
        const float alpha = rsqrtf((float)head_dim);
        const float beta = 0.0f;
        cublasStatus_t st = cublasSgemmStridedBatched(g_cublas,
                                                      CUBLAS_OP_T,
                                                      CUBLAS_OP_N,
                                                      (int)n_keys,
                                                      (int)n_tokens,
                                                      (int)head_dim,
                                                      &alpha,
                                                      (const float *)raw_kv->ptr,
                                                      (int)head_dim,
                                                      0,
                                                      (const float *)q->ptr,
                                                      (int)(n_head * head_dim),
                                                      (long long)head_dim,
                                                      &beta,
                                                      scores,
                                                      (int)n_keys,
                                                      (long long)n_keys * n_tokens,
                                                      (int)n_head);
        if (!cublas_ok(st, "attention raw score gemm")) return 0;
        dim3 sgrid(n_tokens, n_head, 1);
        attention_prefill_raw_softmax_kernel<<<sgrid, 256>>>(scores, sinks, n_tokens, window, n_keys);
        if (!cuda_ok(cudaGetLastError(), "attention raw softmax launch")) return 0;
        const float one = 1.0f;
        st = cublasSgemmStridedBatched(g_cublas,
                                       CUBLAS_OP_N,
                                       CUBLAS_OP_N,
                                       (int)head_dim,
                                       (int)n_tokens,
                                       (int)n_keys,
                                       &one,
                                       (const float *)raw_kv->ptr,
                                       (int)head_dim,
                                       0,
                                       scores,
                                       (int)n_keys,
                                       (long long)n_keys * n_tokens,
                                       &beta,
                                       out_tmp,
                                       (int)head_dim,
                                       (long long)head_dim * n_tokens,
                                       (int)n_head);
        if (!cublas_ok(st, "attention raw value gemm")) return 0;
        uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
        attention_prefill_unpack_heads_kernel<<<(n + 255) / 256, 256>>>((float *)heads->ptr,
                                                                        out_tmp,
                                                                        n_tokens,
                                                                        n_head,
                                                                        head_dim);
        return cuda_ok(cudaGetLastError(), "attention raw unpack launch");
    }
    dim3 grid(n_tokens, n_head, 1);
    attention_prefill_raw_kernel<<<grid, 128>>>((float *)heads->ptr,
                                                sinks,
                                                (const float *)q->ptr,
                                                (const float *)raw_kv->ptr,
                                                n_tokens, window, n_head, head_dim);
    return cuda_ok(cudaGetLastError(), "attention_prefill_raw launch");
}

extern "C" int ds4_gpu_interleave3_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        const ds4_gpu_tensor *c,
        uint32_t              n_rows,
        uint32_t              width) {
    const uint64_t plane_bytes = (uint64_t)n_rows * width * sizeof(float);
    const uint64_t out_bytes = 3u * plane_bytes;
    if (!out || !a || !b || !c || n_rows == 0 || width == 0 ||
        out->bytes < out_bytes || a->bytes < plane_bytes ||
        b->bytes < plane_bytes || c->bytes < plane_bytes) {
        return 0;
    }
    const uint64_t n = (uint64_t)n_rows * 3u * width;
    interleave3_rows_kernel<<<(n + 255u) / 256u, 256>>>(
            (float *)out->ptr,
            (const float *)a->ptr,
            (const float *)b->ptr,
            (const float *)c->ptr,
            n_rows,
            width);
    return cuda_ok(cudaGetLastError(), "interleave3 rows launch");
}

extern "C" int ds4_gpu_dspark_attention_heads_tensor(
        ds4_gpu_tensor       *heads,
        ds4_gpu_tensor       *kv_context,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *main_kv,
        const ds4_gpu_tensor *draft_kv,
        uint32_t              n_main,
        uint32_t              main_cap,
        uint32_t              main_start,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              head_dim) {
    const uint32_t n_keys = n_main + n_tokens;
    if (!heads || !kv_context || !model_map || !q || !main_kv || !draft_kv ||
        n_main == 0 || n_main > main_cap || main_cap == 0 ||
        main_start >= main_cap || n_tokens == 0 || n_keys > 256u ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        main_kv->bytes < (uint64_t)main_cap * head_dim * sizeof(float) ||
        draft_kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float) ||
        kv_context->bytes < (uint64_t)n_keys * head_dim * sizeof(float)) {
        return 0;
    }
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map,
            sinks_offset,
            (uint64_t)n_head * sizeof(float),
            "dspark_attn_sinks");
    if (!sinks) return 0;
    static int notice = 0;
    if (!notice) {
        notice = 1;
        fprintf(stderr,
                "ds4: CUDA DSpark non-causal attention enabled "
                "(main=%u draft=%u heads=%u dim=%u)\n",
                n_main, n_tokens, n_head, head_dim);
    }
    const uint64_t n = (uint64_t)n_keys * head_dim;
    dspark_gather_kv_kernel<<<(n + 255u) / 256u, 256>>>(
            (float *)kv_context->ptr,
            (const float *)main_kv->ptr,
            (const float *)draft_kv->ptr,
            n_main,
            main_cap,
            main_start,
            n_tokens,
            head_dim);
    if (!cuda_ok(cudaGetLastError(), "dspark gather kv launch")) return 0;
    dim3 grid(n_tokens, n_head, 1);
    dspark_attention_kernel<<<grid, 256>>>(
            (float *)heads->ptr,
            sinks,
            (const float *)q->ptr,
            (const float *)kv_context->ptr,
            n_tokens,
            n_keys,
            n_head,
            head_dim);
    return cuda_ok(cudaGetLastError(), "dspark attention launch");
}

extern "C" int ds4_gpu_dspark_confidence_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *hidden,
        const ds4_gpu_tensor *markov,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_tokens,
        uint32_t              hidden_dim,
        uint32_t              markov_dim) {
    const uint64_t weight_count = (uint64_t)hidden_dim + markov_dim;
    if (!out || !hidden || !markov || !model_map || n_tokens == 0 ||
        out->bytes < (uint64_t)n_tokens * sizeof(float) ||
        hidden->bytes < (uint64_t)n_tokens * hidden_dim * sizeof(float) ||
        markov->bytes < (uint64_t)n_tokens * markov_dim * sizeof(float) ||
        weight_offset > model_size ||
        weight_count * sizeof(__half) > model_size - weight_offset) return 0;
    const __half *weight = (const __half *)cuda_model_range_ptr(
            model_map, weight_offset, weight_count * sizeof(__half),
            "dspark_confidence");
    if (!weight) return 0;
    dspark_confidence_kernel<<<n_tokens, 256>>>(
            (float *)out->ptr,
            (const float *)hidden->ptr,
            (const float *)markov->ptr,
            weight,
            hidden_dim,
            markov_dim);
    return cuda_ok(cudaGetLastError(), "dspark confidence launch");
}

static int cuda_attention_tokentile_arch_ok(void) {
    static int initialized;
    static int supported;
    if (!initialized) {
        initialized = 1;
        int device = 0;
        cudaDeviceProp prop;
        if (cudaGetDevice(&device) == cudaSuccess &&
            cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
            supported = prop.major >= 8;
        } else {
            (void)cudaGetLastError();
        }
    }
    return supported;
}

/* Returns 1 after a successful token-tile launch, 0 when the existing path
 * should be used, and -1 for a CUDA launch/configuration failure. */
static int cuda_attention_tokentile_indexed_launch(
        ds4_gpu_tensor       *heads,
        const float          *sinks,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t              comp_kv_f16,
        const int32_t        *topk,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              n_raw,
        uint32_t              raw_cap,
        uint32_t              raw_start,
        uint32_t              n_comp,
        uint32_t              top_k,
        uint32_t              window,
        uint32_t              ratio,
        uint32_t              n_head,
        uint32_t              head_dim) {
    if (!topk || n_tokens < 128u || head_dim != kTTHeadDim || n_head != 64u ||
        top_k != 512u || window != kTTRawWindow || ratio == 0u ||
        n_comp == 0u || n_comp > 32768u || n_raw < n_tokens ||
        (uint64_t)n_raw > (uint64_t)pos0 + n_tokens ||
        g_token_graph_capturing || g_mtp_graph_capturing ||
        !cuda_attention_tokentile_arch_ok()) {
        return 0;
    }

    cuda_nvtx_scope token_tile_scope(
            "ds4/prefill/attention/token_tile/indexed",
            cuda_nvtx_payload(n_comp, n_tokens));

    const uint32_t n_tiles = n_tokens;
    const uint32_t rec_stride = top_k;
    const uint32_t n_mirror_rows = n_tokens + kTTRawWindow - 1u;
    const uint32_t raw_before = n_raw - n_tokens;
    const uint32_t available_before = raw_before < pos0 ? raw_before : pos0;
    const uint32_t available_clamped = available_before < kTTRawWindow - 1u
        ? available_before : kTTRawWindow - 1u;
    const uint32_t raw_row_min = kTTRawWindow - 1u - available_clamped;
    const uint32_t first_raw_pos = pos0 + n_tokens - n_raw;

    uint64_t off = 0;
    const uint64_t raw_mirror_off = off;
    off = cuda_align256_u64(off +
            (uint64_t)n_mirror_rows * kTTHeadDim * sizeof(half));
    const uint64_t comp_mirror_off = off;
    if (!comp_kv_f16) {
        off = cuda_align256_u64(off + 32768ull * kTTHeadDim * sizeof(half));
    }

    unsigned char *scratch = (unsigned char *)
        cuda_attention_tokentile_scratch_alloc(off);
    if (!scratch) return 0;
    half *raw_mirror = (half *)(scratch + raw_mirror_off);
    half *comp_mirror = comp_kv_f16
        ? (half *)comp_kv->ptr
        : (half *)(scratch + comp_mirror_off);

    static int hmma_smem_configured;
    if (!hmma_smem_configured) {
        if (!cuda_ok(cudaFuncSetAttribute(
                    attention_tokentile_hmma_kernel<kTTSparseTileTokens,
                                                    kTTSparseG,
                                                    true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)tt_TokentileSmemBudget<kTTStageRows,
                                                   kTTSparseTileTokens,
                                                   kTTSparseG>::total),
                "attention FlashMLA sparse HMMA shared memory")) {
            return -1;
        }
        hmma_smem_configured = 1;
    }

    {
        cuda_nvtx_scope stage("ds4/prefill/attention/token_tile/raw_mirror",
                              cuda_nvtx_payload(n_mirror_rows, n_tokens));
        attention_tokentile_raw_mirror_kernel<<<n_mirror_rows, 256>>>(
                raw_mirror, (const float *)raw_kv->ptr, NULL, pos0, n_tokens,
                raw_cap, raw_start, first_raw_pos, raw_row_min, head_dim);
        if (!cuda_ok(cudaGetLastError(), "attention token-tile raw mirror launch")) return -1;
    }
    if (!comp_kv_f16) {
        cuda_nvtx_scope stage("ds4/prefill/attention/token_tile/comp_mirror",
                              cuda_nvtx_payload(n_comp, n_tokens));
        attention_tokentile_comp_mirror_kernel<<<n_comp, 256>>>(
                comp_mirror, (const float *)comp_kv->ptr, n_comp, head_dim);
        if (!cuda_ok(cudaGetLastError(), "attention token-tile comp mirror launch")) return -1;
    }

    /* x is the fast launch dimension: place both head groups for one token
     * next to each other so the second CTA can reuse selected KV from L2. */
    dim3 grid(n_head / kTTSparseG, n_tiles, 1);
    {
        cuda_nvtx_scope stage("ds4/prefill/attention/flashmla/hmma",
                              cuda_nvtx_payload(n_tiles, n_tokens));
        attention_tokentile_hmma_kernel<kTTSparseTileTokens, kTTSparseG, true>
            <<<grid, kTTThreads,
                tt_TokentileSmemBudget<kTTStageRows,
                                       kTTSparseTileTokens,
                                       kTTSparseG>::total>>>(
                (float *)heads->ptr, sinks, (const float *)q->ptr,
                raw_mirror, comp_mirror, NULL, NULL, topk,
                pos0, ratio, n_comp, top_k, rec_stride,
                n_tokens, n_head, raw_row_min);
        if (!cuda_ok(cudaGetLastError(), "attention FlashMLA sparse HMMA launch")) {
            return -1;
        }
    }
    static int notice_printed;
    if (!notice_printed) {
        notice_printed = 1;
        fprintf(stderr,
                "ds4: CUDA FlashMLA-style exact sparse prefill enabled "
                "(token=1, heads=32, stage=32, direct-topk=512, comp-kv=%s)\n",
                comp_kv_f16 ? "direct-f16" : "f32-mirror");
    }
    return 1;
}

static int cuda_attention_tokentile_dense_launch(
        ds4_gpu_tensor       *heads,
        const float          *sinks,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t              comp_kv_f16,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              n_raw,
        uint32_t              raw_cap,
        uint32_t              raw_start,
        uint32_t              n_comp,
        uint32_t              window,
        uint32_t              ratio,
        uint32_t              n_head,
        uint32_t              head_dim) {
    if (n_tokens < 128u || head_dim != kTTHeadDim || n_head != 64u ||
        window != kTTRawWindow || ratio == 0u || n_comp > 32768u ||
        (n_comp != 0u && !comp_kv) || n_raw < n_tokens ||
        (uint64_t)n_raw > (uint64_t)pos0 + n_tokens ||
        g_token_graph_capturing || g_mtp_graph_capturing ||
        !cuda_attention_tokentile_arch_ok()) {
        return 0;
    }

    cuda_nvtx_scope token_tile_scope(
            "ds4/prefill/attention/token_tile/dense",
            cuda_nvtx_payload(n_comp, n_tokens));

    const uint32_t n_tiles = (n_tokens + kTTTileTokens - 1u) / kTTTileTokens;
    const uint32_t rec_stride = n_comp ? n_comp : 1u;
    const uint32_t n_mirror_rows = n_tokens + kTTRawWindow - 1u;
    const uint32_t raw_before = n_raw - n_tokens;
    const uint32_t available_before = raw_before < pos0 ? raw_before : pos0;
    const uint32_t available_clamped = available_before < kTTRawWindow - 1u
        ? available_before : kTTRawWindow - 1u;
    const uint32_t raw_row_min = kTTRawWindow - 1u - available_clamped;
    const uint32_t first_raw_pos = pos0 + n_tokens - n_raw;

    uint64_t off = 0;
    const uint64_t records_off = off;
    off = cuda_align256_u64(off + (uint64_t)n_tiles * rec_stride * sizeof(int2));
    const uint64_t counts_off = off;
    off = cuda_align256_u64(off + (uint64_t)n_tiles * sizeof(uint32_t));
    const uint64_t raw_mirror_off = off;
    off = cuda_align256_u64(off +
            (uint64_t)n_mirror_rows * kTTHeadDim * sizeof(half));
    const uint64_t comp_mirror_off = off;
    if (!comp_kv_f16) {
        off = cuda_align256_u64(off + 32768ull * kTTHeadDim * sizeof(half));
    }

    unsigned char *scratch = (unsigned char *)
        cuda_attention_tokentile_scratch_alloc(off);
    if (!scratch) return 0;
    int2 *records = (int2 *)(scratch + records_off);
    uint32_t *counts = (uint32_t *)(scratch + counts_off);
    half *raw_mirror = (half *)(scratch + raw_mirror_off);
    half *comp_mirror = n_comp == 0u
        ? raw_mirror
        : (comp_kv_f16
            ? (half *)comp_kv->ptr
            : (half *)(scratch + comp_mirror_off));

    static int hmma_smem_configured;
    if (!hmma_smem_configured) {
        if (!cuda_ok(cudaFuncSetAttribute(
                    attention_tokentile_hmma_kernel<kTTTileTokens, kTTG, false>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)tt_TokentileSmemBudget<kTTStageRows,
                                                   kTTTileTokens,
                                                   kTTG>::total),
                "attention token-tile HMMA shared memory")) {
            return -1;
        }
        hmma_smem_configured = 1;
    }

    {
        cuda_nvtx_scope stage("ds4/prefill/attention/token_tile/visible_rows",
                              cuda_nvtx_payload(n_comp, n_tokens));
        attention_tokentile_dense_build_kernel<<<n_tiles, kTTThreads>>>(
                records, counts, NULL, pos0, n_tokens, ratio, n_comp, rec_stride);
        if (!cuda_ok(cudaGetLastError(), "attention token-tile dense build launch")) return -1;
    }
    {
        cuda_nvtx_scope stage("ds4/prefill/attention/token_tile/raw_mirror",
                              cuda_nvtx_payload(n_mirror_rows, n_tokens));
        attention_tokentile_raw_mirror_kernel<<<n_mirror_rows, 256>>>(
                raw_mirror, (const float *)raw_kv->ptr, NULL, pos0, n_tokens,
                raw_cap, raw_start, first_raw_pos, raw_row_min, head_dim);
        if (!cuda_ok(cudaGetLastError(), "attention token-tile raw mirror launch")) return -1;
    }
    if (n_comp != 0u && !comp_kv_f16) {
        cuda_nvtx_scope stage("ds4/prefill/attention/token_tile/comp_mirror",
                              cuda_nvtx_payload(n_comp, n_tokens));
        attention_tokentile_comp_mirror_kernel<<<n_comp, 256>>>(
                comp_mirror, (const float *)comp_kv->ptr, n_comp, head_dim);
        if (!cuda_ok(cudaGetLastError(), "attention token-tile comp mirror launch")) return -1;
    }

    dim3 grid(n_tiles, n_head / kTTG, 1);
    {
        cuda_nvtx_scope stage("ds4/prefill/attention/token_tile/hmma",
                              cuda_nvtx_payload(n_tiles, n_tokens));
        attention_tokentile_hmma_kernel<kTTTileTokens, kTTG, false>
            <<<grid, kTTThreads,
                tt_TokentileSmemBudget<kTTStageRows,
                                       kTTTileTokens,
                                       kTTG>::total>>>(
                (float *)heads->ptr, sinks, (const float *)q->ptr,
                raw_mirror, comp_mirror, records, counts, NULL,
                0u, 0u, 0u, 0u, rec_stride,
                n_tokens, n_head, raw_row_min);
        if (!cuda_ok(cudaGetLastError(), "attention token-tile mixed HMMA launch")) {
            return -1;
        }
    }
    static int notice_printed;
    if (!notice_printed) {
        notice_printed = 1;
        fprintf(stderr,
                "ds4: CUDA token-tile HMMA raw/mixed prefill enabled "
                "(tile=16, heads=2, stage=32, comp-kv=%s)\n",
                comp_kv_f16 ? "direct-f16" : "f32-mirror");
    }
    return 1;
}

static int cuda_attention_tokentile_compare(
        ds4_gpu_tensor    *heads_ref,
        ds4_gpu_tensor    *heads_tt,
        std::vector<float> &reference,
        std::vector<float> &candidate,
        uint64_t            count,
        const char         *label) {
    if (!ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(heads_ref, 0, reference.data(),
                             count * sizeof(float)) ||
        !ds4_gpu_tensor_read(heads_tt, 0, candidate.data(),
                             count * sizeof(float))) {
        return 0;
    }
    double diff_sq = 0.0;
    double ref_sq = 0.0;
    double max_abs = 0.0;
    for (uint64_t i = 0; i < count; ++i) {
        const double ref = reference[i];
        const double got = candidate[i];
        if (!isfinite(ref) || !isfinite(got)) return 0;
        const double diff = got - ref;
        diff_sq += diff * diff;
        ref_sq += ref * ref;
        const double ad = fabs(diff);
        if (ad > max_abs) max_abs = ad;
    }
    const double rel_rmse = ref_sq > 0.0
        ? sqrt(diff_sq / ref_sq) : sqrt(diff_sq);
    fprintf(stderr,
            "cuda-regression: token-tile %s rel-rmse=%.6f max-abs=%.6f\n",
            label, rel_rmse, max_abs);
    return rel_rmse <= 0.01 && max_abs <= 0.02;
}

extern "C" int ds4_gpu_attention_tokentile_self_test(void) {
    if (!cuda_attention_tokentile_arch_ok()) return 1;
    const uint32_t n_tokens = 128u;
    const uint32_t n_head = 64u;
    const uint32_t head_dim = 512u;
    /* Sparse metadata is staged directly now; a moderate compressed cache is
     * enough to exercise causal visibility without a 50 MiB host fixture. */
    const uint32_t n_comp = 4608u;
    const uint32_t top_k = 512u;
    const uint32_t pos0 = 1024u;
    const uint32_t n_raw = n_tokens + kTTRawWindow - 1u;
    const uint64_t out_count = (uint64_t)n_tokens * n_head * head_dim;
    const uint64_t raw_count = (uint64_t)n_raw * head_dim;
    const uint64_t comp_count = (uint64_t)n_comp * head_dim;
    const uint64_t topk_count = (uint64_t)n_tokens * top_k;

    std::vector<float> sinks(n_head, 0.0f);
    std::vector<float> q_host(out_count);
    std::vector<float> raw_host(raw_count);
    std::vector<float> comp_host(comp_count);
    std::vector<int32_t> topk_host(topk_count);
    std::vector<float> reference(out_count);
    std::vector<float> candidate(out_count);
    for (uint64_t i = 0; i < out_count; ++i) {
        q_host[i] = (float)((int)(i * 17u % 127u) - 63) * 0.0005f;
    }
    for (uint64_t i = 0; i < raw_count; ++i) {
        raw_host[i] = (float)((int)(i * 29u % 251u) - 125) * 0.0004f;
    }
    for (uint64_t i = 0; i < comp_count; ++i) {
        comp_host[i] = (float)((int)(i * 43u % 257u) - 128) * 0.0004f;
    }
    for (uint32_t t = 0; t < n_tokens; ++t) {
        const uint32_t visible = (pos0 + t + 1u) / 4u;
        for (uint32_t i = 0; i < top_k; ++i) {
            /* Keep the visible prefix expected by the reference kernel, but
             * rotate it per token so sparse gather order is still exercised. */
            topk_host[(uint64_t)t * top_k + i] = i < visible
                ? (int32_t)((i + t * 53u) % visible)
                : (int32_t)i;
        }
    }

    ds4_gpu_tensor *heads_ref = ds4_gpu_tensor_alloc(out_count * sizeof(float));
    ds4_gpu_tensor *heads_tt = ds4_gpu_tensor_alloc(out_count * sizeof(float));
    ds4_gpu_tensor *sinks_dev = ds4_gpu_tensor_alloc((uint64_t)n_head * sizeof(float));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(out_count * sizeof(float));
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_count * sizeof(float));
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(comp_count * sizeof(float));
    ds4_gpu_tensor *comp_f16 = ds4_gpu_tensor_alloc(comp_count * sizeof(uint16_t));
    ds4_gpu_tensor *topk = ds4_gpu_tensor_alloc(topk_count * sizeof(int32_t));
    int ok = heads_ref && heads_tt && sinks_dev && q && raw && comp &&
        comp_f16 && topk &&
        ds4_gpu_tensor_write(sinks_dev, 0, sinks.data(),
                             (uint64_t)n_head * sizeof(float)) &&
        ds4_gpu_tensor_write(q, 0, q_host.data(), out_count * sizeof(float)) &&
        ds4_gpu_tensor_write(raw, 0, raw_host.data(), raw_count * sizeof(float)) &&
        ds4_gpu_tensor_write(comp, 0, comp_host.data(), comp_count * sizeof(float)) &&
        ds4_gpu_tensor_write(topk, 0, topk_host.data(), topk_count * sizeof(int32_t)) &&
        ds4_gpu_tensor_copy_f32_to_f16(comp_f16, 0, comp, 0, comp_count);
    if (ok) {
        dim3 reference_grid(n_tokens, (n_head + 15u) / 16u, 1);
        attention_indexed_mixed_heads8_online_kernel<8, 16, false>
            <<<reference_grid, 512>>>((float *)heads_ref->ptr,
                                      (const float *)sinks_dev->ptr,
                                      (const float *)q->ptr,
                                      (const float *)raw->ptr,
                                      (const float *)comp->ptr,
                                      (const int32_t *)topk->ptr,
                                      n_tokens,
                                      pos0,
                                      n_raw,
                                      n_raw,
                                      0,
                                      n_comp,
                                      top_k,
                                      kTTRawWindow,
                                      4u,
                                      n_head,
                                      head_dim);
        ok = cuda_ok(cudaGetLastError(), "attention token-tile reference launch");
    }
    if (ok) {
        ok = cuda_attention_tokentile_indexed_launch(
                heads_tt, (const float *)sinks_dev->ptr, q, raw, comp, 0u,
                (const int32_t *)topk->ptr,
                n_tokens, pos0, n_raw, n_raw, 0, n_comp, top_k,
                kTTRawWindow, 4u, n_head, head_dim) == 1;
    }
    if (ok) ok = cuda_attention_tokentile_compare(
            heads_ref, heads_tt, reference, candidate, out_count,
            "indexed F32-cache attention");
    if (ok) {
        dim3 reference_grid(n_tokens, (n_head + 7u) / 8u, 1);
        attention_decode_mixed_heads8_online_kernel<false><<<reference_grid, 256>>>(
                (float *)heads_ref->ptr,
                (const float *)sinks_dev->ptr,
                (const float *)q->ptr,
                (const float *)raw->ptr,
                (const float *)comp->ptr,
                n_tokens,
                pos0,
                n_raw,
                n_raw,
                0,
                n_comp,
                kTTRawWindow,
                4u,
                n_head,
                head_dim);
        ok = cuda_ok(cudaGetLastError(),
                     "attention token-tile dense reference launch");
    }
    if (ok) {
        ok = cuda_attention_tokentile_dense_launch(
                heads_tt, (const float *)sinks_dev->ptr, q, raw, comp, 0u,
                n_tokens, pos0, n_raw, n_raw, 0, n_comp,
                kTTRawWindow, 4u, n_head, head_dim) == 1;
    }
    if (ok) ok = cuda_attention_tokentile_compare(
            heads_ref, heads_tt, reference, candidate, out_count,
            "raw/mixed F32-cache attention");

    if (ok) {
        dim3 reference_grid(n_tokens, (n_head + 15u) / 16u, 1);
        attention_indexed_mixed_heads8_online_kernel<8, 16, true>
            <<<reference_grid, 512>>>((float *)heads_ref->ptr,
                                      (const float *)sinks_dev->ptr,
                                      (const float *)q->ptr,
                                      (const float *)raw->ptr,
                                      comp_f16->ptr,
                                      (const int32_t *)topk->ptr,
                                      n_tokens, pos0, n_raw, n_raw, 0,
                                      n_comp, top_k, kTTRawWindow, 4u,
                                      n_head, head_dim);
        ok = cuda_ok(cudaGetLastError(),
                     "attention token-tile F16 indexed reference launch");
    }
    if (ok) {
        ok = cuda_attention_tokentile_indexed_launch(
                heads_tt, (const float *)sinks_dev->ptr, q, raw, comp_f16, 1u,
                (const int32_t *)topk->ptr,
                n_tokens, pos0, n_raw, n_raw, 0, n_comp, top_k,
                kTTRawWindow, 4u, n_head, head_dim) == 1;
    }
    if (ok) ok = cuda_attention_tokentile_compare(
            heads_ref, heads_tt, reference, candidate, out_count,
            "indexed direct-F16 attention");

    if (ok) {
        dim3 reference_grid(n_tokens, (n_head + 7u) / 8u, 1);
        attention_decode_mixed_heads8_online_kernel<true><<<reference_grid, 256>>>(
                (float *)heads_ref->ptr,
                (const float *)sinks_dev->ptr,
                (const float *)q->ptr,
                (const float *)raw->ptr,
                comp_f16->ptr,
                n_tokens, pos0, n_raw, n_raw, 0, n_comp,
                kTTRawWindow, 4u, n_head, head_dim);
        ok = cuda_ok(cudaGetLastError(),
                     "attention token-tile F16 dense reference launch");
    }
    if (ok) {
        ok = cuda_attention_tokentile_dense_launch(
                heads_tt, (const float *)sinks_dev->ptr, q, raw, comp_f16, 1u,
                n_tokens, pos0, n_raw, n_raw, 0, n_comp,
                kTTRawWindow, 4u, n_head, head_dim) == 1;
    }
    if (ok) ok = cuda_attention_tokentile_compare(
            heads_ref, heads_tt, reference, candidate, out_count,
            "raw/mixed direct-F16 attention");

    ds4_gpu_tensor_free(topk);
    ds4_gpu_tensor_free(comp_f16);
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(sinks_dev);
    ds4_gpu_tensor_free(heads_tt);
    ds4_gpu_tensor_free(heads_ref);
    return ok;
}

static int attention_decode_batch_launch(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (comp_kv_f16 > 1u || !heads || !q || !raw_kv || !model_map ||
        n_tokens == 0 ||
        n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap ||
        (n_comp != 0 && !comp_kv) || (use_comp_mask && !comp_mask) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp * head_dim *
             (comp_kv_f16 ? sizeof(half) : sizeof(float))) ||
        (use_comp_mask && comp_mask->bytes < (uint64_t)n_tokens * n_comp * sizeof(float))) {
        return 0;
    }
    if (n_comp != 0 && ratio == 0) return 0;
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    if (!use_comp_mask && getenv("DS4_CUDA_NO_WINDOW_ATTENTION") == NULL &&
        (getenv("DS4_CUDA_WINDOW_ATTENTION") != NULL ||
         (!g_quality_mode && n_tokens >= 128u))) {
        const int tokentile = cuda_attention_tokentile_dense_launch(
                heads, sinks, q, raw_kv, comp_kv, comp_kv_f16,
                n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp,
                window, ratio, n_head, head_dim);
        if (tokentile != 0) return tokentile > 0;
    }
    if (!cuda_attention_score_buffer_fits(n_comp)) {
        if (!use_comp_mask && head_dim == 512u &&
            getenv("DS4_CUDA_NO_WINDOW_ATTENTION") == NULL) {
            dim3 online_grid(n_tokens, (n_head + 7u) / 8u, 1);
            if (comp_kv_f16) {
                attention_decode_mixed_heads8_online_kernel<true><<<online_grid, 256>>>((float *)heads->ptr,
                                                                              sinks,
                                                                              (const float *)q->ptr,
                                                                              (const float *)raw_kv->ptr,
                                                                              n_comp ? comp_kv->ptr : raw_kv->ptr,
                                                                              n_tokens,
                                                                              pos0,
                                                                              n_raw,
                                                                              raw_cap,
                                                                              raw_start,
                                                                              n_comp,
                                                                              window,
                                                                              ratio,
                                                                              n_head,
                                                                              head_dim);
            } else {
                attention_decode_mixed_heads8_online_kernel<false><<<online_grid, 256>>>(
                    (float *)heads->ptr, sinks, (const float *)q->ptr,
                    (const float *)raw_kv->ptr,
                    n_comp ? comp_kv->ptr : raw_kv->ptr,
                    n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp,
                    window, ratio, n_head, head_dim);
            }
            return cuda_ok(cudaGetLastError(), "attention decode online launch");
        }
        fprintf(stderr, "ds4: CUDA attention score buffer too small for %u compressed rows\n", n_comp);
        return 0;
    }
    if (!use_comp_mask && n_tokens > 1 && head_dim == 512 &&
        getenv("DS4_CUDA_NO_WINDOW_ATTENTION") == NULL &&
        (getenv("DS4_CUDA_WINDOW_ATTENTION") != NULL || (!g_quality_mode && n_tokens >= 128u))) {
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
        if (comp_kv_f16) {
            attention_decode_mixed_heads8_online_kernel<true><<<grid, 256>>>((float *)heads->ptr,
                                                                   sinks,
                                                                   (const float *)q->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   n_comp ? comp_kv->ptr : raw_kv->ptr,
                                                                   n_tokens,
                                                                   pos0,
                                                                   n_raw,
                                                                   raw_cap,
                                                                   raw_start,
                                                                   n_comp,
                                                                   window,
                                                                   ratio,
                                                                   n_head,
                                                                   head_dim);
        } else {
            attention_decode_mixed_heads8_online_kernel<false><<<grid, 256>>>(
                (float *)heads->ptr, sinks, (const float *)q->ptr,
                (const float *)raw_kv->ptr,
                n_comp ? comp_kv->ptr : raw_kv->ptr,
                n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp,
                window, ratio, n_head, head_dim);
        }
        return cuda_ok(cudaGetLastError(), "attention decode window launch");
    }
    dim3 grid(n_tokens, n_head, 1);
    if (comp_kv_f16) {
        attention_decode_mixed_kernel<true><<<grid, 256>>>((float *)heads->ptr,
                                                 sinks,
                                                 (const float *)q->ptr,
                                                 (const float *)raw_kv->ptr,
                                                 n_comp ? comp_kv->ptr : raw_kv->ptr,
                                                 use_comp_mask ? (const float *)comp_mask->ptr : NULL,
                                                 use_comp_mask, n_tokens, pos0, n_raw, raw_cap,
                                                 raw_start, n_comp, window, ratio, n_head, head_dim);
    } else {
        attention_decode_mixed_kernel<false><<<grid, 256>>>(
            (float *)heads->ptr, sinks, (const float *)q->ptr,
            (const float *)raw_kv->ptr,
            n_comp ? comp_kv->ptr : raw_kv->ptr,
            use_comp_mask ? (const float *)comp_mask->ptr : NULL,
            use_comp_mask, n_tokens, pos0, n_raw, raw_cap, raw_start,
            n_comp, window, ratio, n_head, head_dim);
    }
    return cuda_ok(cudaGetLastError(), "attention decode batch launch");
}

extern "C" int ds4_gpu_attention_decode_raw_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim) {
    return attention_decode_batch_launch(heads, model_map, model_size, sinks_offset,
                                      q, raw_kv, NULL, 0, NULL, 0, n_tokens, pos0,
                                      n_raw, raw_cap, raw_start, 0, window, 1,
                                      n_head, head_dim);
}

extern "C" int ds4_gpu_attention_decode_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    return attention_decode_batch_launch(heads, model_map, model_size, sinks_offset,
                                      q, raw_kv, comp_kv, comp_kv_f16, comp_mask, use_comp_mask,
                                      n_tokens, pos0, n_raw, raw_cap, raw_start,
                                      n_comp, window, ratio, n_head, head_dim);
}

extern "C" int ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *topk,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                top_k,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (comp_kv_f16 > 1u || !heads || !q || !raw_kv || !comp_kv ||
        !topk || !model_map ||
        n_tokens == 0 || n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap ||
        n_comp == 0 || top_k == 0 ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        comp_kv->bytes < (uint64_t)n_comp * head_dim *
            (comp_kv_f16 ? sizeof(half) : sizeof(float)) ||
        topk->bytes < (uint64_t)n_tokens * top_k * sizeof(int32_t)) {
        return 0;
    }
    if (top_k > 512u) return 0;
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    const int32_t *topk_ptr = (const int32_t *)topk->ptr;
    if (n_tokens == 1u) {
        uint32_t raw_rows = n_raw > 256u ? 256u : n_raw;
        if (window != 0u && raw_rows > window) raw_rows = window;
        uint32_t visible_comp = n_comp;
        if (ratio != 0u) {
            visible_comp = (pos0 + 1u) / ratio;
            if (visible_comp > n_comp) visible_comp = n_comp;
        }
        uint32_t selected_rows = top_k < visible_comp ? top_k : visible_comp;
        const uint32_t heads2_rows = raw_rows + selected_rows;
        if (cuda_attn_heads2_use(heads2_rows,
                                 n_tokens,
                                 n_head,
                                 head_dim)) {
            if (!g_attn_heads2_indexed_notice) {
                g_attn_heads2_indexed_notice = 1;
                fprintf(stderr,
                        "ds4: CUDA dynamic attention selected path=indexed "
                        "rows=%u blocks=%u heads=%u top_k=%u\n",
                        heads2_rows,
                        (n_head + 1u) / 2u,
                        n_head,
                        selected_rows);
            }
            dim3 heads2_grid(1u, (n_head + 1u) / 2u, 1u);
            if (comp_kv_f16) {
                attention_decode_mixed_heads2_kernel<true, true><<<heads2_grid, 256>>>(
                    (float *)heads->ptr,
                    sinks,
                    (const float *)q->ptr,
                    (const float *)raw_kv->ptr,
                    comp_kv->ptr,
                    topk_ptr,
                    n_tokens,
                    pos0,
                    n_raw,
                    raw_cap,
                    raw_start,
                    n_comp,
                    top_k,
                    window,
                    ratio,
                    n_head,
                    head_dim);
            } else {
                attention_decode_mixed_heads2_kernel<true, false><<<heads2_grid, 256>>>(
                    (float *)heads->ptr, sinks, (const float *)q->ptr,
                    (const float *)raw_kv->ptr, comp_kv->ptr, topk_ptr,
                    n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp,
                    top_k, window, ratio, n_head, head_dim);
            }
            return cuda_ok(cudaGetLastError(),
                           "attention indexed dynamic heads2 launch");
        }
    }
    const int tokentile = cuda_attention_tokentile_indexed_launch(
            heads, sinks, q, raw_kv, comp_kv, comp_kv_f16, topk_ptr,
            n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp, top_k,
            window, ratio, n_head, head_dim);
    if (tokentile != 0) return tokentile > 0;
    if (n_tokens > 1u && top_k == 512u &&
        getenv("DS4_CUDA_NO_INDEXED_TOPK_SORT") == NULL) {
        const uint64_t sort_bytes = (uint64_t)n_tokens * top_k * sizeof(int32_t);
        int32_t *sorted = (int32_t *)cuda_tmp_alloc(sort_bytes, "indexed attention topk sort");
        if (!sorted) return 0;
        indexed_topk_sort_512_asc_kernel<<<n_tokens, 512>>>(sorted, topk_ptr, n_tokens);
        if (!cuda_ok(cudaGetLastError(), "indexed attention topk sort launch")) return 0;
        topk_ptr = sorted;
    }
    if (n_tokens > 1 && head_dim == 512 && top_k <= 512u &&
        getenv("DS4_CUDA_NO_INDEXED_HEADS8") == NULL) {
        if (getenv("DS4_CUDA_INDEXED_TWOPASS") == NULL) {
            dim3 grid(n_tokens, (n_head + 15u) / 16u, 1);
            if (comp_kv_f16) {
                attention_indexed_mixed_heads8_online_kernel<8, 16, true><<<grid, 512>>>((float *)heads->ptr,
                                                                               sinks,
                                                                               (const float *)q->ptr,
                                                                               (const float *)raw_kv->ptr,
                                                                               comp_kv->ptr,
                                                                               topk_ptr,
                                                                               n_tokens,
                                                                               pos0,
                                                                               n_raw,
                                                                               raw_cap,
                                                                               raw_start,
                                                                               n_comp,
                                                                               top_k,
                                                                               window,
                                                                               ratio,
                                                                               n_head,
                                                                               head_dim);
            } else {
                attention_indexed_mixed_heads8_online_kernel<8, 16, false><<<grid, 512>>>(
                    (float *)heads->ptr, sinks, (const float *)q->ptr,
                    (const float *)raw_kv->ptr, comp_kv->ptr, topk_ptr,
                    n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp,
                    top_k, window, ratio, n_head, head_dim);
            }
            return cuda_ok(cudaGetLastError(), "attention indexed online launch");
        }
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
        if (comp_kv_f16) {
            attention_indexed_mixed_heads8_rb4_kernel<true><<<grid, 256>>>((float *)heads->ptr,
                                                                 sinks,
                                                                 (const float *)q->ptr,
                                                                 (const float *)raw_kv->ptr,
                                                                 comp_kv->ptr,
                                                                 topk_ptr,
                                                                 n_tokens,
                                                                 pos0,
                                                                 n_raw,
                                                                 raw_cap,
                                                                 raw_start,
                                                                 n_comp,
                                                                 top_k,
                                                                 window,
                                                                 ratio,
                                                                 n_head,
                                                                 head_dim);
        } else {
            attention_indexed_mixed_heads8_rb4_kernel<false><<<grid, 256>>>(
                (float *)heads->ptr, sinks, (const float *)q->ptr,
                (const float *)raw_kv->ptr, comp_kv->ptr, topk_ptr,
                n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp,
                top_k, window, ratio, n_head, head_dim);
        }
        return cuda_ok(cudaGetLastError(), "attention indexed heads8 launch");
    }
    dim3 grid(n_tokens, n_head, 1);
    if (comp_kv_f16) {
        attention_indexed_mixed_kernel<true><<<grid, 256>>>((float *)heads->ptr,
                                                  sinks,
                                                  (const float *)q->ptr,
                                                  (const float *)raw_kv->ptr,
                                                  comp_kv->ptr,
                                                  topk_ptr,
                                                  n_tokens,
                                                  pos0,
                                                  n_raw,
                                                  raw_cap,
                                                  raw_start,
                                                  n_comp,
                                                  top_k,
                                                  window,
                                                  ratio,
                                                  n_head,
                                                  head_dim);
    } else {
        attention_indexed_mixed_kernel<false><<<grid, 256>>>(
            (float *)heads->ptr, sinks, (const float *)q->ptr,
            (const float *)raw_kv->ptr, comp_kv->ptr, topk_ptr,
            n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp,
            top_k, window, ratio, n_head, head_dim);
    }
    return cuda_ok(cudaGetLastError(), "attention indexed mixed launch");
}

static int attention_prefill_mixed_launch(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (comp_kv_f16 > 1u || !heads || !q || !raw_kv || !model_map ||
        n_tokens == 0 || ratio == 0 ||
        (n_comp != 0 && !comp_kv) || (use_comp_mask && !comp_mask) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float) ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp * head_dim *
             (comp_kv_f16 ? sizeof(half) : sizeof(float))) ||
        (use_comp_mask && comp_mask->bytes < (uint64_t)n_tokens * n_comp * sizeof(float))) {
        return 0;
    }
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    if (!use_comp_mask && n_tokens >= 128u &&
        getenv("DS4_CUDA_NO_WINDOW_ATTENTION") == NULL &&
        (getenv("DS4_CUDA_WINDOW_ATTENTION") != NULL || !g_quality_mode)) {
        const int tokentile = cuda_attention_tokentile_dense_launch(
                heads, sinks, q, raw_kv, comp_kv, comp_kv_f16,
                n_tokens, 0, n_tokens, n_tokens, 0, n_comp,
                window, ratio, n_head, head_dim);
        if (tokentile != 0) return tokentile > 0;
    }
    if (!use_comp_mask && n_tokens > 1 && head_dim == 512 &&
        getenv("DS4_CUDA_NO_WINDOW_ATTENTION") == NULL &&
        (getenv("DS4_CUDA_WINDOW_ATTENTION") != NULL || (!g_quality_mode && n_tokens >= 128u))) {
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
        if (comp_kv_f16) {
            attention_static_mixed_heads8_online_kernel<true><<<grid, 256>>>((float *)heads->ptr,
                                                                   sinks,
                                                                   (const float *)q->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   n_comp ? comp_kv->ptr : raw_kv->ptr,
                                                                   n_tokens,
                                                                   n_comp,
                                                                   window,
                                                                   ratio,
                                                                   n_head,
                                                                   head_dim);
        } else {
            attention_static_mixed_heads8_online_kernel<false><<<grid, 256>>>(
                (float *)heads->ptr, sinks, (const float *)q->ptr,
                (const float *)raw_kv->ptr,
                n_comp ? comp_kv->ptr : raw_kv->ptr,
                n_tokens, n_comp, window, ratio, n_head, head_dim);
        }
        return cuda_ok(cudaGetLastError(), "attention mixed window launch");
    }
    if (g_cublas_ready && n_tokens > 1 && head_dim == 512 &&
        getenv("DS4_CUDA_NO_CUBLAS_ATTENTION") == NULL) {
        const uint32_t n_keys = n_tokens + n_comp;
        const uint64_t kv_count = (uint64_t)n_keys * head_dim;
        const uint64_t score_count = (uint64_t)n_head * n_tokens * n_keys;
        const uint64_t out_count = (uint64_t)n_head * n_tokens * head_dim;
        const uint64_t kv_bytes = kv_count * sizeof(float);
        const uint64_t score_offset = (kv_bytes + 255u) & ~255ull;
        const uint64_t score_bytes = score_count * sizeof(float);
        const uint64_t out_offset = score_offset + ((score_bytes + 255u) & ~255ull);
        const uint64_t tmp_bytes = out_offset + out_count * sizeof(float);
        float *tmp = (float *)cuda_tmp_alloc(tmp_bytes, "attention mixed cublas");
        if (!tmp) return 0;
        float *kv = tmp;
        float *scores = (float *)((char *)tmp + score_offset);
        float *out_tmp = (float *)((char *)tmp + out_offset);
        if (comp_kv_f16) {
            attention_prefill_pack_mixed_kv_kernel<true>
                <<<(kv_count + 255) / 256, 256>>>(
                    kv, (const float *)raw_kv->ptr,
                    n_comp ? comp_kv->ptr : raw_kv->ptr,
                    n_tokens, n_comp, head_dim);
        } else {
            attention_prefill_pack_mixed_kv_kernel<false>
                <<<(kv_count + 255) / 256, 256>>>(
                    kv, (const float *)raw_kv->ptr,
                    n_comp ? comp_kv->ptr : raw_kv->ptr,
                    n_tokens, n_comp, head_dim);
        }
        if (!cuda_ok(cudaGetLastError(), "attention mixed kv pack launch")) return 0;
        const float alpha = rsqrtf((float)head_dim);
        const float beta = 0.0f;
        cublasStatus_t st = cublasSgemmStridedBatched(g_cublas,
                                                      CUBLAS_OP_T,
                                                      CUBLAS_OP_N,
                                                      (int)n_keys,
                                                      (int)n_tokens,
                                                      (int)head_dim,
                                                      &alpha,
                                                      kv,
                                                      (int)head_dim,
                                                      0,
                                                      (const float *)q->ptr,
                                                      (int)(n_head * head_dim),
                                                      (long long)head_dim,
                                                      &beta,
                                                      scores,
                                                      (int)n_keys,
                                                      (long long)n_keys * n_tokens,
                                                      (int)n_head);
        if (!cublas_ok(st, "attention mixed score gemm")) return 0;
        dim3 sgrid(n_tokens, n_head, 1);
        attention_prefill_mixed_softmax_kernel<<<sgrid, 256>>>(
                scores,
                sinks,
                use_comp_mask ? (const float *)comp_mask->ptr : NULL,
                use_comp_mask,
                n_tokens,
                n_comp,
                window,
                ratio,
                n_keys);
        if (!cuda_ok(cudaGetLastError(), "attention mixed softmax launch")) return 0;
        const float one = 1.0f;
        st = cublasSgemmStridedBatched(g_cublas,
                                       CUBLAS_OP_N,
                                       CUBLAS_OP_N,
                                       (int)head_dim,
                                       (int)n_tokens,
                                       (int)n_keys,
                                       &one,
                                       kv,
                                       (int)head_dim,
                                       0,
                                       scores,
                                       (int)n_keys,
                                       (long long)n_keys * n_tokens,
                                       &beta,
                                       out_tmp,
                                       (int)head_dim,
                                       (long long)head_dim * n_tokens,
                                       (int)n_head);
        if (!cublas_ok(st, "attention mixed value gemm")) return 0;
        uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
        attention_prefill_unpack_heads_kernel<<<(n + 255) / 256, 256>>>((float *)heads->ptr,
                                                                        out_tmp,
                                                                        n_tokens,
                                                                        n_head,
                                                                        head_dim);
        return cuda_ok(cudaGetLastError(), "attention mixed unpack launch");
    }
    dim3 grid(n_tokens, n_head, 1);
    if (comp_kv_f16) {
        attention_prefill_mixed_kernel<true><<<grid, 256>>>((float *)heads->ptr,
                                                  sinks,
                                                  (const float *)q->ptr,
                                                  (const float *)raw_kv->ptr,
                                                  n_comp ? comp_kv->ptr : raw_kv->ptr,
                                                  use_comp_mask ? (const float *)comp_mask->ptr : NULL,
                                                  use_comp_mask, n_tokens, n_comp, window, ratio,
                                                  n_head, head_dim);
    } else {
        attention_prefill_mixed_kernel<false><<<grid, 256>>>(
            (float *)heads->ptr, sinks, (const float *)q->ptr,
            (const float *)raw_kv->ptr,
            n_comp ? comp_kv->ptr : raw_kv->ptr,
            use_comp_mask ? (const float *)comp_mask->ptr : NULL,
            use_comp_mask, n_tokens, n_comp, window, ratio, n_head, head_dim);
    }
    return cuda_ok(cudaGetLastError(), "attention prefill mixed launch");
}

extern "C" int ds4_gpu_attention_prefill_static_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    return attention_prefill_mixed_launch(heads, model_map, model_size, sinks_offset,
                                       q, raw_kv, comp_kv, comp_kv_f16, NULL, 0, n_tokens,
                                       n_comp, window, ratio, n_head, head_dim);
}

extern "C" int ds4_gpu_attention_prefill_masked_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    return attention_prefill_mixed_launch(heads, model_map, model_size, sinks_offset,
                                       q, raw_kv, comp_kv, comp_kv_f16, comp_mask, 1, n_tokens,
                                       n_comp, window, ratio, n_head, head_dim);
}
static int cuda_attention_output_q8_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        ds4_gpu_tensor       *group_tmp,
        ds4_gpu_tensor       *low_tmp,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens,
        int                     fuse_inverse_rope,
        uint32_t                head_dim,
        uint32_t                n_rot,
        uint32_t                pos0,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow) {
    (void)group_tmp;
    (void)low_tmp;
    if (!out || !low || !heads || !model_map ||
        group_dim == 0 || rank == 0 || n_groups == 0 || out_dim == 0 || n_tokens == 0) {
        return 0;
    }
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t blocks_a = (group_dim + 31) / 32;
    const uint64_t blocks_b = (low_dim + 31) / 32;
    const uint64_t out_a_bytes = (uint64_t)n_groups * rank * blocks_a * 34;
    const uint64_t out_b_bytes = out_dim * blocks_b * 34;
    if (out_a_offset > model_size || out_b_offset > model_size ||
        out_a_bytes > model_size - out_a_offset ||
        out_b_bytes > model_size - out_b_offset ||
        heads->bytes < (uint64_t)n_tokens * n_groups * group_dim * sizeof(float) ||
        low->bytes < (uint64_t)n_tokens * low_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float)) {
        return 0;
    }
    const unsigned char *out_a = reinterpret_cast<const unsigned char *>(
            cuda_model_range_ptr(model_map, out_a_offset, out_a_bytes, "attn_out_a"));
    const unsigned char *out_b = reinterpret_cast<const unsigned char *>(
            cuda_model_range_ptr(model_map, out_b_offset, out_b_bytes, "attn_out_b"));
    if (!out_a || !out_b) return 0;

    const __half *out_a_f16 = NULL;
    uint32_t out_a_cublas_min_tokens = 2u;
    const char *out_a_min_env = getenv("DS4_CUDA_ATTENTION_OUTPUT_A_CUBLAS_MIN");
    if (out_a_min_env && out_a_min_env[0]) {
        char *endp = NULL;
        long v = strtol(out_a_min_env, &endp, 10);
        if (endp != out_a_min_env && v > 1 && v < 4096) out_a_cublas_min_tokens = (uint32_t)v;
    }
    if (!g_quality_mode &&
        g_cublas_ready &&
        n_tokens >= out_a_cublas_min_tokens &&
        getenv("DS4_CUDA_NO_CUBLAS_ATTENTION_OUTPUT_A") == NULL) {
        out_a_f16 = cuda_q8_f16_ptr(model_map, out_a_offset, out_a_bytes, group_dim, low_dim, "attn_output_a");
    }
    if (out_a_f16) {
        const uint64_t heads_h_count = (uint64_t)n_groups * n_tokens * group_dim;
        const uint64_t low_tmp_count = (uint64_t)n_groups * n_tokens * rank;
        const uint64_t heads_h_bytes = heads_h_count * sizeof(__half);
        const uint64_t low_tmp_offset = (heads_h_bytes + 255u) & ~255ull;
        const uint64_t tmp_bytes = low_tmp_offset + low_tmp_count * sizeof(float);
        void *tmp = cuda_tmp_alloc(tmp_bytes, "attention output a cublas");
        if (!tmp) return 0;
        __half *heads_h = (__half *)tmp;
        float *low_packed = (float *)((char *)tmp + low_tmp_offset);
        if (fuse_inverse_rope) {
            if (head_dim == 0u || group_dim % head_dim != 0u ||
                n_rot > head_dim || (n_rot & 1u) != 0u) {
                return 0;
            }
            attention_inverse_rope_pack_group_heads_f16_kernel<<<
                    (heads_h_count / 2u + 255u) / 256u, 256>>>(
                    heads_h,
                    (const float *)heads->ptr,
                    n_tokens,
                    n_groups,
                    group_dim,
                    head_dim,
                    n_rot,
                    pos0,
                    n_ctx_orig,
                    freq_base,
                    freq_scale,
                    ext_factor,
                    attn_factor,
                    beta_fast,
                    beta_slow);
        } else {
            attention_pack_group_heads_f16_kernel<<<
                    (heads_h_count / 2u + 255u) / 256u, 256>>>(
                    heads_h,
                    (const float *)heads->ptr,
                    n_tokens,
                    n_groups,
                    group_dim);
        }
        if (!cuda_ok(cudaGetLastError(), "attention_output_q8_a pack launch")) return 0;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t st = cublasGemmStridedBatchedEx(g_cublas,
                                                       CUBLAS_OP_T,
                                                       CUBLAS_OP_N,
                                                       (int)rank,
                                                       (int)n_tokens,
                                                       (int)group_dim,
                                                       &alpha,
                                                       out_a_f16,
                                                       CUDA_R_16F,
                                                       (int)group_dim,
                                                       (long long)rank * group_dim,
                                                       heads_h,
                                                       CUDA_R_16F,
                                                       (int)group_dim,
                                                       (long long)n_tokens * group_dim,
                                                       &beta,
                                                       low_packed,
                                                       CUDA_R_32F,
                                                       (int)rank,
                                                       (long long)rank * n_tokens,
                                                       (int)n_groups,
                                                       CUDA_R_32F,
                                                       CUBLAS_GEMM_DEFAULT);
        if (!cublas_ok(st, "attention output a gemm")) return 0;
        attention_unpack_group_low_kernel<<<(low_tmp_count / 4u + 255u) / 256u, 256>>>(
                (float *)low->ptr,
                low_packed,
                n_tokens,
                n_groups,
                rank);
        if (!cuda_ok(cudaGetLastError(), "attention_output_q8_a unpack launch")) return 0;
    } else {
        if (fuse_inverse_rope) return 0;
        const uint64_t x_rows = (uint64_t)n_tokens * n_groups;
        const uint64_t xq_bytes = x_rows * blocks_a * 32u;
        const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
        const uint64_t tmp_bytes = scale_offset + x_rows * blocks_a * sizeof(float);
        void *tmp = cuda_tmp_alloc(tmp_bytes, "attention output a q8 prequant");
        if (!tmp) return 0;
        int8_t *xq = (int8_t *)tmp;
        float *xscale = (float *)((char *)tmp + scale_offset);
        const int use_dp4a = cuda_q8_use_dp4a();
        if (!launch_quantize_q8_0_f32_rows(xq,
                                           xscale,
                                           (const float *)heads->ptr,
                                           x_rows,
                                           group_dim,
                                           blocks_a,
                                           "attention_output_q8_a prequant launch")) return 0;
        dim3 grid_a(((unsigned)low_dim + 7u) / 8u, (unsigned)n_tokens, 1);
        grouped_q8_0_a_preq_warp8_kernel<<<grid_a, 256>>>((float *)low->ptr,
                                                          out_a,
                                                          xq,
                                                          xscale,
                                                          group_dim,
                                                          rank,
                                                          n_groups,
                                                          n_tokens,
                                                          blocks_a,
                                                          use_dp4a);
        if (!cuda_ok(cudaGetLastError(), "attention_output_q8_a preq launch")) return 0;
    }

    (void)out_b;
    return cuda_matmul_q8_0_tensor_labeled(out,
                                           model_map,
                                           model_size,
                                           out_b_offset,
                                           low_dim,
                                           out_dim,
                                           low,
                                           n_tokens,
                                           "attn_output_b");
}

extern "C" int ds4_gpu_attention_output_q8_batch_tensor(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *low,
        ds4_gpu_tensor *group_tmp,
        ds4_gpu_tensor *low_tmp,
        const void *model_map,
        uint64_t model_size,
        uint64_t out_a_offset,
        uint64_t out_b_offset,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        uint64_t out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t n_tokens) {
    return cuda_attention_output_q8_batch_tensor(
            out, low, group_tmp, low_tmp,
            model_map, model_size, out_a_offset, out_b_offset,
            group_dim, rank, n_groups, out_dim, heads, n_tokens,
            0, 0u, 0u, 0u, 0u,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
}

extern "C" int ds4_gpu_attention_output_q8_batch_inverse_rope_tensor(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *low,
        ds4_gpu_tensor *group_tmp,
        ds4_gpu_tensor *low_tmp,
        const void *model_map,
        uint64_t model_size,
        uint64_t out_a_offset,
        uint64_t out_b_offset,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        uint64_t out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t n_tokens,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t pos0,
        uint32_t n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    if (n_tokens < 128u) return 0;
    return cuda_attention_output_q8_batch_tensor(
            out, low, group_tmp, low_tmp,
            model_map, model_size, out_a_offset, out_b_offset,
            group_dim, rank, n_groups, out_dim, heads, n_tokens,
            1, head_dim, n_rot, pos0, n_ctx_orig,
            freq_base, freq_scale, ext_factor, attn_factor,
            beta_fast, beta_slow);
}

extern "C" int ds4_gpu_attention_output_q8_batch_f16_tensor(
        ds4_gpu_tensor *out_h,
        ds4_gpu_tensor *low,
        const void *model_map,
        uint64_t model_size,
        uint64_t out_a_offset,
        uint64_t out_b_offset,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        uint64_t out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t n_tokens) {
    (void)out_h; (void)low; (void)model_map; (void)model_size;
    (void)out_a_offset; (void)out_b_offset; (void)group_dim; (void)rank;
    (void)n_groups; (void)out_dim; (void)heads; (void)n_tokens;
    return 0;
}

extern "C" int ds4_gpu_attention_output_low_q8_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        const ds4_gpu_tensor *heads) {
    if (!low || !heads || !model_map || group_dim == 0 || rank == 0 || n_groups == 0) {
        return 0;
    }
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t blocks_a = (group_dim + 31) / 32;
    const uint64_t out_a_bytes = (uint64_t)n_groups * rank * blocks_a * 34;
    if (out_a_offset > model_size ||
        out_a_bytes > model_size - out_a_offset ||
        heads->bytes < (uint64_t)n_groups * group_dim * sizeof(float) ||
        low->bytes < low_dim * sizeof(float)) {
        return 0;
    }
    const unsigned char *out_a = reinterpret_cast<const unsigned char *>(
            cuda_model_range_ptr(model_map, out_a_offset, out_a_bytes, "attn_out_a"));
    if (!out_a) return 0;

    const uint64_t x_rows = (uint64_t)n_groups;
    const uint64_t xq_bytes = x_rows * blocks_a * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + x_rows * blocks_a * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "attention output low q8 prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    if (!launch_quantize_q8_0_f32_rows(xq,
                                       xscale,
                                       (const float *)heads->ptr,
                                       x_rows,
                                       group_dim,
                                       blocks_a,
                                       "attention_output_low_q8 prequant launch")) return 0;
    dim3 grid_a(((unsigned)low_dim + 7u) / 8u, 1, 1);
    grouped_q8_0_a_preq_warp8_kernel<<<grid_a, 256>>>((float *)low->ptr,
                                                      out_a,
                                                      xq,
                                                      xscale,
                                                      group_dim,
                                                      rank,
                                                      n_groups,
                                                      1,
                                                      blocks_a,
                                                      use_dp4a);
    return cuda_ok(cudaGetLastError(), "attention_output_low_q8 launch");
}
extern "C" int ds4_gpu_swiglu_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up, uint32_t n, float clamp, float weight) {
    if (!out || !gate || !up ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        gate->bytes < (uint64_t)n * sizeof(float) ||
        up->bytes < (uint64_t)n * sizeof(float)) return 0;
    swiglu_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)gate->ptr, (const float *)up->ptr, n, clamp, weight);
    return cuda_ok(cudaGetLastError(), "swiglu launch");
}
extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp) {
    if (getenv("DS4_CUDA_DISABLE_SHARED_GATE_UP_PAIR") == NULL) {
        return ds4_gpu_matmul_q8_0_pair_tensor(gate, up,
                                                 model_map, model_size,
                                                 gate_offset, up_offset,
                                                 in_dim, out_dim, out_dim,
                                                 x, 1) &&
               ds4_gpu_swiglu_tensor(mid, gate, up, (uint32_t)out_dim, clamp, 1.0f);
    }
    return ds4_gpu_matmul_q8_0_tensor(gate, model_map, model_size,
                                        gate_offset, in_dim, out_dim, x, 1) &&
           ds4_gpu_matmul_q8_0_tensor(up, model_map, model_size,
                                        up_offset, in_dim, out_dim, x, 1) &&
           ds4_gpu_swiglu_tensor(mid, gate, up, (uint32_t)out_dim, clamp, 1.0f);
}
extern "C" int ds4_gpu_add_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *a, const ds4_gpu_tensor *b, uint32_t n) {
    if (!out || !a || !b ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        a->bytes < (uint64_t)n * sizeof(float) ||
        b->bytes < (uint64_t)n * sizeof(float)) return 0;
    add_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)a->ptr, (const float *)b->ptr, n);
    return cuda_ok(cudaGetLastError(), "add launch");
}
extern "C" int ds4_gpu_directional_steering_project_tensor(
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *directions,
        uint32_t                layer,
        uint32_t                width,
        uint32_t                rows,
        float                   scale) {
    if (!x || !directions || width == 0 || rows == 0 || scale == 0.0f) return 0;
    const uint64_t x_bytes = (uint64_t)width * rows * sizeof(float);
    const uint64_t dir_bytes = (uint64_t)(layer + 1u) * width * sizeof(float);
    if (x->bytes < x_bytes || directions->bytes < dir_bytes) return 0;

    uint32_t nth = 256u;
    while (nth > width && nth > 1u) nth >>= 1;
    directional_steering_project_kernel<<<rows, nth>>>(
            (float *)x->ptr,
            (const float *)directions->ptr,
            layer,
            width,
            rows,
            scale);
    return cuda_ok(cudaGetLastError(), "directional steering launch");
}
extern "C" int ds4_gpu_router_select_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size, uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows, uint32_t token, uint32_t n_expert, uint32_t n_expert_used, float expert_weight_scale, uint32_t n_expert_groups, uint32_t n_group_used, bool has_bias, bool hash_mode, const ds4_gpu_tensor *logits) {
    if (!selected || !weights || !probs || !logits || !model_map || n_expert_groups > 1u || n_group_used > 0u) return 0;
    if (n_expert != 256u || n_expert_used != 6u || fabsf(expert_weight_scale - 1.5f) > 1.0e-6f) return 0;
    int32_t tok = (int32_t)token;
    int ok = 1;
    const float *bias = NULL;
    const int32_t *hash = NULL;
    if (ok && has_bias && !hash_mode) {
        if (bias_offset > model_size || model_size - bias_offset < 256u * sizeof(float)) ok = 0;
        else bias = (const float *)cuda_model_range_ptr(model_map, bias_offset, 256u * sizeof(float), "router_bias");
        if (!bias) ok = 0;
    }
    if (ok && hash_mode) {
        const uint64_t hash_bytes = (uint64_t)hash_rows * 6u * sizeof(int32_t);
        if (hash_offset > model_size || hash_bytes > model_size - hash_offset) ok = 0;
        else hash = (const int32_t *)cuda_model_range_ptr(model_map, hash_offset, hash_bytes, "router_hash");
        if (!hash) ok = 0;
    }
    if (ok) {
        if (getenv("DS4_CUDA_NO_WARP_ROUTER_SELECT") == NULL &&
            getenv("DS4_CUDA_NO_PARALLEL_ROUTER_SELECT") == NULL) {
            dim3 block(32, 4, 1);
            router_select_warp_topk_kernel<<<1, block>>>((int32_t *)selected->ptr, (float *)weights->ptr, (float *)probs->ptr,
                                                         bias, hash, (const float *)logits->ptr, NULL, tok, hash_rows, 1,
                                                         has_bias && !hash_mode, hash_mode);
        } else if (getenv("DS4_CUDA_NO_PARALLEL_ROUTER_SELECT") == NULL) {
            router_select_parallel_kernel<<<1, 256>>>((int32_t *)selected->ptr, (float *)weights->ptr, (float *)probs->ptr,
                                                      bias, hash, (const float *)logits->ptr, NULL, tok, hash_rows, 1,
                                                      has_bias && !hash_mode, hash_mode);
        } else {
            router_select_kernel<<<1, 1>>>((int32_t *)selected->ptr, (float *)weights->ptr, (float *)probs->ptr,
                                          bias, hash, (const float *)logits->ptr, NULL, tok, hash_rows, 1,
                                          has_bias && !hash_mode, hash_mode);
        }
        ok = cuda_ok(cudaGetLastError(), "router_select launch");
    }
    return ok;
}
extern "C" int ds4_gpu_router_select_batch_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size, uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows, uint32_t n_expert_groups, uint32_t n_group_used, bool has_bias, bool hash_mode, const ds4_gpu_tensor *logits, const ds4_gpu_tensor *tokens, uint32_t n_expert, uint32_t n_expert_used, float expert_weight_scale, uint32_t n_tokens) {
    if (n_expert != 256u || n_expert_used != 6u || fabsf(expert_weight_scale - 1.5f) > 1.0e-6f) return 0;
    if (!selected || !weights || !probs || !logits || !tokens || !model_map || n_tokens == 0 ||
        n_expert_groups > 1u || n_group_used > 0u ||
        logits->bytes < (uint64_t)n_tokens * 256u * sizeof(float) ||
        probs->bytes < (uint64_t)n_tokens * 256u * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * 6u * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * 6u * sizeof(float)) {
        return 0;
    }
    const float *bias = NULL;
    const int32_t *hash = NULL;
    if (has_bias && !hash_mode) {
        if (bias_offset > model_size || model_size - bias_offset < 256u * sizeof(float)) return 0;
        bias = (const float *)cuda_model_range_ptr(model_map, bias_offset, 256u * sizeof(float), "router_bias");
        if (!bias) return 0;
    }
    if (hash_mode) {
        const uint64_t hash_bytes = (uint64_t)hash_rows * 6u * sizeof(int32_t);
        if (hash_offset > model_size || hash_bytes > model_size - hash_offset) return 0;
        hash = (const int32_t *)cuda_model_range_ptr(model_map, hash_offset, hash_bytes, "router_hash");
        if (!hash) return 0;
    }
    if (getenv("DS4_CUDA_NO_WARP_ROUTER_SELECT") == NULL &&
        getenv("DS4_CUDA_NO_PARALLEL_ROUTER_SELECT") == NULL) {
        dim3 block(32, 4, 1);
        router_select_warp_topk_kernel<<<(n_tokens + 3u) / 4u, block>>>((int32_t *)selected->ptr,
                                                                        (float *)weights->ptr,
                                                                        (float *)probs->ptr,
                                                                        bias,
                                                                        hash,
                                                                        (const float *)logits->ptr,
                                                                        (const int32_t *)tokens->ptr,
                                                                        0,
                                                                        hash_rows,
                                                                        n_tokens,
                                                                        has_bias && !hash_mode,
                                                                        hash_mode);
    } else if (getenv("DS4_CUDA_NO_PARALLEL_ROUTER_SELECT") == NULL) {
        router_select_parallel_kernel<<<n_tokens, 256>>>((int32_t *)selected->ptr,
                                                         (float *)weights->ptr,
                                                         (float *)probs->ptr,
                                                         bias,
                                                         hash,
                                                         (const float *)logits->ptr,
                                                         (const int32_t *)tokens->ptr,
                                                         0,
                                                         hash_rows,
                                                         n_tokens,
                                                         has_bias && !hash_mode,
                                                         hash_mode);
    } else {
        router_select_kernel<<<n_tokens, 1>>>((int32_t *)selected->ptr,
                                              (float *)weights->ptr,
                                              (float *)probs->ptr,
                                              bias,
                                              hash,
                                              (const float *)logits->ptr,
                                              (const int32_t *)tokens->ptr,
                                              0,
                                              hash_rows,
                                              n_tokens,
                                              has_bias && !hash_mode,
                                              hash_mode);
    }
    return cuda_ok(cudaGetLastError(), "router_select launch");
}

__device__ static float dev_f16_to_f32(uint16_t v) {
    return __half2float(*reinterpret_cast<const __half *>(&v));
}

__device__ __forceinline__ static uint32_t dev_unpack_iq2_signs(uint32_t v) {
    const uint32_t p = __popc(v) & 1u;
    const uint32_t s = v ^ (p << 7u);
    return s * 0x01010101u;
}

__device__ __forceinline__ static int32_t dev_iq2_dp4a_8(uint64_t grid, uint32_t sign, const int8_t *q8, int32_t acc) {
    const uint32_t signs = dev_unpack_iq2_signs(sign);
    const int32_t sm0 = __vcmpne4(signs & 0x08040201u, 0);
    const int32_t sm1 = __vcmpne4(signs & 0x80402010u, 0);
    const int32_t g0 = __vsub4((int32_t)(uint32_t)grid ^ sm0, sm0);
    const int32_t g1 = __vsub4((int32_t)(uint32_t)(grid >> 32) ^ sm1, sm1);
    acc = __dp4a(g0, *(const int32_t *)(q8 + 0), acc);
    acc = __dp4a(g1, *(const int32_t *)(q8 + 4), acc);
    return acc;
}

__device__ static int32_t dev_dot_q2_16(const uint8_t *q2, const int8_t *q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 16; i += 4) {
        const int32_t v = (*(const int32_t *)(q2 + i) >> shift) & 0x03030303;
        sum = __dp4a(v, *(const int32_t *)(q8 + i), sum);
    }
    return sum;
}

__device__ static int32_t dev_dot_iq2_pair_16(uint8_t grid0, uint32_t sign0, uint8_t grid1, uint32_t sign1, const int8_t *q8) {
    int32_t sum = 0;
    sum = dev_iq2_dp4a_8(cuda_iq2xxs_grid[grid0], cuda_ksigns_iq2xs[sign0], q8, sum);
    sum = dev_iq2_dp4a_8(cuda_iq2xxs_grid[grid1], cuda_ksigns_iq2xs[sign1], q8 + 8, sum);
    return sum;
}

template <bool COMPUTED_SIGNS>
__device__ __forceinline__ static void dev_iq2_i8x8_lut_impl(
        const uint64_t *grid,
        const uint8_t *signs,
        uint8_t grid_idx,
        uint32_t sign_idx,
        int32_t *w0,
        int32_t *w1) {
    /* cuda_ksigns_iq2xs[i] is i with its parity bit in bit 7.  Feeding the
     * seven-bit index directly to dev_unpack_iq2_signs produces the exact
     * same byte and avoids a random shared-memory load in the hot IQ2 path. */
    const uint32_t sign_code = COMPUTED_SIGNS ? sign_idx : signs[sign_idx];
    const uint32_t s = dev_unpack_iq2_signs(sign_code);
    const int32_t sm0 = __vcmpne4(s & 0x08040201u, 0);
    const int32_t sm1 = __vcmpne4(s & 0x80402010u, 0);
    const uint64_t g = grid[grid_idx];
    *w0 = __vsub4((int32_t)(uint32_t)g ^ sm0, sm0);
    *w1 = __vsub4((int32_t)(uint32_t)(g >> 32) ^ sm1, sm1);
}

template <bool COMPUTED_SIGNS>
__device__ static float dev_dot_iq2_xxs_q8_K_block_lut_impl(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y,
        const uint64_t *grid,
        const uint8_t *signs) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const int32_t ls = (int32_t)(2u * (aux1 >> 28) + 1u);
        int32_t w[8];
        dev_iq2_i8x8_lut_impl<COMPUTED_SIGNS>(grid, signs, (uint8_t)(aux0 & 0xffu),           (aux1 >> 0)  & 127u, &w[0], &w[1]);
        dev_iq2_i8x8_lut_impl<COMPUTED_SIGNS>(grid, signs, (uint8_t)((aux0 >> 8)  & 0xffu),   (aux1 >> 7)  & 127u, &w[2], &w[3]);
        dev_iq2_i8x8_lut_impl<COMPUTED_SIGNS>(grid, signs, (uint8_t)((aux0 >> 16) & 0xffu),   (aux1 >> 14) & 127u, &w[4], &w[5]);
        dev_iq2_i8x8_lut_impl<COMPUTED_SIGNS>(grid, signs, (uint8_t)((aux0 >> 24) & 0xffu),   (aux1 >> 21) & 127u, &w[6], &w[7]);
        int32_t sumi = 0;
        sumi = __dp4a(w[0], *(const int32_t *)(q8 + ib32 * 32u + 0),  sumi);
        sumi = __dp4a(w[1], *(const int32_t *)(q8 + ib32 * 32u + 4),  sumi);
        sumi = __dp4a(w[2], *(const int32_t *)(q8 + ib32 * 32u + 8),  sumi);
        sumi = __dp4a(w[3], *(const int32_t *)(q8 + ib32 * 32u + 12), sumi);
        sumi = __dp4a(w[4], *(const int32_t *)(q8 + ib32 * 32u + 16), sumi);
        sumi = __dp4a(w[5], *(const int32_t *)(q8 + ib32 * 32u + 20), sumi);
        sumi = __dp4a(w[6], *(const int32_t *)(q8 + ib32 * 32u + 24), sumi);
        sumi = __dp4a(w[7], *(const int32_t *)(q8 + ib32 * 32u + 28), sumi);
        bsum += sumi * ls;
    }
    return 0.125f * xd * y->d * (float)bsum;
}

__device__ static float dev_dot_iq2_xxs_q8_K_block_lut(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y,
        const uint64_t *grid,
        const uint8_t *signs) {
    return dev_dot_iq2_xxs_q8_K_block_lut_impl<false>(x, y, grid, signs);
}

__device__ static float dev_dot_iq2_xxs_q8_K_block_computed_signs(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y,
        const uint64_t *grid) {
    return dev_dot_iq2_xxs_q8_K_block_lut_impl<true>(x, y, grid, NULL);
}

__device__ static float dev_dot_iq2_xxs_q8_K_block(const cuda_block_iq2_xxs *x, const cuda_block_q8_K *y) {
    const float d = dev_f16_to_f32(x->d) * y->d;
    const uint16_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        const uint8_t a0 = (uint8_t)(aux0 & 0xffu);
        const uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
        const uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
        const uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
        int32_t sumi = 0;
        sumi += dev_dot_iq2_pair_16(a0, (aux1 >> 0) & 127u, a1, (aux1 >> 7) & 127u, q8);
        q8 += 16;
        sumi += dev_dot_iq2_pair_16(a2, (aux1 >> 14) & 127u, a3, (aux1 >> 21) & 127u, q8);
        q8 += 16;
        bsum += sumi * (int32_t)ls;
    }
    return 0.125f * d * (float)bsum;
}

/* Logical IQ2_XXS block read from the in-place aligned layout.  The repack
 * separates the half scale from the eight uint2 code words but keeps the raw
 * expert/row/block order, so this reproduces the established Q8_K decode dot
 * without rebuilding a 66-byte block or keeping a second weight layout. */
template <bool COMPUTED_SIGNS>
__device__ static float dev_dot_iq2_xxs_aligned_q8_K_block_lut_impl(
        const __half *dq,
        const uint2 *qs,
        uint64_t blk,
        const cuda_block_q8_K *y,
        const uint64_t *grid,
        const uint8_t *signs) {
    const float xd = __half2float(dq[blk]);
    const int8_t *q8 = y->qs;
    int32_t bsum = 0;
    #pragma unroll
    for (uint32_t ib32 = 0; ib32 < CUDA_QK_K / 32u; ib32++) {
        const uint2 code = qs[blk * 8u + ib32];
        const uint32_t aux0 = code.x;
        const uint32_t aux1 = code.y;
        const int32_t ls = (int32_t)(2u * (aux1 >> 28) + 1u);
        int32_t w[8];
        dev_iq2_i8x8_lut_impl<COMPUTED_SIGNS>(grid, signs,
                                     (uint8_t)(aux0 & 0xffu),
                                     (aux1 >> 0) & 127u, &w[0], &w[1]);
        dev_iq2_i8x8_lut_impl<COMPUTED_SIGNS>(grid, signs,
                                     (uint8_t)((aux0 >> 8) & 0xffu),
                                     (aux1 >> 7) & 127u, &w[2], &w[3]);
        dev_iq2_i8x8_lut_impl<COMPUTED_SIGNS>(grid, signs,
                                     (uint8_t)((aux0 >> 16) & 0xffu),
                                     (aux1 >> 14) & 127u, &w[4], &w[5]);
        dev_iq2_i8x8_lut_impl<COMPUTED_SIGNS>(grid, signs,
                                     (uint8_t)((aux0 >> 24) & 0xffu),
                                     (aux1 >> 21) & 127u, &w[6], &w[7]);
        int32_t sumi = 0;
        sumi = __dp4a(w[0], *(const int32_t *)(q8 + ib32 * 32u + 0u), sumi);
        sumi = __dp4a(w[1], *(const int32_t *)(q8 + ib32 * 32u + 4u), sumi);
        sumi = __dp4a(w[2], *(const int32_t *)(q8 + ib32 * 32u + 8u), sumi);
        sumi = __dp4a(w[3], *(const int32_t *)(q8 + ib32 * 32u + 12u), sumi);
        sumi = __dp4a(w[4], *(const int32_t *)(q8 + ib32 * 32u + 16u), sumi);
        sumi = __dp4a(w[5], *(const int32_t *)(q8 + ib32 * 32u + 20u), sumi);
        sumi = __dp4a(w[6], *(const int32_t *)(q8 + ib32 * 32u + 24u), sumi);
        sumi = __dp4a(w[7], *(const int32_t *)(q8 + ib32 * 32u + 28u), sumi);
        bsum += sumi * ls;
    }
    return 0.125f * xd * y->d * (float)bsum;
}

__device__ static float dev_dot_iq2_xxs_aligned_q8_K_block_lut(
        const __half *dq,
        const uint2 *qs,
        uint64_t blk,
        const cuda_block_q8_K *y,
        const uint64_t *grid,
        const uint8_t *signs) {
    return dev_dot_iq2_xxs_aligned_q8_K_block_lut_impl<false>(
            dq, qs, blk, y, grid, signs);
}

__device__ static float dev_dot_iq2_xxs_aligned_q8_K_block_computed_signs(
        const __half *dq,
        const uint2 *qs,
        uint64_t blk,
        const cuda_block_q8_K *y,
        const uint64_t *grid) {
    return dev_dot_iq2_xxs_aligned_q8_K_block_lut_impl<true>(
            dq, qs, blk, y, grid, NULL);
}

__device__ static void dev_dot_iq2_xxs_q8_K_block8_deq_lut(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8],
        const uint64_t *grid,
        const uint8_t *signs) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    int32_t bsum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int8_t *q8[8] = {
        y0 ? y0->qs : NULL, y1 ? y1->qs : NULL, y2 ? y2->qs : NULL, y3 ? y3->qs : NULL,
        y4 ? y4->qs : NULL, y5 ? y5->qs : NULL, y6 ? y6->qs : NULL, y7 ? y7->qs : NULL,
    };
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const int32_t ls = (int32_t)(2u * (aux1 >> 28) + 1u);
        int32_t w[8];
        dev_iq2_i8x8_lut_impl<false>(grid, signs, (uint8_t)(aux0 & 0xffu),           (aux1 >> 0)  & 127u, &w[0], &w[1]);
        dev_iq2_i8x8_lut_impl<false>(grid, signs, (uint8_t)((aux0 >> 8)  & 0xffu),   (aux1 >> 7)  & 127u, &w[2], &w[3]);
        dev_iq2_i8x8_lut_impl<false>(grid, signs, (uint8_t)((aux0 >> 16) & 0xffu),   (aux1 >> 14) & 127u, &w[4], &w[5]);
        dev_iq2_i8x8_lut_impl<false>(grid, signs, (uint8_t)((aux0 >> 24) & 0xffu),   (aux1 >> 21) & 127u, &w[6], &w[7]);
        for (uint32_t p = 0; p < n; p++) {
            const int8_t *q = q8[p] + ib32 * 32;
            int32_t sumi = 0;
            sumi = __dp4a(w[0], *(const int32_t *)(q + 0),  sumi);
            sumi = __dp4a(w[1], *(const int32_t *)(q + 4),  sumi);
            sumi = __dp4a(w[2], *(const int32_t *)(q + 8),  sumi);
            sumi = __dp4a(w[3], *(const int32_t *)(q + 12), sumi);
            sumi = __dp4a(w[4], *(const int32_t *)(q + 16), sumi);
            sumi = __dp4a(w[5], *(const int32_t *)(q + 20), sumi);
            sumi = __dp4a(w[6], *(const int32_t *)(q + 24), sumi);
            sumi = __dp4a(w[7], *(const int32_t *)(q + 28), sumi);
            bsum[p] += sumi * ls;
        }
    }
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    for (uint32_t p = 0; p < n; p++) acc[p] += 0.125f * xd * ys[p]->d * (float)bsum[p];
}

__device__ static void dev_dot_iq2_xxs_q8_K_block4(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        uint32_t n,
        float acc[4]) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    int32_t bsum[4] = {0, 0, 0, 0};
    const int8_t *q8[4] = {
        y0 ? y0->qs : NULL,
        y1 ? y1->qs : NULL,
        y2 ? y2->qs : NULL,
        y3 ? y3->qs : NULL,
    };
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        const uint8_t a0 = (uint8_t)(aux0 & 0xffu);
        const uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
        const uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
        const uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
        for (uint32_t p = 0; p < n; p++) {
            int32_t sumi = 0;
            sumi += dev_dot_iq2_pair_16(a0, (aux1 >> 0) & 127u, a1, (aux1 >> 7) & 127u, q8[p] + ib32 * 32);
            sumi += dev_dot_iq2_pair_16(a2, (aux1 >> 14) & 127u, a3, (aux1 >> 21) & 127u, q8[p] + ib32 * 32 + 16);
            bsum[p] += sumi * (int32_t)ls;
        }
    }
    const cuda_block_q8_K *ys[4] = { y0, y1, y2, y3 };
    for (uint32_t p = 0; p < n; p++) acc[p] += 0.125f * xd * ys[p]->d * (float)bsum[p];
}

__device__ static DS4_CUDA_UNUSED void dev_dot_iq2_xxs_q8_K_block8(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8]) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    int32_t bsum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int8_t *q8[8] = {
        y0 ? y0->qs : NULL, y1 ? y1->qs : NULL, y2 ? y2->qs : NULL, y3 ? y3->qs : NULL,
        y4 ? y4->qs : NULL, y5 ? y5->qs : NULL, y6 ? y6->qs : NULL, y7 ? y7->qs : NULL,
    };
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        const uint8_t a0 = (uint8_t)(aux0 & 0xffu);
        const uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
        const uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
        const uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
        for (uint32_t p = 0; p < n; p++) {
            int32_t sumi = 0;
            sumi += dev_dot_iq2_pair_16(a0, (aux1 >> 0) & 127u, a1, (aux1 >> 7) & 127u, q8[p] + ib32 * 32);
            sumi += dev_dot_iq2_pair_16(a2, (aux1 >> 14) & 127u, a3, (aux1 >> 21) & 127u, q8[p] + ib32 * 32 + 16);
            bsum[p] += sumi * (int32_t)ls;
        }
    }
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    for (uint32_t p = 0; p < n; p++) acc[p] += 0.125f * xd * ys[p]->d * (float)bsum[p];
}

__device__ static void dev_q4_K_get_scale_min(
        uint32_t j,
        const uint8_t *scales,
        uint8_t *d_out,
        uint8_t *m_out) {
    if (j < 4u) {
        *d_out = scales[j] & 63u;
        *m_out = scales[j + 4u] & 63u;
    } else {
        *d_out = (scales[j + 4u] & 0x0fu) | ((scales[j - 4u] >> 6u) << 4u);
        *m_out = (scales[j + 4u] >> 4u) | ((scales[j] >> 6u) << 4u);
    }
}

__device__ __forceinline__ static int32_t dev_dot_q4_32(const uint8_t *qs, const int8_t *q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        const int32_t v = (*(const int32_t *)(qs + i) >> shift) & 0x0f0f0f0f;
        sum = __dp4a(v, *(const int32_t *)(q8 + i), sum);
    }
    return sum;
}

__device__ static float dev_dot_q4_K_q8_K_block(const cuda_block_q4_K *x, const cuda_block_q8_K *y) {
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    int isum = 0;
    int summs = 0;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        dev_q4_K_get_scale_min(j, x->scales, &sc, &m);
        summs += (int)m * (int)(y->bsums[2u * j] + y->bsums[2u * j + 1u]);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        isum += (int)sc * dev_dot_q4_32(x->qs + byte_off, y->qs + j * 32u, shift);
    }
    return y->d * xd * (float)isum - y->d * xmin * (float)summs;
}

__device__ __forceinline__ static void dev_dot_q4_32_q8_K_block8(
        const uint8_t *qs,
        const cuda_block_q8_K *const ys[8],
        uint32_t n,
        uint32_t y_off,
        int shift,
        int32_t sums[8]) {
    #pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        const int32_t v = (*(const int32_t *)(qs + i) >> shift) & 0x0f0f0f0f;
        #pragma unroll
        for (uint32_t p = 0; p < 8u; p++) {
            if (p < n) sums[p] = __dp4a(v, *(const int32_t *)(ys[p]->qs + y_off + i), sums[p]);
        }
    }
}

__device__ __forceinline__ static void dev_dot_q4_32_q8_K_block8_full(
        const uint8_t *qs,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t y_off,
        int shift,
        int32_t sums[8]) {
    #pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        const int32_t v = (*(const int32_t *)(qs + i) >> shift) & 0x0f0f0f0f;
        sums[0] = __dp4a(v, *(const int32_t *)(y0->qs + y_off + i), sums[0]);
        sums[1] = __dp4a(v, *(const int32_t *)(y1->qs + y_off + i), sums[1]);
        sums[2] = __dp4a(v, *(const int32_t *)(y2->qs + y_off + i), sums[2]);
        sums[3] = __dp4a(v, *(const int32_t *)(y3->qs + y_off + i), sums[3]);
        sums[4] = __dp4a(v, *(const int32_t *)(y4->qs + y_off + i), sums[4]);
        sums[5] = __dp4a(v, *(const int32_t *)(y5->qs + y_off + i), sums[5]);
        sums[6] = __dp4a(v, *(const int32_t *)(y6->qs + y_off + i), sums[6]);
        sums[7] = __dp4a(v, *(const int32_t *)(y7->qs + y_off + i), sums[7]);
    }
}

__device__ static void dev_dot_q4_K_q8_K_block8(
        const cuda_block_q4_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8]) {
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    int isum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int summs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        dev_q4_K_get_scale_min(j, x->scales, &sc, &m);
        const uint32_t y_off = j * 32u;
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        int32_t dots[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        dev_dot_q4_32_q8_K_block8(x->qs + byte_off, ys, n, y_off, shift, dots);
        #pragma unroll
        for (uint32_t p = 0; p < 8u; p++) {
            if (p < n) {
                summs[p] += (int)m * (int)(ys[p]->bsums[2u * j] + ys[p]->bsums[2u * j + 1u]);
                isum[p] += (int)sc * dots[p];
            }
        }
    }
    #pragma unroll
    for (uint32_t p = 0; p < 8u; p++) {
        if (p < n) {
            const float yd = ys[p]->d;
            acc[p] += yd * xd * (float)isum[p] - yd * xmin * (float)summs[p];
        }
    }
}

__device__ static void dev_dot_q4_K_q8_K_block8_full(
        const cuda_block_q4_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        float acc[8]) {
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    int isum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int summs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        dev_q4_K_get_scale_min(j, x->scales, &sc, &m);
        const uint32_t y_off = j * 32u;
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        int32_t dots[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        dev_dot_q4_32_q8_K_block8_full(
            x->qs + byte_off,
            y0, y1, y2, y3, y4, y5, y6, y7,
            y_off,
            shift,
            dots);
        const int ms0 = (int)m * (int)(y0->bsums[2u * j] + y0->bsums[2u * j + 1u]);
        const int ms1 = (int)m * (int)(y1->bsums[2u * j] + y1->bsums[2u * j + 1u]);
        const int ms2 = (int)m * (int)(y2->bsums[2u * j] + y2->bsums[2u * j + 1u]);
        const int ms3 = (int)m * (int)(y3->bsums[2u * j] + y3->bsums[2u * j + 1u]);
        const int ms4 = (int)m * (int)(y4->bsums[2u * j] + y4->bsums[2u * j + 1u]);
        const int ms5 = (int)m * (int)(y5->bsums[2u * j] + y5->bsums[2u * j + 1u]);
        const int ms6 = (int)m * (int)(y6->bsums[2u * j] + y6->bsums[2u * j + 1u]);
        const int ms7 = (int)m * (int)(y7->bsums[2u * j] + y7->bsums[2u * j + 1u]);
        summs[0] += ms0;
        summs[1] += ms1;
        summs[2] += ms2;
        summs[3] += ms3;
        summs[4] += ms4;
        summs[5] += ms5;
        summs[6] += ms6;
        summs[7] += ms7;
        isum[0] += (int)sc * dots[0];
        isum[1] += (int)sc * dots[1];
        isum[2] += (int)sc * dots[2];
        isum[3] += (int)sc * dots[3];
        isum[4] += (int)sc * dots[4];
        isum[5] += (int)sc * dots[5];
        isum[6] += (int)sc * dots[6];
        isum[7] += (int)sc * dots[7];
    }
    acc[0] += y0->d * xd * (float)isum[0] - y0->d * xmin * (float)summs[0];
    acc[1] += y1->d * xd * (float)isum[1] - y1->d * xmin * (float)summs[1];
    acc[2] += y2->d * xd * (float)isum[2] - y2->d * xmin * (float)summs[2];
    acc[3] += y3->d * xd * (float)isum[3] - y3->d * xmin * (float)summs[3];
    acc[4] += y4->d * xd * (float)isum[4] - y4->d * xmin * (float)summs[4];
    acc[5] += y5->d * xd * (float)isum[5] - y5->d * xmin * (float)summs[5];
    acc[6] += y6->d * xd * (float)isum[6] - y6->d * xmin * (float)summs[6];
    acc[7] += y7->d * xd * (float)isum[7] - y7->d * xmin * (float)summs[7];
}

__device__ static float dev_dot_q2_K_q8_K_block(const cuda_block_q2_K *x, const cuda_block_q8_K *y) {
    const uint8_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    const uint8_t *sc = x->scales;
    int summs = 0;
    for (int j = 0; j < 16; j++) summs += y->bsums[j] * (sc[j] >> 4);
    const float dall = y->d * dev_f16_to_f32(x->d);
    const float dmin = y->d * dev_f16_to_f32(x->dmin);
    int isum = 0;
    int is = 0;
    for (int k = 0; k < CUDA_QK_K / 128; k++) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            int d = sc[is++] & 0x0f;
            isum += d * dev_dot_q2_16(q2, q8, shift);
            d = sc[is++] & 0x0f;
            isum += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
            shift += 2;
            q8 += 32;
        }
        q2 += 32;
    }
    return dall * (float)isum - dmin * (float)summs;
}

__device__ __forceinline__ static uint32_t dev_q2_aligned_row_word(
        const uint2 *words,
        uint64_t index,
        uint32_t parity) {
    const uint2 v = words[index];
    return parity ? v.y : v.x;
}

__device__ __forceinline__ static uint8_t dev_q2_aligned_scale_cached(
        uint32_t sc0,
        uint32_t sc1,
        uint32_t sc2,
        uint32_t sc3,
        uint32_t index) {
    const uint32_t word = index < 4u ? sc0 :
                          index < 8u ? sc1 :
                          index < 12u ? sc2 : sc3;
    return (uint8_t)(word >> ((index & 3u) * 8u));
}

__device__ static int32_t dev_dot_q2_aligned_16(
        const uint2 *qs2,
        uint64_t pblk,
        uint32_t parity,
        uint32_t word0,
        const int8_t *q8,
        int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 4u; i++) {
        const uint32_t q = dev_q2_aligned_row_word(
                qs2, pblk * 16u + word0 + i, parity);
        const int32_t v = ((int32_t)q >> shift) & 0x03030303;
        sum = __dp4a(v, *(const int32_t *)(q8 + i * 4u), sum);
    }
    return sum;
}

/* Q2_K x Q8_K with the same scalar chain as dev_dot_q2_K_q8_K_block, but
 * reading one logical row from the row-pair SoA artifact. */
__device__ static float dev_dot_q2_K_aligned_q8_K_block(
        const uint2 *dm2,
        const int4 *sc4,
        const uint2 *qs2,
        uint64_t pblk,
        uint32_t parity,
        const cuda_block_q8_K *y) {
    /* Each logical Q2_K block owns exactly 16 scale bytes. Keep their four
     * packed words in registers instead of reloading the two int4 records for
     * every bsums and dot-product term. */
    const int4 scales_lo = sc4[pblk * 2u];
    const int4 scales_hi = sc4[pblk * 2u + 1u];
    const uint32_t sc0 = parity ? (uint32_t)scales_lo.z : (uint32_t)scales_lo.x;
    const uint32_t sc1 = parity ? (uint32_t)scales_lo.w : (uint32_t)scales_lo.y;
    const uint32_t sc2 = parity ? (uint32_t)scales_hi.z : (uint32_t)scales_hi.x;
    const uint32_t sc3 = parity ? (uint32_t)scales_hi.w : (uint32_t)scales_hi.y;
    int summs = 0;
    #pragma unroll
    for (uint32_t j = 0; j < 16u; j++) {
        summs += y->bsums[j] *
                 (dev_q2_aligned_scale_cached(sc0, sc1, sc2, sc3, j) >> 4u);
    }

    const uint2 dm_pair = dm2[pblk];
    const uint32_t dm = parity ? dm_pair.y : dm_pair.x;
    const float dall = y->d * dev_f16_to_f32((uint16_t)(dm & 0xffffu));
    const float dmin = y->d * dev_f16_to_f32((uint16_t)(dm >> 16u));
    const int8_t *q8 = y->qs;
    int isum = 0;
    uint32_t is = 0;
    #pragma unroll
    for (uint32_t k = 0; k < CUDA_QK_K / 128u; k++) {
        int shift = 0;
        #pragma unroll
        for (uint32_t j = 0; j < 4u; j++) {
            int d = dev_q2_aligned_scale_cached(
                    sc0, sc1, sc2, sc3, is++) & 0x0f;
            isum += d * dev_dot_q2_aligned_16(
                    qs2, pblk, parity, k * 8u, q8, shift);
            d = dev_q2_aligned_scale_cached(
                    sc0, sc1, sc2, sc3, is++) & 0x0f;
            isum += d * dev_dot_q2_aligned_16(
                    qs2, pblk, parity, k * 8u + 4u, q8 + 16u, shift);
            shift += 2;
            q8 += 32u;
        }
    }
    return dall * (float)isum - dmin * (float)summs;
}

__device__ static void dev_dot_q2_K_q8_K_block4(
        const cuda_block_q2_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        uint32_t n,
        float acc[4]) {
    const uint8_t *sc = x->scales;
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    const cuda_block_q8_K *ys[4] = { y0, y1, y2, y3 };
    int isum[4] = {0, 0, 0, 0};
    int summs[4] = {0, 0, 0, 0};
    for (uint32_t p = 0; p < n; p++) {
        for (int j = 0; j < 16; j++) summs[p] += ys[p]->bsums[j] * (sc[j] >> 4);
    }
    for (uint32_t p = 0; p < n; p++) {
        const uint8_t *q2 = x->qs;
        const int8_t *q8 = ys[p]->qs;
        int is = 0;
        for (int k = 0; k < CUDA_QK_K / 128; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2, q8, shift);
                d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        const float yd = ys[p]->d;
        acc[p] += yd * xd * (float)isum[p] - yd * xmin * (float)summs[p];
    }
}

__device__ static void dev_dot_q2_K_q8_K_block8(
        const cuda_block_q2_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8]) {
    const uint8_t *sc = x->scales;
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    int isum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int summs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t p = 0; p < n; p++) {
        for (int j = 0; j < 16; j++) summs[p] += ys[p]->bsums[j] * (sc[j] >> 4);
    }
    for (uint32_t p = 0; p < n; p++) {
        const uint8_t *q2 = x->qs;
        const int8_t *q8 = ys[p]->qs;
        int is = 0;
        for (int k = 0; k < CUDA_QK_K / 128; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2, q8, shift);
                d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        const float yd = ys[p]->d;
        acc[p] += yd * xd * (float)isum[p] - yd * xmin * (float)summs[p];
    }
}

__device__ static float half_warp_sum_f32(float v, uint32_t lane16) {
    uint32_t mask = 0xffffu << (threadIdx.x & 16u);
    for (int offset = 8; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset, 16);
    }
    (void)lane16;
    return v;
}

__device__ static float quarter_warp_sum_f32(float v, uint32_t lane8) {
    uint32_t mask = 0xffu << (threadIdx.x & 24u);
    for (int offset = 4; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset, 8);
    }
    (void)lane8;
    return v;
}

__global__ static void q8_K_quantize_kernel(cuda_block_q8_K *out, const float *x, uint32_t in_dim, uint32_t n_rows) {
    uint32_t b = blockIdx.x;
    uint32_t row = blockIdx.y;
    if (row >= n_rows || b >= in_dim / CUDA_QK_K) return;
    const float *xr = x + (uint64_t)row * in_dim + (uint64_t)b * CUDA_QK_K;
    cuda_block_q8_K *yb = out + (uint64_t)row * (in_dim / CUDA_QK_K) + b;
    __shared__ float abs_part[256];
    __shared__ float val_part[256];
    __shared__ float maxv_s;
    __shared__ float iscale_s;
    uint32_t tid = threadIdx.x;
    float v = tid < CUDA_QK_K ? xr[tid] : 0.0f;
    abs_part[tid] = tid < CUDA_QK_K ? fabsf(v) : 0.0f;
    val_part[tid] = v;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && abs_part[tid + stride] > abs_part[tid]) {
            abs_part[tid] = abs_part[tid + stride];
            val_part[tid] = val_part[tid + stride];
        }
        __syncthreads();
    }
    float amax = abs_part[0];
    if (amax == 0.0f) {
        if (tid == 0) yb->d = 0.0f;
        if (tid < CUDA_QK_K) yb->qs[tid] = 0;
        if (tid < CUDA_QK_K / 16) yb->bsums[tid] = 0;
        return;
    }
    if (tid == 0) {
        maxv_s = val_part[0];
        iscale_s = -127.0f / maxv_s;
    }
    __syncthreads();
    if (tid < CUDA_QK_K) {
        int qv = (int)lrintf(iscale_s * xr[tid]);
        if (qv > 127) qv = 127;
        if (qv < -128) qv = -128;
        yb->qs[tid] = (int8_t)qv;
    }
    __syncthreads();
    if (tid < CUDA_QK_K / 16) {
        int sum = 0;
        for (int i = 0; i < 16; i++) sum += yb->qs[tid * 16 + i];
        yb->bsums[tid] = (int16_t)sum;
    }
    if (tid == 0) yb->d = 1.0f / iscale_s;
}

/* Small-batch target MoE on the aligned in-place weights.  This deliberately
 * retains the production Q8_K activation format and quarter-warp reduction
 * used before the SoA loader was introduced.  The prefill D2R/MMQ path has a
 * separate dispatch and never enters these kernels. */
__global__ static void moe_gate_up_mid_aligned_q8K_qwarp32_kernel(
        float *mid_out,
        const __half *gate_dq,
        const uint2 *gate_qs,
        const __half *up_dq,
        const uint2 *up_qs,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_total_expert,
        uint32_t n_expert,
        float clamp) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row_lane = threadIdx.x >> 3u;
    const uint32_t pair = blockIdx.y;
    const uint32_t tok = pair / n_expert;
    const int32_t expert_i = selected[pair];
    const bool valid = expert_i >= 0 && (uint32_t)expert_i < n_total_expert;
    const uint32_t expert = valid ? (uint32_t)expert_i : 0u;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;

    __shared__ cuda_block_q8_K sxq[16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < xq_blocks; i += blockDim.x) {
            sxq[i] = xqb[i];
        }
    }
    for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) {
        s_iq2_grid[i] = cuda_iq2xxs_grid[i];
    }
    for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) {
        s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
    }
    __syncthreads();
    if (xq_blocks <= 16u) xqb = sxq;

    #pragma unroll 1
    for (uint32_t rr = 0; rr < 4u; rr++) {
        const uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; valid && b < xq_blocks; b += 8u) {
            const uint64_t blk =
                ((uint64_t)expert * expert_mid_dim + row) * xq_blocks + b;
            gate += dev_dot_iq2_xxs_aligned_q8_K_block_lut(
                    gate_dq, gate_qs, blk, xqb + b,
                    s_iq2_grid, s_iq2_signs);
            up += dev_dot_iq2_xxs_aligned_q8_K_block_lut(
                    up_dq, up_qs, blk, xqb + b,
                    s_iq2_grid, s_iq2_signs);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0u) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[pair];
        }
    }
}

/* GB10 small-batch variant of the aligned gate/up kernel. It preserves the
 * quarter-warp dot/reduction order, while coalescing Q8_K staging, deriving
 * the validated IQ2 signs in registers, and amortizing one CTA over 256 rows. */
__global__ static void moe_gate_up_mid_aligned_q8K_gb10_kernel(
        float *mid_out,
        const __half *gate_dq,
        const uint2 *gate_qs,
        const __half *up_dq,
        const uint2 *up_qs,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_total_expert,
        uint32_t n_expert,
        float clamp) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row_lane = threadIdx.x >> 3u;
    const uint32_t pair = blockIdx.y;
    const uint32_t tok = pair / n_expert;
    const int32_t expert_i = selected[pair];
    const bool valid = expert_i >= 0 && (uint32_t)expert_i < n_total_expert;
    const uint32_t expert = valid ? (uint32_t)expert_i : 0u;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;

    __shared__ uint32_t sxq_words[
        16u * sizeof(cuda_block_q8_K) / sizeof(uint32_t)];
    __shared__ uint64_t s_iq2_grid[256];
    if (xq_blocks <= 16u) {
        const uint32_t nwords =
            xq_blocks * sizeof(cuda_block_q8_K) / sizeof(uint32_t);
        const uint32_t *src = reinterpret_cast<const uint32_t *>(xqb);
        for (uint32_t i = threadIdx.x; i < nwords; i += blockDim.x) {
            sxq_words[i] = src[i];
        }
    }
    for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) {
        s_iq2_grid[i] = cuda_iq2xxs_grid_global[i];
    }
    __syncthreads();
    if (xq_blocks <= 16u) {
        xqb = reinterpret_cast<const cuda_block_q8_K *>(sxq_words);
    }

    #pragma unroll 1
    for (uint32_t rr = 0; rr < 8u; rr++) {
        const uint32_t row = blockIdx.x * 256u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; valid && b < xq_blocks; b += 8u) {
            const uint64_t blk =
                ((uint64_t)expert * expert_mid_dim + row) * xq_blocks + b;
            gate += dev_dot_iq2_xxs_aligned_q8_K_block_computed_signs(
                    gate_dq, gate_qs, blk, xqb + b, s_iq2_grid);
            up += dev_dot_iq2_xxs_aligned_q8_K_block_computed_signs(
                    up_dq, up_qs, blk, xqb + b, s_iq2_grid);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0u) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[pair];
        }
    }
}

__global__ static void moe_down_aligned_q8K_sum6_qwarp32_kernel(
        float *out,
        const uint2 *dm2,
        const int4 *sc4,
        const uint2 *qs2,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_tokens,
        uint32_t n_total_expert) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    const uint32_t tok = blockIdx.y;
    if (row >= out_dim || tok >= n_tokens) return;

    const uint32_t row_pair = row >> 1u;
    const uint32_t parity = row & 1u;
    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < 6u; slot++) {
        const uint32_t assignment = tok * 6u + slot;
        const int32_t expert_i = selected[assignment];
        const bool valid = expert_i >= 0 && (uint32_t)expert_i < n_total_expert;
        const uint32_t expert = valid ? (uint32_t)expert_i : 0u;
        const cuda_block_q8_K *xq = midq + (uint64_t)assignment * midq_blocks;
        float acc = 0.0f;
        for (uint32_t b = lane; valid && b < midq_blocks; b += 8u) {
            const uint64_t pblk =
                ((uint64_t)expert * (out_dim / 2u) + row_pair) * midq_blocks + b;
            acc += dev_dot_q2_K_aligned_q8_K_block(
                    dm2, sc4, qs2, pblk, parity, xq + b);
        }
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0u) total += acc;
    }
    if (lane == 0u) out[(uint64_t)tok * out_dim + row] = total;
}

/* Every output row in a token CTA consumes the same six Q8_K activation
 * vectors. Stage them once instead of asking all 32 quarter-warps to reload
 * them independently. Two 32-row waves reuse the same staging, while dynamic
 * shared memory keeps the production footprint at 6 * 8 Q8_K blocks for
 * Flash and retains a global-memory fallback. */
__global__ static void moe_down_aligned_q8K_sum6_gb10_kernel(
        float *out,
        const uint2 *dm2,
        const int4 *sc4,
        const uint2 *qs2,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_tokens,
        uint32_t n_total_expert) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row_lane = threadIdx.x >> 3u;
    const uint32_t tok = blockIdx.y;
    if (tok >= n_tokens) return;

    const cuda_block_q8_K *token_midq =
        midq + (uint64_t)tok * 6u * midq_blocks;
    extern __shared__ uint32_t smidq_words[];
    if (midq_blocks <= 16u) {
        const uint32_t nwords = 6u * midq_blocks *
            sizeof(cuda_block_q8_K) / sizeof(uint32_t);
        const uint32_t *src = reinterpret_cast<const uint32_t *>(token_midq);
        for (uint32_t i = threadIdx.x; i < nwords; i += blockDim.x) {
            smidq_words[i] = src[i];
        }
    }
    __syncthreads();
    if (midq_blocks <= 16u) {
        token_midq = reinterpret_cast<const cuda_block_q8_K *>(smidq_words);
    }

    #pragma unroll 1
    for (uint32_t rr = 0; rr < 2u; rr++) {
        const uint32_t row = blockIdx.x * 64u + row_lane + rr * 32u;
        if (row >= out_dim) continue;
        const uint32_t row_pair = row >> 1u;
        const uint32_t parity = row & 1u;
        float total = 0.0f;
        #pragma unroll
        for (uint32_t slot = 0; slot < 6u; slot++) {
            const uint32_t assignment = tok * 6u + slot;
            const int32_t expert_i = selected[assignment];
            const bool valid =
                expert_i >= 0 && (uint32_t)expert_i < n_total_expert;
            const uint32_t expert = valid ? (uint32_t)expert_i : 0u;
            const cuda_block_q8_K *xq =
                token_midq + (uint64_t)slot * midq_blocks;
            float acc = 0.0f;
            for (uint32_t b = lane; valid && b < midq_blocks; b += 8u) {
                const uint64_t pblk =
                    ((uint64_t)expert * (out_dim / 2u) + row_pair) *
                    midq_blocks + b;
                acc += dev_dot_q2_K_aligned_q8_K_block(
                        dm2, sc4, qs2, pblk, parity, xq + b);
            }
            acc = quarter_warp_sum_f32(acc, lane);
            if (lane == 0u) total += acc;
        }
        if (lane == 0u) out[(uint64_t)tok * out_dim + row] = total;
    }
}

__global__ static DS4_CUDA_UNUSED void moe_gate_up_mid_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = threadIdx.x; b < xq_blocks; b += blockDim.x) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    __shared__ float partial_gate[256];
    __shared__ float partial_up[256];
    partial_gate[threadIdx.x] = gate;
    partial_up[threadIdx.x] = up;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_gate[threadIdx.x] += partial_gate[threadIdx.x + stride];
            partial_up[threadIdx.x] += partial_up[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        gate = partial_gate[0];
        up = partial_up[0];
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static DS4_CUDA_UNUSED void moe_gate_up_mid_warp8_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 31u;
    uint32_t warp = threadIdx.x >> 5u;
    uint32_t row = blockIdx.x * 8u + warp;
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 32u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = warp_sum_f32(gate);
    up = warp_sum_f32(up);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static DS4_CUDA_UNUSED void moe_gate_up_mid_hwarp16_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 15u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 16u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = half_warp_sum_f32(gate, lane);
    up = half_warp_sum_f32(up, lane);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static void moe_gate_up_mid_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    for (uint32_t rr = 0; rr < 4u; rr++) {
        uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
            up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            gate_out[off] = gate;
            up_out[off] = up;
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

__global__ static void moe_gate_up_mid_decode_lut_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    __shared__ cuda_block_q8_K sxq[16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < xq_blocks; i += blockDim.x) sxq[i] = xqb[i];
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
        xqb = sxq;
    }
    for (uint32_t rr = 0; rr < 4u; rr++) {
        uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_iq2_xxs_q8_K_block_lut(gr + b, xqb + b, s_iq2_grid, s_iq2_signs);
            up += dev_dot_iq2_xxs_q8_K_block_lut(ur + b, xqb + b, s_iq2_grid, s_iq2_signs);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate;
                up_out[off] = up;
            }
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

__global__ static void moe_gate_up_mid_decode_gb10_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row_lane = threadIdx.x >> 3u;
    const uint32_t pair = blockIdx.y;
    const uint32_t tok = pair / n_expert;
    const uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;

    /* Flatten the Q8 activation copy so every warp reads consecutive words.
     * The baseline assigns one complete 292-byte block to a single thread. */
    __shared__ uint32_t sxq_words[
        16u * sizeof(cuda_block_q8_K) / sizeof(uint32_t)];
    __shared__ uint64_t s_iq2_grid[256];
    if (xq_blocks <= 16u) {
        const uint32_t nwords =
            xq_blocks * sizeof(cuda_block_q8_K) / sizeof(uint32_t);
        const uint32_t *src = reinterpret_cast<const uint32_t *>(xqb);
        for (uint32_t i = threadIdx.x; i < nwords; i += blockDim.x) {
            sxq_words[i] = src[i];
        }
    }
    for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) {
        s_iq2_grid[i] = cuda_iq2xxs_grid_global[i];
    }
    __syncthreads();
    if (xq_blocks <= 16u) {
        xqb = reinterpret_cast<const cuda_block_q8_K *>(sxq_words);
    }

#pragma unroll 1
    for (uint32_t rr = 0; rr < 8u; rr++) {
        const uint32_t row = blockIdx.x * 256u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr =
            (const cuda_block_iq2_xxs *)(gate_base +
                (uint64_t)expert * gate_expert_bytes +
                (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur =
            (const cuda_block_iq2_xxs *)(up_base +
                (uint64_t)expert * gate_expert_bytes +
                (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_iq2_xxs_q8_K_block_computed_signs(
                gr + b, xqb + b, s_iq2_grid);
            up += dev_dot_iq2_xxs_q8_K_block_computed_signs(
                ur + b, xqb + b, s_iq2_grid);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate;
                up_out[off] = up;
            }
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up *
                           weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

__global__ static void moe_count_sorted_pairs_kernel(
        uint32_t *counts,
        const int32_t *selected,
        uint32_t pair_count) {
    uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t expert_i = selected[pair];
    if (expert_i < 0) expert_i = 0;
    atomicAdd(counts + (uint32_t)expert_i, 1u);
}

__global__ static void moe_prefix_sorted_pairs_kernel(
        uint32_t *offsets,
        uint32_t *cursors,
        const uint32_t *counts,
        uint32_t expert_count) {
    if (threadIdx.x == 0) {
        uint32_t sum = 0;
        for (uint32_t e = 0; e < expert_count; e++) {
            offsets[e] = sum;
            cursors[e] = sum;
            sum += counts[e];
        }
        offsets[expert_count] = sum;
    }
}

__global__ static void moe_scatter_sorted_pairs_kernel(
        uint32_t *sorted_pairs,
        uint32_t *cursors,
        const int32_t *selected,
        uint32_t pair_count) {
    uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t expert_i = selected[pair];
    if (expert_i < 0) expert_i = 0;
    uint32_t pos = atomicAdd(cursors + (uint32_t)expert_i, 1u);
    sorted_pairs[pos] = pair;
}

__global__ static void moe_build_expert_tile_offsets_kernel(
        uint32_t *tile_offsets,
        uint32_t *tile_total,
        const uint32_t *counts,
        uint32_t expert_count,
        uint32_t block_m) {
    if (threadIdx.x == 0) {
        uint32_t sum = 0;
        for (uint32_t e = 0; e < expert_count; e++) {
            tile_offsets[e] = sum;
            sum += (counts[e] + block_m - 1u) / block_m;
        }
        tile_offsets[expert_count] = sum;
        *tile_total = sum;
    }
}

__global__ static void moe_build_expert_tiles_kernel(
        uint32_t *tile_experts,
        uint32_t *tile_starts,
        const uint32_t *tile_offsets,
        const uint32_t *counts,
        uint32_t expert_count,
        uint32_t block_m) {
    uint32_t e = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (e >= expert_count) return;
    uint32_t ntiles = (counts[e] + block_m - 1u) / block_m;
    uint32_t off = tile_offsets[e];
    for (uint32_t t = 0; t < ntiles; t++) {
        tile_experts[off + t] = e;
        tile_starts[off + t] = t * block_m;
    }
}

__global__ static void moe_gate_up_mid_sorted_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = sorted_pairs[blockIdx.y];
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = quarter_warp_sum_f32(gate, lane);
    up = quarter_warp_sum_f32(up, lane);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static DS4_CUDA_UNUSED void moe_gate_up_mid_expert_tile8_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t group = threadIdx.x >> 3u;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_slot = group & 7u;
    uint32_t row_lane = group >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_pair = tile_starts[tile] + pair_slot;
    if (local_pair >= counts[expert]) return;
    uint32_t sorted_idx = offsets[expert] + local_pair;
    uint32_t pair = sorted_pairs[sorted_idx];
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;

    for (uint32_t rr = 0; rr < 2u; rr++) {
        uint32_t row = blockIdx.x * 8u + row_lane + rr * 4u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
            up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            gate_out[off] = gate;
            up_out[off] = up;
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

__global__ static void moe_gate_up_mid_expert_tile4_row32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[4][16];
    uint32_t pair[4] = {0, 0, 0, 0};
    uint32_t tok[4] = {0, 0, 0, 0};
    uint32_t slot[4] = {0, 0, 0, 0};
    const cuda_block_q8_K *xqb[4] = {NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 4u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= expert_mid_dim) return;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        dev_dot_iq2_xxs_q8_K_block4(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                    xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, gate);
        dev_dot_iq2_xxs_q8_K_block4(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                    xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, up);
    }
    for (uint32_t p = 0; p < np; p++) {
        gate[p] = quarter_warp_sum_f32(gate[p], lane);
        up[p] = quarter_warp_sum_f32(up[p], lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate[p] > clamp) gate[p] = clamp;
                if (up[p] > clamp) up[p] = clamp;
                if (up[p] < -clamp) up[p] = -clamp;
            }
            const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate[p];
                up_out[off] = up[p];
            }
            mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
        }
    }
}

__global__ static void moe_gate_up_mid_expert_tile8_row32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= expert_mid_dim) return;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        dev_dot_iq2_xxs_q8_K_block8_deq_lut(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                            xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                            xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                            xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate,
                                            s_iq2_grid, s_iq2_signs);
        dev_dot_iq2_xxs_q8_K_block8_deq_lut(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                            xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                            xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                            xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up,
                                            s_iq2_grid, s_iq2_signs);
    }
    for (uint32_t p = 0; p < np; p++) {
        gate[p] = quarter_warp_sum_f32(gate[p], lane);
        up[p] = quarter_warp_sum_f32(up[p], lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate[p] > clamp) gate[p] = clamp;
                if (up[p] > clamp) up[p] = clamp;
                if (up[p] < -clamp) up[p] = -clamp;
            }
            const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate[p];
                up_out[off] = up[p];
            }
            mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
        }
    }
}

__global__ static void moe_gate_up_mid_expert_tile8_row2048_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < 64u; rr++) {
        uint32_t row = blockIdx.x * 2048u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate,
                                                s_iq2_grid, s_iq2_signs);
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up,
                                                s_iq2_grid, s_iq2_signs);
        }
        for (uint32_t p = 0; p < np; p++) {
            gate[p] = quarter_warp_sum_f32(gate[p], lane);
            up[p] = quarter_warp_sum_f32(up[p], lane);
            if (lane == 0) {
                if (clamp > 1.0e-6f) {
                    if (gate[p] > clamp) gate[p] = clamp;
                    if (up[p] > clamp) up[p] = clamp;
                    if (up[p] < -clamp) up[p] = -clamp;
                }
                const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
                if (write_aux) {
                    gate_out[off] = gate[p];
                    up_out[off] = up[p];
                }
                mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
            }
        }
    }
}

template <uint32_t ROW_SPAN>
__global__ static void moe_gate_up_mid_expert_tile8_rowspan_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate,
                                                s_iq2_grid, s_iq2_signs);
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up,
                                                s_iq2_grid, s_iq2_signs);
        }
        for (uint32_t p = 0; p < np; p++) {
            gate[p] = quarter_warp_sum_f32(gate[p], lane);
            up[p] = quarter_warp_sum_f32(up[p], lane);
            if (lane == 0) {
                if (clamp > 1.0e-6f) {
                    if (gate[p] > clamp) gate[p] = clamp;
                    if (up[p] > clamp) up[p] = clamp;
                    if (up[p] < -clamp) up[p] = -clamp;
                }
                const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
                if (write_aux) {
                    gate_out[off] = gate[p];
                    up_out[off] = up[p];
                }
                mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
            }
        }
    }
}

__global__ static void moe_gate_up_mid_sorted_p2_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t pair_count,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_lane = (threadIdx.x >> 3u) & 1u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t sorted_idx = blockIdx.y * 2u + pair_lane;
    if (row >= expert_mid_dim || sorted_idx >= pair_count) return;
    uint32_t pair = sorted_pairs[sorted_idx];
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = quarter_warp_sum_f32(gate, lane);
    up = quarter_warp_sum_f32(up, lane);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static DS4_CUDA_UNUSED void moe_down_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = threadIdx.x; b < midq_blocks; b += blockDim.x) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) down_out[(uint64_t)pair * out_dim + row] = partial[0];
}

__global__ static DS4_CUDA_UNUSED void moe_down_warp8_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 31u;
    uint32_t warp = threadIdx.x >> 5u;
    uint32_t row = blockIdx.x * 8u + warp;
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 32u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = warp_sum_f32(acc);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static DS4_CUDA_UNUSED void moe_down_hwarp16_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 15u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 16u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = half_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static void moe_down_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static void moe_gate_up_mid_q4K_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    for (uint32_t rr = 0; rr < 4u; rr++) {
        uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_q4_K_q8_K_block(gr + b, xqb + b);
            up += dev_dot_q4_K_q8_K_block(ur + b, xqb + b);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate;
                up_out[off] = up;
            }
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

template <uint32_t ROW_SPAN>
__global__ static void moe_gate_up_mid_q4K_expert_tile8_rowspan_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            if (np == 8u) {
                dev_dot_q4_K_q8_K_block8_full(gr + b, xqb[0] + b, xqb[1] + b,
                                              xqb[2] + b, xqb[3] + b,
                                              xqb[4] + b, xqb[5] + b,
                                              xqb[6] + b, xqb[7] + b, gate);
                dev_dot_q4_K_q8_K_block8_full(ur + b, xqb[0] + b, xqb[1] + b,
                                              xqb[2] + b, xqb[3] + b,
                                              xqb[4] + b, xqb[5] + b,
                                              xqb[6] + b, xqb[7] + b, up);
            } else {
                dev_dot_q4_K_q8_K_block8(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                         xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                         xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                         xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate);
                dev_dot_q4_K_q8_K_block8(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                         xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                         xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                         xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up);
            }
        }
        for (uint32_t p = 0; p < np; p++) {
            gate[p] = quarter_warp_sum_f32(gate[p], lane);
            up[p] = quarter_warp_sum_f32(up[p], lane);
            if (lane == 0) {
                if (clamp > 1.0e-6f) {
                    if (gate[p] > clamp) gate[p] = clamp;
                    if (up[p] > clamp) up[p] = clamp;
                    if (up[p] < -clamp) up[p] = -clamp;
                }
                const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
                if (write_aux) {
                    gate_out[off] = gate[p];
                    up_out[off] = up[p];
                }
                mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
            }
        }
    }
}

__global__ static void moe_down_sum6_qwarp32_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    if (row >= out_dim) return;
    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < 6u; slot++) {
        int32_t expert_i = selected[slot];
        if (expert_i < 0) expert_i = 0;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc;
    }
    if (lane == 0) out[row] = total;
}

__global__ static void moe_down_q4K_sum6_qwarp32_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    if (row >= out_dim) return;
    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < 6u; slot++) {
        int32_t expert_i = selected[slot];
        if (expert_i < 0) expert_i = 0;
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc;
    }
    if (lane == 0) out[row] = total;
}

__global__ static void moe_down_q4K_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

template <uint32_t ROW_SPAN>
__global__ static void moe_down_q4K_expert_tile8_rowspan_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][8];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= out_dim) continue;
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            if (np == 8u) {
                dev_dot_q4_K_q8_K_block8_full(wr + b, xqb[0] + b, xqb[1] + b,
                                              xqb[2] + b, xqb[3] + b,
                                              xqb[4] + b, xqb[5] + b,
                                              xqb[6] + b, xqb[7] + b, acc);
            } else {
                dev_dot_q4_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                         xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                         xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                         xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, acc);
            }
        }
        for (uint32_t p = 0; p < np; p++) {
            acc[p] = quarter_warp_sum_f32(acc[p], lane);
            if (lane == 0) {
                if (atomic_out) {
                    uint32_t tok = pair[p] / n_expert;
                    atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
                } else {
                    down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
                }
            }
        }
    }
}

template <uint32_t ROW_SPAN>
__global__ static void moe_down_q4K_expert_tile16_rowspan_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t local_start = tile_starts[tile];
    if (local_start & 8u) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    __shared__ cuda_block_q8_K sxq[16][8];
    uint32_t pair[16] = {0};
    const cuda_block_q8_K *xqb[16] = {NULL};
    uint32_t np = 0;
    for (; np < 16u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= out_dim) continue;
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc[16] = {0.0f};
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            if (np >= 8u) {
                dev_dot_q4_K_q8_K_block8_full(wr + b, xqb[0] + b, xqb[1] + b,
                                              xqb[2] + b, xqb[3] + b,
                                              xqb[4] + b, xqb[5] + b,
                                              xqb[6] + b, xqb[7] + b, acc);
            } else {
                dev_dot_q4_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                         xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                         xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                         xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, acc);
            }
            if (np > 8u) {
                if (np == 16u) {
                    dev_dot_q4_K_q8_K_block8_full(wr + b, xqb[8] + b, xqb[9] + b,
                                                  xqb[10] + b, xqb[11] + b,
                                                  xqb[12] + b, xqb[13] + b,
                                                  xqb[14] + b, xqb[15] + b, acc + 8);
                } else {
                    dev_dot_q4_K_q8_K_block8(wr + b, xqb[8] ? xqb[8] + b : NULL, xqb[9] ? xqb[9] + b : NULL,
                                             xqb[10] ? xqb[10] + b : NULL, xqb[11] ? xqb[11] + b : NULL,
                                             xqb[12] ? xqb[12] + b : NULL, xqb[13] ? xqb[13] + b : NULL,
                                             xqb[14] ? xqb[14] + b : NULL, xqb[15] ? xqb[15] + b : NULL, np - 8u, acc + 8);
                }
            }
        }
        for (uint32_t p = 0; p < np; p++) {
            acc[p] = quarter_warp_sum_f32(acc[p], lane);
            if (lane == 0) {
                if (atomic_out) {
                    uint32_t tok = pair[p] / n_expert;
                    atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
                } else {
                    down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
                }
            }
        }
    }
}

__global__ static void moe_down_sorted_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = sorted_pairs[blockIdx.y];
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static DS4_CUDA_UNUSED void moe_down_expert_tile8_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t group = threadIdx.x >> 3u;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_slot = group & 7u;
    uint32_t row_lane = group >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_pair = tile_starts[tile] + pair_slot;
    if (local_pair >= counts[expert]) return;
    uint32_t sorted_idx = offsets[expert] + local_pair;
    uint32_t pair = sorted_pairs[sorted_idx];
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;

    for (uint32_t rr = 0; rr < 2u; rr++) {
        uint32_t row = blockIdx.x * 8u + row_lane + rr * 4u;
        if (row >= out_dim) continue;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
    }
}

__global__ static void moe_down_expert_tile4_row32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[4][8];
    uint32_t pair[4] = {0, 0, 0, 0};
    const cuda_block_q8_K *xqb[4] = {NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 4u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= out_dim) return;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_q2_K_q8_K_block4(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, acc);
    }
    for (uint32_t p = 0; p < np; p++) {
        acc[p] = quarter_warp_sum_f32(acc[p], lane);
        if (lane == 0) {
            if (atomic_out) {
                uint32_t tok = pair[p] / n_expert;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
            } else {
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
}

__global__ static void moe_down_expert_tile8_row32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][8];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= out_dim) return;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                 xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                 xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, acc);
    }
    for (uint32_t p = 0; p < np; p++) {
        acc[p] = quarter_warp_sum_f32(acc[p], lane);
        if (lane == 0) {
            if (atomic_out) {
                uint32_t tok = pair[p] / n_expert;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
            } else {
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
}

__global__ static void moe_down_expert_tile16_row32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t local_start = tile_starts[tile];
    if (local_start & 8u) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    __shared__ cuda_block_q8_K sxq[16][8];
    uint32_t pair[16] = {0};
    const cuda_block_q8_K *xqb[16] = {NULL};
    uint32_t np = 0;
    for (; np < 16u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= out_dim) return;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    float acc[16] = {0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                 xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                 xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np < 8u ? np : 8u, acc);
        if (np > 8u) {
            dev_dot_q2_K_q8_K_block8(wr + b, xqb[8] ? xqb[8] + b : NULL, xqb[9] ? xqb[9] + b : NULL,
                                     xqb[10] ? xqb[10] + b : NULL, xqb[11] ? xqb[11] + b : NULL,
                                     xqb[12] ? xqb[12] + b : NULL, xqb[13] ? xqb[13] + b : NULL,
                                     xqb[14] ? xqb[14] + b : NULL, xqb[15] ? xqb[15] + b : NULL, np - 8u, acc + 8);
        }
    }
    for (uint32_t p = 0; p < np; p++) {
        acc[p] = quarter_warp_sum_f32(acc[p], lane);
        if (lane == 0) {
            if (atomic_out) {
                uint32_t tok = pair[p] / n_expert;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
            } else {
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
}

__global__ static void moe_down_expert_tile16_row2048_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t local_start = tile_starts[tile];
    if (local_start & 8u) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    __shared__ cuda_block_q8_K sxq[16][8];
    uint32_t pair[16] = {0};
    const cuda_block_q8_K *xqb[16] = {NULL};
    uint32_t np = 0;
    for (; np < 16u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < 64u; rr++) {
        uint32_t row = blockIdx.x * 2048u + row_lane + rr * 32u;
        if (row >= out_dim) continue;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc[16] = {0.0f};
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                     xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                     xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                     xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np < 8u ? np : 8u, acc);
            if (np > 8u) {
                dev_dot_q2_K_q8_K_block8(wr + b, xqb[8] ? xqb[8] + b : NULL, xqb[9] ? xqb[9] + b : NULL,
                                         xqb[10] ? xqb[10] + b : NULL, xqb[11] ? xqb[11] + b : NULL,
                                         xqb[12] ? xqb[12] + b : NULL, xqb[13] ? xqb[13] + b : NULL,
                                         xqb[14] ? xqb[14] + b : NULL, xqb[15] ? xqb[15] + b : NULL, np - 8u, acc + 8);
            }
        }
        for (uint32_t p = 0; p < np; p++) {
            acc[p] = quarter_warp_sum_f32(acc[p], lane);
            if (lane == 0) {
                if (atomic_out) {
                    uint32_t tok = pair[p] / n_expert;
                    atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
                } else {
                    down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
                }
            }
        }
    }
}

template <uint32_t ROW_SPAN>
__global__ static void moe_down_expert_tile16_rowspan_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t local_start = tile_starts[tile];
    if (local_start & 8u) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    __shared__ cuda_block_q8_K sxq[16][8];
    uint32_t pair[16] = {0};
    const cuda_block_q8_K *xqb[16] = {NULL};
    uint32_t np = 0;
    for (; np < 16u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= out_dim) continue;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc[16] = {0.0f};
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                     xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                     xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                     xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np < 8u ? np : 8u, acc);
            if (np > 8u) {
                dev_dot_q2_K_q8_K_block8(wr + b, xqb[8] ? xqb[8] + b : NULL, xqb[9] ? xqb[9] + b : NULL,
                                         xqb[10] ? xqb[10] + b : NULL, xqb[11] ? xqb[11] + b : NULL,
                                         xqb[12] ? xqb[12] + b : NULL, xqb[13] ? xqb[13] + b : NULL,
                                         xqb[14] ? xqb[14] + b : NULL, xqb[15] ? xqb[15] + b : NULL, np - 8u, acc + 8);
            }
        }
        for (uint32_t p = 0; p < np; p++) {
            acc[p] = quarter_warp_sum_f32(acc[p], lane);
            if (lane == 0) {
                if (atomic_out) {
                    uint32_t tok = pair[p] / n_expert;
                    atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
                } else {
                    down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
                }
            }
        }
    }
}

__global__ static void moe_down_sorted_p2_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t pair_count) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_lane = (threadIdx.x >> 3u) & 1u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t sorted_idx = blockIdx.y * 2u + pair_lane;
    if (row >= out_dim || sorted_idx >= pair_count) return;
    uint32_t pair = sorted_pairs[sorted_idx];
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static void moe_sum_kernel(float *out, const float *down, uint32_t out_dim, uint32_t n_expert, uint32_t n_tokens) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * out_dim;
    if (gid >= n) return;
    uint32_t tok = gid / out_dim;
    uint32_t row = gid - (uint64_t)tok * out_dim;
    float acc = 0.0f;
    for (uint32_t e = 0; e < n_expert; e++) acc += down[((uint64_t)tok * n_expert + e) * out_dim + row];
    out[gid] = acc;
}

__global__ static void moe_mmq_sum_guard_kernel(
        float *out,
        const float *down,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t n_tokens) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_tokens * out_dim;
    if (gid >= n) return;
    const uint32_t tok = (uint32_t)(gid / out_dim);
    const uint32_t row = (uint32_t)(gid - (uint64_t)tok * out_dim);
    float acc = 0.0f;
    for (uint32_t e = 0; e < n_expert; ++e) {
        const float v = down[((uint64_t)tok * n_expert + e) * out_dim + row];
        if (isfinite(v)) acc += v;
    }
    out[gid] = acc;
}

__device__ static float dev_iq2_xxs_dot_f32(const cuda_block_iq2_xxs *row, const float *x, uint32_t nb) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        const cuda_block_iq2_xxs *xb = row + b;
        const float d = dev_f16_to_f32(xb->d);
        const uint16_t *q2 = xb->qs;
        const float *xf = x + (uint64_t)b * CUDA_QK_K;
        for (uint32_t ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
            const uint32_t aux_g = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
            const uint32_t aux_s = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
            q2 += 4;
            const float dl = d * (0.5f + (float)(aux_s >> 28)) * 0.25f;
            const uint8_t grids[4] = {
                (uint8_t)(aux_g & 0xffu),
                (uint8_t)((aux_g >> 8) & 0xffu),
                (uint8_t)((aux_g >> 16) & 0xffu),
                (uint8_t)((aux_g >> 24) & 0xffu),
            };
            for (uint32_t half = 0; half < 2; half++) {
                for (uint32_t g = 0; g < 2; g++) {
                    const uint32_t gi = half * 2 + g;
                    const uint64_t grid = cuda_iq2xxs_grid[grids[gi]];
                    const uint8_t signs = cuda_ksigns_iq2xs[(aux_s >> (14u * half + 7u * g)) & 127u];
                    for (uint32_t i = 0; i < 8; i++) {
                        float w = (float)((grid >> (8u * i)) & 0xffu);
                        if (signs & (1u << i)) w = -w;
                        acc += dl * w * xf[ib32 * 32u + half * 16u + g * 8u + i];
                    }
                }
            }
        }
    }
    return acc;
}

__device__ static float dev_q2_K_dot_f32(const cuda_block_q2_K *row, const float *x, uint32_t nb) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        const cuda_block_q2_K *xb = row + b;
        const float d = dev_f16_to_f32(xb->d);
        const float dmin = dev_f16_to_f32(xb->dmin);
        for (uint32_t il = 0; il < 16; il++) {
            const uint32_t chunk = il / 8u;
            const uint32_t pair = il & 1u;
            const uint32_t shift = ((il / 2u) & 3u) * 2u;
            const uint8_t sc = xb->scales[il];
            const float dl = d * (float)(sc & 0x0fu);
            const float ml = dmin * (float)(sc >> 4);
            const uint8_t *q = xb->qs + 32u * chunk + 16u * pair;
            const float *xf = x + (uint64_t)b * CUDA_QK_K + chunk * 128u + ((il % 8u) / 2u) * 32u + pair * 16u;
            for (uint32_t i = 0; i < 16; i++) {
                const float w = dl * (float)((q[i] >> shift) & 3u) - ml;
                acc += w * xf[i];
            }
        }
    }
    return acc;
}

__global__ static void moe_gate_up_mid_f32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const float *x,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const uint32_t nb = expert_in_dim / CUDA_QK_K;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const float *xr = x + (uint64_t)tok * expert_in_dim;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) {
        gate += dev_iq2_xxs_dot_f32(gr + b, xr + (uint64_t)b * CUDA_QK_K, 1);
        up += dev_iq2_xxs_dot_f32(ur + b, xr + (uint64_t)b * CUDA_QK_K, 1);
    }
    __shared__ float partial_gate[256];
    __shared__ float partial_up[256];
    partial_gate[threadIdx.x] = gate;
    partial_up[threadIdx.x] = up;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_gate[threadIdx.x] += partial_gate[threadIdx.x + stride];
            partial_up[threadIdx.x] += partial_up[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        gate = partial_gate[0];
        up = partial_up[0];
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static void moe_down_f32_kernel(
        float *down_out,
        const char *down_base,
        const float *mid,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const uint32_t nb = expert_mid_dim / CUDA_QK_K;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const float *xr = mid + (uint64_t)pair * expert_mid_dim;
    float acc = 0.0f;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) acc += dev_q2_K_dot_f32(wr + b, xr + (uint64_t)b * CUDA_QK_K, 1);
    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) down_out[(uint64_t)pair * out_dim + row] = partial[0];
}

static int routed_moe_mmq_prefill_allowed(
        uint32_t gate_type, uint32_t down_type,
        uint32_t n_tokens, uint32_t n_expert) {
    return gate_type == 16u && down_type == 10u &&
           n_tokens >= DS4_MMQ_PREFILL_MIN_TOKENS && n_expert == 6u;
}

static int routed_moe_mmq_prefill_launch(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid,
        ds4_gpu_tensor *down,
        const char *gate_w,
        const char *up_w,
        const char *down_w,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const int32_t *selected,
        const float *weights,
        const float *x,
        uint32_t n_total_expert,
        uint32_t n_expert,
        uint32_t n_tokens,
        float clamp,
        bool *mid_is_f16,
        bool defer_sum,
        bool *output_unsummed) {
    if (output_unsummed) *output_unsummed = false;
    if (!g_mmq_prefill_ready || n_tokens < DS4_MMQ_PREFILL_MIN_TOKENS ||
        !out || !gate || !up || !mid || !down ||
        !gate_w || !up_w || !down_w ||
        !selected || !weights || !x || n_total_expert == 0 || n_expert == 0 ||
        n_total_expert > (uint32_t)INT_MAX || n_expert > (uint32_t)INT_MAX ||
        n_tokens > (uint32_t)INT_MAX || expert_in_dim > (uint32_t)INT_MAX ||
        expert_mid_dim > (uint32_t)INT_MAX || out_dim > (uint32_t)INT_MAX) {
        return 0;
    }

    cuda_nvtx_scope routed_scope(
            "ds4/prefill/moe/routed",
            cuda_nvtx_payload(n_tokens, n_expert));

    const uint64_t expected_gate_row =
        (uint64_t)(expert_in_dim / CUDA_QK_K) * sizeof(cuda_block_iq2_xxs);
    const uint64_t expected_down_row =
        (uint64_t)(expert_mid_dim / CUDA_QK_K) * sizeof(cuda_block_q2_K);
    if (gate_row_bytes != expected_gate_row ||
        gate_expert_bytes != (uint64_t)expert_mid_dim * expected_gate_row ||
        down_row_bytes != expected_down_row ||
        down_expert_bytes != (uint64_t)out_dim * expected_down_row) {
        return 0;
    }

    const uint64_t assignment_count = (uint64_t)n_tokens * n_expert;
    if (n_expert != 6u || assignment_count > (uint64_t)INT_MAX) return 0;

#ifdef DS4_CUDA_TOKEN_GRAPH_BUILD
    const cudaStream_t stream = cudaStreamPerThread;
#else
    const cudaStream_t stream = 0;
#endif

    /* One expert-major map now spans gate/up and down. The established MMQ
     * quantizer gathers the weighted SwiGLU rows through that same map. */
    int rc = ds4_mmq_iq2_xxs_q2_K_moe_fused(
            gate_w, up_w, down_w, x, selected, weights,
            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
            (float *)down->ptr,
            (int)expert_mid_dim, (int)expert_in_dim, (int)out_dim,
            (int)n_tokens, (int)n_total_expert, (int)n_expert,
            clamp, stream);
    if (rc != 0) return -1;
    if (mid_is_f16) *mid_is_f16 = false;

    const uint64_t out_values = (uint64_t)n_tokens * out_dim;
    if (defer_sum) {
        if (output_unsummed) *output_unsummed = true;
    } else {
        cuda_nvtx_scope stage("ds4/prefill/moe/sum",
                              cuda_nvtx_payload(n_tokens, out_dim));
        moe_mmq_sum_guard_kernel<<<
                (uint32_t)((out_values + 255u) / 256u), 256, 0, stream>>>(
                (float *)out->ptr, (const float *)down->ptr,
                out_dim, n_expert, n_tokens);
        if (cudaGetLastError() != cudaSuccess) return -2;
    }

    if (!g_mmq_prefill_notice) {
        g_mmq_prefill_notice = 1;
        fprintf(stderr,
                "ds4: CUDA Entrpi batched MMQ MoE prefill enabled "
                "(single-map IQ2 gate/up + Q2 down, token-bound stream-K; decode excluded)\n");
    }
    return 1;
}

static int routed_moe_aligned_launch(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid,
        ds4_gpu_tensor *down,
        const cuda_moe_aligned_range *gate_art,
        const cuda_moe_aligned_range *up_art,
        const cuda_moe_aligned_range *down_art,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const int32_t *selected,
        const float *weights,
        const float *x,
        uint32_t n_total_expert,
        uint32_t n_expert,
        uint32_t n_tokens,
        float clamp,
        bool *mid_is_f16,
        bool defer_sum,
        bool *output_unsummed) {
    if (output_unsummed) *output_unsummed = false;
    if (!g_mmq_prefill_ready || !out || !gate || !up || !mid || !down ||
        !gate_art || !up_art || !down_art || !selected || !weights || !x ||
        gate_art->in_dim != expert_in_dim || up_art->in_dim != expert_in_dim ||
        gate_art->out_dim != expert_mid_dim || up_art->out_dim != expert_mid_dim ||
        down_art->in_dim != expert_mid_dim || down_art->out_dim != out_dim ||
        gate_art->group_count != n_total_expert ||
        up_art->group_count != n_total_expert ||
        down_art->group_count != n_total_expert ||
        n_tokens > (uint32_t)INT_MAX || n_total_expert > (uint32_t)INT_MAX ||
        n_expert != 6u || n_expert > (uint32_t)INT_MAX ||
        expert_in_dim == 0u || expert_in_dim % CUDA_QK_K != 0u ||
        expert_mid_dim == 0u || expert_mid_dim % CUDA_QK_K != 0u ||
        out_dim == 0u || (out_dim & 1u) != 0u ||
        expert_in_dim > (uint32_t)INT_MAX ||
        expert_mid_dim > (uint32_t)INT_MAX || out_dim > (uint32_t)INT_MAX) {
        return -1;
    }
#ifdef DS4_CUDA_TOKEN_GRAPH_BUILD
    const cudaStream_t stream = cudaStreamPerThread;
#else
    const cudaStream_t stream = 0;
#endif

    int rc = 0;
    bool output_summed = false;
    if (n_tokens <= 16u) {
        const bool use_gb10_small_batch =
            n_tokens >= 2u && n_tokens <= 6u &&
            g_moe_gb10_sign_validation == 1;
        const uint32_t xq_blocks = expert_in_dim / CUDA_QK_K;
        const uint32_t midq_blocks = expert_mid_dim / CUDA_QK_K;
        const uint64_t xq_bytes =
            (uint64_t)n_tokens * xq_blocks * sizeof(cuda_block_q8_K);
        const uint64_t midq_bytes =
            (uint64_t)n_tokens * n_expert * midq_blocks * sizeof(cuda_block_q8_K);
        if (down->bytes < xq_bytes || gate->bytes < midq_bytes) return -2;
        cuda_block_q8_K *xq = (cuda_block_q8_K *)down->ptr;
        cuda_block_q8_K *midq = (cuda_block_q8_K *)gate->ptr;

        q8_K_quantize_kernel<<<
                dim3(xq_blocks, n_tokens), 256, 0, stream>>>(
                xq, x, expert_in_dim, n_tokens);
        if (cudaGetLastError() != cudaSuccess) return -3;

        const uint64_t iq2_nblk =
            (uint64_t)n_total_expert * expert_mid_dim * xq_blocks;
        const uint64_t iq2_dq_bytes = (iq2_nblk * sizeof(__half) + 63u) & ~63ull;
        const __half *gate_dq = (const __half *)gate_art->device_ptr;
        const uint2 *gate_qs = (const uint2 *)(
                (const char *)gate_art->device_ptr + iq2_dq_bytes);
        const __half *up_dq = (const __half *)up_art->device_ptr;
        const uint2 *up_qs = (const uint2 *)(
                (const char *)up_art->device_ptr + iq2_dq_bytes);
        if (use_gb10_small_batch) {
            moe_gate_up_mid_aligned_q8K_gb10_kernel<<<
                    dim3((expert_mid_dim + 255u) / 256u,
                         n_tokens * n_expert),
                    256, 0, stream>>>(
                    (float *)mid->ptr,
                    gate_dq, gate_qs, up_dq, up_qs,
                    xq, selected, weights,
                    xq_blocks, expert_mid_dim, n_total_expert, n_expert, clamp);
        } else {
            moe_gate_up_mid_aligned_q8K_qwarp32_kernel<<<
                    dim3((expert_mid_dim + 127u) / 128u,
                         n_tokens * n_expert),
                    256, 0, stream>>>(
                    (float *)mid->ptr,
                    gate_dq, gate_qs, up_dq, up_qs,
                    xq, selected, weights,
                    xq_blocks, expert_mid_dim, n_total_expert, n_expert, clamp);
        }
        if (cudaGetLastError() != cudaSuccess) return -4;

        q8_K_quantize_kernel<<<
                dim3(midq_blocks, n_tokens * n_expert), 256, 0, stream>>>(
                midq, (const float *)mid->ptr,
                expert_mid_dim, n_tokens * n_expert);
        if (cudaGetLastError() != cudaSuccess) return -5;

        const uint64_t q2_npair =
            (uint64_t)n_total_expert * (out_dim / 2u) * midq_blocks;
        const uint64_t q2_dm_bytes = (q2_npair * sizeof(uint2) + 63u) & ~63ull;
        const uint64_t q2_sc_bytes = (q2_npair * sizeof(int4) * 2u + 63u) & ~63ull;
        const uint2 *dm2 = (const uint2 *)down_art->device_ptr;
        const int4 *sc4 = (const int4 *)(
                (const char *)down_art->device_ptr + q2_dm_bytes);
        const uint2 *qs2 = (const uint2 *)(
                (const char *)down_art->device_ptr + q2_dm_bytes + q2_sc_bytes);
        if (use_gb10_small_batch) {
            const size_t smem_bytes = midq_blocks <= 16u
                ? (size_t)6u * midq_blocks * sizeof(cuda_block_q8_K)
                : 0u;
            moe_down_aligned_q8K_sum6_gb10_kernel<<<
                    dim3((out_dim + 63u) / 64u, n_tokens),
                    256, smem_bytes, stream>>>(
                    (float *)out->ptr, dm2, sc4, qs2, midq, selected,
                    midq_blocks, out_dim, n_tokens, n_total_expert);
        } else {
            moe_down_aligned_q8K_sum6_qwarp32_kernel<<<
                    dim3((out_dim + 31u) / 32u, n_tokens), 256, 0, stream>>>(
                    (float *)out->ptr, dm2, sc4, qs2, midq, selected,
                    midq_blocks, out_dim, n_tokens, n_total_expert);
        }
        if (cudaGetLastError() != cudaSuccess) return -6;
        if (use_gb10_small_batch && !g_moe_aligned_small_batch_notice) {
            g_moe_aligned_small_batch_notice = 1;
            fprintf(stderr,
                    "ds4: CUDA GB10 aligned MoE verifier enabled "
                    "(computed IQ2 signs, coalesced Q8 staging, sum6 row span=64)\n");
        }
        output_summed = true;
    } else {
        const bool materialize_intermediates =
            getenv("DS4_METAL_GRAPH_DUMP_PREFIX") != NULL;
        if (!materialize_intermediates) {
            rc = ds4_mmq_iq2_xxs_q2_K_moe_fused_direct_soa(
                    gate_art->device_ptr, up_art->device_ptr,
                    down_art->device_ptr, x, selected, weights,
                    up->ptr, (size_t)up->bytes,
                    gate->ptr, (size_t)gate->bytes,
                    mid->ptr, (size_t)mid->bytes, (float *)down->ptr,
                    (int)expert_mid_dim, (int)expert_in_dim, (int)out_dim,
                    (int)n_tokens, (int)n_total_expert, (int)n_expert,
                    clamp, stream);
            if (rc == 0 && !g_moe_complete_fused_notice) {
                g_moe_complete_fused_notice = 1;
                fprintf(stderr,
                        "ds4: CUDA complete fused MoE D2R prefill enabled "
                        "(preallocated workspace, register gate/up, direct SwiGLU Q8 down)\n");
            }
        } else {
            rc = -1;
        }
        if (rc != 0) {
            rc = ds4_mmq_iq2_xxs_q2_K_moe_fused_soa(
                    gate_art->device_ptr, up_art->device_ptr,
                    down_art->device_ptr, x, selected, weights,
                    (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                    (float *)down->ptr,
                    (int)expert_mid_dim, (int)expert_in_dim, (int)out_dim,
                    (int)n_tokens, (int)n_total_expert, (int)n_expert,
                    clamp, stream);
        }
        if (rc != 0) return -7;
    }

    if (!output_summed && defer_sum) {
        if (output_unsummed) *output_unsummed = true;
    } else if (!output_summed) {
        const uint64_t out_values = (uint64_t)n_tokens * out_dim;
        moe_mmq_sum_guard_kernel<<<
                (uint32_t)((out_values + 255u) / 256u), 256, 0, stream>>>(
                (float *)out->ptr, (const float *)down->ptr,
                out_dim, n_expert, n_tokens);
        if (cudaGetLastError() != cudaSuccess) return -8;
    }
    if (mid_is_f16) *mid_is_f16 = false;
    if (!g_moe_aligned_notice) {
        g_moe_aligned_notice = 1;
        fprintf(stderr,
                "ds4: CUDA in-place aligned MoE execution active "
                "(Q8_K small-batch + fused gate/up/SwiGLU/Q8 D2R prefill)\n");
    }
    return 1;
}

static int routed_moe_launch(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid,
        ds4_gpu_tensor *down,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint32_t gate_type,
        uint32_t down_type,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t n_total_expert,
        uint32_t n_expert,
        float clamp,
        const ds4_gpu_tensor *x,
        uint32_t layer_index,
        uint32_t n_tokens,
        bool *mid_is_f16,
        bool defer_sum,
        bool *output_unsummed) {
    if (output_unsummed) *output_unsummed = false;
    if (!out || !gate || !up || !mid || !down || !model_map || !selected || !weights || !x ||
        n_tokens == 0 || n_total_expert == 0 || n_expert == 0 ||
        expert_in_dim % CUDA_QK_K != 0 || expert_mid_dim % CUDA_QK_K != 0 ||
        gate_offset > model_size || up_offset > model_size || down_offset > model_size ||
        x->bytes < (uint64_t)n_tokens * expert_in_dim * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * n_expert * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * n_expert * sizeof(float) ||
        gate->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
        up->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
        mid->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
        down->bytes < (uint64_t)n_tokens * n_expert * out_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float)) {
        return 0;
    }
    const int q4k_path = (gate_type == 12u && down_type == 12u);
    if (!q4k_path && (gate_type != 16u || down_type != 10u)) return 0;
    const uint64_t gate_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t down_bytes = (uint64_t)n_total_expert * down_expert_bytes;
    if (gate_bytes > model_size - gate_offset ||
        gate_bytes > model_size - up_offset ||
        down_bytes > model_size - down_offset) {
        return 0;
    }
    const uint64_t required_slot_count = (uint64_t)n_tokens * n_expert;
    const int use_stream_selected_cache =
        g_ssd_streaming_mode &&
        g_stream_selected_cache.valid &&
        g_stream_selected_cache.model_map == model_map &&
        g_stream_selected_cache.layer == layer_index &&
        g_stream_selected_cache.n_total_expert == n_total_expert &&
        g_stream_selected_cache.slot_count >= required_slot_count &&
        g_stream_selected_cache.gate_offset == gate_offset &&
        g_stream_selected_cache.up_offset == up_offset &&
        g_stream_selected_cache.down_offset == down_offset &&
        g_stream_selected_cache.gate_expert_bytes == gate_expert_bytes &&
        g_stream_selected_cache.down_expert_bytes == down_expert_bytes &&
        g_stream_selected_cache.gate_ptr &&
        g_stream_selected_cache.up_ptr &&
        g_stream_selected_cache.down_ptr &&
        g_stream_selected_cache.slot_selected_tensor.ptr &&
        g_stream_selected_cache.slot_selected_tensor.bytes >=
            required_slot_count * sizeof(int32_t);
    const ds4_gpu_tensor *selected_tensor =
        use_stream_selected_cache ? &g_stream_selected_cache.slot_selected_tensor : selected;
    const int32_t *selected_ptr = (const int32_t *)selected_tensor->ptr;
    const cuda_moe_aligned_range *aligned_gate = NULL;
    const cuda_moe_aligned_range *aligned_up = NULL;
    const cuda_moe_aligned_range *aligned_down = NULL;
    if (!g_ssd_streaming_mode && !use_stream_selected_cache && !q4k_path) {
        aligned_gate = cuda_moe_aligned_find(
                model_map, gate_offset, gate_bytes,
                DS4_REPACK_IQ2_XXS_ALIGNED_MOE);
        aligned_up = cuda_moe_aligned_find(
                model_map, up_offset, gate_bytes,
                DS4_REPACK_IQ2_XXS_ALIGNED_MOE);
        aligned_down = cuda_moe_aligned_find(
                model_map, down_offset, down_bytes,
                DS4_REPACK_Q2_K_ALIGNED_MOE);
    }
    const uint32_t aligned_count =
        (aligned_gate != NULL) + (aligned_up != NULL) + (aligned_down != NULL);
    const bool model_uses_aligned_moe =
        !q4k_path && g_moe_aligned_ready &&
        model_map == g_moe_aligned_host_base;
    if (model_uses_aligned_moe || aligned_count != 0u) {
        if (aligned_count != 3u) {
            fprintf(stderr,
                    "ds4: CUDA aligned MoE mapping is incomplete at layer %u; "
                    "refusing raw-layout fallback\n",
                    layer_index);
            return 0;
        }
        const int aligned_rc = routed_moe_aligned_launch(
                out, gate, up, mid, down,
                aligned_gate, aligned_up, aligned_down,
                expert_in_dim, expert_mid_dim, out_dim,
                selected_ptr, (const float *)weights->ptr,
                (const float *)x->ptr,
                n_total_expert, n_expert, n_tokens, clamp, mid_is_f16,
                defer_sum, output_unsummed);
        if (aligned_rc > 0) return 1;
        fprintf(stderr,
                "ds4: CUDA aligned MoE launch failed at layer %u (stage=%d); "
                "raw bytes were replaced, aborting the operation\n",
                layer_index, -aligned_rc);
        return 0;
    }
    const char *gate_w = use_stream_selected_cache
        ? g_stream_selected_cache.gate_ptr
        : cuda_model_range_ptr(model_map, gate_offset, gate_bytes, "moe_gate");
    const char *up_w = use_stream_selected_cache
        ? g_stream_selected_cache.up_ptr
        : cuda_model_range_ptr(model_map, up_offset, gate_bytes, "moe_up");
    const char *down_w = use_stream_selected_cache
        ? g_stream_selected_cache.down_ptr
        : cuda_model_range_ptr(model_map, down_offset, down_bytes, "moe_down");
    if (!gate_w || !up_w || !down_w) return 0;

    /* Raw-layout kernels remain the compatibility path for the Q4 DSpark
     * sidecar, SSD/distributed spans, and models without complete in-place
     * aligned replacements. */
    if (routed_moe_mmq_prefill_allowed(
                gate_type, down_type, n_tokens, n_expert) &&
        !g_ssd_streaming_mode && !use_stream_selected_cache) {
        const int mmq_rc = routed_moe_mmq_prefill_launch(
                out, gate, up, mid, down,
                gate_w, up_w, down_w,
                gate_expert_bytes, gate_row_bytes,
                down_expert_bytes, down_row_bytes,
                expert_in_dim, expert_mid_dim, out_dim,
                selected_ptr, (const float *)weights->ptr,
                (const float *)x->ptr,
                n_total_expert, n_expert, n_tokens, clamp, mid_is_f16,
                defer_sum, output_unsummed);
        if (mmq_rc > 0) return 1;
        if (mmq_rc < 0 && !g_mmq_prefill_fallback_notice) {
            g_mmq_prefill_fallback_notice = 1;
            fprintf(stderr,
                    "ds4: CUDA raw-GGUF MMQ MoE prefill launch failed "
                    "(stage=%d); using legacy prefill kernels\n",
                    -mmq_rc);
        }
    }

    int ok = 1;
    const uint32_t xq_blocks = expert_in_dim / CUDA_QK_K;
    const uint32_t midq_blocks = expert_mid_dim / CUDA_QK_K;
    const uint64_t xq_count = (uint64_t)n_tokens * xq_blocks;
    const uint64_t midq_count = (uint64_t)n_tokens * n_expert * midq_blocks;
    const uint64_t xq_bytes = xq_count * sizeof(cuda_block_q8_K);
    const uint64_t midq_bytes = midq_count * sizeof(cuda_block_q8_K);
    if (down->bytes >= xq_bytes && gate->bytes >= midq_bytes) {
        cuda_block_q8_K *xq = (cuda_block_q8_K *)down->ptr;
        cuda_block_q8_K *midq = (cuda_block_q8_K *)gate->ptr;
        const uint32_t profile_moe = getenv("DS4_CUDA_MOE_PROFILE") != NULL;
        cudaEvent_t prof_ev[7] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL};
        if (profile_moe) {
            for (uint32_t i = 0; i < 7u; i++) {
                if (cudaEventCreate(&prof_ev[i]) != cudaSuccess) {
                    for (uint32_t j = 0; j < i; j++) (void)cudaEventDestroy(prof_ev[j]);
                    memset(prof_ev, 0, sizeof(prof_ev));
                    break;
                }
            }
            if (prof_ev[0]) (void)cudaEventRecord(prof_ev[0], 0);
        }
        const uint32_t pair_count = n_tokens * n_expert;
        /* Sorting by expert amortizes weight reads for a real prefill batch,
         * but a DSpark verifier has only 2..6 rows (12..36 routed pairs).  In
         * that regime count/prefix/scatter/tile construction adds up to six
         * kernels per layer and most expert tiles still contain one pair.
         * Keep this opt-in until it is benchmarked on GB10.  The direct
         * kernels already index pair as token*n_expert+slot, so no numerical
         * or layout change is required. */
        const uint32_t tiny_direct_q4_only =
            getenv("DS4_CUDA_MOE_TINY_DIRECT_Q4_ONLY") != NULL;
        const uint32_t use_tiny_direct =
            n_tokens > 1u && n_tokens <= 6u &&
            getenv("DS4_CUDA_MOE_TINY_DIRECT") != NULL &&
            (!tiny_direct_q4_only || q4k_path);
        if (use_tiny_direct && !g_moe_tiny_direct_notice) {
            g_moe_tiny_direct_notice = 1;
            fprintf(stderr,
                    "ds4: CUDA MoE tiny-batch direct enabled "
                    "(2..6 rows, sorting/tiles bypassed%s)\n",
                    tiny_direct_q4_only ? ", Q4 sidecar only" : "");
        }
        const uint32_t use_q4_expert_tiles =
            q4k_path && getenv("DS4_CUDA_MOE_NO_Q4_EXPERT_TILES") == NULL;
        const uint32_t use_sorted_pairs =
            !use_tiny_direct && n_tokens > 1u &&
            (!q4k_path || use_q4_expert_tiles);
        const uint32_t use_expert_tiles = use_sorted_pairs && getenv("DS4_CUDA_MOE_NO_EXPERT_TILES") == NULL;
        const uint32_t expert_tile_m = (!q4k_path && getenv("DS4_CUDA_MOE_TILE4")) ? 4u : 8u;
        const uint32_t write_gate_up = getenv("DS4_CUDA_MOE_WRITE_GATE_UP") != NULL;
        const uint32_t use_p2_sorted = use_sorted_pairs && !q4k_path && getenv("DS4_CUDA_MOE_NO_P2") == NULL;
        const uint32_t use_atomic_down = use_expert_tiles &&
            getenv("DS4_CUDA_MOE_NO_ATOMIC_DOWN") == NULL &&
            (getenv("DS4_CUDA_MOE_ATOMIC_DOWN") != NULL ||
             (!q4k_path && n_tokens >= 128u));
        const uint32_t use_gate_row2048 = use_expert_tiles && expert_tile_m == 8u &&
            (getenv("DS4_CUDA_MOE_GATE_ROW2048") != NULL ||
             getenv("DS4_CUDA_MOE_GATE_ROW256") != NULL ||
             getenv("DS4_CUDA_MOE_GATE_ROW128") != NULL ||
             (n_tokens >= 128u &&
              getenv("DS4_CUDA_MOE_NO_GATE_ROW2048") == NULL &&
              getenv("DS4_CUDA_MOE_NO_GATE_ROW256") == NULL &&
              getenv("DS4_CUDA_MOE_NO_GATE_ROW128") == NULL));
        const uint32_t use_down_tile16 = expert_tile_m == 8u &&
            n_tokens >= 128u && getenv("DS4_CUDA_MOE_NO_DOWN_TILE16") == NULL &&
            (use_atomic_down || q4k_path);
        const uint32_t use_decode_lut_gate =
            !q4k_path && (n_tokens == 1u || use_tiny_direct) &&
            xq_blocks <= 16u &&
            getenv("DS4_CUDA_MOE_NO_DECODE_LUT_GATE") == NULL;
        const uint32_t use_decode_gb10 =
            use_decode_lut_gate &&
            xq_blocks == 16u && expert_mid_dim == 2048u && n_expert == 6u &&
            getenv("DS4_CUDA_MOE_DECODE_GB10") != NULL &&
            g_moe_gb10_sign_validation == 1;
        const uint32_t gate_row_span =
            getenv("DS4_CUDA_MOE_GATE_ROW512") != NULL ? 512u :
            getenv("DS4_CUDA_MOE_GATE_ROW2048") != NULL ? 2048u : 1024u;
        const uint32_t down_row_span =
            getenv("DS4_CUDA_MOE_DOWN_ROW512") != NULL ? 512u :
            getenv("DS4_CUDA_MOE_DOWN_ROW1024") != NULL ? 1024u : 2048u;
        const uint32_t use_down_row2048 = use_expert_tiles && expert_tile_m == 8u &&
            (getenv("DS4_CUDA_MOE_DOWN_ROW2048") != NULL ||
             getenv("DS4_CUDA_MOE_DOWN_ROW256") != NULL ||
             getenv("DS4_CUDA_MOE_DOWN_ROW128") != NULL ||
             getenv("DS4_CUDA_MOE_DOWN_ROW64") != NULL ||
             (use_down_tile16 &&
              getenv("DS4_CUDA_MOE_NO_DOWN_ROW2048") == NULL &&
              getenv("DS4_CUDA_MOE_NO_DOWN_ROW256") == NULL &&
              getenv("DS4_CUDA_MOE_NO_DOWN_ROW128") == NULL &&
              getenv("DS4_CUDA_MOE_NO_DOWN_ROW64") == NULL));
        const uint32_t use_direct_down_sum6 =
            n_tokens == 1u && n_expert == 6u &&
            getenv("DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6") == NULL;
        uint32_t *sorted_pairs = NULL;
        uint32_t *sorted_offsets = NULL;
        uint32_t *sorted_counts = NULL;
        uint32_t *tile_total = NULL;
        uint32_t *tile_experts = NULL;
        uint32_t *tile_starts = NULL;
        uint32_t *tile16_total = NULL;
        uint32_t *tile16_experts = NULL;
        uint32_t *tile16_starts = NULL;
        uint32_t tile_capacity = 0;
        uint32_t tile16_capacity = 0;
        dim3 xq_grid(xq_blocks, n_tokens, 1);
        q8_K_quantize_kernel<<<xq_grid, 256>>>(xq, (const float *)x->ptr, expert_in_dim, n_tokens);
        ok = cuda_ok(cudaGetLastError(), "routed_moe x quantize launch");
        if (prof_ev[1]) (void)cudaEventRecord(prof_ev[1], 0);
        if (ok && use_sorted_pairs) {
            const uint32_t sort_expert_count =
                use_stream_selected_cache ? g_stream_selected_cache.compact_count :
                n_total_expert;
            if (sort_expert_count == 0) ok = 0;
            const uint64_t counts_bytes = (uint64_t)sort_expert_count * sizeof(uint32_t);
            const uint64_t offsets_bytes = ((uint64_t)sort_expert_count + 1ull) * sizeof(uint32_t);
            const uint64_t cursors_bytes = (uint64_t)sort_expert_count * sizeof(uint32_t);
            const uint64_t sorted_bytes = (uint64_t)pair_count * sizeof(uint32_t);
            tile_capacity = (pair_count + expert_tile_m - 1u) / expert_tile_m + sort_expert_count;
            tile16_capacity = use_down_tile16 ? ((pair_count + 15u) / 16u + sort_expert_count) : 0u;
            const uint64_t tile_offsets_bytes = ((uint64_t)sort_expert_count + 1ull) * sizeof(uint32_t);
            const uint64_t tile_total_bytes = sizeof(uint32_t);
            const uint64_t tile_experts_bytes = (uint64_t)tile_capacity * sizeof(uint32_t);
            const uint64_t tile_starts_bytes = (uint64_t)tile_capacity * sizeof(uint32_t);
            const uint64_t tile16_offsets_bytes = use_down_tile16 ? ((uint64_t)sort_expert_count + 1ull) * sizeof(uint32_t) : 0u;
            const uint64_t tile16_total_bytes = use_down_tile16 ? sizeof(uint32_t) : 0u;
            const uint64_t tile16_experts_bytes = (uint64_t)tile16_capacity * sizeof(uint32_t);
            const uint64_t tile16_starts_bytes = (uint64_t)tile16_capacity * sizeof(uint32_t);
            const uint64_t tile_offsets_off = counts_bytes + offsets_bytes + cursors_bytes + sorted_bytes;
            const uint64_t tile_total_off = tile_offsets_off + tile_offsets_bytes;
            const uint64_t tile_experts_off = tile_total_off + tile_total_bytes;
            const uint64_t tile_starts_off = tile_experts_off + tile_experts_bytes;
            const uint64_t tile16_offsets_off = tile_starts_off + tile_starts_bytes;
            const uint64_t tile16_total_off = tile16_offsets_off + tile16_offsets_bytes;
            const uint64_t tile16_experts_off = tile16_total_off + tile16_total_bytes;
            const uint64_t tile16_starts_off = tile16_experts_off + tile16_experts_bytes;
            const uint64_t scratch_bytes = tile16_starts_off + tile16_starts_bytes;
            uint8_t *scratch = (uint8_t *)cuda_tmp_alloc(scratch_bytes,
                                                         "routed_moe sorted pairs");
            if (!scratch) {
                ok = 0;
            } else {
                uint32_t *counts = (uint32_t *)scratch;
                uint32_t *offsets = (uint32_t *)(scratch + counts_bytes);
                uint32_t *cursors = (uint32_t *)(scratch + counts_bytes + offsets_bytes);
                sorted_pairs = (uint32_t *)(scratch + counts_bytes + offsets_bytes + cursors_bytes);
                sorted_offsets = offsets;
                sorted_counts = counts;
                uint32_t *tile_offsets = (uint32_t *)(scratch + tile_offsets_off);
                tile_total = (uint32_t *)(scratch + tile_total_off);
                tile_experts = (uint32_t *)(scratch + tile_experts_off);
                tile_starts = (uint32_t *)(scratch + tile_starts_off);
                uint32_t *tile16_offsets = use_down_tile16 ? (uint32_t *)(scratch + tile16_offsets_off) : NULL;
                tile16_total = use_down_tile16 ? (uint32_t *)(scratch + tile16_total_off) : NULL;
                tile16_experts = use_down_tile16 ? (uint32_t *)(scratch + tile16_experts_off) : NULL;
                tile16_starts = use_down_tile16 ? (uint32_t *)(scratch + tile16_starts_off) : NULL;
                /* Synchronous cudaMemset is forbidden while the N=2 MTP
                 * verifier stream is being captured.  The async memset is a
                 * native graph node and remains ordered with the following
                 * sort kernels on the per-thread default stream. */
                ok = cuda_ok(cudaMemsetAsync(counts,
                                             0,
                                             (size_t)counts_bytes,
                                             cudaStreamPerThread),
                             "routed_moe sorted counts clear");
                if (ok) {
                    moe_count_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(
                        counts,
                        selected_ptr,
                        pair_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted count launch");
                }
                if (ok) {
                    moe_prefix_sorted_pairs_kernel<<<1, 1>>>(offsets, cursors, counts, sort_expert_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted prefix launch");
                }
                if (ok) {
                    moe_scatter_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(
                        sorted_pairs,
                        cursors,
                        selected_ptr,
                        pair_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted scatter launch");
                }
                if (ok && use_expert_tiles) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1>>>(tile_offsets, tile_total, counts, sort_expert_count, expert_tile_m);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile offsets launch");
                }
                if (ok && use_expert_tiles) {
                    moe_build_expert_tiles_kernel<<<(sort_expert_count + 255u) / 256u, 256>>>(tile_experts, tile_starts, tile_offsets, counts, sort_expert_count, expert_tile_m);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tiles launch");
                }
                if (ok && use_expert_tiles && use_down_tile16) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1>>>(tile16_offsets, tile16_total, counts, sort_expert_count, 16u);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile16 offsets launch");
                }
                if (ok && use_expert_tiles && use_down_tile16) {
                    moe_build_expert_tiles_kernel<<<(sort_expert_count + 255u) / 256u, 256>>>(tile16_experts, tile16_starts, tile16_offsets, counts, sort_expert_count, 16u);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile16 launch");
                }
            }
        }
        if (prof_ev[2]) (void)cudaEventRecord(prof_ev[2], 0);
        if (ok) {
            dim3 mgrid((expert_mid_dim + 31u) / 32u, n_tokens * n_expert, 1);
            if (ok && sorted_pairs && use_expert_tiles && sorted_offsets && sorted_counts && tile_total && tile_experts && tile_starts) {
                if (q4k_path) {
                    if (use_gate_row2048) {
                        if (gate_row_span == 512u) {
                            dim3 tgrid((expert_mid_dim + 511u) / 512u, tile_capacity, 1);
                            moe_gate_up_mid_q4K_expert_tile8_rowspan_kernel<512><<<tgrid, 256>>>(
                                (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                                gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                                tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                                gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                                write_gate_up, clamp);
                        } else if (gate_row_span == 1024u) {
                            dim3 tgrid((expert_mid_dim + 1023u) / 1024u, tile_capacity, 1);
                            moe_gate_up_mid_q4K_expert_tile8_rowspan_kernel<1024><<<tgrid, 256>>>(
                                (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                                gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                                tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                                gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                                write_gate_up, clamp);
                        } else {
                            dim3 tgrid((expert_mid_dim + 2047u) / 2048u, tile_capacity, 1);
                            moe_gate_up_mid_q4K_expert_tile8_rowspan_kernel<2048><<<tgrid, 256>>>(
                                (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                                gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                                tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                                gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                                write_gate_up, clamp);
                        }
                    } else {
                        dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity, 1);
                        moe_gate_up_mid_q4K_expert_tile8_rowspan_kernel<32><<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            write_gate_up, clamp);
                    }
                } else if (use_gate_row2048) {
                    if (gate_row_span == 512u) {
                        dim3 tgrid((expert_mid_dim + 511u) / 512u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_rowspan_kernel<512><<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            write_gate_up, clamp);
                    } else if (gate_row_span == 1024u) {
                        dim3 tgrid((expert_mid_dim + 1023u) / 1024u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_rowspan_kernel<1024><<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            write_gate_up, clamp);
                    } else {
                        dim3 tgrid((expert_mid_dim + 2047u) / 2048u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_row2048_kernel<<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            write_gate_up, clamp);
                    }
                } else if (expert_tile_m == 8u) {
                    dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity, 1);
                    moe_gate_up_mid_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                        gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        write_gate_up, clamp);
                } else {
                    dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity, 1);
                    moe_gate_up_mid_expert_tile4_row32_kernel<<<tgrid, 256>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                        gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        write_gate_up, clamp);
                }
            } else if (ok && sorted_pairs && use_p2_sorted) {
                dim3 p2_mgrid((expert_mid_dim + 15u) / 16u, (pair_count + 1u) / 2u, 1);
                moe_gate_up_mid_sorted_p2_qwarp32_kernel<<<p2_mgrid, 256>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    xq,
                    sorted_pairs,
                    selected_ptr,
                    (const float *)weights->ptr,
                    gate_expert_bytes,
                    gate_row_bytes,
                    xq_blocks,
                    expert_mid_dim,
                    n_expert,
                    pair_count,
                    clamp);
            } else if (ok && !q4k_path && sorted_pairs) {
                moe_gate_up_mid_sorted_qwarp32_kernel<<<mgrid, 256>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    xq,
                    sorted_pairs,
                    selected_ptr,
                    (const float *)weights->ptr,
                    gate_expert_bytes,
                    gate_row_bytes,
                    xq_blocks,
                    expert_mid_dim,
                    n_expert,
                    clamp);
            } else if (ok) {
                dim3 qgrid((expert_mid_dim + 127u) / 128u, n_tokens * n_expert, 1);
                if (q4k_path) {
                    moe_gate_up_mid_q4K_qwarp32_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        selected_ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        clamp);
                } else if (use_decode_gb10) {
                    dim3 gb10_grid((expert_mid_dim + 255u) / 256u,
                                   n_tokens * n_expert,
                                   1);
                    moe_gate_up_mid_decode_gb10_kernel<<<gb10_grid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        selected_ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        clamp);
                } else if (use_decode_lut_gate) {
                    moe_gate_up_mid_decode_lut_qwarp32_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        selected_ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        clamp);
                } else {
                    moe_gate_up_mid_qwarp32_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        selected_ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        clamp);
                }
            }
            ok = cuda_ok(cudaGetLastError(), "routed_moe gate/up launch");
        }
        if (prof_ev[3]) (void)cudaEventRecord(prof_ev[3], 0);
        if (ok) {
            dim3 midq_grid(midq_blocks, n_tokens * n_expert, 1);
            q8_K_quantize_kernel<<<midq_grid, 256>>>(midq, (const float *)mid->ptr, expert_mid_dim, n_tokens * n_expert);
            ok = cuda_ok(cudaGetLastError(), "routed_moe mid quantize launch");
        }
        if (prof_ev[4]) (void)cudaEventRecord(prof_ev[4], 0);
        if (ok) {
            dim3 dgrid((out_dim + 31u) / 32u, n_tokens * n_expert, 1);
            uint32_t *down_tile_total = tile_total;
            uint32_t *down_tile_experts = tile_experts;
            uint32_t *down_tile_starts = tile_starts;
            uint32_t down_tile_capacity = tile_capacity;
            if (use_down_tile16 && tile16_total && tile16_experts && tile16_starts) {
                down_tile_total = tile16_total;
                down_tile_experts = tile16_experts;
                down_tile_starts = tile16_starts;
                down_tile_capacity = tile16_capacity;
            }
            if (use_direct_down_sum6) {
                dim3 sgrid((out_dim + 31u) / 32u, 1, 1);
                if (q4k_path) {
                    moe_down_q4K_sum6_qwarp32_kernel<<<sgrid, 256>>>(
                        (float *)out->ptr,
                        down_w,
                        midq,
                        selected_ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim);
                } else {
                    moe_down_sum6_qwarp32_kernel<<<sgrid, 256>>>(
                        (float *)out->ptr,
                        down_w,
                        midq,
                        selected_ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim);
                }
            } else if (use_atomic_down) {
                uint64_t n = (uint64_t)n_tokens * out_dim;
                zero_kernel<<<(n + 255u) / 256u, 256>>>((float *)out->ptr, n);
                ok = cuda_ok(cudaGetLastError(), "routed_moe atomic zero launch");
            }
            if (use_direct_down_sum6) {
                /* The direct decode kernel writes the final token row. */
            } else if (sorted_pairs && use_expert_tiles && sorted_offsets && sorted_counts &&
                down_tile_total && down_tile_experts && down_tile_starts) {
                if (q4k_path) {
                    if (use_down_row2048) {
                        if (down_row_span == 512u) {
                            dim3 tgrid((out_dim + 511u) / 512u, down_tile_capacity, 1);
                            if (use_down_tile16) {
                                moe_down_q4K_expert_tile16_rowspan_kernel<512><<<tgrid, 256>>>(
                                    use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                                    down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                                    down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                                    midq_blocks, out_dim, n_expert, use_atomic_down);
                            } else {
                                moe_down_q4K_expert_tile8_rowspan_kernel<512><<<tgrid, 256>>>(
                                    use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                                    down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                                    down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                                    midq_blocks, out_dim, n_expert, use_atomic_down);
                            }
                        } else if (down_row_span == 1024u) {
                            dim3 tgrid((out_dim + 1023u) / 1024u, down_tile_capacity, 1);
                            if (use_down_tile16) {
                                moe_down_q4K_expert_tile16_rowspan_kernel<1024><<<tgrid, 256>>>(
                                    use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                                    down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                                    down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                                    midq_blocks, out_dim, n_expert, use_atomic_down);
                            } else {
                                moe_down_q4K_expert_tile8_rowspan_kernel<1024><<<tgrid, 256>>>(
                                    use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                                    down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                                    down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                                    midq_blocks, out_dim, n_expert, use_atomic_down);
                            }
                        } else {
                            dim3 tgrid((out_dim + 2047u) / 2048u, down_tile_capacity, 1);
                            if (use_down_tile16) {
                                moe_down_q4K_expert_tile16_rowspan_kernel<2048><<<tgrid, 256>>>(
                                    use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                                    down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                                    down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                                    midq_blocks, out_dim, n_expert, use_atomic_down);
                            } else {
                                moe_down_q4K_expert_tile8_rowspan_kernel<2048><<<tgrid, 256>>>(
                                    use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                                    down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                                    down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                                    midq_blocks, out_dim, n_expert, use_atomic_down);
                            }
                        }
                    } else if (use_down_tile16) {
                        dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                        moe_down_q4K_expert_tile16_rowspan_kernel<32><<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else {
                        dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                        moe_down_q4K_expert_tile8_rowspan_kernel<32><<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    }
                } else if (use_down_row2048) {
                    if (down_row_span == 512u) {
                        dim3 tgrid((out_dim + 511u) / 512u, down_tile_capacity, 1);
                        moe_down_expert_tile16_rowspan_kernel<512><<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else if (down_row_span == 1024u) {
                        dim3 tgrid((out_dim + 1023u) / 1024u, down_tile_capacity, 1);
                        moe_down_expert_tile16_rowspan_kernel<1024><<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else {
                        dim3 tgrid((out_dim + 2047u) / 2048u, down_tile_capacity, 1);
                        moe_down_expert_tile16_row2048_kernel<<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    }
                } else if (use_down_tile16) {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile16_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                } else if (expert_tile_m == 8u) {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                } else {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile4_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                }
            } else if (sorted_pairs && use_p2_sorted) {
                dim3 p2_dgrid((out_dim + 15u) / 16u, (pair_count + 1u) / 2u, 1);
                moe_down_sorted_p2_qwarp32_kernel<<<p2_dgrid, 256>>>(
                    (float *)down->ptr,
                    down_w,
                    midq,
                    sorted_pairs,
                    selected_ptr,
                    down_expert_bytes,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert,
                    pair_count);
            } else if (!q4k_path && sorted_pairs) {
                moe_down_sorted_qwarp32_kernel<<<dgrid, 256>>>(
                    (float *)down->ptr,
                    down_w,
                    midq,
                    sorted_pairs,
                    selected_ptr,
                    down_expert_bytes,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert);
            } else {
                if (q4k_path) {
                    moe_down_q4K_qwarp32_kernel<<<dgrid, 256>>>(
                        (float *)down->ptr,
                        down_w,
                        midq,
                        selected_ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                } else {
                    moe_down_qwarp32_kernel<<<dgrid, 256>>>(
                        (float *)down->ptr,
                        down_w,
                        midq,
                        selected_ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                }
            }
            ok = cuda_ok(cudaGetLastError(), "routed_moe down launch");
        }
        if (prof_ev[5]) (void)cudaEventRecord(prof_ev[5], 0);
        if (ok && !use_atomic_down && !use_direct_down_sum6 && defer_sum) {
            if (output_unsummed) *output_unsummed = true;
        } else if (ok && !use_atomic_down && !use_direct_down_sum6) {
            uint64_t n = (uint64_t)n_tokens * out_dim;
            moe_sum_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
            ok = cuda_ok(cudaGetLastError(), "routed_moe sum launch");
        }
        if (prof_ev[6]) {
            (void)cudaEventRecord(prof_ev[6], 0);
            if (cudaEventSynchronize(prof_ev[6]) == cudaSuccess) {
                float ms_xq = 0.0f, ms_sort = 0.0f, ms_gate = 0.0f, ms_midq = 0.0f, ms_down = 0.0f, ms_sum = 0.0f, ms_total = 0.0f;
                (void)cudaEventElapsedTime(&ms_xq, prof_ev[0], prof_ev[1]);
                (void)cudaEventElapsedTime(&ms_sort, prof_ev[1], prof_ev[2]);
                (void)cudaEventElapsedTime(&ms_gate, prof_ev[2], prof_ev[3]);
                (void)cudaEventElapsedTime(&ms_midq, prof_ev[3], prof_ev[4]);
                (void)cudaEventElapsedTime(&ms_down, prof_ev[4], prof_ev[5]);
                (void)cudaEventElapsedTime(&ms_sum, prof_ev[5], prof_ev[6]);
                (void)cudaEventElapsedTime(&ms_total, prof_ev[0], prof_ev[6]);
                fprintf(stderr,
                        "ds4: CUDA MoE profile tokens=%u pairs=%u xq=%.3f sort=%.3f gateup=%.3f midq=%.3f down=%.3f sum=%.3f total=%.3f ms\n",
                        n_tokens, pair_count, ms_xq, ms_sort, ms_gate, ms_midq, ms_down, ms_sum, ms_total);
            }
            for (uint32_t i = 0; i < 7u; i++) (void)cudaEventDestroy(prof_ev[i]);
        }
        return ok;
    }

    if (ok) {
        dim3 mgrid(expert_mid_dim, n_tokens * n_expert, 1);
        moe_gate_up_mid_f32_kernel<<<mgrid, 256>>>(
            (float *)gate->ptr,
            (float *)up->ptr,
            (float *)mid->ptr,
            gate_w,
            up_w,
            (const float *)x->ptr,
            selected_ptr,
            (const float *)weights->ptr,
            gate_expert_bytes,
            gate_row_bytes,
            expert_in_dim,
            expert_mid_dim,
            n_expert,
            clamp);
        ok = cuda_ok(cudaGetLastError(), "routed_moe gate/up launch");
    }
    if (ok) {
        dim3 dgrid(out_dim, n_tokens * n_expert, 1);
        moe_down_f32_kernel<<<dgrid, 256>>>(
            (float *)down->ptr,
            down_w,
            (const float *)mid->ptr,
            selected_ptr,
            down_expert_bytes,
            down_row_bytes,
            expert_mid_dim,
            out_dim,
            n_expert);
        ok = cuda_ok(cudaGetLastError(), "routed_moe down launch");
    }
    if (ok && defer_sum) {
        if (output_unsummed) *output_unsummed = true;
    } else if (ok) {
        uint64_t n = (uint64_t)n_tokens * out_dim;
        moe_sum_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
        ok = cuda_ok(cudaGetLastError(), "routed_moe sum launch");
    }
    return ok;
}

extern "C" int ds4_gpu_routed_moe_set_selected_override(const int32_t *selected, uint32_t n_selected) {
    (void)selected;
    (void)n_selected;
    return 1;
}

extern "C" int ds4_gpu_routed_moe_one_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x, uint32_t layer_index) {
    return routed_moe_launch(out, gate, up, mid, down, model_map, model_size,
                             gate_offset, up_offset, down_offset,
                             gate_type, down_type,
                             gate_expert_bytes, gate_row_bytes,
                             down_expert_bytes, down_row_bytes,
                             expert_in_dim, expert_mid_dim, out_dim,
                             selected, weights, n_total_expert, n_expert, clamp, x,
                             layer_index, 1, NULL, false, NULL);
}
extern "C" int ds4_gpu_routed_moe_batch_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x, uint32_t layer_index, uint32_t n_tokens, bool *mid_is_f16, bool defer_sum, bool *output_unsummed) {
    if (mid_is_f16) *mid_is_f16 = false;
    return routed_moe_launch(out, gate, up, mid, down, model_map, model_size,
                             gate_offset, up_offset, down_offset,
                             gate_type, down_type,
                             gate_expert_bytes, gate_row_bytes,
                             down_expert_bytes, down_row_bytes,
                             expert_in_dim, expert_mid_dim, out_dim,
                             selected, weights, n_total_expert, n_expert, clamp, x,
                             layer_index, n_tokens, mid_is_f16,
                             defer_sum, output_unsummed);
}

extern "C" int ds4_gpu_mmq_prefill_self_test(void) {
    enum {
        n_tokens = 192,
        n_total_expert = 8,
        n_expert = 6,
        in_dim = 1024,
        mid_dim = 256,
        out_dim = 256
    };
    const float test_clamp = 10.0f;
    if (!g_mmq_prefill_ready ||
        routed_moe_mmq_prefill_allowed(16u, 10u, 1u, n_expert) ||
        routed_moe_mmq_prefill_allowed(16u, 10u, 6u, n_expert) ||
        routed_moe_mmq_prefill_allowed(12u, 12u, DS4_MMQ_PREFILL_MIN_TOKENS, n_expert) ||
        routed_moe_mmq_prefill_allowed(16u, 10u, DS4_MMQ_PREFILL_MIN_TOKENS, 4u) ||
        routed_moe_mmq_prefill_allowed(16u, 10u, DS4_MMQ_PREFILL_MIN_TOKENS - 1u, n_expert) ||
        !routed_moe_mmq_prefill_allowed(16u, 10u, DS4_MMQ_PREFILL_MIN_TOKENS, n_expert)) {
        return 0;
    }

    const uint64_t pair_count = (uint64_t)n_tokens * n_expert;
    const uint64_t gate_row_bytes =
        (uint64_t)(in_dim / CUDA_QK_K) * sizeof(cuda_block_iq2_xxs);
    const uint64_t gate_expert_bytes = (uint64_t)mid_dim * gate_row_bytes;
    const uint64_t down_row_bytes =
        (uint64_t)(mid_dim / CUDA_QK_K) * sizeof(cuda_block_q2_K);
    const uint64_t down_expert_bytes = (uint64_t)out_dim * down_row_bytes;
    const uint64_t gate_block_count =
        (uint64_t)n_total_expert * mid_dim * (in_dim / CUDA_QK_K);
    const uint64_t down_block_count =
        (uint64_t)n_total_expert * out_dim * (mid_dim / CUDA_QK_K);
    const uint64_t pair_values = pair_count * mid_dim;
    const uint64_t out_values = (uint64_t)n_tokens * out_dim;

    std::vector<cuda_block_iq2_xxs> gate_host(gate_block_count);
    std::vector<cuda_block_iq2_xxs> up_host(gate_block_count);
    std::vector<cuda_block_q2_K> down_host(down_block_count);
    std::vector<float> x_host((uint64_t)n_tokens * in_dim);
    std::vector<int32_t> selected_host(pair_count);
    std::vector<float> weights_host(pair_count);
    std::vector<float> gate_ref_host(pair_values);
    std::vector<float> gate_mmq_host(pair_values);
    std::vector<float> up_ref_host(pair_values);
    std::vector<float> up_mmq_host(pair_values);
    std::vector<float> mid_ref_host(pair_values);
    std::vector<float> mid_mmq_host(pair_values);
    std::vector<float> out_ref_host(out_values);
    std::vector<float> out_mmq_host(out_values);
    std::vector<float> gate_soa_host(pair_values);
    std::vector<float> up_soa_host(pair_values);
    std::vector<float> mid_soa_host(pair_values);
    std::vector<float> out_soa_host(out_values);
    std::vector<float> out_fused_direct_host(out_values);
    std::vector<float> out_direct_host((uint64_t)16u * out_dim);
    std::vector<float> out_soa_prefix_host((uint64_t)16u * out_dim);
    std::vector<float> out_q8k_host((uint64_t)16u * out_dim);
    std::vector<float> out_q8k_gb10_host((uint64_t)6u * out_dim);
    std::vector<float> out_ref_prefix_host((uint64_t)16u * out_dim);
    uint64_t q8k_gb10_bad = 0;
    double q8k_gb10_max_abs = 0.0;

    uint32_t rng = 0x6d2b79f5u;
    auto next_u32 = [&rng]() -> uint32_t {
        rng = rng * 1664525u + 1013904223u;
        return rng;
    };
    const uint16_t iq2_scales[] = {0x2a66u, 0x2d1fu, 0x2e66u, 0x30cdu};
    const uint16_t q2_scales[] = {0x251fu, 0x2a66u, 0x2d1fu, 0x2e66u};
    for (uint64_t b = 0; b < gate_block_count; ++b) {
        gate_host[b].d = iq2_scales[next_u32() & 3u];
        up_host[b].d = iq2_scales[next_u32() & 3u];
        for (uint32_t q = 0; q < CUDA_QK_K / 8; ++q) {
            gate_host[b].qs[q] = (uint16_t)(next_u32() >> 16u);
            up_host[b].qs[q] = (uint16_t)(next_u32() >> 16u);
        }
    }
    for (uint64_t b = 0; b < down_block_count; ++b) {
        down_host[b].d = q2_scales[next_u32() & 3u];
        down_host[b].dmin = q2_scales[next_u32() & 3u];
        for (uint32_t s = 0; s < CUDA_QK_K / 16; ++s) {
            const uint8_t lo = (uint8_t)(next_u32() & 0x0fu);
            const uint8_t hi = (uint8_t)(next_u32() & 0x0fu);
            down_host[b].scales[s] = (uint8_t)(lo | (hi << 4u));
        }
        for (uint32_t q = 0; q < CUDA_QK_K / 4; ++q) {
            down_host[b].qs[q] = (uint8_t)(next_u32() >> 24u);
        }
    }
    for (uint64_t i = 0; i < x_host.size(); ++i) {
        const float unit = (float)(next_u32() >> 8u) * (1.0f / 16777216.0f);
        x_host[i] = 2.0f * unit - 1.0f;
    }
    for (uint32_t t = 0; t < n_tokens; ++t) {
        for (uint32_t s = 0; s < n_expert; ++s) {
            const uint64_t pair = (uint64_t)t * n_expert + s;
            /* Expert zero receives the maximum valid top-k load: exactly one
             * assignment from every token. The other five slots remain
             * distinct and rotate over the remaining experts. */
            selected_host[pair] = s == 0u
                ? 0
                : (int32_t)(1u + ((t + s - 1u) % (n_total_expert - 1u)));
            weights_host[pair] = 0.20f + 0.015f * (float)s;
        }
    }

    std::vector<ds4_gpu_tensor *> allocations;
    auto alloc = [&allocations](uint64_t bytes) -> ds4_gpu_tensor * {
        ds4_gpu_tensor *tensor = ds4_gpu_tensor_alloc(bytes);
        if (tensor) allocations.push_back(tensor);
        return tensor;
    };

    ds4_gpu_tensor *gate_w = alloc(gate_block_count * sizeof(cuda_block_iq2_xxs));
    ds4_gpu_tensor *up_w = alloc(gate_block_count * sizeof(cuda_block_iq2_xxs));
    ds4_gpu_tensor *down_w = alloc(down_block_count * sizeof(cuda_block_q2_K));
    ds4_gpu_tensor *gate_soa = alloc(ds4_mmq_iq2_xxs_aligned_bytes(
            mid_dim, in_dim, n_total_expert));
    ds4_gpu_tensor *up_soa = alloc(ds4_mmq_iq2_xxs_aligned_bytes(
            mid_dim, in_dim, n_total_expert));
    ds4_gpu_tensor *down_soa = alloc(ds4_mmq_q2_k_aligned_bytes(
            out_dim, mid_dim, n_total_expert));
    ds4_gpu_tensor *x = alloc(x_host.size() * sizeof(float));
    ds4_gpu_tensor *selected = alloc(pair_count * sizeof(int32_t));
    ds4_gpu_tensor *weights = alloc(pair_count * sizeof(float));
    ds4_gpu_tensor *xq = alloc((uint64_t)n_tokens * (in_dim / CUDA_QK_K) * sizeof(cuda_block_q8_K));
    ds4_gpu_tensor *midq = alloc(pair_count * (mid_dim / CUDA_QK_K) * sizeof(cuda_block_q8_K));
    ds4_gpu_tensor *gate_ref = alloc(pair_values * sizeof(float));
    ds4_gpu_tensor *up_ref = alloc(pair_values * sizeof(float));
    ds4_gpu_tensor *mid_ref = alloc(pair_values * sizeof(float));
    ds4_gpu_tensor *down_ref = alloc(pair_count * out_dim * sizeof(float));
    ds4_gpu_tensor *out_ref = alloc(out_values * sizeof(float));
    ds4_gpu_tensor *gate_mmq = alloc(pair_values * sizeof(float));
    ds4_gpu_tensor *up_mmq = alloc(pair_values * sizeof(float));
    ds4_gpu_tensor *mid_mmq = alloc(pair_values * sizeof(float));
    ds4_gpu_tensor *down_mmq = alloc(pair_count * out_dim * sizeof(float));
    ds4_gpu_tensor *out_mmq = alloc(out_values * sizeof(float));
    /* The synthetic K=1024, M=256 shape has a smaller FP32 up tensor than
     * its gathered Q8_1 input. Production Flash uses K=4096, M=2048 and
     * safely reuses up; keep the capacity guard strict and give only this
     * regression shape a dedicated oversized input workspace. */
    ds4_gpu_tensor *direct_input_q8 = alloc(pair_count * in_dim * sizeof(float));

    int ok = allocations.size() == 22u;
    if (ok) ok = ds4_gpu_tensor_write(gate_w, 0, gate_host.data(), gate_w->bytes);
    if (ok) ok = ds4_gpu_tensor_write(up_w, 0, up_host.data(), up_w->bytes);
    if (ok) ok = ds4_gpu_tensor_write(down_w, 0, down_host.data(), down_w->bytes);
    if (ok) ok = ds4_gpu_tensor_write(x, 0, x_host.data(), x->bytes);
    if (ok) ok = ds4_gpu_tensor_write(selected, 0, selected_host.data(), selected->bytes);
    if (ok) ok = ds4_gpu_tensor_write(weights, 0, weights_host.data(), weights->bytes);

#ifdef DS4_CUDA_TOKEN_GRAPH_BUILD
    const cudaStream_t stream = cudaStreamPerThread;
#else
    const cudaStream_t stream = 0;
#endif

    if (ok) {
        ok = ds4_repack_iq2_aligned_device(
                     gate_soa->ptr, gate_w->ptr,
                     in_dim, mid_dim, n_total_expert, stream) &&
             ds4_repack_iq2_aligned_device(
                     up_soa->ptr, up_w->ptr,
                     in_dim, mid_dim, n_total_expert, stream) &&
             ds4_repack_q2k_aligned_device(
                     down_soa->ptr, down_w->ptr,
                     mid_dim, out_dim, n_total_expert, stream);
    }

    if (ok) {
        q8_K_quantize_kernel<<<dim3(in_dim / CUDA_QK_K, n_tokens), 256, 0, stream>>>(
                (cuda_block_q8_K *)xq->ptr, (const float *)x->ptr, in_dim, n_tokens);
        ok = cudaGetLastError() == cudaSuccess;
    }
    if (ok) {
        moe_gate_up_mid_qwarp32_kernel<<<dim3((mid_dim + 127u) / 128u, (uint32_t)pair_count), 256, 0, stream>>>(
                (float *)gate_ref->ptr, (float *)up_ref->ptr, (float *)mid_ref->ptr,
                (const char *)gate_w->ptr, (const char *)up_w->ptr,
                (const cuda_block_q8_K *)xq->ptr, (const int32_t *)selected->ptr,
                (const float *)weights->ptr, gate_expert_bytes, gate_row_bytes,
                in_dim / CUDA_QK_K, mid_dim, n_expert, test_clamp);
        ok = cudaGetLastError() == cudaSuccess;
    }
    if (ok) {
        q8_K_quantize_kernel<<<dim3(mid_dim / CUDA_QK_K, (uint32_t)pair_count), 256, 0, stream>>>(
                (cuda_block_q8_K *)midq->ptr, (const float *)mid_ref->ptr,
                mid_dim, (uint32_t)pair_count);
        ok = cudaGetLastError() == cudaSuccess;
    }
    if (ok) {
        moe_down_qwarp32_kernel<<<dim3((out_dim + 31u) / 32u, (uint32_t)pair_count), 256, 0, stream>>>(
                (float *)down_ref->ptr, (const char *)down_w->ptr,
                (const cuda_block_q8_K *)midq->ptr, (const int32_t *)selected->ptr,
                down_expert_bytes, down_row_bytes, mid_dim / CUDA_QK_K,
                out_dim, n_expert);
        ok = cudaGetLastError() == cudaSuccess;
    }
    if (ok) {
        moe_sum_kernel<<<(uint32_t)((out_values + 255u) / 256u), 256, 0, stream>>>(
                (float *)out_ref->ptr, (const float *)down_ref->ptr,
                out_dim, n_expert, n_tokens);
        ok = cudaGetLastError() == cudaSuccess;
    }

    if (ok) {
        ok = ds4_mmq_iq2_xxs_q2_K_moe_fused(
                gate_w->ptr, up_w->ptr, down_w->ptr,
                (const float *)x->ptr, (const int32_t *)selected->ptr,
                (const float *)weights->ptr,
                (float *)gate_mmq->ptr, (float *)up_mmq->ptr,
                (float *)mid_mmq->ptr, (float *)down_mmq->ptr,
                mid_dim, in_dim, out_dim,
                n_tokens, n_total_expert, n_expert, test_clamp,
                stream) == 0;
    }
    if (ok) {
        moe_mmq_sum_guard_kernel<<<(uint32_t)((out_values + 255u) / 256u), 256, 0, stream>>>(
                (float *)out_mmq->ptr, (const float *)down_mmq->ptr,
                out_dim, n_expert, n_tokens);
        ok = cudaGetLastError() == cudaSuccess && cudaStreamSynchronize(stream) == cudaSuccess;
    }

    if (ok) ok = ds4_gpu_tensor_read(gate_ref, 0, gate_ref_host.data(), gate_ref->bytes);
    if (ok) ok = ds4_gpu_tensor_read(gate_mmq, 0, gate_mmq_host.data(), gate_mmq->bytes);
    if (ok) ok = ds4_gpu_tensor_read(up_ref, 0, up_ref_host.data(), up_ref->bytes);
    if (ok) ok = ds4_gpu_tensor_read(up_mmq, 0, up_mmq_host.data(), up_mmq->bytes);
    if (ok) ok = ds4_gpu_tensor_read(mid_ref, 0, mid_ref_host.data(), mid_ref->bytes);
    if (ok) ok = ds4_gpu_tensor_read(mid_mmq, 0, mid_mmq_host.data(), mid_mmq->bytes);
    if (ok) ok = ds4_gpu_tensor_read(out_ref, 0, out_ref_host.data(), out_ref->bytes);
    if (ok) ok = ds4_gpu_tensor_read(out_mmq, 0, out_mmq_host.data(), out_mmq->bytes);

    if (ok) {
        ok = ds4_mmq_iq2_xxs_q2_K_moe_fused_soa(
                gate_soa->ptr, up_soa->ptr, down_soa->ptr,
                (const float *)x->ptr, (const int32_t *)selected->ptr,
                (const float *)weights->ptr,
                (float *)gate_mmq->ptr, (float *)up_mmq->ptr,
                (float *)mid_mmq->ptr, (float *)down_mmq->ptr,
                mid_dim, in_dim, out_dim,
                n_tokens, n_total_expert, n_expert, test_clamp,
                stream) == 0;
    }
    if (ok) {
        moe_mmq_sum_guard_kernel<<<
                (uint32_t)((out_values + 255u) / 256u), 256, 0, stream>>>(
                (float *)out_mmq->ptr, (const float *)down_mmq->ptr,
                out_dim, n_expert, n_tokens);
        ok = cudaGetLastError() == cudaSuccess &&
             cudaStreamSynchronize(stream) == cudaSuccess;
    }
    if (ok) ok = ds4_gpu_tensor_read(gate_mmq, 0, gate_soa_host.data(), gate_mmq->bytes);
    if (ok) ok = ds4_gpu_tensor_read(up_mmq, 0, up_soa_host.data(), up_mmq->bytes);
    if (ok) ok = ds4_gpu_tensor_read(mid_mmq, 0, mid_soa_host.data(), mid_mmq->bytes);
    if (ok) ok = ds4_gpu_tensor_read(out_mmq, 0, out_soa_host.data(), out_mmq->bytes);

    if (ok) {
        const int direct_rc = ds4_mmq_iq2_xxs_q2_K_moe_fused_direct_soa(
                gate_soa->ptr, up_soa->ptr, down_soa->ptr,
                (const float *)x->ptr, (const int32_t *)selected->ptr,
                (const float *)weights->ptr,
                direct_input_q8->ptr, (size_t)direct_input_q8->bytes,
                gate_mmq->ptr, (size_t)gate_mmq->bytes,
                mid_mmq->ptr, (size_t)mid_mmq->bytes,
                (float *)down_mmq->ptr,
                mid_dim, in_dim, out_dim,
                n_tokens, n_total_expert, n_expert, test_clamp,
                stream);
        ok = direct_rc == 0;
        if (!ok) {
            fprintf(stderr,
                    "cuda-regression: complete fused D2R launch failed rc=%d\n",
                    direct_rc);
        }
    }
    if (ok) {
        moe_mmq_sum_guard_kernel<<<
                (uint32_t)((out_values + 255u) / 256u), 256, 0, stream>>>(
                (float *)out_mmq->ptr, (const float *)down_mmq->ptr,
                out_dim, n_expert, n_tokens);
        ok = cudaGetLastError() == cudaSuccess &&
             cudaStreamSynchronize(stream) == cudaSuccess;
    }
    if (ok) {
        ok = ds4_gpu_tensor_read(
                out_mmq, 0, out_fused_direct_host.data(), out_mmq->bytes);
    }

    if (ok) {
        ok = ds4_mmq_iq2_xxs_aligned_moe_gate_up_mid_vec(
                gate_soa->ptr, up_soa->ptr,
                (const float *)x->ptr, (const int32_t *)selected->ptr,
                (const float *)weights->ptr, (float *)mid_mmq->ptr,
                mid_dim, in_dim, 16, n_total_expert, n_expert,
                test_clamp, stream) == 0;
    }
    if (ok) {
        ok = ds4_mmq_q2_K_aligned_moe_down_sum6_vec(
                down_soa->ptr, (const float *)mid_mmq->ptr,
                (const int32_t *)selected->ptr, (float *)out_mmq->ptr,
                out_dim, mid_dim, 16, n_total_expert, n_expert,
                stream) == 0 &&
             cudaStreamSynchronize(stream) == cudaSuccess;
    }
    if (ok) {
        ok = ds4_gpu_tensor_read(
                out_mmq, 0, out_direct_host.data(),
                (uint64_t)16u * out_dim * sizeof(float));
    }
    if (ok) {
        memcpy(out_soa_prefix_host.data(), out_soa_host.data(),
               (uint64_t)16u * out_dim * sizeof(float));
    }

    if (ok) {
        q8_K_quantize_kernel<<<dim3(in_dim / CUDA_QK_K, 16u), 256, 0, stream>>>(
                (cuda_block_q8_K *)xq->ptr, (const float *)x->ptr, in_dim, 16u);
        ok = cudaGetLastError() == cudaSuccess;
    }
    const uint64_t iq2_dq_bytes =
        (gate_block_count * sizeof(__half) + 63u) & ~63ull;
    if (ok) {
        moe_gate_up_mid_aligned_q8K_qwarp32_kernel<<<
                dim3((mid_dim + 127u) / 128u, 16u * n_expert),
                256, 0, stream>>>(
                (float *)mid_mmq->ptr,
                (const __half *)gate_soa->ptr,
                (const uint2 *)((const char *)gate_soa->ptr + iq2_dq_bytes),
                (const __half *)up_soa->ptr,
                (const uint2 *)((const char *)up_soa->ptr + iq2_dq_bytes),
                (const cuda_block_q8_K *)xq->ptr,
                (const int32_t *)selected->ptr,
                (const float *)weights->ptr,
                in_dim / CUDA_QK_K, mid_dim, n_total_expert, n_expert,
                test_clamp);
        ok = cudaGetLastError() == cudaSuccess;
    }
    if (ok) {
        q8_K_quantize_kernel<<<
                dim3(mid_dim / CUDA_QK_K, 16u * n_expert),
                256, 0, stream>>>(
                (cuda_block_q8_K *)midq->ptr, (const float *)mid_mmq->ptr,
                mid_dim, 16u * n_expert);
        ok = cudaGetLastError() == cudaSuccess;
    }
    const uint64_t q2_npair =
        (uint64_t)n_total_expert * (out_dim / 2u) * (mid_dim / CUDA_QK_K);
    const uint64_t q2_dm_bytes = (q2_npair * sizeof(uint2) + 63u) & ~63ull;
    const uint64_t q2_sc_bytes =
        (q2_npair * sizeof(int4) * 2u + 63u) & ~63ull;
    if (ok) {
        moe_down_aligned_q8K_sum6_qwarp32_kernel<<<
                dim3((out_dim + 31u) / 32u, 16u), 256, 0, stream>>>(
                (float *)out_mmq->ptr,
                (const uint2 *)down_soa->ptr,
                (const int4 *)((const char *)down_soa->ptr + q2_dm_bytes),
                (const uint2 *)((const char *)down_soa->ptr +
                                q2_dm_bytes + q2_sc_bytes),
                (const cuda_block_q8_K *)midq->ptr,
                (const int32_t *)selected->ptr,
                mid_dim / CUDA_QK_K, out_dim, 16u, n_total_expert);
        ok = cudaGetLastError() == cudaSuccess &&
             cudaStreamSynchronize(stream) == cudaSuccess;
    }
    if (ok) {
        ok = ds4_gpu_tensor_read(
                out_mmq, 0, out_q8k_host.data(),
                (uint64_t)16u * out_dim * sizeof(float));
    }
    if (ok) {
        memcpy(out_ref_prefix_host.data(), out_ref_host.data(),
               (uint64_t)16u * out_dim * sizeof(float));
    }

    /* Exercise every target-verifier width used by DSpark. The optimized
     * kernels retain the old aligned kernel above as their numerical oracle;
     * all five shapes must agree at the same strict production tolerance. */
    if (ok) ok = cuda_moe_gb10_validate_signs();
    for (uint32_t verifier_tokens = 2u;
         ok && verifier_tokens <= 6u;
         verifier_tokens++) {
        moe_gate_up_mid_aligned_q8K_gb10_kernel<<<
                dim3((mid_dim + 255u) / 256u,
                     verifier_tokens * n_expert),
                256, 0, stream>>>(
                (float *)mid_mmq->ptr,
                (const __half *)gate_soa->ptr,
                (const uint2 *)((const char *)gate_soa->ptr + iq2_dq_bytes),
                (const __half *)up_soa->ptr,
                (const uint2 *)((const char *)up_soa->ptr + iq2_dq_bytes),
                (const cuda_block_q8_K *)xq->ptr,
                (const int32_t *)selected->ptr,
                (const float *)weights->ptr,
                in_dim / CUDA_QK_K, mid_dim, n_total_expert, n_expert,
                test_clamp);
        ok = cudaGetLastError() == cudaSuccess;
        if (ok) {
            q8_K_quantize_kernel<<<
                    dim3(mid_dim / CUDA_QK_K,
                         verifier_tokens * n_expert),
                    256, 0, stream>>>(
                    (cuda_block_q8_K *)midq->ptr,
                    (const float *)mid_mmq->ptr,
                    mid_dim, verifier_tokens * n_expert);
            ok = cudaGetLastError() == cudaSuccess;
        }
        if (ok) {
            const size_t smem_bytes = (size_t)6u *
                (mid_dim / CUDA_QK_K) * sizeof(cuda_block_q8_K);
            moe_down_aligned_q8K_sum6_gb10_kernel<<<
                    dim3((out_dim + 63u) / 64u, verifier_tokens),
                    256, smem_bytes, stream>>>(
                    (float *)out_mmq->ptr,
                    (const uint2 *)down_soa->ptr,
                    (const int4 *)((const char *)down_soa->ptr + q2_dm_bytes),
                    (const uint2 *)((const char *)down_soa->ptr +
                                    q2_dm_bytes + q2_sc_bytes),
                    (const cuda_block_q8_K *)midq->ptr,
                    (const int32_t *)selected->ptr,
                    mid_dim / CUDA_QK_K, out_dim,
                    verifier_tokens, n_total_expert);
            ok = cudaGetLastError() == cudaSuccess &&
                 cudaStreamSynchronize(stream) == cudaSuccess;
        }
        const uint64_t verifier_values = (uint64_t)verifier_tokens * out_dim;
        if (ok) {
            ok = ds4_gpu_tensor_read(
                    out_mmq, 0, out_q8k_gb10_host.data(),
                    verifier_values * sizeof(float));
        }
        if (ok) {
            for (uint64_t i = 0; i < verifier_values; i++) {
                const double ref = (double)out_q8k_host[i];
                const double candidate = (double)out_q8k_gb10_host[i];
                const double ae = fabs(candidate - ref);
                const double re = fabs(ref) > 1.0e-12
                    ? ae / fabs(ref)
                    : (ae > 0.0 ? INFINITY : 0.0);
                q8k_gb10_max_abs = fmax(q8k_gb10_max_abs, ae);
                if (!isfinite(candidate) ||
                    (ae > 1.0e-6 && re > 1.0e-6)) {
                    ++q8k_gb10_bad;
                }
            }
        }
    }

    if (ok) {
        /* The established kernel stores gate/up after applying clamp, while
         * MMQ leaves both projections raw and clamps in the weighted SwiGLU
         * pass. Compare like with like without adding a production GPU pass. */
        for (uint64_t i = 0; i < pair_values; ++i) {
            if (gate_mmq_host[i] > test_clamp) gate_mmq_host[i] = test_clamp;
            if (up_mmq_host[i] > test_clamp) up_mmq_host[i] = test_clamp;
            if (up_mmq_host[i] < -test_clamp) up_mmq_host[i] = -test_clamp;
            if (gate_soa_host[i] > test_clamp) gate_soa_host[i] = test_clamp;
            if (up_soa_host[i] > test_clamp) up_soa_host[i] = test_clamp;
            if (up_soa_host[i] < -test_clamp) up_soa_host[i] = -test_clamp;
        }
    }

    struct parity_stats {
        double relative_rmse;
        double max_abs;
        uint64_t bad;
    };
    auto analyze_parity = [](const std::vector<float> &ref,
                             const std::vector<float> &candidate,
                             double abs_tol,
                             double rel_tol) -> parity_stats {
        double error = 0.0;
        double signal = 0.0;
        double max_abs = 0.0;
        uint64_t bad = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            if (!isfinite(ref[i]) || !isfinite(candidate[i])) {
                return {INFINITY, INFINITY, UINT64_MAX};
            }
            const double d = (double)candidate[i] - ref[i];
            const double ae = fabs(d);
            const double re = fabs((double)ref[i]) > 1.0e-12
                ? ae / fabs((double)ref[i])
                : (ae > 0.0 ? INFINITY : 0.0);
            error += d * d;
            signal += (double)ref[i] * ref[i];
            max_abs = fmax(max_abs, ae);
            if (ae > abs_tol && re > rel_tol) ++bad;
        }
        return {sqrt(error / fmax(signal, 1.0e-30)), max_abs, bad};
    };
    int parity_reported = 0;
    if (ok) {
        /* IQ2 MMQ and the established DS4 path quantize activations as Q8_1
         * and Q8_K respectively. Entrpi's parity harness therefore uses an
         * absolute tolerance scaled by sqrt(K), combined with a relative
         * tolerance, instead of rejecting near-zero accumulations on relative
         * error alone. The final MoE output remains the strict integration
         * criterion and uses the model's real clamp value. */
        const double iq2_abs_tol = 0.20 * sqrt((double)in_dim);
        const parity_stats gate_stats = analyze_parity(
                gate_ref_host, gate_mmq_host, iq2_abs_tol, 0.05);
        const parity_stats up_stats = analyze_parity(
                up_ref_host, up_mmq_host, iq2_abs_tol, 0.05);
        const parity_stats mid_stats = analyze_parity(
                mid_ref_host, mid_mmq_host, INFINITY, INFINITY);
        const parity_stats out_stats = analyze_parity(
                out_ref_host, out_mmq_host, INFINITY, INFINITY);
        const parity_stats soa_gate_stats = analyze_parity(
                gate_mmq_host, gate_soa_host, iq2_abs_tol, 0.05);
        const parity_stats soa_up_stats = analyze_parity(
                up_mmq_host, up_soa_host, iq2_abs_tol, 0.05);
        const parity_stats soa_mid_stats = analyze_parity(
                mid_mmq_host, mid_soa_host, INFINITY, INFINITY);
        const parity_stats soa_out_stats = analyze_parity(
                out_mmq_host, out_soa_host, INFINITY, INFINITY);
        const parity_stats fused_direct_stats = analyze_parity(
                out_soa_host, out_fused_direct_host, 2.0e-3, 2.0e-4);
        const parity_stats direct_out_stats = analyze_parity(
                out_soa_prefix_host, out_direct_host, INFINITY, INFINITY);
        const parity_stats q8k_out_stats = analyze_parity(
                out_ref_prefix_host, out_q8k_host, 1.0e-6, 1.0e-6);
        fprintf(stderr,
                "cuda-regression: raw-GGUF MMQ MoE parity "
                "gate=%.5f/%llu up=%.5f/%llu mid=%.5f final=%.5f rel-rmse/bad\n",
                gate_stats.relative_rmse,
                (unsigned long long)gate_stats.bad,
                up_stats.relative_rmse,
                (unsigned long long)up_stats.bad,
                mid_stats.relative_rmse,
                out_stats.relative_rmse);
        fprintf(stderr,
                "cuda-regression: aligned-SoA D2R MoE parity "
                "gate=%.5f/%llu up=%.5f/%llu mid=%.5f final=%.5f rel-rmse/bad\n",
                soa_gate_stats.relative_rmse,
                (unsigned long long)soa_gate_stats.bad,
                soa_up_stats.relative_rmse,
                (unsigned long long)soa_up_stats.bad,
                soa_mid_stats.relative_rmse,
                soa_out_stats.relative_rmse);
        fprintf(stderr,
                "cuda-regression: complete fused D2R MoE parity "
                "final=%.8f max=%.8g bad=%llu\n",
                fused_direct_stats.relative_rmse,
                fused_direct_stats.max_abs,
                (unsigned long long)fused_direct_stats.bad);
        fprintf(stderr,
                "cuda-regression: aligned-SoA direct down+sum6 parity "
                "final=%.5f rel-rmse\n",
                direct_out_stats.relative_rmse);
        fprintf(stderr,
                "cuda-regression: aligned-SoA Q8_K small-batch parity "
                "final=%.8f max=%.8g bad=%llu\n",
                q8k_out_stats.relative_rmse,
                q8k_out_stats.max_abs,
                (unsigned long long)q8k_out_stats.bad);
        fprintf(stderr,
                "cuda-regression: GB10 aligned MoE verifier N=2..6 parity "
                "max=%.8g bad=%llu\n",
                q8k_gb10_max_abs,
                (unsigned long long)q8k_gb10_bad);
        parity_reported = 1;
        ok = gate_stats.bad == 0 && up_stats.bad == 0 &&
             mid_stats.relative_rmse < 0.10 && out_stats.relative_rmse < 0.10 &&
             soa_gate_stats.bad == 0 && soa_up_stats.bad == 0 &&
             soa_mid_stats.relative_rmse < 0.05 &&
             soa_out_stats.relative_rmse < 0.05 &&
             fused_direct_stats.relative_rmse < 1.0e-3 &&
             fused_direct_stats.bad == 0 &&
             direct_out_stats.relative_rmse < 0.05 &&
             q8k_out_stats.bad == 0 && q8k_gb10_bad == 0;
    }

    if (!ok && !parity_reported) {
        fprintf(stderr,
                "cuda-regression: raw-GGUF MMQ MoE self-test failed before parity comparison\n");
    }

    for (ds4_gpu_tensor *tensor : allocations) ds4_gpu_tensor_free(tensor);
    return ok;
}

extern "C" int ds4_gpu_hc_split_sinkhorn_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *mix, const void *model_map, uint64_t model_size, uint64_t scale_offset, uint64_t base_offset, uint32_t n_hc, uint32_t sinkhorn_iters, float eps) {
    if (!out || !mix || !model_map || n_hc != 4) return 0;
    const uint64_t mix_bytes = 24ull * sizeof(float);
    if (scale_offset > model_size || model_size - scale_offset < 3ull * sizeof(float) ||
        base_offset > model_size || model_size - base_offset < mix_bytes ||
        mix->bytes < mix_bytes || out->bytes < mix_bytes) return 0;
    const float *scale = (const float *)cuda_model_range_ptr(model_map, scale_offset, 3ull * sizeof(float), "hc_scale");
    const float *base = (const float *)cuda_model_range_ptr(model_map, base_offset, mix_bytes, "hc_base");
    if (!scale || !base) return 0;
    uint32_t n_rows = (uint32_t)(mix->bytes / mix_bytes);
    if (out->bytes / mix_bytes < n_rows) n_rows = (uint32_t)(out->bytes / mix_bytes);
    hc_split_sinkhorn_kernel<<<(n_rows + 255) / 256, 256>>>(
        (float *)out->ptr, (const float *)mix->ptr,
        scale,
        base,
        n_rows, sinkhorn_iters, eps);
    return cuda_ok(cudaGetLastError(), "hc_split_sinkhorn launch");
}
extern "C" int ds4_gpu_hc_weighted_sum_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *weights, uint32_t n_embd, uint32_t n_hc) {
    if (!out || !residual_hc || !weights || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out->bytes / ((uint64_t)n_embd * sizeof(float)));
    hc_weighted_sum_kernel<<<((uint64_t)n_embd * n_tokens + 255) / 256, 256>>>(
        (float *)out->ptr, (const float *)residual_hc->ptr, (const float *)weights->ptr,
        n_embd, n_hc, n_tokens, n_hc);
    return cuda_ok(cudaGetLastError(), "hc_weighted_sum launch");
}
extern "C" int ds4_gpu_hc_weighted_sum_split_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    if (!out || !residual_hc || !split || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out->bytes / ((uint64_t)n_embd * sizeof(float)));
    uint32_t stride = (uint32_t)(2u * n_hc + n_hc * n_hc);
    hc_weighted_sum_kernel<<<((uint64_t)n_embd * n_tokens + 255) / 256, 256>>>(
        (float *)out->ptr, (const float *)residual_hc->ptr, (const float *)split->ptr,
        n_embd, n_hc, n_tokens, stride);
    return cuda_ok(cudaGetLastError(), "hc_weighted_sum_split launch");
}
extern "C" int ds4_gpu_hc_split_weighted_sum_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps) {
    if (!out || !split || !mix || !residual_hc || !model_map ||
        n_embd == 0 || n_hc != 4) {
        return 0;
    }
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t mix_bytes = mix_hc * sizeof(float);
    const uint64_t out_row_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t residual_row_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    if (out->bytes < out_row_bytes || out->bytes % out_row_bytes != 0 ||
        scale_offset > model_size || 3ull * sizeof(float) > model_size - scale_offset ||
        base_offset > model_size || mix_bytes > model_size - base_offset) {
        return 0;
    }
    uint64_t n_rows = out->bytes / out_row_bytes;
    if (mix->bytes < n_rows * mix_bytes ||
        split->bytes < n_rows * mix_bytes ||
        residual_hc->bytes < n_rows * residual_row_bytes) {
        return 0;
    }
    const float *scale = (const float *)cuda_model_range_ptr(model_map, scale_offset, 3ull * sizeof(float), "hc_scale");
    const float *base = (const float *)cuda_model_range_ptr(model_map, base_offset, mix_bytes, "hc_base");
    if (!scale || !base) return 0;
    hc_split_weighted_sum_fused_kernel<<<(uint32_t)n_rows, 256>>>(
            (float *)out->ptr,
            (float *)split->ptr,
            (const float *)mix->ptr,
            (const float *)residual_hc->ptr,
            scale,
            base,
            n_embd, n_hc, (uint32_t)n_rows, sinkhorn_iters, eps);
    return cuda_ok(cudaGetLastError(), "hc split weighted sum launch");
}
extern "C" int ds4_gpu_hc_split_weighted_sum_norm_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint64_t                norm_weight_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps,
        float                   norm_eps) {
    if (getenv("DS4_CUDA_DISABLE_HC_SPLIT_NORM_FUSED") == NULL) {
        if (!out || !norm_out || !split || !mix || !residual_hc || !model_map ||
            n_embd == 0 || n_hc != 4) {
            return 0;
        }
        const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
        const uint64_t mix_bytes = mix_hc * sizeof(float);
        const uint64_t out_row_bytes = (uint64_t)n_embd * sizeof(float);
        const uint64_t residual_row_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
        if (out->bytes < out_row_bytes || out->bytes % out_row_bytes != 0 ||
            norm_out->bytes < out->bytes ||
            scale_offset > model_size || 3ull * sizeof(float) > model_size - scale_offset ||
            base_offset > model_size || mix_bytes > model_size - base_offset ||
            norm_weight_offset > model_size ||
            (uint64_t)n_embd * sizeof(float) > model_size - norm_weight_offset) {
            return 0;
        }
        uint64_t n_rows = out->bytes / out_row_bytes;
        if (n_rows == 1) {
            if (mix->bytes < n_rows * mix_bytes ||
                split->bytes < n_rows * mix_bytes ||
                residual_hc->bytes < n_rows * residual_row_bytes) {
                return 0;
            }
            const float *scale = (const float *)cuda_model_range_ptr(model_map, scale_offset,
                    3ull * sizeof(float), "hc_scale");
            const float *base = (const float *)cuda_model_range_ptr(model_map, base_offset,
                    mix_bytes, "hc_base");
            const float *norm_w = (const float *)cuda_model_range_ptr(model_map, norm_weight_offset,
                    (uint64_t)n_embd * sizeof(float), "hc_norm_weight");
            if (!scale || !base || !norm_w) return 0;
            hc_split_weighted_sum_norm_fused_kernel<<<(uint32_t)n_rows, 256>>>(
                    (float *)out->ptr,
                    (float *)norm_out->ptr,
                    (float *)split->ptr,
                    (const float *)mix->ptr,
                    (const float *)residual_hc->ptr,
                    scale,
                    base,
                    norm_w,
                    n_embd, n_hc, (uint32_t)n_rows, sinkhorn_iters, eps, norm_eps);
            return cuda_ok(cudaGetLastError(), "hc split weighted sum norm launch");
        }
    }
    return ds4_gpu_hc_split_weighted_sum_tensor(out, split, mix, residual_hc,
                                                  model_map, model_size,
                                                  scale_offset, base_offset,
                                                  n_embd, n_hc,
                                                  sinkhorn_iters, eps) &&
           ds4_gpu_rms_norm_weight_tensor(norm_out, out, model_map, model_size,
                                            norm_weight_offset, n_embd, norm_eps);
}
extern "C" int ds4_gpu_output_hc_weights_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *pre,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        float                   eps) {
    if (!out || !pre || !model_map || n_hc == 0) return 0;
    const uint64_t row_bytes = (uint64_t)n_hc * sizeof(float);
    if (row_bytes == 0 || out->bytes < row_bytes || out->bytes % row_bytes != 0 ||
        pre->bytes < out->bytes ||
        scale_offset > model_size || sizeof(float) > model_size - scale_offset ||
        base_offset > model_size || row_bytes > model_size - base_offset) {
        return 0;
    }
    const uint64_t n_tokens = out->bytes / row_bytes;
    const float *scale = (const float *)cuda_model_range_ptr(model_map, scale_offset, sizeof(float), "output_hc_scale");
    const float *base = (const float *)cuda_model_range_ptr(model_map, base_offset, row_bytes, "output_hc_base");
    if (!scale || !base) return 0;
    uint64_t n = n_tokens * n_hc;
    output_hc_weights_kernel<<<(n + 255) / 256, 256>>>(
            (float *)out->ptr,
            (const float *)pre->ptr,
            scale,
            base,
            n_hc,
            (uint32_t)n_tokens,
            eps);
    return cuda_ok(cudaGetLastError(), "output hc weights launch");
}
extern "C" int ds4_gpu_hc_expand_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *post, const ds4_gpu_tensor *comb, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !block_out || !residual_hc || !post || !comb || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out_hc->bytes / ((uint64_t)n_hc * n_embd * sizeof(float)));
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    hc_expand_kernel<<<(n_elem + 255) / 256, 256>>>((float *)out_hc->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)residual_hc->ptr,
                                                    (const float *)post->ptr,
                                                    (const float *)comb->ptr,
                                                    n_embd, n_hc, n_tokens,
                                                    n_hc, n_hc * n_hc, 0);
    return cuda_ok(cudaGetLastError(), "hc_expand launch");
}
extern "C" int ds4_gpu_hc_expand_split_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !block_out || !residual_hc || !split || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out_hc->bytes / ((uint64_t)n_hc * n_embd * sizeof(float)));
    uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    const float *base = (const float *)split->ptr;
    hc_expand_kernel<<<(n_elem + 255) / 256, 256>>>((float *)out_hc->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)residual_hc->ptr,
                                                    base + n_hc,
                                                    base + 2u * n_hc,
                                                    n_embd, n_hc, n_tokens,
                                                    mix_hc, mix_hc, 0);
    return cuda_ok(cudaGetLastError(), "hc_expand_split launch");
}

static int cuda_hc_expand_split_norm_f16(
        ds4_gpu_tensor *out_hc,
        ds4_gpu_tensor *norm_h,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t n_embd,
        uint32_t n_hc,
        float eps) {
    if (!out_hc || !norm_h || !block_out || !residual_hc || !split ||
        n_embd == 0u || n_hc != 4u) {
        return 0;
    }
    const uint64_t hc_row = (uint64_t)n_hc * n_embd;
    if (out_hc->bytes < hc_row * sizeof(float) ||
        out_hc->bytes % (hc_row * sizeof(float)) != 0u) {
        return 0;
    }
    const uint64_t n_tokens64 = out_hc->bytes / (hc_row * sizeof(float));
    if (n_tokens64 < 128u || n_tokens64 > UINT32_MAX ||
        norm_h->bytes < n_tokens64 * hc_row * sizeof(__half) ||
        block_out->bytes < n_tokens64 * n_embd * sizeof(float) ||
        residual_hc->bytes < n_tokens64 * hc_row * sizeof(float) ||
        split->bytes < n_tokens64 * 24u * sizeof(float) ||
        (block_add && block_add->bytes < n_tokens64 * n_embd * sizeof(float))) {
        return 0;
    }
    hc_expand_norm_f16_kernel<<<(uint32_t)n_tokens64, 256>>>(
            (float *)out_hc->ptr,
            (__half *)norm_h->ptr,
            (const float *)block_out->ptr,
            block_add ? (const float *)block_add->ptr : (const float *)block_out->ptr,
            (const float *)residual_hc->ptr,
            (const float *)split->ptr,
            n_embd,
            n_hc,
            (uint32_t)n_tokens64,
            block_add ? 1 : 0,
            eps);
    return cuda_ok(cudaGetLastError(), "hc_expand_norm_f16 launch");
}

extern "C" int ds4_gpu_hc_expand_split_norm_f16_tensor(
        ds4_gpu_tensor *out_hc,
        ds4_gpu_tensor *norm_h,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t n_embd,
        uint32_t n_hc,
        float eps) {
    return cuda_hc_expand_split_norm_f16(out_hc, norm_h, block_out, NULL,
                                          residual_hc, split,
                                          n_embd, n_hc, eps);
}

extern "C" int ds4_gpu_hc_expand_add_split_norm_f16_tensor(
        ds4_gpu_tensor *out_hc,
        ds4_gpu_tensor *norm_h,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t n_embd,
        uint32_t n_hc,
        float eps) {
    return cuda_hc_expand_split_norm_f16(out_hc, norm_h, block_out, block_add,
                                          residual_hc, split,
                                          n_embd, n_hc, eps);
}

extern "C" int ds4_gpu_moe_down_hc_expand_add_norm_f16_tensor(
        ds4_gpu_tensor *out_hc,
        ds4_gpu_tensor *norm_h,
        const ds4_gpu_tensor *routed_down,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t n_embd,
        uint32_t n_expert,
        float eps) {
    if (!out_hc || !norm_h || !routed_down || !block_add ||
        !residual_hc || !split || n_embd == 0u || n_expert != 6u) {
        return 0;
    }
    const uint64_t hc_row = 4ull * n_embd;
    if (out_hc->bytes < hc_row * sizeof(float) ||
        out_hc->bytes % (hc_row * sizeof(float)) != 0u) {
        return 0;
    }
    const uint64_t n_tokens64 = out_hc->bytes / (hc_row * sizeof(float));
    if (n_tokens64 < 128u || n_tokens64 > UINT32_MAX ||
        norm_h->bytes < n_tokens64 * hc_row * sizeof(__half) ||
        routed_down->bytes < n_tokens64 * n_expert * n_embd * sizeof(float) ||
        block_add->bytes < n_tokens64 * n_embd * sizeof(float) ||
        residual_hc->bytes < n_tokens64 * hc_row * sizeof(float) ||
        split->bytes < n_tokens64 * 24u * sizeof(float)) {
        return 0;
    }
    moe_down_hc_expand_norm_f16_kernel<<<(uint32_t)n_tokens64, 256>>>(
            (float *)out_hc->ptr,
            (__half *)norm_h->ptr,
            (const float *)routed_down->ptr,
            (const float *)block_add->ptr,
            (const float *)residual_hc->ptr,
            (const float *)split->ptr,
            n_embd,
            n_expert,
            (uint32_t)n_tokens64,
            eps);
    return cuda_ok(cudaGetLastError(), "moe_down_hc_expand_norm_f16 launch");
}

extern "C" int ds4_gpu_hc_expand_split_half_tensor(
        ds4_gpu_tensor *out_hc,
        const ds4_gpu_tensor *block_out_h,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t n_embd,
        uint32_t n_hc) {
    (void)out_hc; (void)block_out_h; (void)residual_hc; (void)split;
    (void)n_embd; (void)n_hc;
    return 0;
}

extern "C" int ds4_gpu_hc_expand_add_split_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *block_add, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !block_out || !block_add || !residual_hc || !split || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out_hc->bytes / ((uint64_t)n_hc * n_embd * sizeof(float)));
    uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    const float *base = (const float *)split->ptr;
    hc_expand_kernel<<<(n_elem + 255) / 256, 256>>>((float *)out_hc->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)block_add->ptr,
                                                    (const float *)residual_hc->ptr,
                                                    base + n_hc,
                                                    base + 2u * n_hc,
                                                    n_embd, n_hc, n_tokens,
                                                    mix_hc, mix_hc, 1);
    return cuda_ok(cudaGetLastError(), "hc_expand_add_split launch");
}

extern "C" int ds4_gpu_prefill_epilogue_self_test(void) {
    enum {
        n_tokens = 128,
        n_embd = 64,
        n_hc = 4,
        n_expert = 6,
        split_width = 24
    };
    const float eps = 1.0e-6f;
    const uint64_t block_values = (uint64_t)n_tokens * n_embd;
    const uint64_t hc_values = block_values * n_hc;
    const uint64_t down_values = block_values * n_expert;
    const uint64_t split_values = (uint64_t)n_tokens * split_width;

    std::vector<float> block_host(block_values);
    std::vector<float> add_host(block_values);
    std::vector<float> residual_host(hc_values);
    std::vector<float> split_host(split_values);
    std::vector<float> down_host(down_values);
    uint32_t rng = 0x7f4a7c15u;
    auto random_small = [&rng]() -> float {
        rng = rng * 1664525u + 1013904223u;
        return ((float)(rng >> 8u) * (1.0f / 16777216.0f) - 0.5f) * 0.5f;
    };
    for (float &v : block_host) v = random_small();
    for (float &v : add_host) v = random_small();
    for (float &v : residual_host) v = random_small();
    for (float &v : down_host) v = random_small();
    for (uint32_t t = 0; t < n_tokens; ++t) {
        float *row = split_host.data() + (uint64_t)t * split_width;
        for (uint32_t i = 0; i < split_width; ++i) row[i] = random_small();
        for (uint32_t h = 0; h < n_hc; ++h) row[n_hc + h] += 0.75f;
    }
    down_host[(uint64_t)17u * n_expert * n_embd + 3u * n_embd + 11u] = NAN;
    down_host[(uint64_t)91u * n_expert * n_embd + 5u * n_embd + 37u] = INFINITY;

    std::vector<ds4_gpu_tensor *> allocations;
    auto alloc = [&allocations](uint64_t bytes) -> ds4_gpu_tensor * {
        ds4_gpu_tensor *tensor = ds4_gpu_tensor_alloc(bytes);
        if (tensor) allocations.push_back(tensor);
        return tensor;
    };
    ds4_gpu_tensor *block = alloc(block_values * sizeof(float));
    ds4_gpu_tensor *add = alloc(block_values * sizeof(float));
    ds4_gpu_tensor *residual = alloc(hc_values * sizeof(float));
    ds4_gpu_tensor *split = alloc(split_values * sizeof(float));
    ds4_gpu_tensor *down = alloc(down_values * sizeof(float));
    ds4_gpu_tensor *summed = alloc(block_values * sizeof(float));
    ds4_gpu_tensor *ref_hc = alloc(hc_values * sizeof(float));
    ds4_gpu_tensor *fused_hc = alloc(hc_values * sizeof(float));
    ds4_gpu_tensor *ref_norm = alloc(hc_values * sizeof(float));
    ds4_gpu_tensor *ref_half = alloc(hc_values * sizeof(__half));
    ds4_gpu_tensor *fused_half = alloc(hc_values * sizeof(__half));
    int ok = allocations.size() == 11u;
    if (ok) ok = ds4_gpu_tensor_write(block, 0, block_host.data(), block->bytes);
    if (ok) ok = ds4_gpu_tensor_write(add, 0, add_host.data(), add->bytes);
    if (ok) ok = ds4_gpu_tensor_write(residual, 0, residual_host.data(), residual->bytes);
    if (ok) ok = ds4_gpu_tensor_write(split, 0, split_host.data(), split->bytes);
    if (ok) ok = ds4_gpu_tensor_write(down, 0, down_host.data(), down->bytes);

#ifdef DS4_CUDA_TOKEN_GRAPH_BUILD
    const cudaStream_t stream = cudaStreamPerThread;
#else
    const cudaStream_t stream = 0;
#endif
    auto normalize_reference = [&]() -> int {
        if (!ds4_gpu_rms_norm_plain_rows_tensor(
                    ref_norm, ref_hc, n_hc * n_embd, n_tokens, eps)) {
            return 0;
        }
        f32_to_f16_kernel<<<(hc_values + 255u) / 256u, 256, 0, stream>>>(
                (__half *)ref_half->ptr, (const float *)ref_norm->ptr, hc_values);
        return cudaGetLastError() == cudaSuccess;
    };
    auto compare = [&](const char *label) -> int {
        std::vector<float> ref_hc_host(hc_values);
        std::vector<float> fused_hc_host(hc_values);
        std::vector<uint16_t> ref_half_host(hc_values);
        std::vector<uint16_t> fused_half_host(hc_values);
        if (cudaStreamSynchronize(stream) != cudaSuccess ||
            !ds4_gpu_tensor_read(ref_hc, 0, ref_hc_host.data(), ref_hc->bytes) ||
            !ds4_gpu_tensor_read(fused_hc, 0, fused_hc_host.data(), fused_hc->bytes) ||
            !ds4_gpu_tensor_read(ref_half, 0, ref_half_host.data(), ref_half->bytes) ||
            !ds4_gpu_tensor_read(fused_half, 0, fused_half_host.data(), fused_half->bytes)) {
            return 0;
        }
        double max_abs = 0.0;
        uint64_t bad_f32 = 0;
        uint64_t changed_f16 = 0;
        uint64_t bad_f16 = 0;
        uint32_t max_f16_ulp = 0;
        for (uint64_t i = 0; i < hc_values; ++i) {
            const double ae = fabs((double)fused_hc_host[i] - ref_hc_host[i]);
            max_abs = fmax(max_abs, ae);
            if (!isfinite(fused_hc_host[i]) || ae > 2.0e-6) ++bad_f32;
            if (fused_half_host[i] != ref_half_host[i]) {
                ++changed_f16;
                const uint16_t a = ref_half_host[i];
                const uint16_t b = fused_half_host[i];
                const int32_t ao = (a & 0x8000u)
                    ? 0x8000 - (int32_t)(a & 0x7fffu)
                    : 0x8000 + (int32_t)a;
                const int32_t bo = (b & 0x8000u)
                    ? 0x8000 - (int32_t)(b & 0x7fffu)
                    : 0x8000 + (int32_t)b;
                const uint32_t ulp = (uint32_t)abs(ao - bo);
                if (ulp > max_f16_ulp) max_f16_ulp = ulp;
                if (ulp > 1u) ++bad_f16;
            }
        }
        fprintf(stderr,
                "cuda-regression: fused prefill epilogue %s max_abs=%.3g "
                "f32_bad=%llu f16_changed=%llu max_f16_ulp=%u f16_bad=%llu\n",
                label, max_abs,
                (unsigned long long)bad_f32,
                (unsigned long long)changed_f16,
                max_f16_ulp,
                (unsigned long long)bad_f16);
        return bad_f32 == 0u && bad_f16 == 0u;
    };

    if (ok) {
        ok = ds4_gpu_hc_expand_add_split_tensor(
                     ref_hc, block, add, residual, split, n_embd, n_hc) &&
             normalize_reference() &&
             ds4_gpu_hc_expand_add_split_norm_f16_tensor(
                     fused_hc, fused_half, block, add, residual, split,
                     n_embd, n_hc, eps) &&
             compare("hc+add+rms");
    }
    if (ok) {
        moe_mmq_sum_guard_kernel<<<
                (block_values + 255u) / 256u, 256, 0, stream>>>(
                (float *)summed->ptr, (const float *)down->ptr,
                n_embd, n_expert, n_tokens);
        ok = cudaGetLastError() == cudaSuccess &&
             ds4_gpu_hc_expand_add_split_tensor(
                     ref_hc, summed, add, residual, split, n_embd, n_hc) &&
             normalize_reference() &&
             ds4_gpu_moe_down_hc_expand_add_norm_f16_tensor(
                     fused_hc, fused_half, down, add, residual, split,
                     n_embd, n_expert, eps) &&
             compare("moe-sum+hc+rms");
    }
    if (ok) {
        enum {
            rope_head_dim = 32,
            rope_n_rot = 16,
            rope_n_groups = 4,
            rope_group_dim = 64,
            rope_n_head = rope_n_groups * rope_group_dim / rope_head_dim
        };
        const uint32_t rope_pos0 = 98304u;
        const uint32_t rope_ctx = 16384u;
        const float rope_base = 10000.0f;
        const float rope_scale = 0.25f;
        const float rope_ext = 1.0f;
        const float rope_attn = 0.8f;
        const float rope_beta_fast = 32.0f;
        const float rope_beta_slow = 1.0f;
        ok = ds4_gpu_tensor_write(
                     ref_hc, 0, residual_host.data(), ref_hc->bytes) &&
             ds4_gpu_tensor_write(
                     fused_hc, 0, residual_host.data(), fused_hc->bytes) &&
             ds4_gpu_rope_tail_tensor(
                     ref_hc, n_tokens, rope_n_head, rope_head_dim,
                     rope_n_rot, rope_pos0, rope_ctx, true,
                     rope_base, rope_scale, rope_ext, rope_attn,
                     rope_beta_fast, rope_beta_slow) != 0;
        if (ok) {
            attention_pack_group_heads_f16_kernel<<<
                    (hc_values / 2u + 255u) / 256u, 256, 0, stream>>>(
                    (__half *)ref_half->ptr,
                    (const float *)ref_hc->ptr,
                    n_tokens, rope_n_groups, rope_group_dim);
            ok = cudaGetLastError() == cudaSuccess;
        }
        if (ok) {
            attention_inverse_rope_pack_group_heads_f16_kernel<<<
                    (hc_values / 2u + 255u) / 256u, 256, 0, stream>>>(
                    (__half *)fused_half->ptr,
                    (const float *)fused_hc->ptr,
                    n_tokens, rope_n_groups, rope_group_dim,
                    rope_head_dim, rope_n_rot, rope_pos0, rope_ctx,
                    rope_base, rope_scale, rope_ext, rope_attn,
                    rope_beta_fast, rope_beta_slow);
            ok = cudaGetLastError() == cudaSuccess;
        }
        std::vector<uint16_t> ref_rope_host(hc_values);
        std::vector<uint16_t> fused_rope_host(hc_values);
        if (ok) {
            ok = cudaStreamSynchronize(stream) == cudaSuccess &&
                 ds4_gpu_tensor_read(ref_half, 0, ref_rope_host.data(),
                                      ref_half->bytes) &&
                 ds4_gpu_tensor_read(fused_half, 0, fused_rope_host.data(),
                                      fused_half->bytes);
        }
        uint64_t rope_changed = 0;
        uint64_t rope_bad = 0;
        uint64_t rope_first = UINT64_MAX;
        uint16_t rope_first_ref = 0;
        uint16_t rope_first_fused = 0;
        uint32_t rope_max_ulp = 0;
        if (ok) {
            for (uint64_t i = 0; i < hc_values; ++i) {
                if (ref_rope_host[i] != fused_rope_host[i]) {
                    ++rope_changed;
                    const uint16_t a = ref_rope_host[i];
                    const uint16_t b = fused_rope_host[i];
                    const int32_t ao = (a & 0x8000u)
                        ? 0x8000 - (int32_t)(a & 0x7fffu)
                        : 0x8000 + (int32_t)a;
                    const int32_t bo = (b & 0x8000u)
                        ? 0x8000 - (int32_t)(b & 0x7fffu)
                        : 0x8000 + (int32_t)b;
                    const uint32_t ulp = (uint32_t)abs(ao - bo);
                    if (ulp > rope_max_ulp) rope_max_ulp = ulp;
                    if (rope_first == UINT64_MAX) {
                        rope_first = i;
                        rope_first_ref = a;
                        rope_first_fused = b;
                    }
                    if (ulp > 1u) ++rope_bad;
                }
            }
            fprintf(stderr,
                    "cuda-regression: fused inverse-RoPE pack changed=%llu "
                    "max_ulp=%u bad=%llu first=%llu ref=0x%04x fused=0x%04x\n",
                    (unsigned long long)rope_changed,
                    rope_max_ulp,
                    (unsigned long long)rope_bad,
                    (unsigned long long)rope_first,
                    (unsigned)rope_first_ref,
                    (unsigned)rope_first_fused);
            ok = rope_bad == 0u;
        }
    }
    for (ds4_gpu_tensor *tensor : allocations) ds4_gpu_tensor_free(tensor);
    return ok;
}

extern "C" int ds4_gpu_hc_expand_add_split_half_add_tensor(
        ds4_gpu_tensor *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add_h,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t n_embd,
        uint32_t n_hc) {
    (void)out_hc; (void)block_out; (void)block_add_h; (void)residual_hc;
    (void)split; (void)n_embd; (void)n_hc;
    return 0;
}

extern "C" int ds4_gpu_shared_down_hc_expand_q8_0_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *shared_mid,
        const ds4_gpu_tensor *routed_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (getenv("DS4_CUDA_DISABLE_Q8_HC_EXPAND_FUSED") == NULL) {
        return cuda_matmul_q8_0_hc_expand_tensor_labeled(out_hc, shared_out,
                                                        model_map, model_size,
                                                        weight_offset,
                                                        in_dim, out_dim,
                                                        shared_mid,
                                                        routed_out,
                                                        residual_hc,
                                                        split,
                                                        n_embd, n_hc,
                                                        "shared_down_hc_expand");
    }
    return ds4_gpu_matmul_q8_0_tensor(shared_out, model_map, model_size,
                                        weight_offset, in_dim, out_dim,
                                        shared_mid, 1) &&
           ds4_gpu_hc_expand_add_split_tensor(out_hc, shared_out, routed_out,
                                                residual_hc, split, n_embd, n_hc);
}

extern "C" int ds4_gpu_matmul_q8_0_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (getenv("DS4_CUDA_DISABLE_Q8_HC_EXPAND_FUSED") == NULL) {
        return cuda_matmul_q8_0_hc_expand_tensor_labeled(out_hc, block_out,
                                                        model_map, model_size,
                                                        weight_offset,
                                                        in_dim, out_dim,
                                                        x,
                                                        NULL,
                                                        residual_hc,
                                                        split,
                                                        n_embd, n_hc,
                                                        "q8_hc_expand");
    }
    return ds4_gpu_matmul_q8_0_tensor(block_out, model_map, model_size,
                                        weight_offset, in_dim, out_dim, x, 1) &&
           ds4_gpu_hc_expand_split_tensor(out_hc, block_out, residual_hc,
                                            split, n_embd, n_hc);
}
