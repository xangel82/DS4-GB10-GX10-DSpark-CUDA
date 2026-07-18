// SPDX-License-Identifier: MIT
// ds4_mmq.cu - host wrapper around llama.cpp's vendored mul_mat_q kernels.
//
// Implements the public ds4_mmq_* entry points and explicitly instantiates
// the mul_mat_q_case<T> template for each quant type the caller needs.
//
// Status:
//   Q8_0 dense ............ implemented, parity-tested against CPU reference
//   Q2_K dense ............ pending (Phase 3)
//   IQ2_XXS dense ......... pending (Phase 3)
//   Q8_0 MoE _id .......... pending (Phase 4)
//   Q2_K MoE _id .......... pending (Phase 4)
//   IQ2_XXS MoE _id ....... pending (Phase 4)

#include "ds4_mmq.h"

#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"
#include "ds4_mmq_d2r.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define DS4_MMQ_HAS_NVTX 1
#endif
#endif
#ifndef DS4_MMQ_HAS_NVTX
#define DS4_MMQ_HAS_NVTX 0
#endif

static bool ds4_mmq_nvtx_requested() {
    static int enabled = -1;
    if (enabled < 0) {
        const char *nvtx = getenv("DS4_CUDA_NVTX");
        const char *capture = getenv("DS4_CUDA_NSYS_PREFILL_START_POS");
        enabled = (nvtx != nullptr && std::strcmp(nvtx, "1") == 0) ||
                  (capture != nullptr && capture[0] != '\0');
    }
    return enabled != 0;
}

static uint64_t ds4_mmq_nvtx_payload(uint32_t first, uint32_t second) {
    return ((uint64_t)first << 32) | second;
}

class ds4_mmq_nvtx_scope {
public:
    ds4_mmq_nvtx_scope(const char *name, uint64_t payload, bool enabled)
        : active_(enabled) {
#if DS4_MMQ_HAS_NVTX
        if (active_) {
            nvtxEventAttributes_t attr = {};
            attr.version = NVTX_VERSION;
            attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            attr.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
            attr.payload.ullValue = payload;
            attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
            attr.message.ascii = name;
            (void)nvtxRangePushEx(&attr);
        }
#else
        (void)name;
        (void)payload;
        active_ = false;
#endif
    }

    ~ds4_mmq_nvtx_scope() {
#if DS4_MMQ_HAS_NVTX
        if (active_) (void)nvtxRangePop();
#endif
    }

    ds4_mmq_nvtx_scope(const ds4_mmq_nvtx_scope &) = delete;
    ds4_mmq_nvtx_scope &operator=(const ds4_mmq_nvtx_scope &) = delete;

private:
    bool active_;
};

// ----------------------------------------------------------------------------
// Init
// ----------------------------------------------------------------------------

// Step 7 task #29: experimental persistent Q8_1 scratch buffer.
//
// Hypothesis: ggml_cuda_pool_alloc inside ds4_mmq_moe_vec_impl records a
// cudaMallocAsync graph node into the captured layer graph.  At replay
// time the alloc node returns a (potentially different) address, but the
// matvec kernel's pointer argument was baked in at capture time.  Result:
// the matvec reads stale/wrong memory and produces a different output
// than eager execution, even with identical inputs.
//
// Mitigation under test: pre-allocate a persistent device buffer at
// startup via plain cudaMalloc (NOT cudaMallocAsync, NOT inside any
// capture).  When the env flag DS4_CUDA_MMQ_Q81_PERSISTENT=1 is set,
// ds4_mmq_moe_vec_impl uses this persistent buffer instead of pool_alloc.
// If slot 213 (routed_gate) now matches OFF, the pool's interaction with
// graph capture was the root cause.  If it still differs, the bug is in
// the captured matvec kernel itself.
//
// Sized for V4 Flash decode shapes: gate Q8_1 ~8 KB, down Q8_1 ~14 KB.
// 256 KB allocation gives generous headroom for short prefill batches.
static void *g_q81_scratch_ptr   = nullptr;
static size_t g_q81_scratch_bytes = 0;
static bool   g_q81_scratch_enabled = false;

// Read by ds4_mmq_moe_vec_impl; non-zero means use the persistent buffer.
// Set by ds4_mmq_init once based on env.  (Single-threaded GPU work; no
// atomicity needed.)
extern "C" int ds4_mmq_q81_persistent_enabled(void) {
    return g_q81_scratch_enabled ? 1 : 0;
}

extern "C" void *ds4_mmq_q81_scratch_ptr(void) {
    return g_q81_scratch_ptr;
}

// M2-Inc2a: registry of producer-emitted q8_1 activations (ds4_cuda.cu).
// A hit returns canonical block_q8_1 codes for this exact activation
// pointer (bit-exact vs quantize_row_q8_1_cuda), letting the caller skip
// its quantize prelude.  Only valid for single-token unpadded rows
// (ne10_padded == K); the registry itself guarantees freshness (slots are
// reset by the producing entry every layer and pops are one-shot).
extern "C" int ds4_cuda_q8_fold_take_q81(const void *src, uint64_t in_dim,
                                         const void **q81);
static char *ds4_mmq_folded_q81(const float *X_f32, int64_t K, int n_tokens,
                                int64_t ne10_padded) {
    if (n_tokens != 1 || ne10_padded != K) return nullptr;
    const void *p = nullptr;
    if (!ds4_cuda_q8_fold_take_q81((const void *)X_f32, (uint64_t)K, &p)) return nullptr;
    static int logged = 0;
    if (!logged) {
        logged = 1;
        fprintf(stderr, "ds4: M2-Inc2a q8_1 activation fold active (mmvq decode)\n");
    }
    return (char *)(uintptr_t)p;
}

// Default ON (2026-07-09 gated increment: same-boot ABBA 427->493 tok/s @12k,
// gsm8k 97.5 / mbpp 90). DS4_MMQ_D2R=0 is the kill switch back to the
// mul_mat_q SoA-tile down path.
static bool d2r_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("DS4_MMQ_D2R");
        cached = (env && env[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

static bool d2r_iq2_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("DS4_MMQ_D2R_IQ2");
        cached = (env && env[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

// Blanket output zeroing on the dense/MoE-down/pair GEMM entries.  Added by
// 82b2622 as belt-and-suspenders while root-causing the cont BOS spam; the
// actual roots were fixed in the same commit (stream-K fixup write_back goes
// dense + tmp_fixup zeroed + ncols_max=ne_get_rows), after which every
// element a consumer reads is stored by the GEMM itself and the zeroing was
// ~1.0 s/12k-admission of pure memset tax.  Default OFF (2026-07-09 gated
// increment: L42 deep tensors BIT-IDENTICAL with/without, same-boot ABBA
// 641.5 -> 678 tok/s @12k, gsm8k 119/120 / mbpp 36/40 / canary=[]).
// DS4_MMQ_OUT_MEMSET=1 restores the zeroing.
static bool out_memset_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("DS4_MMQ_OUT_MEMSET");
        cached = (env && env[0] == '1') ? 1 : 0;
        if (cached) {
            fprintf(stderr, "ds4: DS4_MMQ_OUT_MEMSET=1 - blanket GEMM output zeroing restored\n");
        }
    }
    return cached != 0;
}

static int64_t d2r_min_cols() {
    static int64_t cached = -1;
    if (cached < 0) {
        cached = 1024;
        const char *env = getenv("DS4_MMQ_D2R_MIN_COLS");
        if (env && env[0] != '\0') {
            char *end = nullptr;
            const long v = strtol(env, &end, 10);
            if (end != env && v > 0) {
                cached = (int64_t)v;
            }
        }
    }
    return cached;
}

extern "C" size_t ds4_mmq_q81_scratch_bytes(void) {
    return g_q81_scratch_bytes;
}

extern "C" int ds4_mmq_init(int device) {
    if (device < 0) {
        fprintf(stderr, "ds4_mmq_init: invalid device %d\n", device);
        return -1;
    }
    ggml_cuda_set_device(device);
    // Trigger lazy population of the device-info singleton.
    const auto & info = ggml_cuda_info();
    if (info.device_count == 0) {
        fprintf(stderr, "ds4_mmq_init: no CUDA devices found\n");
        return -1;
    }
    if (device >= info.device_count) {
        fprintf(stderr, "ds4_mmq_init: device %d out of range (have %d)\n",
                device, info.device_count);
        return -1;
    }

    // Step 7 task #29: pre-allocate persistent Q8_1 scratch if enabled.
    // Must happen here (before any layer-graph capture) so the cudaMalloc
    // is not forbidden by capture-mode restrictions, and so the kernel
    // pointer arg baked into the captured graph stays valid at replay.
    if (getenv("DS4_CUDA_MMQ_Q81_PERSISTENT") && !g_q81_scratch_ptr) {
        const size_t bytes = 256 * 1024;
        cudaError_t err = cudaMalloc(&g_q81_scratch_ptr, bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4_mmq_init: cudaMalloc(q81_scratch %zu B) failed: %s; "
                            "falling back to pool_alloc\n",
                    bytes, cudaGetErrorString(err));
            g_q81_scratch_ptr = nullptr;
            g_q81_scratch_enabled = false;
        } else {
            g_q81_scratch_bytes = bytes;
            g_q81_scratch_enabled = true;
            fprintf(stderr, "ds4_mmq_init: persistent Q8_1 scratch enabled (%zu B at %p)\n",
                    bytes, g_q81_scratch_ptr);
        }
    }
    return 0;
}

// ----------------------------------------------------------------------------
// Gating: when should the caller choose mmq over dequant+cublas?
//
// Body lifted verbatim from llama.cpp's ggml/src/ggml-cuda/mmq.cu:267-372
// (we do not vendor mmq.cu itself, since its other half talks to ggml_tensor
// and ggml_backend internals we don't carry over).
// ----------------------------------------------------------------------------

static bool ds4_should_use_mmq_impl(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    GGML_UNUSED(type); GGML_UNUSED(cc); GGML_UNUSED(ne11); GGML_UNUSED(n_experts);
    return false;
#endif

    bool mmq_supported;
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }
    if (!mmq_supported) return false;

    if (turing_mma_available(cc)) {
        return true;
    }
    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        return false;
    }
#ifdef GGML_CUDA_FORCE_MMQ
    GGML_UNUSED(ne11); GGML_UNUSED(n_experts);
    return true;
#endif

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }
    if (amd_mfma_available(cc)) {
        if (GGML_CUDA_CC_IS_CDNA3(cc)) return true;
        if (n_experts > 64 || ne11 <= 128) return true;
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 ||
            type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) return true;
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) return true;
        return false;
    }
    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            if (n_experts >= 64) return true;
            switch (type) {
                case GGML_TYPE_Q2_K: return ne11 <= 128;
                case GGML_TYPE_Q6_K: return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default: return true;
            }
        }
        return true;
    }
    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}

extern "C" int ds4_mmq_should_use(int type_x, int64_t ne11, int64_t n_experts) {
    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;
    const enum ggml_type t = (enum ggml_type) type_x;
    return ds4_should_use_mmq_impl(t, cc, ne11, n_experts) ? 1 : 0;
}

// ----------------------------------------------------------------------------
// Dense matmul implementation, shared across all three quant types.
//
// Computes  out[col, row] = sum_k W[row, k] * X[k, col]   with W in the
// type-specific block layout and X / out in F32 (X K-innermost row-major,
// out column-major out[col*M + row]).
//
// Mirrors upstream mmq.cu:154-159 (the no-ids branch) but builds mmq_args
// from plain pointers + shape ints instead of ggml_tensor introspection.
// ----------------------------------------------------------------------------

// Per-device singleton context. Owns the pool for stream-K fixup scratch.
// Phase 4 will make this per-stream as well; for now a single context per
// device is sufficient for the dense path.
namespace {

__global__ static void ds4_mmq_sanitize_f32_kernel(float *p, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = p[i];
    if (!isfinite(v)) p[i] = 0.0f;
}

static void ds4_mmq_sanitize_f32(float *p, uint64_t n, cudaStream_t stream) {
    if (!p || n == 0) return;
    ds4_mmq_sanitize_f32_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, stream>>>(p, n);
}

ggml_backend_cuda_context * get_ctx_for_device(int device) {
    static ggml_backend_cuda_context * cached[GGML_CUDA_MAX_DEVICES] = {};
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) return nullptr;
    if (!cached[device]) {
        cached[device] = new ggml_backend_cuda_context(device);
    }
    return cached[device];
}

template <ggml_type type>
int ds4_mmq_dense_impl(
        const char  * tag,
        const void  * W,
        const float * X_f32,
        float       * out_f32,
        int           M,
        int           N,
        int           K,
        cudaStream_t  stream) {

    if (!W || !X_f32 || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (K <= 0 || M <= 0 || N <= 0) {
        fprintf(stderr, "%s: bad shape M=%d N=%d K=%d\n", tag, M, N, K);
        return -1;
    }
    if (K % 256 != 0) {
        // mmq requires K to be a multiple of the largest super-block size
        // it sees during the inner tile loop, which is QK_K=256.
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;

    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }

    /* Task #22 fix: order the pool's cudaMallocAsync/cudaFreeAsync on the SAME
     * stream the kernels below launch on.  The pool defaults to
     * cudaStreamPerThread; with kernels on the legacy stream the RAII free is
     * ordered on an EMPTY stream, so the driver can recycle/remap the scratch
     * while the in-flight quantize/GEMM still reads it -> intermittent illegal
     * access under shape churn (the batched-draft early-step crash).  The vec
     * impls already do this (graph-capture fix); the batched impls were missed. */
    ds4_pool_set_stream(stream);

    // 1. Quantize the F32 activation into the mmq Q8_1 format. The
    //    target_type parameter only affects the activation scale strategy
    //    that the quantizer picks (matched to the weight type's K-block
    //    layout); the output buffer is always Q8_1.
    const int64_t ne00         = K;
    const int64_t ne10_padded  = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const int64_t ne11         = N;
    const int64_t ne12         = 1;
    const int64_t ne13         = 1;

    const size_t nbytes_src1_q8_1 =
        ne13 * ne12 * ne11 * ne10_padded * sizeof(block_q8_1) / QK8_1 +
        get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);

    ggml_cuda_pool_alloc<char> src1_q8_1(ctx->pool(), nbytes_src1_q8_1);

    // S1.1a fix: the mmq Y (activation) buffer is over-allocated for the kernel's
    // tail-tile reads (the +mmq_x_max blocks above), and ne11 columns may not fill
    // the final column tile -- but quantize_mmq_q8_1_cuda only writes the ne11 valid
    // columns.  The mmq kernel (mmq.cuh:3528) unconditionally loads the full column
    // tile, reading the never-written tail.  Pool allocs reuse stale device memory,
    // so that tail is non-deterministic: any allocator/stream perturbation (e.g. an
    // MTP draft's cudaMalloc) changes it and flips a near-threshold argmax in the
    // batched forward (confirmed by compute-sanitizer --tool initcheck on a PRO6000
    // / sm_120: 4-byte uninitialized __global__ read in mul_mat_q_process_tile).
    // The tail's dot-products are masked out by write_back, so only their
    // non-determinism matters; zero the buffer so the tail is a deterministic zero
    // (a zero q8_1 block contributes 0 to the dot product).
    cudaMemsetAsync(src1_q8_1.get(), 0, nbytes_src1_q8_1, stream);

    quantize_mmq_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1.get(),
        type, /*ne00=*/K, /*s11=*/(int64_t)K, /*s12=*/0, /*s13=*/0,
        /*ne0=*/ne10_padded, /*ne1=*/ne11, /*ne2=*/ne12, /*ne3=*/ne13,
        stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize failed: %s\n", tag, cudaGetErrorString(err));
        return -2;
    }

    // 2. Build mmq_args. stride_row_x is in WEIGHT BLOCKS per row, which
    //    is K / blck_size(type). Q8_0 has block size 32; Q2_K and IQ2_XXS
    //    are K-quants with block size 256.
    const int64_t blck   = ggml_blck_size(type);
    const int64_t s01    = (int64_t)K / blck;
    const int64_t s1     = (int64_t)M;
    const int64_t s12    = ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13    = ne12 * s12;

    const bool use_stream_k =
        (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA) ||
        GGML_CUDA_CC_IS_CDNA(cc);

    if (out_memset_enabled()) {
        cudaMemsetAsync(out_f32, 0, (size_t)M * (size_t)N * sizeof(float), stream);
    }

    const mmq_args args = {
        /*x=*/(const char *)W,
        /*type_x=*/type,
        /*y=*/(const int *)src1_q8_1.get(),
        /*ids_dst=*/nullptr,
        /*expert_bounds=*/nullptr,
        /*dst=*/out_f32,
        /*ncols_x=*/ne00,    /*nrows_x=*/(int64_t)M,    /*ncols_dst=*/ne11,
        /*stride_row_x=*/s01,/*ncols_y=*/ne11,          /*nrows_dst=*/s1,
        /*nchannels_x=*/1,   /*nchannels_y=*/1,
        /*stride_channel_x=*/0, /*stride_channel_y=*/s12, /*stride_channel_dst=*/0,
        /*nsamples_x=*/1,    /*nsamples_y=*/1,
        /*stride_sample_x=*/0, /*stride_sample_y=*/s13, /*stride_sample_dst=*/0,
        /*use_stream_k=*/use_stream_k,
        /*ncols_max=*/ne11,
    };

    mul_mat_q_case<type>(*ctx, args, stream);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: mul_mat_q_case launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    ds4_mmq_sanitize_f32(out_f32, (uint64_t)M * (uint64_t)N, stream);
    return 0;
}

} // anonymous namespace

extern "C" int ds4_mmq_q8_0_dense(
        const void * W, const float * X, float * out,
        int M, int N, int K, cudaStream_t stream) {
    return ds4_mmq_dense_impl<GGML_TYPE_Q8_0>("ds4_mmq_q8_0_dense", W, X, out, M, N, K, stream);
}

// Dense Q8_0 D2R entry: same activation quantize + scratch treatment as
// ds4_mmq_dense_impl (incl. the S1.1a zero for the never-written tail), then
// the D2R kernel on the kind-5 aligned artifact instead of mul_mat_q_case.
// No out-memset / trailing sanitize: the D2R epilogue writes every element
// through an isfinite guard.  Caller (ds4_cuda.cu) resolves W_aligned and
// gates on shape (M%128, K%1024, K<=4096) + n_tok.
extern "C" int ds4_mmq_q8_0_dense_d2r(
        const void * W_aligned, const float * X_f32, float * out_f32,
        int M, int N, int K, cudaStream_t stream) {
    const char *tag = "ds4_mmq_q8_0_dense_d2r";
    if (!W_aligned || !X_f32 || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || (M % 128) != 0 || N <= 0 || K <= 0 || (K % 1024) != 0) {
        fprintf(stderr, "%s: bad shape M=%d N=%d K=%d\n", tag, M, N, K);
        return -1;
    }
    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;
    if (!ds4_mmq_q8_0_dense_d2r_available(cc)) {
        return -1;
    }
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }
    ds4_pool_set_stream(stream);

    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    // Slack: the guarded last col tile reads up to 128 blocks past N*K/128.
    const int64_t slack_blocks = std::max<int64_t>(get_mmq_x_max_host(cc), 128);
    const size_t nbytes_src1_q8_1 =
        (int64_t)N * ne10_padded * sizeof(block_q8_1) / QK8_1 +
        slack_blocks * sizeof(block_q8_1_mmq);

    ggml_cuda_pool_alloc<char> src1_q8_1(ctx->pool(), nbytes_src1_q8_1);
    cudaMemsetAsync(src1_q8_1.get(), 0, nbytes_src1_q8_1, stream);

    quantize_mmq_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1.get(),
        GGML_TYPE_Q8_0, /*ne00=*/K, /*s11=*/(int64_t)K, /*s12=*/0, /*s13=*/0,
        /*ne0=*/ne10_padded, /*ne1=*/(int64_t)N, /*ne2=*/1, /*ne3=*/1,
        stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize failed: %s\n", tag, cudaGetErrorString(err));
        return -2;
    }
    return ds4_mmq_q8_0_dense_d2r_launch(W_aligned, src1_q8_1.get(), out_f32,
                                         M, N, K, stream);
}

extern "C" int ds4_mmq_q2_K_dense(
        const void * W, const float * X, float * out,
        int M, int N, int K, cudaStream_t stream) {
    return ds4_mmq_dense_impl<GGML_TYPE_Q2_K>("ds4_mmq_q2_K_dense", W, X, out, M, N, K, stream);
}

extern "C" int ds4_mmq_iq2_xxs_dense(
        const void * W, const float * X, float * out,
        int M, int N, int K, cudaStream_t stream) {
    return ds4_mmq_dense_impl<GGML_TYPE_IQ2_XXS>("ds4_mmq_iq2_xxs_dense", W, X, out, M, N, K, stream);
}

extern "C" int ds4_mmq_q4_K_dense(
        const void * W, const float * X, float * out,
        int M, int N, int K, cudaStream_t stream) {
    return ds4_mmq_dense_impl<GGML_TYPE_Q4_K>("ds4_mmq_q4_K_dense", W, X, out, M, N, K, stream);
}

// ----------------------------------------------------------------------------
// MoE matmul implementation, shared across all three quant types.
//
// Mirrors upstream mmq.cu:163-222 (the ids != nullptr branch).  Caller
// provides:
//   - per-expert weights stacked contiguously
//   - per-token activations [n_tokens, K]
//   - routing table ids[t, s] = expert id
// The wrapper invokes:
//   1. ggml_cuda_launch_mm_ids_helper to build (ids_src1, ids_dst,
//      expert_bounds) - permutations that sort assignments by expert.
//   2. quantize_mmq_q8_1_cuda with ids_src1 - gathers and quantizes the
//      activation into the expert-major flat layout.
//   3. mul_mat_q_case<type> with ids_dst + expert_bounds - the matmul.
// ----------------------------------------------------------------------------

namespace {

template <ggml_type type>
int ds4_mmq_moe_impl(
        const char    * tag,
        const void    * W,
        const float   * X_f32,
        const int32_t * ids,
        float         * out_f32,
        int             M,
        int             K,
        int             n_tokens,
        int             n_experts,
        int             n_expert_used,
        cudaStream_t    stream,
        /* ds4 (P4 Inc3): optional aligned-SoA artifact; when non-null the mmq
         * kernel loads tiles from it directly and W is ignored (see mmq_args). */
        const char    * x_soa      = NULL,
        int64_t         soa_blocks = 0,
        /* ds4 (P3): false skips the whole-buffer nonfinite pass; only valid
         * when every consumer sanitizes at read (the routed-MoE swiglu/sum
         * kernels do). */
        bool            sanitize_out = true) {

    if (!W || !X_f32 || !ids || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || K <= 0 || n_tokens <= 0 || n_experts <= 0 || n_expert_used <= 0) {
        fprintf(stderr, "%s: bad shape M=%d K=%d ntok=%d nexp=%d nused=%d\n",
                tag, M, K, n_tokens, n_experts, n_expert_used);
        return -1;
    }
    if (K % 256 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }
    if (n_expert_used > n_experts) {
        fprintf(stderr, "%s: n_expert_used=%d > n_experts=%d\n", tag, n_expert_used, n_experts);
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;

    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }

    ds4_pool_set_stream(stream);  /* task #22: pool ops must be stream-ordered with the kernels (see ds4_mmq_dense_impl) */

    const int64_t ne_get_rows  = (int64_t)n_tokens * n_expert_used;
    const int64_t ne00         = K;
    const int64_t ne10_padded  = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const int64_t ne11         = 1;             // src1 rows per channel (one per token)
    const int64_t ne12         = n_tokens;      // src1 channels (= tokens)
    const int64_t blck         = ggml_blck_size(type);
    const int64_t s01          = (int64_t)K / blck;
    const int64_t s02          = (int64_t)M * s01;   // per-expert weight stride in blocks

    // 1. Build the expert-major work map.
    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx->pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx->pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx->pool(), n_experts + 1);

    // Task #22 root-cause fix: mm_ids_helper COMPACTS - it only writes ids_src1
    // entries for in-range router ids and drops invalid ones (the router's NaN
    // path emits -1 by design), so with any dropped id the tail of ids_src1
    // stays unwritten pool memory.  quantize_mmq_q8_1's grid covers all
    // ne_get_rows rows and gathers x rows via ids_src1[i1] unconditionally
    // (quantize.cu:304), so a stale/garbage tail entry becomes a wild OOB read
    // (the intermittent batched-draft illegal access; B200 memcheck-convicted).
    // Zero both id maps so unwritten tail slots gather/scatter row 0 instead:
    // those lanes' output is never consumed (the mmq write-back loop is
    // expert_bounds-bounded), the cost is a few KB of memset on-stream.
    cudaMemsetAsync(ids_src1.get(), 0, ne_get_rows * sizeof(int32_t), stream);
    cudaMemsetAsync(ids_dst.get(),  0, ne_get_rows * sizeof(int32_t), stream);

    // si1 = stride between tokens in the ids tensor, in elements. Our ids is
    // contiguous [n_tokens, n_expert_used] so si1 = n_expert_used.
    // sis1 = stride between src1 channels in row-units. With ne11=1, sis1=1
    //        means each "channel" of src1 is one row of K floats.
    const int si1  = n_expert_used;
    const int sis1 = 1;

    // The smem mm_ids_helper uses n_tokens * 4 bytes of dynamic shared memory;
    // the down matmul reaches here with n_tokens = assignments (6x the forward
    // width), so 8192-row prefill chunks pass 48384 "tokens" > cap.  P5: past
    // the cap the launcher dispatches the bit-identical two-pass global
    // variant instead (mmid.cu mm_ids_helper_global) — refusing here used to
    // throw the WHOLE MoE block (including gate/up mmq work) onto the legacy
    // expert-tile fallback, the W8192 prefill cliff.  DS4_MMID_LARGE=0
    // restores the refusal.
    if ((size_t)n_tokens * 4u > ggml_cuda_info().devices[dev].smpbo && !ds4_mmid_large_enabled()) {
        fprintf(stderr, "%s: n_tokens=%d exceeds mm_ids_helper shared-mem cap; falling back\n",
                tag, n_tokens);
        return -1;
    }

    ggml_cuda_launch_mm_ids_helper(
        ids, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
        n_experts, n_tokens, n_expert_used, /*nchannels_y=*/(int)ne11, si1, sis1, stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: mm_ids_helper failed: %s\n", tag, cudaGetErrorString(err));
        return -2;
    }

    // 2. Gather + quantize the activation into Q8_1.
    const size_t nbytes_src1_q8_1 =
        ne_get_rows * ne10_padded * sizeof(block_q8_1) / QK8_1 +
        get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx->pool(), nbytes_src1_q8_1);

    // S1.1a fix (same as the dense path): the mmq Y buffer is over-allocated for the
    // kernel's tail-tile reads and ne_get_rows columns need not fill the final mmq
    // column tile, but quantize only writes the valid columns.  The mmq kernel
    // (mmq.cuh:3528) unconditionally loads the full tile, reading the never-written
    // tail from stale pool memory -> allocator-perturbation-dependent garbage in the
    // (write_back-masked) tail lanes -> non-deterministic batched-forward output.
    // Zero it so the masked-out tail is a deterministic zero.
    cudaMemsetAsync(src1_q8_1.get(), 0, nbytes_src1_q8_1, stream);

    // src1 logical [K, ne11=1, ne12=n_tokens, ne13=1] - K innermost, then
    // one row per channel, channels = tokens.
    const int64_t s11_src = (int64_t)K;                                 // stride between rows of a channel
    const int64_t s12_src = (int64_t)K * ne11;                          // stride between channels = K*1
    const int64_t s13_src = (int64_t)K * ne11 * ne12;                   // stride between samples

    quantize_mmq_q8_1_cuda(
        X_f32, ids_src1.get(), (void *)src1_q8_1.get(),
        type, /*ne00=*/K, s11_src, s12_src, s13_src,
        /*ne0=*/ne10_padded, /*ne1=*/ne_get_rows, /*ne2=*/1, /*ne3=*/1,
        stream);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_mmq_q8_1_cuda failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }

    // 3. Build mmq_args for the MoE path.
    //
    // dst layout convention matches upstream's MoE branch
    // (mmq.cu:215-220): dst is interpreted as [M, n_expert_used, n_tokens]
    // with M innermost and n_expert_used as the second dim that mmq writes
    // through ids_dst.  s1 = M (the column stride in the flat dst buffer
    // mmq writes into).  The output is column-major: out[col*M + row].
    const int64_t s1            = (int64_t)M;
    // stride_channel_y per upstream: ne11 * ne10_padded * sizeof(block_q8_1)
    //                                     / (QK8_1 * sizeof(int))
    // In MoE mode the kernel zeroes out the channel-stride contribution to
    // offset_y after reading expert_bounds, so the value is permissive -
    // but we set it consistently with upstream.
    const int64_t s12_mmq = ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13_mmq = ne12 * s12_mmq;

    const bool use_stream_k =
        (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA) ||
        GGML_CUDA_CC_IS_CDNA(cc);

    if (out_memset_enabled()) {
        cudaMemsetAsync(out_f32, 0, (size_t)M * (size_t)ne_get_rows * sizeof(float), stream);
    }

    if (type == GGML_TYPE_Q2_K && x_soa != nullptr && d2r_enabled() &&
        K % 256 == 0 && M % 2 == 0 && ne_get_rows >= d2r_min_cols()) {
        static int d2r_avail_cc = -1;
        static int d2r_avail = 0;
        if (d2r_avail_cc != cc) {
            d2r_avail_cc = cc;
            d2r_avail = ds4_mmq_q2_K_moe_d2r_available(cc) ? 1 : 0;
        }
        if (d2r_avail) {
            const size_t d2r_work_bytes =
                ds4_mmq_q2_K_moe_d2r_scratch_bytes(ne_get_rows, n_experts);
            if (d2r_work_bytes != 0) {
                ggml_cuda_pool_alloc<char> d2r_work(ctx->pool(), d2r_work_bytes);
                const int d2r_rc = ds4_mmq_q2_K_moe_d2r_launch(
                    x_soa, soa_blocks, src1_q8_1.get(), ids_dst.get(), expert_bounds.get(),
                    out_f32, M, K, ne_get_rows, n_experts, d2r_work.get(), d2r_work_bytes,
                    stream);
                if (d2r_rc == 0) {
                    return 0;
                }
            }
        }
    }

    const mmq_args args = {
        /*x=*/(const char *)W,
        /*type_x=*/type,
        /*y=*/(const int *)src1_q8_1.get(),
        /*ids_dst=*/ids_dst.get(),
        /*expert_bounds=*/expert_bounds.get(),
        /*dst=*/out_f32,
        /*ncols_x=*/ne00,
        /*nrows_x=*/(int64_t)M,
        /*ncols_dst=*/ne_get_rows,
        /*stride_row_x=*/s01,
        /*ncols_y=*/ne_get_rows,
        /*nrows_dst=*/s1,
        /*nchannels_x=*/(int64_t)n_experts,
        /*nchannels_y=*/(int64_t)n_experts,
        /*stride_channel_x=*/s02,
        /*stride_channel_y=*/s12_mmq,
        /*stride_channel_dst=*/(int64_t)0,
        /*nsamples_x=*/1,
        /*nsamples_y=*/1,
        /*stride_sample_x=*/0,
        /*stride_sample_y=*/s13_mmq,
        /*stride_sample_dst=*/0,
        /*use_stream_k=*/use_stream_k,
        /*ncols_max=*/ne_get_rows,
        /*x_soa=*/x_soa,
        /*soa_blocks=*/soa_blocks,
    };

    mul_mat_q_case<type>(*ctx, args, stream);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: mul_mat_q_case (moe) launch failed: %s\n", tag, cudaGetErrorString(err));
        return -4;
    }
    if (sanitize_out) {
        ds4_mmq_sanitize_f32(out_f32, (uint64_t)M * (uint64_t)ne_get_rows, stream);
    }
    return 0;
}

struct ds4_mmq_fused_down {
    const void  * W;
    const float * router_weights;
    float       * mid_f32;
    float       * out;
    int           out_dim;
    float         clamp;
};

// Produce the weighted SwiGLU rows in their canonical pair-major order. The
// proven upstream quantizer below gathers them through the already available
// ids_dst map, so gate/up and down share one expert-major schedule without a
// second mm_ids_helper.
static __global__ void ds4_swiglu_weighted_f32(
        const float * __restrict__ gate,
        const float * __restrict__ up,
        const float * __restrict__ router_weights,
        float * __restrict__ mid,
        uint64_t n,
        int K,
        float clamp) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint64_t pair = i / (uint64_t)K;
    float g = isfinite(gate[i]) ? gate[i] : 0.0f;
    float u = isfinite(up[i]) ? up[i] : 0.0f;
    if (clamp > 1.0e-6f) {
        g = fminf(g, clamp);
        u = fminf(fmaxf(u, -clamp), clamp);
    }
    mid[i] = (g / (1.0f + expf(-g))) * u * router_weights[pair];
}

// Paired MoE: one helper + one quantize covers both weights.  See the
// header comment on ds4_mmq_iq2_xxs_moe_pair for motivation.  Internal
// structure mirrors ds4_mmq_moe_impl above; the only differences are the
// two W pointers, the two output pointers, and the second mul_mat_q_case
// launch with a fresh (x, dst) pair.
template <ggml_type type, bool profile_fused_prefill = false>
int ds4_mmq_moe_pair_impl(
        const char    * tag,
        const void    * W_a,
        const void    * W_b,
        const float   * X_f32,
        const int32_t * ids,
        float         * out_a,
        float         * out_b,
        int             M,
        int             K,
        int             n_tokens,
        int             n_experts,
        int             n_expert_used,
        cudaStream_t    stream,
        /* ds4 (P4 Inc3): optional aligned-SoA artifacts for W_a / W_b (same
         * shape, so one block count); see ds4_mmq_moe_impl. */
        const char    * xa_soa     = NULL,
        const char    * xb_soa     = NULL,
        int64_t         soa_blocks = 0,
        /* ds4 (P3): see ds4_mmq_moe_impl. */
        bool            sanitize_out = true,
        const ds4_mmq_fused_down *fused_down = nullptr) {

    if (!W_a || !W_b || !X_f32 || !ids || !out_a || !out_b) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || K <= 0 || n_tokens <= 0 || n_experts <= 0 || n_expert_used <= 0) {
        fprintf(stderr, "%s: bad shape M=%d K=%d ntok=%d nexp=%d nused=%d\n",
                tag, M, K, n_tokens, n_experts, n_expert_used);
        return -1;
    }
    if (K % 256 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }
    if (n_expert_used > n_experts) {
        fprintf(stderr, "%s: n_expert_used=%d > n_experts=%d\n", tag, n_expert_used, n_experts);
        return -1;
    }
    if (fused_down &&
        (type != GGML_TYPE_IQ2_XXS || !fused_down->W ||
         !fused_down->router_weights || !fused_down->mid_f32 ||
         !fused_down->out || fused_down->out_dim <= 0 || M % 256 != 0)) {
        fprintf(stderr, "%s: invalid fused Q2_K down configuration\n", tag);
        return -1;
    }

    const bool nvtx_prefill = profile_fused_prefill &&
                              fused_down != nullptr &&
                              n_tokens >= 1024 &&
                              ds4_mmq_nvtx_requested();
    ds4_mmq_nvtx_scope fused_scope(
            "ds4/prefill/moe/mmq_fused",
            ds4_mmq_nvtx_payload((uint32_t)n_tokens, (uint32_t)n_expert_used),
            nvtx_prefill);

    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;

    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }

    ds4_pool_set_stream(stream);  /* task #22: pool ops must be stream-ordered with the kernels (see ds4_mmq_dense_impl) */

    const int64_t ne_get_rows  = (int64_t)n_tokens * n_expert_used;
    const int64_t ne00         = K;
    const int64_t ne10_padded  = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const int64_t ne11         = 1;
    const int64_t ne12         = n_tokens;
    const int64_t blck         = ggml_blck_size(type);
    const int64_t s01          = (int64_t)K / blck;
    const int64_t s02          = (int64_t)M * s01;

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx->pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx->pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx->pool(), n_experts + 1);

    const int si1  = n_expert_used;
    const int sis1 = 1;

    // Same cap guard as ds4_mmq_moe_impl (see comment there): past the smem
    // cap the launcher takes the bit-identical global variant (P5); only
    // refuse with DS4_MMID_LARGE=0.
    if ((size_t)n_tokens * 4u > ggml_cuda_info().devices[dev].smpbo && !ds4_mmid_large_enabled()) {
        fprintf(stderr, "%s: n_tokens=%d exceeds mm_ids_helper shared-mem cap; falling back\n",
                tag, n_tokens);
        return -1;
    }

    cudaError_t err = cudaSuccess;
    {
        ds4_mmq_nvtx_scope stage(
                "ds4/prefill/moe/expert_map",
                ds4_mmq_nvtx_payload((uint32_t)n_tokens, (uint32_t)n_experts),
                nvtx_prefill);
        // Task #22 root-cause fix (same as ds4_mmq_moe_impl): zero the id maps
        // so entries dropped by mm_ids_helper never expose stale pool memory.
        cudaMemsetAsync(ids_src1.get(), 0, ne_get_rows * sizeof(int32_t), stream);
        cudaMemsetAsync(ids_dst.get(),  0, ne_get_rows * sizeof(int32_t), stream);
        ggml_cuda_launch_mm_ids_helper(
            ids, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            n_experts, n_tokens, n_expert_used, /*nchannels_y=*/(int)ne11,
            si1, sis1, stream);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: mm_ids_helper failed: %s\n", tag, cudaGetErrorString(err));
            return -2;
        }
    }

    const bool use_stream_k =
        (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA) ||
        GGML_CUDA_CC_IS_CDNA(cc);
    /* The fused target-prefill path receives a true top-k assignment: one
     * token cannot select the same expert twice, so no expert bucket can
     * exceed n_tokens rows. Keep the conservative gathered-row bound for all
     * generic MMQ callers, including DSpark/MTP. */
    const int64_t routed_ncols_max = fused_down
        ? (int64_t)n_tokens
        : ne_get_rows;

    /* The gate/up Q8_1 can be stream-freed before allocating the down Q8_1;
     * ids_dst and expert_bounds remain alive outside this scope. */
    {
    const size_t nbytes_src1_q8_1 =
        ne_get_rows * ne10_padded * sizeof(block_q8_1) / QK8_1 +
        get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx->pool(), nbytes_src1_q8_1);

    // S1.1a fix (same as the dense/moe paths): zero the over-allocated mmq Y buffer
    // so the kernel's unconditional masked-out tail-tile read (mmq.cuh:3528) returns
    // a deterministic zero instead of allocator-perturbation-dependent stale memory.
    const int64_t s11_src = (int64_t)K;
    const int64_t s12_src = (int64_t)K * ne11;
    const int64_t s13_src = (int64_t)K * ne11 * ne12;
    {
        ds4_mmq_nvtx_scope stage(
                "ds4/prefill/moe/input_quant_q8_1",
                ds4_mmq_nvtx_payload((uint32_t)ne_get_rows, (uint32_t)K),
                nvtx_prefill);
        cudaMemsetAsync(src1_q8_1.get(), 0, nbytes_src1_q8_1, stream);
        quantize_mmq_q8_1_cuda(
            X_f32, ids_src1.get(), (void *)src1_q8_1.get(),
            type, /*ne00=*/K, s11_src, s12_src, s13_src,
            /*ne0=*/ne10_padded, /*ne1=*/ne_get_rows, /*ne2=*/1, /*ne3=*/1,
            stream);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: quantize_mmq_q8_1_cuda failed: %s\n", tag, cudaGetErrorString(err));
            return -3;
        }
    }

    const int64_t s1      = (int64_t)M;
    const int64_t s12_mmq = ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13_mmq = ne12 * s12_mmq;

    if (out_memset_enabled()) {
        cudaMemsetAsync(out_a, 0, (size_t)M * (size_t)ne_get_rows * sizeof(float), stream);
        cudaMemsetAsync(out_b, 0, (size_t)M * (size_t)ne_get_rows * sizeof(float), stream);
    }

    if (type == GGML_TYPE_IQ2_XXS && xa_soa != nullptr && xb_soa != nullptr &&
        d2r_enabled() && d2r_iq2_enabled() && K % 256 == 0 &&
        ne_get_rows >= d2r_min_cols()) {
        static int d2r_iq2_avail_cc = -1;
        static int d2r_iq2_avail = 0;
        if (d2r_iq2_avail_cc != cc) {
            d2r_iq2_avail_cc = cc;
            d2r_iq2_avail = ds4_mmq_iq2_xxs_moe_d2r_available(cc) ? 1 : 0;
        }
        if (d2r_iq2_avail) {
            const size_t d2r_work_bytes =
                ds4_mmq_iq2_xxs_moe_d2r_pair_scratch_bytes(ne_get_rows, n_experts);
            if (d2r_work_bytes != 0) {
                ggml_cuda_pool_alloc<char> d2r_work(ctx->pool(), d2r_work_bytes);
                ds4_mmq_nvtx_scope stage(
                        "ds4/prefill/moe/iq2_gate_up_d2r",
                        ds4_mmq_nvtx_payload((uint32_t)ne_get_rows, (uint32_t)M),
                        nvtx_prefill);
                const int d2r_rc = ds4_mmq_iq2_xxs_moe_d2r_pair_launch(
                        xa_soa, xb_soa, soa_blocks, src1_q8_1.get(), ids_dst.get(),
                        expert_bounds.get(), out_a, out_b, M, K, ne_get_rows, n_experts,
                        d2r_work.get(), d2r_work_bytes, stream);
                if (d2r_rc == 0) {
                    return 0;
                }
            }
        }
    }

    mmq_args args = {
        /*x=*/(const char *)W_a,
        /*type_x=*/type,
        /*y=*/(const int *)src1_q8_1.get(),
        /*ids_dst=*/ids_dst.get(),
        /*expert_bounds=*/expert_bounds.get(),
        /*dst=*/out_a,
        /*ncols_x=*/ne00,
        /*nrows_x=*/(int64_t)M,
        /*ncols_dst=*/ne_get_rows,
        /*stride_row_x=*/s01,
        /*ncols_y=*/ne_get_rows,
        /*nrows_dst=*/s1,
        /*nchannels_x=*/(int64_t)n_experts,
        /*nchannels_y=*/(int64_t)n_experts,
        /*stride_channel_x=*/s02,
        /*stride_channel_y=*/s12_mmq,
        /*stride_channel_dst=*/(int64_t)0,
        /*nsamples_x=*/1,
        /*nsamples_y=*/1,
        /*stride_sample_x=*/0,
        /*stride_sample_y=*/s13_mmq,
        /*stride_sample_dst=*/0,
        /*use_stream_k=*/use_stream_k,
        /*ncols_max=*/routed_ncols_max,
        /*x_soa=*/xa_soa,
        /*soa_blocks=*/soa_blocks,
    };

    {
        ds4_mmq_nvtx_scope stage(
                "ds4/prefill/moe/iq2_gate",
                ds4_mmq_nvtx_payload((uint32_t)ne_get_rows, (uint32_t)M),
                nvtx_prefill);
        mul_mat_q_case<type>(*ctx, args, stream);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: mul_mat_q_case (pair a) launch failed: %s\n", tag, cudaGetErrorString(err));
            return -4;
        }
    }

    // Second matmul over the same activation buffer and same routing map.
    args.x     = (const char *)W_b;
    args.dst   = out_b;
    args.x_soa = xb_soa;
    {
        ds4_mmq_nvtx_scope stage(
                "ds4/prefill/moe/iq2_up",
                ds4_mmq_nvtx_payload((uint32_t)ne_get_rows, (uint32_t)M),
                nvtx_prefill);
        mul_mat_q_case<type>(*ctx, args, stream);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: mul_mat_q_case (pair b) launch failed: %s\n", tag, cudaGetErrorString(err));
            return -5;
        }
    }
    }

    if (fused_down) {
        const int64_t down_ne10_padded = GGML_PAD((int64_t)M, MATRIX_ROW_PADDING);
        const size_t logical_q8_bytes =
            (size_t)ne_get_rows * (size_t)down_ne10_padded * sizeof(block_q8_1) / QK8_1;
        const size_t tail_q8_bytes =
            (size_t)get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
        ggml_cuda_pool_alloc<char> down_q8_1(
            ctx->pool(), logical_q8_bytes + tail_q8_bytes);

        const uint64_t mid_values = (uint64_t)ne_get_rows * (uint64_t)M;
        {
            ds4_mmq_nvtx_scope stage(
                    "ds4/prefill/moe/swiglu_down_quant",
                    ds4_mmq_nvtx_payload((uint32_t)ne_get_rows, (uint32_t)M),
                    nvtx_prefill);
            cudaMemsetAsync(down_q8_1.get(), 0, logical_q8_bytes + tail_q8_bytes, stream);
            ds4_swiglu_weighted_f32<<<
                (uint32_t)((mid_values + 255u) / 256u), 256, 0, stream>>>(
                    out_a, out_b, fused_down->router_weights,
                    fused_down->mid_f32, mid_values, M, fused_down->clamp);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "%s: weighted SwiGLU launch failed: %s\n",
                        tag, cudaGetErrorString(err));
                return -6;
            }

            quantize_mmq_q8_1_cuda(
                fused_down->mid_f32, ids_dst.get(), (void *)down_q8_1.get(),
                GGML_TYPE_Q2_K, /*ne00=*/M, /*s01=*/M,
                /*s02=*/(int64_t)M, /*s03=*/(int64_t)M * ne_get_rows,
                /*ne0=*/down_ne10_padded, /*ne1=*/ne_get_rows,
                /*ne2=*/1, /*ne3=*/1, stream);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "%s: down quantize_mmq_q8_1_cuda failed: %s\n",
                        tag, cudaGetErrorString(err));
                return -7;
            }
        }

        if (out_memset_enabled()) {
            cudaMemsetAsync(fused_down->out, 0,
                    (size_t)fused_down->out_dim * (size_t)ne_get_rows * sizeof(float),
                    stream);
        }
        const int64_t down_s01 = (int64_t)M / ggml_blck_size(GGML_TYPE_Q2_K);
        const int64_t down_s02 = (int64_t)fused_down->out_dim * down_s01;
        const int64_t down_s12 =
            down_ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const mmq_args down_args = {
            /*x=*/(const char *)fused_down->W,
            /*type_x=*/GGML_TYPE_Q2_K,
            /*y=*/(const int *)down_q8_1.get(),
            /*ids_dst=*/ids_dst.get(),
            /*expert_bounds=*/expert_bounds.get(),
            /*dst=*/fused_down->out,
            /*ncols_x=*/(int64_t)M,
            /*nrows_x=*/(int64_t)fused_down->out_dim,
            /*ncols_dst=*/ne_get_rows,
            /*stride_row_x=*/down_s01,
            /*ncols_y=*/ne_get_rows,
            /*nrows_dst=*/(int64_t)fused_down->out_dim,
            /*nchannels_x=*/(int64_t)n_experts,
            /*nchannels_y=*/(int64_t)n_experts,
            /*stride_channel_x=*/down_s02,
            /*stride_channel_y=*/down_s12,
            /*stride_channel_dst=*/(int64_t)0,
            /*nsamples_x=*/1,
            /*nsamples_y=*/1,
            /*stride_sample_x=*/0,
            /*stride_sample_y=*/ne_get_rows * down_s12,
            /*stride_sample_dst=*/0,
            /*use_stream_k=*/use_stream_k,
            /*ncols_max=*/routed_ncols_max,
            /*x_soa=*/nullptr,
            /*soa_blocks=*/0,
        };
        {
            ds4_mmq_nvtx_scope stage(
                    "ds4/prefill/moe/q2_down",
                    ds4_mmq_nvtx_payload((uint32_t)ne_get_rows,
                                         (uint32_t)fused_down->out_dim),
                    nvtx_prefill);
            mul_mat_q_case<GGML_TYPE_Q2_K>(*ctx, down_args, stream);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "%s: fused Q2_K down launch failed: %s\n",
                        tag, cudaGetErrorString(err));
                return -8;
            }
        }
    }
    if (sanitize_out) {
        ds4_mmq_sanitize_f32(out_a, (uint64_t)M * (uint64_t)ne_get_rows, stream);
        ds4_mmq_sanitize_f32(out_b, (uint64_t)M * (uint64_t)ne_get_rows, stream);
    }
    return 0;
}

} // anonymous namespace

extern "C" int ds4_mmq_q8_0_moe(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_impl<GGML_TYPE_Q8_0>("ds4_mmq_q8_0_moe", W, X, ids, out, M, K,
                                            n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_q2_K_moe(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_impl<GGML_TYPE_Q2_K>("ds4_mmq_q2_K_moe", W, X, ids, out, M, K,
                                            n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_q2_K_moe_consumer_sanitizes(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_impl<GGML_TYPE_Q2_K>(
        "ds4_mmq_q2_K_moe_consumer_sanitizes", W, X, ids, out, M, K,
        n_tokens, n_experts, n_expert_used, stream,
        /*x_soa=*/NULL, /*soa_blocks=*/0, /*sanitize_out=*/false);
}

extern "C" int ds4_mmq_iq2_xxs_moe(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_impl<GGML_TYPE_IQ2_XXS>("ds4_mmq_iq2_xxs_moe", W, X, ids, out, M, K,
                                               n_tokens, n_experts, n_expert_used, stream);
}

/* ds4 (P4 Inc3): mmq MoE over the aligned row-pair-SoA Q2_K artifact
 * (weight server --repack-q2k-aligned) -- no raw-layout weights and no
 * derepack scratch involved; the mul_mat_q tile loader reads the SoA
 * sections directly (load_tiles_q2_K_soa, bit-identical tiles). */
extern "C" int ds4_mmq_q2_K_moe_soa(
        const void * W_soa, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    if (M <= 0 || M % 2 != 0 || K <= 0 || K % 256 != 0 || n_experts <= 0) {
        fprintf(stderr, "ds4_mmq_q2_K_moe_soa: bad shape M=%d K=%d nexp=%d\n", M, K, n_experts);
        return -1;
    }
    const int64_t npair = (int64_t)n_experts * (int64_t)(M/2) * (int64_t)(K/256);
    /* W_soa doubles as the (unused) raw pointer so the impl's null checks
     * hold.  sanitize_out=false: the routed-MoE consumers (swiglu / moe_sum)
     * sanitize at read, saving the whole-buffer pass (P3). */
    return ds4_mmq_moe_impl<GGML_TYPE_Q2_K>("ds4_mmq_q2_K_moe_soa", W_soa, X, ids, out, M, K,
                                            n_tokens, n_experts, n_expert_used, stream,
                                            (const char *)W_soa, npair,
                                            /*sanitize_out=*/false);
}

extern "C" int ds4_mmq_q4_K_moe(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_impl<GGML_TYPE_Q4_K>("ds4_mmq_q4_K_moe", W, X, ids, out, M, K,
                                            n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_iq2_xxs_moe_pair(
        const void * W_a, const void * W_b,
        const float * X, const int32_t * ids, float * out_a, float * out_b,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_pair_impl<GGML_TYPE_IQ2_XXS>(
        "ds4_mmq_iq2_xxs_moe_pair", W_a, W_b, X, ids, out_a, out_b,
        M, K, n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_iq2_xxs_moe_pair_consumer_sanitizes(
        const void * W_a, const void * W_b,
        const float * X, const int32_t * ids, float * out_a, float * out_b,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_pair_impl<GGML_TYPE_IQ2_XXS>(
        "ds4_mmq_iq2_xxs_moe_pair_consumer_sanitizes",
        W_a, W_b, X, ids, out_a, out_b,
        M, K, n_tokens, n_experts, n_expert_used, stream,
        /*xa_soa=*/NULL, /*xb_soa=*/NULL, /*soa_blocks=*/0,
        /*sanitize_out=*/false);
}

extern "C" int ds4_mmq_iq2_xxs_q2_K_moe_fused(
        const void * W_gate, const void * W_up, const void * W_down,
        const float * X, const int32_t * ids, const float * router_weights,
        float * gate, float * up, float * mid_f32, float * down,
        int expert_mid_dim, int expert_in_dim, int out_dim,
        int n_tokens, int n_experts, int n_expert_used,
        float clamp, cudaStream_t stream) {
    const ds4_mmq_fused_down fused_down = {
        W_down,
        router_weights,
        mid_f32,
        down,
        out_dim,
        clamp,
    };
    return ds4_mmq_moe_pair_impl<GGML_TYPE_IQ2_XXS, true>(
        "ds4_mmq_iq2_xxs_q2_K_moe_fused",
        W_gate, W_up, X, ids, gate, up,
        expert_mid_dim, expert_in_dim, n_tokens, n_experts, n_expert_used,
        stream, /*xa_soa=*/NULL, /*xb_soa=*/NULL, /*soa_blocks=*/0,
        /*sanitize_out=*/false, &fused_down);
}

/* ds4 (P4 Inc3): paired mmq MoE over the aligned-SoA IQ2_XXS gate/up
 * artifacts (weight server --repack-iq2-aligned); same contract as
 * ds4_mmq_q2_K_moe_soa. */
extern "C" int ds4_mmq_iq2_xxs_moe_pair_soa(
        const void * Wa_soa, const void * Wb_soa,
        const float * X, const int32_t * ids, float * out_a, float * out_b,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    if (M <= 0 || K <= 0 || K % 256 != 0 || n_experts <= 0) {
        fprintf(stderr, "ds4_mmq_iq2_xxs_moe_pair_soa: bad shape M=%d K=%d nexp=%d\n", M, K, n_experts);
        return -1;
    }
    const int64_t nblk = (int64_t)n_experts * (int64_t)M * (int64_t)(K/256);
    /* sanitize_out=false: see ds4_mmq_q2_K_moe_soa. */
    return ds4_mmq_moe_pair_impl<GGML_TYPE_IQ2_XXS>(
        "ds4_mmq_iq2_xxs_moe_pair_soa", Wa_soa, Wb_soa, X, ids, out_a, out_b,
        M, K, n_tokens, n_experts, n_expert_used, stream,
        (const char *)Wa_soa, (const char *)Wb_soa, nblk,
        /*sanitize_out=*/false);
}

extern "C" int ds4_mmq_q4_K_moe_pair(
        const void * W_a, const void * W_b,
        const float * X, const int32_t * ids, float * out_a, float * out_b,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_pair_impl<GGML_TYPE_Q4_K>(
        "ds4_mmq_q4_K_moe_pair", W_a, W_b, X, ids, out_a, out_b,
        M, K, n_tokens, n_experts, n_expert_used, stream);
}

// ----------------------------------------------------------------------------
// mmvq-backed entry points (Step 6 of the optimization plan).
//
// mmvq is upstream's matrix-vector matmul family, optimised for the
// n_tokens <= MMVQ_MAX_BATCH_SIZE=8 regime. Unlike mmq it consumes the
// CANONICAL block_q8_1 layout (via quantize_row_q8_1_cuda), not the
// interleaved block_q8_1_mmq that quantize_mmq_q8_1_cuda produces.
//
// The single-W _moe_vec entries cover:
//   - the down matmul at decode (treating [n_tokens=1, n_expert_used=6]
//     as [n_tokens=6, n_expert_used=1])
//   - dense attention projections at decode (n_tokens=1, no MoE)
//   - any small-batch path where mmvq's per-token grid wins over mmq's
//     tile-based approach
//
// The pair-fused _moe_pair_vec entries cover the gate+up matmuls at
// decode using mmvq's built-in fusion. fusion.gate is the up_w pointer
// and fusion.glu_op is GGML_GLU_OP_SWIGLU - the kernel computes
// silu(gate@x) * (up@x) in a single launch. mmvq's fusion is supported
// only at ncols_dst=1, so n_tokens=1 is the only valid case.
// ----------------------------------------------------------------------------

#include "mmvq.cuh"

namespace {

template <ggml_type type>
int ds4_mmq_moe_vec_impl(
        const char    * tag,
        const void    * W,
        const float   * X_f32,
        const int32_t * ids,
        float         * out_f32,
        int             M,
        int             K,
        int             n_tokens,
        int             n_experts,
        int             n_expert_used,
        cudaStream_t    stream) {

    if (!W || !X_f32 || !ids || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || K <= 0 || n_tokens <= 0 || n_experts <= 0 || n_expert_used <= 0) {
        fprintf(stderr, "%s: bad shape M=%d K=%d ntok=%d nexp=%d nused=%d\n",
                tag, M, K, n_tokens, n_experts, n_expert_used);
        return -1;
    }
    if (K % 256 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }
    if (n_expert_used > n_experts) {
        fprintf(stderr, "%s: n_expert_used=%d > n_experts=%d\n", tag, n_expert_used, n_experts);
        return -1;
    }
    // mmvq's per-arch batch cap. ncols_dst as computed below is
    // max(n_tokens, n_expert_used) depending on which dim we route into.
    // We follow upstream's convention: ne_y = n_tokens, ne_dst = n_expert_used.
    // So ncols_dst = n_tokens and nchannels_dst = n_expert_used.
    // FD Inc2a: n_tokens beyond the per-launch column cap no longer rejects;
    // the launch loop below splits the column dim into capped chunks.

    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }

    // Route the pool's cudaMallocAsync / cudaFreeAsync through the same
    // stream the caller uses for kernel launches.  Required for Step 8
    // (CUDA Graph capture): pool allocations on a different stream than
    // the capture stream would invalidate the capture.
    ds4_pool_set_stream(stream);

    // 1. Quantize X into CANONICAL Q8_1 (NOT the MMQ-interleaved variant).
    //    Layout: [ne13=1, ne12=n_tokens, ne11=1, ne10_padded blocks].
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t  nbytes_q8_1 = (size_t)n_tokens * ne10_padded *
                                sizeof(block_q8_1) / QK8_1;
    // Step 7 task #29: experimental persistent Q8_1 scratch.  Avoids
    // pool_alloc (cudaMallocAsync) graph nodes whose pointer baked at
    // capture time may not match the address resolved at replay.  When
    // disabled (default) or when the persistent buffer is too small,
    // fall back to the pool path.  See ds4_mmq_init for setup.
    ggml_cuda_pool_alloc<char> src1_q8_1_pool;
    char *src1_q8_1_ptr = nullptr;
    if (g_q81_scratch_enabled && g_q81_scratch_ptr && g_q81_scratch_bytes >= nbytes_q8_1) {
        src1_q8_1_ptr = (char *)g_q81_scratch_ptr;
    } else {
        src1_q8_1_pool.alloc(ctx->pool(), nbytes_q8_1);
        src1_q8_1_ptr = src1_q8_1_pool.get();
    }

    // s11 = stride between rows of an src1 channel in source-float units.
    //       Logical src1 [K, ne11=1, ne12=n_tokens, ne13=1] - K innermost.
    // s12 = stride between channels = K * ne11 = K.
    // s13 = stride between samples = K * ne11 * ne12 = K * n_tokens.
    quantize_row_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1_ptr,
        type, /*ne00=*/K,
        /*s11=*/(int64_t)K, /*s12=*/(int64_t)K, /*s13=*/(int64_t)K * n_tokens,
        /*ne0=*/ne10_padded, /*ne1=*/1, /*ne2=*/n_tokens, /*ne3=*/1,
        stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_row_q8_1_cuda failed: %s\n",
                tag, cudaGetErrorString(err));
        return -2;
    }

    // 2. mmvq stride setup. Mirror upstream's ggml_cuda_mul_mat_vec_q
    //    dispatch (mmvq.cu:1101-1136).
    //
    //    For MoE (ids != nullptr): per the dispatch math at line 1121-1130,
    //      ncols_dst          = ne2  = n_tokens
    //      nchannels_y        = ne11 = 1
    //      nchannels_dst      = ne1  = n_expert_used
    //      stride_col_y       = s12  = ne11 * (ne10_padded / QK8_1)
    //      stride_col_dst     = s2   = n_expert_used * M (token stride in dst)
    //      stride_channel_y   = s11  = ne10_padded / QK8_1
    //      stride_channel_dst = s1   = M (channel/slot stride in dst)
    //      ids_stride         = stride between rows of ids[] tensor
    //
    //    FD Inc2a stride fix: stride_col_dst was previously M, same as the
    //    channel stride.  That was invisible while every caller degenerated
    //    one dim (gate/up at n_tokens=1: col index always 0; down at
    //    n_expert_used=1: channel index always 0, and 1 * M == M keeps it
    //    bit-identical here).  At n_tokens >= 2 with n_expert_used > 1 the
    //    multi-token MoE kernel writes dst[chan*s1 + col*s2 + row], and
    //    equal strides collide (token=0,slot=1) with (token=1,slot=0).
    //    s2 = n_expert_used * M yields the row-major
    //    [token * n_expert_used + slot, M] layout the swiglu consumer
    //    expects.
    const int64_t blck      = ggml_blck_size(type);
    const int64_t s01_row   = (int64_t)K / blck;            // weight row stride in blocks
    const int64_t s02_chan  = (int64_t)M * s01_row;         // expert-stack stride
    const int64_t s11_y     = ne10_padded / QK8_1;          // src1 channel stride in blocks
    const int64_t s12_y     = (int64_t)1 * s11_y;           // ne11 * s11
    const int64_t s1_dst    = (int64_t)M;                   // dst channel (slot) stride
    const int64_t s2_dst    = (int64_t)n_expert_used * M;   // dst col (token) stride

    // ids_stride: stride between rows of the ids tensor in int32 elements.
    // Caller passes ids[t * n_expert_used + s], so stride between tokens
    // is n_expert_used.
    const int ids_stride = n_expert_used;

    ggml_cuda_mm_fusion_args_device fusion = {};

    cudaMemsetAsync(out_f32, 0, (size_t)M * (size_t)n_tokens * (size_t)n_expert_used * sizeof(float), stream);

    // FD Inc2a: one mmvq launch serves at most col_cap columns -- the moe
    // kernel runs one warp per column (block.y = ncols_dst) under
    // __launch_bounds__ baked per COMPILED arch + type
    // (get_mmvq_mmid_max_batch_for_device).  The runtime device cc can
    // exceed the compiled arch (CUDA_ARCH= builds run default-arch PTX on
    // newer GPUs), so the host cap MUST be looked up at the compiled arch:
    // asking the runtime cc says 8 where the compiled bounds say 7 (e.g.
    // Q2_K builds at turing_plus -> 7*warp_size threads) and the launch
    // dies with cudaErrorInvalidValue.  Wider batches run as
    // ceil(n_tokens / col_cap) launches; every per-column stride (vy, ids,
    // dst) is uniform, so a chunk is plain pointer offsets.  The single
    // quantize above already covers all columns.
    const int cc      = ggml_cuda_info().devices[dev].cc;
    const int col_cap = get_mmvq_mmid_max_batch(type, ggml_cuda_highest_compiled_arch(cc));

    for (int c0 = 0; c0 < n_tokens; c0 += col_cap) {
        const int ncols = (n_tokens - c0 < col_cap) ? (n_tokens - c0) : col_cap;
        mul_mat_vec_q_switch_type(
            /*vx=*/W, /*type_x=*/type,
            /*vy=*/(const void *)(src1_q8_1_ptr + (size_t)c0 * s12_y * sizeof(block_q8_1)),
            /*ids=*/ids + (size_t)c0 * ids_stride, /*fusion=*/fusion,
            /*dst=*/out_f32 + (int64_t)c0 * s2_dst,
            /*ncols_x=*/K, /*nrows_x=*/M, /*ncols_dst=*/ncols,
            /*stride_row_x=*/(int)s01_row,
            /*stride_col_y=*/(int)s12_y,
            /*stride_col_dst=*/(int)s2_dst,
            /*nchannels_x=*/n_experts,
            /*nchannels_y=*/1,
            /*nchannels_dst=*/n_expert_used,
            /*stride_channel_x=*/(int)s02_chan,
            /*stride_channel_y=*/(int)s11_y,
            /*stride_channel_dst=*/(int)s1_dst,
            /*nsamples_x=*/1, /*nsamples_dst=*/1,
            /*stride_sample_x=*/0, /*stride_sample_y=*/0, /*stride_sample_dst=*/0,
            /*ids_stride=*/ids_stride, stream);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: mul_mat_vec_q_switch_type launch failed: %s (cols %d..%d cap %d)\n",
                    tag, cudaGetErrorString(err), c0, c0 + ncols - 1, col_cap);
            return -3;
        }
    }

    ds4_mmq_sanitize_f32(out_f32, (uint64_t)M * (uint64_t)n_tokens * (uint64_t)n_expert_used, stream);
    return 0;
}

// ---------------------------------------------------------------------------
// Aligned-SoA IQ2_XXS decode matvec (megakernel program M1-Inc1).
//
// Layout contract (see ds4_mmq.h): W_aligned = [__half dq[nblk]][pad to 64B]
// [uint2 qs[nblk*8]], nblk = n_experts * M * (K/256), block linear order equal
// to the raw tensor byte order.  Per-pair integer math is bit-identical to
// vec_dot_iq2_xxs_q8_1 (vecdotq.cuh); only the float accumulation order
// differs (per-warp-row here vs per-mmvq-tile there).  Proven +12% over the
// raw-layout vec path at the production decode shape
// (cuda/mmq/test/proto_iq2_aligned.cu).
__global__ void iq2_xxs_aligned_moe_vec_kernel(
        float             *out,        // [n_tokens*n_expert_used, M]
        const uint2       *qs,         // 64B-aligned code pairs
        const __half      *dq,         // block scales
        const block_q8_1  *x8,         // [n_tokens][nyb] canonical Q8_1 activations
        const int32_t     *ids,        // [n_tokens*n_expert_used] expert ids
        int                M,
        int                nb,         // IQ2_XXS blocks per row = K/256
        int                nyb,        // Q8_1 blocks per activation row
        int                n_expert_used)
{
    const int row  = blockIdx.x;
    const int slot = blockIdx.y;       // flat assignment = token*n_expert_used+slot
    const int lane = threadIdx.x;      // 32 lanes: lane covers (block b, pair p)
    // The router's NaN path emits -1 expert ids by design (same guard as
    // mul_mat_vec_q_moe): clamp the pointer math to expert 0, skip the dot
    // loop, write a clean 0.
    const int32_t id_raw = ids[slot];
    const bool invalid_id = id_raw < 0;
    const long long rbase = ((long long)(invalid_id ? 0 : id_raw) * M + row) * nb;
    x8 += (long long)(slot / n_expert_used) * nyb;

    float acc = 0.0f;
    // 32 lanes cover 4 blocks x 8 pairs per pass.
    for (int b0 = 0; !invalid_id && b0 < nb; b0 += 4) {
        const int b = b0 + (lane >> 3);
        const int p = lane & 7;
        const uint2 cw   = qs[(rbase + b) * 8 + p];   // aligned 8B load
        const uint32_t q2 = cw.x, aux32 = cw.y;
        const uint8_t *aux8 = (const uint8_t *)&q2;

        int sumi = 0;
        const int q8i = (b * 256 + p * 32) / 32;   // q8_1 block covering these 32 values
        const int *u = (const int *)x8[q8i].qs;
#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            const uint2 grid_pos = ((const uint2 *)iq2xxs_grid)[aux8[k0 / 2]];
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));

            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int grid0  = __vsub4(grid_pos.x ^ signs0, signs0);
            sumi = ggml_cuda_dp4a(grid0, u[k0 + 0], sumi);

            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            const int grid1  = __vsub4(grid_pos.y ^ signs1, signs1);
            sumi = ggml_cuda_dp4a(grid1, u[k0 + 1], sumi);
        }
        const int ls = aux32 >> 27 | 1;
        sumi = sumi * ls / 8;
        const float d = __half2float(dq[rbase + b]) * __low2float(x8[q8i].ds);
        acc += d * (float)sumi;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) out[(long long)slot * M + row] = acc;
}

// M1-Inc2 variant P: one launch covers gate and up (blockIdx.z selects the
// weight stream); nonfinite accs are zeroed in-kernel so no sanitize pass is
// needed.  Same per-warp math as iq2_xxs_aligned_moe_vec_kernel.
__global__ void iq2_xxs_aligned_moe_pair_vec_kernel(
        float             *out_gate,   // [n_tokens*n_expert_used, M]
        float             *out_up,     // [n_tokens*n_expert_used, M]
        const uint2       *qs_gate,
        const __half      *dq_gate,
        const uint2       *qs_up,
        const __half      *dq_up,
        const block_q8_1  *x8,         // [n_tokens][nyb]
        const int32_t     *ids,        // [n_tokens*n_expert_used]
        int                M,
        int                nb,
        int                nyb,
        int                n_expert_used)
{
    const int row  = blockIdx.x;
    const int slot = blockIdx.y;       // flat assignment = token*n_expert_used+slot
    const int lane = threadIdx.x;
    const uint2  *qs = blockIdx.z ? qs_up : qs_gate;
    const __half *dq = blockIdx.z ? dq_up : dq_gate;
    float        *out = blockIdx.z ? out_up : out_gate;
    const int32_t id_raw = ids[slot];
    const bool invalid_id = id_raw < 0;
    const long long rbase = ((long long)(invalid_id ? 0 : id_raw) * M + row) * nb;
    x8 += (long long)(slot / n_expert_used) * nyb;

    float acc = 0.0f;
    for (int b0 = 0; !invalid_id && b0 < nb; b0 += 4) {
        const int b = b0 + (lane >> 3);
        const int p = lane & 7;
        const uint2 cw   = qs[(rbase + b) * 8 + p];
        const uint32_t q2 = cw.x, aux32 = cw.y;
        const uint8_t *aux8 = (const uint8_t *)&q2;

        int sumi = 0;
        const int q8i = (b * 256 + p * 32) / 32;
        const int *u = (const int *)x8[q8i].qs;
#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            const uint2 grid_pos = ((const uint2 *)iq2xxs_grid)[aux8[k0 / 2]];
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));

            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int grid0  = __vsub4(grid_pos.x ^ signs0, signs0);
            sumi = ggml_cuda_dp4a(grid0, u[k0 + 0], sumi);

            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            const int grid1  = __vsub4(grid_pos.y ^ signs1, signs1);
            sumi = ggml_cuda_dp4a(grid1, u[k0 + 1], sumi);
        }
        const int ls = aux32 >> 27 | 1;
        sumi = sumi * ls / 8;
        const float d = __half2float(dq[rbase + b]) * __low2float(x8[q8i].ds);
        acc += d * (float)sumi;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) {
        if (!isfinite(acc)) acc = 0.0f;
        out[(long long)slot * M + row] = acc;
    }
}

// M1-Inc2 variant F: gate and up accumulated in the same warp (interleaved so
// each q8 activation block is loaded once), clamp/SwiGLU/router-weight
// epilogue folded in (semantics copied from
// ds4_mmq_moe_gate_up_mid_q8_1_qwarp32_kernel) -> mid directly.  Replaces
// quantize+gate+up+sanitize+swiglu with quantize+one launch.
__global__ void iq2_xxs_aligned_moe_gate_up_mid_kernel(
        float             *mid,        // [n_tokens*n_expert_used, M]
        const uint2       *qs_gate,
        const __half      *dq_gate,
        const uint2       *qs_up,
        const __half      *dq_up,
        const block_q8_1  *x8,         // [n_tokens][nyb]
        const int32_t     *ids,        // [n_tokens*n_expert_used]
        const float       *weights,    // [n_tokens*n_expert_used] router weights
        int                M,
        int                nb,
        int                nyb,
        int                n_expert_used,
        float              clamp)
{
    const int row  = blockIdx.x;
    const int slot = blockIdx.y;       // flat assignment = token*n_expert_used+slot
    const int lane = threadIdx.x;
    const int32_t id_raw = ids[slot];
    const bool invalid_id = id_raw < 0;
    const long long rbase = ((long long)(invalid_id ? 0 : id_raw) * M + row) * nb;
    x8 += (long long)(slot / n_expert_used) * nyb;

    float acc_g = 0.0f;
    float acc_u = 0.0f;
    for (int b0 = 0; !invalid_id && b0 < nb; b0 += 4) {
        const int b = b0 + (lane >> 3);
        const int p = lane & 7;
        const int q8i = (b * 256 + p * 32) / 32;
        const int *u = (const int *)x8[q8i].qs;
        const float d8 = __low2float(x8[q8i].ds);

        const uint2 cwg = qs_gate[(rbase + b) * 8 + p];
        const uint2 cwu = qs_up[(rbase + b) * 8 + p];
        const uint8_t *aux8g = (const uint8_t *)&cwg.x;
        const uint8_t *aux8u = (const uint8_t *)&cwu.x;

        int sumi_g = 0;
        int sumi_u = 0;
#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            {
                const uint2 grid_pos = ((const uint2 *)iq2xxs_grid)[aux8g[k0 / 2]];
                const uint32_t signs = unpack_ksigns(cwg.y >> (7 * k0 / 2));
                const int signs0 = __vcmpne4(signs & 0x08040201, 0);
                const int grid0  = __vsub4(grid_pos.x ^ signs0, signs0);
                sumi_g = ggml_cuda_dp4a(grid0, u[k0 + 0], sumi_g);
                const int signs1 = __vcmpne4(signs & 0x80402010, 0);
                const int grid1  = __vsub4(grid_pos.y ^ signs1, signs1);
                sumi_g = ggml_cuda_dp4a(grid1, u[k0 + 1], sumi_g);
            }
            {
                const uint2 grid_pos = ((const uint2 *)iq2xxs_grid)[aux8u[k0 / 2]];
                const uint32_t signs = unpack_ksigns(cwu.y >> (7 * k0 / 2));
                const int signs0 = __vcmpne4(signs & 0x08040201, 0);
                const int grid0  = __vsub4(grid_pos.x ^ signs0, signs0);
                sumi_u = ggml_cuda_dp4a(grid0, u[k0 + 0], sumi_u);
                const int signs1 = __vcmpne4(signs & 0x80402010, 0);
                const int grid1  = __vsub4(grid_pos.y ^ signs1, signs1);
                sumi_u = ggml_cuda_dp4a(grid1, u[k0 + 1], sumi_u);
            }
        }
        const int ls_g = cwg.y >> 27 | 1;
        const int ls_u = cwu.y >> 27 | 1;
        acc_g += __half2float(dq_gate[rbase + b]) * d8 * (float)(sumi_g * ls_g / 8);
        acc_u += __half2float(dq_up[rbase + b])   * d8 * (float)(sumi_u * ls_u / 8);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc_g += __shfl_down_sync(0xffffffffu, acc_g, off);
        acc_u += __shfl_down_sync(0xffffffffu, acc_u, off);
    }
    if (lane == 0) {
        float gate = acc_g;
        float up = acc_u;
        if (!isfinite(gate)) gate = 0.0f;
        if (!isfinite(up)) up = 0.0f;
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const float silu = gate / (1.0f + expf(-gate));
        mid[(long long)slot * M + row] = silu * up * weights[slot];
    }
}

template <ggml_type type>
int ds4_mmq_moe_pair_raw_vec_impl(
        const char    * tag,
        const void    * W_a,
        const void    * W_b,
        const float   * X_f32,
        const int32_t * ids,
        float         * out_a,
        float         * out_b,
        int             M,
        int             K,
        int             n_tokens,
        int             n_experts,
        int             n_expert_used,
        cudaStream_t    stream) {

    if (!W_a || !W_b || !X_f32 || !ids || !out_a || !out_b) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || K <= 0 || n_tokens <= 0 || n_experts <= 0 || n_expert_used <= 0) {
        fprintf(stderr, "%s: bad shape M=%d K=%d ntok=%d nexp=%d nused=%d\n",
                tag, M, K, n_tokens, n_experts, n_expert_used);
        return -1;
    }
    if (K % 256 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }
    if (n_expert_used > n_experts) {
        fprintf(stderr, "%s: n_expert_used=%d > n_experts=%d\n", tag, n_expert_used, n_experts);
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }

    ds4_pool_set_stream(stream);

    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t  nbytes_q8_1 = (size_t)n_tokens * ne10_padded *
                                sizeof(block_q8_1) / QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1_pool;
    char *src1_q8_1_ptr = nullptr;
    if (g_q81_scratch_enabled && g_q81_scratch_ptr && g_q81_scratch_bytes >= nbytes_q8_1) {
        src1_q8_1_ptr = (char *)g_q81_scratch_ptr;
    } else {
        src1_q8_1_pool.alloc(ctx->pool(), nbytes_q8_1);
        src1_q8_1_ptr = src1_q8_1_pool.get();
    }

    quantize_row_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1_ptr,
        type, /*ne00=*/K,
        /*s11=*/(int64_t)K, /*s12=*/(int64_t)K, /*s13=*/(int64_t)K * n_tokens,
        /*ne0=*/ne10_padded, /*ne1=*/1, /*ne2=*/n_tokens, /*ne3=*/1,
        stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_row_q8_1_cuda failed: %s\n",
                tag, cudaGetErrorString(err));
        return -2;
    }

    const int64_t blck      = ggml_blck_size(type);
    const int64_t s01_row   = (int64_t)K / blck;
    const int64_t s02_chan  = (int64_t)M * s01_row;
    const int64_t s11_y     = ne10_padded / QK8_1;
    const int64_t s12_y     = (int64_t)1 * s11_y;
    const int64_t s1_dst    = (int64_t)M;
    const int64_t s2_dst    = (int64_t)n_expert_used * M;
    const int ids_stride    = n_expert_used;
    const int cc            = ggml_cuda_info().devices[dev].cc;
    const int col_cap       = get_mmvq_mmid_max_batch(type, ggml_cuda_highest_compiled_arch(cc));
    ggml_cuda_mm_fusion_args_device fusion = {};

    const size_t out_bytes = (size_t)M * (size_t)n_tokens * (size_t)n_expert_used * sizeof(float);
    cudaMemsetAsync(out_a, 0, out_bytes, stream);
    cudaMemsetAsync(out_b, 0, out_bytes, stream);

    for (int c0 = 0; c0 < n_tokens; c0 += col_cap) {
        const int ncols = (n_tokens - c0 < col_cap) ? (n_tokens - c0) : col_cap;
        const void *vy = (const void *)(src1_q8_1_ptr + (size_t)c0 * s12_y * sizeof(block_q8_1));
        const int32_t *ids_chunk = ids + (size_t)c0 * ids_stride;
        float *out_a_chunk = out_a + (int64_t)c0 * s2_dst;
        float *out_b_chunk = out_b + (int64_t)c0 * s2_dst;

        mul_mat_vec_q_switch_type(
            /*vx=*/W_a, /*type_x=*/type,
            /*vy=*/vy, /*ids=*/ids_chunk, /*fusion=*/fusion,
            /*dst=*/out_a_chunk,
            /*ncols_x=*/K, /*nrows_x=*/M, /*ncols_dst=*/ncols,
            /*stride_row_x=*/(int)s01_row,
            /*stride_col_y=*/(int)s12_y,
            /*stride_col_dst=*/(int)s2_dst,
            /*nchannels_x=*/n_experts,
            /*nchannels_y=*/1,
            /*nchannels_dst=*/n_expert_used,
            /*stride_channel_x=*/(int)s02_chan,
            /*stride_channel_y=*/(int)s11_y,
            /*stride_channel_dst=*/(int)s1_dst,
            /*nsamples_x=*/1, /*nsamples_dst=*/1,
            /*stride_sample_x=*/0, /*stride_sample_y=*/0, /*stride_sample_dst=*/0,
            /*ids_stride=*/ids_stride, stream);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: mul_mat_vec_q_switch_type (a) failed: %s (cols %d..%d cap %d)\n",
                    tag, cudaGetErrorString(err), c0, c0 + ncols - 1, col_cap);
            return -3;
        }

        mul_mat_vec_q_switch_type(
            /*vx=*/W_b, /*type_x=*/type,
            /*vy=*/vy, /*ids=*/ids_chunk, /*fusion=*/fusion,
            /*dst=*/out_b_chunk,
            /*ncols_x=*/K, /*nrows_x=*/M, /*ncols_dst=*/ncols,
            /*stride_row_x=*/(int)s01_row,
            /*stride_col_y=*/(int)s12_y,
            /*stride_col_dst=*/(int)s2_dst,
            /*nchannels_x=*/n_experts,
            /*nchannels_y=*/1,
            /*nchannels_dst=*/n_expert_used,
            /*stride_channel_x=*/(int)s02_chan,
            /*stride_channel_y=*/(int)s11_y,
            /*stride_channel_dst=*/(int)s1_dst,
            /*nsamples_x=*/1, /*nsamples_dst=*/1,
            /*stride_sample_x=*/0, /*stride_sample_y=*/0, /*stride_sample_dst=*/0,
            /*ids_stride=*/ids_stride, stream);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: mul_mat_vec_q_switch_type (b) failed: %s (cols %d..%d cap %d)\n",
                    tag, cudaGetErrorString(err), c0, c0 + ncols - 1, col_cap);
            return -4;
        }
    }

    const uint64_t out_count = (uint64_t)M * (uint64_t)n_tokens * (uint64_t)n_expert_used;
    ds4_mmq_sanitize_f32(out_a, out_count, stream);
    ds4_mmq_sanitize_f32(out_b, out_count, stream);
    return 0;
}

template <ggml_type type>
int ds4_mmq_moe_pair_vec_impl(
        const char    * tag,
        const void    * W_a,
        const void    * W_b,
        const float   * X_f32,
        const int32_t * ids,
        float         * out_silu,
        int             M,
        int             K,
        int             n_experts,
        int             n_expert_used,
        cudaStream_t    stream) {

    if (!W_a || !W_b || !X_f32 || !ids || !out_silu) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || K <= 0 || n_experts <= 0 || n_expert_used <= 0) {
        fprintf(stderr, "%s: bad shape M=%d K=%d nexp=%d nused=%d\n",
                tag, M, K, n_experts, n_expert_used);
        return -1;
    }
    if (K % 256 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }
    if (n_expert_used > n_experts) {
        fprintf(stderr, "%s: n_expert_used=%d > n_experts=%d\n", tag, n_expert_used, n_experts);
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }

    // Route the pool's cudaMallocAsync through the caller-supplied stream
    // for Step 8 / CUDA Graph compatibility.  See ds4_mmq_moe_vec_impl.
    ds4_pool_set_stream(stream);

    const int n_tokens = 1;  // fusion only supported at ncols_dst=1.

    // Quantize X (single token) into canonical Q8_1.
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t  nbytes_q8_1 = (size_t)n_tokens * ne10_padded *
                                sizeof(block_q8_1) / QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx->pool(), nbytes_q8_1);

    quantize_row_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1.get(),
        type, /*ne00=*/K,
        /*s11=*/(int64_t)K, /*s12=*/(int64_t)K, /*s13=*/(int64_t)K * n_tokens,
        /*ne0=*/ne10_padded, /*ne1=*/1, /*ne2=*/n_tokens, /*ne3=*/1,
        stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_row_q8_1_cuda failed: %s\n",
                tag, cudaGetErrorString(err));
        return -2;
    }

    const int64_t blck      = ggml_blck_size(type);
    const int64_t s01_row   = (int64_t)K / blck;
    const int64_t s02_chan  = (int64_t)M * s01_row;
    const int64_t s11_y     = ne10_padded / QK8_1;
    const int64_t s12_y     = (int64_t)1 * s11_y;
    const int64_t s1_dst    = (int64_t)M;
    const int ids_stride    = n_expert_used;

    // Configure fusion: gate=W_b (up weights), glu_op=SWIGLU.
    // mmvq's kernel will compute, for each (channel_dst, row):
    //   a = vec_dot(W_a, x); b = vec_dot(W_b, x);
    //   dst = silu(a) * b
    ggml_cuda_mm_fusion_args_device fusion = {};
    fusion.gate   = W_b;
    fusion.glu_op = GGML_GLU_OP_SWIGLU;

    cudaMemsetAsync(out_silu, 0, (size_t)M * (size_t)n_expert_used * sizeof(float), stream);

    mul_mat_vec_q_switch_type(
        /*vx=*/W_a, /*type_x=*/type,
        /*vy=*/(const void *)src1_q8_1.get(),
        /*ids=*/ids, /*fusion=*/fusion,
        /*dst=*/out_silu,
        /*ncols_x=*/K, /*nrows_x=*/M, /*ncols_dst=*/n_tokens,
        /*stride_row_x=*/(int)s01_row,
        /*stride_col_y=*/(int)s12_y,
        /*stride_col_dst=*/(int)s1_dst,
        /*nchannels_x=*/n_experts,
        /*nchannels_y=*/1,
        /*nchannels_dst=*/n_expert_used,
        /*stride_channel_x=*/(int)s02_chan,
        /*stride_channel_y=*/(int)s11_y,
        /*stride_channel_dst=*/(int)s1_dst,
        /*nsamples_x=*/1, /*nsamples_dst=*/1,
        /*stride_sample_x=*/0, /*stride_sample_y=*/0, /*stride_sample_dst=*/0,
        /*ids_stride=*/ids_stride, stream);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: mul_mat_vec_q_switch_type (fused) launch failed: %s\n",
                tag, cudaGetErrorString(err));
        return -3;
    }
    ds4_mmq_sanitize_f32(out_silu, (uint64_t)M * (uint64_t)n_expert_used, stream);
    return 0;
}

template <ggml_type type>
int ds4_mmq_dense_vec_impl(
        const char  * tag,
        const void  * W,
        const float * X_f32,
        float       * out_f32,
        int           M,
        int           N,
        int           K,
        cudaStream_t  stream) {

    if (!W || !X_f32 || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        fprintf(stderr, "%s: bad shape M=%d N=%d K=%d\n", tag, M, N, K);
        return -1;
    }
    if (K % 256 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }
    if (N > MMVQ_MAX_BATCH_SIZE) {
        fprintf(stderr, "%s: N=%d exceeds MMVQ_MAX_BATCH_SIZE=%d\n",
                tag, N, MMVQ_MAX_BATCH_SIZE);
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }

    // Route the pool's cudaMallocAsync through the caller-supplied stream
    // for Step 8 / CUDA Graph compatibility.  See ds4_mmq_moe_vec_impl.
    ds4_pool_set_stream(stream);

    // Dense: no MoE, ids=null. Layout [K, N, 1, 1] for src1.
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t  nbytes_q8_1 = (size_t)N * ne10_padded *
                                sizeof(block_q8_1) / QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx->pool(), nbytes_q8_1);

    // Dense src1 layout: K innermost, N next; ne11=N, ne12=1, ne13=1.
    quantize_row_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1.get(),
        type, /*ne00=*/K,
        /*s11=*/(int64_t)K, /*s12=*/(int64_t)K * N, /*s13=*/(int64_t)K * N,
        /*ne0=*/ne10_padded, /*ne1=*/N, /*ne2=*/1, /*ne3=*/1,
        stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_row_q8_1_cuda failed: %s\n",
                tag, cudaGetErrorString(err));
        return -2;
    }

    // Dense (no ids): per upstream dispatch (mmvq.cu:1121-1127),
    //   ncols_dst          = ne1  = N
    //   nchannels_y        = ne12 = 1
    //   nchannels_dst      = ne2  = 1
    //   stride_col_y       = s11  = ne10_padded / QK8_1
    //   stride_channel_y   = s12  = N * (ne10_padded / QK8_1)
    const int64_t blck      = ggml_blck_size(type);
    const int64_t s01_row   = (int64_t)K / blck;
    const int64_t s11_y     = ne10_padded / QK8_1;
    const int64_t s12_y     = (int64_t)N * s11_y;
    const int64_t s1_dst    = (int64_t)M;

    ggml_cuda_mm_fusion_args_device fusion = {};

    cudaMemsetAsync(out_f32, 0, (size_t)M * (size_t)N * sizeof(float), stream);

    mul_mat_vec_q_switch_type(
        /*vx=*/W, /*type_x=*/type,
        /*vy=*/(const void *)src1_q8_1.get(),
        /*ids=*/nullptr, /*fusion=*/fusion,
        /*dst=*/out_f32,
        /*ncols_x=*/K, /*nrows_x=*/M, /*ncols_dst=*/N,
        /*stride_row_x=*/(int)s01_row,
        /*stride_col_y=*/(int)s11_y,
        /*stride_col_dst=*/(int)s1_dst,
        /*nchannels_x=*/1,
        /*nchannels_y=*/1,
        /*nchannels_dst=*/1,
        /*stride_channel_x=*/0,
        /*stride_channel_y=*/(int)s12_y,
        /*stride_channel_dst=*/0,
        /*nsamples_x=*/1, /*nsamples_dst=*/1,
        /*stride_sample_x=*/0, /*stride_sample_y=*/0, /*stride_sample_dst=*/0,
        /*ids_stride=*/0, stream);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: mul_mat_vec_q_switch_type (dense) launch failed: %s\n",
                tag, cudaGetErrorString(err));
        return -3;
    }
    ds4_mmq_sanitize_f32(out_f32, (uint64_t)M * (uint64_t)N, stream);
    return 0;
}

template <ggml_type type> struct ds4_mmq_vdr_mmvq_value;
template <> struct ds4_mmq_vdr_mmvq_value<GGML_TYPE_IQ2_XXS> { static constexpr int value = VDR_IQ2_XXS_Q8_1_MMVQ; };
template <> struct ds4_mmq_vdr_mmvq_value<GGML_TYPE_Q2_K>    { static constexpr int value = VDR_Q2_K_Q8_1_MMVQ; };
template <> struct ds4_mmq_vdr_mmvq_value<GGML_TYPE_Q4_K>    { static constexpr int value = VDR_Q4_K_Q8_1_MMVQ; };

template <ggml_type type>
static __device__ __forceinline__ float ds4_mmq_vec_dot_q8_1(
        const void * __restrict__ W,
        const block_q8_1 * __restrict__ X_q8,
        const int & kbx,
        const int & iqs) {
    if constexpr (type == GGML_TYPE_IQ2_XXS) {
        return vec_dot_iq2_xxs_q8_1(W, X_q8, kbx, iqs);
    } else if constexpr (type == GGML_TYPE_Q2_K) {
        return vec_dot_q2_K_q8_1(W, X_q8, kbx, iqs);
    } else {
        static_assert(type == GGML_TYPE_Q4_K, "unsupported fused vector type");
        return vec_dot_q4_K_q8_1(W, X_q8, kbx, iqs);
    }
}

static __device__ __forceinline__ float ds4_mmq_half_warp_sum_f32(float v) {
    const uint32_t mask = 0xffffu << (threadIdx.x & 16u);
    for (int offset = 8; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset, 16);
    }
    return v;
}

template <ggml_type type>
static __global__ void ds4_mmq_moe_down_sum6_q8_1_qwarp32_kernel(
        const void       * __restrict__ W,
        const block_q8_1 * __restrict__ X_q8,
        const int32_t    * __restrict__ ids,
        float            * __restrict__ out,
        const uint32_t ncols_x,
        const uint32_t nrows_x,
        const uint32_t n_tokens,
        const uint32_t n_experts,
        const uint32_t stride_row_x,
        const uint32_t stride_col_y,
        const uint32_t stride_channel_x) {

    constexpr int top_k = 6;
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int q8_per_k = qk / QK8_1;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = ds4_mmq_vdr_mmvq_value<type>::value;
    constexpr int lanes_per_k = qi / vdr;
    constexpr int blocks_per_iter = vdr * 16 / qi;
    const uint32_t lane = threadIdx.x & 15u;
    const uint32_t row_lane = threadIdx.x >> 4u;
    const uint32_t tok  = blockIdx.y;
    if (tok >= n_tokens) return;

    const uint32_t blocks_per_row_x = ncols_x / qk;
    const uint32_t kbx0 = lane / lanes_per_k;
    const int kqs = vdr * (lane % lanes_per_k);

#pragma unroll
    for (uint32_t rr = 0; rr < 4u; ++rr) {
        const uint32_t row = blockIdx.x * 64u + row_lane + rr * 16u;
        if (row >= nrows_x) continue;
        float total = 0.0f;
#pragma unroll
        for (uint32_t slot = 0; slot < top_k; ++slot) {
            const uint32_t assignment = tok * top_k + slot;
            const int32_t id_raw = ids[assignment];
            const bool invalid_id = id_raw < 0 || (uint32_t)id_raw >= n_experts;
            const uint32_t expert = invalid_id ? 0u : (uint32_t)id_raw;
            const block_q8_1 * xq = X_q8 + (uint64_t)assignment * stride_col_y;
            const int kbx_base = (int)(expert * stride_channel_x + row * stride_row_x);
            float acc = 0.0f;
            for (uint32_t b = kbx0; !invalid_id && b < blocks_per_row_x; b += blocks_per_iter) {
                acc += ds4_mmq_vec_dot_q8_1<type>(
                    W, xq + (uint64_t)b * q8_per_k, kbx_base + (int)b, kqs);
            }
            acc = ds4_mmq_half_warp_sum_f32(acc);
            if (lane == 0) {
                if (!isfinite(acc)) acc = 0.0f;
                total += acc;
            }
        }
        if (lane == 0) {
            out[(uint64_t)tok * nrows_x + row] = total;
        }
    }
}

template <ggml_type type>
static __global__ void ds4_mmq_moe_gate_up_mid_q8_1_qwarp32_kernel(
        const void       * __restrict__ W_gate,
        const void       * __restrict__ W_up,
        const block_q8_1 * __restrict__ X_q8,
        const int32_t    * __restrict__ ids,
        const float      * __restrict__ weights,
        float            * __restrict__ mid,
        const uint32_t ncols_x,
        const uint32_t nrows_x,
        const uint32_t n_tokens,
        const uint32_t n_experts,
        const uint32_t stride_row_x,
        const uint32_t stride_col_y,
        const uint32_t stride_channel_x,
        const float clamp) {

    constexpr int top_k = 6;
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int q8_per_k = qk / QK8_1;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = ds4_mmq_vdr_mmvq_value<type>::value;
    constexpr int lanes_per_k = qi / vdr;
    constexpr int blocks_per_iter = vdr * 16 / qi;
    const uint32_t lane = threadIdx.x & 15u;
    const uint32_t row_lane = threadIdx.x >> 4u;
    const uint32_t assignment = blockIdx.y;
    const uint32_t tok = assignment / top_k;
    const uint32_t slot = assignment - tok * top_k;
    if (tok >= n_tokens) return;

    const int32_t id_raw = ids[(uint64_t)tok * top_k + slot];
    const bool invalid_id = id_raw < 0 || (uint32_t)id_raw >= n_experts;
    const uint32_t expert = invalid_id ? 0u : (uint32_t)id_raw;
    const block_q8_1 * xq = X_q8 + (uint64_t)tok * stride_col_y;
    const uint32_t blocks_per_row_x = ncols_x / qk;
    const uint32_t kbx0 = lane / lanes_per_k;
    const int kqs = vdr * (lane % lanes_per_k);

#pragma unroll
    for (uint32_t rr = 0; rr < 4u; ++rr) {
        const uint32_t row = blockIdx.x * 64u + row_lane + rr * 16u;
        if (row >= nrows_x) continue;
        const int kbx_base = (int)(expert * stride_channel_x + row * stride_row_x);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = kbx0; !invalid_id && b < blocks_per_row_x; b += blocks_per_iter) {
            const block_q8_1 * xb = xq + (uint64_t)b * q8_per_k;
            const int kbx = kbx_base + (int)b;
            gate += ds4_mmq_vec_dot_q8_1<type>(W_gate, xb, kbx, kqs);
            up   += ds4_mmq_vec_dot_q8_1<type>(W_up,   xb, kbx, kqs);
        }
        gate = ds4_mmq_half_warp_sum_f32(gate);
        up   = ds4_mmq_half_warp_sum_f32(up);
        if (lane == 0) {
            if (!isfinite(gate)) gate = 0.0f;
            if (!isfinite(up)) up = 0.0f;
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const float silu = gate / (1.0f + expf(-gate));
            mid[(uint64_t)assignment * nrows_x + row] = silu * up * weights[(uint64_t)tok * top_k + slot];
        }
    }
}

template <ggml_type type, int c_rows_per_block>
static __global__ void ds4_mmq_moe_down_sum6_vec_kernel(
        const void       * __restrict__ W,
        const block_q8_1 * __restrict__ X_q8,
        const int32_t    * __restrict__ ids,
        float            * __restrict__ out,
        const uint32_t ncols_x,
        const uint32_t nrows_x,
        const uint32_t n_tokens,
        const uint32_t n_experts,
        const uint32_t stride_row_x,
        const uint32_t stride_col_y,
        const uint32_t stride_channel_x) {

    constexpr int top_k = 6;
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = ds4_mmq_vdr_mmvq_value<type>::value;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const uint32_t slot  = threadIdx.y;
    const uint32_t token = blockIdx.y;
    const uint32_t row0  = c_rows_per_block * blockIdx.x;

    if (slot >= top_k || token >= n_tokens) {
        return;
    }

    const uint32_t assignment = token * top_k + slot;
    const int32_t  id_raw     = ids[assignment];
    const bool     invalid_id = id_raw < 0 || (uint32_t)id_raw >= n_experts;
    const uint32_t expert     = invalid_id ? 0u : (uint32_t)id_raw;

    const int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * warp_size / qi;

    const block_q8_1 * y = X_q8 + (uint64_t)assignment * stride_col_y;
    const int kbx_offset = (int)(expert * stride_channel_x + row0 * stride_row_x);

    float tmp[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi / vdr); !invalid_id && kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi / vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] += ds4_mmq_vec_dot_q8_1<type>(
                W, &y[kby], kbx_offset + i * stride_row_x + kbx, kqs);
        }
    }

#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
    }

    __shared__ float partial[top_k][c_rows_per_block];
    if (threadIdx.x < c_rows_per_block) {
        const uint32_t row = row0 + threadIdx.x;
        partial[slot][threadIdx.x] = row < nrows_x ? tmp[threadIdx.x] : 0.0f;
    }
    __syncthreads();

    if (slot == 0 && threadIdx.x < c_rows_per_block) {
        const uint32_t row = row0 + threadIdx.x;
        if (row < nrows_x) {
            float sum = 0.0f;
#pragma unroll
            for (int s = 0; s < top_k; ++s) {
                sum += partial[s][threadIdx.x];
            }
            out[(uint64_t)token * nrows_x + row] = sum;
        }
    }
}

template <ggml_type type>
int ds4_mmq_moe_down_sum6_vec_impl(
        const char    * tag,
        const void    * W,
        const float   * X_f32,
        const int32_t * ids,
        float         * out_f32,
        int             M,
        int             K,
        int             n_tokens,
        int             n_experts,
        int             n_expert_used,
        cudaStream_t    stream) {

    if (!W || !X_f32 || !ids || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || K <= 0 || n_tokens <= 0 || n_experts <= 0 || n_expert_used != 6) {
        fprintf(stderr, "%s: bad shape M=%d K=%d ntok=%d nexp=%d nused=%d\n",
                tag, M, K, n_tokens, n_experts, n_expert_used);
        return -1;
    }
    if (K % 256 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }

    ds4_pool_set_stream(stream);

    const int n_assignments = n_tokens * n_expert_used;
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t nbytes_q8_1 = (size_t)n_assignments * ne10_padded *
                               sizeof(block_q8_1) / QK8_1;

    ggml_cuda_pool_alloc<char> src1_q8_1_pool;
    char *src1_q8_1_ptr = nullptr;
    if (g_q81_scratch_enabled && g_q81_scratch_ptr && g_q81_scratch_bytes >= nbytes_q8_1) {
        src1_q8_1_ptr = (char *)g_q81_scratch_ptr;
    } else {
        src1_q8_1_pool.alloc(ctx->pool(), nbytes_q8_1);
        src1_q8_1_ptr = src1_q8_1_pool.get();
    }

    quantize_row_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1_ptr,
        type, /*ne00=*/K,
        /*s11=*/(int64_t)K, /*s12=*/(int64_t)K, /*s13=*/(int64_t)K * n_assignments,
        /*ne0=*/ne10_padded, /*ne1=*/1, /*ne2=*/n_assignments, /*ne3=*/1,
        stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_row_q8_1_cuda failed: %s\n",
                tag, cudaGetErrorString(err));
        return -2;
    }

    const int64_t blck = ggml_blck_size(type);
    const uint32_t stride_row_x     = (uint32_t)((int64_t)K / blck);
    const uint32_t stride_col_y     = (uint32_t)(ne10_padded / QK8_1);
    const uint32_t stride_channel_x = (uint32_t)((int64_t)M * stride_row_x);

    const dim3 block_nums((M + 63) / 64, n_tokens);
    const dim3 block_dims(256);

    ds4_mmq_moe_down_sum6_q8_1_qwarp32_kernel<type><<<block_nums, block_dims, 0, stream>>>(
        W, (const block_q8_1 *)src1_q8_1_ptr, ids, out_f32,
        (uint32_t)K, (uint32_t)M, (uint32_t)n_tokens, (uint32_t)n_experts,
        stride_row_x, stride_col_y, stride_channel_x);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: fused down+sum launch failed: %s\n",
                tag, cudaGetErrorString(err));
        return -3;
    }

    return 0;
}

template <ggml_type type, int c_rows_per_block>
static __global__ void ds4_mmq_moe_gate_up_mid_vec_kernel(
        const void       * __restrict__ W_gate,
        const void       * __restrict__ W_up,
        const block_q8_1 * __restrict__ X_q8,
        const int32_t    * __restrict__ ids,
        const float      * __restrict__ weights,
        float            * __restrict__ mid,
        const uint32_t ncols_x,
        const uint32_t nrows_x,
        const uint32_t n_tokens,
        const uint32_t n_experts,
        const uint32_t stride_row_x,
        const uint32_t stride_col_y,
        const uint32_t stride_channel_x,
        const float clamp) {

    constexpr int top_k = 6;
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = ds4_mmq_vdr_mmvq_value<type>::value;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const uint32_t slot  = threadIdx.y;
    const uint32_t token = blockIdx.y;
    const uint32_t row0  = c_rows_per_block * blockIdx.x;

    const uint32_t assignment = token * top_k + slot;
    const int32_t  id_raw     = ids[assignment];
    const bool     invalid_id = id_raw < 0 || (uint32_t)id_raw >= n_experts;
    const uint32_t expert     = invalid_id ? 0u : (uint32_t)id_raw;

    const int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * warp_size / qi;

    const block_q8_1 * y = X_q8 + (uint64_t)token * stride_col_y;
    const int kbx_offset = (int)(expert * stride_channel_x + row0 * stride_row_x);

    float gate[c_rows_per_block] = {0.0f};
    float up[c_rows_per_block]   = {0.0f};

    for (int kbx = threadIdx.x / (qi / vdr); !invalid_id && kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi / vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            const int row_kbx = kbx_offset + i * stride_row_x + kbx;
            gate[i] += ds4_mmq_vec_dot_q8_1<type>(W_gate, &y[kby], row_kbx, kqs);
            up[i]   += ds4_mmq_vec_dot_q8_1<type>(W_up,   &y[kby], row_kbx, kqs);
        }
    }

#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        gate[i] = warp_reduce_sum<warp_size>(gate[i]);
        up[i]   = warp_reduce_sum<warp_size>(up[i]);
    }

    if (threadIdx.x < c_rows_per_block) {
        const uint32_t row = row0 + threadIdx.x;
        if (row < nrows_x) {
            float g = gate[threadIdx.x];
            float u = up[threadIdx.x];
            if (!isfinite(g)) g = 0.0f;
            if (!isfinite(u)) u = 0.0f;
            if (clamp > 1.0e-6f) {
                if (g > clamp) g = clamp;
                if (u > clamp) u = clamp;
                if (u < -clamp) u = -clamp;
            }
            const float silu = g / (1.0f + expf(-g));
            mid[(uint64_t)assignment * nrows_x + row] = silu * u * weights[assignment];
        }
    }
}

template <ggml_type type, int c_rows_per_block>
static __global__ void ds4_mmq_moe_gate_up_mid_vec_by_slot_kernel(
        const void       * __restrict__ W_gate,
        const void       * __restrict__ W_up,
        const block_q8_1 * __restrict__ X_q8,
        const int32_t    * __restrict__ ids,
        const float      * __restrict__ weights,
        float            * __restrict__ mid,
        const uint32_t ncols_x,
        const uint32_t nrows_x,
        const uint32_t n_experts,
        const uint32_t stride_row_x,
        const uint32_t stride_col_y,
        const uint32_t stride_channel_x,
        const uint32_t token0,
        const float clamp) {

    constexpr int top_k = 6;
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = ds4_mmq_vdr_mmvq_value<type>::value;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const uint32_t slot  = blockIdx.y;
    const uint32_t token = token0 + threadIdx.y;
    const uint32_t row0  = c_rows_per_block * blockIdx.x;

    const uint32_t assignment = token * top_k + slot;
    const int32_t  id_raw     = ids[assignment];
    const bool     invalid_id = id_raw < 0 || (uint32_t)id_raw >= n_experts;
    const uint32_t expert     = invalid_id ? 0u : (uint32_t)id_raw;

    const int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * warp_size / qi;

    const block_q8_1 * y = X_q8 + (uint64_t)token * stride_col_y;
    const int kbx_offset = (int)(expert * stride_channel_x + row0 * stride_row_x);

    float gate[c_rows_per_block] = {0.0f};
    float up[c_rows_per_block]   = {0.0f};

    for (int kbx = threadIdx.x / (qi / vdr); !invalid_id && kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi / vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            const int row_kbx = kbx_offset + i * stride_row_x + kbx;
            gate[i] += ds4_mmq_vec_dot_q8_1<type>(W_gate, &y[kby], row_kbx, kqs);
            up[i]   += ds4_mmq_vec_dot_q8_1<type>(W_up,   &y[kby], row_kbx, kqs);
        }
    }

#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        gate[i] = warp_reduce_sum<warp_size>(gate[i]);
        up[i]   = warp_reduce_sum<warp_size>(up[i]);
    }

    if (threadIdx.x < c_rows_per_block) {
        const uint32_t row = row0 + threadIdx.x;
        if (row < nrows_x) {
            float g = gate[threadIdx.x];
            float u = up[threadIdx.x];
            if (!isfinite(g)) g = 0.0f;
            if (!isfinite(u)) u = 0.0f;
            if (clamp > 1.0e-6f) {
                if (g > clamp) g = clamp;
                if (u > clamp) u = clamp;
                if (u < -clamp) u = -clamp;
            }
            const float silu = g / (1.0f + expf(-g));
            mid[(uint64_t)assignment * nrows_x + row] = silu * u * weights[assignment];
        }
    }
}

template <ggml_type type>
int ds4_mmq_moe_gate_up_mid_vec_impl(
        const char    * tag,
        const void    * W_gate,
        const void    * W_up,
        const float   * X_f32,
        const int32_t * ids,
        const float   * weights,
        float         * mid_f32,
        int             M,
        int             K,
        int             n_tokens,
        int             n_experts,
        int             n_expert_used,
        float           clamp,
        cudaStream_t    stream) {

    if (!W_gate || !W_up || !X_f32 || !ids || !weights || !mid_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || K <= 0 || n_tokens <= 0 || n_experts <= 0 || n_expert_used != 6) {
        fprintf(stderr, "%s: bad shape M=%d K=%d ntok=%d nexp=%d nused=%d\n",
                tag, M, K, n_tokens, n_experts, n_expert_used);
        return -1;
    }
    if (K % 256 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }

    ds4_pool_set_stream(stream);

    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t nbytes_q8_1 = (size_t)n_tokens * ne10_padded *
                               sizeof(block_q8_1) / QK8_1;

    ggml_cuda_pool_alloc<char> src1_q8_1_pool;
    // M2-Inc2a: the fused HC stage may have emitted this activation's q8_1
    // codes already (ffn_norm) -- take them and skip the quantize prelude.
    char *src1_q8_1_ptr = ds4_mmq_folded_q81(X_f32, K, n_tokens, ne10_padded);
    cudaError_t err;
    if (!src1_q8_1_ptr) {
    if (g_q81_scratch_enabled && g_q81_scratch_ptr && g_q81_scratch_bytes >= nbytes_q8_1) {
        src1_q8_1_ptr = (char *)g_q81_scratch_ptr;
    } else {
        src1_q8_1_pool.alloc(ctx->pool(), nbytes_q8_1);
        src1_q8_1_ptr = src1_q8_1_pool.get();
    }

    quantize_row_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1_ptr,
        type, /*ne00=*/K,
        /*s11=*/(int64_t)K, /*s12=*/(int64_t)K, /*s13=*/(int64_t)K * n_tokens,
        /*ne0=*/ne10_padded, /*ne1=*/1, /*ne2=*/n_tokens, /*ne3=*/1,
        stream);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_row_q8_1_cuda failed: %s\n",
                tag, cudaGetErrorString(err));
        return -2;
    }
    }

    const int64_t blck = ggml_blck_size(type);
    const uint32_t stride_row_x     = (uint32_t)((int64_t)K / blck);
    const uint32_t stride_col_y     = (uint32_t)(ne10_padded / QK8_1);
    const uint32_t stride_channel_x = (uint32_t)((int64_t)M * stride_row_x);

    const dim3 block_nums((M + 63) / 64, n_tokens * n_expert_used);
    const dim3 block_dims(256);
    ds4_mmq_moe_gate_up_mid_q8_1_qwarp32_kernel<type><<<block_nums, block_dims, 0, stream>>>(
        W_gate, W_up, (const block_q8_1 *)src1_q8_1_ptr, ids, weights, mid_f32,
        (uint32_t)K, (uint32_t)M, (uint32_t)n_tokens, (uint32_t)n_experts,
        stride_row_x, stride_col_y, stride_channel_x, clamp);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: fused gate+up qwarp launch failed: %s\n",
                tag, cudaGetErrorString(err));
        return -3;
    }

    return 0;
}

} // anonymous namespace

extern "C" int ds4_mmq_q8_0_moe_vec(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_vec_impl<GGML_TYPE_Q8_0>(
        "ds4_mmq_q8_0_moe_vec", W, X, ids, out, M, K,
        n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_q2_K_moe_vec(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_vec_impl<GGML_TYPE_Q2_K>(
        "ds4_mmq_q2_K_moe_vec", W, X, ids, out, M, K,
        n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_iq2_xxs_moe_vec(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_vec_impl<GGML_TYPE_IQ2_XXS>(
        "ds4_mmq_iq2_xxs_moe_vec", W, X, ids, out, M, K,
        n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_q4_K_moe_vec(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_vec_impl<GGML_TYPE_Q4_K>(
        "ds4_mmq_q4_K_moe_vec", W, X, ids, out, M, K,
        n_tokens, n_experts, n_expert_used, stream);
}

// M1-Inc2b: exact inverse of the weight-server repack
// (repack_iq2_xxs_aligned_kernel, tools/ds4_weight_server.cu): aligned-SoA
// artifact -> raw block_iq2_xxs byte stream (66B = [half d][8 x uint2
// codes]).  Device->device fill of a raw-layout scratch so the batched/mmq
// consumers keep their layout while the raw spans stay excluded from the
// upload.  One thread per (block, pair); p==0 additionally writes the
// 2-byte scale.  Destination blocks are 66B so stores are byte-granular.
__global__ void iq2_xxs_aligned_derepack_kernel(
        unsigned char     *raw,        // [nblk * 66]
        const uint2       *qs,         // 64B-aligned code pairs
        const __half      *dq,         // block scales
        uint64_t           nblk)
{
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nblk * 8ull) return;
    const uint64_t blk = i >> 3;
    const uint32_t p = (uint32_t)(i & 7u);
    unsigned char *dst = raw + blk * 66ull;
    if (p == 0u) {
        const uint16_t h = __half_as_ushort(dq[blk]);
        memcpy(dst, &h, 2u);
    }
    const uint2 v = qs[blk * 8ull + p];
    memcpy(dst + 2u + (uint64_t)p * 8u, &v, 8u);
}

extern "C" int ds4_mmq_iq2_xxs_aligned_derepack(
        const void * W_aligned, void * raw_out,
        int M, int K, int n_experts, cudaStream_t stream) {
    const char *tag = "ds4_mmq_iq2_xxs_aligned_derepack";
    if (!W_aligned || !raw_out) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || K <= 0 || n_experts <= 0 || K % 256 != 0) return -1;
    const uint64_t nblk = (uint64_t)n_experts * (uint64_t)M * (uint64_t)(K / 256);
    const uint64_t dq_bytes = (nblk * 2u + 63u) & ~63ull;
    const uint64_t n_threads = nblk * 8ull;
    iq2_xxs_aligned_derepack_kernel<<<(unsigned)((n_threads + 255ull) / 256ull), 256, 0, stream>>>(
        (unsigned char *)raw_out,
        (const uint2 *)((const char *)W_aligned + dq_bytes),
        (const __half *)W_aligned,
        nblk);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: kernel launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Aligned-SoA Q8_0 dense decode matvec (megakernel program M1-Inc3).
//
// block_q8_0 is 34 bytes ([half d][int8 qs[32]]), so the raw code stream is
// only 2-byte aligned — the same misalignment class proto_iq2_aligned proved
// costly.  Artifact layout (weight server --repack-q8-aligned, derived kind
// DERIVED_Q8_0_ALIGNED_DENSE): [__half dq[nblk]][pad to 64B][int8 qs[nblk*32]]
// with nblk = M * (K/32), block order equal to the raw tensor byte order.
// Unlike the IQ2 expert repack, the raw spans stay SERVED (dense tensors are
// ~6 GiB total, affordable to duplicate), so every other consumer is
// unchanged.  proto_q8_aligned.cu A/B (GB10, L2-defeating rotation, double-ref
// parity): attn_q_b 217->235, mid 2048x4096 172->218, out_a 8192x4096
// 199->230, head 224->243 GB/s; the warp-per-row accumulation is also ~1000x
// closer to the double reference than the mmvq tile order at K>=4096.
__global__ void q8_0_aligned_dense_vec_kernel(
        float             *out,        // [M]
        const int4        *qs,         // aligned codes, 2 int4 per block
        const __half      *dq,         // block scales
        const block_q8_1  *x8,         // [K/32] canonical Q8_1 activation
        int                M,
        int                nb)         // blocks per row = K/32
{
    const int row  = blockIdx.x;
    const int lane = threadIdx.x;
    const long long rbase = (long long)row * nb;

    float acc = 0.0f;
    for (int b0 = 0; b0 < nb; b0 += 32) {
        const int b = b0 + lane;
        const int4 w0 = qs[(rbase + b) * 2 + 0];   // aligned 16B loads
        const int4 w1 = qs[(rbase + b) * 2 + 1];
        const int *u = (const int *)x8[b].qs;
        int sumi = 0;
        sumi = ggml_cuda_dp4a(w0.x, u[0], sumi);
        sumi = ggml_cuda_dp4a(w0.y, u[1], sumi);
        sumi = ggml_cuda_dp4a(w0.z, u[2], sumi);
        sumi = ggml_cuda_dp4a(w0.w, u[3], sumi);
        sumi = ggml_cuda_dp4a(w1.x, u[4], sumi);
        sumi = ggml_cuda_dp4a(w1.y, u[5], sumi);
        sumi = ggml_cuda_dp4a(w1.z, u[6], sumi);
        sumi = ggml_cuda_dp4a(w1.w, u[7], sumi);
        acc += __half2float(dq[rbase + b]) * __low2float(x8[b].ds) * (float)sumi;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) out[row] = acc;
}

extern "C" uint64_t ds4_mmq_q8_0_aligned_bytes(int M, int K) {
    if (M <= 0 || K <= 0 || K % 1024 != 0) return 0;
    const uint64_t nblk = (uint64_t)M * (uint64_t)(K / 32);
    const uint64_t dq_bytes = (nblk * 2u + 63u) & ~63ull;
    return dq_bytes + nblk * 32u;
}

extern "C" int ds4_mmq_q8_0_aligned_dense_vec(
        const void * W_aligned, const float * X_f32, float * out_f32,
        int M, int N, int K, cudaStream_t stream) {
    const char *tag = "ds4_mmq_q8_0_aligned_dense_vec";
    if (!W_aligned || !X_f32 || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    // K % 1024: the kernel's 32-blocks-per-pass loop needs nb % 32 == 0.
    if (N != 1 || M <= 0 || K <= 0 || K % 1024 != 0) return -1;

    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }
    ds4_pool_set_stream(stream);
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t  nbytes_q8_1 = (size_t)ne10_padded * sizeof(block_q8_1) / QK8_1;
    ggml_cuda_pool_alloc<char> q8_pool;
    // M2-Inc2a: producer-emitted q8_1 codes (qr_norm from the qkv-rms
    // kernel) -- take them and skip the quantize prelude.
    char *x8 = ds4_mmq_folded_q81(X_f32, K, 1, ne10_padded);
    cudaError_t err;
    if (!x8) {
    if (g_q81_scratch_enabled && g_q81_scratch_ptr && g_q81_scratch_bytes >= nbytes_q8_1) {
        x8 = (char *)g_q81_scratch_ptr;
    } else {
        q8_pool.alloc(ctx->pool(), nbytes_q8_1);
        x8 = q8_pool.get();
    }
    quantize_row_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)x8,
        GGML_TYPE_Q8_0, /*ne00=*/K,
        /*s11=*/(int64_t)K, /*s12=*/(int64_t)K, /*s13=*/(int64_t)K,
        /*ne0=*/ne10_padded, /*ne1=*/1, /*ne2=*/1, /*ne3=*/1,
        stream);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_row_q8_1_cuda failed: %s\n", tag, cudaGetErrorString(err));
        return -2;
    }
    }

    const uint64_t nblk = (uint64_t)M * (uint64_t)(K / 32);
    const uint64_t dq_bytes = (nblk * 2u + 63u) & ~63ull;
    q8_0_aligned_dense_vec_kernel<<<(unsigned)M, 32, 0, stream>>>(
        out_f32,
        (const int4 *)((const char *)W_aligned + dq_bytes),
        (const __half *)W_aligned,
        (const block_q8_1 *)x8, M, K / 32);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: kernel launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Aligned row-pair-SoA Q2_K routed-expert decode matvec (megakernel program
// M2, moe-down increment).  The production down leg runs
// mul_mat_vec_q_moe<GGML_TYPE_Q2_K, 2> over raw 84-byte block_q2_K stacks at
// ~190 GB/s (per-lane loads: one 4B qs int, four scale BYTES, one 4B half2 --
// 12 load instructions per lane-iteration).  W_aligned is a repacked copy of
// the SAME bytes keyed to that kernel's rows_per_block == 2: for the row pair
// (2p, 2p+1) of an expert, each lane-iteration needs exactly one 8B qs load
// (both rows' int), one 16B scales-window load (both rows' 8B half), one 8B
// dm load.  Layout contract (shared with the weight server
// --repack-q2k-aligned, DERIVED_Q2_K_ALIGNED_MOE, and ds4_mmq.h):
//
//   npair = n_experts * (M/2) * (K/256)      pair-blocks, expert-major then
//                                            row-pair then block (raw order)
//   [ uint2 dm2[npair] ]        {row0 half2(d,dmin), row1 half2}
//   [ pad to 64B ]
//   [ int4  sc4[npair*2] ]      half h: {row0 scales[8h..8h+3], row0 [8h+4..
//                               8h+7], row1 [8h..8h+3], row1 [8h+4..8h+7]}
//   [ pad to 64B ]
//   [ uint2 qs2[npair*16] ]     iqs: {row0 qs int[iqs], row1 qs int[iqs]}
//
// Lane mapping, scale-byte values, q8 side and the float accumulation order
// are copied verbatim from mul_mat_vec_q_moe/vec_dot_q2_K_q8_1 -> outputs are
// bit-identical to the raw path (proto_m2_q2k.cu: 240/240 parity + graph
// capture/replay, and 214 GB/s vs 154 raw on the same rotating rig).
// ---------------------------------------------------------------------------

// Same float chain as vec_dot_q2_K_q8_1_impl_mmvq; the four scale bytes come
// from the two pre-loaded 32-bit window words (byte lo+2i of the 8B window ==
// scales[scale_offset + 2i] of the raw block).
static __device__ __forceinline__ float q2_k_vec_dot_windowed(
        const int v, const int * __restrict__ u, const uint32_t w0, const uint32_t w1,
        const int lo, const half2 dm2, const float * __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        const int bidx = lo + 2*i;
        const uint32_t w = (bidx < 4) ? w0 : w1;
        const int sc = (int)((w >> ((bidx & 3) * 8)) & 0xFFu);

        const int vi = (v >> (2*i)) & 0x03030303;

        sumf_d += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * (sc & 0xF));

        int m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;
        sumf_m += d8[i] * ggml_cuda_dp4a(m, u[i], 0);
    }
    const float2 dm2f = __half22float2(dm2);
    return dm2f.x*sumf_d - dm2f.y*sumf_m;
}

// Twin of mul_mat_vec_q_moe<GGML_TYPE_Q2_K, 2> at the down-leg call shape
// (nchannels_dst == 1, ids_stride == 1): grid (M/2, 1), block (32, ncols_dst),
// warp per assignment column.  Keeps the -1 router-id guard (task #23).
__launch_bounds__(8*32, 1)   /* MMVQ_MAX_BATCH_SIZE (mmvq.cuh) * warp; not included here */
__global__ static void q2_k_aligned_moe_vec_kernel(
        const uint2 * __restrict__ dm2_soa,
        const int4  * __restrict__ sc4_soa,
        const uint2 * __restrict__ qs2_soa,
        const block_q8_1 * __restrict__ vy, const int32_t * __restrict__ ids,
        float * __restrict__ dst,
        const uint32_t ncols_x, const uint32_t nrows_x,
        const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t ncols_dst) {
    constexpr int qi  = 16;   // QI2_K
    constexpr int vdr = 1;    // VDR_Q2_K_Q8_1_MMVQ
    constexpr int warp_size = 32;

    const uint32_t token_idx = threadIdx.y;
    const int      row0      = 2*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / QK_K;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;   // 2

    if (token_idx >= ncols_dst) {
        return;
    }

    const int32_t  id_raw     = ids[token_idx];
    const bool     invalid_id = id_raw < 0;
    const uint32_t channel_x  = invalid_id ? 0u : (uint32_t)id_raw;

    const block_q8_1 * y = vy + token_idx*stride_col_y;
    const size_t pair_base = ((size_t)channel_x * (nrows_x/2u) + (size_t)blockIdx.x)
                           * (size_t)blocks_per_row_x;

    float tmp[2] = {0.0f, 0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); !invalid_id && kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (QK_K/QK8_1);
        const int iqs = vdr * (threadIdx.x % (qi/vdr));

        const int bq8_offset = QR2_K * (iqs / QI8_1);
        const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);
        const int whalf = iqs / QI8_1;
        const int lo    = scale_offset - 8*whalf;
        const block_q8_1 * bq8_1 = &y[kby];

        int    u[QR2_K];
        float d8[QR2_K];
#pragma unroll
        for (int i = 0; i < QR2_K; ++i) {
            u[i]  = get_int_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
            d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
        }

        const size_t pblk = pair_base + (size_t)kbx;
        const uint2 v2  = qs2_soa[pblk*16u + (unsigned)iqs];
        const uint2 dmw = dm2_soa[pblk];
        const int4  scw = sc4_soa[pblk*2u + (unsigned)whalf];
        const half2 dm0 = *(const half2 *)&dmw.x;
        const half2 dm1 = *(const half2 *)&dmw.y;

        tmp[0] += q2_k_vec_dot_windowed((int)v2.x, u, (uint32_t)scw.x, (uint32_t)scw.y, lo, dm0, d8);
        tmp[1] += q2_k_vec_dot_windowed((int)v2.y, u, (uint32_t)scw.z, (uint32_t)scw.w, lo, dm1, d8);
    }

#pragma unroll
    for (int i = 0; i < 2; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
    }

    if (threadIdx.x < 2 && uint32_t(row0 + threadIdx.x) < nrows_x) {
        dst[token_idx*stride_col_dst + row0 + threadIdx.x] = tmp[threadIdx.x];
    }
}

// Exact inverse of the weight-server repack (repack_q2_k_aligned_kernel,
// tools/ds4_weight_server.cu): pair-SoA -> raw block_q2_K byte stream.  One
// thread per (raw block, qs int); p < 4 additionally restores a scales word,
// p == 0 the dm word.
__global__ static void q2_k_aligned_derepack_kernel(
        unsigned char *raw_out,
        const uint2   * __restrict__ dm2_soa,
        const int4    * __restrict__ sc4_soa,
        const uint2   * __restrict__ qs2_soa,
        uint64_t nblk,       // raw blocks total
        uint32_t nb_row,     // blocks per row = K/256
        uint32_t nrows) {    // rows per expert = M
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nblk * 16ull) return;
    const uint64_t g = i >> 4;
    const uint32_t p = (uint32_t)(i & 15u);
    const uint32_t b = (uint32_t)(g % nb_row);
    const uint32_t r = (uint32_t)((g / nb_row) % nrows);
    const uint64_t e = g / ((uint64_t)nb_row * nrows);
    const uint64_t pblk = ((uint64_t)e * (nrows/2u) + r/2u) * nb_row + b;
    const uint32_t parity = r & 1u;
    unsigned char *dst = raw_out + g * 84ull;
    const uint2 q = qs2_soa[pblk*16u + p];
    const uint32_t qw = parity ? q.y : q.x;
    memcpy(dst + 16u + (uint64_t)p * 4u, &qw, 4u);
    if (p < 4u) {
        const int4 s = sc4_soa[pblk*2u + (p >> 1)];
        const uint32_t sw = parity ? ((p & 1u) ? (uint32_t)s.w : (uint32_t)s.z)
                                   : ((p & 1u) ? (uint32_t)s.y : (uint32_t)s.x);
        memcpy(dst + ((p >> 1) * 8u + (p & 1u) * 4u), &sw, 4u);
    }
    if (p == 0u) {
        const uint2 d = dm2_soa[pblk];
        const uint32_t dw = parity ? d.y : d.x;
        memcpy(dst + 80u, &dw, 4u);
    }
}

extern "C" uint64_t ds4_mmq_q2_k_aligned_bytes(int M, int K, int n_experts) {
    if (M <= 0 || K <= 0 || n_experts <= 0 || K % 256 != 0 || M % 2 != 0) return 0;
    const uint64_t npair = (uint64_t)n_experts * (uint64_t)(M/2) * (uint64_t)(K / 256);
    const uint64_t dm_bytes = (npair * 8u + 63u) & ~63ull;
    const uint64_t sc_bytes = (npair * 32u + 63u) & ~63ull;
    return dm_bytes + sc_bytes + npair * 128u;
}

extern "C" int ds4_mmq_q2_K_aligned_moe_vec(
        const void * W_aligned, const float * X_f32, const int32_t * ids,
        float * out_f32, int M, int K, int n_tokens, int n_experts,
        int n_expert_used, cudaStream_t stream) {
    const char *tag = "ds4_mmq_q2_K_aligned_moe_vec";
    if (!W_aligned || !X_f32 || !ids || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    /* Down-leg call shape only: each (token, slot) assignment arrives as its
     * own "token" with one expert (n_expert_used == 1, ids_stride == 1). */
    if (n_expert_used != 1 || n_tokens < 1 || M <= 0 || M % 2 != 0 || K <= 0 ||
        K % 256 != 0 || n_experts <= 0) {
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }
    ds4_pool_set_stream(stream);

    /* Quantize verbatim from ds4_mmq_moe_vec_impl<GGML_TYPE_Q2_K> so the q8_1
     * codes feeding the twin are bit-identical to the raw path's. */
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t  nbytes_q8_1 = (size_t)n_tokens * ne10_padded * sizeof(block_q8_1) / QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1_pool;
    char *src1_q8_1_ptr = nullptr;
    if (g_q81_scratch_enabled && g_q81_scratch_ptr && g_q81_scratch_bytes >= nbytes_q8_1) {
        src1_q8_1_ptr = (char *)g_q81_scratch_ptr;
    } else {
        src1_q8_1_pool.alloc(ctx->pool(), nbytes_q8_1);
        src1_q8_1_ptr = src1_q8_1_pool.get();
    }
    quantize_row_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1_ptr,
        GGML_TYPE_Q2_K, /*ne00=*/K,
        /*s11=*/(int64_t)K, /*s12=*/(int64_t)K, /*s13=*/(int64_t)K * n_tokens,
        /*ne0=*/ne10_padded, /*ne1=*/1, /*ne2=*/n_tokens, /*ne3=*/1,
        stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_row_q8_1_cuda failed: %s\n", tag, cudaGetErrorString(err));
        return -2;
    }

    const int64_t s12_y  = ne10_padded / QK8_1;
    const int64_t s2_dst = (int64_t)M;   /* n_expert_used == 1 */

    cudaMemsetAsync(out_f32, 0, (size_t)M * (size_t)n_tokens * sizeof(float), stream);

    const uint64_t npair = (uint64_t)n_experts * (uint64_t)(M/2) * (uint64_t)(K / 256);
    const uint64_t dm_bytes = (npair * 8u + 63u) & ~63ull;
    const uint64_t sc_bytes = (npair * 32u + 63u) & ~63ull;
    const uint2 *dm2 = (const uint2 *)W_aligned;
    const int4  *sc4 = (const int4 *)((const char *)W_aligned + dm_bytes);
    const uint2 *qs2 = (const uint2 *)((const char *)W_aligned + dm_bytes + sc_bytes);

    const int col_cap = 8;   /* MMVQ_MAX_BATCH_SIZE; matches __launch_bounds__ */
    for (int c0 = 0; c0 < n_tokens; c0 += col_cap) {
        const int ncols = (n_tokens - c0 < col_cap) ? (n_tokens - c0) : col_cap;
        dim3 grid((unsigned)(M/2), 1);
        dim3 block(32, (unsigned)ncols);
        q2_k_aligned_moe_vec_kernel<<<grid, block, 0, stream>>>(
            dm2, sc4, qs2,
            (const block_q8_1 *)(src1_q8_1_ptr + (size_t)c0 * s12_y * sizeof(block_q8_1)),
            ids + c0,
            out_f32 + (int64_t)c0 * s2_dst,
            (uint32_t)K, (uint32_t)M,
            (uint32_t)s12_y, (uint32_t)s2_dst,
            (uint32_t)ncols);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: kernel launch failed: %s (cols %d..%d)\n",
                    tag, cudaGetErrorString(err), c0, c0 + ncols - 1);
            return -3;
        }
    }

    ds4_mmq_sanitize_f32(out_f32, (uint64_t)M * (uint64_t)n_tokens, stream);
    return 0;
}

extern "C" int ds4_mmq_q2_K_aligned_derepack(
        const void * W_aligned, void * raw_out,
        int M, int K, int n_experts, cudaStream_t stream) {
    const char *tag = "ds4_mmq_q2_K_aligned_derepack";
    if (!W_aligned || !raw_out) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || M % 2 != 0 || K <= 0 || K % 256 != 0 || n_experts <= 0) return -1;
    const uint64_t nblk = (uint64_t)n_experts * (uint64_t)M * (uint64_t)(K / 256);
    const uint64_t npair = nblk / 2u;
    const uint64_t dm_bytes = (npair * 8u + 63u) & ~63ull;
    const uint64_t sc_bytes = (npair * 32u + 63u) & ~63ull;
    const uint64_t n_threads = nblk * 16ull;
    q2_k_aligned_derepack_kernel<<<(unsigned)((n_threads + 255ull) / 256ull), 256, 0, stream>>>(
        (unsigned char *)raw_out,
        (const uint2 *)W_aligned,
        (const int4 *)((const char *)W_aligned + dm_bytes),
        (const uint2 *)((const char *)W_aligned + dm_bytes + sc_bytes),
        nblk, (uint32_t)(K / 256), (uint32_t)M);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: kernel launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

extern "C" uint64_t ds4_mmq_iq2_xxs_aligned_bytes(int M, int K, int n_experts) {
    if (M <= 0 || K <= 0 || n_experts <= 0 || K % 256 != 0) return 0;
    const uint64_t nblk = (uint64_t)n_experts * (uint64_t)M * (uint64_t)(K / 256);
    const uint64_t dq_bytes = (nblk * 2u + 63u) & ~63ull;
    return dq_bytes + nblk * 64u;
}

// Shared single-token canonical-Q8_1 quantize for the aligned IQ2_XXS
// entries.  Returns the device pointer (persistent scratch when enabled,
// pool otherwise) or nullptr on failure; *pool must outlive the launches.
static char *iq2_aligned_quantize_xn(
        const char *tag, const float *X_f32, int K, int n_tokens,
        ggml_cuda_pool_alloc<char> *pool, cudaStream_t stream) {
    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return nullptr;
    }
    ds4_pool_set_stream(stream);
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t  nbytes_q8_1 = (size_t)n_tokens * ne10_padded * sizeof(block_q8_1) / QK8_1;
    // M2-Inc2a: producer-emitted q8_1 codes (ffn_norm from the fused HC
    // stage) -- take them and skip the quantize prelude.
    char *folded = ds4_mmq_folded_q81(X_f32, K, n_tokens, ne10_padded);
    if (folded) {
        // C3-Inc4 fold twin selftest (DS4_Q8_FOLD_SELFTEST=<call budget>,
        // eager legs only -- syncs the stream): the taken sidecar must be
        // byte-identical to the fresh quantize this prelude would have run.
        // Do NOT combine with DS4_HC_STAGE_BATCH_PARITY (the probe rewrites
        // norm_out after the sidecar was emitted).
        static int fold_st = -1;
        if (fold_st < 0) {
            const char *st = getenv("DS4_Q8_FOLD_SELFTEST");
            fold_st = st && *st ? atoi(st) : 0;
            if (st && *st && fold_st <= 1) fold_st = 512;
        }
        cudaStreamCaptureStatus fold_cs = cudaStreamCaptureStatusNone;
        if (fold_st > 0) (void)cudaStreamIsCapturing(stream, &fold_cs);
        if (fold_st > 0 && fold_cs == cudaStreamCaptureStatusNone &&
            nbytes_q8_1 <= 16384u) {
            fold_st--;
            static char h[2][16384];
            pool->alloc(ctx->pool(), nbytes_q8_1);
            char *fresh = pool->get();
            quantize_row_q8_1_cuda(
                X_f32, /*ids=*/nullptr, (void *)fresh,
                GGML_TYPE_IQ2_XXS, /*ne00=*/K,
                /*s11=*/(int64_t)K, /*s12=*/(int64_t)K, /*s13=*/(int64_t)K * n_tokens,
                /*ne0=*/ne10_padded, /*ne1=*/1, /*ne2=*/n_tokens, /*ne3=*/1,
                stream);
            if (cudaGetLastError() == cudaSuccess &&
                cudaStreamSynchronize(stream) == cudaSuccess &&
                cudaMemcpy(h[0], folded, nbytes_q8_1, cudaMemcpyDeviceToHost) == cudaSuccess &&
                cudaMemcpy(h[1], fresh, nbytes_q8_1, cudaMemcpyDeviceToHost) == cudaSuccess) {
                fprintf(stderr, "ds4: Q8F-SELFTEST(q81 moe) K=%d %s\n", K,
                        memcmp(h[0], h[1], nbytes_q8_1) == 0 ? "PASS" : "FAIL");
            } else {
                fprintf(stderr, "ds4: Q8F-SELFTEST(q81 moe) SKIP (setup failed)\n");
            }
        }
        return folded;
    }
    char *ptr = nullptr;
    if (g_q81_scratch_enabled && g_q81_scratch_ptr && g_q81_scratch_bytes >= nbytes_q8_1) {
        ptr = (char *)g_q81_scratch_ptr;
    } else {
        pool->alloc(ctx->pool(), nbytes_q8_1);
        ptr = pool->get();
    }
    quantize_row_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)ptr,
        GGML_TYPE_IQ2_XXS, /*ne00=*/K,
        /*s11=*/(int64_t)K, /*s12=*/(int64_t)K, /*s13=*/(int64_t)K * n_tokens,
        /*ne0=*/ne10_padded, /*ne1=*/1, /*ne2=*/n_tokens, /*ne3=*/1,
        stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_row_q8_1_cuda failed: %s\n", tag, cudaGetErrorString(err));
        return nullptr;
    }
    return ptr;
}

extern "C" int ds4_mmq_iq2_xxs_aligned_moe_pair_vec(
        const void * W_gate_aligned, const void * W_up_aligned,
        const float * X_f32, const int32_t * ids,
        float * gate_out, float * up_out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    const char *tag = "ds4_mmq_iq2_xxs_aligned_moe_pair_vec";
    if (!W_gate_aligned || !W_up_aligned || !X_f32 || !ids || !gate_out || !up_out) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (n_tokens < 1 || n_tokens > 16 || M <= 0 || K <= 0 || n_experts <= 0 ||
        n_expert_used <= 0 || n_expert_used > n_experts || K % 1024 != 0) {
        return -1;
    }
    ggml_cuda_pool_alloc<char> q8_pool;
    char *x8 = iq2_aligned_quantize_xn(tag, X_f32, K, n_tokens, &q8_pool, stream);
    if (!x8) return -2;

    const uint64_t nblk = (uint64_t)n_experts * (uint64_t)M * (uint64_t)(K / 256);
    const uint64_t dq_bytes = (nblk * 2u + 63u) & ~63ull;
    dim3 grid((unsigned)M, (unsigned)(n_tokens * n_expert_used), 2);
    iq2_xxs_aligned_moe_pair_vec_kernel<<<grid, 32, 0, stream>>>(
        gate_out, up_out,
        (const uint2 *)((const char *)W_gate_aligned + dq_bytes),
        (const __half *)W_gate_aligned,
        (const uint2 *)((const char *)W_up_aligned + dq_bytes),
        (const __half *)W_up_aligned,
        (const block_q8_1 *)x8, ids, M, K / 256, K / 32, n_expert_used);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: kernel launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

extern "C" int ds4_mmq_iq2_xxs_aligned_moe_gate_up_mid_vec(
        const void * W_gate_aligned, const void * W_up_aligned,
        const float * X_f32, const int32_t * ids, const float * weights,
        float * mid_f32,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        float clamp, cudaStream_t stream) {
    const char *tag = "ds4_mmq_iq2_xxs_aligned_moe_gate_up_mid_vec";
    if (!W_gate_aligned || !W_up_aligned || !X_f32 || !ids || !weights || !mid_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (n_tokens < 1 || n_tokens > 16 || M <= 0 || K <= 0 || n_experts <= 0 ||
        n_expert_used <= 0 || n_expert_used > n_experts || K % 1024 != 0) {
        return -1;
    }
    ggml_cuda_pool_alloc<char> q8_pool;
    char *x8 = iq2_aligned_quantize_xn(tag, X_f32, K, n_tokens, &q8_pool, stream);
    if (!x8) return -2;

    const uint64_t nblk = (uint64_t)n_experts * (uint64_t)M * (uint64_t)(K / 256);
    const uint64_t dq_bytes = (nblk * 2u + 63u) & ~63ull;
    dim3 grid((unsigned)M, (unsigned)(n_tokens * n_expert_used), 1);
    iq2_xxs_aligned_moe_gate_up_mid_kernel<<<grid, 32, 0, stream>>>(
        mid_f32,
        (const uint2 *)((const char *)W_gate_aligned + dq_bytes),
        (const __half *)W_gate_aligned,
        (const uint2 *)((const char *)W_up_aligned + dq_bytes),
        (const __half *)W_up_aligned,
        (const block_q8_1 *)x8, ids, weights, M, K / 256, K / 32, n_expert_used, clamp);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: kernel launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

extern "C" int ds4_mmq_iq2_xxs_aligned_moe_vec(
        const void * W_aligned, const float * X_f32, const int32_t * ids, float * out_f32,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    const char *tag = "ds4_mmq_iq2_xxs_aligned_moe_vec";
    if (!W_aligned || !X_f32 || !ids || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    // n_tokens 1..16 (the vec-tier envelope; each warp reads one activation
    // row selected by assignment/n_expert_used).
    // K % 1024: the lane->(block,pair) mapping covers 4 blocks per pass.
    if (n_tokens < 1 || n_tokens > 16 || M <= 0 || K <= 0 || n_experts <= 0 ||
        n_expert_used <= 0 || n_expert_used > n_experts || K % 1024 != 0) {
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }
    ds4_pool_set_stream(stream);

    // Quantize X into canonical Q8_1, exactly as ds4_mmq_moe_vec_impl does, so
    // the aligned path shares its activation numerics (and its persistent
    // scratch when enabled).
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t  nbytes_q8_1 = (size_t)n_tokens * ne10_padded * sizeof(block_q8_1) / QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1_pool;
    char *src1_q8_1_ptr = nullptr;
    if (g_q81_scratch_enabled && g_q81_scratch_ptr && g_q81_scratch_bytes >= nbytes_q8_1) {
        src1_q8_1_ptr = (char *)g_q81_scratch_ptr;
    } else {
        src1_q8_1_pool.alloc(ctx->pool(), nbytes_q8_1);
        src1_q8_1_ptr = src1_q8_1_pool.get();
    }
    quantize_row_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1_ptr,
        GGML_TYPE_IQ2_XXS, /*ne00=*/K,
        /*s11=*/(int64_t)K, /*s12=*/(int64_t)K, /*s13=*/(int64_t)K * n_tokens,
        /*ne0=*/ne10_padded, /*ne1=*/1, /*ne2=*/n_tokens, /*ne3=*/1,
        stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_row_q8_1_cuda failed: %s\n", tag, cudaGetErrorString(err));
        return -2;
    }

    const uint64_t nblk = (uint64_t)n_experts * (uint64_t)M * (uint64_t)(K / 256);
    const uint64_t dq_bytes = (nblk * 2u + 63u) & ~63ull;
    const __half *dq = (const __half *)W_aligned;
    const uint2  *qs = (const uint2 *)((const char *)W_aligned + dq_bytes);

    dim3 grid((unsigned)M, (unsigned)(n_tokens * n_expert_used), 1);
    iq2_xxs_aligned_moe_vec_kernel<<<grid, 32, 0, stream>>>(
        out_f32, qs, dq, (const block_q8_1 *)src1_q8_1_ptr, ids, M, K / 256,
        K / 32, n_expert_used);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: kernel launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }

    ds4_mmq_sanitize_f32(out_f32, (uint64_t)n_tokens * (uint64_t)M * (uint64_t)n_expert_used, stream);
    return 0;
}

extern "C" int ds4_mmq_q2_K_moe_down_sum6_vec(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_down_sum6_vec_impl<GGML_TYPE_Q2_K>(
        "ds4_mmq_q2_K_moe_down_sum6_vec", W, X, ids, out, M, K,
        n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_q4_K_moe_down_sum6_vec(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_down_sum6_vec_impl<GGML_TYPE_Q4_K>(
        "ds4_mmq_q4_K_moe_down_sum6_vec", W, X, ids, out, M, K,
        n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_iq2_xxs_moe_gate_up_mid_vec(
        const void * W_gate, const void * W_up,
        const float * X, const int32_t * ids, const float * weights, float * mid,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        float clamp, cudaStream_t stream) {
    return ds4_mmq_moe_gate_up_mid_vec_impl<GGML_TYPE_IQ2_XXS>(
        "ds4_mmq_iq2_xxs_moe_gate_up_mid_vec", W_gate, W_up, X, ids, weights, mid,
        M, K, n_tokens, n_experts, n_expert_used, clamp, stream);
}

extern "C" int ds4_mmq_q4_K_moe_gate_up_mid_vec(
        const void * W_gate, const void * W_up,
        const float * X, const int32_t * ids, const float * weights, float * mid,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        float clamp, cudaStream_t stream) {
    return ds4_mmq_moe_gate_up_mid_vec_impl<GGML_TYPE_Q4_K>(
        "ds4_mmq_q4_K_moe_gate_up_mid_vec", W_gate, W_up, X, ids, weights, mid,
        M, K, n_tokens, n_experts, n_expert_used, clamp, stream);
}

extern "C" int ds4_mmq_iq2_xxs_moe_pair_vec(
        const void * W_a, const void * W_b,
        const float * X, const int32_t * ids, float * out_silu,
        int M, int K, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_pair_vec_impl<GGML_TYPE_IQ2_XXS>(
        "ds4_mmq_iq2_xxs_moe_pair_vec", W_a, W_b, X, ids, out_silu,
        M, K, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_q4_K_moe_pair_vec(
        const void * W_a, const void * W_b,
        const float * X, const int32_t * ids, float * out_silu,
        int M, int K, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_pair_vec_impl<GGML_TYPE_Q4_K>(
        "ds4_mmq_q4_K_moe_pair_vec", W_a, W_b, X, ids, out_silu,
        M, K, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_iq2_xxs_moe_pair_raw_vec(
        const void * W_a, const void * W_b,
        const float * X, const int32_t * ids, float * out_a, float * out_b,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_pair_raw_vec_impl<GGML_TYPE_IQ2_XXS>(
        "ds4_mmq_iq2_xxs_moe_pair_raw_vec", W_a, W_b, X, ids, out_a, out_b,
        M, K, n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_q4_K_moe_pair_raw_vec(
        const void * W_a, const void * W_b,
        const float * X, const int32_t * ids, float * out_a, float * out_b,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_pair_raw_vec_impl<GGML_TYPE_Q4_K>(
        "ds4_mmq_q4_K_moe_pair_raw_vec", W_a, W_b, X, ids, out_a, out_b,
        M, K, n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_q8_0_dense_vec(
        const void * W, const float * X, float * out,
        int M, int N, int K, cudaStream_t stream) {
    return ds4_mmq_dense_vec_impl<GGML_TYPE_Q8_0>(
        "ds4_mmq_q8_0_dense_vec", W, X, out, M, N, K, stream);
}

// Explicit instantiations. One per quant type the public API exposes.
// Each instantiation drags in the load_tiles_<type> + vec_dot_<type>_*
// device functions from mmq.cuh, so the .o objects below contain everything
// needed to link against the public C entries.
template void mul_mat_q_case<GGML_TYPE_Q8_0>(
    ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
template void mul_mat_q_case<GGML_TYPE_Q2_K>(
    ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
template void mul_mat_q_case<GGML_TYPE_IQ2_XXS>(
    ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
template void mul_mat_q_case<GGML_TYPE_Q4_K>(
    ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
