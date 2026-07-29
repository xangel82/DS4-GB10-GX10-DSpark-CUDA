#ifndef DS4_H
#define DS4_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "ds4_ssd.h"

/* Public engine boundary.
 *
 * The CLI and server should treat ds4_engine as the loaded model and
 * ds4_session as one mutable inference timeline.  A session owns the live KV
 * cache and logits; callers provide full token prefixes and let
 * ds4_session_sync() reuse, extend, or rebuild the graph state.  Keep this
 * header narrow so HTTP/CLI code does not depend on tensor internals. */

typedef enum {
    DS4_BACKEND_METAL,
    DS4_BACKEND_CUDA,
    DS4_BACKEND_CPU,
} ds4_backend;

typedef enum {
    DS4_THINK_NONE,
    DS4_THINK_HIGH,
    DS4_THINK_MAX,
} ds4_think_mode;

typedef enum {
    DS4_LOG_DEFAULT,
    DS4_LOG_PREFILL,
    DS4_LOG_GENERATION,
    DS4_LOG_KVCACHE,
    DS4_LOG_TOOL,
    DS4_LOG_WARNING,
    DS4_LOG_TIMING,
    DS4_LOG_OK,
    DS4_LOG_ERROR,
} ds4_log_type;

typedef struct {
    int *v;
    int len;
    int cap;
} ds4_tokens;

typedef struct {
    int id;
    float logit;
    float logprob;
} ds4_token_score;

#define DS4_DEFAULT_TEMPERATURE 1.0f
#define DS4_DEFAULT_TOP_P 1.0f
#define DS4_DEFAULT_MIN_P 0.05f

typedef struct ds4_engine ds4_engine;
typedef struct ds4_session ds4_session;

typedef void (*ds4_session_progress_fn)(void *ud, const char *event, int current, int total);
typedef bool (*ds4_session_cancel_fn)(void *ud);

#define DS4_SESSION_SYNC_INTERRUPTED 2

typedef enum {
    DS4_DISTRIBUTED_NONE = 0,
    DS4_DISTRIBUTED_COORDINATOR,
    DS4_DISTRIBUTED_WORKER,
} ds4_distributed_role;

typedef struct {
    uint32_t start;
    uint32_t end;
    bool has_output;
    bool set;
} ds4_distributed_layers;

typedef struct {
    ds4_distributed_role role;
    ds4_distributed_layers layers;
    const char *listen_host;
    int listen_port;
    const char *coordinator_host;
    int coordinator_port;
    uint32_t prefill_chunk;
    uint32_t prefill_window;
    uint32_t activation_bits;
    bool replay_check;
    bool debug;
} ds4_distributed_options;

typedef struct {
    const char *model_path;
    const char *mtp_path;
    const char *dspark_path;
    ds4_backend backend;
    int n_threads;
    uint32_t prefill_chunk;
    int mtp_draft_tokens;
    float mtp_margin;
    int dspark_draft_tokens;
    const char *directional_steering_file;
    const char *expert_profile_path;
    float directional_steering_attn;
    float directional_steering_ffn;
    int power_percent;
    uint32_t ssd_streaming_cache_experts;
    uint64_t ssd_streaming_cache_bytes;
    uint32_t ssd_streaming_preload_experts;
    uint64_t simulate_used_memory_bytes;
    bool warm_weights;
    bool quality;
    bool ssd_streaming;
    bool ssd_streaming_cold;
    bool inspect_only;
    bool load_slice;
    uint32_t load_layer_start;
    uint32_t load_layer_end;
    bool load_output;
    ds4_distributed_options distributed;
} ds4_engine_options;

typedef void (*ds4_token_emit_fn)(void *ud, int token);
typedef void (*ds4_generation_done_fn)(void *ud);

typedef struct {
    uint64_t total_bytes;
    uint64_t raw_bytes;
    uint64_t compressed_bytes;
    uint64_t scratch_bytes;
    uint32_t prefill_cap;
    uint32_t raw_cap;
    uint32_t comp_cap;
} ds4_context_memory;

typedef struct {
    uint8_t *ptr;
    uint64_t len;
    uint64_t cap;
} ds4_session_snapshot;

typedef struct {
    char *path;
    uint64_t bytes;
} ds4_session_payload_file;

int ds4_engine_open(ds4_engine **out, const ds4_engine_options *opt);
void ds4_engine_close(ds4_engine *e);
void ds4_engine_summary(ds4_engine *e);
int ds4_engine_vocab_size(ds4_engine *e);
int ds4_engine_power(ds4_engine *e);
int ds4_engine_set_power(ds4_engine *e, int power_percent);
const char *ds4_engine_model_name(ds4_engine *e);
int ds4_engine_layer_count(ds4_engine *e);
uint32_t ds4_engine_layer_compress_ratio(ds4_engine *e, uint32_t layer);
uint64_t ds4_engine_hidden_f32_values(ds4_engine *e);
/* Stable id for cache compatibility.  0 is the original Flash shape, so old
 * KV files with the previously-zero reserved byte remain Flash-compatible;
 * Pro and later shapes must use nonzero ids. */
int ds4_engine_model_id(ds4_engine *e);
const char *ds4_backend_name(ds4_backend backend);
bool ds4_think_mode_enabled(ds4_think_mode mode);
const char *ds4_think_mode_name(ds4_think_mode mode);
const char *ds4_think_max_prefix(void);
uint32_t ds4_think_max_min_context(void);
ds4_think_mode ds4_think_mode_for_context(ds4_think_mode mode, int ctx_size);
/* Uses the active model shape selected by ds4_engine_open(); call after opening
 * the GGUF so Flash/Pro dimensions are known. */
ds4_context_memory ds4_context_memory_estimate(ds4_backend backend, int ctx_size);
ds4_context_memory ds4_context_memory_estimate_with_prefill(
        ds4_backend backend,
        int ctx_size,
        uint32_t prefill_chunk);
bool ds4_log_is_tty(FILE *fp);
void ds4_log(FILE *fp, ds4_log_type type, const char *fmt, ...);
int ds4_engine_generate_argmax(ds4_engine *e, const ds4_tokens *prompt,
                               int n_predict, int ctx_size,
                               ds4_token_emit_fn emit,
                               ds4_generation_done_fn done,
                               void *emit_ud,
                               ds4_session_progress_fn progress,
                               void *progress_ud);
int ds4_engine_collect_imatrix(ds4_engine *e,
                               const char *dataset_path,
                               const char *output_path,
                               int ctx_size,
                               int max_prompts,
                               int max_tokens);
void ds4_engine_dump_tokens(ds4_engine *e, const ds4_tokens *tokens);
int ds4_dump_text_tokenization(const char *model_path, const char *text, FILE *fp);
int ds4_engine_head_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_first_token_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_full_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_prompt_test(ds4_engine *e, const ds4_tokens *prompt, int ctx_size);

void ds4_tokens_push(ds4_tokens *tv, int token);
void ds4_tokens_free(ds4_tokens *tv);
void ds4_tokens_copy(ds4_tokens *dst, const ds4_tokens *src);
bool ds4_tokens_starts_with(const ds4_tokens *tokens, const ds4_tokens *prefix);

void ds4_tokenize_text(ds4_engine *e, const char *text, ds4_tokens *out);
void ds4_tokenize_rendered_chat(ds4_engine *e, const char *text, ds4_tokens *out);
void ds4_chat_begin(ds4_engine *e, ds4_tokens *tokens);
void ds4_encode_chat_prompt(
        ds4_engine *e,
        const char *system,
        const char *prompt,
        ds4_think_mode think_mode,
        ds4_tokens *out);
void ds4_chat_append_max_effort_prefix(ds4_engine *e, ds4_tokens *tokens);
void ds4_chat_append_message(ds4_engine *e, ds4_tokens *tokens, const char *role, const char *content);
void ds4_chat_append_assistant_prefix(ds4_engine *e, ds4_tokens *tokens, ds4_think_mode think_mode);

char *ds4_token_text(ds4_engine *e, int token, size_t *len);
int ds4_token_eos(ds4_engine *e);
int ds4_token_user(ds4_engine *e);
int ds4_token_assistant(ds4_engine *e);

int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size);
/* Create an independent CUDA timeline that reuses owner's transient graph
 * arena. The lane owns its KV/compressor/DSpark state and host RNG/statistics,
 * but owner must outlive it. This is the memory-bounded prerequisite for a
 * physical multi-session verifier; shared lanes must not be evaluated
 * concurrently except through the R=n executor. */
int ds4_session_create_shared(ds4_session **out, ds4_session *owner);
void ds4_session_free(ds4_session *s);
bool ds4_session_is_shared(ds4_session *s);
uint64_t ds4_session_private_device_bytes(ds4_session *s);

typedef struct {
    ds4_session *session;
    const ds4_tokens *tokens;
    uint32_t start;
    uint32_t rows;
    uint32_t capture_prefixes;
    /* Physical tail rows used only to stabilize a CUDA launch shape. They are
     * excluded from rejection sampling and can never be committed. */
    uint32_t shadow_tail_rows;
    int *row_tops;
    /* Optional rows * vocabulary row-major target logits.  This is intended
     * for validation and CPU sampling; production code should prefer the
     * device-resident transaction output below. */
    float *row_logits;
    float *continuation_logits;
} ds4_physical_verify_request;

typedef struct ds4_physical_verify_txn ds4_physical_verify_txn;

enum {
    DS4_PHYSICAL_DSPARK_MAX_DRAFT = 5,
};

typedef struct {
    ds4_session *session;
    int pending_token;
    uint32_t proposal_count;
    float temperature;
    float min_p;
    uint64_t *rng;
    int draft_tokens[DS4_PHYSICAL_DSPARK_MAX_DRAFT];
    float confidence_logits[DS4_PHYSICAL_DSPARK_MAX_DRAFT];
    float accept_uniforms[DS4_PHYSICAL_DSPARK_MAX_DRAFT];
    float residual_uniforms[DS4_PHYSICAL_DSPARK_MAX_DRAFT];
} ds4_physical_draft_request;

typedef struct {
    ds4_session *session;
    uint64_t request_id;
    int first_token;
    int max_tokens;
    int eos_token;
    float temperature;
    int top_k;
    float top_p;
    float min_p;
    uint64_t *rng;
    int *accepted;
    int accepted_cap;
    int result;
} ds4_speculative_request;

/* Run each lane's DSpark drafter against its private history. Multi-request
 * cohorts flatten embedding, Transformer FFN and vocabulary work into one
 * physical CUDA batch; attention KV and causal Markov state remain lane-local.
 * Sampled q rows stay device-resident until verify_suffix_rn_reject(); only
 * tokens, confidence and uniforms cross to the host scheduler. RNG state is
 * request-owned and advances exactly as in the single-session path. */
int ds4_sessions_prepare_dspark_rn(
        ds4_physical_draft_request *requests,
        uint32_t request_count,
        char *err,
        size_t errlen);

/* Execute one lossless speculative cycle for independent sessions sharing the
 * same CUDA arena. Coordinator singletons remain in the physical executor so
 * an R>1 tail cannot trigger cold legacy graph capture; explicit serial mode
 * restores the established R=1 path. Compatible multi-request stochastic
 * DSpark cycles use one physical target verifier batch; unsupported policies
 * fall back to serialized R=1 evaluation without changing target sampling
 * semantics. */
int ds4_sessions_eval_speculative_sample_rn(
        ds4_speculative_request *requests,
        uint32_t request_count,
        char *err,
        size_t errlen);

/* Begin one CUDA target microbatch over independent speculative suffixes.
 * Attention and persistent KV remain session-local; dense/FFN/output work is
 * flattened request-major.  The transaction retains every pre-verify frontier
 * until finish/abort, and therefore must be resolved before any other operation
 * uses one of these sessions or their shared scratch arena. */
int ds4_sessions_verify_suffix_rn_begin(
        ds4_physical_verify_request *requests,
        uint32_t request_count,
        ds4_physical_verify_txn **out,
        char *err,
        size_t errlen);

typedef struct {
    /* Optional row-major q distributions for validation. Production leaves
     * this NULL because the lane-private DSpark drafter wrote them directly. */
    const float *draft_probabilities;
    const float *accept_uniforms;
    const float *residual_uniforms;
    float temperature;
    float min_p;
    bool block_verify;
    uint32_t committed_drafts;
    int correction_token;
} ds4_physical_rejection_request;

/* Apply lossless p/q rejection independently to every request slice while
 * target logits remain in the shared device output. This does not consume or
 * commit the transaction; callers pass 1 + committed_drafts to finish(). */
int ds4_sessions_verify_suffix_rn_reject(
        ds4_physical_verify_txn *txn,
        ds4_physical_rejection_request *rejection,
        char *err,
        size_t errlen);

/* Commit keep_rows[r] verified input rows independently for every lane.
 * A zero restores that lane, a partial prefix restores its rejected ring tail
 * and commits the captured compressor frontier, and the full row count keeps
 * the already-computed state.  Selected continuation logits are materialized
 * directly from the shared device output.  The transaction is consumed. */
int ds4_sessions_verify_suffix_rn_finish(
        ds4_physical_verify_txn *txn,
        const uint32_t *keep_rows,
        char *err,
        size_t errlen);

/* Restore all request frontiers and consume the transaction. */
int ds4_sessions_verify_suffix_rn_abort(
        ds4_physical_verify_txn *txn,
        char *err,
        size_t errlen);

/* Evaluate independent speculative suffixes as an observational CUDA target
 * microbatch. row_tops receives rows-1 target ids, row_logits optionally
 * receives every target distribution, and continuation_logits receives the
 * final distribution. Every frontier is restored before return. */
int ds4_sessions_verify_suffix_rn(
        ds4_physical_verify_request *requests,
        uint32_t request_count,
        char *err,
        size_t errlen);
int ds4_session_power(ds4_session *s);
int ds4_session_set_power(ds4_session *s, int power_percent);
bool ds4_session_is_distributed(ds4_session *s);
void ds4_session_set_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud);
/* UI-only progress. It may report fine-grained progress inside a prefill chunk;
 * callers must not treat it as a durable KV checkpoint boundary. */
void ds4_session_set_display_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud);
/* Optional cooperative cancellation.  ds4_session_sync() checks it only at
 * safe boundaries where the live checkpoint is either unchanged or represents a
 * valid token prefix, and returns DS4_SESSION_SYNC_INTERRUPTED when it stops. */
void ds4_session_set_cancel(ds4_session *s, ds4_session_cancel_fn fn, void *ud);
void ds4_session_report_progress(ds4_session *s, const char *event, int current, int total);
/* Distributed coordinator sessions return 1 when the full layer route is
 * available, 0 when it is still incomplete, and -1 for a local API error. */
int ds4_session_distributed_route_ready(ds4_session *s, char *err, size_t errlen);

typedef enum {
    DS4_SESSION_REWRITE_ERROR = -1,
    DS4_SESSION_REWRITE_OK = 0,
    /* The live backend state cannot be rewritten safely in place.  The caller should
     * restore an older checkpoint if it has one, then sync to the prompt. */
    DS4_SESSION_REWRITE_REBUILD_NEEDED = 1,
} ds4_session_rewrite_result;

/* Synchronize the live session to a full prompt token prefix.  If the current
 * checkpoint is a prefix, only the suffix is evaluated; otherwise the backend
 * state is refilled from scratch. */
int ds4_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen);
bool ds4_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common);
ds4_session_rewrite_result ds4_session_rewrite_from_common(
        ds4_session *s, const ds4_tokens *prompt, int common,
        char *err, size_t errlen);
/* Keep one bounded accelerator-resident prompt frontier for tool-call
 * canonicalization.  Restore returns 1 when the saved frontier matched and was
 * restored, 0 when no matching frontier exists, and -1 on a backend error.
 * A NULL prompt lets a caller that already proved the rendered-text prefix
 * restore the saved exact tokenization. */
int ds4_session_frontier_capture(ds4_session *s, uint64_t *snapshot_id,
                                 char *err, size_t errlen);
int ds4_session_frontier_restore(ds4_session *s, const ds4_tokens *prompt,
                                 uint64_t snapshot_id,
                                 char *err, size_t errlen);
int ds4_session_common_prefix(ds4_session *s, const ds4_tokens *prompt);
int ds4_session_argmax(ds4_session *s);
int ds4_session_argmax_excluding(ds4_session *s, int excluded_id);
int ds4_sample_logits(const float *logits, int n_vocab, float temperature,
                      int top_k, float top_p, float min_p, uint64_t *rng);
int ds4_speculative_rejection_sample(const float *target_logits,
                                     const float *draft_probs,
                                     int n_vocab,
                                     float temperature,
                                     float min_p,
                                     int draft_token,
                                     uint64_t *rng,
                                     bool *accepted);
int ds4_speculative_rejection_self_test(void);
int ds4_session_sample(ds4_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng);
int ds4_session_top_logprobs(ds4_session *s, ds4_token_score *out, int k);
int ds4_session_token_logprob(ds4_session *s, int token, ds4_token_score *out);
int ds4_session_copy_logits(ds4_session *s, float *out, int cap);
int ds4_session_set_logits(ds4_session *s, const float *logits, int n);
int ds4_session_materialize_logits(ds4_session *s, char *err, size_t errlen);
int ds4_session_eval(ds4_session *s, int token, char *err, size_t errlen);
int ds4_session_eval_greedy(ds4_session *s, int token, int *next_token,
                            char *err, size_t errlen);
int ds4_session_eval_speculative_argmax(ds4_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen);
int ds4_session_eval_speculative_sample(ds4_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        float temperature, int top_k,
                                        float top_p, float min_p,
                                        uint64_t *rng,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen);
void ds4_session_invalidate(ds4_session *s);
void ds4_session_rewind(ds4_session *s, int pos);
int ds4_session_pos(ds4_session *s);
int ds4_session_ctx(ds4_session *s);
int ds4_session_prefill_cap(ds4_session *s);
int ds4_engine_routed_quant_bits(ds4_engine *e);
bool ds4_engine_has_output_head(ds4_engine *e);
bool ds4_engine_has_mtp(ds4_engine *e);
int ds4_engine_mtp_draft_tokens(ds4_engine *e);
bool ds4_engine_has_dspark(ds4_engine *e);
int ds4_engine_dspark_draft_tokens(ds4_engine *e);
int ds4_engine_spec_draft_tokens(ds4_engine *e);
const ds4_tokens *ds4_session_tokens(ds4_session *s);

/* Low-level graph slice entry points used by distributed inference.  The
 * transport/session routing logic lives in ds4_distributed.c. */
int ds4_session_layer_slice_reset(ds4_session *s, char *err, size_t errlen);
int ds4_session_eval_layer_slice(ds4_session *s,
                                 const int *tokens,
                                 uint32_t n_tokens,
                                 uint32_t pos0,
                                 uint32_t layer_start,
                                 uint32_t layer_end,
                                 const float *input_hc,
                                 float *output_hc,
                                 bool output_logits,
                                 float *logits,
                                 char *err,
                                 size_t errlen);
int ds4_session_eval_output_head_from_hc(ds4_session *s,
                                         const float *hidden_hc,
                                         uint32_t n_tokens,
                                         float *logits,
                                         char *err,
                                         size_t errlen);

/* Disk KV payload helpers.  HTTP/agent code owns the outer file header and
 * persistence policy; the engine owns the DS4-specific serialized graph state. */
#define DS4_SESSION_PAYLOAD_MAGIC UINT32_C(0x34565344) /* "DSV4" */
#define DS4_SESSION_PAYLOAD_VERSION UINT32_C(4)
#define DS4_SESSION_PAYLOAD_VERSION_F32_INDEXER UINT32_C(3)
#define DS4_SESSION_PAYLOAD_U32_FIELDS 13u
#define DS4_SESSION_INDEXER_PACKED_FLAG UINT32_C(0x80000000)
#define DS4_SESSION_INDEXER_PACKED_ROW_BYTES UINT32_C(68)
#define DS4_SESSION_DSPARK_MAGIC UINT32_C(0x4b505344) /* "DSPK" */
#define DS4_SESSION_DSPARK_VERSION UINT32_C(1)
#define DS4_SESSION_DSPARK_U32_FIELDS 6u
#define DS4_SESSION_LAYER_PAYLOAD_MAGIC UINT32_C(0x4c565344) /* "DSVL" */
#define DS4_SESSION_LAYER_PAYLOAD_VERSION UINT32_C(2)
#define DS4_SESSION_LAYER_PAYLOAD_VERSION_F32_INDEXER UINT32_C(1)
#define DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS 14u

uint64_t ds4_session_payload_bytes(ds4_session *s);
int ds4_session_stage_payload(ds4_session *s, ds4_session_payload_file *out,
                              char *err, size_t errlen);
int ds4_session_write_staged_payload(const ds4_session_payload_file *payload,
                                     FILE *fp, char *err, size_t errlen);
void ds4_session_payload_file_free(ds4_session_payload_file *payload);
int ds4_session_save_payload(ds4_session *s, FILE *fp, char *err, size_t errlen);
int ds4_session_load_payload(ds4_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen);
int ds4_session_save_snapshot(ds4_session *s, ds4_session_snapshot *snap, char *err, size_t errlen);
int ds4_session_load_snapshot(ds4_session *s, const ds4_session_snapshot *snap, char *err, size_t errlen);
void ds4_session_snapshot_free(ds4_session_snapshot *snap);

uint64_t ds4_session_layer_payload_bytes(ds4_session *s,
                                         uint32_t layer_start,
                                         uint32_t layer_end);
int ds4_session_save_layer_payload(ds4_session *s, FILE *fp,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen);
int ds4_session_load_layer_payload(ds4_session *s, FILE *fp,
                                   uint64_t payload_bytes,
                                   const int *tokens, uint32_t n_tokens,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen);

#endif
