/* GB10/GX10 DSpark CUDA modifications:
 * Copyright (c) 2026 Marco Palaferri. Licensed under the MIT License. */

#ifndef DS4_DSPARK_SCHEDULER_H
#define DS4_DSPARK_SCHEDULER_H

#include <stdint.h>

enum {
    DS4_DSPARK_SCHEDULER_MAX_REQUESTS = 256,
    DS4_DSPARK_SCHEDULER_MAX_PREFIX = 5,
    DS4_DSPARK_SCHEDULER_MAX_ROWS =
        DS4_DSPARK_SCHEDULER_MAX_REQUESTS *
        (DS4_DSPARK_SCHEDULER_MAX_PREFIX + 1),
    DS4_DSPARK_NIGHTJAR_MAX_CONTEXTS = 64,
    DS4_DSPARK_NIGHTJAR_MAX_ARMS = 64,
};

typedef struct {
    /* Stable key used only for deterministic tie-breaking. Direct callers may
     * leave it zero; the stateful production API fills it from request_id. */
    uint64_t identity;
    /* Hardware may make a short prefix effectively free. Such rows belong to
     * the baseline batch and remain subject to exact target verification. */
    uint32_t minimum_prefix;
    uint32_t max_prefix;
    double conditional[DS4_DSPARK_SCHEDULER_MAX_PREFIX];
} ds4_dspark_schedule_request;

typedef struct {
    uint32_t prefix[DS4_DSPARK_SCHEDULER_MAX_REQUESTS];
    uint32_t request_count;
    uint32_t batch_size;
    uint32_t capacity_batch_size;
    uint32_t admitted_candidates;
    uint32_t evaluated_candidates;
    uint32_t stop_batch_size;
    uint32_t stop_request;
    uint32_t stop_prefix;
    double expected_tokens;
    double throughput;
    double baseline_throughput;
    double capacity_throughput;
    int stopped;
} ds4_dspark_schedule_result;

typedef struct {
    uint64_t request_id;
    /* Absolute position of the pending target token. The scheduler ignores
     * this field; the flattened executor contract uses it to build per-row
     * positions without padding independent sequences together. */
    uint32_t position;
    ds4_dspark_schedule_request request;
} ds4_dspark_schedule_item;

typedef struct {
    uint64_t request_id;
    ds4_dspark_schedule_request history[2];
    uint64_t history_step[2];
    uint64_t last_seen_step;
    uint32_t history_count;
    uint32_t in_use;
} ds4_dspark_scheduler_entry;

/* Fixed-size state keeps scheduler work allocation-free on the decode path.
 * Histories are keyed by request identity, so reordering an active cohort
 * cannot associate confidence from one session with another. */
typedef struct {
    ds4_dspark_scheduler_entry
        entries[DS4_DSPARK_SCHEDULER_MAX_REQUESTS];
    uint64_t step;
    uint32_t entry_count;
} ds4_dspark_scheduler_state;

typedef struct {
    ds4_dspark_schedule_result selected;
    ds4_dspark_schedule_result causal;
    uint32_t used_async;
    uint32_t history_ready_requests;
} ds4_dspark_schedule_step_result;

/* Optional exact-shape hardware curve. Returning a non-positive or non-finite
 * value falls back to sps[batch_size]. The callback may inspect the complete
 * prefix vector, but the asynchronous scheduler invokes it on historical
 * confidence while selecting capacity, preserving the t-2 causal barrier. */
typedef double (*ds4_dspark_shape_sps_fn)(
        const uint32_t *prefix,
        uint32_t request_count,
        uint32_t batch_size,
        void *opaque);

/* Nightjar keeps one independent bin/block timeline for each serving
 * context. The DSpark integration uses an aggregate neural draft budget as
 * the arm; executor selection remains an independent hardware-shape
 * decision. */
typedef struct {
    uint32_t arm;
    double predicted_loss;
    double switch_loss;
} ds4_dspark_nightjar_candidate;

typedef struct {
    uint32_t arm;
    double mean_loss;
    double recent_loss;
    double total_latency;
    uint64_t total_tokens;
    uint64_t samples;
    uint32_t recent_samples;
    uint32_t in_use;
} ds4_dspark_nightjar_arm;

typedef struct {
    uint64_t key;
    uint64_t last_seen;
    uint64_t block_horizon;
    uint64_t observations;
    uint64_t rng;
    uint32_t block_index;
    uint32_t bin_index;
    uint32_t round_in_bin;
    uint32_t locked_arm;
    uint32_t previous_arm;
    uint32_t locked;
    uint32_t previous_valid;
    uint32_t revoked_pending;
    uint32_t in_use;
    double safe_ratio;
    ds4_dspark_nightjar_arm arm[DS4_DSPARK_NIGHTJAR_MAX_ARMS];
} ds4_dspark_nightjar_context;

typedef struct {
    ds4_dspark_nightjar_context
        context[DS4_DSPARK_NIGHTJAR_MAX_CONTEXTS];
    uint64_t clock;
} ds4_dspark_nightjar_state;

typedef struct {
    uint32_t arm;
    uint32_t exploration;
    uint32_t locked;
    uint32_t revoked;
    uint32_t block_index;
    uint32_t bin_index;
    uint32_t round_in_bin;
    uint32_t arm_samples;
    double estimated_loss;
    double exploration_probability;
} ds4_dspark_nightjar_decision;

/* Request-major physical layout consumed by a variable-prefix verifier.
 * Prefix row zero is the mandatory pending target token; rows 1..K are the
 * selected DSpark draft prefix. row_request is the marker tensor payload. */
typedef struct {
    uint64_t request_id[DS4_DSPARK_SCHEDULER_MAX_REQUESTS];
    uint32_t request_position[DS4_DSPARK_SCHEDULER_MAX_REQUESTS];
    uint32_t request_offset[DS4_DSPARK_SCHEDULER_MAX_REQUESTS];
    uint32_t request_rows[DS4_DSPARK_SCHEDULER_MAX_REQUESTS];
    uint32_t row_request[DS4_DSPARK_SCHEDULER_MAX_ROWS];
    uint32_t row_prefix[DS4_DSPARK_SCHEDULER_MAX_ROWS];
    uint32_t row_position[DS4_DSPARK_SCHEDULER_MAX_ROWS];
    uint32_t request_count;
    uint32_t row_count;
} ds4_dspark_flatten_plan;

/* Complete host-side input contract for one physical target microbatch.
 * Rows are contiguous and unpadded. KV/RNG remain request-owned; rng_state is
 * metadata for the executor and must never be shared across requests. */
typedef struct {
    ds4_dspark_flatten_plan layout;
    int32_t row_token[DS4_DSPARK_SCHEDULER_MAX_ROWS];
    uint64_t rng_state[DS4_DSPARK_SCHEDULER_MAX_REQUESTS];
} ds4_dspark_physical_batch;

typedef struct {
    uint32_t committed_prefix[DS4_DSPARK_SCHEDULER_MAX_REQUESTS];
    uint32_t emitted_tokens[DS4_DSPARK_SCHEDULER_MAX_REQUESTS];
    uint32_t continuation_row[DS4_DSPARK_SCHEDULER_MAX_REQUESTS];
    uint32_t request_count;
    uint32_t emitted_total;
} ds4_dspark_physical_result;

/* Implements Algorithm 1 from the DSpark paper. sps[b] is the profiled
 * engine step rate for a physical target batch of b rows. */
int ds4_dspark_hardware_schedule(
        const ds4_dspark_schedule_request *requests,
        uint32_t request_count,
        const double *sps,
        uint32_t sps_count,
        ds4_dspark_schedule_result *result);

/* Production adaptation from Section 5.2 of the DSpark paper. The first
 * request set must come from an earlier decode step and is used only to choose
 * a physical capacity through an unrestricted search over the jagged SPS
 * curve. That fixed capacity is then filled with the highest-survival current
 * candidates. Keeping capacity selection historical preserves causality. */
int ds4_dspark_hardware_schedule_async(
        const ds4_dspark_schedule_request *capacity_requests,
        const ds4_dspark_schedule_request *current_requests,
        uint32_t request_count,
        const double *sps,
        uint32_t sps_count,
        ds4_dspark_schedule_result *result);

/* Fill an already selected physical capacity with the highest-survival
 * current candidates. Nightjar uses this after selecting an aggregate budget
 * to compare physical and serial execution without changing that capacity. */
int ds4_dspark_hardware_schedule_fixed_capacity_shape(
        const ds4_dspark_schedule_request *requests,
        uint32_t request_count,
        uint32_t capacity_batch_size,
        const double *sps,
        uint32_t sps_count,
        ds4_dspark_shape_sps_fn shape_sps,
        void *shape_sps_opaque,
        ds4_dspark_schedule_result *result);

int ds4_dspark_hardware_schedule_fixed_capacity(
        const ds4_dspark_schedule_request *requests,
        uint32_t request_count,
        uint32_t capacity_batch_size,
        const double *sps,
        uint32_t sps_count,
        ds4_dspark_schedule_result *result);

void ds4_dspark_nightjar_state_reset(
        ds4_dspark_nightjar_state *state);

/* Warm-started ADA-BINGREEDY. Offline SPS supplies a prior for unseen arms;
 * measured token-weighted loss and a recent EWMA replace it after observation.
 * Exploration is probationary rather than bin-locked, and a degraded lock is
 * revoked before the next decision. safe_ratio bounds eligible exploration. */
int ds4_dspark_nightjar_select(
        ds4_dspark_nightjar_state *state,
        uint64_t context_key,
        const ds4_dspark_nightjar_candidate *candidates,
        uint32_t candidate_count,
        double safe_ratio,
        ds4_dspark_nightjar_decision *decision);

int ds4_dspark_nightjar_observe(
        ds4_dspark_nightjar_state *state,
        uint64_t context_key,
        uint32_t arm,
        double latency_seconds,
        uint32_t emitted_tokens);

void ds4_dspark_scheduler_state_reset(
        ds4_dspark_scheduler_state *state);

/* Drop delayed confidence after a scheduling gap while preserving the global
 * step. The next two cohort steps therefore fall back to causal scheduling,
 * but unrelated runtime maturity/probe counters do not re-enter startup. */
void ds4_dspark_scheduler_state_reset_history(
        ds4_dspark_scheduler_state *state);

/* Read the confidence snapshot that the next asynchronous scheduler step
 * would use for capacity selection. The lookup is exact t-2 and never
 * advances or mutates the scheduler timeline. */
int ds4_dspark_scheduler_peek_delayed_request(
        const ds4_dspark_scheduler_state *state,
        uint64_t request_id,
        ds4_dspark_schedule_request *request);

/* Forgetting a retired request prevents a later request that reuses its
 * identifier from inheriting stale confidence. Returns one when found. */
int ds4_dspark_scheduler_state_forget(
        ds4_dspark_scheduler_state *state,
        uint64_t request_id);

/* Production cohort scheduler from Section 5.2. Capacity comes exclusively
 * from confidence observed exactly two scheduler steps earlier, while current
 * confidence only ranks candidates inside that fixed capacity. If any active
 * request has no observation at that exact step (startup, join, or scheduling
 * gap), the whole cohort falls back to causal Algorithm 1 for this step. */
int ds4_dspark_hardware_schedule_step(
        ds4_dspark_scheduler_state *state,
        const ds4_dspark_schedule_item *items,
        uint32_t request_count,
        const double *sps,
        uint32_t sps_count,
        ds4_dspark_schedule_step_result *result);

/* Evaluate two executor curves and their independent exact-shape models
 * against the same t-2 history, then advance the shared history exactly once.
 * This keeps executor selection causal while allowing physical and serial
 * implementations to choose different prefixes. */
int ds4_dspark_hardware_schedule_step_pair_shape(
        ds4_dspark_scheduler_state *state,
        const ds4_dspark_schedule_item *items,
        uint32_t request_count,
        const double *first_sps,
        ds4_dspark_shape_sps_fn first_shape_sps,
        void *first_shape_sps_opaque,
        const double *second_sps,
        ds4_dspark_shape_sps_fn second_shape_sps,
        void *second_shape_sps_opaque,
        uint32_t sps_count,
        ds4_dspark_schedule_step_result *first_result,
        ds4_dspark_schedule_step_result *second_result);

/* Row-only compatibility wrapper around the paired shape-aware API. */
int ds4_dspark_hardware_schedule_step_pair(
        ds4_dspark_scheduler_state *state,
        const ds4_dspark_schedule_item *items,
        uint32_t request_count,
        const double *first_sps,
        const double *second_sps,
        uint32_t sps_count,
        ds4_dspark_schedule_step_result *first_result,
        ds4_dspark_schedule_step_result *second_result);

/* Shape-aware production variant. The ordinary API remains the deterministic
 * row-only reference used by existing callers and tests. */
int ds4_dspark_hardware_schedule_step_shape(
        ds4_dspark_scheduler_state *state,
        const ds4_dspark_schedule_item *items,
        uint32_t request_count,
        const double *sps,
        uint32_t sps_count,
        ds4_dspark_shape_sps_fn shape_sps,
        void *shape_sps_opaque,
        ds4_dspark_schedule_step_result *result);

/* A single exact-shape timing is not mature enough for normal exploitation,
 * but it can justify one bounded confirmation probe. observed_rate receives
 * the measured physical rate when the inputs are valid. */
int ds4_dspark_should_confirm_physical(
        uint64_t profile_samples,
        double draft_seconds,
        double verify_seconds,
        double expected_tokens,
        double serial_rate,
        double required_gain,
        double *observed_rate);

int ds4_dspark_schedule_flatten(
        const ds4_dspark_schedule_result *schedule,
        const ds4_dspark_schedule_item *items,
        ds4_dspark_flatten_plan *plan);

int ds4_dspark_physical_batch_build(
        const ds4_dspark_schedule_result *schedule,
        const ds4_dspark_schedule_item *items,
        const int32_t *pending_tokens,
        const int32_t *draft_tokens,
        uint32_t draft_stride,
        const uint64_t *rng_state,
        ds4_dspark_physical_batch *batch);

/* Scatter verifier prefix lengths without crossing request boundaries.
 * continuation_row identifies the final accepted physical row whose target
 * logits/frontier belong to that request. */
int ds4_dspark_physical_batch_scatter(
        const ds4_dspark_physical_batch *batch,
        const uint32_t *committed_prefix,
        ds4_dspark_physical_result *result);

#endif
