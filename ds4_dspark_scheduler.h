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

void ds4_dspark_scheduler_state_reset(
        ds4_dspark_scheduler_state *state);

/* Drop delayed confidence after a scheduling gap while preserving the global
 * step. The next two cohort steps therefore fall back to causal scheduling,
 * but unrelated runtime maturity/probe counters do not re-enter startup. */
void ds4_dspark_scheduler_state_reset_history(
        ds4_dspark_scheduler_state *state);

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
