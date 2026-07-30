/* GB10/GX10 DSpark CUDA modifications:
 * Copyright (c) 2026 Marco Palaferri. Licensed under the MIT License. */

#include "ds4_dspark_scheduler.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    double survival;
    uint64_t identity;
    uint32_t request;
    uint32_t prefix;
} ds4_dspark_candidate;

enum {
    DS4_DSPARK_SCHEDULER_MAX_CANDIDATES =
        DS4_DSPARK_SCHEDULER_MAX_REQUESTS *
        DS4_DSPARK_SCHEDULER_MAX_PREFIX,
};

static int ds4_dspark_candidate_compare(const void *lhs_ptr,
                                        const void *rhs_ptr) {
    const ds4_dspark_candidate *lhs = lhs_ptr;
    const ds4_dspark_candidate *rhs = rhs_ptr;
    if (lhs->survival > rhs->survival) return -1;
    if (lhs->survival < rhs->survival) return 1;
    if (lhs->identity < rhs->identity) return -1;
    if (lhs->identity > rhs->identity) return 1;
    if (lhs->request < rhs->request) return -1;
    if (lhs->request > rhs->request) return 1;
    if (lhs->prefix < rhs->prefix) return -1;
    if (lhs->prefix > rhs->prefix) return 1;
    return 0;
}

static int ds4_dspark_build_candidates(
        const ds4_dspark_schedule_request *requests,
        uint32_t request_count,
        ds4_dspark_candidate *candidates,
        uint32_t *candidate_count_out,
        uint32_t *baseline_batch_out,
        double *baseline_expected_out) {
    if (!requests || !candidates || !candidate_count_out ||
        !baseline_batch_out || !baseline_expected_out ||
        request_count == 0 ||
        request_count > DS4_DSPARK_SCHEDULER_MAX_REQUESTS) {
        return 1;
    }
    uint32_t candidate_count = 0;
    uint32_t baseline_batch = request_count;
    double baseline_expected = (double)request_count;
    for (uint32_t r = 0; r < request_count; r++) {
        if (requests[r].max_prefix > DS4_DSPARK_SCHEDULER_MAX_PREFIX ||
            requests[r].minimum_prefix > requests[r].max_prefix) {
            return 1;
        }
        baseline_batch += requests[r].minimum_prefix;
        double survival = 1.0;
        for (uint32_t j = 0; j < requests[r].max_prefix; j++) {
            double conditional = requests[r].conditional[j];
            if (!isfinite(conditional)) return 1;
            if (conditional < 0.0) conditional = 0.0;
            if (conditional > 1.0) conditional = 1.0;
            survival *= conditional;
            if (!isfinite(survival)) return 1;
            if (j < requests[r].minimum_prefix) {
                baseline_expected += survival;
                continue;
            }
            if (!(survival > 0.0)) break;
            candidates[candidate_count++] = (ds4_dspark_candidate) {
                .survival = survival,
                .identity = requests[r].identity,
                .request = r,
                .prefix = j + 1u,
            };
        }
    }
    qsort(candidates, candidate_count, sizeof(candidates[0]),
          ds4_dspark_candidate_compare);
    *candidate_count_out = candidate_count;
    *baseline_batch_out = baseline_batch;
    *baseline_expected_out = baseline_expected;
    return 0;
}

static int ds4_dspark_validate_sps(
        const double *sps, uint32_t sps_count,
        uint32_t baseline_batch, uint32_t candidate_count) {
    if (!sps) return 1;
    const uint32_t max_batch = baseline_batch + candidate_count;
    if (sps_count <= max_batch) return 1;
    for (uint32_t b = baseline_batch; b <= max_batch; b++) {
        if (!(sps[b] > 0.0) || !isfinite(sps[b])) return 1;
    }
    return 0;
}

static double ds4_dspark_schedule_sps(
        const double *sps,
        uint32_t batch_size,
        const uint32_t *prefix,
        uint32_t request_count,
        ds4_dspark_shape_sps_fn shape_sps,
        void *shape_sps_opaque) {
    if (shape_sps) {
        const double exact = shape_sps(
                prefix, request_count, batch_size, shape_sps_opaque);
        if (exact > 0.0 && isfinite(exact)) return exact;
    }
    return sps[batch_size];
}

static int ds4_dspark_hardware_schedule_impl(
        const ds4_dspark_schedule_request *requests,
        uint32_t request_count,
        const double *sps,
        uint32_t sps_count,
        ds4_dspark_shape_sps_fn shape_sps,
        void *shape_sps_opaque,
        ds4_dspark_schedule_result *result) {
    if (!requests || !sps || !result || request_count == 0 ||
        request_count > DS4_DSPARK_SCHEDULER_MAX_REQUESTS) {
        return 1;
    }
    memset(result, 0, sizeof(*result));

    ds4_dspark_candidate candidates[
        DS4_DSPARK_SCHEDULER_MAX_CANDIDATES];
    uint32_t candidate_count = 0;
    uint32_t baseline_batch = 0;
    double baseline_expected = 0.0;
    if (ds4_dspark_build_candidates(
            requests, request_count, candidates, &candidate_count,
            &baseline_batch, &baseline_expected) != 0 ||
        ds4_dspark_validate_sps(
            sps, sps_count, baseline_batch, candidate_count) != 0) {
        return 1;
    }

    uint32_t current_prefix[DS4_DSPARK_SCHEDULER_MAX_REQUESTS] = {0};
    for (uint32_t r = 0; r < request_count; r++) {
        current_prefix[r] = requests[r].minimum_prefix;
    }
    uint32_t batch_size = baseline_batch;
    double expected_tokens = baseline_expected;
    double best_throughput = expected_tokens * ds4_dspark_schedule_sps(
            sps, batch_size, current_prefix, request_count,
            shape_sps, shape_sps_opaque);

    result->request_count = request_count;
    result->batch_size = batch_size;
    result->capacity_batch_size = batch_size;
    result->expected_tokens = expected_tokens;
    result->throughput = best_throughput;
    result->baseline_throughput = best_throughput;
    result->capacity_throughput = best_throughput;
    result->stop_request = UINT32_MAX;

    for (uint32_t i = 0; i < candidate_count; i++) {
        const ds4_dspark_candidate *candidate = &candidates[i];
        if (candidate->prefix !=
            current_prefix[candidate->request] + 1u) {
            return 1;
        }
        current_prefix[candidate->request] = candidate->prefix;
        batch_size++;
        expected_tokens += candidate->survival;
        const double throughput = expected_tokens * ds4_dspark_schedule_sps(
                sps, batch_size, current_prefix, request_count,
                shape_sps, shape_sps_opaque);
        result->evaluated_candidates++;

        if (throughput > best_throughput) {
            best_throughput = throughput;
            result->batch_size = batch_size;
            result->capacity_batch_size = batch_size;
            result->admitted_candidates = i + 1u;
            result->expected_tokens = expected_tokens;
            result->throughput = throughput;
            result->capacity_throughput = throughput;
            memcpy(result->prefix, current_prefix,
                   (size_t)request_count * sizeof(result->prefix[0]));
            continue;
        }

        result->stopped = 1;
        result->stop_batch_size = batch_size;
        result->stop_request = candidate->request;
        result->stop_prefix = candidate->prefix;
        break;
    }
    return 0;
}

int ds4_dspark_hardware_schedule(
        const ds4_dspark_schedule_request *requests,
        uint32_t request_count,
        const double *sps,
        uint32_t sps_count,
        ds4_dspark_schedule_result *result) {
    return ds4_dspark_hardware_schedule_impl(
            requests, request_count, sps, sps_count,
            NULL, NULL, result);
}

static int ds4_dspark_hardware_schedule_async_impl(
        const ds4_dspark_schedule_request *capacity_requests,
        const ds4_dspark_schedule_request *current_requests,
        uint32_t request_count,
        const double *sps,
        uint32_t sps_count,
        ds4_dspark_shape_sps_fn shape_sps,
        void *shape_sps_opaque,
        ds4_dspark_schedule_result *result) {
    if (!capacity_requests || !current_requests || !sps || !result ||
        request_count == 0 ||
        request_count > DS4_DSPARK_SCHEDULER_MAX_REQUESTS) {
        return 1;
    }
    memset(result, 0, sizeof(*result));

    ds4_dspark_candidate capacity_candidates[
        DS4_DSPARK_SCHEDULER_MAX_CANDIDATES];
    ds4_dspark_candidate current_candidates[
        DS4_DSPARK_SCHEDULER_MAX_CANDIDATES];
    uint32_t capacity_count = 0;
    uint32_t current_count = 0;
    uint32_t capacity_baseline_batch = 0;
    uint32_t current_baseline_batch = 0;
    double capacity_baseline_expected = 0.0;
    double current_baseline_expected = 0.0;
    if (ds4_dspark_build_candidates(
            capacity_requests, request_count, capacity_candidates,
            &capacity_count, &capacity_baseline_batch,
            &capacity_baseline_expected) != 0 ||
        ds4_dspark_build_candidates(
            current_requests, request_count, current_candidates,
            &current_count, &current_baseline_batch,
            &current_baseline_expected) != 0 ||
        capacity_baseline_batch != current_baseline_batch) {
        return 1;
    }
    const uint32_t max_candidates =
        capacity_count > current_count ? capacity_count : current_count;
    if (ds4_dspark_validate_sps(
            sps, sps_count, current_baseline_batch, max_candidates) != 0) {
        return 1;
    }

    uint32_t capacity_admitted = 0;
    uint32_t capacity_batch = capacity_baseline_batch;
    double capacity_expected = capacity_baseline_expected;
    uint32_t capacity_prefix[DS4_DSPARK_SCHEDULER_MAX_REQUESTS] = {0};
    for (uint32_t r = 0; r < request_count; r++) {
        capacity_prefix[r] = capacity_requests[r].minimum_prefix;
    }
    double capacity_throughput = capacity_expected *
        ds4_dspark_schedule_sps(
                sps, capacity_batch, capacity_prefix, request_count,
                shape_sps, shape_sps_opaque);
    for (uint32_t i = 0; i < capacity_count; i++) {
        const ds4_dspark_candidate *candidate = &capacity_candidates[i];
        if (candidate->prefix !=
            capacity_prefix[candidate->request] + 1u) {
            return 1;
        }
        capacity_prefix[candidate->request] = candidate->prefix;
        capacity_batch++;
        capacity_expected += candidate->survival;
        const double throughput = capacity_expected *
            ds4_dspark_schedule_sps(
                    sps, capacity_batch, capacity_prefix, request_count,
                    shape_sps, shape_sps_opaque);
        if (throughput > capacity_throughput) {
            capacity_throughput = throughput;
            capacity_admitted = i + 1u;
        }
    }
    if (capacity_admitted > current_count) capacity_admitted = current_count;

    /* A current EOS/context boundary may expose fewer candidates than the
     * historical search selected. Keep the physical limit feasible and make
     * its diagnostic throughput describe that clamped limit. */
    capacity_expected = capacity_baseline_expected;
    for (uint32_t i = 0; i < capacity_admitted; i++) {
        capacity_expected += capacity_candidates[i].survival;
    }
    memset(capacity_prefix, 0, sizeof(capacity_prefix));
    for (uint32_t r = 0; r < request_count; r++) {
        capacity_prefix[r] = capacity_requests[r].minimum_prefix;
    }
    for (uint32_t i = 0; i < capacity_admitted; i++) {
        const ds4_dspark_candidate *candidate = &capacity_candidates[i];
        capacity_prefix[candidate->request] = candidate->prefix;
    }
    capacity_throughput = capacity_expected *
        ds4_dspark_schedule_sps(
                sps, capacity_baseline_batch + capacity_admitted,
                capacity_prefix, request_count,
                shape_sps, shape_sps_opaque);

    uint32_t current_prefix[DS4_DSPARK_SCHEDULER_MAX_REQUESTS] = {0};
    for (uint32_t r = 0; r < request_count; r++) {
        current_prefix[r] = current_requests[r].minimum_prefix;
    }
    double current_expected = current_baseline_expected;
    for (uint32_t i = 0; i < capacity_admitted; i++) {
        const ds4_dspark_candidate *candidate = &current_candidates[i];
        if (candidate->prefix !=
            current_prefix[candidate->request] + 1u) {
            return 1;
        }
        current_prefix[candidate->request] = candidate->prefix;
        current_expected += candidate->survival;
    }

    result->request_count = request_count;
    result->batch_size = current_baseline_batch + capacity_admitted;
    result->capacity_batch_size =
        current_baseline_batch + capacity_admitted;
    result->admitted_candidates = capacity_admitted;
    result->evaluated_candidates = capacity_count;
    result->stop_request = UINT32_MAX;
    result->expected_tokens = current_expected;
    result->throughput = current_expected * ds4_dspark_schedule_sps(
            sps, result->batch_size, current_prefix, request_count,
            shape_sps, shape_sps_opaque);
    uint32_t current_baseline_prefix[
        DS4_DSPARK_SCHEDULER_MAX_REQUESTS] = {0};
    for (uint32_t r = 0; r < request_count; r++) {
        current_baseline_prefix[r] =
            current_requests[r].minimum_prefix;
    }
    result->baseline_throughput = current_baseline_expected *
        ds4_dspark_schedule_sps(
                sps, current_baseline_batch, current_baseline_prefix,
                request_count, shape_sps, shape_sps_opaque);
    result->capacity_throughput = capacity_throughput;
    memcpy(result->prefix, current_prefix,
           (size_t)request_count * sizeof(result->prefix[0]));
    return 0;
}

int ds4_dspark_hardware_schedule_async(
        const ds4_dspark_schedule_request *capacity_requests,
        const ds4_dspark_schedule_request *current_requests,
        uint32_t request_count,
        const double *sps,
        uint32_t sps_count,
        ds4_dspark_schedule_result *result) {
    return ds4_dspark_hardware_schedule_async_impl(
            capacity_requests, current_requests, request_count,
            sps, sps_count, NULL, NULL, result);
}

void ds4_dspark_scheduler_state_reset(
        ds4_dspark_scheduler_state *state) {
    if (state) memset(state, 0, sizeof(*state));
}

void ds4_dspark_scheduler_state_reset_history(
        ds4_dspark_scheduler_state *state) {
    if (!state) return;
    const uint64_t step = state->step;
    memset(state->entries, 0, sizeof(state->entries));
    state->entry_count = 0u;
    state->step = step;
}

int ds4_dspark_should_confirm_physical(
        uint64_t profile_samples,
        double draft_seconds,
        double verify_seconds,
        double expected_tokens,
        double serial_rate,
        double required_gain,
        double *observed_rate) {
    if (observed_rate) *observed_rate = 0.0;
    if (profile_samples != 1u ||
        !(draft_seconds >= 0.0) || !isfinite(draft_seconds) ||
        !(verify_seconds > 0.0) || !isfinite(verify_seconds) ||
        !(expected_tokens > 0.0) || !isfinite(expected_tokens) ||
        !(serial_rate > 0.0) || !isfinite(serial_rate) ||
        !(required_gain >= 1.0) || !isfinite(required_gain)) {
        return 0;
    }
    const double seconds = draft_seconds + verify_seconds;
    if (!(seconds > 0.0) || !isfinite(seconds)) return 0;
    const double rate = expected_tokens / seconds;
    if (!(rate > 0.0) || !isfinite(rate)) return 0;
    if (observed_rate) *observed_rate = rate;
    return rate >= serial_rate * required_gain;
}

static ds4_dspark_scheduler_entry *ds4_dspark_scheduler_find(
        ds4_dspark_scheduler_state *state,
        uint64_t request_id) {
    if (!state) return NULL;
    for (uint32_t i = 0;
         i < DS4_DSPARK_SCHEDULER_MAX_REQUESTS;
         i++) {
        ds4_dspark_scheduler_entry *entry = &state->entries[i];
        if (entry->in_use && entry->request_id == request_id) {
            return entry;
        }
    }
    return NULL;
}

static int ds4_dspark_items_contain(
        const ds4_dspark_schedule_item *items,
        uint32_t request_count,
        uint64_t request_id) {
    for (uint32_t i = 0; i < request_count; i++) {
        if (items[i].request_id == request_id) return 1;
    }
    return 0;
}

static ds4_dspark_scheduler_entry *ds4_dspark_scheduler_allocate(
        ds4_dspark_scheduler_state *state,
        const ds4_dspark_schedule_item *items,
        uint32_t request_count,
        uint64_t request_id) {
    ds4_dspark_scheduler_entry *entry =
        ds4_dspark_scheduler_find(state, request_id);
    if (entry) return entry;

    for (uint32_t i = 0;
         i < DS4_DSPARK_SCHEDULER_MAX_REQUESTS;
         i++) {
        if (!state->entries[i].in_use) {
            entry = &state->entries[i];
            state->entry_count++;
            break;
        }
    }
    if (!entry) {
        uint64_t oldest_step = UINT64_MAX;
        for (uint32_t i = 0;
             i < DS4_DSPARK_SCHEDULER_MAX_REQUESTS;
             i++) {
            ds4_dspark_scheduler_entry *candidate = &state->entries[i];
            if (!ds4_dspark_items_contain(
                    items, request_count, candidate->request_id) &&
                candidate->last_seen_step < oldest_step) {
                oldest_step = candidate->last_seen_step;
                entry = candidate;
            }
        }
    }
    if (!entry) return NULL;

    memset(entry, 0, sizeof(*entry));
    entry->request_id = request_id;
    entry->in_use = 1;
    return entry;
}

int ds4_dspark_scheduler_state_forget(
        ds4_dspark_scheduler_state *state,
        uint64_t request_id) {
    ds4_dspark_scheduler_entry *entry =
        ds4_dspark_scheduler_find(state, request_id);
    if (!entry) return 0;
    memset(entry, 0, sizeof(*entry));
    if (state->entry_count != 0) state->entry_count--;
    return 1;
}

int ds4_dspark_hardware_schedule_step_shape(
        ds4_dspark_scheduler_state *state,
        const ds4_dspark_schedule_item *items,
        uint32_t request_count,
        const double *sps,
        uint32_t sps_count,
        ds4_dspark_shape_sps_fn shape_sps,
        void *shape_sps_opaque,
        ds4_dspark_schedule_step_result *result) {
    if (!state || !items || !sps || !result ||
        request_count == 0 ||
        request_count > DS4_DSPARK_SCHEDULER_MAX_REQUESTS) {
        return 1;
    }
    memset(result, 0, sizeof(*result));

    ds4_dspark_schedule_request
        current[DS4_DSPARK_SCHEDULER_MAX_REQUESTS];
    ds4_dspark_schedule_request
        capacity[DS4_DSPARK_SCHEDULER_MAX_REQUESTS];
    const uint64_t current_step = state->step + 1u;
    const uint64_t capacity_step =
        current_step > 2u ? current_step - 2u : 0u;
    for (uint32_t i = 0; i < request_count; i++) {
        if (items[i].request.max_prefix >
            DS4_DSPARK_SCHEDULER_MAX_PREFIX) {
            return 1;
        }
        for (uint32_t j = 0; j < i; j++) {
            if (items[j].request_id == items[i].request_id) return 1;
        }
        current[i] = items[i].request;
        current[i].identity = items[i].request_id;
        ds4_dspark_scheduler_entry *entry =
            ds4_dspark_scheduler_find(state, items[i].request_id);
        int history_slot = -1;
        if (entry && capacity_step != 0u) {
            for (uint32_t h = 0; h < entry->history_count; h++) {
                if (entry->history_step[h] == capacity_step) {
                    history_slot = (int)h;
                    break;
                }
            }
        }
        if (history_slot >= 0) {
            capacity[i] = entry->history[history_slot];
            capacity[i].identity = items[i].request_id;
            if (capacity[i].max_prefix > current[i].max_prefix) {
                capacity[i].max_prefix = current[i].max_prefix;
            }
            capacity[i].minimum_prefix = current[i].minimum_prefix;
            result->history_ready_requests++;
        } else {
            memset(&capacity[i], 0, sizeof(capacity[i]));
        }
    }

    if (ds4_dspark_hardware_schedule_impl(
            current, request_count, sps, sps_count,
            shape_sps, shape_sps_opaque,
            &result->causal) != 0) {
        return 1;
    }
    result->selected = result->causal;
    if (result->history_ready_requests == request_count) {
        if (ds4_dspark_hardware_schedule_async_impl(
                capacity, current, request_count, sps, sps_count,
                shape_sps, shape_sps_opaque,
                &result->selected) != 0) {
            return 1;
        }
        result->used_async = 1;
    }

    for (uint32_t i = 0; i < request_count; i++) {
        ds4_dspark_scheduler_entry *entry =
            ds4_dspark_scheduler_allocate(
                state, items, request_count, items[i].request_id);
        if (!entry) return 1;
        entry->history[1] = entry->history[0];
        entry->history_step[1] = entry->history_step[0];
        entry->history[0] = items[i].request;
        entry->history_step[0] = current_step;
        if (entry->history_count < 2u) entry->history_count++;
        entry->last_seen_step = current_step;
    }
    state->step = current_step;
    return 0;
}

int ds4_dspark_hardware_schedule_step(
        ds4_dspark_scheduler_state *state,
        const ds4_dspark_schedule_item *items,
        uint32_t request_count,
        const double *sps,
        uint32_t sps_count,
        ds4_dspark_schedule_step_result *result) {
    return ds4_dspark_hardware_schedule_step_shape(
            state, items, request_count, sps, sps_count,
            NULL, NULL, result);
}

int ds4_dspark_schedule_flatten(
        const ds4_dspark_schedule_result *schedule,
        const ds4_dspark_schedule_item *items,
        ds4_dspark_flatten_plan *plan) {
    if (!schedule || !items || !plan || schedule->request_count == 0 ||
        schedule->request_count > DS4_DSPARK_SCHEDULER_MAX_REQUESTS) {
        return 1;
    }
    memset(plan, 0, sizeof(*plan));
    plan->request_count = schedule->request_count;
    for (uint32_t r = 0; r < schedule->request_count; r++) {
        for (uint32_t previous = 0; previous < r; previous++) {
            if (items[previous].request_id == items[r].request_id) return 1;
        }
        const uint32_t prefix = schedule->prefix[r];
        if (items[r].request.max_prefix >
                DS4_DSPARK_SCHEDULER_MAX_PREFIX ||
            prefix > items[r].request.max_prefix) {
            return 1;
        }
        const uint32_t rows = prefix + 1u;
        if (items[r].position > UINT32_MAX - prefix) return 1;
        if (plan->row_count >
            DS4_DSPARK_SCHEDULER_MAX_ROWS - rows) {
            return 1;
        }
        plan->request_id[r] = items[r].request_id;
        plan->request_position[r] = items[r].position;
        plan->request_offset[r] = plan->row_count;
        plan->request_rows[r] = rows;
        for (uint32_t j = 0; j < rows; j++) {
            plan->row_request[plan->row_count] = r;
            plan->row_prefix[plan->row_count] = j;
            plan->row_position[plan->row_count] =
                items[r].position + j;
            plan->row_count++;
        }
    }
    if (plan->row_count != schedule->batch_size) {
        return 1;
    }
    return 0;
}

int ds4_dspark_physical_batch_build(
        const ds4_dspark_schedule_result *schedule,
        const ds4_dspark_schedule_item *items,
        const int32_t *pending_tokens,
        const int32_t *draft_tokens,
        uint32_t draft_stride,
        const uint64_t *rng_state,
        ds4_dspark_physical_batch *batch) {
    if (!schedule || !items || !pending_tokens || !draft_tokens ||
        !rng_state || !batch ||
        draft_stride < DS4_DSPARK_SCHEDULER_MAX_PREFIX) {
        return 1;
    }
    memset(batch, 0, sizeof(*batch));
    if (ds4_dspark_schedule_flatten(
            schedule, items, &batch->layout) != 0) {
        return 1;
    }
    for (uint32_t request = 0;
         request < batch->layout.request_count; request++) {
        batch->rng_state[request] = rng_state[request];
        const uint32_t offset = batch->layout.request_offset[request];
        const uint32_t rows = batch->layout.request_rows[request];
        batch->row_token[offset] = pending_tokens[request];
        for (uint32_t prefix = 1; prefix < rows; prefix++) {
            batch->row_token[offset + prefix] =
                draft_tokens[(uint64_t)request * draft_stride +
                             prefix - 1u];
        }
    }
    return 0;
}

int ds4_dspark_physical_batch_scatter(
        const ds4_dspark_physical_batch *batch,
        const uint32_t *committed_prefix,
        ds4_dspark_physical_result *result) {
    if (!batch || !committed_prefix || !result ||
        batch->layout.request_count == 0u ||
        batch->layout.request_count >
            DS4_DSPARK_SCHEDULER_MAX_REQUESTS) {
        return 1;
    }
    memset(result, 0, sizeof(*result));
    result->request_count = batch->layout.request_count;
    for (uint32_t request = 0;
         request < batch->layout.request_count; request++) {
        const uint32_t selected_prefix =
            batch->layout.request_rows[request] - 1u;
        const uint32_t committed = committed_prefix[request];
        if (committed > selected_prefix) return 1;
        result->committed_prefix[request] = committed;
        result->emitted_tokens[request] = committed + 1u;
        result->continuation_row[request] =
            batch->layout.request_offset[request] + committed;
        result->emitted_total += committed + 1u;
    }
    return 0;
}
