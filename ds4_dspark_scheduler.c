/* GB10/GX10 DSpark CUDA modifications:
 * Copyright (c) 2026 Marco Palaferri. Licensed under the MIT License. */

#include "ds4_dspark_scheduler.h"

#include <float.h>
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

int ds4_dspark_hardware_schedule_fixed_capacity_shape(
        const ds4_dspark_schedule_request *requests,
        uint32_t request_count,
        uint32_t capacity_batch_size,
        const double *sps,
        uint32_t sps_count,
        ds4_dspark_shape_sps_fn shape_sps,
        void *shape_sps_opaque,
        ds4_dspark_schedule_result *result) {
    if (!requests || !sps || !result || request_count == 0u ||
        request_count > DS4_DSPARK_SCHEDULER_MAX_REQUESTS) {
        return 1;
    }
    memset(result, 0, sizeof(*result));

    ds4_dspark_candidate candidates[
        DS4_DSPARK_SCHEDULER_MAX_CANDIDATES];
    uint32_t candidate_count = 0u;
    uint32_t baseline_batch = 0u;
    double baseline_expected = 0.0;
    if (ds4_dspark_build_candidates(
            requests, request_count, candidates, &candidate_count,
            &baseline_batch, &baseline_expected) != 0 ||
        ds4_dspark_validate_sps(
            sps, sps_count, baseline_batch, candidate_count) != 0 ||
        capacity_batch_size < baseline_batch ||
        capacity_batch_size > baseline_batch + candidate_count) {
        return 1;
    }

    uint32_t prefix[DS4_DSPARK_SCHEDULER_MAX_REQUESTS] = {0};
    for (uint32_t r = 0; r < request_count; r++) {
        prefix[r] = requests[r].minimum_prefix;
    }
    const uint32_t admitted = capacity_batch_size - baseline_batch;
    double expected = baseline_expected;
    for (uint32_t i = 0; i < admitted; i++) {
        const ds4_dspark_candidate *candidate = &candidates[i];
        if (candidate->prefix != prefix[candidate->request] + 1u) {
            return 1;
        }
        prefix[candidate->request] = candidate->prefix;
        expected += candidate->survival;
    }

    uint32_t baseline_prefix[DS4_DSPARK_SCHEDULER_MAX_REQUESTS] = {0};
    for (uint32_t r = 0; r < request_count; r++) {
        baseline_prefix[r] = requests[r].minimum_prefix;
    }
    result->request_count = request_count;
    result->batch_size = capacity_batch_size;
    result->capacity_batch_size = capacity_batch_size;
    result->admitted_candidates = admitted;
    result->evaluated_candidates = candidate_count;
    result->stop_request = UINT32_MAX;
    result->expected_tokens = expected;
    result->throughput = expected * ds4_dspark_schedule_sps(
            sps, capacity_batch_size, prefix, request_count,
            shape_sps, shape_sps_opaque);
    result->baseline_throughput = baseline_expected *
        ds4_dspark_schedule_sps(
                sps, baseline_batch, baseline_prefix, request_count,
                shape_sps, shape_sps_opaque);
    result->capacity_throughput = result->throughput;
    memcpy(result->prefix, prefix,
           (size_t)request_count * sizeof(result->prefix[0]));
    return 0;
}

int ds4_dspark_hardware_schedule_fixed_capacity(
        const ds4_dspark_schedule_request *requests,
        uint32_t request_count,
        uint32_t capacity_batch_size,
        const double *sps,
        uint32_t sps_count,
        ds4_dspark_schedule_result *result) {
    return ds4_dspark_hardware_schedule_fixed_capacity_shape(
            requests, request_count, capacity_batch_size,
            sps, sps_count, NULL, NULL, result);
}

void ds4_dspark_nightjar_state_reset(
        ds4_dspark_nightjar_state *state) {
    if (state) memset(state, 0, sizeof(*state));
}

static uint64_t ds4_dspark_nightjar_random(uint64_t *state) {
    uint64_t x = *state;
    if (x == 0u) x = UINT64_C(0x9e3779b97f4a7c15);
    x ^= x >> 12u;
    x ^= x << 25u;
    x ^= x >> 27u;
    *state = x;
    return x * UINT64_C(2685821657736338717);
}

static ds4_dspark_nightjar_context *ds4_dspark_nightjar_context_get(
        ds4_dspark_nightjar_state *state,
        uint64_t context_key,
        int create) {
    if (!state) return NULL;
    ds4_dspark_nightjar_context *free_context = NULL;
    ds4_dspark_nightjar_context *oldest = NULL;
    for (uint32_t i = 0; i < DS4_DSPARK_NIGHTJAR_MAX_CONTEXTS; i++) {
        ds4_dspark_nightjar_context *context = &state->context[i];
        if (context->in_use && context->key == context_key) {
            context->last_seen = ++state->clock;
            return context;
        }
        if (!context->in_use && !free_context) free_context = context;
        if (context->in_use &&
            (!oldest || context->last_seen < oldest->last_seen)) {
            oldest = context;
        }
    }
    if (!create) return NULL;
    ds4_dspark_nightjar_context *context =
        free_context ? free_context : oldest;
    if (!context) return NULL;
    memset(context, 0, sizeof(*context));
    context->in_use = 1u;
    context->key = context_key;
    context->last_seen = ++state->clock;
    context->block_index = 1u;
    context->block_horizon = 1u;
    context->bin_index = 1u;
    context->rng = context_key ^ UINT64_C(0xd1b54a32d192ed03);
    if (context->rng == 0u) context->rng = 1u;
    return context;
}

static ds4_dspark_nightjar_arm *ds4_dspark_nightjar_arm_get(
        ds4_dspark_nightjar_context *context,
        uint32_t arm,
        int create) {
    if (!context) return NULL;
    ds4_dspark_nightjar_arm *free_arm = NULL;
    for (uint32_t i = 0; i < DS4_DSPARK_NIGHTJAR_MAX_ARMS; i++) {
        ds4_dspark_nightjar_arm *slot = &context->arm[i];
        if (slot->in_use && slot->arm == arm) return slot;
        if (!slot->in_use && !free_arm) free_arm = slot;
    }
    if (!create || !free_arm) return NULL;
    memset(free_arm, 0, sizeof(*free_arm));
    free_arm->in_use = 1u;
    free_arm->arm = arm;
    return free_arm;
}

static int ds4_dspark_nightjar_arm_available(
        const ds4_dspark_nightjar_context *context,
        const ds4_dspark_nightjar_arm *arm) {
    return context && arm &&
        (arm->reject_until == 0u ||
         context->observations >= arm->reject_until);
}

int ds4_dspark_nightjar_seed_arm(
        ds4_dspark_nightjar_state *state,
        uint64_t context_key,
        uint32_t arm,
        double mean_loss,
        double recent_loss,
        uint64_t samples,
        uint32_t recent_samples) {
    if (!state || !(mean_loss > 0.0) || !isfinite(mean_loss) ||
        !(recent_loss > 0.0) || !isfinite(recent_loss) ||
        samples == 0u || recent_samples == 0u) {
        return 1;
    }
    ds4_dspark_nightjar_context *context =
        ds4_dspark_nightjar_context_get(state, context_key, 1);
    if (!context) return 1;
    ds4_dspark_nightjar_arm *slot =
        ds4_dspark_nightjar_arm_get(context, arm, 1);
    if (!slot) return 1;

    if (context->observations >= slot->samples) {
        context->observations -= slot->samples;
    } else {
        context->observations = 0u;
    }
    const uint64_t effective_samples = samples > 8u ? 8u : samples;
    uint32_t effective_recent = recent_samples > 4u ? 4u : recent_samples;
    if ((uint64_t)effective_recent > effective_samples) {
        effective_recent = (uint32_t)effective_samples;
    }
    slot->samples = effective_samples;
    slot->total_tokens = effective_samples;
    slot->total_latency = mean_loss * (double)effective_samples;
    slot->mean_loss = mean_loss;
    slot->recent_loss = recent_loss;
    slot->recent_samples = effective_recent;
    slot->reject_until = 0u;
    slot->reject_streak = 0u;
    if (UINT64_MAX - context->observations < effective_samples) {
        context->observations = UINT64_MAX;
    } else {
        context->observations += effective_samples;
    }
    /* Never persist a bin lock. It is cheap to recover, but unsafe to carry
     * across a restart where clocks, contention and request mix may differ. */
    context->locked = 0u;
    context->previous_valid = 0u;
    context->revoked_pending = 0u;
    return 0;
}

static uint32_t ds4_dspark_nightjar_bin_span(uint64_t horizon) {
    if (horizon == 0u) return 1u;
    const double root = sqrt((double)horizon);
    uint64_t span = (uint64_t)ceil(root);
    if (span == 0u) span = 1u;
    if (span > UINT32_MAX) span = UINT32_MAX;
    return (uint32_t)span;
}

static int ds4_dspark_nightjar_candidate_index(
        const ds4_dspark_nightjar_candidate *candidates,
        uint32_t candidate_count,
        uint32_t arm) {
    for (uint32_t i = 0; i < candidate_count; i++) {
        if (candidates[i].arm == arm) return (int)i;
    }
    return -1;
}

static double ds4_dspark_nightjar_arm_loss(
        const ds4_dspark_nightjar_arm *arm,
        double predicted_loss) {
    if (!arm || arm->samples == 0u) return predicted_loss;
    double observed = arm->mean_loss;
    if (arm->recent_samples < 2u || !(arm->recent_loss > 0.0)) {
        return predicted_loss > 0.0
            ? observed * 0.65 + predicted_loss * 0.35
            : observed;
    }
    observed = arm->mean_loss * 0.35 + arm->recent_loss * 0.65;
    if (!(predicted_loss > 0.0)) return observed;
    /* The offline SPS/confidence model is a cold-start prior. Once an arm has
     * enough real token-weighted observations, retaining a large prior weight
     * can pin the scheduler to a locally inferior budget after the workload
     * changes. Keep a small stabilizing contribution without overruling the
     * measured recent loss. */
    const double predicted_weight = arm->samples >= 4u ? 0.10 : 0.25;
    return observed * (1.0 - predicted_weight) +
           predicted_loss * predicted_weight;
}

int ds4_dspark_nightjar_select(
        ds4_dspark_nightjar_state *state,
        uint64_t context_key,
        const ds4_dspark_nightjar_candidate *candidates,
        uint32_t candidate_count,
        double safe_ratio,
        ds4_dspark_nightjar_decision *decision) {
    if (!state || !candidates || !decision || candidate_count == 0u ||
        candidate_count > DS4_DSPARK_NIGHTJAR_MAX_ARMS ||
        !(safe_ratio >= 1.0) || !isfinite(safe_ratio)) {
        return 1;
    }
    memset(decision, 0, sizeof(*decision));
    for (uint32_t i = 0; i < candidate_count; i++) {
        if (!(candidates[i].predicted_loss > 0.0) ||
            !isfinite(candidates[i].predicted_loss) ||
            !(candidates[i].switch_loss >= 0.0) ||
            !isfinite(candidates[i].switch_loss)) {
            return 1;
        }
        for (uint32_t j = 0; j < i; j++) {
            if (candidates[j].arm == candidates[i].arm) return 1;
        }
    }

    ds4_dspark_nightjar_context *context =
        ds4_dspark_nightjar_context_get(state, context_key, 1);
    if (!context) return 1;
    context->safe_ratio = safe_ratio;

    uint32_t available_arms = 0u;
    for (uint32_t i = 0; i < candidate_count; i++) {
        ds4_dspark_nightjar_arm *slot =
            ds4_dspark_nightjar_arm_get(
                    context, candidates[i].arm, 1);
        if (!slot) return 1;
        if (ds4_dspark_nightjar_arm_available(context, slot)) {
            available_arms++;
        }
    }
#define DS4_NIGHTJAR_ARM_SKIP(slot_) \
    (available_arms != 0u && \
     !ds4_dspark_nightjar_arm_available(context, (slot_)))

    int selected_index = -1;
    int bootstrap_index = -1;
    double bootstrap_loss = DBL_MAX;
    if (!context->locked && context->observations == 0u) {
        double best_current = DBL_MAX;
        for (uint32_t i = 0; i < candidate_count; i++) {
            ds4_dspark_nightjar_arm *slot =
                ds4_dspark_nightjar_arm_get(
                        context, candidates[i].arm, 1);
            if (!slot) return 1;
            if (DS4_NIGHTJAR_ARM_SKIP(slot)) continue;
            double loss = ds4_dspark_nightjar_arm_loss(
                    slot, candidates[i].predicted_loss);
            if (context->previous_valid &&
                context->previous_arm == 0u &&
                candidates[i].arm != 0u) {
                loss += candidates[i].switch_loss;
            }
            if (loss < best_current) best_current = loss;
        }
        const double bootstrap_limit = best_current * safe_ratio;
        for (uint32_t i = 0; i < candidate_count; i++) {
            ds4_dspark_nightjar_arm *slot =
                ds4_dspark_nightjar_arm_get(
                        context, candidates[i].arm, 1);
            if (!slot) return 1;
            if (DS4_NIGHTJAR_ARM_SKIP(slot)) continue;
            if (slot->samples != 0u) continue;
            double loss = candidates[i].predicted_loss;
            if (context->previous_valid &&
                context->previous_arm == 0u &&
                candidates[i].arm != 0u) {
                loss += candidates[i].switch_loss;
            }
            if (loss > bootstrap_limit) continue;
            if (loss < bootstrap_loss ||
                (loss == bootstrap_loss &&
                 (bootstrap_index < 0 ||
                  candidates[i].arm <
                      candidates[bootstrap_index].arm))) {
                bootstrap_loss = loss;
                bootstrap_index = (int)i;
            }
        }
    }
    int locked_index = context->locked
        ? ds4_dspark_nightjar_candidate_index(
                candidates, candidate_count, context->locked_arm)
        : -1;
    if (locked_index >= 0) {
        ds4_dspark_nightjar_arm *locked_slot =
            ds4_dspark_nightjar_arm_get(
                    context, candidates[locked_index].arm, 0);
        if (!locked_slot || DS4_NIGHTJAR_ARM_SKIP(locked_slot)) {
            locked_index = -1;
            context->locked = 0u;
        }
    }
    const uint32_t reused_lock = locked_index >= 0 ? 1u : 0u;
    uint32_t exploration = 0u;
    double exploration_probability = 0.0;
    if (bootstrap_index >= 0) {
        selected_index = bootstrap_index;
        exploration = context->observations != 0u ? 1u : 0u;
        exploration_probability = exploration ? 1.0 : 0.0;
        context->locked = 0u;
    } else if (locked_index >= 0) {
        selected_index = locked_index;
    } else {
        context->locked = 0u;
        exploration_probability =
            1.0 / (double)(context->bin_index ? context->bin_index : 1u);
        if (exploration_probability > 0.10) {
            exploration_probability = 0.10;
        }
        if (context->observations != 0u) {
            const double unit =
                (double)(ds4_dspark_nightjar_random(&context->rng) >> 11u) *
                (1.0 / 9007199254740992.0);
            exploration = unit < exploration_probability ? 1u : 0u;
        }

        if (exploration) {
            uint32_t safe[DS4_DSPARK_NIGHTJAR_MAX_ARMS];
            uint32_t safe_count = 0u;
            double current_loss[DS4_DSPARK_NIGHTJAR_MAX_ARMS];
            double best_current = DBL_MAX;
            int best_current_index = -1;
            for (uint32_t i = 0; i < candidate_count; i++) {
                ds4_dspark_nightjar_arm *slot =
                    ds4_dspark_nightjar_arm_get(
                            context, candidates[i].arm, 1);
                if (!slot) return 1;
                if (DS4_NIGHTJAR_ARM_SKIP(slot)) {
                    current_loss[i] = DBL_MAX;
                    continue;
                }
                double loss = ds4_dspark_nightjar_arm_loss(
                        slot, candidates[i].predicted_loss);
                if (context->previous_valid &&
                    context->previous_arm == 0u &&
                    candidates[i].arm != 0u) {
                    loss += candidates[i].switch_loss;
                }
                current_loss[i] = loss;
                if (loss < best_current ||
                    (loss == best_current &&
                     (best_current_index < 0 ||
                      candidates[i].arm <
                          candidates[best_current_index].arm))) {
                    best_current = loss;
                    best_current_index = (int)i;
                }
            }
            if (best_current_index < 0) return 1;
            const double limit = best_current * safe_ratio;
            for (uint32_t i = 0; i < candidate_count; i++) {
                if (current_loss[i] < DBL_MAX &&
                    current_loss[i] <= limit) {
                    safe[safe_count++] = i;
                }
            }
            if (safe_count == 0u) return 1;
            const uint32_t best_arm =
                candidates[best_current_index].arm;
            uint32_t best_distance = UINT32_MAX;
            uint64_t fewest_samples = UINT64_MAX;
            double neighbor_loss = DBL_MAX;
            for (uint32_t i = 0; i < safe_count; i++) {
                const uint32_t index = safe[i];
                const uint32_t arm = candidates[index].arm;
                if ((int)index == best_current_index) continue;
                const uint32_t distance = arm > best_arm
                    ? arm - best_arm : best_arm - arm;
                ds4_dspark_nightjar_arm *slot =
                    ds4_dspark_nightjar_arm_get(context, arm, 1);
                if (!slot) return 1;
                if (distance < best_distance ||
                    (distance == best_distance &&
                     slot->samples < fewest_samples) ||
                    (distance == best_distance &&
                     slot->samples == fewest_samples &&
                     current_loss[index] < neighbor_loss)) {
                    selected_index = (int)index;
                    best_distance = distance;
                    fewest_samples = slot->samples;
                    neighbor_loss = current_loss[index];
                }
            }
            if (selected_index < 0) {
                selected_index = best_current_index;
                exploration = 0u;
            }
        } else {
            double best_loss = DBL_MAX;
            uint64_t best_samples = 0u;
            for (uint32_t i = 0; i < candidate_count; i++) {
                ds4_dspark_nightjar_arm *slot =
                    ds4_dspark_nightjar_arm_get(
                            context, candidates[i].arm, 1);
                if (!slot) return 1;
                if (DS4_NIGHTJAR_ARM_SKIP(slot)) continue;
                double loss = ds4_dspark_nightjar_arm_loss(
                        slot, candidates[i].predicted_loss);
                if (context->previous_valid &&
                    context->previous_arm == 0u &&
                    candidates[i].arm != 0u) {
                    loss += candidates[i].switch_loss;
                }
                if (loss < best_loss ||
                    (loss == best_loss && slot->samples > best_samples) ||
                    (loss == best_loss && slot->samples == best_samples &&
                     (selected_index < 0 ||
                      candidates[i].arm <
                          candidates[selected_index].arm))) {
                    selected_index = (int)i;
                    best_loss = loss;
                    best_samples = slot->samples;
                }
            }
        }
        if (selected_index < 0) return 1;
        context->locked_arm = candidates[selected_index].arm;
    }

    ds4_dspark_nightjar_arm *selected =
        ds4_dspark_nightjar_arm_get(
                context, candidates[selected_index].arm, 1);
    if (!selected) return 1;
    if (context->observations != 0u && selected->samples == 0u) {
        exploration = 1u;
    }
    /* A new arm gets two real observations before it can own a bin, whether
     * it arrived through random exploration or because its prior became the
     * best prediction after a phase change. */
    context->locked = selected->samples >= 2u && !exploration;
    double estimated = ds4_dspark_nightjar_arm_loss(
            selected, candidates[selected_index].predicted_loss);
    if (context->previous_valid &&
        context->previous_arm == 0u &&
        candidates[selected_index].arm != 0u) {
        estimated += candidates[selected_index].switch_loss;
    }
    decision->arm = candidates[selected_index].arm;
    decision->exploration = exploration;
    decision->locked = reused_lock;
    decision->revoked = context->revoked_pending;
    context->revoked_pending = 0u;
    decision->block_index = context->block_index;
    decision->bin_index = context->bin_index;
    decision->round_in_bin = context->round_in_bin + 1u;
    decision->arm_samples = selected->samples > UINT32_MAX
        ? UINT32_MAX : (uint32_t)selected->samples;
    decision->estimated_loss = estimated;
    decision->reference_loss = DBL_MAX;
    for (uint32_t i = 0; i < candidate_count; i++) {
        ds4_dspark_nightjar_arm *slot =
            ds4_dspark_nightjar_arm_get(
                    context, candidates[i].arm, 1);
        if (!slot) return 1;
        if (DS4_NIGHTJAR_ARM_SKIP(slot)) continue;
        double loss = ds4_dspark_nightjar_arm_loss(
                slot, candidates[i].predicted_loss);
        if (context->previous_valid && context->previous_arm == 0u &&
            candidates[i].arm != 0u) {
            loss += candidates[i].switch_loss;
        }
        if (loss < decision->reference_loss) {
            decision->reference_loss = loss;
        }
    }
    if (!(decision->reference_loss > 0.0) ||
        !isfinite(decision->reference_loss)) {
        return 1;
    }
    decision->exploration_probability = exploration_probability;

    context->previous_arm = decision->arm;
    context->previous_valid = 1u;
    context->round_in_bin++;
    const uint32_t span =
        ds4_dspark_nightjar_bin_span(context->block_horizon);
    if (context->round_in_bin >= span) {
        context->round_in_bin = 0u;
        context->locked = 0u;
        context->bin_index++;
        if (context->bin_index > span) {
            context->block_index++;
            if (context->block_horizon <= UINT64_MAX / 2u) {
                context->block_horizon *= 2u;
            }
            context->bin_index = 1u;
        }
    }
#undef DS4_NIGHTJAR_ARM_SKIP
    return 0;
}

int ds4_dspark_nightjar_observe(
        ds4_dspark_nightjar_state *state,
        uint64_t context_key,
        uint32_t arm,
        double latency_seconds,
        uint32_t emitted_tokens) {
    if (!state || !(latency_seconds > 0.0) ||
        !isfinite(latency_seconds) || emitted_tokens == 0u) {
        return 1;
    }
    ds4_dspark_nightjar_context *context =
        ds4_dspark_nightjar_context_get(state, context_key, 0);
    if (!context) return 1;
    ds4_dspark_nightjar_arm *slot =
        ds4_dspark_nightjar_arm_get(context, arm, 0);
    if (!slot) return 1;
    slot->samples++;
    slot->total_latency += latency_seconds;
    slot->total_tokens += emitted_tokens;
    slot->mean_loss = slot->total_tokens != 0u
        ? slot->total_latency / (double)slot->total_tokens
        : 0.0;
    const double sample_loss = latency_seconds / (double)emitted_tokens;
    const double alpha = slot->recent_samples < 4u ? 0.5 : 0.25;
    slot->recent_loss = slot->recent_samples == 0u
        ? sample_loss
        : slot->recent_loss * (1.0 - alpha) + sample_loss * alpha;
    slot->recent_samples++;
    slot->reject_until = 0u;
    slot->reject_streak = 0u;
    context->observations++;

    if (context->locked && context->locked_arm == arm) {
        double best_other = DBL_MAX;
        for (uint32_t i = 0; i < DS4_DSPARK_NIGHTJAR_MAX_ARMS; i++) {
            const ds4_dspark_nightjar_arm *candidate = &context->arm[i];
            if (!candidate->in_use || candidate->arm == arm ||
                candidate->samples == 0u) {
                continue;
            }
            const double loss = ds4_dspark_nightjar_arm_loss(candidate, 0.0);
            if (loss > 0.0 && loss < best_other) best_other = loss;
        }
        const double selected_loss =
            ds4_dspark_nightjar_arm_loss(slot, 0.0);
        double abort_ratio = context->safe_ratio;
        if (!(abort_ratio >= 1.0) || !isfinite(abort_ratio)) {
            abort_ratio = 1.08;
        }
        if (abort_ratio > 1.08) abort_ratio = 1.08;
        if (best_other < DBL_MAX && selected_loss > 0.0 &&
            selected_loss > best_other * abort_ratio) {
            context->locked = 0u;
            context->revoked_pending = 1u;
        }
    }
    return 0;
}

int ds4_dspark_nightjar_reject(
        ds4_dspark_nightjar_state *state,
        uint64_t context_key,
        uint32_t arm,
        uint32_t cooldown_enabled) {
    ds4_dspark_nightjar_context *context =
        ds4_dspark_nightjar_context_get(state, context_key, 0);
    if (!context) return 1;
    ds4_dspark_nightjar_arm *slot =
        ds4_dspark_nightjar_arm_get(context, arm, 0);
    if (!slot) return 1;
    if (cooldown_enabled) {
        if (slot->reject_streak < 5u) slot->reject_streak++;
        const uint64_t cooldown = UINT64_C(1) << slot->reject_streak;
        slot->reject_until = context->observations > UINT64_MAX - cooldown
            ? UINT64_MAX : context->observations + cooldown;
    }
    if (context->locked && context->locked_arm == arm) {
        context->locked = 0u;
    }
    if (context->previous_valid && context->previous_arm == arm) {
        context->previous_valid = 0u;
    }
    context->revoked_pending = 1u;
    return 0;
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

int ds4_dspark_scheduler_peek_delayed_request(
        const ds4_dspark_scheduler_state *state,
        uint64_t request_id,
        ds4_dspark_schedule_request *request) {
    if (!state || !request || request_id == 0u) return 1;
    const uint64_t current_step = state->step + 1u;
    const uint64_t capacity_step =
        current_step > 2u ? current_step - 2u : 0u;
    if (capacity_step == 0u) return 1;
    for (uint32_t i = 0; i < DS4_DSPARK_SCHEDULER_MAX_REQUESTS; i++) {
        const ds4_dspark_scheduler_entry *entry = &state->entries[i];
        if (!entry->in_use || entry->request_id != request_id) continue;
        for (uint32_t h = 0; h < entry->history_count; h++) {
            if (entry->history_step[h] == capacity_step) {
                *request = entry->history[h];
                return 0;
            }
        }
        return 1;
    }
    return 1;
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
        ds4_dspark_schedule_step_result *second_result) {
    if (!state || !items || !first_sps || !second_sps ||
        !first_result || !second_result ||
        first_result == second_result) {
        return 1;
    }

    ds4_dspark_scheduler_state first_state = *state;
    ds4_dspark_scheduler_state second_state = *state;
    if (ds4_dspark_hardware_schedule_step_shape(
            &first_state, items, request_count,
            first_sps, sps_count,
            first_shape_sps, first_shape_sps_opaque,
            first_result) != 0 ||
        ds4_dspark_hardware_schedule_step_shape(
            &second_state, items, request_count,
            second_sps, sps_count,
            second_shape_sps, second_shape_sps_opaque,
            second_result) != 0 ||
        memcmp(&first_state, &second_state, sizeof(first_state)) != 0) {
        return 1;
    }

    *state = first_state;
    return 0;
}

int ds4_dspark_hardware_schedule_step_pair(
        ds4_dspark_scheduler_state *state,
        const ds4_dspark_schedule_item *items,
        uint32_t request_count,
        const double *first_sps,
        const double *second_sps,
        uint32_t sps_count,
        ds4_dspark_schedule_step_result *first_result,
        ds4_dspark_schedule_step_result *second_result) {
    return ds4_dspark_hardware_schedule_step_pair_shape(
            state, items, request_count,
            first_sps, NULL, NULL,
            second_sps, NULL, NULL,
            sps_count, first_result, second_result);
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
