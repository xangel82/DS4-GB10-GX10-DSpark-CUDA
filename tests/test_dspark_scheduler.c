/* Regression tests for the lossless DSpark hardware-aware scheduler. */

#include "ds4_dspark_scheduler.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void require_true(int condition, const char *message) {
    if (condition) return;
    fprintf(stderr, "dspark scheduler regression: %s\n", message);
    exit(1);
}

static void require_close(double actual, double expected,
                          const char *message) {
    if (fabs(actual - expected) <= 1.0e-9) return;
    fprintf(stderr,
            "dspark scheduler regression: %s (actual=%.12f expected=%.12f)\n",
            message, actual, expected);
    exit(1);
}

static void test_single_request(void) {
    ds4_dspark_schedule_request request = {
        .max_prefix = 3,
        .conditional = {0.90, 0.80, 0.70},
    };
    const double sps[] = {0.0, 10.0, 9.0, 8.0, 7.0};
    ds4_dspark_schedule_result result;
    require_true(ds4_dspark_hardware_schedule(
                     &request, 1, sps, 5, &result) == 0,
                 "single-request schedule failed");
    require_true(result.prefix[0] == 3, "single-request K must be 3");
    require_true(result.batch_size == 4, "single-request batch must be 4");
    require_close(result.expected_tokens, 3.124,
                  "single-request expected tokens");
    require_close(result.throughput, 21.868,
                  "single-request throughput");
}

static void test_causal_stop(void) {
    ds4_dspark_schedule_request request = {
        .max_prefix = 3,
        .conditional = {0.90, 0.80, 0.70},
    };
    const double sps[] = {0.0, 10.0, 8.0, 5.0, 100.0};
    ds4_dspark_schedule_result result;
    require_true(ds4_dspark_hardware_schedule(
                     &request, 1, sps, 5, &result) == 0,
                 "causal-stop schedule failed");
    require_true(result.prefix[0] == 1,
                 "causal stop must preserve the last improving prefix");
    require_true(result.stopped == 1 && result.stop_prefix == 2,
                 "causal stop must occur at K=2");
    require_true(result.evaluated_candidates == 2,
                 "causal stop must not inspect K=3");
}

static void test_two_request_global_allocation(void) {
    ds4_dspark_schedule_request request[2];
    memset(request, 0, sizeof(request));
    request[0].max_prefix = 2;
    request[0].conditional[0] = 0.90;
    request[0].conditional[1] = 0.80;
    request[1].max_prefix = 2;
    request[1].conditional[0] = 0.85;
    request[1].conditional[1] = 0.50;
    const double sps[] = {0.0, 0.0, 10.0, 9.0, 8.0, 7.0, 5.0};
    ds4_dspark_schedule_result result;
    require_true(ds4_dspark_hardware_schedule(
                     request, 2, sps, 7, &result) == 0,
                 "two-request schedule failed");
    require_true(result.prefix[0] == 2 && result.prefix[1] == 1,
                 "global allocation must choose A2/B1");
    require_true(result.batch_size == 5,
                 "two-request selected batch must be 5");
    require_close(result.expected_tokens, 4.47,
                  "two-request expected tokens");
    require_close(result.throughput, 31.29,
                  "two-request throughput");
    require_true(result.stopped == 1 &&
                 result.stop_request == 1 &&
                 result.stop_prefix == 2,
                 "two-request stop candidate");
}

static void test_zero_prefix_and_ties(void) {
    ds4_dspark_schedule_request request[2];
    memset(request, 0, sizeof(request));
    request[0].max_prefix = 2;
    request[0].conditional[0] = 1.0;
    request[0].conditional[1] = 1.0;
    request[1].max_prefix = 1;
    request[1].conditional[0] = 1.0;
    const double sps[] = {0.0, 0.0, 10.0, 10.0, 10.0, 10.0};
    ds4_dspark_schedule_result result;
    require_true(ds4_dspark_hardware_schedule(
                     request, 2, sps, 6, &result) == 0,
                 "tie schedule failed");
    require_true(result.prefix[0] == 2 && result.prefix[1] == 1,
                 "ties must retain prefix closure deterministically");

    ds4_dspark_schedule_request zero = {
        .max_prefix = 1,
        .conditional = {0.10},
    };
    const double declining_sps[] = {0.0, 10.0, 4.0};
    require_true(ds4_dspark_hardware_schedule(
                     &zero, 1, declining_sps, 3, &result) == 0,
                 "zero-prefix schedule failed");
    require_true(result.prefix[0] == 0 && result.batch_size == 1,
                 "K=0 must remain a valid hardware-aware decision");
}

static void test_hardware_minimum_prefix(void) {
    ds4_dspark_schedule_request request[2];
    memset(request, 0, sizeof(request));
    request[0].minimum_prefix = request[1].minimum_prefix = 1;
    request[0].max_prefix = request[1].max_prefix = 2;
    request[0].conditional[0] = 0.90;
    request[0].conditional[1] = 0.80;
    request[1].conditional[0] = 0.70;
    request[1].conditional[1] = 0.60;
    const double sps[] =
        {0.0, 0.0, 0.0, 0.0, 10.0, 9.0, 5.0};
    ds4_dspark_schedule_result result;
    require_true(ds4_dspark_hardware_schedule(
                     request, 2, sps, 7, &result) == 0,
                 "minimum-prefix schedule failed");
    require_true(result.prefix[0] == 2 && result.prefix[1] == 1,
                 "minimum-prefix allocation must choose A2/B1");
    require_true(result.batch_size == 5,
                 "minimum-prefix physical batch must be 5");
    require_close(result.expected_tokens, 4.32,
                  "minimum-prefix expected tokens");
    require_close(result.baseline_throughput, 36.0,
                  "minimum-prefix baseline throughput");

    request[0].minimum_prefix = 3;
    require_true(ds4_dspark_hardware_schedule(
                     request, 2, sps, 7, &result) != 0,
                 "minimum prefix above max must be rejected");
}

static void test_async_crosses_jagged_hardware_cliff(void) {
    ds4_dspark_schedule_request history = {
        .max_prefix = 3,
        .conditional = {0.90, 0.80, 0.70},
    };
    ds4_dspark_schedule_request current = {
        .max_prefix = 3,
        .conditional = {0.80, 0.75, 0.60},
    };
    /* The synchronous causal rule stops at K=1 and cannot observe the later
     * hardware recovery. Historical capacity selection may inspect it. */
    const double sps[] = {0.0, 10.0, 8.0, 5.0, 9.0};
    ds4_dspark_schedule_result sync;
    ds4_dspark_schedule_result async;
    require_true(ds4_dspark_hardware_schedule(
                     &history, 1, sps, 5, &sync) == 0,
                 "jagged synchronous schedule failed");
    require_true(sync.prefix[0] == 1,
                 "synchronous schedule must stop at the first cliff");
    require_true(ds4_dspark_hardware_schedule_async(
                     &history, &current, 1, sps, 5, &async) == 0,
                 "jagged asynchronous schedule failed");
    require_true(async.prefix[0] == 3 && async.batch_size == 4,
                 "historical capacity must cross the SPS cliff");
    require_close(async.expected_tokens, 2.76,
                  "asynchronous current expected tokens");
    require_close(async.throughput, 24.84,
                  "asynchronous current throughput");
}

static void test_async_capacity_is_historical(void) {
    ds4_dspark_schedule_request history[2];
    ds4_dspark_schedule_request current[2];
    memset(history, 0, sizeof(history));
    memset(current, 0, sizeof(current));
    history[0].max_prefix = history[1].max_prefix = 2;
    history[0].conditional[0] = 0.95;
    history[0].conditional[1] = 0.90;
    history[1].conditional[0] = 0.85;
    history[1].conditional[1] = 0.50;
    current[0].max_prefix = current[1].max_prefix = 2;
    current[0].conditional[0] = 0.40;
    current[0].conditional[1] = 0.20;
    current[1].conditional[0] = 0.99;
    current[1].conditional[1] = 0.95;
    const double sps[] = {0.0, 0.0, 10.0, 9.0, 8.0, 6.0, 4.0};
    ds4_dspark_schedule_result result;
    require_true(ds4_dspark_hardware_schedule_async(
                     history, current, 2, sps, 7, &result) == 0,
                 "two-request asynchronous schedule failed");
    require_true(result.admitted_candidates == 2 &&
                 result.batch_size == 4,
                 "historical scores alone must determine capacity");
    require_true(result.prefix[0] == 0 && result.prefix[1] == 2,
                 "current scores must allocate the fixed capacity globally");
}

static void test_async_current_confidence_cannot_expand_capacity(void) {
    ds4_dspark_schedule_request history = {
        .max_prefix = 1,
        .conditional = {0.10},
    };
    ds4_dspark_schedule_request current = {
        .max_prefix = 1,
        .conditional = {0.99},
    };
    const double sps[] = {0.0, 10.0, 4.0};
    ds4_dspark_schedule_result result;
    require_true(ds4_dspark_hardware_schedule_async(
                     &history, &current, 1, sps, 3, &result) == 0,
                 "zero-capacity asynchronous schedule failed");
    require_true(result.admitted_candidates == 0 &&
                 result.prefix[0] == 0 &&
                 result.batch_size == 1,
                 "current confidence must not expand historical capacity");
}

static ds4_dspark_schedule_item schedule_item(
        uint64_t id, double c1, double c2) {
    ds4_dspark_schedule_item item;
    memset(&item, 0, sizeof(item));
    item.request_id = id;
    item.request.max_prefix = 2;
    item.request.conditional[0] = c1;
    item.request.conditional[1] = c2;
    return item;
}

static void test_stateful_two_step_barrier(void) {
    ds4_dspark_scheduler_state state;
    ds4_dspark_scheduler_state_reset(&state);
    const double sps[] = {0.0, 10.0, 8.0, 9.0};
    ds4_dspark_schedule_step_result result;
    ds4_dspark_schedule_item item = schedule_item(42, 0.90, 0.80);

    require_true(ds4_dspark_hardware_schedule_step(
                     &state, &item, 1, sps, 4, &result) == 0,
                 "stateful first step failed");
    require_true(result.used_async == 0,
                 "first step must use causal scheduling");
    require_true(ds4_dspark_hardware_schedule_step(
                     &state, &item, 1, sps, 4, &result) == 0,
                 "stateful second step failed");
    require_true(result.used_async == 0,
                 "second step must use causal scheduling");

    item.request.conditional[0] = 0.99;
    item.request.conditional[1] = 0.99;
    require_true(ds4_dspark_hardware_schedule_step(
                     &state, &item, 1, sps, 4, &result) == 0,
                 "stateful third step failed");
    require_true(result.used_async == 1,
                 "third step must cross the two-step barrier");
    require_true(result.selected.prefix[0] == 2,
                 "historical capacity must be filled by current confidence");
}

static void test_history_reset_preserves_runtime_step(void) {
    ds4_dspark_scheduler_state state;
    ds4_dspark_scheduler_state_reset(&state);
    const double sps[] = {0.0, 10.0, 8.0, 9.0};
    ds4_dspark_schedule_step_result result;
    ds4_dspark_schedule_item item = schedule_item(42, 0.90, 0.80);

    for (int step = 0; step < 3; step++) {
        require_true(ds4_dspark_hardware_schedule_step(
                         &state, &item, 1, sps, 4, &result) == 0,
                     "history-reset warmup failed");
    }
    require_true(state.step == 3u && state.entry_count == 1u,
                 "history-reset warmup state");

    ds4_dspark_scheduler_state_reset_history(&state);
    require_true(state.step == 3u && state.entry_count == 0u,
                 "history reset must preserve the global step only");

    require_true(ds4_dspark_hardware_schedule_step(
                     &state, &item, 1, sps, 4, &result) == 0 &&
                 result.used_async == 0 && state.step == 4u,
                 "first post-gap step must be causal");
    require_true(ds4_dspark_hardware_schedule_step(
                     &state, &item, 1, sps, 4, &result) == 0 &&
                 result.used_async == 0 && state.step == 5u,
                 "second post-gap step must be causal");
    require_true(ds4_dspark_hardware_schedule_step(
                     &state, &item, 1, sps, 4, &result) == 0 &&
                 result.used_async == 1 && state.step == 6u,
                 "third post-gap step must recover exact t-2 history");
}

static void test_physical_confirmation_probe(void) {
    double observed = -1.0;
    require_true(!ds4_dspark_should_confirm_physical(
                     0, 0.02, 0.18, 4.0, 17.0, 1.10, &observed),
                 "an unmeasured shape must not request confirmation");
    require_close(observed, 0.0,
                  "an unmeasured shape must not expose a rate");

    require_true(ds4_dspark_should_confirm_physical(
                     1, 0.02, 0.18, 4.0, 17.0, 1.10, &observed),
                 "a strong first sample must request confirmation");
    require_close(observed, 20.0,
                  "confirmation must report the observed physical rate");

    require_true(!ds4_dspark_should_confirm_physical(
                     1, 0.02, 0.18, 4.0, 19.0, 1.10, &observed),
                 "a marginal first sample must retain the serial gate");
    require_close(observed, 20.0,
                  "a marginal valid sample still reports its rate");

    require_true(!ds4_dspark_should_confirm_physical(
                     2, 0.02, 0.18, 4.0, 17.0, 1.10, &observed),
                 "a mature profile must use normal exploitation");
    require_close(observed, 0.0,
                  "a mature profile is not a confirmation sample");
}

static void test_stateful_cohort_reordering(void) {
    ds4_dspark_scheduler_state state;
    ds4_dspark_scheduler_state_reset(&state);
    const double sps[] = {0.0, 0.0, 10.0, 9.0, 8.0, 6.0, 4.0};
    ds4_dspark_schedule_step_result result;
    ds4_dspark_schedule_item items[2] = {
        schedule_item(11, 0.95, 0.90),
        schedule_item(22, 0.30, 0.20),
    };
    for (int step = 0; step < 2; step++) {
        require_true(ds4_dspark_hardware_schedule_step(
                         &state, items, 2, sps, 7, &result) == 0,
                     "cohort history warmup failed");
    }

    ds4_dspark_schedule_item reordered[2] = {
        schedule_item(22, 0.99, 0.95),
        schedule_item(11, 0.20, 0.10),
    };
    require_true(ds4_dspark_hardware_schedule_step(
                     &state, reordered, 2, sps, 7, &result) == 0,
                 "reordered cohort schedule failed");
    require_true(result.used_async == 1,
                 "reordered mature cohort must use async scheduling");
    require_true(result.selected.prefix[0] == 2 &&
                 result.selected.prefix[1] == 0,
                 "current rank must follow request identity after reorder");
}

static void test_stateful_ties_follow_request_identity(void) {
    ds4_dspark_scheduler_state first_state;
    ds4_dspark_scheduler_state second_state;
    ds4_dspark_scheduler_state_reset(&first_state);
    ds4_dspark_scheduler_state_reset(&second_state);
    const double sps[] = {0.0, 0.0, 10.0, 8.0, 1.0};
    ds4_dspark_schedule_step_result first_result;
    ds4_dspark_schedule_step_result second_result;
    ds4_dspark_schedule_item first_order[2] = {
        schedule_item(22, 0.90, 0.0),
        schedule_item(11, 0.90, 0.0),
    };
    ds4_dspark_schedule_item second_order[2] = {
        first_order[1],
        first_order[0],
    };

    require_true(ds4_dspark_hardware_schedule_step(
                     &first_state, first_order, 2, sps, 5,
                     &first_result) == 0,
                 "identity tie first order failed");
    require_true(ds4_dspark_hardware_schedule_step(
                     &second_state, second_order, 2, sps, 5,
                     &second_result) == 0,
                 "identity tie second order failed");
    require_true(first_result.selected.prefix[0] == 0 &&
                 first_result.selected.prefix[1] == 1,
                 "lower identity must win the first ordering tie");
    require_true(second_result.selected.prefix[0] == 1 &&
                 second_result.selected.prefix[1] == 0,
                 "lower identity must win after cohort reorder");
}

static void test_stateful_join_and_forget(void) {
    ds4_dspark_scheduler_state state;
    ds4_dspark_scheduler_state_reset(&state);
    const double sps[] = {0.0, 10.0, 10.0, 9.0, 8.0, 6.0, 4.0};
    ds4_dspark_schedule_step_result result;
    ds4_dspark_schedule_item first = schedule_item(7, 0.90, 0.80);
    for (int step = 0; step < 3; step++) {
        require_true(ds4_dspark_hardware_schedule_step(
                         &state, &first, 1, sps, 7, &result) == 0,
                     "single member warmup failed");
    }

    ds4_dspark_schedule_item pair[2] = {
        first,
        schedule_item(8, 0.95, 0.90),
    };
    require_true(ds4_dspark_hardware_schedule_step(
                     &state, pair, 2, sps, 7, &result) == 0,
                 "joining cohort schedule failed");
    require_true(result.used_async == 0 &&
                 result.history_ready_requests == 1,
                 "a new member must force causal cohort scheduling");
    require_true(ds4_dspark_hardware_schedule_step(
                     &state, pair, 2, sps, 7, &result) == 0,
                 "joining cohort second schedule failed");
    require_true(result.used_async == 0,
                 "new member needs two complete observations");
    require_true(ds4_dspark_hardware_schedule_step(
                     &state, pair, 2, sps, 7, &result) == 0,
                 "mature joined cohort schedule failed");
    require_true(result.used_async == 1,
                 "mature joined cohort must enable async scheduling");

    require_true(ds4_dspark_scheduler_state_forget(&state, 8) == 1,
                 "forget must remove a known request");
    require_true(ds4_dspark_scheduler_state_forget(&state, 8) == 0,
                 "forget must report an unknown request");
    require_true(ds4_dspark_hardware_schedule_step(
                     &state, pair, 2, sps, 7, &result) == 0,
                 "forgotten member schedule failed");
    require_true(result.used_async == 0 &&
                 result.history_ready_requests == 1,
                 "forgotten identity must not inherit stale history");
}

static void test_stateful_current_boundary_clamps_history(void) {
    ds4_dspark_scheduler_state state;
    ds4_dspark_scheduler_state_reset(&state);
    const double sps[] = {0.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0};
    ds4_dspark_schedule_step_result result;
    ds4_dspark_schedule_item item;
    memset(&item, 0, sizeof(item));
    item.request_id = 99;
    item.request.max_prefix = 5;
    for (uint32_t k = 0; k < 5; k++) {
        item.request.conditional[k] = 0.95;
    }
    for (int step = 0; step < 2; step++) {
        require_true(ds4_dspark_hardware_schedule_step(
                         &state, &item, 1, sps, 7, &result) == 0,
                     "boundary history warmup failed");
    }

    item.request.max_prefix = 1;
    require_true(ds4_dspark_hardware_schedule_step(
                     &state, &item, 1, sps, 3, &result) == 0,
                 "shrinking current boundary schedule failed");
    require_true(result.used_async == 1 &&
                 result.selected.prefix[0] <= 1 &&
                 result.selected.batch_size <= 2,
                 "historical capacity must respect the current boundary");
    require_close(result.selected.capacity_throughput, 1.95 * sps[2],
                  "clamped capacity throughput must match its physical batch");
}

static void test_stateful_history_uses_exact_global_step(void) {
    ds4_dspark_scheduler_state state;
    ds4_dspark_scheduler_state_reset(&state);
    const double sps[] = {0.0, 10.0, 9.0, 8.0, 7.0};
    ds4_dspark_schedule_step_result result;
    ds4_dspark_schedule_item a = schedule_item(101, 0.90, 0.80);
    ds4_dspark_schedule_item b = schedule_item(202, 0.70, 0.60);

    require_true(ds4_dspark_hardware_schedule_step(
                     &state, &a, 1, sps, 5, &result) == 0,
                 "exact-step history A1 failed");
    require_true(ds4_dspark_hardware_schedule_step(
                     &state, &a, 1, sps, 5, &result) == 0,
                 "exact-step history A2 failed");
    require_true(ds4_dspark_hardware_schedule_step(
                     &state, &b, 1, sps, 5, &result) == 0,
                 "exact-step history B3 failed");

    /* At global step 4, A's step-2 confidence is available even though A was
     * absent at step 3. */
    require_true(ds4_dspark_hardware_schedule_step(
                     &state, &a, 1, sps, 5, &result) == 0,
                 "exact-step history A4 failed");
    require_true(result.used_async == 1,
                 "step 4 must use A's exact step-2 history");

    /* At global step 5, A has no observation from step 3. Reusing its second
     * previous observation would be a temporal-offset bug. */
    require_true(ds4_dspark_hardware_schedule_step(
                     &state, &a, 1, sps, 5, &result) == 0,
                 "exact-step history A5 failed");
    require_true(result.used_async == 0,
                 "a missing exact step-3 observation must force causal mode");

    require_true(ds4_dspark_hardware_schedule_step(
                     &state, &a, 1, sps, 5, &result) == 0,
                 "exact-step history A6 failed");
    require_true(result.used_async == 1,
                 "step 6 must recover with A's exact step-4 history");
}

static void test_flattened_marker_layout(void) {
    ds4_dspark_schedule_result schedule;
    memset(&schedule, 0, sizeof(schedule));
    schedule.request_count = 3;
    schedule.prefix[0] = 2;
    schedule.prefix[1] = 0;
    schedule.prefix[2] = 1;
    schedule.batch_size = 6;
    ds4_dspark_schedule_item items[3] = {
        schedule_item(7001, 0.9, 0.8),
        schedule_item(7002, 0.9, 0.8),
        schedule_item(7003, 0.9, 0.8),
    };
    items[0].position = 100;
    items[1].position = 200;
    items[2].position = 300;

    ds4_dspark_flatten_plan plan;
    require_true(ds4_dspark_schedule_flatten(
                     &schedule, items, &plan) == 0,
                 "flattened marker plan failed");
    require_true(plan.request_count == 3 && plan.row_count == 6,
                 "flattened plan dimensions");
    require_true(plan.request_offset[0] == 0 &&
                 plan.request_offset[1] == 3 &&
                 plan.request_offset[2] == 4,
                 "flattened request offsets");
    require_true(plan.request_rows[0] == 3 &&
                 plan.request_rows[1] == 1 &&
                 plan.request_rows[2] == 2,
                 "flattened request row counts");
    const uint32_t expected_request[] = {0, 0, 0, 1, 2, 2};
    const uint32_t expected_prefix[] = {0, 1, 2, 0, 0, 1};
    const uint32_t expected_position[] = {100, 101, 102, 200, 300, 301};
    for (uint32_t row = 0; row < plan.row_count; row++) {
        require_true(plan.row_request[row] == expected_request[row] &&
                     plan.row_prefix[row] == expected_prefix[row] &&
                     plan.row_position[row] == expected_position[row],
                     "flattened marker row order");
    }
    require_true(plan.request_id[0] == 7001 &&
                 plan.request_id[1] == 7002 &&
                 plan.request_id[2] == 7003,
                 "flattened request identities");

    schedule.batch_size = 5;
    require_true(ds4_dspark_schedule_flatten(
                     &schedule, items, &plan) != 0,
                 "flatten must reject a mismatched physical batch");

    schedule.batch_size = 6;
    items[2].request_id = items[1].request_id;
    require_true(ds4_dspark_schedule_flatten(
                     &schedule, items, &plan) != 0,
                 "flatten must reject duplicate request identities");

    items[2].request_id = 7003;
    items[2].request.max_prefix = 0;
    require_true(ds4_dspark_schedule_flatten(
                     &schedule, items, &plan) != 0,
                 "flatten must reject a prefix not offered by its request");
}

static void test_physical_batch_build_and_scatter(void) {
    ds4_dspark_schedule_result schedule;
    memset(&schedule, 0, sizeof(schedule));
    schedule.request_count = 3;
    schedule.prefix[0] = 2;
    schedule.prefix[1] = 0;
    schedule.prefix[2] = 1;
    schedule.batch_size = 6;
    ds4_dspark_schedule_item items[3] = {
        schedule_item(71, 0.9, 0.8),
        schedule_item(72, 0.9, 0.8),
        schedule_item(73, 0.9, 0.8),
    };
    items[0].position = 100;
    items[1].position = 200;
    items[2].position = 300;
    const int32_t pending[] = {1000, 2000, 3000};
    const int32_t drafts[3][DS4_DSPARK_SCHEDULER_MAX_PREFIX] = {
        {1001, 1002, 1003, 1004, 1005},
        {2001, 2002, 2003, 2004, 2005},
        {3001, 3002, 3003, 3004, 3005},
    };
    const uint64_t rng[] = {111, 222, 333};
    ds4_dspark_physical_batch batch;
    require_true(ds4_dspark_physical_batch_build(
                     &schedule,
                     items,
                     pending,
                     &drafts[0][0],
                     DS4_DSPARK_SCHEDULER_MAX_PREFIX,
                     rng,
                     &batch) == 0,
                 "physical batch build failed");
    const int32_t expected_tokens[] =
        {1000, 1001, 1002, 2000, 3000, 3001};
    for (uint32_t row = 0; row < batch.layout.row_count; row++) {
        require_true(batch.row_token[row] == expected_tokens[row],
                     "physical batch must be request-major and unpadded");
    }
    require_true(batch.rng_state[0] == 111 &&
                 batch.rng_state[1] == 222 &&
                 batch.rng_state[2] == 333,
                 "physical batch must preserve per-request RNG");

    const uint32_t committed[] = {1, 0, 1};
    ds4_dspark_physical_result result;
    require_true(ds4_dspark_physical_batch_scatter(
                     &batch, committed, &result) == 0,
                 "physical result scatter failed");
    require_true(result.emitted_total == 5 &&
                 result.emitted_tokens[0] == 2 &&
                 result.emitted_tokens[1] == 1 &&
                 result.emitted_tokens[2] == 2,
                 "physical result emitted counts");
    require_true(result.continuation_row[0] == 1 &&
                 result.continuation_row[1] == 3 &&
                 result.continuation_row[2] == 5,
                 "physical result continuation rows");

    const uint32_t invalid_committed[] = {3, 0, 1};
    require_true(ds4_dspark_physical_batch_scatter(
                     &batch, invalid_committed, &result) != 0,
                 "physical result crossed a request prefix boundary");
}

int main(void) {
    test_single_request();
    test_causal_stop();
    test_two_request_global_allocation();
    test_zero_prefix_and_ties();
    test_hardware_minimum_prefix();
    test_async_crosses_jagged_hardware_cliff();
    test_async_capacity_is_historical();
    test_async_current_confidence_cannot_expand_capacity();
    test_stateful_two_step_barrier();
    test_history_reset_preserves_runtime_step();
    test_physical_confirmation_probe();
    test_stateful_cohort_reordering();
    test_stateful_ties_follow_request_identity();
    test_stateful_join_and_forget();
    test_stateful_current_boundary_clamps_history();
    test_stateful_history_uses_exact_global_step();
    test_flattened_marker_layout();
    test_physical_batch_build_and_scatter();
    puts("dspark hardware scheduler regression: OK");
    return 0;
}
