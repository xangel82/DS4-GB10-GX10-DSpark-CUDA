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

static void test_fixed_capacity_reranks_current_requests(void) {
    ds4_dspark_schedule_request request[2];
    memset(request, 0, sizeof(request));
    request[0].max_prefix = request[1].max_prefix = 2;
    request[0].conditional[0] = 0.40;
    request[0].conditional[1] = 0.20;
    request[1].conditional[0] = 0.99;
    request[1].conditional[1] = 0.95;
    const double sps[] = {0.0, 0.0, 10.0, 9.0, 8.0, 7.0, 6.0};
    ds4_dspark_schedule_result result;
    require_true(ds4_dspark_hardware_schedule_fixed_capacity(
                     request, 2, 4, sps, 7, &result) == 0,
                 "fixed-capacity schedule failed");
    require_true(result.prefix[0] == 0 && result.prefix[1] == 2,
                 "fixed capacity must use current global rank");
    require_true(result.admitted_candidates == 2 &&
                 result.batch_size == 4,
                 "fixed capacity must retain the locked row count");
    require_close(result.expected_tokens, 2.99 + 0.9405,
                  "fixed-capacity expected tokens");
    require_true(ds4_dspark_hardware_schedule_fixed_capacity(
                     request, 2, 7, sps, 7, &result) != 0,
                 "unreachable fixed capacity must be rejected");
}

static void test_nightjar_capacity_does_not_lock_historical_lane(void) {
    ds4_dspark_nightjar_state state;
    ds4_dspark_nightjar_state_reset(&state);
    const ds4_dspark_nightjar_candidate candidate[] = {
        {.arm = 1u, .predicted_loss = 0.080, .switch_loss = 0.0},
        {.arm = 2u, .predicted_loss = 0.050, .switch_loss = 0.0},
    };
    ds4_dspark_nightjar_decision decision;
    require_true(ds4_dspark_nightjar_select(
                     &state, 91u, candidate, 2u, 1.08, &decision) == 0,
                 "Nightjar capacity selection failed");
    require_true(decision.arm == 2u,
                 "Nightjar did not select the expected aggregate budget");

    ds4_dspark_schedule_request current[2];
    memset(current, 0, sizeof(current));
    current[0].max_prefix = current[1].max_prefix = 2u;
    current[0].conditional[0] = 0.20;
    current[0].conditional[1] = 0.10;
    current[1].conditional[0] = 0.99;
    current[1].conditional[1] = 0.95;
    const double sps[] = {0.0, 0.0, 10.0, 9.0, 8.0, 7.0, 6.0};
    ds4_dspark_schedule_result result;
    require_true(ds4_dspark_hardware_schedule_fixed_capacity(
                     current, 2u, 2u + decision.arm,
                     sps, 7u, &result) == 0,
                 "Nightjar fixed-capacity rerank failed");
    require_true(result.prefix[0] == 0u && result.prefix[1] == 2u,
                 "t-2 capacity must not lock the historical lane allocation");
}

static void test_nightjar_warm_start_and_bin_lock(void) {
    ds4_dspark_nightjar_state state;
    ds4_dspark_nightjar_state_reset(&state);
    const ds4_dspark_nightjar_candidate candidate[] = {
        {.arm = 10u, .predicted_loss = 0.050, .switch_loss = 0.0},
        {.arm = 20u, .predicted_loss = 0.040, .switch_loss = 0.0},
    };
    ds4_dspark_nightjar_decision decision;
    require_true(ds4_dspark_nightjar_select(
                     &state, 7u, candidate, 2u, 1.0, &decision) == 0,
                 "Nightjar warm start failed");
    require_true(decision.arm == 20u && decision.exploration == 0u,
                 "Nightjar must warm-start from the SPS prior");
    require_true(ds4_dspark_nightjar_observe(
                     &state, 7u, decision.arm, 0.041, 1u) == 0,
                 "Nightjar first observation failed");

    const ds4_dspark_nightjar_candidate shifted[] = {
        {.arm = 10u, .predicted_loss = 0.030, .switch_loss = 0.0},
        {.arm = 20u, .predicted_loss = 0.040, .switch_loss = 0.0},
    };
    /* An unseen arm is measured when its prior becomes competitive, without
     * exhaustively bootstrapping every possible budget. */
    require_true(ds4_dspark_nightjar_select(
                     &state, 7u, shifted, 2u, 1.0, &decision) == 0,
                 "Nightjar bootstrap exploration failed");
    require_true(decision.arm == 10u && decision.exploration == 1u,
                 "Nightjar must sample the remaining unobserved arm");
    require_true(state.context[0].locked == 0u,
                 "Nightjar exploratory probation must not lock a bin");
    require_true(ds4_dspark_nightjar_observe(
                     &state, 7u, decision.arm, 0.042, 1u) == 0,
                 "Nightjar second observation failed");

    require_true(ds4_dspark_nightjar_select(
                     &state, 7u, candidate, 2u, 2.0, &decision) == 0,
                 "Nightjar first post-bootstrap round failed");

    require_true(ds4_dspark_nightjar_select(
                     &state, 8u, candidate, 2u, 1.0, &decision) == 0,
                 "Nightjar independent context failed");
    require_true(decision.arm == 20u && decision.arm_samples == 0u,
                 "Nightjar contexts must keep independent statistics");
}

static void test_peek_delayed_request_is_exact_and_read_only(void) {
    ds4_dspark_scheduler_state state;
    ds4_dspark_scheduler_state_reset(&state);
    state.step = 4u;
    state.entry_count = 1u;
    state.entries[0].in_use = 1u;
    state.entries[0].request_id = 77u;
    state.entries[0].history_count = 2u;
    state.entries[0].history_step[0] = 4u;
    state.entries[0].history_step[1] = 3u;
    state.entries[0].history[0].max_prefix = 5u;
    state.entries[0].history[0].conditional[0] = 0.11;
    state.entries[0].history[1].max_prefix = 5u;
    state.entries[0].history[1].conditional[0] = 0.73;

    ds4_dspark_schedule_request delayed;
    require_true(ds4_dspark_scheduler_peek_delayed_request(
                     &state, 77u, &delayed) == 0,
                 "exact t-2 confidence lookup failed");
    require_close(delayed.conditional[0], 0.73,
                  "exact t-2 confidence value");
    require_true(state.step == 4u &&
                 state.entries[0].history_step[0] == 4u,
                 "t-2 confidence lookup mutated scheduler state");
    require_true(ds4_dspark_scheduler_peek_delayed_request(
                     &state, 78u, &delayed) != 0,
                 "unknown request unexpectedly found t-2 confidence");
}

static void test_nightjar_skips_unsafe_bootstrap_arms(void) {
    ds4_dspark_nightjar_state state;
    ds4_dspark_nightjar_state_reset(&state);
    const ds4_dspark_nightjar_candidate candidate[] = {
        {.arm = 2u, .predicted_loss = 0.030, .switch_loss = 0.0},
        {.arm = 9u, .predicted_loss = 0.060, .switch_loss = 0.0},
    };
    ds4_dspark_nightjar_decision decision;
    require_true(ds4_dspark_nightjar_select(
                     &state, 0x17u, candidate, 2u, 1.15, &decision) == 0 &&
                 decision.arm == 2u,
                 "Nightjar safe bootstrap setup failed");
    require_true(ds4_dspark_nightjar_observe(
                     &state, 0x17u, 2u, 0.030, 1u) == 0,
                 "Nightjar safe bootstrap observation failed");
    require_true(ds4_dspark_nightjar_select(
                     &state, 0x17u, candidate, 2u, 1.15, &decision) == 0 &&
                 decision.arm == 2u && decision.exploration == 0u,
                 "Nightjar sampled an unsafe unobserved arm");
}

static void test_nightjar_revokes_a_degraded_lock(void) {
    ds4_dspark_nightjar_state state;
    ds4_dspark_nightjar_state_reset(&state);
    const ds4_dspark_nightjar_candidate candidate[] = {
        {.arm = 2u, .predicted_loss = 0.030, .switch_loss = 0.0},
        {.arm = 5u, .predicted_loss = 0.034, .switch_loss = 0.0},
    };
    const ds4_dspark_nightjar_candidate shifted[] = {
        {.arm = 2u, .predicted_loss = 0.030, .switch_loss = 0.0},
        {.arm = 5u, .predicted_loss = 0.025, .switch_loss = 0.0},
    };
    ds4_dspark_nightjar_decision decision;

    require_true(ds4_dspark_nightjar_select(
                     &state, 0x91u, candidate, 2u, 1.15, &decision) == 0 &&
                 decision.arm == 2u,
                 "Nightjar revocation setup must select the best prior");
    require_true(ds4_dspark_nightjar_observe(
                     &state, 0x91u, 2u, 0.030, 1u) == 0,
                 "Nightjar revocation setup observation failed");
    require_true(ds4_dspark_nightjar_select(
                     &state, 0x91u, shifted, 2u, 1.0, &decision) == 0 &&
                 decision.arm == 5u && decision.exploration == 1u,
                 "Nightjar must probation the unseen arm");
    require_true(ds4_dspark_nightjar_observe(
                     &state, 0x91u, 5u, 0.040, 1u) == 0,
                 "Nightjar probation observation failed");

    for (uint32_t attempt = 0u;
         attempt < 8u && state.context[0].locked == 0u;
         attempt++) {
        require_true(ds4_dspark_nightjar_select(
                         &state, 0x91u, candidate, 2u, 1.15,
                         &decision) == 0 && decision.arm == 2u,
                     "Nightjar must select the measured incumbent");
        if (state.context[0].locked == 0u) {
            require_true(ds4_dspark_nightjar_observe(
                             &state, 0x91u, 2u, 0.030, 1u) == 0,
                         "Nightjar incumbent warm-up failed");
        }
    }
    require_true(state.context[0].locked == 1u,
                 "Nightjar incumbent was not bin-locked");
    require_true(ds4_dspark_nightjar_observe(
                     &state, 0x91u, 2u, 0.200, 1u) == 0,
                 "Nightjar degraded observation failed");
    require_true(state.context[0].locked == 0u,
                 "Nightjar did not revoke a degraded lock");
    require_true(ds4_dspark_nightjar_select(
                     &state, 0x91u, candidate, 2u, 1.15, &decision) == 0 &&
                 decision.arm == 5u && decision.revoked == 1u,
                 "Nightjar did not expose and recover from revocation");
}

static void test_nightjar_safe_exploration_uses_observed_loss(void) {
    ds4_dspark_nightjar_state state;
    ds4_dspark_nightjar_state_reset(&state);
    const ds4_dspark_nightjar_candidate candidates[] = {
        {.arm = 2u, .predicted_loss = 0.031, .switch_loss = 0.0},
        {.arm = 5u, .predicted_loss = 0.030, .switch_loss = 0.0},
    };
    ds4_dspark_nightjar_decision decision;
    require_true(ds4_dspark_nightjar_select(
                         &state, 0x52u, candidates, 2u, 1.15, &decision) == 0,
                 "Nightjar observed-loss setup select failed");
    require_true(decision.arm == 5u,
                 "Nightjar observed-loss setup must use the best prior");
    require_true(ds4_dspark_nightjar_observe(
                         &state, 0x52u, 5u, 0.029, 1u) == 0,
                 "Nightjar observed-loss setup observe failed");

    /* Advance beyond the first one-round block.  Arm 2 still has an
     * optimistic SPS prior here, so make its measured cost explicitly bad. */
    require_true(ds4_dspark_nightjar_select(
                         &state, 0x52u, candidates, 2u, 1.15, &decision) == 0,
                 "Nightjar observed-loss second select failed");
    require_true(ds4_dspark_nightjar_observe(
                         &state, 0x52u, 2u, 0.046, 1u) == 0,
                 "Nightjar observed-loss bad-arm observe failed");

    for (uint32_t i = 0; i < 32u; i++) {
        require_true(ds4_dspark_nightjar_select(
                             &state, 0x52u, candidates, 2u, 1.15,
                             &decision) == 0,
                     "Nightjar observed-loss exploration select failed");
        require_true(decision.locked || decision.arm == 5u,
                     "Nightjar exploration crossed the observed-loss guard");
        require_true(ds4_dspark_nightjar_observe(
                             &state, 0x52u, decision.arm,
                             decision.arm == 5u ? 0.029 : 0.046,
                             1u) == 0,
                     "Nightjar observed-loss exploration observe failed");
    }
}

static void test_nightjar_switch_cost_only_wakes_drafter(void) {
    ds4_dspark_nightjar_state state;
    ds4_dspark_nightjar_state_reset(&state);
    ds4_dspark_nightjar_decision decision;
    const ds4_dspark_nightjar_candidate sleep_candidates[] = {
        {.arm = 0u, .predicted_loss = 0.012, .switch_loss = 0.0},
        {.arm = 1u, .predicted_loss = 0.020, .switch_loss = 0.005},
    };
    require_true(ds4_dspark_nightjar_select(
                         &state, 0x91u, sleep_candidates, 2u, 1.0,
                         &decision) == 0 && decision.arm == 0u,
                 "Nightjar sleep arm setup failed");
    require_true(ds4_dspark_nightjar_observe(
                         &state, 0x91u, 0u, 0.012, 1u) == 0,
                 "Nightjar sleep observation failed");
    const ds4_dspark_nightjar_candidate wake_candidates[] = {
        {.arm = 0u, .predicted_loss = 0.012, .switch_loss = 0.0},
        {.arm = 1u, .predicted_loss = 0.010, .switch_loss = 0.005},
    };
    require_true(ds4_dspark_nightjar_observe(
                         &state, 0x91u, 1u, 0.010, 1u) == 0,
                 "Nightjar wake-arm observation failed");
    state.context[0].previous_arm = 0u;
    state.context[0].previous_valid = 1u;
    state.context[0].locked = 0u;
    require_true(ds4_dspark_nightjar_select(
                         &state, 0x91u, wake_candidates, 2u, 1.0,
                         &decision) == 0 && decision.arm == 0u,
                 "Nightjar must charge the K0-to-draft wake cost");

    ds4_dspark_nightjar_state_reset(&state);
    const ds4_dspark_nightjar_candidate positive_setup[] = {
        {.arm = 1u, .predicted_loss = 0.010, .switch_loss = 0.100},
        {.arm = 2u, .predicted_loss = 0.020, .switch_loss = 0.100},
    };
    require_true(ds4_dspark_nightjar_select(
                         &state, 0x92u, positive_setup, 2u, 1.0,
                         &decision) == 0 && decision.arm == 1u,
                 "Nightjar positive-arm setup failed");
    require_true(ds4_dspark_nightjar_observe(
                         &state, 0x92u, 1u, 0.010, 1u) == 0,
                 "Nightjar positive-arm observation failed");
    const ds4_dspark_nightjar_candidate positive_switch[] = {
        {.arm = 1u, .predicted_loss = 0.020, .switch_loss = 0.100},
        {.arm = 2u, .predicted_loss = 0.009, .switch_loss = 0.100},
    };
    require_true(ds4_dspark_nightjar_select(
                         &state, 0x92u, positive_switch, 2u, 1.0,
                         &decision) == 0 && decision.arm == 2u,
                 "Nightjar bootstrap must measure the second positive arm");
    require_true(ds4_dspark_nightjar_observe(
                         &state, 0x92u, 2u, 0.009, 1u) == 0,
                 "Nightjar second positive-arm observation failed");
    state.context[0].previous_arm = 1u;
    state.context[0].previous_valid = 1u;
    state.context[0].locked = 0u;
    require_true(ds4_dspark_nightjar_select(
                         &state, 0x92u, positive_switch, 2u, 1.0,
                         &decision) == 0 && decision.arm == 2u,
                 "Nightjar must not charge gamma-to-gamma switching");
}

static void test_nightjar_reward_is_token_weighted(void) {
    ds4_dspark_nightjar_state state;
    ds4_dspark_nightjar_state_reset(&state);
    const ds4_dspark_nightjar_candidate candidate[] = {
        {.arm = 2u, .predicted_loss = 0.060, .switch_loss = 0.0},
    };
    ds4_dspark_nightjar_decision decision;
    require_true(ds4_dspark_nightjar_select(
                         &state, 0xa1u, candidate, 1u, 1.0,
                         &decision) == 0,
                 "Nightjar weighted reward setup failed");
    require_true(ds4_dspark_nightjar_observe(
                         &state, 0xa1u, 2u, 0.100, 1u) == 0 &&
                 ds4_dspark_nightjar_observe(
                         &state, 0xa1u, 2u, 0.100, 3u) == 0,
                 "Nightjar weighted reward observations failed");
    require_close(state.context[0].arm[0].mean_loss, 0.050,
                  "Nightjar reward must be total latency / total tokens");
}

static void test_nightjar_mature_measurements_override_stale_prior(void) {
    ds4_dspark_nightjar_state state;
    ds4_dspark_nightjar_state_reset(&state);
    const uint64_t context = UINT64_C(0x7711);
    const ds4_dspark_nightjar_candidate initial[] = {
        {.arm = 2u, .predicted_loss = 0.030, .switch_loss = 0.0},
        {.arm = 5u, .predicted_loss = 0.040, .switch_loss = 0.0},
    };
    ds4_dspark_nightjar_decision decision;
    require_true(ds4_dspark_nightjar_select(
                     &state, context, initial, 2u, 2.0, &decision) == 0,
                 "Nightjar mature-prior setup failed");

    ds4_dspark_nightjar_context *ctx = &state.context[0];
    memset(ctx->arm, 0, sizeof(ctx->arm));
    ctx->observations = 8u;
    ctx->locked = 0u;
    ctx->bin_index = 100u;
    ctx->arm[0].in_use = 1u;
    ctx->arm[0].arm = 2u;
    ctx->arm[0].samples = 4u;
    ctx->arm[0].total_tokens = 4u;
    ctx->arm[0].total_latency = 0.200;
    ctx->arm[0].mean_loss = 0.050;
    ctx->arm[0].recent_loss = 0.050;
    ctx->arm[0].recent_samples = 4u;
    ctx->arm[1].in_use = 1u;
    ctx->arm[1].arm = 5u;
    ctx->arm[1].samples = 4u;
    ctx->arm[1].total_tokens = 4u;
    ctx->arm[1].total_latency = 0.160;
    ctx->arm[1].mean_loss = 0.040;
    ctx->arm[1].recent_loss = 0.040;
    ctx->arm[1].recent_samples = 4u;

    const ds4_dspark_nightjar_candidate stale[] = {
        {.arm = 2u, .predicted_loss = 0.020, .switch_loss = 0.0},
        {.arm = 5u, .predicted_loss = 0.080, .switch_loss = 0.0},
    };
    require_true(ds4_dspark_nightjar_select(
                     &state, context, stale, 2u, 2.0, &decision) == 0,
                 "Nightjar mature-prior selection failed");
    require_true(decision.arm == 5u,
                 "mature measured loss did not override the stale prior");
}

static void test_nightjar_seed_is_bounded_and_adaptive(void) {
    ds4_dspark_nightjar_state state;
    ds4_dspark_nightjar_state_reset(&state);
    const uint64_t context = UINT64_C(0x7331);
    require_true(ds4_dspark_nightjar_seed_arm(
                     &state, context, 2u, 0.050, 0.045, 100u, 40u) == 0,
                 "Nightjar persisted-arm seed failed");
    require_true(ds4_dspark_nightjar_seed_arm(
                     &state, context, 5u, 0.035, 0.030, 12u, 8u) == 0,
                 "Nightjar second persisted-arm seed failed");
    const ds4_dspark_nightjar_context *ctx = &state.context[0];
    require_true(ctx->observations == 16u && !ctx->locked,
                 "Nightjar warm start must cap evidence and discard locks");
    require_true(ctx->arm[0].samples == 8u &&
                 ctx->arm[0].recent_samples == 4u,
                 "Nightjar warm-start confidence was not bounded");

    const ds4_dspark_nightjar_candidate candidates[] = {
        {.arm = 2u, .predicted_loss = 0.025, .switch_loss = 0.0},
        {.arm = 5u, .predicted_loss = 0.060, .switch_loss = 0.0},
    };
    ds4_dspark_nightjar_decision decision;
    require_true(ds4_dspark_nightjar_select(
                     &state, context, candidates, 2u, 2.0,
                     &decision) == 0 && decision.arm == 5u,
                 "bounded warm start did not preserve mature arm ordering");
    require_true(ds4_dspark_nightjar_seed_arm(
                     &state, context, 1u, 0.0, 0.1, 4u, 4u) != 0,
                 "Nightjar accepted an invalid persisted loss");
}

static void test_nightjar_exposes_stale_lock_reference_loss(void) {
    ds4_dspark_nightjar_state state;
    ds4_dspark_nightjar_state_reset(&state);
    const uint64_t context = UINT64_C(0x7332);
    require_true(ds4_dspark_nightjar_seed_arm(
                     &state, context, 1u, 0.050, 0.050, 8u, 4u) == 0 &&
                 ds4_dspark_nightjar_seed_arm(
                     &state, context, 5u, 0.010, 0.010, 8u, 4u) == 0,
                 "Nightjar stale-lock setup failed");
    state.context[0].locked = 1u;
    state.context[0].locked_arm = 1u;

    const ds4_dspark_nightjar_candidate shifted[] = {
        {.arm = 1u, .predicted_loss = 0.050, .switch_loss = 0.0},
        {.arm = 5u, .predicted_loss = 0.010, .switch_loss = 0.0},
    };
    ds4_dspark_nightjar_decision decision;
    require_true(ds4_dspark_nightjar_select(
                     &state, context, shifted, 2u, 2.0,
                     &decision) == 0,
                 "Nightjar stale-lock decision failed");
    require_true(decision.estimated_loss >
                     decision.reference_loss * 1.03,
                 "Nightjar did not expose a guardable stale lock");
    require_true(ds4_dspark_nightjar_reject(
                     &state, context, decision.arm, 0u) == 0 &&
                 !state.context[0].locked &&
                 !state.context[0].previous_valid &&
                 state.context[0].arm[0].reject_until == 0u &&
                 state.context[0].arm[0].reject_streak == 0u &&
                 state.context[0].revoked_pending,
                 "Nightjar non-R3 rejection changed scheduling state");
}

static void test_nightjar_rejected_arm_cools_down_and_recovers(void) {
    ds4_dspark_nightjar_state state;
    ds4_dspark_nightjar_state_reset(&state);
    const uint64_t context = UINT64_C(0x7333);
    const ds4_dspark_nightjar_candidate candidates[] = {
        {.arm = 2u, .predicted_loss = 0.020, .switch_loss = 0.0},
        {.arm = 4u, .predicted_loss = 0.021, .switch_loss = 0.0},
    };
    ds4_dspark_nightjar_decision decision;
    require_true(ds4_dspark_nightjar_select(
                     &state, context, candidates, 2u, 1.10,
                     &decision) == 0 && decision.arm == 2u,
                 "Nightjar cooldown setup did not select best arm");
    require_true(ds4_dspark_nightjar_reject(
                     &state, context, 2u, 1u) == 0,
                 "Nightjar cooldown reject failed");
    require_true(ds4_dspark_nightjar_select(
                     &state, context, candidates, 2u, 1.10,
                     &decision) == 0 && decision.arm == 4u,
                 "Nightjar immediately retried a guarded arm");
    require_true(ds4_dspark_nightjar_observe(
                     &state, context, 4u, 0.021, 1u) == 0,
                 "Nightjar cooldown progress observation failed");
    require_true(ds4_dspark_nightjar_select(
                     &state, context, candidates, 2u, 1.10,
                     &decision) == 0 && decision.arm == 4u,
                 "Nightjar cooldown expired before its bounded horizon");
    require_true(ds4_dspark_nightjar_observe(
                     &state, context, 4u, 0.021, 1u) == 0,
                 "Nightjar cooldown expiry observation failed");
    require_true(ds4_dspark_nightjar_select(
                     &state, context, candidates, 2u, 1.10,
                     &decision) == 0 && decision.arm == 2u,
                 "Nightjar rejected arm did not recover after cooldown");
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

typedef struct {
    uint32_t calls;
    uint32_t historical_cliff_calls;
} shape_curve_probe;

static double test_shape_curve(
        const uint32_t *prefix,
        uint32_t request_count,
        uint32_t batch_size,
        void *opaque) {
    shape_curve_probe *probe = opaque;
    probe->calls++;
    uint32_t admitted = 0u;
    for (uint32_t r = 0; r < request_count; r++) {
        admitted += prefix[r];
    }
    require_true(batch_size == request_count + admitted,
                 "shape callback batch must match the prefix vector");
    if (request_count == 2u && admitted >= 2u) {
        probe->historical_cliff_calls++;
        return 2.0;
    }
    return 10.0;
}

static double test_shape_curve_fallback(
        const uint32_t *prefix,
        uint32_t request_count,
        uint32_t batch_size,
        void *opaque) {
    (void)prefix;
    (void)request_count;
    (void)batch_size;
    (void)opaque;
    return 0.0;
}

static void test_shape_aware_step_and_fallback(void) {
    const double sps[] = {0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0};
    ds4_dspark_schedule_item items[2] = {
        schedule_item(11, 0.95, 0.90),
        schedule_item(22, 0.90, 0.85),
    };
    ds4_dspark_scheduler_state shaped;
    ds4_dspark_scheduler_state fallback;
    ds4_dspark_scheduler_state_reset(&shaped);
    ds4_dspark_scheduler_state_reset(&fallback);
    ds4_dspark_schedule_step_result shaped_result;
    ds4_dspark_schedule_step_result fallback_result;
    shape_curve_probe probe = {0};

    require_true(ds4_dspark_hardware_schedule_step_shape(
                     &shaped, items, 2u, sps, 7u,
                     test_shape_curve, &probe, &shaped_result) == 0,
                 "shape-aware causal step failed");
    require_true(shaped_result.selected.admitted_candidates == 1u &&
                 shaped_result.selected.batch_size == 3u,
                 "exact shape cliff must stop the causal prefix at one row");
    require_true(probe.calls >= 3u && probe.historical_cliff_calls != 0u,
                 "shape callback must evaluate candidate shapes");

    require_true(ds4_dspark_hardware_schedule_step_shape(
                     &fallback, items, 2u, sps, 7u,
                     test_shape_curve_fallback, NULL,
                     &fallback_result) == 0,
                 "shape fallback step failed");
    ds4_dspark_scheduler_state reference;
    ds4_dspark_scheduler_state_reset(&reference);
    ds4_dspark_schedule_step_result reference_result;
    require_true(ds4_dspark_hardware_schedule_step(
                     &reference, items, 2u, sps, 7u,
                     &reference_result) == 0,
                 "row-only reference step failed");
    require_true(fallback_result.selected.batch_size ==
                     reference_result.selected.batch_size &&
                 fallback_result.selected.admitted_candidates ==
                     reference_result.selected.admitted_candidates,
                 "non-positive exact cost must preserve the row curve");
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

static void test_executor_pair_advances_history_once(void) {
    ds4_dspark_scheduler_state state;
    ds4_dspark_scheduler_state_reset(&state);
    ds4_dspark_schedule_item item = schedule_item(77, 0.95, 0.90);
    const double physical_sps[] = {0.0, 5.0, 8.0, 10.0};
    const double serial_sps[] = {0.0, 10.0, 5.0, 2.0};
    ds4_dspark_schedule_step_result physical;
    ds4_dspark_schedule_step_result serial;

    require_true(ds4_dspark_hardware_schedule_step_pair(
                     &state, &item, 1u,
                     physical_sps, serial_sps, 4u,
                     &physical, &serial) == 0,
                 "paired executor scheduling failed");
    require_true(physical.selected.prefix[0] == 2u &&
                 serial.selected.prefix[0] == 0u,
                 "executor curves must produce independent prefixes");
    require_true(state.step == 1u && state.entry_count == 1u &&
                 state.entries[0].history_count == 1u,
                 "paired scheduling must advance shared history once");

    require_true(ds4_dspark_hardware_schedule_step_pair(
                     &state, &item, 1u,
                     physical_sps, serial_sps, 4u,
                     &physical, &serial) == 0,
                 "second paired executor scheduling failed");
    require_true(state.step == 2u &&
                 state.entries[0].history_count == 2u,
                 "second paired step must retain two observations");

    item.request.conditional[0] = 0.40;
    item.request.conditional[1] = 0.20;
    require_true(ds4_dspark_hardware_schedule_step_pair(
                     &state, &item, 1u,
                     physical_sps, serial_sps, 4u,
                     &physical, &serial) == 0,
                 "asynchronous paired executor scheduling failed");
    require_true(state.step == 3u &&
                 physical.used_async == 1u &&
                 serial.used_async == 1u,
                 "both executor candidates must share exact t-2 readiness");
}

typedef struct {
    uint32_t calls;
    uint32_t prefer_long;
} pair_shape_probe;

static double test_pair_shape_curve(
        const uint32_t *prefix,
        uint32_t request_count,
        uint32_t batch_size,
        void *opaque) {
    pair_shape_probe *probe = (pair_shape_probe *)opaque;
    require_true(probe != NULL && prefix != NULL &&
                 request_count == 1u &&
                 batch_size == prefix[0] + 1u,
                 "paired shape callback contract");
    probe->calls++;
    if (probe->prefer_long) {
        return prefix[0] == 2u ? 20.0 :
               prefix[0] == 1u ? 10.0 : 5.0;
    }
    return prefix[0] == 0u ? 20.0 : 1.0;
}

static void test_executor_pair_uses_independent_shapes(void) {
    ds4_dspark_scheduler_state state;
    ds4_dspark_scheduler_state_reset(&state);
    ds4_dspark_schedule_item item = schedule_item(91, 0.95, 0.90);
    const double same_sps[] = {0.0, 5.0, 5.0, 5.0};
    pair_shape_probe physical_probe = {.prefer_long = 1u};
    pair_shape_probe serial_probe = {.prefer_long = 0u};
    ds4_dspark_schedule_step_result physical;
    ds4_dspark_schedule_step_result serial;

    require_true(ds4_dspark_hardware_schedule_step_pair_shape(
                     &state, &item, 1u,
                     same_sps,
                     test_pair_shape_curve, &physical_probe,
                     same_sps,
                     test_pair_shape_curve, &serial_probe,
                     4u, &physical, &serial) == 0,
                 "paired shape-aware scheduling failed");
    require_true(physical.selected.prefix[0] == 2u &&
                 serial.selected.prefix[0] == 0u,
                 "each executor must optimize its own exact shape");
    require_true(physical_probe.calls >= 3u &&
                 serial_probe.calls >= 2u,
                 "both paired shape curves must be evaluated");
    require_true(state.step == 1u &&
                 state.entries[0].history_count == 1u,
                 "paired shape scheduling must commit history once");
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
    test_fixed_capacity_reranks_current_requests();
    test_nightjar_capacity_does_not_lock_historical_lane();
    test_nightjar_warm_start_and_bin_lock();
    test_peek_delayed_request_is_exact_and_read_only();
    test_nightjar_skips_unsafe_bootstrap_arms();
    test_nightjar_safe_exploration_uses_observed_loss();
    test_nightjar_revokes_a_degraded_lock();
    test_nightjar_switch_cost_only_wakes_drafter();
    test_nightjar_reward_is_token_weighted();
    test_nightjar_mature_measurements_override_stale_prior();
    test_nightjar_seed_is_bounded_and_adaptive();
    test_nightjar_exposes_stale_lock_reference_loss();
    test_nightjar_rejected_arm_cools_down_and_recovers();
    test_shape_aware_step_and_fallback();
    test_stateful_two_step_barrier();
    test_executor_pair_advances_history_once();
    test_executor_pair_uses_independent_shapes();
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
