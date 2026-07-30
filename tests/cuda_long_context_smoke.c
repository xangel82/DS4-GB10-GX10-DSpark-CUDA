#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static double getenv_seconds(const char *name, double fallback) {
    const char *s = getenv(name);
    if (!s || !s[0]) return fallback;
    char *end = NULL;
    const double v = strtod(s, &end);
    return end != s && v > 0.0 ? v : fallback;
}

static int check_large_topk(void) {
    const uint32_t n_comp = 32768;
    const uint32_t n_tokens = 32;
    const uint32_t top_k = 512;
    const uint64_t score_count = (uint64_t)n_comp * n_tokens;
    float *scores_host = (float *)malloc((size_t)score_count * sizeof(float));
    uint32_t *selected_host = (uint32_t *)malloc((size_t)n_tokens * top_k * sizeof(uint32_t));
    if (!scores_host || !selected_host) return 1;

    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t i = 0; i < n_comp; i++) {
            float value = (float)i;
            if (t == 1u) value = 42.0f;
            if (t == 2u) value = -(float)i;
            if (t == 3u) value = NAN;
            if (t == 4u) value = -INFINITY;
            if (t == 4u && i == 0u) value = INFINITY;
            if (t == 4u && i == 1u) value = 0.0f;
            if (t == 4u && i == 2u) value = -0.0f;
            if (t == 5u) value = (i & 1u) ? -0.0f : 0.0f;
            scores_host[(uint64_t)t * n_comp + i] = value;
        }
    }

    ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc(score_count * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc((uint64_t)n_tokens * top_k * sizeof(uint32_t));
    int rc = 1;
    double elapsed = 0.0;
    if (scores && selected &&
        ds4_gpu_tensor_write(scores, 0, scores_host, score_count * sizeof(float))) {
        /* Exclude one-time CUDA module/kernel setup from the throughput guard. */
        if (!ds4_gpu_indexer_topk_tensor(selected, scores, n_comp, n_tokens, top_k) ||
            !ds4_gpu_synchronize()) {
            rc = 1;
            goto cleanup;
        }
        const double t0 = monotonic_seconds();
        if (ds4_gpu_indexer_topk_tensor(selected, scores, n_comp, n_tokens, top_k) &&
            ds4_gpu_synchronize()) {
            elapsed = monotonic_seconds() - t0;
            rc = ds4_gpu_tensor_read(selected, 0, selected_host,
                                     (uint64_t)n_tokens * top_k * sizeof(uint32_t)) ? 0 : 1;
        }
    }
    if (rc == 0) {
        for (uint32_t t = 0; t < n_tokens && rc == 0; t++) {
            for (uint32_t i = 0; i < top_k; i++) {
                const uint32_t expected = (t >= 1u && t <= 5u)
                    ? i
                    : n_comp - 1u - i;
                const uint32_t got = selected_host[(uint64_t)t * top_k + i];
                if (got != expected) {
                    fprintf(stderr, "top-k mismatch token=%u rank=%u got=%u expected=%u\n",
                            t, i, got, expected);
                    rc = 1;
                    break;
                }
            }
        }
    }
    if (rc == 0) {
        const double max_seconds = getenv_seconds("DS4_CUDA_TOPK_REGRESSION_SEC", 2.0);
        fprintf(stderr, "cuda-regression: top-k n_comp=%u n_tokens=%u elapsed=%.3fs\n",
                n_comp, n_tokens, elapsed);
        if (elapsed > max_seconds) {
            fprintf(stderr, "top-k regression: %.3fs exceeds %.3fs\n", elapsed, max_seconds);
            rc = 1;
        }
    }

cleanup:
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(scores);
    free(selected_host);
    free(scores_host);
    return rc;
}

static int check_frontier_copy_primitives(void) {
    enum { RING_CAP = 8, ROWS = 4, WIDTH = 4 };
    float ring_host[RING_CAP * WIDTH];
    float zero_host[RING_CAP * WIDTH] = {0};
    float got_host[RING_CAP * WIDTH] = {0};
    float state_host[WIDTH] = {11.0f, 12.0f, 13.0f, 14.0f};
    float state_got[WIDTH] = {0};
    for (uint32_t i = 0; i < RING_CAP * WIDTH; i++) {
        ring_host[i] = (float)(1000u + i);
    }

    ds4_gpu_tensor *ring = ds4_gpu_tensor_alloc(sizeof(ring_host));
    ds4_gpu_tensor *backup =
        ds4_gpu_tensor_alloc((uint64_t)ROWS * WIDTH * sizeof(float));
    ds4_gpu_tensor *state = ds4_gpu_tensor_alloc(sizeof(state_host));
    ds4_gpu_tensor *state_backup = ds4_gpu_tensor_alloc(sizeof(state_host));
    int rc = 1;
    if (!ring || !backup || !state || !state_backup ||
        !ds4_gpu_tensor_write(ring, 0, ring_host, sizeof(ring_host)) ||
        !ds4_gpu_tensor_write(state, 0, state_host, sizeof(state_host)) ||
        !ds4_gpu_ring_rows_save_tensor(backup, ring, RING_CAP, 6, ROWS, WIDTH) ||
        !ds4_gpu_tensor_copy_async(state_backup, 0, state, 0,
                                   sizeof(state_host)) ||
        !ds4_gpu_end_commands() ||
        !ds4_gpu_tensor_write(ring, 0, zero_host, sizeof(zero_host)) ||
        !ds4_gpu_tensor_write(state, 0, zero_host, sizeof(state_host)) ||
        !ds4_gpu_ring_rows_restore_tensor(ring, backup, RING_CAP, 6, 0,
                                          ROWS, WIDTH) ||
        !ds4_gpu_tensor_copy_async(state, 0, state_backup, 0,
                                   sizeof(state_host)) ||
        !ds4_gpu_end_commands() ||
        !ds4_gpu_tensor_read(ring, 0, got_host, sizeof(got_host)) ||
        !ds4_gpu_tensor_read(state, 0, state_got, sizeof(state_got))) {
        goto cleanup;
    }

    for (uint32_t r = 0; r < ROWS; r++) {
        const uint32_t phys = (6u + r) % RING_CAP;
        for (uint32_t c = 0; c < WIDTH; c++) {
            const float expected = ring_host[phys * WIDTH + c];
            if (got_host[phys * WIDTH + c] != expected) goto cleanup;
        }
    }
    if (memcmp(state_host, state_got, sizeof(state_host)) != 0) goto cleanup;
    fprintf(stderr,
            "cuda-regression: frontier ring + batched async D2D restore OK\n");
    rc = 0;

cleanup:
    ds4_gpu_tensor_free(state_backup);
    ds4_gpu_tensor_free(state);
    ds4_gpu_tensor_free(backup);
    ds4_gpu_tensor_free(ring);
    return rc;
}

static int check_dspark_hybrid_blockv(void) {
    enum { VOCAB = 2, ROWS = 2 };
    const float equal_logits[(ROWS + 1) * VOCAB] = {
        logf(0.25f), logf(0.75f),
        logf(0.25f), logf(0.75f),
        logf(0.25f), logf(0.75f),
    };
    const float equal_q[ROWS * VOCAB] = {
        0.25f, 0.75f,
        0.25f, 0.75f,
    };
    const int32_t equal_tokens[ROWS + 1] = {0, 1, 0};
    const float uniforms[ROWS] = {0.99f, 0.99f};
    int32_t got_tokens[ROWS] = {-1, -1};
    int32_t got_accept[ROWS] = {0, 0};

    ds4_gpu_tensor *logits =
        ds4_gpu_tensor_alloc(sizeof(equal_logits));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(sizeof(equal_q));
    ds4_gpu_tensor *tokens =
        ds4_gpu_tensor_alloc(sizeof(equal_tokens));
    ds4_gpu_tensor *accept_uniforms =
        ds4_gpu_tensor_alloc(sizeof(uniforms));
    ds4_gpu_tensor *residual_uniforms =
        ds4_gpu_tensor_alloc(sizeof(uniforms));
    ds4_gpu_tensor *out_tokens =
        ds4_gpu_tensor_alloc(sizeof(got_tokens));
    ds4_gpu_tensor *out_accept =
        ds4_gpu_tensor_alloc(sizeof(got_accept));
    int rc = 1;
    if (!logits || !q || !tokens || !accept_uniforms ||
        !residual_uniforms || !out_tokens || !out_accept ||
        !ds4_gpu_tensor_write(logits, 0, equal_logits,
                              sizeof(equal_logits)) ||
        !ds4_gpu_tensor_write(q, 0, equal_q, sizeof(equal_q)) ||
        !ds4_gpu_tensor_write(tokens, 0, equal_tokens,
                              sizeof(equal_tokens)) ||
        !ds4_gpu_tensor_write(accept_uniforms, 0, uniforms,
                              sizeof(uniforms)) ||
        !ds4_gpu_tensor_write(residual_uniforms, 0, uniforms,
                              sizeof(uniforms)) ||
        !ds4_gpu_dspark_block_verify_tensor(
                out_tokens, out_accept, logits, q, tokens,
                accept_uniforms, residual_uniforms,
                ROWS, VOCAB, 1.0f, 0.0f) ||
        !ds4_gpu_tensor_read(out_tokens, 0, got_tokens,
                             sizeof(got_tokens)) ||
        !ds4_gpu_tensor_read(out_accept, 0, got_accept,
                             sizeof(got_accept))) {
        goto cleanup;
    }
    if (got_accept[0] != 1 || got_accept[1] != 1) goto cleanup;

    {
        const float edge_uniforms[][ROWS] = {
            {0.0f, 0.0f},
            {nextafterf(0.0f, 1.0f), nextafterf(0.0f, 1.0f)},
            {nextafterf(1.0f, 0.0f), nextafterf(1.0f, 0.0f)},
        };
        for (uint32_t edge = 0;
             edge < sizeof(edge_uniforms) / sizeof(edge_uniforms[0]);
             edge++) {
            memset(got_accept, 0, sizeof(got_accept));
            if (!ds4_gpu_tensor_write(
                        accept_uniforms, 0, edge_uniforms[edge],
                        sizeof(edge_uniforms[edge])) ||
                !ds4_gpu_tensor_write(
                        residual_uniforms, 0, edge_uniforms[edge],
                        sizeof(edge_uniforms[edge])) ||
                !ds4_gpu_dspark_block_verify_tensor(
                        out_tokens, out_accept, logits, q, tokens,
                        accept_uniforms, residual_uniforms,
                        ROWS, VOCAB, 1.0f, 0.0f) ||
                !ds4_gpu_tensor_read(out_accept, 0, got_accept,
                                     sizeof(got_accept))) {
                goto cleanup;
            }
            if (got_accept[0] != 1 || got_accept[1] != 1) goto cleanup;
        }
    }

    {
        const float reject_logits[2 * VOCAB] = {
            logf(0.7f), logf(0.3f),
            logf(0.7f), logf(0.3f),
        };
        const float reject_q[VOCAB] = {0.2f, 0.8f};
        const int32_t reject_tokens[2] = {0, 1};
        const float reject_uniform = 0.9f;
        if (!ds4_gpu_tensor_write(logits, 0, reject_logits,
                                  sizeof(reject_logits)) ||
            !ds4_gpu_tensor_write(q, 0, reject_q, sizeof(reject_q)) ||
            !ds4_gpu_tensor_write(tokens, 0, reject_tokens,
                                  sizeof(reject_tokens)) ||
            !ds4_gpu_tensor_write(accept_uniforms, 0, &reject_uniform,
                                  sizeof(reject_uniform)) ||
            !ds4_gpu_tensor_write(residual_uniforms, 0, &reject_uniform,
                                  sizeof(reject_uniform)) ||
            !ds4_gpu_dspark_block_verify_tensor(
                    out_tokens, out_accept, logits, q, tokens,
                    accept_uniforms, residual_uniforms,
                    1, VOCAB, 1.0f, 0.0f) ||
            !ds4_gpu_tensor_read(out_tokens, 0, got_tokens,
                                 sizeof(got_tokens[0])) ||
            !ds4_gpu_tensor_read(out_accept, 0, got_accept,
                                 sizeof(got_accept[0]))) {
            goto cleanup;
        }
        if (got_accept[0] != 0 || got_tokens[0] != 0) goto cleanup;
    }

    {
        float zero_q[ROWS * VOCAB] = {0};
        const int32_t one_hot_tokens[ROWS + 1] = {0, 1, 0};
        float one_hot_got[ROWS * VOCAB] = {0};
        if (!ds4_gpu_tensor_write(q, 0, zero_q, sizeof(zero_q)) ||
            !ds4_gpu_tensor_write(tokens, 0, one_hot_tokens,
                                  sizeof(one_hot_tokens)) ||
            !ds4_gpu_dspark_one_hot_draft_rows_tensor(
                    q, tokens, 0, ROWS, VOCAB) ||
            !ds4_gpu_tensor_read(q, 0, one_hot_got,
                                 sizeof(one_hot_got))) {
            goto cleanup;
        }
        if (one_hot_got[0] != 0.0f || one_hot_got[1] != 1.0f ||
            one_hot_got[2] != 1.0f || one_hot_got[3] != 0.0f) {
            goto cleanup;
        }

        const uint8_t sparse_sizes[ROWS] = {2, 1};
        const int32_t sparse_ids[ROWS * VOCAB] = {0, 1, 1, 0};
        const float sparse_probs[ROWS * VOCAB] = {
            0.25f, 0.75f, 1.0f, 0.0f
        };
        const int32_t sparse_proposed[ROWS] = {1, 1};
        memset(one_hot_got, 0, sizeof(one_hot_got));
        if (!ds4_gpu_dspark_sparse_draft_rows_tensor(
                    q, 0, ROWS, VOCAB,
                    sparse_sizes, sparse_ids, sparse_probs,
                    sparse_proposed, VOCAB) ||
            !ds4_gpu_tensor_read(q, 0, one_hot_got,
                                 sizeof(one_hot_got))) {
            goto cleanup;
        }
        if (fabsf(one_hot_got[0] - 0.25f) > 1.0e-7f ||
            fabsf(one_hot_got[1] - 0.75f) > 1.0e-7f ||
            one_hot_got[2] != 0.0f || one_hot_got[3] != 1.0f) {
            goto cleanup;
        }

        {
            const uint8_t two[1] = {2};
            const uint8_t one[1] = {1};
            const int32_t duplicate_ids[2] = {0, 0};
            const int32_t valid_ids[2] = {0, 1};
            const int32_t out_of_range_ids[2] = {0, VOCAB};
            const float one_prob[1] = {1.0f};
            const float half_probs[2] = {0.5f, 0.5f};
            const float bad_sum_probs[2] = {0.4f, 0.5f};
            const float nan_probs[2] = {NAN, 1.0f};
            const int32_t proposed_zero[1] = {0};
            const int32_t proposed_one[1] = {1};
            if (ds4_gpu_dspark_sparse_draft_rows_tensor(
                        q, 0, 1, VOCAB, two, duplicate_ids, half_probs,
                        proposed_zero, 2) ||
                ds4_gpu_dspark_sparse_draft_rows_tensor(
                        q, 0, 1, VOCAB, one, valid_ids, one_prob,
                        proposed_one, 2) ||
                ds4_gpu_dspark_sparse_draft_rows_tensor(
                        q, 0, 1, VOCAB, two, valid_ids, bad_sum_probs,
                        proposed_zero, 2) ||
                ds4_gpu_dspark_sparse_draft_rows_tensor(
                        q, 0, 1, VOCAB, two, valid_ids, nan_probs,
                        proposed_zero, 2) ||
                ds4_gpu_dspark_sparse_draft_rows_tensor(
                        q, 0, 1, VOCAB, two, out_of_range_ids, half_probs,
                        proposed_zero, 2) ||
                ds4_gpu_dspark_one_hot_draft_rows_tensor(
                        q, tokens, 15, 1, VOCAB) ||
                ds4_gpu_dspark_block_verify_tensor(
                        out_tokens, out_accept, logits, q, tokens,
                        accept_uniforms, residual_uniforms,
                        16, VOCAB, 1.0f, 0.0f) ||
                ds4_gpu_dspark_block_verify_tensor(
                        out_tokens, out_accept, logits, q, tokens,
                        accept_uniforms, residual_uniforms,
                        1, VOCAB, 0.0f, 0.0f)) {
                goto cleanup;
            }
        }
    }

    {
        enum { LONG_ROWS = 15, LONG_VOCAB = 2 };
        float long_logits[(LONG_ROWS + 1) * LONG_VOCAB];
        uint8_t long_sizes[LONG_ROWS];
        int32_t long_ids[LONG_ROWS * LONG_VOCAB];
        float long_probs[LONG_ROWS * LONG_VOCAB];
        int32_t long_tokens[LONG_ROWS + 1];
        int32_t long_proposed[LONG_ROWS];
        float long_uniforms[LONG_ROWS];
        int32_t long_accept[LONG_ROWS];
        for (int i = 0; i < (LONG_ROWS + 1) * LONG_VOCAB; i++) {
            long_logits[i] = logf(0.5f);
        }
        long_tokens[0] = 0;
        for (int row = 0; row < LONG_ROWS; row++) {
            long_sizes[row] = 2;
            long_ids[row * LONG_VOCAB] = 0;
            long_ids[row * LONG_VOCAB + 1] = 1;
            long_probs[row * LONG_VOCAB] = 0.5f;
            long_probs[row * LONG_VOCAB + 1] = 0.5f;
            long_proposed[row] = row & 1;
            long_tokens[row + 1] = long_proposed[row];
            long_uniforms[row] = 0.99f;
            long_accept[row] = 0;
        }
        ds4_gpu_tensor *long_logits_tensor =
            ds4_gpu_tensor_alloc(sizeof(long_logits));
        ds4_gpu_tensor *long_q =
            ds4_gpu_tensor_alloc(sizeof(long_probs));
        ds4_gpu_tensor *long_tokens_tensor =
            ds4_gpu_tensor_alloc(sizeof(long_tokens));
        ds4_gpu_tensor *long_uniform_tensor =
            ds4_gpu_tensor_alloc(sizeof(long_uniforms));
        ds4_gpu_tensor *long_out_tokens =
            ds4_gpu_tensor_alloc(sizeof(long_accept));
        ds4_gpu_tensor *long_out_accept =
            ds4_gpu_tensor_alloc(sizeof(long_accept));
        const bool long_ok =
            long_logits_tensor && long_q && long_tokens_tensor &&
            long_uniform_tensor && long_out_tokens && long_out_accept &&
            ds4_gpu_tensor_write(long_logits_tensor, 0, long_logits,
                                 sizeof(long_logits)) &&
            ds4_gpu_tensor_write(long_tokens_tensor, 0, long_tokens,
                                 sizeof(long_tokens)) &&
            ds4_gpu_tensor_write(long_uniform_tensor, 0, long_uniforms,
                                 sizeof(long_uniforms)) &&
            ds4_gpu_dspark_sparse_draft_rows_tensor(
                    long_q, 0, LONG_ROWS, LONG_VOCAB,
                    long_sizes, long_ids, long_probs, long_proposed,
                    LONG_VOCAB) &&
            ds4_gpu_dspark_block_verify_tensor(
                    long_out_tokens, long_out_accept,
                    long_logits_tensor, long_q, long_tokens_tensor,
                    long_uniform_tensor, long_uniform_tensor,
                    LONG_ROWS, LONG_VOCAB, 1.0f, 0.0f) &&
            ds4_gpu_tensor_read(long_out_accept, 0, long_accept,
                                sizeof(long_accept));
        ds4_gpu_tensor_free(long_out_accept);
        ds4_gpu_tensor_free(long_out_tokens);
        ds4_gpu_tensor_free(long_uniform_tensor);
        ds4_gpu_tensor_free(long_tokens_tensor);
        ds4_gpu_tensor_free(long_q);
        ds4_gpu_tensor_free(long_logits_tensor);
        if (!long_ok) goto cleanup;
        for (int row = 0; row < LONG_ROWS; row++) {
            if (long_accept[row] != 1) goto cleanup;
        }
    }

    fprintf(stderr,
            "cuda-regression: HybridLC sparse q + lossless BlockV "
            "(rows=1..15) OK\n");
    rc = 0;

cleanup:
    ds4_gpu_tensor_free(out_accept);
    ds4_gpu_tensor_free(out_tokens);
    ds4_gpu_tensor_free(residual_uniforms);
    ds4_gpu_tensor_free(accept_uniforms);
    ds4_gpu_tensor_free(tokens);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(logits);
    return rc;
}

static int check_gvr_topk(void) {
    const uint32_t n_comp = 16384;
    const uint32_t n_tokens = 2;
    const uint32_t top_k = 512;
    const uint64_t score_count = (uint64_t)n_comp * n_tokens;
    const uint64_t selected_count = (uint64_t)n_tokens * top_k;
    float *previous_host = (float *)malloc((size_t)score_count * sizeof(float));
    float *current_host = (float *)malloc((size_t)score_count * sizeof(float));
    uint32_t *hint_host = (uint32_t *)malloc(
            (size_t)selected_count * sizeof(uint32_t));
    uint32_t *expected_host = (uint32_t *)malloc(
            (size_t)selected_count * sizeof(uint32_t));
    uint32_t *got_host = (uint32_t *)malloc(
            (size_t)selected_count * sizeof(uint32_t));
    uint8_t fallback_host[2] = {0, 0};
    if (!previous_host || !current_host || !hint_host ||
        !expected_host || !got_host) {
        free(got_host);
        free(expected_host);
        free(hint_host);
        free(current_host);
        free(previous_host);
        return 1;
    }

    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t i = 0; i < n_comp; i++) {
            const float base = -(float)i - (float)t * 0.125f;
            previous_host[(uint64_t)t * n_comp + i] = base;
            current_host[(uint64_t)t * n_comp + i] =
                base + 0.001f * sinf((float)(i + 17u * t));
        }
    }

    ds4_gpu_tensor *previous = ds4_gpu_tensor_alloc(score_count * sizeof(float));
    ds4_gpu_tensor *current = ds4_gpu_tensor_alloc(score_count * sizeof(float));
    ds4_gpu_tensor *hint = ds4_gpu_tensor_alloc(selected_count * sizeof(uint32_t));
    ds4_gpu_tensor *expected = ds4_gpu_tensor_alloc(selected_count * sizeof(uint32_t));
    ds4_gpu_tensor *got = ds4_gpu_tensor_alloc(selected_count * sizeof(uint32_t));
    ds4_gpu_tensor *fallback = ds4_gpu_tensor_alloc(n_tokens);
    int rc = 1;
    if (!previous || !current || !hint || !expected || !got || !fallback ||
        !ds4_gpu_tensor_write(previous, 0, previous_host,
                              score_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(current, 0, current_host,
                              score_count * sizeof(float)) ||
        !ds4_gpu_indexer_topk_tensor(hint, previous, n_comp, n_tokens, top_k) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(hint, 0, hint_host,
                             selected_count * sizeof(uint32_t))) {
        goto cleanup;
    }

    /* n_tokens=2 and n_comp>8192 must exercise the parallel small-batch
     * chunk tree rather than the prefill Radix path.  The synthetic rows are
     * strictly descending, so this is also an independent exact-order check. */
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t rank = 0; rank < top_k; rank++) {
            const uint32_t value = hint_host[(uint64_t)t * top_k + rank];
            if (value != rank) {
                fprintf(stderr,
                        "small-batch top-k mismatch token=%u rank=%u got=%u\n",
                        t, rank, value);
                goto cleanup;
            }
        }
    }

    /* Token zero exercises the temporal fast path. Token one deliberately
     * corrupts one hint so the exact masked-Radix fallback is also covered. */
    hint_host[top_k] = n_comp;
    if (!ds4_gpu_tensor_write(hint, 0, hint_host,
                              selected_count * sizeof(uint32_t)) ||
        !ds4_gpu_indexer_topk_tensor(expected, current,
                                     n_comp, n_tokens, top_k) ||
        !ds4_gpu_indexer_topk_gvr_tensor(got, current, hint, fallback,
                                         n_comp, n_tokens) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(expected, 0, expected_host,
                             selected_count * sizeof(uint32_t)) ||
        !ds4_gpu_tensor_read(got, 0, got_host,
                             selected_count * sizeof(uint32_t)) ||
        !ds4_gpu_tensor_read(fallback, 0, fallback_host, n_tokens)) {
        goto cleanup;
    }
    for (uint64_t i = 0; i < selected_count; i++) {
        if (got_host[i] != expected_host[i]) {
            fprintf(stderr,
                    "GVR top-k mismatch token=%u rank=%u got=%u expected=%u\n",
                    (uint32_t)(i / top_k), (uint32_t)(i % top_k),
                    got_host[i], expected_host[i]);
            goto cleanup;
        }
    }
    if (fallback_host[0] != 0u || fallback_host[1] != 1u) {
        fprintf(stderr,
                "GVR dispatch mismatch fast=%u fallback=%u\n",
                (unsigned)fallback_host[0], (unsigned)fallback_host[1]);
        goto cleanup;
    }
    fprintf(stderr,
            "cuda-regression: exact small-batch chunk-tree + GVR Top-512 "
            "fast-path/fallback OK\n");
    rc = 0;

cleanup:
    ds4_gpu_tensor_free(fallback);
    ds4_gpu_tensor_free(got);
    ds4_gpu_tensor_free(expected);
    ds4_gpu_tensor_free(hint);
    ds4_gpu_tensor_free(current);
    ds4_gpu_tensor_free(previous);
    free(got_host);
    free(expected_host);
    free(hint_host);
    free(current_host);
    free(previous_host);
    return rc;
}

static int check_decode_attention_overflow_path(void) {
    const uint32_t n_head = 8;
    const uint32_t head_dim = 512;
    const uint32_t n_raw = 128;
    const uint32_t n_comp = 8100;
    const uint64_t q_count = (uint64_t)n_head * head_dim;
    const uint64_t raw_count = (uint64_t)n_raw * head_dim;
    const uint64_t comp_count = (uint64_t)n_comp * head_dim;

    float *sinks = (float *)calloc(n_head, sizeof(float));
    float *q_host = (float *)calloc((size_t)q_count, sizeof(float));
    float *raw_host = (float *)calloc((size_t)raw_count, sizeof(float));
    float *comp_host = (float *)calloc((size_t)comp_count, sizeof(float));
    float *heads_host = (float *)calloc((size_t)q_count, sizeof(float));
    if (!sinks || !q_host || !raw_host || !comp_host || !heads_host) return 1;

    for (uint32_t c = 0; c < n_comp; c++) {
        comp_host[(uint64_t)c * head_dim] = 1.0f;
    }

    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_count * sizeof(float));
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(comp_count * sizeof(float));
    int rc = 1;
    if (heads && q && raw && comp &&
        ds4_gpu_tensor_write(q, 0, q_host, q_count * sizeof(float)) &&
        ds4_gpu_tensor_write(raw, 0, raw_host, raw_count * sizeof(float)) &&
        ds4_gpu_tensor_write(comp, 0, comp_host, comp_count * sizeof(float)) &&
        ds4_gpu_attention_decode_heads_tensor(heads,
                                              sinks,
                                              n_head * sizeof(float),
                                              0,
                                              q,
                                              raw,
                                              n_raw,
                                              n_raw,
                                              0,
                                              comp,
                                              0,
                                              n_comp,
                                              NULL,
                                              0,
                                              n_head,
                                              head_dim) &&
        ds4_gpu_synchronize() &&
        ds4_gpu_tensor_read(heads, 0, heads_host, q_count * sizeof(float))) {
        rc = 0;
        for (uint32_t h = 0; h < n_head; h++) {
            const float v = heads_host[(uint64_t)h * head_dim];
            if (v < 0.90f) {
                fprintf(stderr, "attention fallback ignored compressed rows for head=%u value=%f\n",
                        h, (double)v);
                rc = 1;
            }
        }
    }

    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(heads);
    free(heads_host);
    free(comp_host);
    free(raw_host);
    free(q_host);
    free(sinks);
    return rc;
}

static int check_physical_rn_dspark_attention(void) {
    enum {
        REQUESTS = 3,
        BLOCK_TOKENS = 5,
        N_HEAD = 2,
        HEAD_DIM = 8,
        MAIN_CAP = 8,
        CONTEXT_STRIDE = MAIN_CAP + BLOCK_TOKENS,
    };
    const uint32_t n_main[REQUESTS] = {6, 7, 8};
    const uint32_t main_start[REQUESTS] = {2, 5, 1};
    const uint32_t total_rows = REQUESTS * BLOCK_TOKENS;
    const uint64_t head_count =
        (uint64_t)total_rows * N_HEAD * HEAD_DIM;
    const uint64_t draft_count =
        (uint64_t)total_rows * HEAD_DIM;
    const uint64_t main_count =
        (uint64_t)MAIN_CAP * HEAD_DIM;
    static float sinks_page[1024] __attribute__((aligned(4096)));
    float *sinks = sinks_page;
    sinks[0] = -0.25f;
    sinks[1] = 0.125f;

    float q_host[head_count];
    float draft_host[draft_count];
    float main_host[REQUESTS][main_count];
    float rn_host[head_count];
    float ref_host[head_count];
    for (uint64_t i = 0; i < head_count; i++) {
        q_host[i] =
            ((float)((i * 17u) % 43u) - 21.0f) / 29.0f;
    }
    for (uint64_t i = 0; i < draft_count; i++) {
        draft_host[i] =
            ((float)((i * 13u) % 37u) - 18.0f) / 23.0f;
    }
    for (uint32_t r = 0; r < REQUESTS; r++) {
        for (uint64_t i = 0; i < main_count; i++) {
            main_host[r][i] =
                ((float)(((i + 11u * r) * 19u) % 47u) - 23.0f) /
                31.0f;
        }
    }
    memset(rn_host, 0, sizeof(rn_host));
    memset(ref_host, 0, sizeof(ref_host));

    ds4_gpu_tensor *heads_rn =
        ds4_gpu_tensor_alloc(sizeof(rn_host));
    ds4_gpu_tensor *heads_ref =
        ds4_gpu_tensor_alloc(sizeof(ref_host));
    ds4_gpu_tensor *q =
        ds4_gpu_tensor_alloc(sizeof(q_host));
    ds4_gpu_tensor *draft =
        ds4_gpu_tensor_alloc(sizeof(draft_host));
    ds4_gpu_tensor *context_rn = ds4_gpu_tensor_alloc(
        (uint64_t)REQUESTS * CONTEXT_STRIDE * HEAD_DIM *
        sizeof(float));
    ds4_gpu_tensor *main_kv[REQUESTS] = {0};
    ds4_gpu_tensor *heads_slice[REQUESTS] = {0};
    ds4_gpu_tensor *q_slice[REQUESTS] = {0};
    ds4_gpu_tensor *draft_slice[REQUESTS] = {0};
    ds4_gpu_tensor *context_slice[REQUESTS] = {0};
    int rc = 1;
    const uint64_t head_slice_bytes =
        (uint64_t)BLOCK_TOKENS * N_HEAD * HEAD_DIM * sizeof(float);
    const uint64_t draft_slice_bytes =
        (uint64_t)BLOCK_TOKENS * HEAD_DIM * sizeof(float);
    bool ok = heads_rn && heads_ref && q && draft && context_rn &&
              ds4_gpu_tensor_write(
                  q, 0, q_host, sizeof(q_host)) != 0 &&
              ds4_gpu_tensor_write(
                  draft, 0, draft_host, sizeof(draft_host)) != 0;
    for (uint32_t r = 0; ok && r < REQUESTS; r++) {
        main_kv[r] = ds4_gpu_tensor_alloc(
                main_count * sizeof(float));
        heads_slice[r] = ds4_gpu_tensor_view(
                heads_ref,
                (uint64_t)r * head_slice_bytes,
                head_slice_bytes);
        q_slice[r] = ds4_gpu_tensor_view(
                q,
                (uint64_t)r * head_slice_bytes,
                head_slice_bytes);
        draft_slice[r] = ds4_gpu_tensor_view(
                draft,
                (uint64_t)r * draft_slice_bytes,
                draft_slice_bytes);
        context_slice[r] = ds4_gpu_tensor_alloc(
                (uint64_t)(n_main[r] + BLOCK_TOKENS) *
                HEAD_DIM * sizeof(float));
        ok = main_kv[r] && heads_slice[r] && q_slice[r] &&
             draft_slice[r] && context_slice[r] &&
             ds4_gpu_tensor_write(
                 main_kv[r],
                 0,
                 main_host[r],
                 main_count * sizeof(float)) != 0;
    }
    for (uint32_t r = 0; ok && r < REQUESTS; r++) {
        ok = ds4_gpu_dspark_attention_heads_tensor(
                heads_slice[r],
                context_slice[r],
                sinks,
                N_HEAD * sizeof(float),
                0,
                q_slice[r],
                main_kv[r],
                draft_slice[r],
                n_main[r],
                MAIN_CAP,
                main_start[r],
                BLOCK_TOKENS,
                N_HEAD,
                HEAD_DIM) != 0;
    }
    if (ok) {
        const ds4_gpu_tensor *main_const[REQUESTS] = {
            main_kv[0], main_kv[1], main_kv[2]
        };
        ok = ds4_gpu_dspark_attention_rn_heads_tensor(
                heads_rn,
                context_rn,
                sinks,
                N_HEAD * sizeof(float),
                0,
                q,
                main_const,
                draft,
                n_main,
                main_start,
                REQUESTS,
                MAIN_CAP,
                BLOCK_TOKENS,
                N_HEAD,
                HEAD_DIM) != 0;
    }
    if (ok) {
        ok = ds4_gpu_synchronize() != 0 &&
             ds4_gpu_tensor_read(
                 heads_ref, 0, ref_host, sizeof(ref_host)) != 0 &&
             ds4_gpu_tensor_read(
                 heads_rn, 0, rn_host, sizeof(rn_host)) != 0;
    }
    if (ok) {
        rc = 0;
        for (uint64_t i = 0; i < head_count; i++) {
            if (fabsf(ref_host[i] - rn_host[i]) > 1.0e-6f) {
                fprintf(stderr,
                        "physical R=n DSpark attention mismatch "
                        "at=%llu got=%g ref=%g\n",
                        (unsigned long long)i,
                        (double)rn_host[i],
                        (double)ref_host[i]);
                rc = 1;
                break;
            }
        }
    }
    if (rc == 0) {
        fprintf(stderr,
                "cuda-regression: physical R=3 DSpark attention "
                "matches independent R=1 KV rings\n");
    }
    for (uint32_t r = 0; r < REQUESTS; r++) {
        ds4_gpu_tensor_free(context_slice[r]);
        ds4_gpu_tensor_free(draft_slice[r]);
        ds4_gpu_tensor_free(q_slice[r]);
        ds4_gpu_tensor_free(heads_slice[r]);
        ds4_gpu_tensor_free(main_kv[r]);
    }
    ds4_gpu_tensor_free(context_rn);
    ds4_gpu_tensor_free(draft);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(heads_ref);
    ds4_gpu_tensor_free(heads_rn);
    return rc;
}

static int check_physical_rn_indexed_attention(void) {
    enum {
        REQUESTS = 2,
        ROWS = 3,
        N_HEAD = 2,
        HEAD_DIM = 8,
        RAW_CAP = 4,
        RAW_STRIDE = 4,
        COMP_STRIDE = 3,
        TOP_K = 2,
    };
    const uint64_t q_count = (uint64_t)ROWS * N_HEAD * HEAD_DIM;
    const uint64_t raw_count =
        (uint64_t)REQUESTS * RAW_STRIDE * HEAD_DIM;
    const uint64_t comp_count =
        (uint64_t)REQUESTS * COMP_STRIDE * HEAD_DIM;
    /* The CUDA backend maps model ranges read-only. Keep the synthetic
     * model-map on its own page so it cannot cover D2H result buffers that
     * happen to share this function's stack page. */
    static float sinks_page[1024] __attribute__((aligned(4096)));
    float *sinks = sinks_page;
    sinks[0] = -0.25f;
    sinks[1] = 0.125f;
    float q_host[q_count];
    float raw_host[raw_count];
    float comp_host[comp_count];
    float rn_host[q_count];
    float ptr_host[q_count];
    float ref_host[q_count];
    const int32_t topk_host[ROWS * TOP_K] = {
        0, 1,
        2, 0,
        1, 0,
    };
    const uint32_t row_request_host[ROWS] = {0, 0, 1};
    const uint32_t row_position_host[ROWS] = {4, 5, 10};
    const uint32_t request_position_host[REQUESTS] = {4, 10};
    const uint32_t request_rows_host[REQUESTS] = {2, 1};
    const uint32_t request_n_raw_host[REQUESTS] = {4, 3};
    const uint32_t request_raw_start_host[REQUESTS] = {1, 0};
    const uint32_t request_n_comp_host[REQUESTS] = {3, 2};
    for (uint64_t i = 0; i < q_count; i++) {
        q_host[i] = ((float)((i * 17u) % 29u) - 14.0f) / 19.0f;
    }
    for (uint64_t i = 0; i < raw_count; i++) {
        raw_host[i] = ((float)((i * 11u) % 31u) - 15.0f) / 23.0f;
    }
    for (uint64_t i = 0; i < comp_count; i++) {
        comp_host[i] = ((float)((i * 7u) % 37u) - 18.0f) / 29.0f;
    }
    memset(rn_host, 0, sizeof(rn_host));
    memset(ref_host, 0, sizeof(ref_host));

    ds4_gpu_tensor *heads_rn =
        ds4_gpu_tensor_alloc(sizeof(rn_host));
    ds4_gpu_tensor *heads_ref =
        ds4_gpu_tensor_alloc(sizeof(ref_host));
    ds4_gpu_tensor *heads_ptr =
        ds4_gpu_tensor_alloc(sizeof(ptr_host));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(sizeof(q_host));
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(sizeof(raw_host));
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(sizeof(comp_host));
    ds4_gpu_tensor *topk = ds4_gpu_tensor_alloc(sizeof(topk_host));
    ds4_gpu_tensor *row_request =
        ds4_gpu_tensor_alloc(sizeof(row_request_host));
    ds4_gpu_tensor *row_position =
        ds4_gpu_tensor_alloc(sizeof(row_position_host));
    ds4_gpu_tensor *request_position =
        ds4_gpu_tensor_alloc(sizeof(request_position_host));
    ds4_gpu_tensor *request_rows =
        ds4_gpu_tensor_alloc(sizeof(request_rows_host));
    ds4_gpu_tensor *request_n_raw =
        ds4_gpu_tensor_alloc(sizeof(request_n_raw_host));
    ds4_gpu_tensor *request_raw_start =
        ds4_gpu_tensor_alloc(sizeof(request_raw_start_host));
    ds4_gpu_tensor *request_n_comp =
        ds4_gpu_tensor_alloc(sizeof(request_n_comp_host));
    ds4_gpu_tensor *heads_ref0 = NULL;
    ds4_gpu_tensor *heads_ref1 = NULL;
    ds4_gpu_tensor *q0 = NULL;
    ds4_gpu_tensor *q1 = NULL;
    ds4_gpu_tensor *raw0 = NULL;
    ds4_gpu_tensor *raw1 = NULL;
    ds4_gpu_tensor *comp0 = NULL;
    ds4_gpu_tensor *comp1 = NULL;
    ds4_gpu_tensor *topk0 = NULL;
    ds4_gpu_tensor *topk1 = NULL;
    ds4_gpu_tensor *raw_ptr0 = NULL;
    ds4_gpu_tensor *raw_ptr1 = NULL;
    ds4_gpu_tensor *comp_ptr0 = NULL;
    ds4_gpu_tensor *comp_ptr1 = NULL;
    ds4_gpu_tensor *raw_table = NULL;
    ds4_gpu_tensor *comp_table = NULL;
    int rc = 1;
    const uint64_t row_bytes =
        (uint64_t)N_HEAD * HEAD_DIM * sizeof(float);
    if (!heads_rn || !heads_ref || !heads_ptr ||
        !q || !raw || !comp || !topk ||
        !row_request || !row_position || !request_position ||
        !request_rows || !request_n_raw || !request_raw_start ||
        !request_n_comp ||
        !ds4_gpu_tensor_write(q, 0, q_host, sizeof(q_host)) ||
        !ds4_gpu_tensor_write(raw, 0, raw_host, sizeof(raw_host)) ||
        !ds4_gpu_tensor_write(comp, 0, comp_host, sizeof(comp_host)) ||
        !ds4_gpu_tensor_write(topk, 0, topk_host, sizeof(topk_host)) ||
        !ds4_gpu_tensor_write(row_request, 0, row_request_host,
                              sizeof(row_request_host)) ||
        !ds4_gpu_tensor_write(row_position, 0, row_position_host,
                              sizeof(row_position_host)) ||
        !ds4_gpu_tensor_write(request_position, 0, request_position_host,
                              sizeof(request_position_host)) ||
        !ds4_gpu_tensor_write(request_rows, 0, request_rows_host,
                              sizeof(request_rows_host)) ||
        !ds4_gpu_tensor_write(request_n_raw, 0, request_n_raw_host,
                              sizeof(request_n_raw_host)) ||
        !ds4_gpu_tensor_write(request_raw_start, 0,
                              request_raw_start_host,
                              sizeof(request_raw_start_host)) ||
        !ds4_gpu_tensor_write(request_n_comp, 0, request_n_comp_host,
                              sizeof(request_n_comp_host))) {
        goto cleanup;
    }

    heads_ref0 = ds4_gpu_tensor_view(heads_ref, 0, 2u * row_bytes);
    heads_ref1 = ds4_gpu_tensor_view(heads_ref, 2u * row_bytes, row_bytes);
    q0 = ds4_gpu_tensor_view(q, 0, 2u * row_bytes);
    q1 = ds4_gpu_tensor_view(q, 2u * row_bytes, row_bytes);
    raw0 = ds4_gpu_tensor_view(
        raw, 0, (uint64_t)RAW_STRIDE * HEAD_DIM * sizeof(float));
    raw1 = ds4_gpu_tensor_view(
        raw,
        (uint64_t)RAW_STRIDE * HEAD_DIM * sizeof(float),
        (uint64_t)RAW_STRIDE * HEAD_DIM * sizeof(float));
    comp0 = ds4_gpu_tensor_view(
        comp, 0, (uint64_t)COMP_STRIDE * HEAD_DIM * sizeof(float));
    comp1 = ds4_gpu_tensor_view(
        comp,
        (uint64_t)COMP_STRIDE * HEAD_DIM * sizeof(float),
        (uint64_t)COMP_STRIDE * HEAD_DIM * sizeof(float));
    topk0 = ds4_gpu_tensor_view(
        topk, 0, 2u * TOP_K * sizeof(int32_t));
    topk1 = ds4_gpu_tensor_view(
        topk, 2u * TOP_K * sizeof(int32_t),
        TOP_K * sizeof(int32_t));
    raw_ptr0 = ds4_gpu_tensor_alloc(
        (uint64_t)RAW_STRIDE * HEAD_DIM * sizeof(float));
    raw_ptr1 = ds4_gpu_tensor_alloc(
        (uint64_t)RAW_STRIDE * HEAD_DIM * sizeof(float));
    comp_ptr0 = ds4_gpu_tensor_alloc(
        (uint64_t)COMP_STRIDE * HEAD_DIM * sizeof(float));
    comp_ptr1 = ds4_gpu_tensor_alloc(
        (uint64_t)COMP_STRIDE * HEAD_DIM * sizeof(float));
    raw_table = ds4_gpu_tensor_alloc(
        REQUESTS * sizeof(void *));
    comp_table = ds4_gpu_tensor_alloc(
        REQUESTS * sizeof(void *));
    const ds4_gpu_tensor *raw_ptrs[REQUESTS] = {
        raw_ptr0, raw_ptr1
    };
    const ds4_gpu_tensor *comp_ptrs[REQUESTS] = {
        comp_ptr0, comp_ptr1
    };
    if (!heads_ref0 || !heads_ref1 || !q0 || !q1 || !raw0 || !raw1 ||
        !comp0 || !comp1 || !topk0 || !topk1 ||
        !raw_ptr0 || !raw_ptr1 || !comp_ptr0 || !comp_ptr1 ||
        !raw_table || !comp_table ||
        !ds4_gpu_tensor_write(
            raw_ptr0, 0, raw_host,
            (uint64_t)RAW_STRIDE * HEAD_DIM * sizeof(float)) ||
        !ds4_gpu_tensor_write(
            raw_ptr1, 0,
            raw_host + (uint64_t)RAW_STRIDE * HEAD_DIM,
            (uint64_t)RAW_STRIDE * HEAD_DIM * sizeof(float)) ||
        !ds4_gpu_tensor_write(
            comp_ptr0, 0, comp_host,
            (uint64_t)COMP_STRIDE * HEAD_DIM * sizeof(float)) ||
        !ds4_gpu_tensor_write(
            comp_ptr1, 0,
            comp_host + (uint64_t)COMP_STRIDE * HEAD_DIM,
            (uint64_t)COMP_STRIDE * HEAD_DIM * sizeof(float)) ||
        !ds4_gpu_tensor_pointer_table_write(
            raw_table, raw_ptrs, REQUESTS) ||
        !ds4_gpu_tensor_pointer_table_write(
            comp_table, comp_ptrs, REQUESTS) ||
        !ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
            heads_ref0, sinks, N_HEAD * sizeof(float), 0,
            q0, raw0, comp0, 0,
            topk0, 2, 4, 4, RAW_CAP, 1, 3, TOP_K, 4, 2,
            N_HEAD, HEAD_DIM) ||
        !ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
            heads_ref1, sinks, N_HEAD * sizeof(float), 0,
            q1, raw1, comp1, 0,
            topk1, 1, 10, 3, RAW_CAP, 0, 2, TOP_K, 4, 2,
            N_HEAD, HEAD_DIM) ||
        !ds4_gpu_attention_indexed_mixed_rn_heads_tensor(
            heads_rn, sinks, N_HEAD * sizeof(float), 0,
            q, raw, comp, 0, topk,
            row_request, row_position, request_position, request_rows,
            request_n_raw, request_raw_start, request_n_comp,
            ROWS, REQUESTS, RAW_CAP, RAW_STRIDE, COMP_STRIDE, TOP_K,
            4, 2, N_HEAD, HEAD_DIM) ||
        !ds4_gpu_attention_indexed_mixed_rn_ptrs_heads_tensor(
            heads_ptr, sinks, N_HEAD * sizeof(float), 0, q,
            raw_table, comp_table, 0, topk,
            row_request, row_position, request_position, request_rows,
            request_n_raw, request_raw_start, request_n_comp,
            ROWS, REQUESTS, RAW_CAP, COMP_STRIDE, TOP_K,
            4, 2, N_HEAD, HEAD_DIM) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(heads_ref, 0, ref_host, sizeof(ref_host)) ||
        !ds4_gpu_tensor_read(heads_rn, 0, rn_host, sizeof(rn_host)) ||
        !ds4_gpu_tensor_read(heads_ptr, 0, ptr_host, sizeof(ptr_host))) {
        goto cleanup;
    }
    for (uint64_t i = 0; i < q_count; i++) {
        if (fabsf(ref_host[i] - rn_host[i]) > 1.0e-6f ||
            fabsf(ref_host[i] - ptr_host[i]) > 1.0e-6f) {
            fprintf(stderr,
                    "physical R=n attention mismatch at=%llu "
                    "arena=%g ptr=%g ref=%g\n",
                    (unsigned long long)i,
                    (double)rn_host[i],
                    (double)ptr_host[i],
                    (double)ref_host[i]);
            goto cleanup;
        }
    }
    fprintf(stderr,
            "cuda-regression: physical R=2 indexed attention matches "
            "independent R=1 frontiers (arena + pointer table)\n");
    rc = 0;

cleanup:
    ds4_gpu_tensor_free(comp_table);
    ds4_gpu_tensor_free(raw_table);
    ds4_gpu_tensor_free(comp_ptr1);
    ds4_gpu_tensor_free(comp_ptr0);
    ds4_gpu_tensor_free(raw_ptr1);
    ds4_gpu_tensor_free(raw_ptr0);
    ds4_gpu_tensor_free(topk1);
    ds4_gpu_tensor_free(topk0);
    ds4_gpu_tensor_free(comp1);
    ds4_gpu_tensor_free(comp0);
    ds4_gpu_tensor_free(raw1);
    ds4_gpu_tensor_free(raw0);
    ds4_gpu_tensor_free(q1);
    ds4_gpu_tensor_free(q0);
    ds4_gpu_tensor_free(heads_ref1);
    ds4_gpu_tensor_free(heads_ref0);
    ds4_gpu_tensor_free(request_n_comp);
    ds4_gpu_tensor_free(request_raw_start);
    ds4_gpu_tensor_free(request_n_raw);
    ds4_gpu_tensor_free(request_rows);
    ds4_gpu_tensor_free(request_position);
    ds4_gpu_tensor_free(row_position);
    ds4_gpu_tensor_free(row_request);
    ds4_gpu_tensor_free(topk);
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(heads_ref);
    ds4_gpu_tensor_free(heads_ptr);
    ds4_gpu_tensor_free(heads_rn);
    return rc;
}

static int check_physical_rn_layout_primitives(void) {
    enum {
        REQUESTS = 2,
        ROWS = 3,
        N_HEAD = 2,
        HEAD_DIM = 8,
        N_ROT = 4,
        RAW_CAP = 4,
        RAW_STRIDE = 4,
    };
    const uint32_t row_request_host[ROWS] = {0, 0, 1};
    const uint32_t row_position_host[ROWS] = {3, 4, 11};
    const uint64_t rope_count =
        (uint64_t)ROWS * N_HEAD * HEAD_DIM;
    const uint64_t raw_count =
        (uint64_t)REQUESTS * RAW_STRIDE * HEAD_DIM;
    const uint64_t raw_slice_bytes =
        (uint64_t)RAW_STRIDE * HEAD_DIM * sizeof(float);
    float rope_input[rope_count];
    float rope_rn_host[rope_count];
    float rope_ref_host[rope_count];
    float kv_host[(uint64_t)ROWS * HEAD_DIM];
    float raw_rn_host[raw_count];
    float raw_ptr_host[raw_count];
    float raw_ref_host[raw_count];
    for (uint64_t i = 0; i < rope_count; i++) {
        rope_input[i] =
            ((float)((i * 13u) % 41u) - 20.0f) / 17.0f;
    }
    for (uint64_t i = 0; i < (uint64_t)ROWS * HEAD_DIM; i++) {
        kv_host[i] =
            ((float)((i * 19u) % 43u) - 21.0f) / 23.0f;
    }
    memset(raw_rn_host, 0, sizeof(raw_rn_host));
    memset(raw_ptr_host, 0, sizeof(raw_ptr_host));
    memset(raw_ref_host, 0, sizeof(raw_ref_host));

    ds4_gpu_tensor *rope_rn =
        ds4_gpu_tensor_alloc(sizeof(rope_input));
    ds4_gpu_tensor *rope_ref =
        ds4_gpu_tensor_alloc(sizeof(rope_input));
    ds4_gpu_tensor *positions =
        ds4_gpu_tensor_alloc(sizeof(row_position_host));
    ds4_gpu_tensor *row_request =
        ds4_gpu_tensor_alloc(sizeof(row_request_host));
    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc(sizeof(kv_host));
    ds4_gpu_tensor *raw_rn =
        ds4_gpu_tensor_alloc(sizeof(raw_rn_host));
    ds4_gpu_tensor *raw_ref =
        ds4_gpu_tensor_alloc(sizeof(raw_ref_host));
    ds4_gpu_tensor *raw_ptr0 =
        ds4_gpu_tensor_alloc(raw_slice_bytes);
    ds4_gpu_tensor *raw_ptr1 =
        ds4_gpu_tensor_alloc(raw_slice_bytes);
    ds4_gpu_tensor *raw_table =
        ds4_gpu_tensor_alloc(REQUESTS * sizeof(void *));
    ds4_gpu_tensor *rope_ref0 = NULL;
    ds4_gpu_tensor *rope_ref1 = NULL;
    ds4_gpu_tensor *kv0 = NULL;
    ds4_gpu_tensor *kv1 = NULL;
    ds4_gpu_tensor *raw_ref0 = NULL;
    ds4_gpu_tensor *raw_ref1 = NULL;
    int rc = 1;
    const uint64_t rope_row_bytes =
        (uint64_t)N_HEAD * HEAD_DIM * sizeof(float);
    const uint64_t kv_row_bytes =
        (uint64_t)HEAD_DIM * sizeof(float);
    const ds4_gpu_tensor *raw_ptrs[REQUESTS] = {
        raw_ptr0, raw_ptr1
    };
    if (!rope_rn || !rope_ref || !positions || !row_request ||
        !kv || !raw_rn || !raw_ref ||
        !raw_ptr0 || !raw_ptr1 || !raw_table ||
        !ds4_gpu_tensor_write(rope_rn, 0, rope_input,
                              sizeof(rope_input)) ||
        !ds4_gpu_tensor_write(rope_ref, 0, rope_input,
                              sizeof(rope_input)) ||
        !ds4_gpu_tensor_write(positions, 0, row_position_host,
                              sizeof(row_position_host)) ||
        !ds4_gpu_tensor_write(row_request, 0, row_request_host,
                              sizeof(row_request_host)) ||
        !ds4_gpu_tensor_write(kv, 0, kv_host, sizeof(kv_host)) ||
        !ds4_gpu_tensor_write(raw_rn, 0, raw_rn_host,
                              sizeof(raw_rn_host)) ||
        !ds4_gpu_tensor_write(raw_ref, 0, raw_ref_host,
                              sizeof(raw_ref_host)) ||
        !ds4_gpu_tensor_write(raw_ptr0, 0, raw_ref_host,
                              raw_slice_bytes) ||
        !ds4_gpu_tensor_write(raw_ptr1, 0,
                              raw_ref_host + RAW_STRIDE * HEAD_DIM,
                              raw_slice_bytes) ||
        !ds4_gpu_tensor_pointer_table_write(
            raw_table, raw_ptrs, REQUESTS)) {
        goto cleanup;
    }
    rope_ref0 = ds4_gpu_tensor_view(
        rope_ref, 0, 2u * rope_row_bytes);
    rope_ref1 = ds4_gpu_tensor_view(
        rope_ref, 2u * rope_row_bytes, rope_row_bytes);
    kv0 = ds4_gpu_tensor_view(kv, 0, 2u * kv_row_bytes);
    kv1 = ds4_gpu_tensor_view(kv, 2u * kv_row_bytes, kv_row_bytes);
    raw_ref0 = ds4_gpu_tensor_view(raw_ref, 0, raw_slice_bytes);
    raw_ref1 = ds4_gpu_tensor_view(
        raw_ref, raw_slice_bytes, raw_slice_bytes);
    if (!rope_ref0 || !rope_ref1 || !kv0 || !kv1 ||
        !raw_ref0 || !raw_ref1 ||
        !ds4_gpu_rope_tail_tensor(
            rope_ref0, 2, N_HEAD, HEAD_DIM, N_ROT, 3, 0, false,
            10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f) ||
        !ds4_gpu_rope_tail_tensor(
            rope_ref1, 1, N_HEAD, HEAD_DIM, N_ROT, 11, 0, false,
            10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f) ||
        !ds4_gpu_rope_tail_positions_tensor(
            rope_rn, positions, ROWS, N_HEAD, HEAD_DIM, N_ROT, 0,
            false, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f) ||
        !ds4_gpu_store_raw_kv_batch_tensor(
            raw_ref0, kv0, RAW_CAP, 3, 2, HEAD_DIM) ||
        !ds4_gpu_store_raw_kv_batch_tensor(
            raw_ref1, kv1, RAW_CAP, 11, 1, HEAD_DIM) ||
        !ds4_gpu_store_raw_kv_rn_tensor(
            raw_rn, kv, row_request, positions, ROWS, REQUESTS,
            RAW_CAP, RAW_STRIDE, HEAD_DIM) ||
        !ds4_gpu_store_raw_kv_rn_ptrs_tensor(
            raw_table, kv, row_request, positions, ROWS, REQUESTS,
            RAW_CAP, HEAD_DIM) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(
            rope_rn, 0, rope_rn_host, sizeof(rope_rn_host)) ||
        !ds4_gpu_tensor_read(
            rope_ref, 0, rope_ref_host, sizeof(rope_ref_host)) ||
        !ds4_gpu_tensor_read(
            raw_rn, 0, raw_rn_host, sizeof(raw_rn_host)) ||
        !ds4_gpu_tensor_read(
            raw_ref, 0, raw_ref_host, sizeof(raw_ref_host))) {
        goto cleanup;
    }
    if (!ds4_gpu_tensor_read(
            raw_ptr0, 0, raw_ptr_host, raw_slice_bytes) ||
        !ds4_gpu_tensor_read(
            raw_ptr1, 0,
            raw_ptr_host + RAW_STRIDE * HEAD_DIM,
            raw_slice_bytes)) {
        goto cleanup;
    }
    for (uint64_t i = 0; i < rope_count; i++) {
        if (fabsf(rope_rn_host[i] - rope_ref_host[i]) > 1.0e-6f) {
            fprintf(stderr,
                    "physical R=n RoPE mismatch at=%llu got=%g ref=%g\n",
                    (unsigned long long)i,
                    (double)rope_rn_host[i],
                    (double)rope_ref_host[i]);
            goto cleanup;
        }
    }
    for (uint64_t i = 0; i < raw_count; i++) {
        if (raw_rn_host[i] != raw_ref_host[i] ||
            raw_ptr_host[i] != raw_ref_host[i]) {
            fprintf(stderr,
                    "physical R=n raw store mismatch at=%llu "
                    "arena=%g ptr=%g ref=%g\n",
                    (unsigned long long)i,
                    (double)raw_rn_host[i],
                    (double)raw_ptr_host[i],
                    (double)raw_ref_host[i]);
            goto cleanup;
        }
    }
    fprintf(stderr,
            "cuda-regression: physical R=2 RoPE/raw-ring markers match "
            "independent R=1 sequences (arena + pointer table)\n");
    rc = 0;

cleanup:
    ds4_gpu_tensor_free(raw_table);
    ds4_gpu_tensor_free(raw_ptr1);
    ds4_gpu_tensor_free(raw_ptr0);
    ds4_gpu_tensor_free(raw_ref1);
    ds4_gpu_tensor_free(raw_ref0);
    ds4_gpu_tensor_free(kv1);
    ds4_gpu_tensor_free(kv0);
    ds4_gpu_tensor_free(rope_ref1);
    ds4_gpu_tensor_free(rope_ref0);
    ds4_gpu_tensor_free(raw_ref);
    ds4_gpu_tensor_free(raw_rn);
    ds4_gpu_tensor_free(kv);
    ds4_gpu_tensor_free(row_request);
    ds4_gpu_tensor_free(positions);
    ds4_gpu_tensor_free(rope_ref);
    ds4_gpu_tensor_free(rope_rn);
    return rc;
}

static float mxfp4_e2m1_value(uint32_t code) {
    static const float values[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
    };
    return values[code & 7u];
}

static int topk_set_contains(const uint32_t *ids, uint32_t top_k, uint32_t id) {
    for (uint32_t i = 0; i < top_k; i++) {
        if (ids[i] == id) return 1;
    }
    return 0;
}

static int topk_boundary_score_close(float a, float b) {
    if (a == b) return 1;
    if (!isfinite(a) || !isfinite(b)) return 0;
    const float scale = fmaxf(1.0f, fmaxf(fabsf(a), fabsf(b)));
    return fabsf(a - b) <= 1.0e-6f + 5.0e-4f * scale;
}

static uint32_t nonnegative_f32_ulp_distance(float a, float b) {
    if (a == b) return 0u;
    if (!isfinite(a) || !isfinite(b) || a < 0.0f || b < 0.0f) {
        return UINT32_MAX;
    }
    uint32_t a_bits;
    uint32_t b_bits;
    memcpy(&a_bits, &a, sizeof(a_bits));
    memcpy(&b_bits, &b, sizeof(b_bits));
    return a_bits > b_bits ? a_bits - b_bits : b_bits - a_bits;
}

/* The native MMA scorer can reorder FP32 additions. A changed Top-K set is
 * acceptable only when both scorers place every exchanged row on their own
 * numerical Kth boundary; arbitrary overlap loss is not accepted. */
static int topk_boundary_equivalent(
        const uint32_t *reference,
        const uint32_t *candidate,
        const float    *reference_scores,
        const float    *candidate_scores,
        uint32_t        n_comp,
        uint32_t        top_k,
        uint32_t       *overlap_out) {
    float reference_boundary = INFINITY;
    float candidate_boundary = INFINITY;
    uint32_t overlap = 0;
    for (uint32_t i = 0; i < top_k; i++) {
        if (reference[i] >= n_comp || candidate[i] >= n_comp) return 0;
        for (uint32_t j = 0; j < i; j++) {
            if (reference[j] == reference[i] || candidate[j] == candidate[i]) return 0;
        }
        reference_boundary = fminf(reference_boundary, reference_scores[reference[i]]);
        candidate_boundary = fminf(candidate_boundary, candidate_scores[candidate[i]]);
        overlap += (uint32_t)topk_set_contains(reference, top_k, candidate[i]);
    }
    for (uint32_t i = 0; i < top_k; i++) {
        const uint32_t id = candidate[i];
        if (!topk_set_contains(reference, top_k, id) &&
            !topk_boundary_score_close(reference_scores[id], reference_boundary)) {
            return 0;
        }
    }
    for (uint32_t i = 0; i < top_k; i++) {
        const uint32_t id = reference[i];
        if (!topk_set_contains(candidate, top_k, id) &&
            !topk_boundary_score_close(candidate_scores[id], candidate_boundary)) {
            return 0;
        }
    }
    if (overlap_out) *overlap_out = overlap;
    return 1;
}

static void mxfp4_unpack_reference(
        float         *dst,
        const uint8_t *packed,
        uint32_t       n_rows) {
    for (uint32_t row = 0; row < n_rows; row++) {
        float scales[4];
        for (uint32_t block = 0; block < 4u; block++) {
            const uint32_t bits =
                (uint32_t)packed[(uint64_t)row * DS4_GPU_INDEXER_FP4_ROW_BYTES +
                                 64u + block] << 23u;
            memcpy(&scales[block], &bits, sizeof(bits));
        }
        const uint8_t *src = packed +
            (uint64_t)row * DS4_GPU_INDEXER_FP4_ROW_BYTES;
        for (uint32_t d = 0; d < 128u; d++) {
            const uint32_t block = d >> 5u;
            const uint32_t lane = d & 31u;
            const uint8_t byte = src[block * 16u + (lane & 15u)];
            const uint32_t code = (byte >> ((lane >> 4u) * 4u)) & 15u;
            const float sign = (code & 8u) != 0u ? -1.0f : 1.0f;
            dst[(uint64_t)row * 128u + d] =
                sign * mxfp4_e2m1_value(code) * scales[block];
        }
    }
}

static int check_mxfp4_indexer(void) {
    const uint32_t n_tokens = 17;
    const uint32_t n_head = 64;
    const uint32_t head_dim = 128;
    const uint32_t n_comp = 641;
    const uint32_t top_k = 512;
    const uint32_t causal_pos0 = 2047;
    const uint32_t q_rows = n_tokens * n_head;
    const uint64_t q_count = (uint64_t)q_rows * head_dim;
    const uint64_t k_count = (uint64_t)n_comp * head_dim;
    const uint64_t score_count = (uint64_t)n_tokens * n_comp;
    const uint64_t q_packed_count =
        (uint64_t)q_rows * DS4_GPU_INDEXER_FP4_ROW_BYTES;
    const uint64_t k_packed_count =
        (uint64_t)n_comp * DS4_GPU_INDEXER_FP4_ROW_BYTES;
    int rc = 1;

    float *q_host = (float *)malloc((size_t)q_count * sizeof(float));
    float *k_host = (float *)malloc((size_t)k_count * sizeof(float));
    float *w_host = (float *)malloc((size_t)n_tokens * n_head * sizeof(float));
    float *q_ref_host = (float *)malloc((size_t)q_count * sizeof(float));
    float *q_unpack_host = (float *)malloc((size_t)q_count * sizeof(float));
    float *k_ref_host = (float *)malloc((size_t)k_count * sizeof(float));
    float *k_unpack_host = (float *)malloc((size_t)k_count * sizeof(float));
    float *score_ref_host = (float *)malloc((size_t)score_count * sizeof(float));
    float *score_fp4_host = (float *)malloc((size_t)score_count * sizeof(float));
    float *score_causal_ref_host =
        (float *)malloc((size_t)score_count * sizeof(float));
    float *score_causal_fp4_host =
        (float *)malloc((size_t)score_count * sizeof(float));
    float *score_shape_host = (float *)malloc((size_t)score_count * sizeof(float));
    uint8_t *q_packed_repeat_host = (uint8_t *)malloc((size_t)q_packed_count);
    uint8_t *q_packed_host = (uint8_t *)malloc((size_t)q_packed_count);
    uint8_t *k_packed_repeat_host = (uint8_t *)malloc((size_t)k_packed_count);
    uint8_t *k_packed_host = (uint8_t *)malloc((size_t)k_packed_count);
    uint32_t *topk_ref_host = (uint32_t *)malloc(
            (size_t)n_tokens * top_k * sizeof(uint32_t));
    uint32_t *topk_fp4_host = (uint32_t *)malloc(
            (size_t)n_tokens * top_k * sizeof(uint32_t));
    uint32_t *topk_causal_ref_host = (uint32_t *)malloc(
            (size_t)n_tokens * top_k * sizeof(uint32_t));
    uint32_t *topk_causal_fp4_host = (uint32_t *)malloc(
            (size_t)n_tokens * top_k * sizeof(uint32_t));
    if (!q_host || !k_host || !w_host || !q_ref_host || !q_unpack_host ||
        !k_ref_host || !k_unpack_host || !score_ref_host || !score_fp4_host ||
        !score_causal_ref_host || !score_causal_fp4_host || !score_shape_host ||
        !q_packed_repeat_host || !q_packed_host || !k_packed_repeat_host ||
        !k_packed_host || !topk_ref_host || !topk_fp4_host ||
        !topk_causal_ref_host || !topk_causal_fp4_host) {
        goto host_cleanup;
    }

    uint32_t rng = 0x9e3779b9u;
    for (uint32_t row = 0; row < q_rows; row++) {
        const uint32_t token = row / n_head;
        const uint32_t head = row % n_head;
        for (uint32_t d = 0; d < head_dim; d++) {
            float value = 0.0f;
            switch (token) {
            case 0:
                value = 0.7f * sinf((float)(row * head_dim + d) * 0.017f) +
                        0.2f * cosf((float)(row * head_dim + d) * 0.071f);
                break;
            case 1:
                value = 0.0f;
                break;
            case 2:
                value = 0.125f + (float)(head & 3u) * 0.0625f;
                break;
            case 3: {
                static const float boundary[] = {
                    0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f
                };
                value = boundary[(d + head) % 7u] * ((d & 1u) ? -1.0f : 1.0f);
                break;
            }
            case 4:
                value = d == ((head * 17u) & 127u)
                    ? 5.5f
                    : ldexpf((d & 1u) ? -1.0f : 1.0f, -20);
                break;
            default:
                rng = rng * 1664525u + 1013904223u;
                value = 2.0f * (float)(rng >> 8u) * (1.0f / 16777216.0f) - 1.0f;
                break;
            }
            q_host[(uint64_t)row * head_dim + d] = value;
        }
    }
    for (uint32_t row = 0; row < n_comp; row++) {
        for (uint32_t d = 0; d < head_dim; d++) {
            float value;
            switch (row % 6u) {
            case 0:
                value = 0.6f * cosf((float)(row * head_dim + d) * 0.013f) -
                        0.3f * sinf((float)(row * head_dim + d) * 0.037f);
                break;
            case 1:
                value = 0.0f;
                break;
            case 2:
                value = (float)((int32_t)(row & 7u) - 3) * 0.125f;
                break;
            case 3:
                value = (d & 1u) ? -3.5f : 3.5f;
                break;
            case 4:
                value = d == ((row * 29u) & 127u) ? 6.0f : 0.0f;
                break;
            default:
                rng = rng * 1664525u + 1013904223u;
                value = 2.0f * (float)(rng >> 8u) * (1.0f / 16777216.0f) - 1.0f;
                break;
            }
            k_host[(uint64_t)row * head_dim + d] = value;
        }
    }
    for (uint32_t i = 0; i < n_tokens * n_head; i++) {
        w_host[i] = 0.5f + 0.5f * sinf((float)i * 0.11f);
    }
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *q_ref = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *q_unpack = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *q_packed = ds4_gpu_tensor_alloc(
            (uint64_t)q_rows * DS4_GPU_INDEXER_FP4_ROW_BYTES);
    ds4_gpu_tensor *keys = ds4_gpu_tensor_alloc(k_count * sizeof(float));
    ds4_gpu_tensor *keys_ref = ds4_gpu_tensor_alloc(k_count * sizeof(float));
    ds4_gpu_tensor *keys_unpack = ds4_gpu_tensor_alloc(k_count * sizeof(float));
    ds4_gpu_tensor *keys_packed = ds4_gpu_tensor_alloc(
            (uint64_t)n_comp * DS4_GPU_INDEXER_FP4_ROW_BYTES);
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(
            (uint64_t)n_tokens * n_head * sizeof(float));
    ds4_gpu_tensor *score_ref = ds4_gpu_tensor_alloc(score_count * sizeof(float));
    ds4_gpu_tensor *score_fp4 = ds4_gpu_tensor_alloc(score_count * sizeof(float));
    ds4_gpu_tensor *score_causal_ref =
        ds4_gpu_tensor_alloc(score_count * sizeof(float));
    ds4_gpu_tensor *score_causal_fp4 =
        ds4_gpu_tensor_alloc(score_count * sizeof(float));
    ds4_gpu_tensor *score_shape = ds4_gpu_tensor_alloc(score_count * sizeof(float));
    ds4_gpu_tensor *topk_ref = ds4_gpu_tensor_alloc(
            (uint64_t)n_tokens * top_k * sizeof(uint32_t));
    ds4_gpu_tensor *topk_fp4 = ds4_gpu_tensor_alloc(
            (uint64_t)n_tokens * top_k * sizeof(uint32_t));
    ds4_gpu_tensor *topk_causal_ref = ds4_gpu_tensor_alloc(
            (uint64_t)n_tokens * top_k * sizeof(uint32_t));
    ds4_gpu_tensor *topk_causal_fp4 = ds4_gpu_tensor_alloc(
            (uint64_t)n_tokens * top_k * sizeof(uint32_t));

    if (!q || !q_ref || !q_unpack || !q_packed || !keys || !keys_ref ||
        !keys_unpack || !keys_packed || !weights || !score_ref || !score_fp4 ||
        !score_causal_ref || !score_causal_fp4 || !score_shape ||
        !topk_ref || !topk_fp4 || !topk_causal_ref || !topk_causal_fp4) {
        goto cleanup;
    }
    if (!ds4_gpu_tensor_write(q, 0, q_host, q_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(q_ref, 0, q_host, q_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(keys, 0, k_host, k_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(keys_ref, 0, k_host, k_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(weights, 0, w_host,
                              (uint64_t)n_tokens * n_head * sizeof(float)) ||
        !ds4_gpu_dsv4_indexer_qat_tensor(q_ref, q_rows, head_dim) ||
        !ds4_gpu_dsv4_indexer_qat_tensor(keys_ref, n_comp, head_dim) ||
        !ds4_gpu_dsv4_indexer_pack_tensor(q_packed, q, q_rows) ||
        !ds4_gpu_dsv4_indexer_unpack_tensor(q_unpack, q_packed, q_rows) ||
        !ds4_gpu_dsv4_indexer_pack_tensor(keys_packed, keys, n_comp) ||
        !ds4_gpu_dsv4_indexer_unpack_tensor(keys_unpack, keys_packed, n_comp) ||
        !ds4_gpu_indexer_scores_decode_batch_tensor(
                score_ref, q_unpack, weights, keys_unpack,
                n_comp, n_tokens, n_comp * 4u,
                n_head, head_dim, 4, 1.0f) ||
        !ds4_gpu_indexer_scores_packed_tensor(
                score_fp4, q_packed, weights, keys_packed,
                n_comp, n_tokens, n_comp * 4u,
                n_head, 4, 1.0f, 1) ||
        !ds4_gpu_indexer_scores_decode_batch_tensor(
                score_causal_ref, q_unpack, weights, keys_unpack,
                n_comp, n_tokens, causal_pos0,
                n_head, head_dim, 4, 1.0f) ||
        !ds4_gpu_indexer_scores_packed_tensor(
                score_causal_fp4, q_packed, weights, keys_packed,
                n_comp, n_tokens, causal_pos0,
                n_head, 4, 1.0f, 1) ||
        !ds4_gpu_indexer_topk_tensor(topk_ref, score_ref,
                                     n_comp, n_tokens, top_k) ||
        !ds4_gpu_indexer_topk_tensor(topk_fp4, score_fp4,
                                     n_comp, n_tokens, top_k) ||
        !ds4_gpu_indexer_topk_tensor(topk_causal_ref, score_causal_ref,
                                     n_comp, n_tokens, top_k) ||
        !ds4_gpu_indexer_topk_tensor(topk_causal_fp4, score_causal_fp4,
                                     n_comp, n_tokens, top_k) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(q_ref, 0, q_ref_host, q_count * sizeof(float)) ||
        !ds4_gpu_tensor_read(q_unpack, 0, q_unpack_host,
                             q_count * sizeof(float)) ||
        !ds4_gpu_tensor_read(q_packed, 0, q_packed_host, q_packed_count) ||
        !ds4_gpu_tensor_read(keys_ref, 0, k_ref_host,
                             k_count * sizeof(float)) ||
        !ds4_gpu_tensor_read(keys_unpack, 0, k_unpack_host,
                             k_count * sizeof(float)) ||
        !ds4_gpu_tensor_read(keys_packed, 0, k_packed_host, k_packed_count) ||
        !ds4_gpu_tensor_read(score_ref, 0, score_ref_host,
                             score_count * sizeof(float)) ||
        !ds4_gpu_tensor_read(score_fp4, 0, score_fp4_host,
                             score_count * sizeof(float)) ||
        !ds4_gpu_tensor_read(score_causal_ref, 0, score_causal_ref_host,
                             score_count * sizeof(float)) ||
        !ds4_gpu_tensor_read(score_causal_fp4, 0, score_causal_fp4_host,
                             score_count * sizeof(float)) ||
        !ds4_gpu_tensor_read(topk_ref, 0, topk_ref_host,
                             (uint64_t)n_tokens * top_k * sizeof(uint32_t)) ||
        !ds4_gpu_tensor_read(topk_fp4, 0, topk_fp4_host,
                             (uint64_t)n_tokens * top_k * sizeof(uint32_t)) ||
        !ds4_gpu_tensor_read(topk_causal_ref, 0, topk_causal_ref_host,
                             (uint64_t)n_tokens * top_k * sizeof(uint32_t)) ||
        !ds4_gpu_tensor_read(topk_causal_fp4, 0, topk_causal_fp4_host,
                             (uint64_t)n_tokens * top_k * sizeof(uint32_t))) {
        goto cleanup;
    }

    double score_ss = 0.0;
    double score_ref_ss = 0.0;
    double causal_ss = 0.0;
    double causal_ref_ss = 0.0;
    float qat_max = 0.0f;
    for (uint64_t i = 0; i < q_count; i++) {
        const float d = fabsf(q_ref_host[i] - q_unpack_host[i]);
        if (d > qat_max) qat_max = d;
    }
    for (uint64_t i = 0; i < k_count; i++) {
        const float d = fabsf(k_ref_host[i] - k_unpack_host[i]);
        if (d > qat_max) qat_max = d;
    }
    for (uint64_t i = 0; i < score_count; i++) {
        const double d = (double)score_ref_host[i] - score_fp4_host[i];
        score_ss += d * d;
        score_ref_ss += (double)score_ref_host[i] * score_ref_host[i];
    }
    uint32_t overlap = 0;
    uint32_t causal_overlap = 0;
    int topk_equivalent = 1;
    int causal_topk_equivalent = 1;
    for (uint32_t t = 0; t < n_tokens; t++) {
        const uint32_t visible = (causal_pos0 + t + 1u) / 4u;
        for (uint32_t c = 0; c < n_comp; c++) {
            const uint64_t i = (uint64_t)t * n_comp + c;
            if (c >= visible) {
                if (!(isinf(score_causal_ref_host[i]) &&
                      score_causal_ref_host[i] < 0.0f &&
                      isinf(score_causal_fp4_host[i]) &&
                      score_causal_fp4_host[i] < 0.0f)) {
                    fprintf(stderr,
                            "MXFP4 causal-mask mismatch token=%u comp=%u visible=%u\n",
                            t, c, visible);
                    goto cleanup;
                }
            } else {
                const double d = (double)score_causal_ref_host[i] -
                                 score_causal_fp4_host[i];
                causal_ss += d * d;
                causal_ref_ss += (double)score_causal_ref_host[i] *
                                 score_causal_ref_host[i];
            }
        }
        uint32_t token_overlap = 0;
        if (!topk_boundary_equivalent(
                    topk_ref_host + (uint64_t)t * top_k,
                    topk_fp4_host + (uint64_t)t * top_k,
                    score_ref_host + (uint64_t)t * n_comp,
                    score_fp4_host + (uint64_t)t * n_comp,
                    n_comp, top_k, &token_overlap)) {
            fprintf(stderr, "MXFP4 Top-K boundary mismatch token=%u\n", t);
            topk_equivalent = 0;
        }
        overlap += token_overlap;
        token_overlap = 0;
        if (!topk_boundary_equivalent(
                    topk_causal_ref_host + (uint64_t)t * top_k,
                    topk_causal_fp4_host + (uint64_t)t * top_k,
                    score_causal_ref_host + (uint64_t)t * n_comp,
                    score_causal_fp4_host + (uint64_t)t * n_comp,
                    n_comp, top_k, &token_overlap)) {
            fprintf(stderr, "MXFP4 causal Top-K boundary mismatch token=%u\n", t);
            causal_topk_equivalent = 0;
        }
        causal_overlap += token_overlap;
    }
    const double rel_rmse = score_ref_ss > 0.0
        ? sqrt(score_ss / score_ref_ss) : sqrt(score_ss);
    const double causal_rel_rmse = causal_ref_ss > 0.0
        ? sqrt(causal_ss / causal_ref_ss) : sqrt(causal_ss);

    float shape_max = 0.0f;
    uint32_t shape_max_ulp = 0u;
    uint64_t shape_changed = 0u;
    for (uint32_t shape = 1; shape <= n_tokens; shape++) {
        if (!ds4_gpu_indexer_scores_packed_tensor(
                    score_shape, q_packed, weights, keys_packed,
                    n_comp, shape, n_comp * 4u,
                    n_head, 4, 1.0f, 1) ||
            !ds4_gpu_synchronize() ||
            !ds4_gpu_tensor_read(score_shape, 0, score_shape_host,
                                 (uint64_t)shape * n_comp * sizeof(float))) {
            goto cleanup;
        }
        for (uint64_t i = 0; i < (uint64_t)shape * n_comp; i++) {
            const float d = fabsf(score_shape_host[i] - score_fp4_host[i]);
            if (d > shape_max) shape_max = d;
            const uint32_t ulp = nonnegative_f32_ulp_distance(
                score_shape_host[i], score_fp4_host[i]);
            if (ulp != 0u) shape_changed++;
            if (ulp > shape_max_ulp) shape_max_ulp = ulp;
        }
    }

    if (!ds4_gpu_dsv4_indexer_pack_tensor(q_packed, q, q_rows) ||
        !ds4_gpu_dsv4_indexer_pack_tensor(keys_packed, keys, n_comp) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(q_packed, 0, q_packed_repeat_host,
                             q_packed_count) ||
        !ds4_gpu_tensor_read(keys_packed, 0, k_packed_repeat_host,
                             k_packed_count)) {
        goto cleanup;
    }
    const uint64_t packed_repeat_mismatches =
        (uint64_t)(memcmp(q_packed_repeat_host, q_packed_host,
                          (size_t)q_packed_count) != 0) +
        (uint64_t)(memcmp(k_packed_repeat_host, k_packed_host,
                          (size_t)k_packed_count) != 0);
    mxfp4_unpack_reference(q_host, q_packed_host, q_rows);
    mxfp4_unpack_reference(k_host, k_packed_host, n_comp);
    float wire_max = 0.0f;
    for (uint64_t i = 0; i < q_count; i++) {
        const float d = fabsf(q_host[i] - q_unpack_host[i]);
        if (d > wire_max) wire_max = d;
    }
    for (uint64_t i = 0; i < k_count; i++) {
        const float d = fabsf(k_host[i] - k_unpack_host[i]);
        if (d > wire_max) wire_max = d;
    }

    fprintf(stderr,
            "cuda-regression: MXFP4 indexer packed-repeat=%llu wire-max=%.8g "
            "qat-max=%.8g "
            "score-rel-rmse=%.6g causal=%.6g "
            "shape-consistency-max=%.8g ulp=%u changed=%llu "
            "topk=%u/%u causal-topk=%u/%u\n",
            (unsigned long long)packed_repeat_mismatches,
            (double)wire_max, (double)qat_max,
            rel_rmse, causal_rel_rmse, (double)shape_max,
            shape_max_ulp, (unsigned long long)shape_changed,
            overlap, n_tokens * top_k,
            causal_overlap, n_tokens * top_k);
    if (packed_repeat_mismatches == 0u && wire_max <= 1.0e-6f &&
        qat_max <= 1.0e-6f &&
        rel_rmse <= 5.0e-4 && causal_rel_rmse <= 5.0e-4 &&
        shape_max_ulp <= 2u && topk_equivalent &&
        causal_topk_equivalent) {
        rc = 0;
    }

cleanup:
    ds4_gpu_tensor_free(topk_causal_fp4);
    ds4_gpu_tensor_free(topk_causal_ref);
    ds4_gpu_tensor_free(topk_fp4);
    ds4_gpu_tensor_free(topk_ref);
    ds4_gpu_tensor_free(score_shape);
    ds4_gpu_tensor_free(score_causal_fp4);
    ds4_gpu_tensor_free(score_causal_ref);
    ds4_gpu_tensor_free(score_fp4);
    ds4_gpu_tensor_free(score_ref);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(keys_packed);
    ds4_gpu_tensor_free(keys_unpack);
    ds4_gpu_tensor_free(keys_ref);
    ds4_gpu_tensor_free(keys);
    ds4_gpu_tensor_free(q_packed);
    ds4_gpu_tensor_free(q_unpack);
    ds4_gpu_tensor_free(q_ref);
    ds4_gpu_tensor_free(q);
host_cleanup:
    free(topk_causal_fp4_host);
    free(topk_causal_ref_host);
    free(topk_fp4_host);
    free(topk_ref_host);
    free(k_packed_host);
    free(k_packed_repeat_host);
    free(q_packed_host);
    free(q_packed_repeat_host);
    free(score_shape_host);
    free(score_causal_fp4_host);
    free(score_causal_ref_host);
    free(score_fp4_host);
    free(score_ref_host);
    free(k_unpack_host);
    free(k_ref_host);
    free(q_unpack_host);
    free(q_ref_host);
    free(w_host);
    free(k_host);
    free(q_host);
    return rc;
}

static int check_physical_rn_mxfp4_topk(void) {
    enum {
        REQUESTS = 2,
        ROWS = 3,
        N_HEAD = 4,
        HEAD_DIM = 128,
        COMP_STRIDE = 640,
        TOP_K = 512,
    };
    const uint32_t request_offset[REQUESTS] = {0, 2};
    const uint32_t request_rows[REQUESTS] = {2, 1};
    const uint32_t request_position[REQUESTS] = {4096, 8192};
    const uint32_t request_n_comp[REQUESTS] = {640, 576};
    const uint64_t q_rows = (uint64_t)ROWS * N_HEAD;
    const uint64_t q_count = q_rows * HEAD_DIM;
    const uint64_t key_rows = (uint64_t)REQUESTS * COMP_STRIDE;
    const uint64_t key_count = key_rows * HEAD_DIM;
    const uint64_t packed_q_bytes =
        q_rows * DS4_GPU_INDEXER_FP4_ROW_BYTES;
    const uint64_t packed_key_bytes =
        key_rows * DS4_GPU_INDEXER_FP4_ROW_BYTES;
    const uint64_t selected_count = (uint64_t)ROWS * TOP_K;
    const uint64_t max_score_count =
        (uint64_t)request_rows[0] * request_n_comp[0];
    float *q_host = (float *)malloc((size_t)q_count * sizeof(float));
    float *key_host =
        (float *)malloc((size_t)key_count * sizeof(float));
    float weights_host[(uint64_t)ROWS * N_HEAD];
    uint32_t selected_rn_host[selected_count];
    uint32_t selected_ptr_host[selected_count];
    uint32_t selected_ref_host[selected_count];
    if (!q_host || !key_host) {
        free(key_host);
        free(q_host);
        return 1;
    }
    for (uint64_t i = 0; i < q_count; i++) {
        q_host[i] =
            ((float)((i * 23u) % 101u) - 50.0f) / 37.0f;
    }
    for (uint64_t i = 0; i < key_count; i++) {
        key_host[i] =
            ((float)((i * 29u) % 113u) - 56.0f) / 41.0f;
    }
    for (uint64_t i = 0; i < (uint64_t)ROWS * N_HEAD; i++) {
        weights_host[i] = 0.25f + (float)(i % N_HEAD) * 0.125f;
    }
    memset(selected_rn_host, 0xff, sizeof(selected_rn_host));
    memset(selected_ptr_host, 0xff, sizeof(selected_ptr_host));
    memset(selected_ref_host, 0xff, sizeof(selected_ref_host));

    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *keys =
        ds4_gpu_tensor_alloc(key_count * sizeof(float));
    ds4_gpu_tensor *q_packed =
        ds4_gpu_tensor_alloc(packed_q_bytes);
    ds4_gpu_tensor *keys_packed =
        ds4_gpu_tensor_alloc(packed_key_bytes);
    ds4_gpu_tensor *weights =
        ds4_gpu_tensor_alloc(sizeof(weights_host));
    ds4_gpu_tensor *score_rn =
        ds4_gpu_tensor_alloc(max_score_count * sizeof(float));
    ds4_gpu_tensor *score_ref =
        ds4_gpu_tensor_alloc(max_score_count * sizeof(float));
    ds4_gpu_tensor *selected_rn =
        ds4_gpu_tensor_alloc(selected_count * sizeof(uint32_t));
    ds4_gpu_tensor *selected_ptr =
        ds4_gpu_tensor_alloc(selected_count * sizeof(uint32_t));
    ds4_gpu_tensor *selected_ref =
        ds4_gpu_tensor_alloc(selected_count * sizeof(uint32_t));
    ds4_gpu_tensor *key_ptr0 = NULL;
    ds4_gpu_tensor *key_ptr1 = NULL;
    int rc = 1;
    if (!q || !keys || !q_packed || !keys_packed || !weights ||
        !score_rn || !score_ref || !selected_rn || !selected_ptr ||
        !selected_ref ||
        !ds4_gpu_tensor_write(q, 0, q_host,
                              q_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(keys, 0, key_host,
                              key_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(weights, 0, weights_host,
                              sizeof(weights_host)) ||
        !ds4_gpu_dsv4_indexer_pack_tensor(
            q_packed, q, (uint32_t)q_rows) ||
        !ds4_gpu_dsv4_indexer_pack_tensor(
            keys_packed, keys, (uint32_t)key_rows) ||
        !ds4_gpu_indexer_packed_topk_rn_tensor(
            selected_rn, score_rn, q_packed, weights, keys_packed,
            request_offset, request_rows, request_position,
            request_n_comp, REQUESTS, ROWS, COMP_STRIDE, N_HEAD, 4,
            1.0f / sqrtf((float)(N_HEAD * HEAD_DIM)), TOP_K)) {
        goto cleanup;
    }
    key_ptr0 = ds4_gpu_tensor_view(
        keys_packed, 0,
        (uint64_t)request_n_comp[0] *
            DS4_GPU_INDEXER_FP4_ROW_BYTES);
    key_ptr1 = ds4_gpu_tensor_view(
        keys_packed,
        (uint64_t)COMP_STRIDE * DS4_GPU_INDEXER_FP4_ROW_BYTES,
        (uint64_t)request_n_comp[1] *
            DS4_GPU_INDEXER_FP4_ROW_BYTES);
    const ds4_gpu_tensor *key_ptrs[REQUESTS] = {
        key_ptr0, key_ptr1
    };
    if (!key_ptr0 || !key_ptr1 ||
        !ds4_gpu_indexer_packed_topk_rn_ptrs_tensor(
            selected_ptr, score_rn, q_packed, weights, key_ptrs,
            request_offset, request_rows, request_position,
            request_n_comp, REQUESTS, ROWS, N_HEAD, 4,
            1.0f / sqrtf((float)(N_HEAD * HEAD_DIM)), TOP_K)) {
        goto cleanup;
    }

    for (uint32_t request = 0; request < REQUESTS; request++) {
        const uint32_t offset = request_offset[request];
        const uint32_t rows = request_rows[request];
        const uint32_t n_comp = request_n_comp[request];
        ds4_gpu_tensor *q_view = ds4_gpu_tensor_view(
            q_packed,
            (uint64_t)offset * N_HEAD *
                DS4_GPU_INDEXER_FP4_ROW_BYTES,
            (uint64_t)rows * N_HEAD *
                DS4_GPU_INDEXER_FP4_ROW_BYTES);
        ds4_gpu_tensor *weight_view = ds4_gpu_tensor_view(
            weights,
            (uint64_t)offset * N_HEAD * sizeof(float),
            (uint64_t)rows * N_HEAD * sizeof(float));
        ds4_gpu_tensor *key_view = ds4_gpu_tensor_view(
            keys_packed,
            (uint64_t)request * COMP_STRIDE *
                DS4_GPU_INDEXER_FP4_ROW_BYTES,
            (uint64_t)n_comp *
                DS4_GPU_INDEXER_FP4_ROW_BYTES);
        ds4_gpu_tensor *selected_view = ds4_gpu_tensor_view(
            selected_ref,
            (uint64_t)offset * TOP_K * sizeof(uint32_t),
            (uint64_t)rows * TOP_K * sizeof(uint32_t));
        const int ok =
            q_view && weight_view && key_view && selected_view &&
            ds4_gpu_indexer_scores_packed_tensor(
                score_ref, q_view, weight_view, key_view, n_comp, rows,
                request_position[request], N_HEAD, 4,
                1.0f / sqrtf((float)(N_HEAD * HEAD_DIM)), 1) &&
            ds4_gpu_indexer_topk_tensor(
                selected_view, score_ref, n_comp, rows, TOP_K);
        ds4_gpu_tensor_free(selected_view);
        ds4_gpu_tensor_free(key_view);
        ds4_gpu_tensor_free(weight_view);
        ds4_gpu_tensor_free(q_view);
        if (!ok) goto cleanup;
    }
    if (!ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(
            selected_rn, 0, selected_rn_host,
            sizeof(selected_rn_host)) ||
        !ds4_gpu_tensor_read(
            selected_ptr, 0, selected_ptr_host,
            sizeof(selected_ptr_host)) ||
        !ds4_gpu_tensor_read(
            selected_ref, 0, selected_ref_host,
            sizeof(selected_ref_host))) {
        goto cleanup;
    }
    for (uint64_t i = 0; i < selected_count; i++) {
        if (selected_rn_host[i] != selected_ref_host[i] ||
            selected_ptr_host[i] != selected_ref_host[i]) {
            fprintf(stderr,
                    "physical R=n MXFP4 Top-K mismatch at=%llu "
                    "arena=%u ptr=%u ref=%u\n",
                    (unsigned long long)i,
                    selected_rn_host[i],
                    selected_ptr_host[i],
                    selected_ref_host[i]);
            goto cleanup;
        }
    }
    fprintf(stderr,
            "cuda-regression: physical R=2 MXFP4 score+exact Top-K "
            "matches independent R=1 slices (arena + pointer handles)\n");
    rc = 0;

cleanup:
    ds4_gpu_tensor_free(key_ptr1);
    ds4_gpu_tensor_free(key_ptr0);
    ds4_gpu_tensor_free(selected_ref);
    ds4_gpu_tensor_free(selected_ptr);
    ds4_gpu_tensor_free(selected_rn);
    ds4_gpu_tensor_free(score_ref);
    ds4_gpu_tensor_free(score_rn);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(keys_packed);
    ds4_gpu_tensor_free(q_packed);
    ds4_gpu_tensor_free(keys);
    ds4_gpu_tensor_free(q);
    free(key_host);
    free(q_host);
    return rc;
}

static int check_physical_rn_compressor_frontiers(void) {
    enum {
        REQUESTS = 2,
        ROWS = 3,
        HEAD_DIM = 8,
        RATIO = 4,
        WIDTH = 2 * HEAD_DIM,
        STATE_ROWS = 2 * RATIO,
        STATE_VALUES = STATE_ROWS * WIDTH,
        COMP_CAP = 4,
    };
    const uint32_t request_offset[REQUESTS] = {0, 2};
    const uint32_t request_rows[REQUESTS] = {2, 1};
    const uint32_t request_position[REQUESTS] = {3, 7};
    const uint32_t request_n_comp[REQUESTS] = {1, 2};
    uint32_t request_n_comp_after[REQUESTS] = {0, 0};
    float kv_host[(uint64_t)ROWS * WIDTH];
    float sc_host[(uint64_t)ROWS * WIDTH];
    float state_kv_init[STATE_VALUES];
    float state_score_init[STATE_VALUES];
    float comp_init[(uint64_t)COMP_CAP * HEAD_DIM];
    for (uint64_t i = 0; i < (uint64_t)ROWS * WIDTH; i++) {
        kv_host[i] =
            ((float)((i * 17u) % 53u) - 26.0f) / 31.0f;
        sc_host[i] =
            ((float)((i * 13u) % 47u) - 23.0f) / 29.0f;
    }
    memset(state_kv_init, 0, sizeof(state_kv_init));
    for (uint32_t i = 0; i < STATE_VALUES; i++) {
        state_score_init[i] = -INFINITY;
    }
    memset(comp_init, 0, sizeof(comp_init));

    static float model_page[1024] __attribute__((aligned(4096)));
    memset(model_page, 0, sizeof(model_page));
    const uint64_t ape_offset = 0;
    const uint64_t norm_offset =
        (uint64_t)WIDTH * RATIO * sizeof(float);
    float *norm = (float *)((char *)model_page + norm_offset);
    for (uint32_t i = 0; i < HEAD_DIM; i++) norm[i] = 1.0f;

    ds4_gpu_tensor *kv =
        ds4_gpu_tensor_alloc(sizeof(kv_host));
    ds4_gpu_tensor *sc =
        ds4_gpu_tensor_alloc(sizeof(sc_host));
    ds4_gpu_tensor *state_kv_rn[REQUESTS] = {NULL, NULL};
    ds4_gpu_tensor *state_sc_rn[REQUESTS] = {NULL, NULL};
    ds4_gpu_tensor *comp_rn[REQUESTS] = {NULL, NULL};
    ds4_gpu_tensor *state_kv_ref[REQUESTS] = {NULL, NULL};
    ds4_gpu_tensor *state_sc_ref[REQUESTS] = {NULL, NULL};
    ds4_gpu_tensor *comp_ref[REQUESTS] = {NULL, NULL};
    int rc = 1;
    for (uint32_t request = 0; request < REQUESTS; request++) {
        state_kv_rn[request] =
            ds4_gpu_tensor_alloc(sizeof(state_kv_init));
        state_sc_rn[request] =
            ds4_gpu_tensor_alloc(sizeof(state_score_init));
        comp_rn[request] = ds4_gpu_tensor_alloc(sizeof(comp_init));
        state_kv_ref[request] =
            ds4_gpu_tensor_alloc(sizeof(state_kv_init));
        state_sc_ref[request] =
            ds4_gpu_tensor_alloc(sizeof(state_score_init));
        comp_ref[request] = ds4_gpu_tensor_alloc(sizeof(comp_init));
        if (!state_kv_rn[request] || !state_sc_rn[request] ||
            !comp_rn[request] || !state_kv_ref[request] ||
            !state_sc_ref[request] || !comp_ref[request] ||
            !ds4_gpu_tensor_write(
                state_kv_rn[request], 0, state_kv_init,
                sizeof(state_kv_init)) ||
            !ds4_gpu_tensor_write(
                state_sc_rn[request], 0, state_score_init,
                sizeof(state_score_init)) ||
            !ds4_gpu_tensor_write(
                comp_rn[request], 0, comp_init, sizeof(comp_init)) ||
            !ds4_gpu_tensor_write(
                state_kv_ref[request], 0, state_kv_init,
                sizeof(state_kv_init)) ||
            !ds4_gpu_tensor_write(
                state_sc_ref[request], 0, state_score_init,
                sizeof(state_score_init)) ||
            !ds4_gpu_tensor_write(
                comp_ref[request], 0, comp_init, sizeof(comp_init))) {
            goto cleanup;
        }
    }
    if (!kv || !sc ||
        !ds4_gpu_tensor_write(kv, 0, kv_host, sizeof(kv_host)) ||
        !ds4_gpu_tensor_write(sc, 0, sc_host, sizeof(sc_host)) ||
        !ds4_gpu_compressor_update_rn_tensor(
            kv, sc,
            state_kv_rn, state_sc_rn, comp_rn,
            request_offset, request_rows, request_position,
            request_n_comp, request_n_comp_after,
            REQUESTS, ROWS,
            model_page, sizeof(model_page),
            ape_offset, 0, norm_offset, 0,
            HEAD_DIM, RATIO, 4, 0,
            10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f,
            1.0e-6f)) {
        goto cleanup;
    }

    for (uint32_t request = 0; request < REQUESTS; request++) {
        uint32_t comp_row = request_n_comp[request];
        for (uint32_t row = 0; row < request_rows[request]; row++) {
            const uint32_t physical = request_offset[request] + row;
            ds4_gpu_tensor *kv_row = ds4_gpu_tensor_view(
                kv, (uint64_t)physical * WIDTH * sizeof(float),
                (uint64_t)WIDTH * sizeof(float));
            ds4_gpu_tensor *sc_row = ds4_gpu_tensor_view(
                sc, (uint64_t)physical * WIDTH * sizeof(float),
                (uint64_t)WIDTH * sizeof(float));
            const uint32_t pos = request_position[request] + row;
            const int ok = kv_row && sc_row &&
                ds4_gpu_compressor_update_tensor(
                    kv_row, sc_row,
                    state_kv_ref[request], state_sc_ref[request],
                    comp_ref[request],
                    model_page, sizeof(model_page),
                    ape_offset, 0, norm_offset, 0,
                    HEAD_DIM, RATIO, pos, comp_row, 4, 0,
                    10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f,
                    1.0e-6f);
            ds4_gpu_tensor_free(sc_row);
            ds4_gpu_tensor_free(kv_row);
            if (!ok) goto cleanup;
            if (((pos + 1u) % RATIO) == 0u) comp_row++;
        }
        if (request_n_comp_after[request] != comp_row) {
            fprintf(stderr,
                    "physical R=n compressor frontier mismatch "
                    "request=%u got=%u ref=%u\n",
                    request, request_n_comp_after[request], comp_row);
            goto cleanup;
        }
    }
    if (!ds4_gpu_synchronize()) goto cleanup;

    for (uint32_t request = 0; request < REQUESTS; request++) {
        float state_kv_rn_host[STATE_VALUES];
        float state_sc_rn_host[STATE_VALUES];
        float state_kv_ref_host[STATE_VALUES];
        float state_sc_ref_host[STATE_VALUES];
        float comp_rn_host[(uint64_t)COMP_CAP * HEAD_DIM];
        float comp_ref_host[(uint64_t)COMP_CAP * HEAD_DIM];
        if (!ds4_gpu_tensor_read(
                state_kv_rn[request], 0, state_kv_rn_host,
                sizeof(state_kv_rn_host)) ||
            !ds4_gpu_tensor_read(
                state_sc_rn[request], 0, state_sc_rn_host,
                sizeof(state_sc_rn_host)) ||
            !ds4_gpu_tensor_read(
                state_kv_ref[request], 0, state_kv_ref_host,
                sizeof(state_kv_ref_host)) ||
            !ds4_gpu_tensor_read(
                state_sc_ref[request], 0, state_sc_ref_host,
                sizeof(state_sc_ref_host)) ||
            !ds4_gpu_tensor_read(
                comp_rn[request], 0, comp_rn_host,
                sizeof(comp_rn_host)) ||
            !ds4_gpu_tensor_read(
                comp_ref[request], 0, comp_ref_host,
                sizeof(comp_ref_host))) {
            goto cleanup;
        }
        for (uint32_t i = 0; i < STATE_VALUES; i++) {
            const bool kv_equal =
                state_kv_rn_host[i] == state_kv_ref_host[i] ||
                fabsf(state_kv_rn_host[i] -
                      state_kv_ref_host[i]) <= 1.0e-6f;
            const bool sc_equal =
                state_sc_rn_host[i] == state_sc_ref_host[i] ||
                (isinf(state_sc_rn_host[i]) &&
                 isinf(state_sc_ref_host[i])) ||
                fabsf(state_sc_rn_host[i] -
                      state_sc_ref_host[i]) <= 1.0e-6f;
            if (!kv_equal || !sc_equal) {
                fprintf(stderr,
                        "physical R=n compressor state mismatch "
                        "request=%u at=%u\n",
                        request, i);
                goto cleanup;
            }
        }
        for (uint32_t i = 0; i < COMP_CAP * HEAD_DIM; i++) {
            if (comp_rn_host[i] != comp_ref_host[i] &&
                fabsf(comp_rn_host[i] - comp_ref_host[i]) > 1.0e-6f) {
                fprintf(stderr,
                        "physical R=n compressor cache mismatch "
                        "request=%u at=%u got=%g ref=%g\n",
                        request, i,
                        (double)comp_rn_host[i],
                        (double)comp_ref_host[i]);
                goto cleanup;
            }
        }
    }
    fprintf(stderr,
            "cuda-regression: physical R=2 compressor frontiers match "
            "independent R=1 sessions\n");
    rc = 0;

cleanup:
    for (uint32_t request = 0; request < REQUESTS; request++) {
        ds4_gpu_tensor_free(comp_ref[request]);
        ds4_gpu_tensor_free(state_sc_ref[request]);
        ds4_gpu_tensor_free(state_kv_ref[request]);
        ds4_gpu_tensor_free(comp_rn[request]);
        ds4_gpu_tensor_free(state_sc_rn[request]);
        ds4_gpu_tensor_free(state_kv_rn[request]);
    }
    ds4_gpu_tensor_free(sc);
    ds4_gpu_tensor_free(kv);
    return rc;
}

int main(void) {
    if (!ds4_gpu_init()) return 1;
    ds4_gpu_nvtx_range_push("ds4/regression/nvtx-link", 0);
    ds4_gpu_nvtx_range_pop();
    int rc = check_large_topk();
    if (rc != 0) fprintf(stderr, "cuda-regression: FAILED exact Top-K\n");
    if (check_frontier_copy_primitives() != 0) {
        fprintf(stderr, "cuda-regression: FAILED frontier copy primitives\n");
        rc = 1;
    }
    if (check_dspark_hybrid_blockv() != 0) {
        fprintf(stderr,
                "cuda-regression: FAILED HybridLC Block Verification\n");
        rc = 1;
    }
    if (check_gvr_topk() != 0) {
        fprintf(stderr, "cuda-regression: FAILED exact GVR Top-K\n");
        rc = 1;
    }
    if (check_mxfp4_indexer() != 0) {
        fprintf(stderr, "cuda-regression: FAILED packed MXFP4 indexer\n");
        rc = 1;
    }
    if (!ds4_gpu_mmq_prefill_self_test()) {
        fprintf(stderr, "cuda-regression: FAILED routed-MoE MMQ\n");
        rc = 1;
    }
    if (!ds4_gpu_prefill_epilogue_self_test()) {
        fprintf(stderr, "cuda-regression: FAILED fused prefill epilogue\n");
        rc = 1;
    }
    if (!ds4_gpu_attention_tokentile_self_test()) {
        fprintf(stderr, "cuda-regression: FAILED token-tile attention\n");
        rc = 1;
    }
    if (check_decode_attention_overflow_path() != 0) {
        fprintf(stderr, "cuda-regression: FAILED decode attention overflow\n");
        rc = 1;
    }
    if (check_physical_rn_dspark_attention() != 0) {
        fprintf(stderr,
                "cuda-regression: FAILED physical R=n DSpark attention\n");
        rc = 1;
    }
    if (check_physical_rn_indexed_attention() != 0) {
        fprintf(stderr,
                "cuda-regression: FAILED physical R=n indexed attention\n");
        rc = 1;
    }
    if (check_physical_rn_layout_primitives() != 0) {
        fprintf(stderr,
                "cuda-regression: FAILED physical R=n layout primitives\n");
        rc = 1;
    }
    if (check_physical_rn_mxfp4_topk() != 0) {
        fprintf(stderr,
                "cuda-regression: FAILED physical R=n MXFP4 Top-K\n");
        rc = 1;
    }
    if (check_physical_rn_compressor_frontiers() != 0) {
        fprintf(stderr,
                "cuda-regression: FAILED physical R=n compressor\n");
        rc = 1;
    }
    ds4_gpu_cleanup();
    if (rc == 0) puts("cuda long-context regression: OK");
    return rc;
}
