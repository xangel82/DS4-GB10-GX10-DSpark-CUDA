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
    const uint32_t n_tokens = 6;
    const uint32_t n_head = 64;
    const uint32_t head_dim = 128;
    const uint32_t n_comp = 640;
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
            "score-rel-rmse=%.6g causal=%.6g shapes1-6-max=%.8g "
            "topk=%u/%u causal-topk=%u/%u\n",
            (unsigned long long)packed_repeat_mismatches,
            (double)wire_max, (double)qat_max,
            rel_rmse, causal_rel_rmse, (double)shape_max,
            overlap, n_tokens * top_k,
            causal_overlap, n_tokens * top_k);
    if (packed_repeat_mismatches == 0u && wire_max <= 1.0e-6f &&
        qat_max <= 1.0e-6f &&
        rel_rmse <= 5.0e-4 && causal_rel_rmse <= 5.0e-4 &&
        shape_max <= 1.0e-6f && topk_equivalent &&
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

int main(void) {
    if (!ds4_gpu_init()) return 1;
    ds4_gpu_nvtx_range_push("ds4/regression/nvtx-link", 0);
    ds4_gpu_nvtx_range_pop();
    int rc = check_large_topk();
    if (rc != 0) fprintf(stderr, "cuda-regression: FAILED exact Top-K\n");
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
    if (!ds4_gpu_attention_tokentile_self_test()) {
        fprintf(stderr, "cuda-regression: FAILED token-tile attention\n");
        rc = 1;
    }
    if (check_decode_attention_overflow_path() != 0) {
        fprintf(stderr, "cuda-regression: FAILED decode attention overflow\n");
        rc = 1;
    }
    ds4_gpu_cleanup();
    if (rc == 0) puts("cuda long-context regression: OK");
    return rc;
}
