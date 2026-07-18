#include "ds4_topk_gvr.h"

#include "ds4_topk_radix.h"

#include <cub/block/block_radix_sort.cuh>
#include <math_constants.h>

namespace {

constexpr uint32_t kThreads = 512u;
constexpr uint32_t kTopK = 512u;
constexpr uint32_t kCandidateCap = 2048u;
constexpr uint32_t kItemsPerThread = kCandidateCap / kThreads;
constexpr uint32_t kMaxSecantIterations = 6u;

__device__ __forceinline__ float canonical_score(float value) {
    if (isnan(value)) return -CUDART_INF_F;
    return value == 0.0f ? 0.0f : value;
}

__device__ __forceinline__ uint64_t score_key(float value, uint32_t index) {
    value = canonical_score(value);
    const uint32_t bits = __float_as_uint(value);
    const uint32_t ordered = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
    return ((uint64_t)ordered << 32u) | (uint64_t)(0xffffffffu - index);
}

__device__ __forceinline__ float warp_min(float value) {
    for (uint32_t delta = 16u; delta != 0u; delta >>= 1u) {
        value = fminf(value, __shfl_down_sync(0xffffffffu, value, delta));
    }
    return value;
}

__device__ __forceinline__ float warp_max(float value) {
    for (uint32_t delta = 16u; delta != 0u; delta >>= 1u) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, delta));
    }
    return value;
}

__device__ __forceinline__ uint32_t warp_sum_u32(uint32_t value) {
    for (uint32_t delta = 16u; delta != 0u; delta >>= 1u) {
        value += __shfl_down_sync(0xffffffffu, value, delta);
    }
    return value;
}

using CandidateSort =
    cub::BlockRadixSort<uint64_t, kThreads, kItemsPerThread>;

__global__ void gvr_exact_512_kernel(
        uint32_t       *selected,
        const float    *scores,
        const uint32_t *previous,
        uint8_t        *fallback_mask,
        uint32_t        n_comp,
        uint32_t        n_tokens) {
    const uint32_t token = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (token >= n_tokens || tid >= kThreads) return;

    const float *row = scores + (uint64_t)token * n_comp;
    const uint32_t *hint = previous + (uint64_t)token * kTopK;

    __shared__ uint64_t candidates[kCandidateCap];
    __shared__ uint32_t per_thread_counts[kThreads];
    __shared__ uint32_t warp_counts[16];
    __shared__ uint32_t warp_offsets[16];
    __shared__ float warp_mins[16];
    __shared__ float warp_maxs[16];
    __shared__ typename CandidateSort::TempStorage sort_storage;
    __shared__ float threshold;
    __shared__ float threshold_lo;
    __shared__ float threshold_hi;
    __shared__ uint32_t count_lo;
    __shared__ uint32_t count_hi;
    __shared__ uint32_t candidate_count;
    __shared__ uint32_t guess_done;
    __shared__ uint32_t invalid_hint;

    if (tid == 0u) {
        fallback_mask[token] = 1u;
        guess_done = 0u;
        invalid_hint = 0u;
        threshold = 0.0f;
        threshold_lo = 0.0f;
        threshold_hi = 0.0f;
        count_lo = n_comp;
        count_hi = 0u;
        candidate_count = 0u;
    }
    __syncthreads();

    const uint32_t predicted = hint[tid];
    float predicted_score = 0.0f;
    if (predicted >= n_comp) {
        atomicOr(&invalid_hint, 1u);
    } else {
        predicted_score = canonical_score(row[predicted]);
        if (!isfinite(predicted_score)) atomicOr(&invalid_hint, 1u);
    }

    float min_value = warp_min(predicted_score);
    float max_value = warp_max(predicted_score);
    if (lane == 0u) {
        warp_mins[warp] = min_value;
        warp_maxs[warp] = max_value;
    }
    __syncthreads();
    if (warp == 0u) {
        min_value = lane < 16u ? warp_mins[lane] : CUDART_INF_F;
        max_value = lane < 16u ? warp_maxs[lane] : -CUDART_INF_F;
        min_value = warp_min(min_value);
        max_value = warp_max(max_value);
        if (lane == 0u && invalid_hint == 0u && min_value < max_value) {
            threshold_lo = min_value;
            threshold_hi = max_value;
            count_lo = n_comp;
            count_hi = 1u;
            /* The weakest score in the previous exact Top-K is the useful
             * temporal guess. With unique hints it starts with at least K
             * candidates, while the former mean almost always started above
             * the Kth score and forced a full Radix fallback. */
            threshold = min_value;
        }
    }
    __syncthreads();
    if (invalid_hint != 0u || !(threshold_lo < threshold_hi) ||
        !(threshold >= threshold_lo && threshold < threshold_hi)) {
        return;
    }

    for (uint32_t iteration = 0u;
         iteration < kMaxSecantIterations;
         iteration++) {
        const float current_threshold = threshold;
        uint32_t local_count = 0u;
        for (uint32_t i = tid; i < n_comp; i += kThreads) {
            local_count += canonical_score(row[i]) >= current_threshold ? 1u : 0u;
        }
        per_thread_counts[tid] = local_count;
        uint32_t block_count = warp_sum_u32(local_count);
        if (lane == 0u) warp_counts[warp] = block_count;
        __syncthreads();
        if (warp == 0u) {
            block_count = lane < 16u ? warp_counts[lane] : 0u;
            block_count = warp_sum_u32(block_count);
            if (lane == 0u) candidate_count = block_count;
        }
        __syncthreads();

        if (tid == 0u) {
            const uint32_t count = candidate_count;
            if (count >= kTopK && count <= kCandidateCap) {
                guess_done = 1u;
            } else if (count < kTopK) {
                /* Duplicate/corrupt hints can violate the expected lower
                 * bound. Preserve exactness by taking the masked Radix path. */
                guess_done = 2u;
            } else {
                threshold_lo = current_threshold;
                count_lo = count;
                const float target =
                    0.5f * (float)(kTopK + kCandidateCap);
                const float denominator =
                    (float)count_lo - (float)count_hi;
                float fraction = denominator > 0.0f
                    ? ((float)count_lo - target) / denominator
                    : 0.5f;
                if (!isfinite(fraction)) fraction = 0.5f;
                fraction = fminf(fmaxf(fraction, 0.05f), 0.95f);
                if (iteration == 0u) fraction = fminf(fraction, 0.50f);
                float next = fmaf(fraction,
                                  threshold_hi - threshold_lo,
                                  threshold_lo);
                if (!(next > threshold_lo && next < threshold_hi)) {
                    next = threshold_lo +
                           0.5f * (threshold_hi - threshold_lo);
                }
                threshold = next;
            }
        }
        __syncthreads();
        if (guess_done != 0u) break;
    }
    if (guess_done != 1u) return;

    /* Per-thread counts from the final verification scan become deterministic
     * write ranges.  Each thread rescans only its strided lane and writes
     * without ballots, shuffles, or atomics in the collection pass. */
    uint32_t inclusive = per_thread_counts[tid];
#pragma unroll
    for (uint32_t delta = 1u; delta < 32u; delta <<= 1u) {
        const uint32_t other = __shfl_up_sync(0xffffffffu, inclusive, delta);
        if (lane >= delta) inclusive += other;
    }
    const uint32_t warp_exclusive = inclusive - per_thread_counts[tid];
    if (lane == 31u) warp_counts[warp] = inclusive;
    __syncthreads();
    if (warp == 0u) {
        uint32_t warp_inclusive = lane < 16u ? warp_counts[lane] : 0u;
#pragma unroll
        for (uint32_t delta = 1u; delta < 32u; delta <<= 1u) {
            const uint32_t other =
                __shfl_up_sync(0xffffffffu, warp_inclusive, delta);
            if (lane >= delta) warp_inclusive += other;
        }
        if (lane < 16u) {
            warp_offsets[lane] = warp_inclusive - warp_counts[lane];
        }
    }
    __syncthreads();

    uint32_t write = warp_offsets[warp] + warp_exclusive;
    const float final_threshold = threshold;
    for (uint32_t i = tid; i < n_comp; i += kThreads) {
        const float value = canonical_score(row[i]);
        if (value >= final_threshold) {
            candidates[write++] = score_key(value, i);
        }
    }
    __syncthreads();

    uint64_t keys[kItemsPerThread];
    const uint32_t base = tid * kItemsPerThread;
#pragma unroll
    for (uint32_t item = 0u; item < kItemsPerThread; item++) {
        const uint32_t rank = base + item;
        keys[item] = rank < candidate_count ? candidates[rank] : 0u;
    }
    __syncthreads();
    CandidateSort(sort_storage).SortDescending(keys);
#pragma unroll
    for (uint32_t item = 0u; item < kItemsPerThread; item++) {
        const uint32_t rank = base + item;
        if (rank < kTopK) {
            selected[(uint64_t)token * kTopK + rank] =
                0xffffffffu - (uint32_t)keys[item];
        }
    }
    if (tid == 0u) fallback_mask[token] = 0u;
}

} // namespace

extern "C" cudaError_t ds4_topk_gvr_exact_512(
        uint32_t       *selected,
        const float    *scores,
        const uint32_t *previous,
        uint8_t        *fallback_mask,
        uint32_t        n_comp,
        uint32_t        n_tokens,
        cudaStream_t    stream) {
    if (!selected || !scores || !previous || !fallback_mask ||
        n_comp < kTopK || n_tokens == 0u) {
        return cudaErrorInvalidValue;
    }
    gvr_exact_512_kernel<<<n_tokens, kThreads, 0, stream>>>(
        selected, scores, previous, fallback_mask, n_comp, n_tokens);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    return ds4_topk_radix_exact_512_masked(
        selected, scores, fallback_mask, n_comp, n_tokens, stream);
}
