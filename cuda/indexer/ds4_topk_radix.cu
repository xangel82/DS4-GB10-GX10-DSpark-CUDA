#include "ds4_topk_radix.h"

#include <cub/block/block_radix_sort.cuh>
#include <math_constants.h>

namespace {

__device__ __forceinline__ uint64_t score_key(float value, uint32_t index) {
    if (isnan(value)) value = -CUDART_INF_F;
    if (value == 0.0f) value = 0.0f; /* +0 and -0 compare equal in FP32. */
    const uint32_t bits = __float_as_uint(value);
    const uint32_t ordered = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
    return ((uint64_t)ordered << 32u) | (uint64_t)(0xffffffffu - index);
}

template <int ITEMS>
__device__ __forceinline__ void write_sorted_indices(
        uint32_t *selected,
        uint64_t (&keys)[ITEMS],
        uint32_t token) {
    const uint32_t rank0 = threadIdx.x * ITEMS;
#pragma unroll
    for (uint32_t item = 0; item < ITEMS; item++) {
        const uint32_t rank = rank0 + item;
        if (rank < 512u) {
            selected[(uint64_t)token * 512u + rank] =
                0xffffffffu - (uint32_t)keys[item];
        }
    }
}

__global__ void topk_radix_exact_512_kernel(
        uint32_t    *selected,
        const float *scores,
        const uint8_t *fallback_mask,
        uint32_t     n_comp,
        uint32_t     n_tokens) {
    constexpr uint32_t THREADS = 512u;
    constexpr uint32_t CANDIDATE_CAP = 1024u;
    using Sort1024 = cub::BlockRadixSort<uint64_t, THREADS, 2>;
    using Sort512 = cub::BlockRadixSort<uint64_t, THREADS, 1>;
    union SortStorage {
        typename Sort1024::TempStorage sort1024;
        typename Sort512::TempStorage sort512;
    };

    const uint32_t token = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (token >= n_tokens || tid >= THREADS ||
        (fallback_mask && fallback_mask[token] == 0u)) return;
    const float *row = scores + (uint64_t)token * n_comp;

    __shared__ uint32_t histogram[16][256];
    __shared__ uint64_t candidates[CANDIDATE_CAP];
    __shared__ uint64_t prefix;
    __shared__ uint64_t threshold;
    __shared__ uint32_t rank_in_bucket;
    __shared__ uint32_t candidate_count;
    __shared__ SortStorage sort_storage;

    if (tid == 0u) {
        prefix = 0u;
        rank_in_bucket = 511u;
    }
    __syncthreads();

    /* The common path narrows with the two most-significant key bytes.  This
     * is enough for ordinary indexer score distributions and scans the score
     * row only three times before sorting at most 1024 candidates. */
    for (uint32_t pass = 0; pass < 2u; pass++) {
        for (uint32_t i = tid; i < 16u * 256u; i += THREADS) {
            histogram[i >> 8u][i & 255u] = 0u;
        }
        __syncthreads();
        const uint32_t shift = 56u - pass * 8u;
        for (uint32_t i = tid; i < n_comp; i += THREADS) {
            const uint64_t key = score_key(row[i], i);
            if (pass == 0u || (key >> (64u - pass * 8u)) == prefix) {
                atomicAdd(&histogram[tid >> 5u][(key >> shift) & 0xffu], 1u);
            }
        }
        __syncthreads();
        if (tid < 256u) {
            uint32_t total = histogram[0][tid];
#pragma unroll
            for (uint32_t w = 1u; w < 16u; w++) total += histogram[w][tid];
            histogram[0][tid] = total;
        }
        __syncthreads();
        if (tid == 0u) {
            uint32_t rank = rank_in_bucket;
            uint32_t above = 0u;
            uint32_t chosen = 0u;
            for (int bucket = 255; bucket >= 0; bucket--) {
                const uint32_t count = histogram[0][bucket];
                if (rank < above + count) {
                    chosen = (uint32_t)bucket;
                    rank -= above;
                    break;
                }
                above += count;
            }
            prefix = (prefix << 8u) | chosen;
            rank_in_bucket = rank;
        }
        __syncthreads();
    }

    if (tid == 0u) candidate_count = 0u;
    __syncthreads();
    for (uint32_t i = tid; i < n_comp; i += THREADS) {
        const uint64_t key = score_key(row[i], i);
        if ((key >> 48u) >= prefix) {
            const uint32_t slot = atomicAdd(&candidate_count, 1u);
            if (slot < CANDIDATE_CAP) candidates[slot] = key;
        }
    }
    __syncthreads();

    if (candidate_count <= CANDIDATE_CAP) {
        uint64_t keys[2];
        const uint32_t base = tid * 2u;
        keys[0] = base < candidate_count ? candidates[base] : 0u;
        keys[1] = base + 1u < candidate_count ? candidates[base + 1u] : 0u;
        __syncthreads();
        Sort1024(sort_storage.sort1024).SortDescending(keys);
        write_sorted_indices(selected, keys, token);
        return;
    }

    /* Highly clustered scores can put more than 1024 values in the same
     * high-16 bucket. Continue the exact MSD radix selection to a unique
     * score/index key. This path is bounded and allocates no global scratch. */
    for (uint32_t pass = 2u; pass < 8u; pass++) {
        for (uint32_t i = tid; i < 16u * 256u; i += THREADS) {
            histogram[i >> 8u][i & 255u] = 0u;
        }
        __syncthreads();
        const uint32_t shift = 56u - pass * 8u;
        for (uint32_t i = tid; i < n_comp; i += THREADS) {
            const uint64_t key = score_key(row[i], i);
            if ((key >> (64u - pass * 8u)) == prefix) {
                atomicAdd(&histogram[tid >> 5u][(key >> shift) & 0xffu], 1u);
            }
        }
        __syncthreads();
        if (tid < 256u) {
            uint32_t total = histogram[0][tid];
#pragma unroll
            for (uint32_t w = 1u; w < 16u; w++) total += histogram[w][tid];
            histogram[0][tid] = total;
        }
        __syncthreads();
        if (tid == 0u) {
            uint32_t rank = rank_in_bucket;
            uint32_t above = 0u;
            uint32_t chosen = 0u;
            for (int bucket = 255; bucket >= 0; bucket--) {
                const uint32_t count = histogram[0][bucket];
                if (rank < above + count) {
                    chosen = (uint32_t)bucket;
                    rank -= above;
                    break;
                }
                above += count;
            }
            prefix = (prefix << 8u) | chosen;
            rank_in_bucket = rank;
        }
        __syncthreads();
    }
    if (tid == 0u) {
        threshold = prefix;
        candidate_count = 0u;
    }
    __syncthreads();
    for (uint32_t i = tid; i < n_comp; i += THREADS) {
        const uint64_t key = score_key(row[i], i);
        if (key >= threshold) {
            const uint32_t slot = atomicAdd(&candidate_count, 1u);
            if (slot < 512u) candidates[slot] = key;
        }
    }
    __syncthreads();

    uint64_t keys[1] = {
        tid < candidate_count && tid < 512u ? candidates[tid] : 0u
    };
    __syncthreads();
    Sort512(sort_storage.sort512).SortDescending(keys);
    write_sorted_indices(selected, keys, token);
}

} // namespace

extern "C" cudaError_t ds4_topk_radix_exact_512(
        uint32_t    *selected,
        const float *scores,
        uint32_t     n_comp,
        uint32_t     n_tokens,
        cudaStream_t stream) {
    if (!selected || !scores || n_comp < 512u || n_tokens == 0u) {
        return cudaErrorInvalidValue;
    }
    topk_radix_exact_512_kernel<<<n_tokens, 512, 0, stream>>>(
        selected, scores, NULL, n_comp, n_tokens);
    return cudaGetLastError();
}

extern "C" cudaError_t ds4_topk_radix_exact_512_masked(
        uint32_t       *selected,
        const float    *scores,
        const uint8_t  *fallback_mask,
        uint32_t        n_comp,
        uint32_t        n_tokens,
        cudaStream_t    stream) {
    if (!selected || !scores || !fallback_mask || n_comp < 512u ||
        n_tokens == 0u) {
        return cudaErrorInvalidValue;
    }
    topk_radix_exact_512_kernel<<<n_tokens, 512, 0, stream>>>(
        selected, scores, fallback_mask, n_comp, n_tokens);
    return cudaGetLastError();
}
