#include "ds4_indexer_sm121.h"

#include <math_constants.h>

namespace {

__device__ __forceinline__ float e2m1_value(uint32_t code) {
    switch (code & 7u) {
    case 0: return 0.0f;
    case 1: return 0.5f;
    case 2: return 1.0f;
    case 3: return 1.5f;
    case 4: return 2.0f;
    case 5: return 3.0f;
    case 6: return 4.0f;
    default: return 6.0f;
    }
}

__device__ __forceinline__ uint32_t e2m1_encode(float x) {
    const float ax = fminf(fabsf(x), 6.0f);
    uint32_t best = 0;
    float best_diff = ax;
#pragma unroll
    for (uint32_t code = 1; code < 8u; code++) {
        const float diff = fabsf(ax - e2m1_value(code));
        if (diff < best_diff ||
            (diff == best_diff && (code & 1u) == 0u && (best & 1u) != 0u)) {
            best = code;
            best_diff = diff;
        }
    }
    return best | (x < 0.0f ? 8u : 0u);
}

__device__ __forceinline__ float ue8m0_value(uint8_t scale) {
    return __uint_as_float((uint32_t)scale << 23u);
}

__device__ __forceinline__ float fp4_value(const uint8_t *row, uint32_t d) {
    const uint32_t block = d >> 5u;
    const uint32_t lane = d & 31u;
    const uint8_t packed = row[block * 16u + (lane & 15u)];
    const uint32_t code = (packed >> ((lane >> 4u) * 4u)) & 15u;
    const float value = e2m1_value(code);
    const float sign = (code & 8u) != 0u ? -1.0f : 1.0f;
    return sign * value * ue8m0_value(row[DS4_INDEXER_FP4_DATA_BYTES + (d >> 5u)]);
}

__global__ void indexer_pack_kernel(
        uint8_t     *packed,
        const float *src,
        uint32_t     n_rows) {
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (row >= n_rows || tid >= DS4_INDEXER_DIM) return;

    __shared__ float values[DS4_INDEXER_DIM];
    __shared__ float magnitudes[DS4_INDEXER_DIM];
    __shared__ float scales[DS4_INDEXER_FP4_SCALE_BYTES];

    values[tid] = src[(uint64_t)row * DS4_INDEXER_DIM + tid];
    __syncthreads();

    for (uint32_t stride = 1u; stride < DS4_INDEXER_DIM; stride <<= 1u) {
        if ((tid & stride) == 0u) {
            const uint32_t base =
                (tid & ~(2u * stride - 1u)) + (tid & (stride - 1u));
            const float a = values[base];
            const float b = values[base + stride];
            values[base] = a + b;
            values[base + stride] = a - b;
        }
        __syncthreads();
    }

    values[tid] *= 0.08838834764831845f;
    magnitudes[tid] = fabsf(values[tid]);
    __syncthreads();

    const uint32_t block = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t base = block * DS4_INDEXER_FP4_BLOCK;
    for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            magnitudes[base + lane] =
                fmaxf(magnitudes[base + lane], magnitudes[base + lane + stride]);
        }
        __syncthreads();
    }

    uint8_t *dst = packed + (uint64_t)row * DS4_INDEXER_FP4_ROW_BYTES;
    if (lane == 0u) {
        const float amax = fmaxf(magnitudes[base], 7.052966104933725e-38f);
        const float scale = exp2f(ceilf(log2f(amax / 6.0f)));
        scales[block] = scale;
        dst[DS4_INDEXER_FP4_DATA_BYTES + block] =
            (uint8_t)((__float_as_uint(scale) >> 23u) & 0xffu);
    }
    __syncthreads();

    if (tid < DS4_INDEXER_FP4_DATA_BYTES) {
        const uint32_t block = tid >> 4u;
        const uint32_t lane = tid & 15u;
        const uint32_t d0 = block * DS4_INDEXER_FP4_BLOCK + lane;
        const uint32_t d1 = d0 + 16u;
        const uint32_t c0 = e2m1_encode(values[d0] / scales[d0 >> 5u]);
        const uint32_t c1 = e2m1_encode(values[d1] / scales[d1 >> 5u]);
        dst[tid] = (uint8_t)(c0 | (c1 << 4u));
    }
}

__global__ void indexer_unpack_kernel(
        float         *dst,
        const uint8_t *packed,
        uint32_t       n_rows) {
    const uint32_t row = blockIdx.x;
    const uint32_t d = threadIdx.x;
    if (row >= n_rows || d >= DS4_INDEXER_DIM) return;
    dst[(uint64_t)row * DS4_INDEXER_DIM + d] =
        fp4_value(packed + (uint64_t)row * DS4_INDEXER_FP4_ROW_BYTES, d);
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (uint32_t mask = 16u; mask != 0u; mask >>= 1u) {
        value += __shfl_down_sync(0xffffffffu, value, mask);
    }
    return value;
}

__global__ void indexer_scores_packed_scalar_kernel(
        float         *scores,
        const uint8_t *q,
        const float   *weights,
        const uint8_t *keys,
        uint32_t       n_comp,
        uint32_t       n_tokens,
        uint32_t       pos0,
        uint32_t       n_head,
        uint32_t       ratio,
        float          scale,
        int            causal) {
    const uint32_t comp = blockIdx.x;
    const uint32_t token = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (comp >= n_comp || token >= n_tokens || tid >= 128u) return;
    if (causal && comp >= (pos0 + token + 1u) / ratio) {
        if (tid == 0u) scores[(uint64_t)token * n_comp + comp] = -CUDART_INF_F;
        return;
    }

    __shared__ float key_row[DS4_INDEXER_DIM];
    __shared__ float partial[4];
    const uint8_t *key = keys + (uint64_t)comp * DS4_INDEXER_FP4_ROW_BYTES;
    key_row[tid] = fp4_value(key, tid);
    __syncthreads();

    float total = 0.0f;
    for (uint32_t h0 = 0; h0 < n_head; h0 += 4u) {
        const uint32_t h = h0 + warp;
        float dot = 0.0f;
        if (h < n_head) {
            const uint8_t *qrow = q +
                ((uint64_t)token * n_head + h) * DS4_INDEXER_FP4_ROW_BYTES;
            const uint32_t d0 = lane * 4u;
#pragma unroll
            for (uint32_t j = 0; j < 4u; j++) {
                dot += fp4_value(qrow, d0 + j) * key_row[d0 + j];
            }
            dot = warp_sum(dot);
        }
        if (lane == 0u) {
            partial[warp] = h < n_head
                ? fmaxf(dot, 0.0f) * weights[(uint64_t)token * n_head + h]
                : 0.0f;
        }
        __syncthreads();
        if (tid == 0u) total += partial[0] + partial[1] + partial[2] + partial[3];
        __syncthreads();
    }
    if (tid == 0u) scores[(uint64_t)token * n_comp + comp] = total * scale;
}

#if defined(DS4_CUDA_SM121A_MXF4_MMA) && \
    defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
__device__ __forceinline__ void mxfp4_mma_m16n8k64(
        float &d0, float &d1, float &d2, float &d3,
        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
        uint32_t b0, uint32_t b1,
        float c0, float c1, float c2, float c3,
        uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X."
        "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0, %1, %2, %3},"
        "{%4, %5, %6, %7},"
        "{%8, %9},"
        "{%10, %11, %12, %13},"
        "%14, {0, 0}, %15, {0, 0};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3),
          "r"(sfa), "r"(sfb));
}
#endif

__device__ __forceinline__ uint32_t load_u32(const uint8_t *ptr) {
    return *reinterpret_cast<const uint32_t *>(ptr);
}

__device__ __forceinline__ uint32_t load_scale_pair(
        const uint8_t *row,
        uint32_t       half) {
    const uint32_t off = DS4_INDEXER_FP4_DATA_BYTES + half * 2u;
    return (uint32_t)row[off] | ((uint32_t)row[off + 1u] << 8u);
}

__global__ void indexer_scores_mxfp4_kernel(
        float         *scores,
        const uint8_t *q,
        const float   *weights,
        const uint8_t *keys,
        uint32_t       n_comp,
        uint32_t       n_tokens,
        uint32_t       pos0,
        uint32_t       n_head,
        uint32_t       ratio,
        float          scale,
        int            causal) {
#if defined(DS4_CUDA_SM121A_MXF4_MMA) && \
    defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const uint32_t tile_token = blockIdx.y * 16u;
    const uint32_t tile_comp = blockIdx.x * 64u;
    const uint32_t comp_base = tile_comp + warp * 8u;

    __shared__ __align__(16) uint8_t q_shared[16 * DS4_INDEXER_FP4_ROW_BYTES];
    __shared__ float weight_shared[16];

    const uint32_t n = comp_base + (lane >> 2u);
    const uint32_t k_byte = (lane & 3u) * 4u;
    uint32_t b00 = 0, b01 = 0, b10 = 0, b11 = 0;
    uint32_t sfb0 = 0, sfb1 = 0;
    if (n < n_comp) {
        const uint8_t *key = keys + (uint64_t)n * DS4_INDEXER_FP4_ROW_BYTES;
        b00 = load_u32(key + k_byte);
        b01 = load_u32(key + 16u + k_byte);
        b10 = load_u32(key + 32u + k_byte);
        b11 = load_u32(key + 48u + k_byte);
        sfb0 = load_scale_pair(key, 0u);
        sfb1 = load_scale_pair(key, 1u);
    }

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (uint32_t h = 0; h < n_head; h++) {
        for (uint32_t i = tid; i < 16u * 17u; i += blockDim.x) {
            const uint32_t m = i / 17u;
            const uint32_t word = i - m * 17u;
            const uint32_t token = tile_token + m;
            uint32_t value = 0;
            if (token < n_tokens) {
                const uint8_t *src = q +
                    ((uint64_t)token * n_head + h) * DS4_INDEXER_FP4_ROW_BYTES;
                value = load_u32(src + word * 4u);
            }
            reinterpret_cast<uint32_t *>(q_shared +
                (uint64_t)m * DS4_INDEXER_FP4_ROW_BYTES)[word] = value;
        }
        if (tid < 16u) {
            const uint32_t token = tile_token + tid;
            weight_shared[tid] = token < n_tokens
                ? weights[(uint64_t)token * n_head + h]
                : 0.0f;
        }
        __syncthreads();

        const uint32_t m0 = lane >> 2u;
        const uint32_t m1 = m0 + 8u;
        const uint32_t scale_m = (lane & 1u) * 8u + (lane >> 2u);
        const uint8_t *a_row0 = q_shared + (uint64_t)m0 * DS4_INDEXER_FP4_ROW_BYTES;
        const uint8_t *a_row1 = q_shared + (uint64_t)m1 * DS4_INDEXER_FP4_ROW_BYTES;
        const uint8_t *a_scale = q_shared +
            (uint64_t)scale_m * DS4_INDEXER_FP4_ROW_BYTES;

        float d0, d1, d2, d3;
        mxfp4_mma_m16n8k64(
            d0, d1, d2, d3,
            load_u32(a_row0 + k_byte),
            load_u32(a_row1 + k_byte),
            load_u32(a_row0 + 16u + k_byte),
            load_u32(a_row1 + 16u + k_byte),
            b00, b01,
            0.0f, 0.0f, 0.0f, 0.0f,
            load_scale_pair(a_scale, 0u), sfb0);
        mxfp4_mma_m16n8k64(
            d0, d1, d2, d3,
            load_u32(a_row0 + 32u + k_byte),
            load_u32(a_row1 + 32u + k_byte),
            load_u32(a_row0 + 48u + k_byte),
            load_u32(a_row1 + 48u + k_byte),
            b10, b11,
            d0, d1, d2, d3,
            load_scale_pair(a_scale, 1u), sfb1);

        acc0 += fmaxf(d0, 0.0f) * weight_shared[m0];
        acc1 += fmaxf(d1, 0.0f) * weight_shared[m0];
        acc2 += fmaxf(d2, 0.0f) * weight_shared[m1];
        acc3 += fmaxf(d3, 0.0f) * weight_shared[m1];
        __syncthreads();
    }

    const uint32_t col0 = comp_base + (lane & 3u) * 2u;
    const uint32_t row0 = tile_token + (lane >> 2u);
    const uint32_t row1 = row0 + 8u;
    if (row0 < n_tokens) {
        if (col0 < n_comp) {
            scores[(uint64_t)row0 * n_comp + col0] =
                causal && col0 >= (pos0 + row0 + 1u) / ratio
                    ? -CUDART_INF_F : acc0 * scale;
        }
        if (col0 + 1u < n_comp) {
            scores[(uint64_t)row0 * n_comp + col0 + 1u] =
                causal && col0 + 1u >= (pos0 + row0 + 1u) / ratio
                    ? -CUDART_INF_F : acc1 * scale;
        }
    }
    if (row1 < n_tokens) {
        if (col0 < n_comp) {
            scores[(uint64_t)row1 * n_comp + col0] =
                causal && col0 >= (pos0 + row1 + 1u) / ratio
                    ? -CUDART_INF_F : acc2 * scale;
        }
        if (col0 + 1u < n_comp) {
            scores[(uint64_t)row1 * n_comp + col0 + 1u] =
                causal && col0 + 1u >= (pos0 + row1 + 1u) / ratio
                    ? -CUDART_INF_F : acc3 * scale;
        }
    }
#else
    (void)scores; (void)q; (void)weights; (void)keys;
    (void)n_comp; (void)n_tokens; (void)pos0; (void)n_head;
    (void)ratio; (void)scale; (void)causal;
#endif
}

} // namespace

extern "C" cudaError_t ds4_indexer_sm121_pack(
        void       *packed,
        const float *src,
        uint32_t     n_rows,
        cudaStream_t stream) {
    if (!packed || !src || n_rows == 0u) return cudaErrorInvalidValue;
    indexer_pack_kernel<<<n_rows, DS4_INDEXER_DIM, 0, stream>>>(
        static_cast<uint8_t *>(packed), src, n_rows);
    return cudaGetLastError();
}

extern "C" cudaError_t ds4_indexer_sm121_unpack(
        float       *dst,
        const void  *packed,
        uint32_t     n_rows,
        cudaStream_t stream) {
    if (!dst || !packed || n_rows == 0u) return cudaErrorInvalidValue;
    indexer_unpack_kernel<<<n_rows, DS4_INDEXER_DIM, 0, stream>>>(
        dst, static_cast<const uint8_t *>(packed), n_rows);
    return cudaGetLastError();
}

extern "C" int ds4_indexer_sm121_has_native_mxfp4(void) {
#if defined(DS4_CUDA_SM121A_MXF4_MMA)
    return 1;
#else
    return 0;
#endif
}

extern "C" cudaError_t ds4_indexer_sm121_scores(
        float       *scores,
        const void  *q_packed,
        const float *weights,
        const void  *key_packed,
        uint32_t     n_comp,
        uint32_t     n_tokens,
        uint32_t     pos0,
        uint32_t     n_head,
        uint32_t     ratio,
        float        scale,
        int          causal,
        cudaStream_t stream) {
    if (!scores || !q_packed || !weights || !key_packed ||
        n_comp == 0u || n_tokens == 0u || n_head == 0u ||
        (causal && ratio == 0u)) {
        return cudaErrorInvalidValue;
    }

    static int cached_device = -1;
    static int cached_native = -1;
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return err;
    if (cached_device != device || cached_native < 0) {
        cudaDeviceProp prop{};
        err = cudaGetDeviceProperties(&prop, device);
        if (err != cudaSuccess) return err;
        cached_device = device;
#if defined(DS4_CUDA_SM121A_MXF4_MMA)
        cached_native = prop.major == 12 && prop.minor == 1 ? 1 : 0;
#else
        cached_native = 0;
#endif
    }
    if (cached_native) {
        const dim3 grid((n_comp + 63u) / 64u, (n_tokens + 15u) / 16u, 1u);
        indexer_scores_mxfp4_kernel<<<grid, 256, 0, stream>>>(
            scores,
            static_cast<const uint8_t *>(q_packed),
            weights,
            static_cast<const uint8_t *>(key_packed),
            n_comp, n_tokens, pos0, n_head, ratio, scale, causal);
    } else {
        const dim3 grid(n_comp, n_tokens, 1u);
        indexer_scores_packed_scalar_kernel<<<grid, 128, 0, stream>>>(
            scores,
            static_cast<const uint8_t *>(q_packed),
            weights,
            static_cast<const uint8_t *>(key_packed),
            n_comp, n_tokens, pos0, n_head, ratio, scale, causal);
    }
    return cudaGetLastError();
}
