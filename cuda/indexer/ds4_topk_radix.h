#ifndef DS4_TOPK_RADIX_H
#define DS4_TOPK_RADIX_H

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t ds4_topk_radix_exact_512(
        uint32_t    *selected,
        const float *scores,
        uint32_t     n_comp,
        uint32_t     n_tokens,
        cudaStream_t stream);

/* Run the exact radix fallback only for rows whose mask byte is non-zero. */
cudaError_t ds4_topk_radix_exact_512_masked(
        uint32_t       *selected,
        const float    *scores,
        const uint8_t  *fallback_mask,
        uint32_t        n_comp,
        uint32_t        n_tokens,
        cudaStream_t    stream);

#ifdef __cplusplus
}
#endif

#endif
