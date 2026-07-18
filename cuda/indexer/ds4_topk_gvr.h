#ifndef DS4_TOPK_GVR_H
#define DS4_TOPK_GVR_H

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Exact K=512 Guess-Verify-Refine.  Every row owns a 512-index hint and one
 * fallback byte.  Failed guesses are completed by the exact radix kernel on
 * the same stream, without a host synchronization. */
cudaError_t ds4_topk_gvr_exact_512(
        uint32_t       *selected,
        const float    *scores,
        const uint32_t *previous,
        uint8_t        *fallback_mask,
        uint32_t        n_comp,
        uint32_t        n_tokens,
        cudaStream_t    stream);

#ifdef __cplusplus
}
#endif

#endif
