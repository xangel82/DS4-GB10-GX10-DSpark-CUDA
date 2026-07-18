#ifndef DS4_INDEXER_SM121_H
#define DS4_INDEXER_SM121_H

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    DS4_INDEXER_DIM = 128,
    DS4_INDEXER_FP4_BLOCK = 32,
    DS4_INDEXER_FP4_DATA_BYTES = 64,
    DS4_INDEXER_FP4_SCALE_BYTES = 4,
    DS4_INDEXER_FP4_ROW_BYTES = 68,
};

/* Apply DS4's normalized Hadamard transform and store four MXFP4 blocks.
 * Each 32-value block uses the native MMA nibble order a0/a16, a1/a17, ...;
 * a row is 64 E2M1 bytes followed by four UE8M0 scales. */
cudaError_t ds4_indexer_sm121_pack(
        void       *packed,
        const float *src,
        uint32_t     n_rows,
        cudaStream_t stream);

/* Expand packed rows to the exact F32 values represented by E2M1*UE8M0.
 * This is used by checkpoint compatibility and numerical regression tests. */
cudaError_t ds4_indexer_sm121_unpack(
        float       *dst,
        const void  *packed,
        uint32_t     n_rows,
        cudaStream_t stream);

/* Returns non-zero when this object contains the architecture-specific
 * SM121a block-scaled MMA implementation. */
int ds4_indexer_sm121_has_native_mxfp4(void);

/* Native SM121a block-scaled MMA scorer with a scalar packed fallback. The
 * score equation remains sum_h weight[t,h] * relu(dot(q[t,h], key[c])) * scale. */
cudaError_t ds4_indexer_sm121_scores(
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
        cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
