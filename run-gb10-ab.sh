#!/usr/bin/env bash
# GB10/GX10 DSpark CUDA modifications:
# Copyright (c) 2026 Marco Palaferri. Licensed under the MIT License.
set -euo pipefail

# Run this script from the ds4 checkout on Athena.
# It never changes the existing ds4/athena service; it only selects runtime
# flags for this process.

MODE="${1:-baseline}"
MODEL="${DS4_MODEL:-/home/athena/ds4/ds4flash.gguf}"
PROMPT_FILE="${DS4_PROMPT_FILE:-speed-bench/promessi_sposi.txt}"
CTX_MAX="${DS4_CTX_MAX:-32768}"
GEN_TOKENS="${DS4_GEN_TOKENS:-256}"

case "$MODE" in
  baseline)
    # Remove experiment-only variables inherited from the shell/service.
    exec env \
      -u DS4_CUDA_COPY_MODEL \
      -u DS4_CUDA_COPY_MODEL_CHUNKED \
      -u DS4_CUDA_Q8_F16_ALL \
      -u DS4_CUDA_Q8_F16_CACHE_MB \
      -u DS4_CUDA_DEFER_END_SYNC \
      -u DS4_CUDA_FUSED_COMPRESSOR_UPDATE \
      -u DS4_CUDA_TOKEN_GRAPH \
      -u DS4_CUDA_TOKEN_GRAPH_PIPELINE \
      -u DS4_CUDA_COALESCED_F16_MATMUL \
      -u DS4_CUDA_Q8_U16_LOADS \
      -u DS4_CUDA_GREEDY_ARGMAX \
      -u DS4_CUDA_ATTN_HEADS2 \
      -u DS4_CUDA_ATTN_HEADS2_MIN_ROWS \
      -u DS4_CUDA_WEIGHT_CACHE_LIMIT_GB \
      -u DS4_CUDA_WEIGHT_CACHE_VERBOSE \
      -u DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS \
      ./ds4-bench -m "$MODEL" \
        --prompt-file "$PROMPT_FILE" \
        --ctx-start 2048 \
        --ctx-max "$CTX_MAX" \
        --step-incr 4096 \
        --gen-tokens "$GEN_TOKENS"
    ;;
  experiment)
    # Experiment 1: keep the complete Q2 model in device memory.
    # Experiment 2: enable only the existing selective Q8->F16 cache, capped
    # conservatively so the 128 GB Spark keeps workspace/KV headroom.
    exec env \
      DS4_CUDA_COPY_MODEL=1 \
      DS4_CUDA_WEIGHT_CACHE_LIMIT_GB="${DS4_CUDA_WEIGHT_CACHE_LIMIT_GB:-96}" \
      DS4_CUDA_Q8_F16_CACHE_MB="${DS4_CUDA_Q8_F16_CACHE_MB:-12288}" \
      DS4_CUDA_DEFER_END_SYNC=1 \
      DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS=0 \
      DS4_CUDA_FUSED_COMPRESSOR_UPDATE=1 \
      DS4_CUDA_TOKEN_GRAPH=1 \
      DS4_CUDA_TOKEN_GRAPH_PIPELINE=1 \
      DS4_CUDA_COALESCED_F16_MATMUL=1 \
      DS4_CUDA_Q8_U16_LOADS=1 \
      DS4_CUDA_ATTN_HEADS2=1 \
      DS4_CUDA_ATTN_HEADS2_MIN_ROWS="${DS4_ATTN_HEADS2_MIN_ROWS:-384}" \
      ./ds4-bench -m "$MODEL" \
        --prompt-file "$PROMPT_FILE" \
        --ctx-start 2048 \
        --ctx-max "$CTX_MAX" \
        --step-incr 4096 \
        --gen-tokens "$GEN_TOKENS"
    ;;
  *)
    echo "usage: $0 {baseline|experiment}" >&2
    exit 2
    ;;
esac
