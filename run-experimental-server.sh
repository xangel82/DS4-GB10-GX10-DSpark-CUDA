#!/usr/bin/env bash
# GB10/GX10 DSpark CUDA modifications:
# Copyright (c) 2026 Marco Palaferri. Licensed under the MIT License.
set -euo pipefail

# Isolated experimental server. Run this from the lab checkout on Athena.
# It replaces the original process temporarily while keeping the same API port.

MODEL="${DS4_MODEL:-/home/athena/ds4/ds4flash.gguf}"
KV_DIR="${DS4_EXPERIMENT_KV_DIR:-/tmp/ds4-gb10-experiment-kv}"
CTX="${DS4_CTX:-131072}"
MAX_TOKENS="${DS4_MAX_TOKENS:-2200}"
THREADS="${DS4_THREADS:-10}"
TELEMETRY="${DS4_TELEMETRY:-0}"
ENABLE_PIPELINE="${DS4_ENABLE_TOKEN_PIPELINE:-1}"
ENABLE_GREEDY_ARGMAX="${DS4_ENABLE_GREEDY_ARGMAX:-1}"
ENABLE_ATTN_HEADS2="${DS4_ENABLE_ATTN_HEADS2:-1}"
ATTN_HEADS2_MIN_ROWS="${DS4_ATTN_HEADS2_MIN_ROWS:-384}"

mkdir -p "$KV_DIR"

export DS4_CUDA_COPY_MODEL="${DS4_CUDA_COPY_MODEL:-1}"
export DS4_CUDA_WEIGHT_CACHE_LIMIT_GB="${DS4_CUDA_WEIGHT_CACHE_LIMIT_GB:-96}"
export DS4_CUDA_Q8_F16_CACHE_MB="${DS4_CUDA_Q8_F16_CACHE_MB:-12288}"
export DS4_CUDA_DEFER_END_SYNC="${DS4_CUDA_DEFER_END_SYNC:-1}"
export DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS="${DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS:-0}"
export DS4_CUDA_FUSED_COMPRESSOR_UPDATE="${DS4_CUDA_FUSED_COMPRESSOR_UPDATE:-1}"
export DS4_CUDA_TOKEN_GRAPH="${DS4_CUDA_TOKEN_GRAPH:-1}"
export DS4_CUDA_COALESCED_F16_MATMUL="${DS4_CUDA_COALESCED_F16_MATMUL:-1}"
export DS4_CUDA_Q8_U16_LOADS="${DS4_CUDA_Q8_U16_LOADS:-1}"
if [[ "$ENABLE_PIPELINE" == "1" ]]; then
  export DS4_CUDA_TOKEN_GRAPH_PIPELINE=1
else
  unset DS4_CUDA_TOKEN_GRAPH_PIPELINE
fi
if [[ "$ENABLE_GREEDY_ARGMAX" == "1" ]]; then
  export DS4_CUDA_GREEDY_ARGMAX=1
else
  unset DS4_CUDA_GREEDY_ARGMAX
fi
if [[ "$ENABLE_ATTN_HEADS2" == "1" ]]; then
  export DS4_CUDA_ATTN_HEADS2=1
  export DS4_CUDA_ATTN_HEADS2_MIN_ROWS="$ATTN_HEADS2_MIN_ROWS"
else
  unset DS4_CUDA_ATTN_HEADS2
  unset DS4_CUDA_ATTN_HEADS2_MIN_ROWS
fi
if [[ "$TELEMETRY" == "1" ]]; then
  export DS4_CUDA_TOKEN_GRAPH_VERBOSE=1
  export DS4_CUDA_TOKEN_GRAPH_PIPELINE_VERBOSE=1
else
  unset DS4_CUDA_TOKEN_GRAPH_VERBOSE
  unset DS4_CUDA_TOKEN_GRAPH_PIPELINE_VERBOSE
  unset DS4_CUDA_TOKEN_GRAPH_TIMING
fi
unset DS4_CUDA_PARALLEL_FFN
unset DS4_CUDA_PARALLEL_FFN_VERBOSE
unset DS4_CUDA_WEIGHT_CACHE_VERBOSE

echo "Model:  $MODEL"
echo "Graph:  token=1 pipeline=$ENABLE_PIPELINE"
echo "Kernels: compressor=1 f16_coalesced=1 q8_u16=1"
echo "Attention: dynamic_heads2=$ENABLE_ATTN_HEADS2 min_rows=$ATTN_HEADS2_MIN_ROWS"
echo "Greedy: device_argmax=$ENABLE_GREEDY_ARGMAX (used only for globally greedy requests)"
echo "Server: http://0.0.0.0:30007"

exec ./ds4-server \
  --cuda \
  -m "$MODEL" \
  -c "$CTX" \
  -n "$MAX_TOKENS" \
  -t "$THREADS" \
  --host 0.0.0.0 \
  --port 30007 \
  --kv-disk-dir "$KV_DIR" \
  --kv-disk-space-mb 65536
