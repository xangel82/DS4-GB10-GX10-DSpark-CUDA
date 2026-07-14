#!/usr/bin/env bash
# GB10/GX10 DSpark CUDA modifications:
# Copyright (c) 2026 Marco Palaferri. Licensed under the MIT License.
set -euo pipefail

MODEL="${DS4_MODEL:-/home/athena/ds4/ds4flash.gguf}"
DSPARK="${DS4_DSPARK_MODEL:-/home/athena/ds4/DeepSeek-V4-Flash-DSpark-Q4K-Q8.gguf}"
KV_DIR="${DS4_EXPERIMENT_KV_DIR:-/tmp/ds4-gb10-dspark-kv}"
CTX="${DS4_CTX:-131072}"
MAX_TOKENS="${DS4_MAX_TOKENS:-2200}"
THREADS="${DS4_THREADS:-10}"
PORT="${DS4_PORT:-30007}"
DRAFT="${DS4_DSPARK_DRAFT:-5}"
TELEMETRY="${DS4_TELEMETRY:-0}"

if [[ ! -f "$MODEL" ]]; then
  echo "Main model not found: $MODEL" >&2
  exit 2
fi
if [[ ! -f "$DSPARK" ]]; then
  echo "DSpark sidecar not found: $DSPARK" >&2
  echo "Build it with: ./build-dspark-sidecar.sh" >&2
  exit 2
fi
mkdir -p "$KV_DIR"

export DS4_CUDA_COPY_MODEL=1
export DS4_CUDA_WEIGHT_CACHE_LIMIT_GB="${DS4_CUDA_WEIGHT_CACHE_LIMIT_GB:-112}"
# The GB10 has enough unified-memory headroom for the measured 80.8 GiB target,
# 10.7 GiB sidecar, context buffers and a 12 GiB hot Q8->F16 cache.  Keeping the
# hot projections resident avoids the 6 GiB cache ceiling seen in early runs.
export DS4_CUDA_Q8_F16_CACHE_MB="${DS4_CUDA_Q8_F16_CACHE_MB:-12288}"
export DS4_CUDA_DSPARK_CACHE_PRIORITY=1
export DS4_CUDA_DEFER_END_SYNC=1
export DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS=0
export DS4_CUDA_FUSED_COMPRESSOR_UPDATE=1
export DS4_CUDA_TOKEN_GRAPH=1
export DS4_CUDA_DSPARK_GRAPH=1
export DS4_CUDA_COALESCED_F16_MATMUL=1
export DS4_CUDA_Q8_U16_LOADS=1
if [[ "${DS4_CUDA_DSPARK_TENSOR_CORES:-1}" == "1" ]]; then
  export DS4_CUDA_DSPARK_TENSOR_CORES=1
  export DS4_CUDA_DSPARK_TC_PAD_N="${DS4_CUDA_DSPARK_TC_PAD_N:-8}"
else
  unset DS4_CUDA_DSPARK_TENSOR_CORES
fi
if [[ "${DS4_CUDA_DSPARK_TENSOR_CORES_Q8:-1}" == "1" ]]; then
  export DS4_CUDA_DSPARK_TENSOR_CORES_Q8=1
else
  unset DS4_CUDA_DSPARK_TENSOR_CORES_Q8
fi
# Athena serves one active decode stream, so never stop collecting DSpark
# confidence/K telemetry because of a stale historical estimate.  K remains
# adaptive after the draft; only the pre-draft performance bypass is removed.
if [[ "${DS4_DSPARK_ALWAYS_DRAFT:-1}" == "1" ]]; then
  export DS4_DSPARK_ALWAYS_DRAFT=1
else
  unset DS4_DSPARK_ALWAYS_DRAFT
fi
if [[ "${DS4_DSPARK_CIRCUIT_BREAKER:-0}" == "1" ]]; then
  unset DS4_DSPARK_NO_CIRCUIT_BREAKER
else
  export DS4_DSPARK_NO_CIRCUIT_BREAKER=1
fi
# K3/K4 verifier batches are bandwidth-bound in uncached Q8 projections.
# Compute all tiny-batch rows in one block so each packed weight row is read
# once.  Keep the direct-MoE shortcut on the Q4 DSpark sidecar; the Q2 target
# verifier benefits from its expert-grouped path at four rows.
if [[ "${DS4_CUDA_Q8_BATCH_REUSE:-1}" == "1" ]]; then
  export DS4_CUDA_Q8_BATCH_REUSE=1
else
  unset DS4_CUDA_Q8_BATCH_REUSE
fi
if [[ "${DS4_CUDA_MOE_TINY_DIRECT:-1}" == "1" ]]; then
  export DS4_CUDA_MOE_TINY_DIRECT=1
else
  unset DS4_CUDA_MOE_TINY_DIRECT
fi
if [[ "${DS4_CUDA_MOE_TINY_DIRECT_Q4_ONLY:-1}" == "1" ]]; then
  export DS4_CUDA_MOE_TINY_DIRECT_Q4_ONLY=1
else
  unset DS4_CUDA_MOE_TINY_DIRECT_Q4_ONLY
fi

# The look-ahead host capture is intentionally not used with DSpark yet.  The
# normal token graphs remain available for fallback; separate K-aware families
# handle the drafter and the fused [current + K draft] target verifier.
unset DS4_CUDA_TOKEN_GRAPH_PIPELINE
unset DS4_CUDA_MTP_GRAPH
unset DS4_CUDA_MTP_TENSOR_CORES

if [[ "$TELEMETRY" == "1" ]]; then
  export DS4_DSPARK_TIMING=1
  export DS4_DSPARK_LOG=1
  export DS4_CUDA_TOKEN_GRAPH_VERBOSE=1
  export DS4_CUDA_DSPARK_GRAPH_VERBOSE=1
else
  unset DS4_DSPARK_TIMING
  unset DS4_DSPARK_LOG
  unset DS4_CUDA_TOKEN_GRAPH_VERBOSE
  unset DS4_CUDA_DSPARK_GRAPH_VERBOSE
fi

echo "Target: $MODEL"
echo "DSpark: $DSPARK (draft=$DRAFT)"
echo "Cache:  Q8->F16=${DS4_CUDA_Q8_F16_CACHE_MB} MiB, weight limit=${DS4_CUDA_WEIGHT_CACHE_LIMIT_GB} GiB"
echo "DSpark scheduler: full 5-slot draft, adaptive verifier K=0..$DRAFT, always-draft=${DS4_DSPARK_ALWAYS_DRAFT:-0}, circuit-breaker=${DS4_DSPARK_CIRCUIT_BREAKER:-0}, fused K+1 verifier, graphs=on, telemetry=$TELEMETRY"
echo "DSpark sampling: lossless p/q rejection for top_k=0 top_p=1 min-p policy (rollback DS4_DSPARK_REJECTION_DISABLE=1)"
echo "GB10 verifier: Q8 batch-reuse=${DS4_CUDA_Q8_BATCH_REUSE:-0}, Q4-sidecar direct-MoE=${DS4_CUDA_MOE_TINY_DIRECT_Q4_ONLY:-0}, tiny-TC=${DS4_CUDA_DSPARK_TENSOR_CORES:-0}, tiny-TC-Q8=${DS4_CUDA_DSPARK_TENSOR_CORES_Q8:-0}"
echo "Server: http://0.0.0.0:$PORT"

exec ./ds4-server \
  --cuda \
  -m "$MODEL" \
  --dspark "$DSPARK" \
  --dspark-draft "$DRAFT" \
  -c "$CTX" \
  -n "$MAX_TOKENS" \
  -t "$THREADS" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --kv-disk-dir "$KV_DIR" \
  --kv-disk-space-mb 65536
