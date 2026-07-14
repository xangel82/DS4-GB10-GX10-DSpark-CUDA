#!/usr/bin/env bash
# GB10/GX10 DSpark CUDA modifications:
# Copyright (c) 2026 Marco Palaferri. Licensed under the MIT License.
set -euo pipefail

# Experimental DeepSeek V4 Flash server for Athena/GB10.
# It keeps both the target and MTP GGUF resident, enables the tiny-batch
# Tensor Core path, and leaves every optimization opt-in to this process.

MODEL="${DS4_MODEL:-/home/athena/ds4/ds4flash.gguf}"
MTP_NAME="DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
MTP_MODEL="${DS4_MTP_MODEL:-}"
if [[ -z "$MTP_MODEL" ]]; then
  for candidate in \
    "/home/athena/ds4/gguf/$MTP_NAME" \
    "/home/athena/ds4/$MTP_NAME" \
    "./gguf/$MTP_NAME"; do
    if [[ -f "$candidate" ]]; then
      MTP_MODEL="$candidate"
      break
    fi
  done
fi

if [[ ! -f "$MODEL" ]]; then
  echo "Main model not found: $MODEL" >&2
  exit 2
fi
if [[ -z "$MTP_MODEL" || ! -f "$MTP_MODEL" ]]; then
  echo "MTP model not found." >&2
  echo "Download it with: ./download_model.sh mtp" >&2
  echo "Or set DS4_MTP_MODEL=/absolute/path/$MTP_NAME" >&2
  exit 2
fi

KV_DIR="${DS4_EXPERIMENT_KV_DIR:-/tmp/ds4-gb10-mtp-tc-kv}"
CTX="${DS4_CTX:-131072}"
MAX_TOKENS="${DS4_MAX_TOKENS:-2200}"
THREADS="${DS4_THREADS:-10}"
PORT="${DS4_PORT:-30007}"
MTP_DRAFT="${DS4_MTP_DRAFT:-2}"
MTP_MARGIN="${DS4_MTP_MARGIN:-3}"
TELEMETRY="${DS4_TELEMETRY:-1}"

mkdir -p "$KV_DIR"

export DS4_CUDA_COPY_MODEL=1
export DS4_CUDA_WEIGHT_CACHE_LIMIT_GB="${DS4_CUDA_WEIGHT_CACHE_LIMIT_GB:-96}"
export DS4_CUDA_Q8_F16_CACHE_MB="${DS4_CUDA_Q8_F16_CACHE_MB:-12288}"
export DS4_CUDA_DEFER_END_SYNC=1
export DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS=0
export DS4_CUDA_FUSED_COMPRESSOR_UPDATE=1
export DS4_CUDA_TOKEN_GRAPH=1
if [[ "$TELEMETRY" == "1" && "${DS4_CUDA_TOKEN_GRAPH_VERBOSE:-1}" == "1" ]]; then
  export DS4_CUDA_TOKEN_GRAPH_VERBOSE=1
else
  unset DS4_CUDA_TOKEN_GRAPH_VERBOSE
fi
export DS4_CUDA_COALESCED_F16_MATMUL=1
export DS4_CUDA_Q8_U16_LOADS=1

if [[ "${DS4_ENABLE_MTP_GRAPH:-1}" == "1" ]]; then
  export DS4_CUDA_MTP_GRAPH=1
  if [[ "$TELEMETRY" == "1" && "${DS4_CUDA_MTP_GRAPH_VERBOSE:-1}" == "1" ]]; then
    export DS4_CUDA_MTP_GRAPH_VERBOSE=1
  else
    unset DS4_CUDA_MTP_GRAPH_VERBOSE
  fi
else
  unset DS4_CUDA_MTP_GRAPH
  unset DS4_CUDA_MTP_GRAPH_VERBOSE
fi

if [[ "${DS4_ENABLE_MTP_TC:-1}" == "1" ]]; then
  export DS4_CUDA_MTP_TENSOR_CORES=1
  export DS4_CUDA_MTP_TC_PAD_N="${DS4_CUDA_MTP_TC_PAD_N:-8}"
  export DS4_CUDA_MTP_TC_WORKSPACE_MB="${DS4_CUDA_MTP_TC_WORKSPACE_MB:-64}"
  if [[ "$TELEMETRY" == "1" && "${DS4_CUDA_MTP_TC_VERBOSE:-1}" == "1" ]]; then
    export DS4_CUDA_MTP_TC_VERBOSE=1
  else
    unset DS4_CUDA_MTP_TC_VERBOSE
  fi
  if [[ "${DS4_CUDA_MTP_TC_AUTOTUNE:-1}" == "1" ]]; then
    export DS4_CUDA_MTP_TC_AUTOTUNE=1
  else
    unset DS4_CUDA_MTP_TC_AUTOTUNE
  fi
else
  unset DS4_CUDA_MTP_TENSOR_CORES
  unset DS4_CUDA_MTP_TC_PAD_N
  unset DS4_CUDA_MTP_TC_WORKSPACE_MB
  unset DS4_CUDA_MTP_TC_VERBOSE
  unset DS4_CUDA_MTP_TC_AUTOTUNE
fi

# DS4's current speculative verifier is greedy-only. Thinking mode otherwise
# forces the default non-zero temperature and silently bypasses MTP.
export DS4_MTP_GREEDY_THINK=1
if [[ "$TELEMETRY" == "1" && "${DS4_MTP_LOG_TIMING:-1}" == "1" ]]; then
  export DS4_MTP_TIMING=1
  export DS4_MTP_SPEC_LOG=1
else
  unset DS4_MTP_TIMING
  unset DS4_MTP_SPEC_LOG
fi

# Explicitly reject experiments that were neutral or slower on GB10.
unset DS4_CUDA_MOE_DECODE_GB10
unset DS4_CUDA_PARALLEL_FFN
unset DS4_CUDA_PARALLEL_FFN_VERBOSE
unset DS4_CUDA_WEIGHT_CACHE_VERBOSE

echo "Target: $MODEL"
echo "MTP:    $MTP_MODEL (draft=$MTP_DRAFT margin=$MTP_MARGIN)"
echo "TC:     ${DS4_ENABLE_MTP_TC:-1} (pad_n=${DS4_CUDA_MTP_TC_PAD_N:-off} autotune=${DS4_CUDA_MTP_TC_AUTOTUNE:-off})"
echo "Graph:  ${DS4_ENABLE_MTP_GRAPH:-1} (dedicated MTP draft + verifier graphs)"
echo "Logs:   telemetry=$TELEMETRY"
echo "Server: http://0.0.0.0:$PORT"

exec ./ds4-server \
  --cuda \
  -m "$MODEL" \
  --mtp "$MTP_MODEL" \
  --mtp-draft "$MTP_DRAFT" \
  --mtp-margin "$MTP_MARGIN" \
  -c "$CTX" \
  -n "$MAX_TOKENS" \
  -t "$THREADS" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --kv-disk-dir "$KV_DIR" \
  --kv-disk-space-mb 65536
