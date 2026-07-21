#!/usr/bin/env bash
# Reproducible steady-state DSpark decode capture for the GB10.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

NSYS="${NSYS:-/usr/local/cuda/bin/nsys}"
MODEL="${DS4_MODEL:-/home/athena/ds4/ds4flash.gguf}"
DSPARK="${DS4_DSPARK_MODEL:-/home/athena/ds4/DeepSeek-V4-Flash-DSpark-Q4K-Q8.gguf}"
PROMPT="${DS4_NSYS_DECODE_PROMPT:-speed-bench/promessi_sposi.txt}"
FRONTIER="${DS4_NSYS_DECODE_FRONTIER:-65536}"
WARMUP="${DS4_NSYS_DECODE_WARMUP:-128}"
CAPTURE_TOKENS="${DS4_NSYS_DECODE_TOKENS:-64}"
TAIL_TOKENS="${DS4_NSYS_DECODE_TAIL:-16}"
DRAFT="${DS4_NSYS_DECODE_DRAFT:-5}"
FIXED_VERIFY="${DS4_NSYS_DECODE_FIXED_VERIFY:-0}"
GRAPHS="${DS4_NSYS_DECODE_GRAPHS:-1}"
CTX="${DS4_CTX:-131072}"
CHUNK="${DS4_PREFILL_CHUNK:-8192}"
THREADS="${DS4_THREADS:-10}"
OUTPUT="${DS4_NSYS_DECODE_OUTPUT:-/tmp/ds4-decode-dspark-$(date +%Y%m%d-%H%M%S)}"

for value in "$FRONTIER" "$WARMUP" "$CAPTURE_TOKENS" "$TAIL_TOKENS" "$DRAFT" "$FIXED_VERIFY" "$GRAPHS" "$CTX" "$CHUNK" "$THREADS"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "Decode profile values must be non-negative integers" >&2
    exit 2
  fi
done
if (( FRONTIER == 0 || CAPTURE_TOKENS == 0 || DRAFT == 0 || DRAFT > 5 ||
      FIXED_VERIFY > 1 || GRAPHS > 1 || CHUNK == 0 || THREADS == 0 )); then
  echo "Frontier, capture tokens, draft, chunk and threads must be valid; fixed-verify/graphs must be 0 or 1" >&2
  exit 2
fi
CAPTURE_START=$((FRONTIER + WARMUP))
GEN_TOKENS=$((WARMUP + CAPTURE_TOKENS + TAIL_TOKENS))
if (( FRONTIER + GEN_TOKENS >= CTX )); then
  echo "Decode profile does not fit ctx=$CTX" >&2
  exit 2
fi

for path in "$NSYS" "$MODEL" "$DSPARK" "$PROMPT" ./ds4-bench; do
  if [[ ! -e "$path" ]]; then
    echo "Required path not found: $path" >&2
    exit 2
  fi
done
if pgrep -x ds4-server >/dev/null 2>&1 || pgrep -x ds4-bench >/dev/null 2>&1; then
  echo "A ds4-server or ds4-bench process is already running; stop it before profiling" >&2
  exit 2
fi

export DS4_CUDA_COPY_MODEL=1
export DS4_CUDA_WEIGHT_CACHE_LIMIT_GB="${DS4_CUDA_WEIGHT_CACHE_LIMIT_GB:-112}"
export DS4_CUDA_DROP_COPIED_MODEL_PAGES=1
export DS4_CUDA_Q8_F16_CACHE_MB="${DS4_CUDA_Q8_F16_CACHE_MB:-12288}"
export DS4_CUDA_COPY_SECONDARY_MODEL=1
export DS4_CUDA_DSPARK_CACHE_PRIORITY=1
export DS4_CUDA_DEFER_END_SYNC=1
export DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS=0
export DS4_CUDA_FUSED_COMPRESSOR_UPDATE=1
export DS4_CUDA_TOKEN_GRAPH=1
if (( GRAPHS == 1 )); then
  export DS4_CUDA_DSPARK_GRAPH=1
else
  unset DS4_CUDA_DSPARK_GRAPH
fi
export DS4_CUDA_COALESCED_F16_MATMUL=1
export DS4_CUDA_Q8_U16_LOADS=1
export DS4_CUDA_Q8_BATCH_REUSE=1
export DS4_CUDA_MOE_TINY_DIRECT=1
export DS4_CUDA_MOE_TINY_DIRECT_Q4_ONLY=1
export DS4_CUDA_DSPARK_TENSOR_CORES=1
export DS4_CUDA_DSPARK_TENSOR_CORES_Q8=1
export DS4_CUDA_DSPARK_TC_PAD_N=8
export DS4_DSPARK_ALWAYS_DRAFT=1
export DS4_DSPARK_NO_CIRCUIT_BREAKER=1
if (( FIXED_VERIFY == 1 )); then
  export DS4_DSPARK_FIXED_VERIFY=1
else
  unset DS4_DSPARK_FIXED_VERIFY
fi
export DS4_PREFILL_FINAL_LOGITS_ONLY=1
export DS4_CUDA_NSYS_CAPTURE_START_POS="$CAPTURE_START"
export DS4_CUDA_NSYS_CAPTURE_TOKENS="$CAPTURE_TOKENS"
unset DS4_CUDA_TOKEN_GRAPH_PIPELINE
unset DS4_CUDA_MTP_GRAPH
unset DS4_CUDA_MTP_TENSOR_CORES
unset DS4_CUDA_TOKEN_GRAPH_TIMING
unset DS4_DSPARK_TIMING
unset DS4_DSPARK_LOG

echo "Nsight DSpark decode: frontier=$FRONTIER warmup=$WARMUP capture=$CAPTURE_TOKENS gen=$GEN_TOKENS draft=$DRAFT fixed-verify=$FIXED_VERIFY graphs=$GRAPHS"
echo "Report: ${OUTPUT}.nsys-rep"

"$NSYS" profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop-shutdown \
  --kill=none \
  --cuda-graph-trace=node \
  --force-overwrite=true \
  -o "$OUTPUT" \
  ./ds4-bench \
    --cuda \
    --model "$MODEL" \
    --dspark "$DSPARK" \
    --dspark-draft "$DRAFT" \
    --prompt-file "$PROMPT" \
    --frontiers "$FRONTIER" \
    --ctx-alloc "$CTX" \
    --prefill-chunk "$CHUNK" \
    --gen-tokens "$GEN_TOKENS" \
    --threads "$THREADS"

REPORT="${OUTPUT}.nsys-rep"
if [[ ! -f "$REPORT" ]]; then
  echo "Nsight produced no report at $REPORT" >&2
  exit 1
fi
STATS="${OUTPUT}.stats.txt"
"$NSYS" stats --force-export=true \
  --report nvtx_gpu_proj_sum,nvtx_kern_sum,cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
  "$REPORT" | tee "$STATS"

echo "Decode report: $REPORT"
echo "Decode stats:  $STATS"
