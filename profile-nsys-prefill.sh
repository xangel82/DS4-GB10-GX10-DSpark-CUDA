#!/usr/bin/env bash
# Reproducible multi-depth Nsight Systems capture for the GB10 prefill path.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

NSYS="${NSYS:-/usr/local/cuda/bin/nsys}"
MODEL="${DS4_MODEL:-/home/athena/ds4/ds4flash.gguf}"
DSPARK="${DS4_DSPARK_MODEL:-/home/athena/ds4/DeepSeek-V4-Flash-DSpark-Q4K-Q8.gguf}"
PROMPT="${DS4_NSYS_PREFILL_PROMPT:-speed-bench/promessi_sposi.txt}"
POSITIONS="${DS4_CUDA_NSYS_PREFILL_START_POSITIONS:-0,8192,32768,65536,98304}"
CHUNK="${DS4_PREFILL_CHUNK:-8192}"
CTX="${DS4_CTX:-131072}"
THREADS="${DS4_THREADS:-10}"
OUTPUT="${DS4_NSYS_PREFILL_OUTPUT:-/tmp/ds4-prefill-depth-$(date +%Y%m%d-%H%M%S)}"

IFS=',' read -r -a starts <<< "$POSITIONS"
if (( ${#starts[@]} == 0 )); then
  echo "No prefill capture positions configured" >&2
  exit 2
fi
previous=-1
for start in "${starts[@]}"; do
  if [[ ! "$start" =~ ^[0-9]+$ ]] || (( start <= previous )); then
    echo "Capture positions must be strictly increasing integers: $POSITIONS" >&2
    exit 2
  fi
  previous=$start
done
frontier=$((previous + CHUNK))
if (( frontier >= CTX )); then
  echo "Final capture frontier $frontier does not fit ctx=$CTX" >&2
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
export DS4_CUDA_COPY_SECONDARY_MODEL="${DS4_CUDA_COPY_SECONDARY_MODEL:-1}"
export DS4_CUDA_DSPARK_CACHE_PRIORITY=1
export DS4_CUDA_DEFER_END_SYNC=1
export DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS=0
export DS4_CUDA_FUSED_COMPRESSOR_UPDATE=1
export DS4_CUDA_TOKEN_GRAPH=1
export DS4_CUDA_DSPARK_GRAPH=1
export DS4_CUDA_COALESCED_F16_MATMUL=1
export DS4_CUDA_Q8_U16_LOADS=1
export DS4_CUDA_Q8_BATCH_REUSE=1
export DS4_CUDA_MOE_TINY_DIRECT=1
export DS4_CUDA_MOE_TINY_DIRECT_Q4_ONLY=1
export DS4_CUDA_DSPARK_TENSOR_CORES=1
export DS4_CUDA_DSPARK_TENSOR_CORES_Q8=1
export DS4_DSPARK_ALWAYS_DRAFT=1
export DS4_DSPARK_NO_CIRCUIT_BREAKER=1
export DS4_PREFILL_FINAL_LOGITS_ONLY=1
export DS4_CUDA_NSYS_PREFILL_START_POSITIONS="$POSITIONS"

echo "Nsight prefill windows: $POSITIONS (chunk=$CHUNK frontier=$frontier)"
echo "Reports: ${OUTPUT}*.nsys-rep"

"$NSYS" profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end="repeat-shutdown:${#starts[@]}" \
  --kill=none \
  --force-overwrite=true \
  -o "$OUTPUT" \
  ./ds4-bench \
    --cuda \
    --model "$MODEL" \
    --dspark "$DSPARK" \
    --dspark-draft 5 \
    --prompt-file "$PROMPT" \
    --frontiers "$frontier" \
    --ctx-alloc "$CTX" \
    --prefill-chunk "$CHUNK" \
    --gen-tokens 0 \
    --threads "$THREADS"

shopt -s nullglob
report_glob=("${OUTPUT}"*.nsys-rep)
if (( ${#report_glob[@]} == 0 )); then
  echo "Nsight produced no reports matching ${OUTPUT}*.nsys-rep" >&2
  exit 1
fi
mapfile -t reports < <(printf '%s\n' "${report_glob[@]}" | sort -V)
if (( ${#reports[@]} != ${#starts[@]} )); then
  echo "Expected ${#starts[@]} Nsight reports, found ${#reports[@]}" >&2
  exit 1
fi
for i in "${!reports[@]}"; do
  report="${reports[$i]}"
  stats="${report%.nsys-rep}.stats.txt"
  echo "Nsight prefill window $((i + 1))/${#reports[@]} pos=${starts[$i]} report=$report"
  "$NSYS" stats --force-export=true \
    --report nvtx_gpu_proj_sum,nvtx_kern_sum,cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
    "$report" | tee "$stats"
done
