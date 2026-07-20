#!/usr/bin/env bash
# Profile one long-context prefill attention launch with Nsight Compute.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

NCU="${NCU:-/usr/local/cuda/bin/ncu}"
MODEL="${DS4_MODEL:-/home/athena/ds4/ds4flash.gguf}"
DSPARK="${DS4_DSPARK_MODEL:-/home/athena/ds4/DeepSeek-V4-Flash-DSpark-Q4K-Q8.gguf}"
PROMPT="${DS4_NCU_PREFILL_PROMPT:-speed-bench/promessi_sposi.txt}"
START="${DS4_NCU_PREFILL_START_POS:-98304}"
CHUNK="${DS4_NCU_PREFILL_CHUNK:-1024}"
CTX="${DS4_CTX:-131072}"
THREADS="${DS4_THREADS:-10}"
RANGE="${DS4_NCU_PREFILL_RANGE:-indexed}"
MODE="${DS4_NCU_PREFILL_MODE:-occupancy}"
OUTPUT="${DS4_NCU_PREFILL_OUTPUT:-/tmp/ds4-ncu-prefill-${RANGE}-${MODE}-$(date +%Y%m%d-%H%M%S)}"

case "$RANGE" in
  indexed|dense)
    NVTX_INCLUDE="regex:ds4\\/prefill\\/attention\\/token_tile\\/${RANGE}/"
    KERNEL_NAME="regex:attention_tokentile_hmma_kernel"
    ;;
  scorer)
    NVTX_INCLUDE="regex:ds4\\/prefill\\/indexer\\/score-mxfp4/"
    KERNEL_NAME="regex:indexer_scores_mxfp4_kernel"
    ;;
  *)
    echo "DS4_NCU_PREFILL_RANGE must be indexed, dense or scorer" >&2
    exit 2
    ;;
esac
case "$MODE" in
  occupancy|runtime) ;;
  *)
    echo "DS4_NCU_PREFILL_MODE must be occupancy or runtime" >&2
    exit 2
    ;;
esac
if [[ ! "$START" =~ ^[0-9]+$ || ! "$CHUNK" =~ ^[0-9]+$ ]] ||
   (( CHUNK < 128 || START + CHUNK >= CTX )); then
  echo "Invalid prefill capture shape: start=$START chunk=$CHUNK ctx=$CTX" >&2
  exit 2
fi
for path in "$NCU" "$MODEL" "$DSPARK" "$PROMPT" ./ds4-bench; do
  if [[ ! -e "$path" ]]; then
    echo "Required path not found: $path" >&2
    exit 2
  fi
done
if pgrep -x ds4-server >/dev/null 2>&1 || pgrep -x ds4-bench >/dev/null 2>&1; then
  echo "A ds4-server or ds4-bench process is already running; stop it before profiling" >&2
  exit 2
fi
if (( EUID != 0 )); then
  echo "Warning: GB10 performance counters normally require sudo; use: sudo -E ./profile-ncu-prefill.sh" >&2
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
export DS4_CUDA_NSYS_PREFILL_START_POS="$START"
export DS4_LOCK_FILE="${DS4_LOCK_FILE:-/tmp/ds4-ncu-root.lock}"

echo "Nsight Compute prefill: range=$RANGE mode=$MODE start=$START chunk=$CHUNK"
echo "Report: ${OUTPUT}.ncu-rep"

ncu_args=(
  --target-processes all
  --profile-from-start off
  --nvtx
  --nvtx-include "$NVTX_INCLUDE"
  --kernel-name "$KERNEL_NAME"
  --launch-count 1
  --cache-control none
  --clock-control none
  --force-overwrite
  -o "$OUTPUT"
)

if [[ "$MODE" == occupancy ]]; then
  # Even static sections enter NCU's replay machinery. Application replay
  # avoids snapshotting the nearly full unified-memory address space.
  ncu_args+=(
    --replay-mode application
    --app-replay-buffer file
    --app-replay-mode relaxed
    --app-replay-match grid
    --section LaunchStats
    --section Occupancy
  )
else
  # Kernel replay snapshots all accessible allocations. The resident GB10
  # model leaves too little memory for that snapshot, so replay the process.
  ncu_args+=(
    --replay-mode application
    --app-replay-buffer file
    --app-replay-mode relaxed
    --app-replay-match grid
    --section SpeedOfLight
    --section MemoryWorkloadAnalysis
    --section WarpStateStats
  )
fi

"$NCU" "${ncu_args[@]}" \
  ./ds4-bench \
    --cuda \
    --model "$MODEL" \
    --dspark "$DSPARK" \
    --dspark-draft 5 \
    --prompt-file "$PROMPT" \
    --frontiers "$((START + CHUNK))" \
    --ctx-alloc "$CTX" \
    --prefill-chunk "$CHUNK" \
    --gen-tokens 0 \
    --threads "$THREADS"

if [[ ! -f "${OUTPUT}.ncu-rep" ]]; then
  echo "Nsight Compute produced no report" >&2
  exit 1
fi
"$NCU" --import "${OUTPUT}.ncu-rep" --page details > "${OUTPUT}.txt"
echo "Text report: ${OUTPUT}.txt"
