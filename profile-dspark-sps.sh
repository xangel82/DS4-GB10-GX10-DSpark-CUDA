#!/usr/bin/env bash
# GB10/GX10 DSpark CUDA modifications:
# Copyright (c) 2026 Marco Palaferri. Licensed under the MIT License.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${DS4_MODEL_DIR:-$HOME/ds4}"
MODEL="${DS4_MODEL:-$MODEL_DIR/ds4flash.gguf}"
VARIANT="${DS4_DSPARK_VARIANT:-q2}"
case "$VARIANT" in
  q2)
    DSPARK_DEFAULT="$MODEL_DIR/DeepSeek-V4-Flash-DSpark-IQ2XXS-Q2K-Q8.gguf"
    ;;
  q4)
    DSPARK_DEFAULT="$MODEL_DIR/DeepSeek-V4-Flash-DSpark-Q4K-Q8.gguf"
    ;;
  *)
    echo "Invalid DS4_DSPARK_VARIANT: $VARIANT (expected q2 or q4)" >&2
    exit 2
    ;;
esac
DSPARK="${DS4_DSPARK_MODEL:-$DSPARK_DEFAULT}"
PROMPT="${DS4_DSPARK_SPS_PROMPT:-$ROOT/speed-bench/promessi_sposi.txt}"
PREFIX="${DS4_DSPARK_SPS_PREFIX:-98304}"
PREFIXES="${DS4_DSPARK_SPS_PREFIXES:-$PREFIX}"
CTX="${DS4_CTX:-262144}"
RUNS="${DS4_DSPARK_SPS_RUNS:-10}"
MAX_R="${DS4_DSPARK_SPS_MAX_R:-3}"
MULTI_R_MAX_PREFIX="${DS4_DSPARK_SPS_MULTI_R_MAX_PREFIX:-524287}"
INCREMENTAL="${DS4_DSPARK_SPS_INCREMENTAL:-1}"
BASE_LOGS="${DS4_DSPARK_SPS_BASE_LOGS:-}"
MAX_CV_ERROR="${DS4_DSPARK_SPS_MAX_CV_ERROR:-0.05}"
OUTPUT="${DS4_DSPARK_SPS_PROFILE:-$MODEL_DIR/dspark-sps-${VARIANT}.conf}"
LOG_BASE="${DS4_DSPARK_SPS_LOG_BASE:-/tmp/ds4-sps-${VARIANT}}"

for file in "$MODEL" "$DSPARK" "$PROMPT"; do
  if [[ ! -f "$file" ]]; then
    echo "Required file not found: $file" >&2
    exit 2
  fi
done
if (( RUNS < 10 )); then
  echo "DS4_DSPARK_SPS_RUNS must be at least 10 (2 warm-up + 8 retained)" >&2
  exit 2
fi
if (( MAX_R < 1 || MAX_R > 4 )); then
  echo "DS4_DSPARK_SPS_MAX_R must be in 1..4" >&2
  exit 2
fi
if ! [[ "$MULTI_R_MAX_PREFIX" =~ ^[0-9]+$ ]]; then
    echo "Invalid DS4_DSPARK_SPS_MULTI_R_MAX_PREFIX: $MULTI_R_MAX_PREFIX" >&2
    exit 2
fi
if [[ "$INCREMENTAL" != "0" && "$INCREMENTAL" != "1" ]]; then
    echo "DS4_DSPARK_SPS_INCREMENTAL must be 0 or 1" >&2
    exit 2
fi

normalized_prefixes="${PREFIXES//,/ }"
normalized_prefixes="${normalized_prefixes//;/ }"
read -r -a prefix_list <<< "$normalized_prefixes"
if (( ${#prefix_list[@]} == 0 )); then
  echo "DS4_DSPARK_SPS_PREFIXES did not contain any prefix" >&2
  exit 2
fi

declare -A seen_buckets=()
logs=()
if [[ -n "$BASE_LOGS" ]]; then
  normalized_base_logs="${BASE_LOGS//,/ }"
  normalized_base_logs="${normalized_base_logs//;/ }"
  read -r -a base_log_list <<< "$normalized_base_logs"
  for log in "${base_log_list[@]}"; do
    if [[ ! -f "$log" ]]; then
      echo "Base SPS log not found: $log" >&2
      exit 2
    fi
    logs+=("$log")
  done
fi
new_logs=()
for prefix in "${prefix_list[@]}"; do
  if [[ ! "$prefix" =~ ^[0-9]+$ ]] ||
      (( prefix < 512 || CTX <= prefix + 128 )); then
    echo "Invalid SPS prefix '$prefix': need 512 <= prefix < DS4_CTX-128" >&2
    exit 2
  fi
  bucket=$((prefix / 32768))
  if (( bucket > 31 )); then
    bucket=31
  fi
  if [[ -n "${seen_buckets[$bucket]:-}" ]]; then
    echo "Duplicate context bucket $bucket in DS4_DSPARK_SPS_PREFIXES" >&2
    exit 2
  fi
  seen_buckets[$bucket]=1
  new_logs+=("${LOG_BASE}-b${bucket}.log")
done

echo "DSpark SPS calibration: variant=$VARIANT prefixes=${prefix_list[*]} ctx=$CTX runs=$RUNS max-r=$MAX_R multi-r-max-prefix=$MULTI_R_MAX_PREFIX incremental=$INCREMENTAL"
echo "Profile: $OUTPUT"

if [[ "$INCREMENTAL" == "1" ]]; then
  incremental_log="${LOG_BASE}-anchors.log"
  frontier_csv="$(IFS=,; echo "${prefix_list[*]}")"
  echo
  echo "Calibrating incremental frontiers=$frontier_csv telemetry=$incremental_log"
  DS4_BENCH_PHYSICAL_RN_SMOKE="$MAX_R" \
  DS4_BENCH_SPS_PROFILE=1 \
  DS4_BENCH_SPS_RUNS="$RUNS" \
  DS4_DSPARK_SPS_MULTI_R_MAX_PREFIX="$MULTI_R_MAX_PREFIX" \
  DS4_DSPARK_SPS_PROFILE="$OUTPUT" \
  "$ROOT/ds4-bench" \
    --cuda \
    --model "$MODEL" \
    --dspark "$DSPARK" \
    --dspark-draft 5 \
    --prompt-file "$PROMPT" \
    --frontiers "$frontier_csv" \
    --ctx-alloc "$CTX" \
    --gen-tokens 128 \
    2>&1 | tee "$incremental_log"
  logs+=("$incremental_log")
else
  for i in "${!prefix_list[@]}"; do
    prefix="${prefix_list[$i]}"
    bucket=$((prefix / 32768))
    if (( bucket > 31 )); then
      bucket=31
    fi
    bucket_max_r="$MAX_R"
    if (( prefix > MULTI_R_MAX_PREFIX )); then
      bucket_max_r=1
    fi
    log="${new_logs[$i]}"
    echo
    echo "Calibrating bucket=$bucket prefix=$prefix max-r=$bucket_max_r telemetry=$log"
    DS4_BENCH_PHYSICAL_RN_SMOKE="$bucket_max_r" \
    DS4_BENCH_PHYSICAL_RN_PREFIX="$prefix" \
    DS4_BENCH_SPS_PROFILE=1 \
    DS4_BENCH_SPS_RUNS="$RUNS" \
    DS4_DSPARK_SPS_PROFILE="$OUTPUT" \
    "$ROOT/ds4-bench" \
      --cuda \
      --model "$MODEL" \
      --dspark "$DSPARK" \
      --dspark-draft 5 \
      --prompt-file "$PROMPT" \
      --ctx-start "$prefix" \
      --ctx-max "$prefix" \
      --ctx-alloc "$CTX" \
      --gen-tokens 128 \
      2>&1 | tee "$log"
    logs+=("$log")
  done
fi

calibrate_args=()
for log in "${logs[@]}"; do
  calibrate_args+=(--log "$log")
done
python3 "$ROOT/tools/calibrate_dspark_sps.py" \
  "${calibrate_args[@]}" \
  --output "$OUTPUT" \
  --skip-first 0 \
  --interpolate-context \
  --max-context-cv-error "$MAX_CV_ERROR"
