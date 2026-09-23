#!/usr/bin/env bash
# GB10/GX10 DSpark CUDA modifications:
# Copyright (c) 2026 Marco Palaferri. Licensed under the MIT License.
#
# A/B for the EXPERIMENTAL ratio-128 sparse undo journal.
#
#   A (baseline) : DS4_DSPARK_R128_UNDO_JOURNAL unset -> legacy full-frontier
#                  snapshot + K prefix captures
#   D (journal)  : DS4_DSPARK_R128_UNDO_JOURNAL=1     -> sparse per-row journal
#
# Everything else is held identical and mirrors the productive decode path
# selected by run-dspark-server.sh, including the presence-based variables
# (`unset` and `=0` are NOT interchangeable for those).  The ONLY difference
# between the two modes is the journal flag.
#
# Run from the ds4 checkout on Athena.  Never touches an existing service.
set -euo pipefail

MODE="${1:-}"
case "$MODE" in
  baseline|journal) ;;
  *) echo "usage: $0 {baseline|journal}" >&2; exit 2 ;;
esac

MODEL_DIR="${DS4_MODEL_DIR:-$HOME/ds4}"
TARGET_0731_FILE="DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf"
MODEL="${DS4_MODEL:-$MODEL_DIR/$TARGET_0731_FILE}"
DSPARK="${DS4_DSPARK_MODEL:-$MODEL_DIR/DeepSeek-V4-Flash-DSpark-IQ2XXS-Q2K-Q8.gguf}"
DRAFT="${DS4_DSPARK_DRAFT:-5}"
PROMPT_FILE="${DS4_PROMPT_FILE:-speed-bench/promessi_sposi.txt}"
CTX_START="${DS4_CTX_START:-4096}"
CTX_MAX="${DS4_CTX_MAX:-16384}"
STEP_INCR="${DS4_STEP_INCR:-4096}"
GEN_TOKENS="${DS4_GEN_TOKENS:-256}"
THREADS="${DS4_THREADS:-10}"
CSV_OUT="${DS4_CSV_OUT:-}"

for f in "$MODEL" "$DSPARK" "$PROMPT_FILE"; do
  [[ -f "$f" ]] || { echo "missing: $f" >&2; exit 2; }
done

# ---------------------------------------------------------------------------
# Productive DSpark environment, copied from run-dspark-server.sh.  Held
# identical across both modes.
# ---------------------------------------------------------------------------
export DS4_CUDA_COPY_MODEL=1
export DS4_CUDA_SECONDARY_COPY_PIPELINED=1
export DS4_CUDA_WEIGHT_CACHE_LIMIT_GB="${DS4_CUDA_WEIGHT_CACHE_LIMIT_GB:-112}"
export DS4_CUDA_DROP_COPIED_MODEL_PAGES=1
unset  DS4_CUDA_KEEP_MODEL_PAGES
export DS4_CUDA_Q8_F16_CACHE_MB="${DS4_CUDA_Q8_F16_CACHE_MB:-12288}"
export DS4_CUDA_COPY_SECONDARY_MODEL=1
unset  DS4_CUDA_DSPARK_CACHE_COMPACT
export DS4_CUDA_DSPARK_CACHE_PRIORITY=1
export DS4_CUDA_DEFER_END_SYNC=1
export DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS=0
export DS4_CUDA_FUSED_COMPRESSOR_UPDATE=1
export DS4_CUDA_TOKEN_GRAPH=1
export DS4_CUDA_DSPARK_GRAPH=1
export DS4_CUDA_COALESCED_F16_MATMUL=1
export DS4_CUDA_Q8_U16_LOADS=1
export DS4_CUDA_DSPARK_TENSOR_CORES=1
export DS4_CUDA_DSPARK_TC_PAD_N=8
export DS4_CUDA_DSPARK_TENSOR_CORES_Q8=1
export DS4_CUDA_Q8_BATCH_REUSE=1
export DS4_CUDA_MOE_TINY_DIRECT=1
export DS4_CUDA_MOE_TINY_DIRECT_Q4_ONLY=1
export DS4_DSPARK_ALWAYS_DRAFT=1
export DS4_DSPARK_NO_CIRCUIT_BREAKER=1
export DS4_DSPARK_HYBRID_LC=1
export DS4_DSPARK_NIGHTJAR=1
export DS4_DSPARK_NIGHTJAR_MAX_R="${DS4_DSPARK_NIGHTJAR_MAX_R:-3}"
export DS4_DSPARK_NIGHTJAR_MAX_PREDICTED_REGRESSION=0.03
export DS4_DSPARK_RN_MIN_PHYSICAL_ROWS=0
export DS4_DSPARK_HW_PROFILE="${DS4_DSPARK_HW_PROFILE:-$MODEL_DIR/dspark-hardware-q2.profile}"
export DS4_DSPARK_SPS_PROFILE="${DS4_DSPARK_SPS_PROFILE:-$MODEL_DIR/dspark-sps-q2.conf}"
export DS4_DSPARK_SPS_FORCE_PROFILE=0
export DS4_CUDA_MTP_GRAPH_CACHE_POLICY=slru
export DS4_CUDA_MTP_GRAPH_MAX_LIVE=24
export DS4_CUDA_MTP_GRAPH_SMALL_RESERVE=8
# Presence-tested rollbacks that the launcher leaves unset.
unset DS4_DSPARK_CONFIDENCE_POST_NORM
unset DS4_CUDA_DSPARK_GRAPH_TOPOLOGY_CACHE_DISABLE
unset DS4_CUDA_TOKEN_GRAPH_PIPELINE
unset DS4_CUDA_MTP_GRAPH
unset DS4_CUDA_MTP_TENSOR_CORES
unset DS4_DSPARK_HYBRID_LC_SHADOW
unset DS4_DSPARK_SCHEDULER_SHADOW
unset DS4_DSPARK_SCHEDULER_DETERMINISTIC
unset DS4_DSPARK_NIGHTJAR_SHADOW
unset DS4_DSPARK_TIMING
unset DS4_DSPARK_LOG

# ---------------------------------------------------------------------------
# The one and only difference between the two arms.
# ---------------------------------------------------------------------------
if [[ "$MODE" == "journal" ]]; then
  export DS4_DSPARK_R128_UNDO_JOURNAL=1
else
  unset DS4_DSPARK_R128_UNDO_JOURNAL
fi
# Instrumentation is OFF in the timed arms so it cannot perturb them.  Opt in
# with DS4_AB_STATS=1 for a separate, non-timed accounting run.
if [[ "${DS4_AB_STATS:-0}" == "1" ]]; then
  export DS4_DSPARK_R128_JOURNAL_STATS=1
else
  unset DS4_DSPARK_R128_JOURNAL_STATS
fi

echo "=== ratio-128 A/B arm: $MODE ==="
echo "target : $MODEL"
echo "dspark : $DSPARK (draft=$DRAFT)"
echo "journal: ${DS4_DSPARK_R128_UNDO_JOURNAL:-unset}"
echo "sweep  : ctx $CTX_START..$CTX_MAX step $STEP_INCR, gen $GEN_TOKENS"

args=( -m "$MODEL"
       --dspark "$DSPARK"
       --dspark-draft "$DRAFT"
       --prompt-file "$PROMPT_FILE"
       --ctx-start "$CTX_START"
       --ctx-max "$CTX_MAX"
       --step-incr "$STEP_INCR"
       --gen-tokens "$GEN_TOKENS"
       -t "$THREADS" )
[[ -n "$CSV_OUT" ]] && args+=( --csv "$CSV_OUT" )

exec ./ds4-bench "${args[@]}"
