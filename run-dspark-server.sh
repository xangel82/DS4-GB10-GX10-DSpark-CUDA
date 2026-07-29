#!/usr/bin/env bash
# GB10/GX10 DSpark CUDA modifications:
# Copyright (c) 2026 Marco Palaferri. Licensed under the MIT License.
set -euo pipefail

MODEL_DIR="${DS4_MODEL_DIR:-$HOME/ds4}"
MODEL="${DS4_MODEL:-$MODEL_DIR/ds4flash.gguf}"
DSPARK_VARIANT="${DS4_DSPARK_VARIANT:-q2}"
case "$DSPARK_VARIANT" in
  q4)
    DSPARK_DEFAULT="$MODEL_DIR/DeepSeek-V4-Flash-DSpark-Q4K-Q8.gguf"
    ;;
  q2)
    DSPARK_DEFAULT="$MODEL_DIR/DeepSeek-V4-Flash-DSpark-IQ2XXS-Q2K-Q8.gguf"
    ;;
  *)
    echo "Invalid DS4_DSPARK_VARIANT: $DSPARK_VARIANT (expected q4 or q2)" >&2
    exit 2
    ;;
esac
if [[ "${DS4_DSPARK_MODEL:-}" != "" ]]; then
  DSPARK="$DS4_DSPARK_MODEL"
  DSPARK_SELECTION="custom"
else
  DSPARK="$DSPARK_DEFAULT"
  DSPARK_SELECTION="$DSPARK_VARIANT"
fi
KV_DIR="${DS4_EXPERIMENT_KV_DIR:-/tmp/ds4-gb10-dspark-kv}"
CTX="${DS4_CTX:-262144}"
MAX_TOKENS="${DS4_MAX_TOKENS:-2200}"
THREADS="${DS4_THREADS:-10}"
PORT="${DS4_PORT:-30007}"
KV_DISK_SPACE_MB="${DS4_KV_DISK_SPACE_MB:-16384}"
DRAFT="${DS4_DSPARK_DRAFT:-5}"
TELEMETRY="${DS4_TELEMETRY:-0}"
SCHEDULER_SHADOW="${DS4_DSPARK_SCHEDULER_SHADOW:-$TELEMETRY}"
SCHEDULER_DETERMINISTIC="${DS4_DSPARK_SCHEDULER_DETERMINISTIC:-0}"
COORDINATOR_LANES="${DS4_SERVER_DSPARK_LANES:-3}"
COORDINATOR_HOT_LANES="${DS4_SERVER_DSPARK_HOT_LANES:-2}"
COORDINATOR_LANE_RESERVE_MB="${DS4_SERVER_DSPARK_LANE_RESERVE_MB:-1536}"
COORDINATOR_COALESCE_US="${DS4_SERVER_DSPARK_COALESCE_US:-500}"
COORDINATOR_ACTIVE_COALESCE_US="${DS4_SERVER_DSPARK_ACTIVE_COALESCE_US:-20000}"
COORDINATOR_MIN_PHYSICAL_ROWS="${DS4_DSPARK_RN_MIN_PHYSICAL_ROWS:-10}"
if [[ -n "${DS4_DSPARK_STS_PROFILE+x}" ]]; then
  STS_PROFILE="$DS4_DSPARK_STS_PROFILE"
else
  STS_PROFILE_DEFAULT="$MODEL_DIR/dspark-sts-${DSPARK_VARIANT}.conf"
  STS_PROFILE_REPO="$(cd "$(dirname "$0")" && pwd)/profiles/dspark-sts-${DSPARK_VARIANT}.conf"
  STS_PROFILE=""
  if [[ -f "$STS_PROFILE_DEFAULT" ]]; then
    STS_PROFILE="$STS_PROFILE_DEFAULT"
  elif [[ -f "$STS_PROFILE_REPO" ]]; then
    STS_PROFILE="$STS_PROFILE_REPO"
  fi
fi
STREAM_HEARTBEAT_SEC="${DS4_STREAM_HEARTBEAT_SEC:-140}"
PREFILL_POLICY="${DS4_KV_PREFILL_CHECKPOINT_POLICY:-canonical-only}"
MEMORY_PROFILE="${DS4_MEMORY_PROFILE:-balanced}"
PREFILL_CHUNK="${DS4_PREFILL_CHUNK:-}"
KV_COLD_MAX_TOKENS="${DS4_KV_CACHE_COLD_MAX_TOKENS:-$CTX}"
ADVERTISE_CONTEXT_PCT="${DS4_ADVERTISE_CONTEXT_PCT:-85}"
LONG_ANCHOR_MIN_TOKENS="${DS4_KV_LONG_COLD_ANCHOR_MIN_TOKENS:-$((CTX / 2))}"
LONG_ANCHOR_TRIM_TOKENS="${DS4_KV_LONG_COLD_ANCHOR_TRIM_TOKENS:-$((CTX / 16))}"
CONFIDENCE_POST_NORM="${DS4_DSPARK_CONFIDENCE_POST_NORM:-0}"
GRAPH_TOPOLOGY_CACHE_DISABLE="${DS4_CUDA_DSPARK_GRAPH_TOPOLOGY_CACHE_DISABLE:-0}"
SECONDARY_COPY_PIPELINED="${DS4_CUDA_SECONDARY_COPY_PIPELINED:-1}"
HYBRID_LC_SHADOW="${DS4_DSPARK_HYBRID_LC_SHADOW:-0}"
if [[ -n "${DS4_DSPARK_HYBRID_LC+x}" ]]; then
  HYBRID_LC="$DS4_DSPARK_HYBRID_LC"
elif [[ "$HYBRID_LC_SHADOW" == "1" ]]; then
  HYBRID_LC=0
else
  HYBRID_LC=1
fi
HYBRID_WIDTH="${DS4_DSPARK_HYBRID_WIDTH:-}"

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

case "$PREFILL_POLICY" in
  canonical-only|coalesced|legacy) ;;
  *) echo "Invalid DS4_KV_PREFILL_CHECKPOINT_POLICY: $PREFILL_POLICY" >&2; exit 2 ;;
esac
case "$CONFIDENCE_POST_NORM" in
  0|1) ;;
  *) echo "Invalid DS4_DSPARK_CONFIDENCE_POST_NORM: $CONFIDENCE_POST_NORM (expected 0 or 1)" >&2; exit 2 ;;
esac
case "$GRAPH_TOPOLOGY_CACHE_DISABLE" in
  0|1) ;;
  *) echo "Invalid DS4_CUDA_DSPARK_GRAPH_TOPOLOGY_CACHE_DISABLE: $GRAPH_TOPOLOGY_CACHE_DISABLE (expected 0 or 1)" >&2; exit 2 ;;
esac
case "$SECONDARY_COPY_PIPELINED" in
  0|1) ;;
  *) echo "Invalid DS4_CUDA_SECONDARY_COPY_PIPELINED: $SECONDARY_COPY_PIPELINED (expected 0 or 1)" >&2; exit 2 ;;
esac
case "$SCHEDULER_SHADOW" in
  0|1) ;;
  *) echo "Invalid DS4_DSPARK_SCHEDULER_SHADOW: $SCHEDULER_SHADOW (expected 0 or 1)" >&2; exit 2 ;;
esac
case "$SCHEDULER_DETERMINISTIC" in
  0|1) ;;
  *) echo "Invalid DS4_DSPARK_SCHEDULER_DETERMINISTIC: $SCHEDULER_DETERMINISTIC (expected 0 or 1)" >&2; exit 2 ;;
esac
case "$COORDINATOR_LANES" in
  1|2|3) ;;
  *) echo "Invalid DS4_SERVER_DSPARK_LANES: $COORDINATOR_LANES (expected 1, 2 or 3)" >&2; exit 2 ;;
esac
case "$COORDINATOR_HOT_LANES" in
  1|2|3) ;;
  *) echo "Invalid DS4_SERVER_DSPARK_HOT_LANES: $COORDINATOR_HOT_LANES (expected 1, 2 or 3)" >&2; exit 2 ;;
esac
if (( COORDINATOR_HOT_LANES > COORDINATOR_LANES )); then
  echo "Invalid DS4_SERVER_DSPARK_HOT_LANES: $COORDINATOR_HOT_LANES exceeds max lanes $COORDINATOR_LANES" >&2
  exit 2
fi
if (( COORDINATOR_LANES > 1 && COORDINATOR_HOT_LANES < 2 )); then
  echo "Invalid DS4_SERVER_DSPARK_HOT_LANES: elastic lanes require two pristine hot lanes" >&2
  exit 2
fi
if ! [[ "$COORDINATOR_LANE_RESERVE_MB" =~ ^[0-9]+$ ]]; then
  echo "Invalid DS4_SERVER_DSPARK_LANE_RESERVE_MB: $COORDINATOR_LANE_RESERVE_MB" >&2
  exit 2
fi
if [[ ! "$COORDINATOR_COALESCE_US" =~ ^[0-9]+$ ]] ||
   (( COORDINATOR_COALESCE_US > 10000 )); then
  echo "Invalid DS4_SERVER_DSPARK_COALESCE_US: $COORDINATOR_COALESCE_US (expected 0..10000)" >&2
  exit 2
fi
if [[ ! "$COORDINATOR_ACTIVE_COALESCE_US" =~ ^[0-9]+$ ]] ||
   (( COORDINATOR_ACTIVE_COALESCE_US > 100000 )); then
  echo "Invalid DS4_SERVER_DSPARK_ACTIVE_COALESCE_US: $COORDINATOR_ACTIVE_COALESCE_US (expected 0..100000)" >&2
  exit 2
fi
if [[ ! "$COORDINATOR_MIN_PHYSICAL_ROWS" =~ ^[0-9]+$ ]] ||
   (( COORDINATOR_MIN_PHYSICAL_ROWS > 64 )); then
  echo "Invalid DS4_DSPARK_RN_MIN_PHYSICAL_ROWS: $COORDINATOR_MIN_PHYSICAL_ROWS (expected 0..64)" >&2
  exit 2
fi
if [[ "$SCHEDULER_DETERMINISTIC" == "1" ]]; then
  SCHEDULER_SHADOW=1
fi
case "$HYBRID_LC" in
  0|1) ;;
  *) echo "Invalid DS4_DSPARK_HYBRID_LC: $HYBRID_LC (expected 0 or 1)" >&2; exit 2 ;;
esac
case "$HYBRID_LC_SHADOW" in
  0|1) ;;
  *) echo "Invalid DS4_DSPARK_HYBRID_LC_SHADOW: $HYBRID_LC_SHADOW (expected 0 or 1)" >&2; exit 2 ;;
esac
if [[ "$HYBRID_LC" == "1" && "$HYBRID_LC_SHADOW" == "1" ]]; then
  echo "DS4_DSPARK_HYBRID_LC and DS4_DSPARK_HYBRID_LC_SHADOW are mutually exclusive" >&2
  exit 2
fi
case "$HYBRID_WIDTH" in
  ""|8|12|16) ;;
  *) echo "Invalid DS4_DSPARK_HYBRID_WIDTH: $HYBRID_WIDTH (expected 8, 12, or 16)" >&2; exit 2 ;;
esac
if [[ -n "$HYBRID_WIDTH" && "$HYBRID_LC" != "1" ]]; then
  echo "DS4_DSPARK_HYBRID_WIDTH requires DS4_DSPARK_HYBRID_LC=1" >&2
  exit 2
fi

export DS4_CUDA_COPY_MODEL=1
export DS4_CUDA_SECONDARY_COPY_PIPELINED="$SECONDARY_COPY_PIPELINED"
export DS4_CUDA_WEIGHT_CACHE_LIMIT_GB="${DS4_CUDA_WEIGHT_CACHE_LIMIT_GB:-112}"
if [[ "${DS4_CUDA_DROP_COPIED_MODEL_PAGES:-1}" == "1" ]]; then
  export DS4_CUDA_DROP_COPIED_MODEL_PAGES=1
  unset DS4_CUDA_KEEP_MODEL_PAGES
else
  export DS4_CUDA_KEEP_MODEL_PAGES=1
fi
# The GB10 has enough unified-memory headroom for the measured 80.8 GiB target,
# 10.7 GiB sidecar, context buffers and a 12 GiB hot Q8->F16 cache.  Keeping the
# hot projections resident avoids the 6 GiB cache ceiling seen in early runs.
case "$MEMORY_PROFILE" in
  balanced)
    export DS4_CUDA_Q8_F16_CACHE_MB="${DS4_CUDA_Q8_F16_CACHE_MB:-12288}"
    export DS4_CUDA_COPY_SECONDARY_MODEL="${DS4_CUDA_COPY_SECONDARY_MODEL:-1}"
    unset DS4_CUDA_DSPARK_CACHE_COMPACT
    if [[ "$PREFILL_CHUNK" == "" ]]; then
      PREFILL_CHUNK=8192
    fi
    ;;
  prefill-fast)
    export DS4_CUDA_Q8_F16_CACHE_MB="${DS4_CUDA_Q8_F16_CACHE_MB:-12288}"
    export DS4_CUDA_COPY_SECONDARY_MODEL="${DS4_CUDA_COPY_SECONDARY_MODEL:-0}"
    unset DS4_CUDA_DSPARK_CACHE_COMPACT
    if [[ "$PREFILL_CHUNK" == "" ]]; then
      PREFILL_CHUNK=4096
    fi
    ;;
  lean)
    export DS4_CUDA_Q8_F16_CACHE_MB="${DS4_CUDA_Q8_F16_CACHE_MB:-11264}"
    export DS4_CUDA_COPY_SECONDARY_MODEL="${DS4_CUDA_COPY_SECONDARY_MODEL:-1}"
    export DS4_CUDA_DSPARK_CACHE_COMPACT="${DS4_CUDA_DSPARK_CACHE_COMPACT:-1}"
    if [[ "$PREFILL_CHUNK" == "" ]]; then
      PREFILL_CHUNK=4096
    fi
    ;;
  *)
    echo "Invalid DS4_MEMORY_PROFILE: $MEMORY_PROFILE (expected balanced, prefill-fast or lean)" >&2
    exit 2
    ;;
esac
export DS4_CUDA_DSPARK_CACHE_PRIORITY=1
export DS4_CUDA_DEFER_END_SYNC=1
export DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS=0
export DS4_CUDA_FUSED_COMPRESSOR_UPDATE=1
export DS4_CUDA_TOKEN_GRAPH=1
export DS4_CUDA_DSPARK_GRAPH=1
export DS4_CUDA_COALESCED_F16_MATMUL=1
export DS4_CUDA_Q8_U16_LOADS=1
export DS4_STREAM_HEARTBEAT_SEC="$STREAM_HEARTBEAT_SEC"
# Long retries must find the exact full-prompt checkpoint even when the prompt
# exceeds cold_max_tokens.  canonical-only writes no intermediate frontiers.
export DS4_KV_KEEP_LONG_TEXT_HITS="${DS4_KV_KEEP_LONG_TEXT_HITS:-1}"
export DS4_KV_PREFILL_CHECKPOINT_POLICY="$PREFILL_POLICY"
export DS4_KV_CANONICAL_LONG_PREFILL="${DS4_KV_CANONICAL_LONG_PREFILL:-1}"
export DS4_KV_CANONICAL_PREFILL_MIN_SEC="${DS4_KV_CANONICAL_PREFILL_MIN_SEC:-30}"
export DS4_KV_LONG_COLD_ANCHOR_MIN_TOKENS="$LONG_ANCHOR_MIN_TOKENS"
export DS4_KV_LONG_COLD_ANCHOR_TRIM_TOKENS="$LONG_ANCHOR_TRIM_TOKENS"
if [[ "${DS4_PREFILL_FINAL_LOGITS_ONLY:-}" != "" ]]; then
  export DS4_PREFILL_FINAL_LOGITS_ONLY
elif [[ "$PREFILL_POLICY" == "canonical-only" ]]; then
  export DS4_PREFILL_FINAL_LOGITS_ONLY=1
else
  unset DS4_PREFILL_FINAL_LOGITS_ONLY
fi
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

# Match the released DeepSeek forward_head contract and retain both recurring
# verifier graph topologies by default.  Normalize explicit zero values by
# unsetting the rollback variables because the C/CUDA paths test their
# presence, not their textual value.
if [[ "$CONFIDENCE_POST_NORM" == "1" ]]; then
  export DS4_DSPARK_CONFIDENCE_POST_NORM=1
  CONFIDENCE_INPUT="post-RMSNorm (rollback)"
else
  unset DS4_DSPARK_CONFIDENCE_POST_NORM
  CONFIDENCE_INPUT="pre-RMSNorm"
fi
if [[ "$GRAPH_TOPOLOGY_CACHE_DISABLE" == "1" ]]; then
  export DS4_CUDA_DSPARK_GRAPH_TOPOLOGY_CACHE_DISABLE=1
  GRAPH_TOPOLOGY_CACHE="single-slot (rollback)"
else
  unset DS4_CUDA_DSPARK_GRAPH_TOPOLOGY_CACHE_DISABLE
  GRAPH_TOPOLOGY_CACHE="two-slot"
fi

# The look-ahead host capture is intentionally not used with DSpark yet.  The
# normal token graphs remain available for fallback; separate K-aware families
# handle the drafter and the fused [current + K draft] target verifier.
unset DS4_CUDA_TOKEN_GRAPH_PIPELINE
unset DS4_CUDA_MTP_GRAPH
unset DS4_CUDA_MTP_TENSOR_CORES
if [[ "$HYBRID_LC" == "1" ]]; then
  export DS4_DSPARK_HYBRID_LC=1
  if [[ -n "$HYBRID_WIDTH" ]]; then
    export DS4_DSPARK_HYBRID_WIDTH="$HYBRID_WIDTH"
  else
    unset DS4_DSPARK_HYBRID_WIDTH
  fi
else
  unset DS4_DSPARK_HYBRID_LC
  unset DS4_DSPARK_HYBRID_WIDTH
fi
if [[ "$HYBRID_LC_SHADOW" == "1" ]]; then
  export DS4_DSPARK_HYBRID_LC_SHADOW=1
else
  unset DS4_DSPARK_HYBRID_LC_SHADOW
fi

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
if [[ "$SCHEDULER_SHADOW" == "1" ]]; then
  export DS4_DSPARK_SCHEDULER_SHADOW=1
else
  unset DS4_DSPARK_SCHEDULER_SHADOW
fi
if [[ "$SCHEDULER_DETERMINISTIC" == "1" ]]; then
  export DS4_DSPARK_SCHEDULER_DETERMINISTIC=1
else
  unset DS4_DSPARK_SCHEDULER_DETERMINISTIC
fi
export DS4_SERVER_DSPARK_LANES="$COORDINATOR_LANES"
export DS4_SERVER_DSPARK_HOT_LANES="$COORDINATOR_HOT_LANES"
export DS4_SERVER_DSPARK_LANE_RESERVE_MB="$COORDINATOR_LANE_RESERVE_MB"
export DS4_SERVER_DSPARK_COALESCE_US="$COORDINATOR_COALESCE_US"
export DS4_SERVER_DSPARK_ACTIVE_COALESCE_US="$COORDINATOR_ACTIVE_COALESCE_US"
export DS4_DSPARK_RN_MIN_PHYSICAL_ROWS="$COORDINATOR_MIN_PHYSICAL_ROWS"
if [[ -n "$STS_PROFILE" ]]; then
  if [[ ! -f "$STS_PROFILE" ]]; then
    echo "DSpark STS profile not found: $STS_PROFILE" >&2
    exit 2
  fi
  export DS4_DSPARK_STS_PROFILE="$STS_PROFILE"
else
  unset DS4_DSPARK_STS_PROFILE
fi

echo "Target: $MODEL"
echo "DSpark: $DSPARK (selection=$DSPARK_SELECTION draft=$DRAFT)"
echo "Cache:  profile=$MEMORY_PROFILE Q8->F16=${DS4_CUDA_Q8_F16_CACHE_MB} MiB compact-priority=${DS4_CUDA_DSPARK_CACHE_COMPACT:-0}, weight limit=${DS4_CUDA_WEIGHT_CACHE_LIMIT_GB} GiB"
echo "Memory: secondary-copy=${DS4_CUDA_COPY_SECONDARY_MODEL:-1}, secondary-pipelined=$DS4_CUDA_SECONDARY_COPY_PIPELINED, drop-copied-source-pages=${DS4_CUDA_DROP_COPIED_MODEL_PAGES:-0}"
echo "Prefill: chunk=$PREFILL_CHUNK final-logits-only=${DS4_PREFILL_FINAL_LOGITS_ONLY:-0}"
echo "Streaming: decode-heartbeat=${DS4_STREAM_HEARTBEAT_SEC}s"
echo "KV:     policy=$DS4_KV_PREFILL_CHECKPOINT_POLICY keep-long-text-hits=$DS4_KV_KEEP_LONG_TEXT_HITS canonical-min-sec=$DS4_KV_CANONICAL_PREFILL_MIN_SEC cold-max=$KV_COLD_MAX_TOKENS long-anchor-min=$DS4_KV_LONG_COLD_ANCHOR_MIN_TOKENS trim=$DS4_KV_LONG_COLD_ANCHOR_TRIM_TOKENS disk-mb=$KV_DISK_SPACE_MB"
echo "Context guard: physical=$CTX advertise=${ADVERTISE_CONTEXT_PCT}%"
echo "DSpark scheduler: hardware-aware Algorithm 1 + exact t-2 production capacity, full 5-slot draft, adaptive verifier K=0..$DRAFT, always-draft=${DS4_DSPARK_ALWAYS_DRAFT:-0}, circuit-breaker=${DS4_DSPARK_CIRCUIT_BREAKER:-0}, fused K+1 verifier, graphs=on, telemetry=$TELEMETRY shadow=$SCHEDULER_SHADOW deterministic=$SCHEDULER_DETERMINISTIC"
echo "DSpark coordinator: resident=$COORDINATOR_HOT_LANES max=$COORDINATOR_LANES reserve=${COORDINATOR_LANE_RESERVE_MB}MiB coalesce=${COORDINATOR_COALESCE_US}us active-rendezvous=${COORDINATOR_ACTIVE_COALESCE_US}us min-physical-rows=$COORDINATOR_MIN_PHYSICAL_ROWS load-aware-prefix=1 physical-R1..R${COORDINATOR_LANES}=cost-gated serial-fallback=1"
echo "DSpark STS: profile=${STS_PROFILE:-online-fallback} capture=${DS4_DSPARK_STS_CAPTURE:-disabled}"
echo "DSpark parity: confidence-input=$CONFIDENCE_INPUT, verifier-topology-cache=$GRAPH_TOPOLOGY_CACHE"
echo "DSpark sampling: lossless p/q rejection for top_k=0 top_p=1 min-p policy (rollback DS4_DSPARK_REJECTION_DISABLE=1)"
echo "HybridLC: enabled=$HYBRID_LC shadow=$HYBRID_LC_SHADOW indexed-suffix=8-token transition-q=top8 BlockV=lossless max-draft=15 graph-rows=8/12/16 forced-width=${HYBRID_WIDTH:-auto}"
echo "GB10 verifier: Q8 batch-reuse=${DS4_CUDA_Q8_BATCH_REUSE:-0}, Q4-sidecar direct-MoE=${DS4_CUDA_MOE_TINY_DIRECT_Q4_ONLY:-0}, tiny-TC=${DS4_CUDA_DSPARK_TENSOR_CORES:-0}, tiny-TC-Q8=${DS4_CUDA_DSPARK_TENSOR_CORES_Q8:-0}"
if [[ "${DS4_CUDA_NVTX:-0}" == "1" ||
      "${DS4_CUDA_NSYS_PREFILL_START_POS:-}" != "" ||
      "${DS4_CUDA_NSYS_PREFILL_START_POSITIONS:-}" != "" ||
      "${DS4_CUDA_NSYS_CAPTURE_START_POS:-}" != "" ]]; then
  echo "Profiling: NVTX=1 prefill-capture-start=${DS4_CUDA_NSYS_PREFILL_START_POS:-disabled} prefill-capture-list=${DS4_CUDA_NSYS_PREFILL_START_POSITIONS:-disabled} decode-capture-start=${DS4_CUDA_NSYS_CAPTURE_START_POS:-disabled} decode-capture-tokens=${DS4_CUDA_NSYS_CAPTURE_TOKENS:-disabled}"
fi
echo "Server: http://0.0.0.0:$PORT"

exec ./ds4-server \
  --cuda \
  -m "$MODEL" \
  --dspark "$DSPARK" \
  --dspark-draft "$DRAFT" \
  --prefill-chunk "$PREFILL_CHUNK" \
  -c "$CTX" \
  --advertise-context-pct "$ADVERTISE_CONTEXT_PCT" \
  -n "$MAX_TOKENS" \
  -t "$THREADS" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --kv-disk-dir "$KV_DIR" \
  --kv-cache-cold-max-tokens "$KV_COLD_MAX_TOKENS" \
  --kv-disk-space-mb "$KV_DISK_SPACE_MB"
