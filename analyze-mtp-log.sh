#!/usr/bin/env bash
# GB10/GX10 DSpark CUDA modifications:
# Copyright (c) 2026 Marco Palaferri. Licensed under the MIT License.
set -euo pipefail

LOG="${1:-/tmp/ds4-mtp-tc.log}"
SKIP_SUMMARIES="${2:-1}"
if [[ ! -f "$LOG" ]]; then
  echo "Log not found: $LOG" >&2
  exit 2
fi
if ! [[ "$SKIP_SUMMARIES" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 LOG [SKIP_SUMMARIES=1]" >&2
  exit 2
fi

awk -v summary_skip="$SKIP_SUMMARIES" '
function value(prefix,    i,a) {
  for (i = 1; i <= NF; i++) {
    split($i, a, "=")
    if (a[1] == prefix) return a[2] + 0
  }
  return -1
}
/mtp spec miss first/ { first_miss++ }
/CUDA MTP graph launches=/ {
  graph_launches = value("launches")
  v = value("draft")
  if (v >= 0) graph_draft_launches = v
  v = value("verifier")
  if (v >= 0) graph_verifier_launches = v
  graph_updates = value("updates")
  graph_rebuilds = value("rebuilds")
}
/CUDA MTP graph .*failed|CUDA MTP graph capture aborted/ { graph_failures++ }
/graph fallback replay enabled/ { graph_replays++ }
/CUDA MTP Tensor Core gemms=/ {
  tc_gemms = value("gemms")
  tc_q8 = value("q8")
  tc_f16 = value("f16")
  tc_fallbacks = value("fallbacks")
}
/mtp timing/ {
  cycles++
  d = value("drafted")
  c = value("committed")
  t = value("total")
  if (d >= 0) drafted += d
  if (c >= 0) {
    committed += c
    if (c == d) full++
    else partial++
  }
  if (t >= 0) total_ms += t
}
/ds4-server: decode summary / {
  summary_seen++
  if (summary_seen <= summary_skip) next
  summary_n++
  v = value("gen")
  if (v >= 0) summary_tokens += v
  v = value("seconds")
  if (v >= 0) summary_seconds += v
  v = value("tps")
  if (v >= 0) last_decode_avg = v
}
/decoding chunk=/ {
  for (i = 1; i <= NF; i++) {
    if ($i ~ /^chunk=/) {
      split($i, a, "=")
      # Chunk rates are diagnostic only.  Very short final intervals can
      # produce meaningless spikes, so never average them into throughput.
      decode_n++
    }
    if ($i ~ /^avg=/) {
      split($i, a, "=")
      last_decode_avg = a[2] + 0
    }
  }
}
END {
  attempts = cycles + first_miss
  printf "MTP cycles timed:          %d\n", cycles
  printf "First-draft misses:        %d\n", first_miss
  printf "Speculative attempts:      %d\n", attempts
  printf "Drafted suffix tokens:     %d\n", drafted
  printf "Committed suffix tokens:   %d\n", committed
  if (drafted > 0) printf "Suffix acceptance:         %.2f%%\n", 100.0 * committed / drafted
  printf "Full / partial accepts:    %d / %d\n", full, partial
  if (cycles > 0) printf "Mean MTP cycle time:       %.3f ms\n", total_ms / cycles
  if (summary_n > 0 && summary_seconds > 0) printf "Weighted request decode:   %.3f t/s (%d summaries)\n", summary_tokens / summary_seconds, summary_n
  else if (last_decode_avg > 0) printf "Final cumulative decode:   %.3f t/s (legacy log)\n", last_decode_avg
  printf "MTP graph launch/update:   %d / %d (draft=%d verifier=%d rebuilds=%d failures=%d replays=%d)\n", graph_launches, graph_updates, graph_draft_launches, graph_verifier_launches, graph_rebuilds, graph_failures, graph_replays
  printf "Tensor Core GEMMs:         %d (q8=%d f16=%d fallbacks=%d)\n", tc_gemms, tc_q8, tc_f16, tc_fallbacks
}
' "$LOG"
