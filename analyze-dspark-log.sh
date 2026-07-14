#!/usr/bin/env bash
# GB10/GX10 DSpark CUDA modifications:
# Copyright (c) 2026 Marco Palaferri. Licensed under the MIT License.
set -euo pipefail

LOG="${1:-/tmp/ds4-dspark.log}"
SKIP_SUMMARIES="${2:-1}"
if [[ ! -f "$LOG" ]]; then
  echo "Log not found: $LOG" >&2
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
function text_value(prefix,    i,a) {
  for (i = 1; i <= NF; i++) {
    split($i, a, "=")
    if (a[1] == prefix) return a[2]
  }
  return ""
}
/dspark first-draft miss/ { first_miss++ }
/dspark scheduler selected=/ {
  b = value("block"); if (b >= 0) { block_sum += b; block_n++ }
  p = value("proposed"); if (p >= 0) { proposed_sum += p; proposed_n++ }
  h = value("history"); if (h == 1) history_active++
  k = value("selected")
  if (k >= 0 && k <= 5) selected[k]++
  k = value("early_stop")
  if (k >= 1 && k <= 5) early_stop[k]++
  k = value("champion")
  if (k >= 1 && k <= 5) champion[k]++
}
/dspark scheduler bypass/ { bypass++ }
/dspark pre-draft bypass/ { predraft_bypass++ }
/dspark scheduler K0 cooldown=/ { k0_cooldowns++ }
/CUDA speculative graph launches=/ {
  v = value("launches"); if (v >= 0) graph_launches = v
  v = value("dspark_draft"); if (v >= 0) graph_draft = v
  v = value("dspark_verify"); if (v >= 0) graph_verify = v
  v = value("updates"); if (v >= 0) graph_updates = v
  v = value("rebuilds"); if (v >= 0) graph_rebuilds = v
}
/dspark timing/ {
  cycles++
  f = value("fused")
  if (f == 1) fused_cycles++
  d = value("drafted")
  c = value("committed")
  e = value("emitted")
  r = value("target_rows")
  dr = value("draft")
  vr = value("verify")
  if (vr < 0) vr = value("verify-replay")
  if (vr < 0) vr = value("verify-commit")
  total = value("total")
  rejection = value("rejection")
  residual = value("residual")
  if (rejection == 1) rejection_cycles++
  if (residual == 1) residual_cycles++
  if (e >= 0) emitted += e
  if (dr >= 0) draft_ms += dr
  if (vr >= 0) verify_ms += vr
  if (total >= 0) total_ms += total

  # target_rows=1 is an ordinary current-token fallback.  It may have drafted
  # candidates before K=0 was selected, but none of those candidates entered a
  # target verifier and therefore must not dilute acceptance.
  if (r > 1) {
    verifier_cycles++
    k = r - 1
    if (d >= 0) verifier_drafted += d
    if (c >= 0) {
      verifier_committed += c
      if (c == d) full++
      else partial++
    }
    verifier_emitted += e
    verifier_rows += r
    verifier_draft_ms += dr
    verifier_verify_ms += vr
    verifier_total_ms += total
    if (k >= 1 && k <= 5) {
      kn[k]++
      kd[k] += d
      kc[k] += c
      ke[k] += e
      kdr[k] += dr
      kvr[k] += vr
      kt[k] += total
    }
  } else if (r == 1) {
    fallback_cycles++
    reason = text_value("fallback")
    if (reason == "") reason = "unspecified"
    fallback_reason[reason]++
  } else {
    legacy_cycles++
    if (d >= 0) legacy_drafted += d
    if (c >= 0) legacy_committed += c
  }
}
/ds4-server: decode summary / {
  summary_seen++
  if (summary_seen <= summary_skip) next
  summary_n++
  v = value("gen"); if (v >= 0) summary_tokens += v
  v = value("seconds"); if (v >= 0) summary_seconds += v
  v = value("tps"); if (v >= 0) last_tps = v
}
/decoding chunk=/ {
  v = value("avg"); if (v >= 0) last_tps = v
}
END {
  # Old anchor-first logs omitted timing rows on a first-draft miss.  Every
  # fused K+1 attempt has a timing row, so misses are already a subset.
  attempts = cycles + (fused_cycles > 0 ? 0 : first_miss)
  printf "DSpark cycles timed:       %d\n", cycles
  if (fused_cycles > 0) {
    printf "Fused verifier cycles:     %d\n", verifier_cycles
    printf "Ordinary/fallback cycles:  %d\n", fallback_cycles
  }
  printf "First-draft misses:        %d\n", first_miss
  printf "Speculative attempts:      %d\n", attempts
  if (rejection_cycles > 0) {
    printf "P/Q rejection cycles:      %d\n", rejection_cycles
    printf "Residual corrections:      %d\n", residual_cycles
  }
  if (fused_cycles > 0) {
    printf "Verified draft tokens:     %d\n", verifier_drafted
    printf "Committed draft tokens:    %d\n", verifier_committed
  } else {
    printf "Drafted suffix tokens:     %d\n", legacy_drafted
    printf "Committed suffix tokens:   %d\n", legacy_committed
  }
  if (emitted > 0) printf "Total emitted by cycles:   %d\n", emitted
  if (verifier_drafted > 0) printf "Verifier acceptance:       %.2f%%\n", 100.0 * verifier_committed / verifier_drafted
  else if (legacy_drafted > 0) printf "Suffix acceptance:         %.2f%%\n", 100.0 * legacy_committed / legacy_drafted
  printf "Full / partial accepts:    %d / %d\n", full, partial
  printf "Scheduler K=0..5:          %d %d %d %d %d %d\n", selected[0], selected[1], selected[2], selected[3], selected[4], selected[5]
  printf "Causal stops K=1..5:       %d %d %d %d %d\n", early_stop[1], early_stop[2], early_stop[3], early_stop[4], early_stop[5]
  printf "Champion K=1..5:           %d %d %d %d %d\n", champion[1], champion[2], champion[3], champion[4], champion[5]
  if (block_n > 0) printf "Mean DSpark block/proposed: %.3f / %.3f\n", block_sum / block_n, proposed_sum / proposed_n
  printf "History-latched cycles:    %d\n", history_active
  printf "K0 cooldown activations:   %d\n", k0_cooldowns
  printf "Pre-draft history bypasses:%6d\n", predraft_bypass
  printf "Circuit-breaker bypasses:  %d\n", bypass
  if (graph_launches > 0) printf "DSpark graph launch/update: %d / %d (draft=%d verify=%d rebuilds=%d)\n", graph_launches, graph_updates, graph_draft, graph_verify, graph_rebuilds
  if (verifier_cycles > 0) {
    printf "Mean verifier target rows: %.3f\n", verifier_rows / verifier_cycles
    printf "Mean verifier draft time:  %.3f ms\n", verifier_draft_ms / verifier_cycles
    printf "Mean verifier target time: %.3f ms\n", verifier_verify_ms / verifier_cycles
    printf "Mean fused cycle:          %.3f ms\n", verifier_total_ms / verifier_cycles
    if (verifier_total_ms > 0) printf "Verifier-cycle throughput: %.3f t/s\n", 1000.0 * verifier_emitted / verifier_total_ms
    for (k = 1; k <= 5; k++) if (kn[k] > 0) {
      printf "  K=%d n=%d accept=%.2f%% emitted=%.3f draft=%.3fms target=%.3fms total=%.3fms rate=%.3ft/s\n",
             k, kn[k], (kd[k] > 0 ? 100.0 * kc[k] / kd[k] : 0.0),
             ke[k] / kn[k], kdr[k] / kn[k], kvr[k] / kn[k],
             kt[k] / kn[k], (kt[k] > 0 ? 1000.0 * ke[k] / kt[k] : 0.0)
    }
  }
  if (fallback_cycles > 0) {
    printf "Fallback reasons:"
    for (reason in fallback_reason) printf " %s=%d", reason, fallback_reason[reason]
    printf "\n"
  }
  if (summary_n > 0 && summary_seconds > 0) printf "Weighted request decode:   %.3f t/s (%d summaries)\n", summary_tokens / summary_seconds, summary_n
  else if (last_tps > 0) printf "Final cumulative decode:   %.3f t/s (legacy log)\n", last_tps
}
' "$LOG"
