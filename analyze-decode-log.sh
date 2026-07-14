#!/usr/bin/env bash
# GB10/GX10 DSpark CUDA modifications:
# Copyright (c) 2026 Marco Palaferri. Licensed under the MIT License.
set -euo pipefail

LOG="${1:-/tmp/ds4-gb10.log}"
SKIP_FIRST="${2:-1}"
MIN_TOKENS="${3:-50}"

if [[ ! -f "$LOG" ]]; then
  echo "Log not found: $LOG" >&2
  exit 2
fi
if ! [[ "$SKIP_FIRST" =~ ^[0-9]+$ && "$MIN_TOKENS" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 LOG [SKIP_FIRST=1] [MIN_TOKENS=50]" >&2
  exit 2
fi

awk -v skip="$SKIP_FIRST" -v min_tokens="$MIN_TOKENS" '
function value(key,    i,a) {
  for (i = 1; i <= NF; i++) {
    split($i, a, "=")
    if (a[1] == key) return a[2]
  }
  return ""
}
/ds4-server: decode summary / {
  req = value("req")
  kind = value("kind")
  prompt = value("prompt") + 0
  gen = value("gen") + 0
  sec = value("seconds") + 0
  tps = value("tps") + 0
  greedy = value("greedy_gpu") + 0
  finish = value("finish")
  summaries++
  if (summaries <= skip) {
    warmup++
    next
  }
  if (gen < min_tokens || sec <= 0) {
    short++
    next
  }
  used++
  tokens += gen
  seconds += sec
  sum_tps += tps
  if (used == 1 || tps < min_tps) min_tps = tps
  if (used == 1 || tps > max_tps) max_tps = tps
  printf "%2d  %-24s %-10s prompt=%-6d gen=%-5d %8.3fs  %7.3f t/s  greedy_gpu=%d  %s\n",
         used, req, kind, prompt, gen, sec, tps, greedy, finish
}
END {
  if (used == 0) {
    printf "No usable decode summaries (summaries=%d skip=%d min_tokens=%d short=%d).\n",
           summaries, skip, min_tokens, (short > 0 ? short : 0)
    exit 3
  }
  printf "\nRequests used:          %d (warm-up skipped=%d, short ignored=%d)\n",
         used, warmup, (short > 0 ? short : 0)
  printf "Total generated:        %d tokens\n", tokens
  printf "Total decode time:      %.3f s\n", seconds
  printf "Weighted throughput:    %.3f t/s\n", tokens / seconds
  printf "Request mean/min/max:   %.3f / %.3f / %.3f t/s\n",
         sum_tps / used, min_tps, max_tps
}
' "$LOG"
