#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$(mktemp "${TMPDIR:-/tmp}/ds4-shadow-log.XXXXXX")"
OUT="$(mktemp "${TMPDIR:-/tmp}/ds4-shadow-out.XXXXXX")"
trap 'rm -f "$LOG" "$OUT"' EXIT

printf '%s\n' \
  'ds4: dspark scheduler selected=3 legacy=3 deterministic=1 block=5 proposed=5 configured=5 pos=65536 bucket=2 shadow_k=4 shadow_ready=0 shadow_rate=20.00 shadow_stop=0 early_stop=0 champion=3 history=1' \
  'ds4: dspark timing drafted=3 target_rows=4 committed=2 emitted=3 draft=20.000 ms verify=140.000 ms total=160.000 ms fused=1' \
  'ds4: dspark scheduler selected=3 legacy=4 deterministic=1 block=5 proposed=5 configured=5 pos=65539 bucket=2 shadow_k=4 shadow_ready=1 shadow_rate=22.00 shadow_local_ready=0 probe_k=3 shadow_stop=5 early_stop=0 champion=4 history=1' \
  'ds4: dspark timing drafted=3 target_rows=4 committed=3 emitted=3 draft=20.000 ms verify=140.000 ms total=160.000 ms fused=1' \
  'ds4: dspark scheduler selected=4 legacy=4 deterministic=1 block=5 proposed=5 configured=5 pos=65542 bucket=2 shadow_k=3 shadow_ready=1 shadow_rate=21.00 shadow_stop=4 early_stop=0 champion=4 history=1' \
  >"$LOG"

"$ROOT/analyze-dspark-log.sh" "$LOG" 0 >"$OUT"

grep -Fq 'DSpark cycles timed:       2' "$OUT"
grep -Fq 'Verifier acceptance:       83.33%' "$OUT"
grep -Fq 'Shadow K=1..5:             0 0 1 2 0' "$OUT"
grep -Fq 'Shadow ready/agreement:    2 / 0 (0.00% agreement)' "$OUT"
grep -Fq 'Deterministic probes K=1..5: 0 0 1 0 0 (total=1)' "$OUT"
grep -Fq 'Shadow local-profile ready: 0 / 3 (0.00%)' "$OUT"
grep -Fq 'Shadow causal stops:       0 0 0 1 1' "$OUT"
grep -Fq 'Deterministic cycles/overrides: 3 / 1' "$OUT"
grep -Fq 'Mean shadow predicted rate: 21.000 t/s' "$OUT"
grep -Fq 'Verifier-cycle throughput: 18.750 t/s' "$OUT"

printf 'dspark analyzer shadow regression: OK\n'
