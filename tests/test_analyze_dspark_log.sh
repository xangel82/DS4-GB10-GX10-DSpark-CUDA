#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$(mktemp "${TMPDIR:-/tmp}/ds4-shadow-log.XXXXXX")"
OUT="$(mktemp "${TMPDIR:-/tmp}/ds4-shadow-out.XXXXXX")"
trap 'rm -f "$LOG" "$OUT"' EXIT

printf '%s\n' \
  'ds4: dspark scheduler selected=3 legacy=3 deterministic=1 block=5 proposed=5 configured=5 pos=65536 bucket=2 shadow_k=4 shadow_ready=0 shadow_rate=20.00 shadow_stop=0 hw_r=1 hw_batch=5 hw_base=15.00 hw_sync_k=4 hw_sync_rate=20.00 hw_async=0 hw_capacity=5 hw_capacity_rate=20.00 hw_age=0 hw_stop_batch=0 hw_admitted=4 early_stop=0 champion=3 history=1' \
  'ds4: dspark timing drafted=3 target_rows=4 committed=2 emitted=3 draft=20.000 ms verify=140.000 ms total=160.000 ms fused=1 neural=3 hybrid=0 requested_r=1' \
  'ds4: dspark scheduler selected=3 legacy=4 deterministic=1 block=5 proposed=5 configured=5 pos=65539 bucket=2 shadow_k=4 shadow_ready=1 shadow_rate=22.00 shadow_local_ready=0 probe_k=3 shadow_stop=5 hw_r=1 hw_batch=5 hw_base=15.00 hw_sync_k=3 hw_sync_rate=21.00 hw_async=1 hw_capacity=5 hw_capacity_rate=22.00 hw_age=2 hw_stop_batch=6 hw_admitted=4 early_stop=0 champion=4 history=1' \
  'ds4: dspark timing drafted=3 target_rows=4 committed=3 emitted=3 draft=20.000 ms verify=140.000 ms total=160.000 ms fused=1 neural=3 hybrid=0 requested_r=1' \
  'ds4: dspark scheduler selected=4 legacy=4 deterministic=1 block=5 proposed=5 configured=5 pos=65542 bucket=2 shadow_k=3 shadow_ready=1 shadow_rate=21.00 shadow_stop=4 hw_r=1 hw_batch=4 hw_base=15.00 hw_sync_k=4 hw_sync_rate=22.00 hw_async=1 hw_capacity=4 hw_capacity_rate=21.00 hw_age=2 hw_stop_batch=5 hw_admitted=3 early_stop=0 champion=4 history=1' \
  >"$LOG"

"$ROOT/analyze-dspark-log.sh" "$LOG" 0 >"$OUT"

grep -Fq 'DSpark cycles timed:       2' "$OUT"
grep -Fq 'Verifier acceptance:       83.33%' "$OUT"
grep -Fq 'Shadow K=0..5:             0 0 0 1 2 0' "$OUT"
grep -Fq 'Shadow ready/agreement:    2 / 0 (0.00% agreement)' "$OUT"
grep -Fq 'Deterministic probes K=1..5: 0 0 1 0 0 (total=1)' "$OUT"
grep -Fq 'Shadow local-profile ready: 0 / 3 (0.00%)' "$OUT"
grep -Fq 'Shadow causal stops:       0 0 0 1 1' "$OUT"
grep -Fq 'Deterministic cycles/overrides: 3 / 1' "$OUT"
grep -Fq 'Mean shadow predicted rate: 21.000 t/s' "$OUT"
grep -Fq 'Hardware scheduler mean R/batch: 1.000 / 4.667' "$OUT"
grep -Fq 'Hardware admitted/cycle:   3.667 (causal stops=2)' "$OUT"
grep -Fq 'Hardware K0 baseline:      15.000 t/s' "$OUT"
grep -Fq 'Hardware causal K=0..5:    0 0 0 1 2 0' "$OUT"
grep -Fq 'Hardware async cycles/changes: 2 / 2' "$OUT"
grep -Fq 'Hardware mean capacity/rate: 4.667 / 21.000 t/s' "$OUT"
grep -Fq 'Verifier-cycle throughput: 18.750 t/s' "$OUT"
grep -Fq 'Neural scheduler K=0..5:   0 0 0 2 0 0' "$OUT"
grep -Fq 'neural R=1 K=0..5: 0 0 0 2 0 0' "$OUT"
grep -Fq 'R=1 K=3 n=2 neural_accept=83.33%' "$OUT"
grep -Fq 'verifier-width=3 n=2 accept=83.33%' "$OUT"

printf '%s\n' \
  'ds4: dspark admission R=2 executor=physical batch=8 capacity_batch=8 capacity_age=2 expected=7.0 rate=25.0 capacity_rate=25.0 hw_prefix=3,3 sps_curve=offline physical_curve=offline serial_curve=offline shape_curve=3/4 physical_prefix=3,3 serial_prefix=2,2 conditional=0.9/0.9' \
  'ds4: dspark nightjar mode=shadow R=2 bucket=1 context=0000000101000201 arm=7 budget=3 generated=10 executor=physical hint=serial rows=8 block=4 bin=2 round=2 lock=1 explore=0 guard=0 p=0.500000 samples=12 predicted_loss=41.000000 ms/token observed=0' \
  'ds4: dspark nightjar reward R=2 bucket=1 context=0000000101000201 gamma=3 emitted=7 latency=40.000000 ms/token pure_neural=1' \
  'ds4: dspark nightjar reward R=2 bucket=1 context=0000000101000201 budget=2 selected_budget=3 guarded=1 emitted=3 latency=20.000000 ms/token pure_neural=1' \
  'ds4: dspark coordinator mode=physical R=2 scheduled_batch=8 physical_rate=25.00 serial_batch=8 serial_rate=18.18 physical_samples=2 serial_samples=2 physical_shape_samples=2 serial_shape_samples=2 bucket=0 shape_mature=1 step=7 reason=predicted' \
  'ds4: dspark cohort timing cohort=7 requested_r=2 executor=physical fallback=none path=neural drafted=6 committed=4 emitted=7 rows=8 wait_us=1000 draft=40.000 ms verify=240.000 ms total=280.000 ms' \
  'ds4: dspark admission R=2 executor=serial batch=8 capacity_batch=8 capacity_age=2 expected=6.0 rate=18.18 capacity_rate=18.18 hw_prefix=3,3 sps_curve=lane-model physical_curve=generic serial_curve=lane-model shape_curve=5/5 physical_prefix=3,3 serial_prefix=3,3 conditional=0.8/0.8' \
  'ds4: dspark coordinator mode=serial R=2 scheduled_batch=8 physical_rate=17.00 serial_batch=8 serial_rate=18.18 physical_samples=2 serial_samples=2 physical_shape_samples=2 serial_shape_samples=2 bucket=0 shape_mature=1 step=8 reason=predicted' \
  'ds4: dspark cohort timing cohort=8 requested_r=2 executor=serial fallback=none path=neural drafted=6 committed=3 emitted=6 rows=8 wait_us=2000 draft=40.000 ms verify=290.000 ms total=330.000 ms' \
  'ds4-server: decode summary req=chatcmpl-1 kind=chat prompt=128 gen=100 seconds=10.000000 tps=10.000 finish=stop' \
  >"$LOG"

"$ROOT/analyze-dspark-log.sh" "$LOG" 0 >"$OUT"

grep -Fq 'Coordinator cohorts:       2 (physical=1 serial=1 target-only=0 fallback=0)' "$OUT"
grep -Fq 'Coordinator wait mean/max: 1.500 / 2.000 ms (>=10ms=0 >=19ms=0)' "$OUT"
grep -Fq 'rendezvous R=2 n=2 mean=1.500ms max=2.000ms' "$OUT"
grep -Fq 'Cohort aggregate throughput: 21.311 t/s' "$OUT"
grep -Fq 'physical R=2 n=1 aggregate=25.000t/s accept=66.67% rows=8.000' "$OUT"
grep -Fq 'serial R=2 n=1 aggregate=18.182t/s accept=50.00% rows=8.000' "$OUT"
grep -Fq 'Pure-neural cohort budgets (aggregate B, not lane K):' "$OUT"
grep -Fq 'physical R=2 B=6 n=1 aggregate=25.000t/s accept=66.67%' "$OUT"
grep -Fq 'serial R=2 B=6 n=1 aggregate=18.182t/s accept=50.00%' "$OUT"
grep -Fq 'Coordinator dispatch reasons: predicted=2' "$OUT"
grep -Fq 'dispatch physical R=2 reason=predicted n=1' "$OUT"
grep -Fq 'dispatch serial R=2 reason=predicted n=1' "$OUT"
grep -Fq 'Hardware shape curve:      8 / 9 hits (88.89%), full=1/2 cohorts' "$OUT"
grep -Fq 'Per-request decode (overlap): 10.000 t/s (1 summaries; not aggregate)' "$OUT"
grep -Fq 'Offline SPS curves:       1/1 cohorts (100.00%), generic=0' "$OUT"
grep -Fq 'Shape-aware SPS selections: shape=0 lane-model=1' "$OUT"
grep -Fq 'Nightjar decisions:        1 (active=0 shadow=1)' "$OUT"
grep -Fq 'Nightjar lock/explore/revoke: 1 / 0 / 0 (100.00% / 0.00% / 0.00%)' "$OUT"
grep -Fq 'Nightjar executor:         physical=1 serial=0' "$OUT"
grep -Fq 'Nightjar executor hint:    physical=0 serial=1' "$OUT"
grep -Fq 'Nightjar aggregate budget: B3=1' "$OUT"
grep -Fq 'Nightjar generated/decision: 10.00 tokens (aggregate rows across R)' "$OUT"
grep -Fq 'Nightjar observed reward:  34.000 ms/token (2 cohorts, 10 emitted)' "$OUT"
grep -Fq 'Nightjar guarded feedback: 1 / 2 rewarded cohorts' "$OUT"
grep -Fq 'physical candidate: offline=1 generic=1' "$OUT"
grep -Fq 'serial candidate:   offline=1 generic=0' "$OUT"
grep -Fq 'exact/model curves:  physical-shape=0 serial-shape=0 serial-lane-model=1' "$OUT"

printf 'dspark analyzer shadow regression: OK\n'
