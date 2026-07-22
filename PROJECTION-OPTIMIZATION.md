# GB10 decode projection optimization

The CUDA projection work is deliberately guarded by separate, stable
operation policies. Every production scope defaults to `legacy`; candidate
paths must pass the corresponding GB10 correctness and performance gates
before any default is changed.

## Frozen reference

The pre-change reference is commit `0af740697b6fa41f378ffe89273101f200ac610b`.
The locally frozen binaries and SHA-256 values are:

| Binary | Frozen copy | SHA-256 |
| --- | --- | --- |
| `ds4-server` | `/tmp/ds4-server-projection-baseline-0af7406` | `2cdfa4c96d58608ac95494528fded7381f668c61c3253ae957ca1cac32172139` |
| `ds4-bench` | `/tmp/ds4-bench-projection-baseline-0af7406` | `135de331f0d448e04e17e1e4d943db48a29b8d2bafa3b1df1193ca986a8a3d3f` |

The copies are host-local artifacts, not repository inputs. Preserve them with
the corresponding Git diff and benchmark JSON when moving the experiment to
the GB10 host.

## Dispatch policies

Projection policy is scoped by the operation being timed:

- `DS4_CUDA_TARGET_PROJECTION_POLICY` controls only the large-model verifier;
- `DS4_CUDA_DSPARK_PROJECTION_POLICY` controls only DSpark decode;
- `DS4_CUDA_MTP_PROJECTION_POLICY` controls only the optional MTP model.

All three default to `legacy`. The older `DS4_CUDA_PROJECTION_POLICY` remains
as a compatibility fallback for the target only; it never changes DSpark or
MTP implicitly.

| Policy | Purpose |
| --- | --- |
| `legacy` | Existing cache and Q8 dispatch; production rollback path and default. |
| `auto` | Hybrid N=2..6 dispatch: padded F16 vocabulary output, native Q8 weight reuse for Q-A/KV/gate/up/indexer, and F16-output fusion for Q-B/attention output/shared-down. |
| `f16-real` | Diagnostic unpadded cuBLAS F16 path with FP32 outputs. |
| `f16-pad8` | Diagnostic padded tiny-batch F16 path with FP32 outputs. |
| `f16-out` | Diagnostic unpadded F16 path with the Q-B/output/shared-down half-output fusions. |
| `q8-reuse` | Diagnostic native Q8 path that bypasses the expanded-weight caches. |

`DS4_CUDA_PROJECTION_VERBOSE=1` prints per-projection dispatch counters every
1024 calls, including the actual path, N, cache hit/miss/fallback counts and
resident FP16 bytes per category. Policies are parsed once so CUDA Graph
warm-up and capture see stable dispatch. Model mappings are registered with an
explicit target/DSpark/MTP role; dispatch never infers the role from tensor
names. The active operation scope overrides weight ownership: the target
vocabulary head reused by a DSpark draft follows the DSpark policy, while
DSpark-side state updated inside target verification follows the target
policy. Target prefill is a separate phase that always keeps legacy arithmetic
and cache ordering, including a final short N=2..6 chunk. The target fills its
configured cache before the sidecar is visited: DSpark may use only residual
capacity and cannot evict target-prefill weights. A future dedicated sidecar
F16 budget must be added separately rather than borrowed from target.
Allocation or cuBLAS failure disables only the active operation scope; it does
not flush valid cache entries belonging to prefill or another decode path.
Runtime statistics expose target, DSpark, and MTP fallback counters separately;
the benchmark gates only the counter belonging to `--projection-scope` and
rejects fallback growth over the corresponding legacy baseline.

Non-CUDA backends accept the projection identity but retain their prior
dispatch. The paired indexer-compressor operation falls back to its original
two calls on a backend that cannot pair the active batch shape.

## Candidate fusions

For verifier batches of two through six rows, `auto` keeps these intermediates
in F16:

- Q-B output through head RMS normalization and tail RoPE;
- attention output A/B through the HC residual expansion;
- shared-down output through the routed-output add and HC expansion.

Q-A/KV and shared gate/up use one activation quantization for both native Q8
projections. The two F16 indexer compressors use the paired tiny-batch GEMM so
they share activation conversion and padding.

`auto` also contains the phase-6 narrow-F16 candidate for the exact
`16384 -> 24`, `4096 -> 64`, and `4096 -> 256` shapes. One warp owns an output
row and accumulates all N=2..6 verifier rows, reusing each F16 weight load
across the batch; the ratio-4 compressor pair reuses the activation in one
paired kernel. The dispatch has exact shape guards, performs no allocation,
and falls back to the existing F16 path for every other shape. Set
`DS4_CUDA_DISABLE_NARROW_F16=1` to isolate `auto` without this candidate;
`legacy` never enables it.

If an end-to-end candidate changes the greedy token hash, use the diagnostic
`DS4_CUDA_PROJECTION_AUTO_COMPONENTS` list to attribute the change without
adding production policies. Its default is `all`; accepted entries are
`q8-reuse`, `f16-gemm`, `f16-out`, and `narrow-f16`, while `none` makes AUTO
use legacy arithmetic for every classified projection. Cache ordering remains
AUTO's in every ablation, so the comparison also verifies that ordering alone
does not affect tokens. `tools/benchmark_gb10.py --auto-components ...` sets
and records the mask explicitly, preventing a stale shell environment from
contaminating a gate run.

## GB10 gate

Build and run the CUDA regression first:

```sh
make cuda-spark-graph-sm121
make cuda-regression
```

Before any end-to-end graph run, measure the real projection blocks. Each
process fixes one policy before loading the model; `ds4-bench` then allocates
persistent scratch for N=2..6, performs five untimed warm-ups and measures with
CUDA events. N=5 receives twice the requested iterations. The driver repeats
each policy three times and writes `selection.csv`, retaining a diagnostic path
only when its median complete-block gain over `legacy` is at least 3%:

```sh
python3 tools/benchmark_projection_blocks.py \
  --model /home/athena/ds4/ds4flash.gguf \
  --dspark /home/athena/ds4/DeepSeek-V4-Flash-DSpark-Q4K-Q8.gguf \
  --iterations 50 \
  --repeats 3 \
  --policies legacy,f16-real,f16-pad8,f16-out,q8-reuse,auto \
  --output-dir benchmark-results/gb10-projection-blocks
```

The measured blocks include Q-A/KV, Q-B plus RMS/RoPE, attention output plus
HC expansion, shared gate/up plus SwiGLU, shared-down plus routed/HC expansion,
the indexer Q-B and compressor pair, the narrow HC/router/indexer F16 shapes,
and the vocabulary projection. The driver
records historical `legacy` fallback but rejects fallback in every candidate
policy, or any CUDA Graph construction, during this pre-capture phase.
Use the selected diagnostic path to update only the corresponding cells in the
static `(projection_kind, N)` table; the following end-to-end gate remains
authoritative for verifier regressions.

Then run the end-to-end policy matrix. The benchmark exports per-frontier
`draft_ms`, `target_ms`, `total_ms`, draft/target/emitted rows per cycle and
acceptance directly from session counters. It also snapshots graph rebuilds,
live graph node counts, DSpark graph launches, projection fallbacks, and the
resident FP16 cache without enabling verbose hot-path telemetry. `target_ms`
is the primary verifier indicator. This command runs adaptive scheduling and
fixed K=4/N=5, stores separate raw logs, checks deterministic greedy hashes
across every policy and schedule, and enforces the 10% throughput criterion:

```sh
python3 tools/benchmark_gb10.py \
  --model /home/athena/ds4/ds4flash.gguf \
  --dspark /home/athena/ds4/DeepSeek-V4-Flash-DSpark-Q4K-Q8.gguf \
  --prompt speed-bench/promessi_sposi.txt \
  --frontiers 27000,63000,90500,93500 \
  --ctx 131072 \
  --gen-tokens 512 \
  --schedules adaptive,fixed-k4 \
  --projection-policies legacy,auto,f16-real,f16-pad8,f16-out,q8-reuse \
  --projection-scope target \
  --min-auto-target-gain 4 \
  --min-auto-decode-gain 10 \
  --append-only \
  --repeats 3 \
  --output-dir benchmark-results/gb10-projections
```

Run the same matrix with `--projection-scope dspark` to measure only the small
model (`draft_ms`), and use `--projection-scope both` only after the two scoped
matrices pass independently. `target_ms` remains authoritative for the large
model verifier; `draft_ms` is authoritative for DSpark decode. `prefill_tps`
is a non-regression guard and is never used to select a decode candidate.

`summary.json` also records the benchmark binary hash, Git revision/status,
effective `DS4_*` configuration and input identities. `worktree.diff` freezes
the tracked source delta next to the measurements.

To use the alternative verifier-cycle criterion, replace
`--min-auto-decode-gain 10` with `--min-fixed-target-ms-saved 12`. Supplying
both options intentionally enforces both gates.

Keep projection telemetry off for the timed gate. Run one additional diagnostic
matrix with `--projection-telemetry` to capture dispatch/cache residency in the
per-run logs; those counters are opt-in so normal measurements do not pay their
hot-path cost.

Acceptance requires all of the following:

- CUDA regression succeeds without non-finite values or launch errors;
- greedy token hashes match for all compared policies;
- aggregate `auto` verifier `target_ms` improves by at least 4%, with no
  measured frontier regressing by more than 0.5%;
- the median aggregate `auto` result is at least 10% above `legacy` (the
  historical 23.46 t/s reference implies at least 25.81 t/s);
- fixed K=4 reduces average `target_ms` by at least 12 ms with unchanged
  acceptance, or the throughput gate above passes;
- `auto` records no projection fallback, no more graph rebuilds or live graph
  nodes than `legacy`, and no larger resident FP16 cache;
- prefill throughput regresses by at most 1%, startup by at most 5%, and
  process peak memory grows by at most 256 MiB (all configurable CLI gates);
- repeated runs show no per-shape regression large enough to erase the
  aggregate gain.

Promote `auto` for one scope in `run-dspark-server.sh` only after that scope
records a passing result. Until then, production startup remains explicitly
`legacy`. Immediate rollback is:

```sh
DS4_CUDA_TARGET_PROJECTION_POLICY=legacy \
DS4_CUDA_DSPARK_PROJECTION_POLICY=legacy \
./run-dspark-server.sh
```

## Final scheduler selection

The four-frontier configuration sweep rejected `adaptive`: its greedy output
was not deterministic between repeats at any measured frontier. All fixed
schedules K=1..5 retained the same reference hash at each frontier. Their
aggregate result selected fixed K=2:

| Schedule | Decode t/s | Acceptance | Target rows/token | Target ms |
| --- | ---: | ---: | ---: | ---: |
| fixed-k1 | 10.787 | 0.499 | 1.333 | 119.01 |
| **fixed-k2** | **11.096** | **0.343** | **1.777** | **131.56** |
| fixed-k3 | 10.577 | 0.250 | 2.278 | 144.64 |
| fixed-k4 | 9.862 | 0.195 | 2.795 | 159.12 |
| fixed-k5 | 9.308 | 0.158 | 3.335 | 169.97 |

Fixed K=2 improves decode throughput by 12.51% over fixed K=4 and reduces
verifier `target_ms` by 17.32%. Relative to adaptive it is 3.49% faster, while
also restoring deterministic generation. `run-dspark-server.sh` therefore
defaults to `DS4_DSPARK_DRAFT=2` with fixed verification enabled. Target and
DSpark projection policies remain `legacy`, since no projection candidate
passed the correctness and verifier-performance gates.

The production default can be made explicit with:

```sh
DS4_DSPARK_DRAFT=2 \
DS4_DSPARK_FIXED_VERIFY=1 \
DS4_CUDA_TARGET_PROJECTION_POLICY=legacy \
DS4_CUDA_DSPARK_PROJECTION_POLICY=legacy \
./run-dspark-server.sh
```

Diagnostic rollback to the previous adaptive K=0..5 policy is:

```sh
DS4_DSPARK_DRAFT=5 \
DS4_DSPARK_FIXED_VERIFY=0 \
DS4_CUDA_TARGET_PROJECTION_POLICY=legacy \
DS4_CUDA_DSPARK_PROJECTION_POLICY=legacy \
./run-dspark-server.sh
```
