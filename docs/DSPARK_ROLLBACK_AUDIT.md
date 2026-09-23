# DSpark rollback legacy audit

Host-only audit of the DSpark speculative-decoding state lifecycle in the
`ds4-gb10-lab` legacy. No CUDA was run, no GB10 was used, and no production
path was modified.

Model: **DeepSeek-V4-Flash** (`g_ds4_shape`, [ds4.c:265](../ds4.c#L265)).
Build target: **GB10 / CUDA** (`!__APPLE__ && !DS4_ROCM_BUILD`), so
`DS4_GPU_INDEXER_PACKED == 1` ([ds4.c:10493](../ds4.c#L10493)).

Launcher flags that define the productive path
(`run-dspark-server.sh`, resolved with presence semantics):

| variable | resolved | note |
|---|---|---|
| `--dspark-draft` | **5** | `DRAFT=${DS4_DSPARK_DRAFT:-5}` → K ≤ 5 |
| `DS4_DSPARK_HYBRID_LC` | **exported =1** | default branch; raises `spec_prefix_cap` 5 → **15** |
| `DS4_DSPARK_HYBRID_WIDTH` | **unset** | auto |
| `DS4_DSPARK_NIGHTJAR` | **exported =1** | R=n executor active |
| `DS4_DSPARK_NIGHTJAR_MAX_R` | **3** | |
| `DS4_SERVER_DSPARK_LANES / HOT_LANES` | **3 / 2** | |
| `DS4_DSPARK_ALWAYS_DRAFT` | **exported =1** | pre-draft bypass removed |
| `DS4_DSPARK_NO_CIRCUIT_BREAKER` | **exported =1** | `DS4_DSPARK_CIRCUIT_BREAKER` defaults 0 → the *inverse* var is exported |
| `DS4_DSPARK_CONFIDENCE_POST_NORM` | **unset** | pre-RMSNorm (presence-tested) |
| `DS4_CUDA_DSPARK_GRAPH` | **exported =1** | selects the DSpark graph capture in the verifier |
| `DS4_CUDA_MTP_GRAPH`, `DS4_CUDA_TOKEN_GRAPH_PIPELINE` | **unset** | `unset`, not `=0` — presence-tested |
| `DS4_DSPARK_RN_MIN_PHYSICAL_ROWS` | **0** | |

`HYBRID_LC=1` matters for cost: it is what makes `spec_prefix_cap = 15`
([ds4.c:11387](../ds4.c#L11387)) and allows draft depth up to 15 rows, so the
frontier machinery is provisioned and can be driven well past K=5.

---

## 1. PRODUCTIVE FLOW

The server does **not** use the single-session cycle. `ds4-server` decode goes
through the DSpark coordinator cohort:

```
ds4_server.c:11427  ds4_sessions_eval_speculative_sample_rn(cohort, selected_count)
  ds4.c:38586       ds4_sessions_eval_speculative_sample_rn
    per lane r:     DSpark draft (neural K) + HybridLC retrieval suffix
    ds4.c:40060     ds4_sessions_verify_suffix_rn_begin(verify, active_count, &txn)
      ds4.c:33174     spec_frontier_snapshot(&txn->frontier[r], session)   <- PER LANE
      ds4.c:33176     dspark_spec_ring_save(&session->graph, start, rows)  <- PER LANE
      ds4.c:33189     metal_graph_verify_suffix_rn(...)   mutates canonical state
                        ds4.c:19868  metal_graph_capture_prefix_attn_state(g, il, t)
                        ds4.c:20276  metal_graph_capture_prefix_index_state(g, il, t)
                        ds4.c:14199  prefix GVR-hint capture inside indexer Top-K
    ds4.c:40067     ds4_sessions_verify_suffix_rn_reject(txn, rejection)   acceptance
    ds4.c:40077     keep_rows[a] = 1 + committed_drafts
    ds4.c:40079     ds4_sessions_verify_suffix_rn_finish(txn, keep_rows)
      ds4.c:33489     keep == rows  -> `continue`, NOTHING is undone
      ds4.c:33489     keep <  rows  -> dspark_spec_ring_restore(start, keep, rows)
      ds4.c:33498                      spec_frontier_commit_prefix(session, keep)
      ds4.c:33495     keep == 0     -> spec_frontier_restore   (unreachable, see below)
    ds4.c:33206/33507 error paths   -> physical_verify_txn_restore (full rollback)
```

The single-session twin, used by the CLI/bench API, is
`ds4_session_eval_dspark_cycle` ([ds4.c:36281](../ds4.c#L36281), snapshot at
36847, commit at 36988, restore at 37141). It is structurally identical; every
number below applies to both, per lane.

**Two findings fall straight out of the control flow.**

1. **Full accept already keeps the final state.** Both `rn_finish`
   ([ds4.c:33489](../ds4.c#L33489)) and the single-session path
   ([ds4.c:36982](../ds4.c#L36982)) skip commit and restore entirely when
   `keep == rows`. The hypothesis's claim that legacy always rolls back is
   **false for the full-accept case**.
2. **`spec_frontier_restore` is unreachable on the normal partial/reject
   path.** `keep = 1 + committed_drafts` is always ≥ 1, because row 0 is the
   current token and is never rejected. So the full baseline restore fires only
   on a *hard verifier failure* (CUDA launch failure, tensor read failure,
   HybridLC self-check failure). The 11.68 MiB baseline snapshot taken on
   **every** cycle is insurance against an error that essentially never
   happens.

---

## 2. CURRENT ROLLBACK MECHANISMS

| mechanism | file:function | bytes | copies | canonical or temp | when |
|---|---|---|---|---|---|
| `spec_frontier_snapshot` | [ds4.c:30340](../ds4.c#L30340) | **12 249 088 (11.682 MiB)** | **145** | reads canonical → writes `spec_*` temp | **every cycle, every lane** |
| `dspark_spec_ring_save` | [ds4.c:35787](../ds4.c#L35787) | 36 864 (K=5) | 3 | reads canonical ring → temp backup | **every cycle, every lane** |
| prefix capture (attn) | [ds4.c:14112](../ds4.c#L14112) | 2 × attn tensor per layer per prefix | 2/layer/prefix | reads canonical → `spec_prefix_*` temp | **every draft token, inside the verifier** |
| prefix capture (index) | [ds4.c:14125](../ds4.c#L14125) | 2 × index tensor per r4 layer per prefix | 2/layer/prefix | temp | **every draft token** |
| prefix capture (GVR) | [ds4.c:14138](../ds4.c#L14138) / [:14199](../ds4.c#L14199) | 2 048 per r4 layer per prefix | 1/layer/prefix | temp | **every draft token** |
| `spec_frontier_commit_prefix` | [ds4.c:30424](../ds4.c#L30424) | **12 249 088** | **145** | **writes canonical** | partial accept **and** reject only |
| `dspark_spec_ring_restore` | [ds4.c:35801](../ds4.c#L35801) | `(rows-keep) × 3 × 2 048` | 3 | **writes canonical** | partial / reject / failure |
| `spec_frontier_restore` | [ds4.c:30381](../ds4.c#L30381) | **12 249 088** | **145** | **writes canonical** | hard failure only |

**Small DSpark ring backup vs compressor/indexer snapshots — kept strictly
separate.** The DSpark raw ring backup is 3 blocks × `verify_rows` rows ×
512 f32 = **36 KiB at K=5**, three kernel launches
([ds4_cuda.cu:16203](../ds4_cuda.cu#L16203)). It is *0.3 %* of a frontier
snapshot. The compressor/indexer snapshot is the 11.68 MiB item. Conflating
them would understate the problem by ~330×.

Raw target SWA rows are **not** backed up at all, by design — the comment at
[ds4.c:14105](../ds4.c#L14105) explains that the raw ring is larger than the
128-token logical SWA window, so speculative future rows are position-addressed
garbage that no reader can see.

---

## 3. FRONTIER SIZE

Layer layout from `ds4_expected_layer_compress_ratio`
([ds4.c:645](../ds4.c#L645)): FLASH → layers 0–1 ratio 0, even ≥2 → ratio 4,
odd ≥3 → ratio 128. Tensor shapes from
[ds4.c:11496–11557](../ds4.c#L11496): `coff = ratio==4 ? 2 : 1`,
`attn_width = coff·512`, `attn_rows = coff·ratio`,
`index_width = coff·128`, `index_rows = coff·ratio`, GVR = 512 × u32.

```
layers                : 43 total -> 21 ratio-4, 20 ratio-128, 2 uncompressed

ratio-4   per layer   :     83 968 B   (attn 65 536 + index 16 384 + gvr 2 048)   5 copies
ratio-128 per layer   :    524 288 B   (attn only)                                2 copies

ratio-4   total       :  1 763 328 B =  1.682 MiB   105 copies
ratio-128 total       : 10 485 760 B = 10.000 MiB    40 copies
  attention compressor: 11 862 016 B
  indexer compressor  :    344 064 B
  GVR / hints         :     43 008 B

ONE FULL FRONTIER     : 12 249 088 B = 11.682 MiB   145 copies
DSpark ring backup    :     36 864 B (3 x 6 rows x 512 f32)   3 copies
```

**The 20 ratio-128 layers are 86 % of the frontier** while holding a 128-row
window that a 6-row speculative block barely touches.

---

## 4. K=5 COST

Derived from the shapes above and confirmed by the harness
(`tests/test_dspark_rollback.c`, 400 deterministic fixtures):

```
                          full accept      partial       reject      hard failure
baseline snapshot         12 249 088     12 249 088   12 249 088     12 249 088
prefix captures (K=5)     61 245 440     61 245 440   61 245 440     61 245 440
commit prefix                      0     12 249 088   12 249 088              0
restore                            0              0            0     12 249 088
DSpark ring save              36 864         36 864       36 864         36 864
DSpark ring restore                0     <= 30 720       30 720         36 864
------------------------------------------------------------------------------
TOTAL bytes               73 531 392     85 800 000   85 811 200    85 817 344
TOTAL MiB                      70.12          81.83        81.84         81.84
TOTAL copies                     873          1 021        1 021         1 021
```

`reject` here means "only the current token survives" (`keep = 1`); it takes the
same `commit_prefix` path as any other partial accept, not the restore path.

### Why this is the wrong shape

Per accepted/rejected token, the compressor kernels write **one row**:
`compressor_store_kernel` ([ds4_cuda.cu:11659](../ds4_cuda.cu#L11659)) writes
`dst_row = ratio==4 ? ratio + pos%4 : pos%128` — 2 048 B in a ratio-128 layer,
4 096 B in a ratio-4 layer, plus a 32 KiB shift on the 1-in-4 emitting token
(`compressor_shift_ratio4_kernel`, [ds4_cuda.cu:11820](../ds4_cuda.cu#L11820)).

```
true state delta, one token, all 43 layers + ring : ~346 KiB
bytes copied to protect 6 tokens                  : 73.5 MiB  (full accept)
true delta of those 6 tokens                      : ~2.1 MiB
copy amplification                                : ~35x
per ratio-128 layer, per prefix capture           : 524 288 B copied for 4 096 B changed = 128x
```

Every prefix capture copies a whole 128-row window to record a single new row.

### K sweep (harness, per-cycle bytes)

| K | A full | A partial | B full | B partial | C full | C partial |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24.5 MB | 36.8 MB | 12.3 MB | 24.9 MB | 1.29 MB | 2.25 MB |
| 2 | 36.8 MB | 49.0 MB | 12.3 MB | 25.3 MB | 1.95 MB | 2.91 MB |
| 3 | 49.0 MB | 61.3 MB | 12.3 MB | 25.2 MB | 2.59 MB | 4.47 MB |
| **5** | **73.5 MB** | **85.8 MB** | **12.3 MB** | **25.9 MB** | **3.83 MB** | **6.23 MB** |
| 8 | 110.3 MB | 122.6 MB | 12.3 MB | 26.9 MB | 5.78 MB | 8.80 MB |
| 15 | 196.1 MB | 208.4 MB | 12.3 MB | 28.8 MB | 10.35 MB | 15.71 MB |

A grows linearly in K at 12.25 MB per extra draft token. This is the ceiling on
how deep HybridLC drafting can usefully go under the current design.

---

## 5. EXPERIMENT A — LEGACY

`tests/test_dspark_rollback.c`, mode A, reproduces the exact lifecycle: full
baseline snapshot, `dspark_spec_ring_save`, K prefix captures interleaved into
the per-token compressor loop at the same place the real code puts them, then
`commit_prefix` / `ring_restore` on partial accept and full restore on hard
failure. The per-token write pattern, the ratio-4 shift, the emit condition
`(pos+1) % ratio == 0` and the `n_comp` / `n_index_comp` advance are transcribed
from the CUDA kernels, so the *which rows change and when* is exact even though
the arithmetic values are stand-ins.

## 6. EXPERIMENT B — SNAPSHOT + REPLAY

Mode B: one baseline snapshot, **no prefix captures**. Full accept keeps the
final state. Partial accept restores the baseline and replays only the accepted
prefix. Failure restores the baseline.

Mode B is not enabled anywhere in the engine. It exists only inside the
standalone harness; `ds4.c` behaviour is unchanged.

### Mode C — delta journal (mutate-on-commit)

Added because B is not the interesting alternative once the write pattern is
known. Mode C takes **no snapshot and no captures**: each speculative write is
journalled with its pre-image, and a partial accept undoes the journal tail.
This prices the "separate speculative frontier / publish only the accepted
prefix" architecture honestly, including its own bookkeeping.

---

## 7. CORRECTNESS

400 deterministic fixtures, K=5, mixed full / partial / hard-failure outcomes,
comparing byte-exact state after every cycle:

```
accept count                : identical (decision computed from verifier
                              logits before any frontier bookkeeping)
correction-token decision   : identical
canonical frontier          : EQUAL (byte-exact)
n_comp                      : EQUAL
n_index_comp                : EQUAL
compressor recurrent state  : EQUAL (byte-exact)
indexer recurrent state     : EQUAL (byte-exact)
GVR hints + validity        : EQUAL (byte-exact)
DSpark frontier ring        : EQUAL (byte-exact)
visible compressed rows     : EQUAL (rows past n_comp treated as invisible
                              garbage, exactly as the legacy does)

RESULT: PASS  for both B and C
```

Draft tokens, output token and RNG are **not** exercised: they live in CUDA
kernels (`ds4_gpu_dspark_rejection_verify_tensor`,
`ds4_gpu_dspark_block_verify_tensor`) and in `sample_top_p_min_p`, and none of
the three modes touch sampling or acceptance — the state-management change is
downstream of the decision. This is a stated gap, not a pass.

The gate is live: an early harness bug produced
`C NOT SEMANTICALLY EQUIVALENT ... compressed_cache_visible_rows` and had to be
fixed before the run above.

---

## 8. HOST MICROBENCHMARK

Mac host, 200 repetitions per outcome, `clock_gettime(CLOCK_MONOTONIC)`,
buffers at the real derived sizes. Median / p90 in ms.

```
outcome          A med    A p90     B med    B p90     C med    C p90
full accept     1.9920   2.1960    0.6430   0.7030    0.3820   0.4550
partial         2.6550   2.7350    1.1020   1.1790    0.4660   0.5270
hard failure    2.6860   2.7650    1.0370   1.0980    0.5140   0.5650

bytes copied (400-cycle run)      A 30 776 MiB    B 7 717 MiB    C 1 980 MiB
copy count                        A    383 684    B    94 356    C   462 648
```

C moves **15.5× less data than A** but issues **more, much smaller** operations.
On the GB10 that trade is launch-bound, not bandwidth-bound, unless the journal
undo is batched into one row-list kernel — the same shape as the existing
`ring_rows_restore_kernel`. That is the single design constraint C imposes.

**These are host memcpy timings. They measure copy amplification only. They are
not a GB10 decode tokens/s estimate.**

---

## 9. BREAK-EVEN

Expected bytes per cycle, K=5, hard-failure held at 2 %, remainder split between
full and partial accept:

```
full-accept rate    E[A] MiB    E[B] MiB    E[C] MiB    winner
      0%              81.82       24.74        5.83       C
     20%              79.48       22.13        5.41       C
     40%              77.14       19.52        4.99       C
     60%              74.81       16.91        4.56       C
     80%              72.47       14.30        4.14       C
    100%              71.76       12.19        3.83       C
```

**There is no break-even.** A is never the cheapest at any acceptance
distribution, for any K, for any average accepted prefix length. The crossover
question the hypothesis expected does not exist, because A's dominant cost —
`(1 + K) × 11.68 MiB` of snapshot and captures — is paid **before** the
acceptance outcome is known and is therefore independent of the distribution.
Acceptance rate only moves the small tail (the 11.68 MiB commit).

Caveat on B: its partial-accept cost above **excludes the target forward pass**
its replay would need on the GB10. That is exactly the cost the legacy comment
at [ds4.c:30414](../ds4.c#L30414) says the prefix captures were introduced to
avoid, and that reasoning is sound. B is cheaper than A in bytes but buys it
with recomputation; **C is cheaper than both without recomputing anything.**

---

## 10. NIGHTJAR / R=N

`ds4_sessions_verify_suffix_rn_begin` ([ds4.c:33174](../ds4.c#L33174)) calls
`spec_frontier_snapshot` **once per request** and `dspark_spec_ring_save` once
per request; `ds4_sessions_verify_suffix_rn_finish`
([ds4.c:33487](../ds4.c#L33487)) runs one `ring_restore` + `commit_prefix` per
request; `physical_verify_txn_restore` ([ds4.c:33047](../ds4.c#L33047)) runs one
full restore per request on any failure. Lanes are distinct `ds4_session`s with
cloned private state ([ds4.c:12118](../ds4.c#L12118)), so nothing is shared.

Launcher: `max-r=3`, `lanes=3`, `hot-lanes=2`.

```
R    snapshots/cycle   restores/cycle   A bytes/cycle    A copies   C bytes/cycle   C copies
1          1                 1            80 677 637         959       5 191 588      1 157
2          2                 2           161 355 274       1 918      10 383 176      2 313
3          3                 3           242 032 911       2 878      15 574 764      3 470
```

This is a **per-lane** cost, not a per-token amplification — each lane is doing
distinct useful work. But the waste scales linearly with the hot-lane count, so
at the configured `hot-lanes=2` the engine pays ~154 MiB/cycle of frontier
traffic of which ~140 MiB is avoidable. Nightjar was not modified.

---

## 11. RECOMMENDATION FOR FUTURE ATHENA DECODE

**SEPARATE SPECULATIVE FRONTIER / MUTATE-ON-COMMIT.**

Not "snapshot + replay" — B trades bytes for recomputation and the legacy was
right to reject it. The evidence points past both options in the original
hypothesis.

Classification requested in STEP 7:

**Unavoidable copies.** The recurrent compressor window row and the indexer
window row written per token (~346 KiB/token across all 43 layers, including
the ratio-4 shift on emitting tokens and the 3 × 512 f32 DSpark ring rows).
These are the model's actual state transition; nothing can remove them.

**Copies caused purely by mutate-then-rollback.** Everything else:
- the 11.68 MiB baseline snapshot, taken every cycle for an error path that
  `keep ≥ 1` makes unreachable in normal operation;
- the K × 11.68 MiB prefix captures, of which at most one is ever consumed and
  only on a partial accept;
- the 11.68 MiB `commit_prefix` write-back, which restores a full frontier to
  rewind at most K rows.

At K=5 that is **~71 MiB of the 73.5 MiB full-accept cost**, and ~83 MiB of the
85.8 MiB partial-accept cost.

**Append-only state that can stay as invisible garbage.** The compressed KV
cache rows past `n_comp`, the indexer compressed rows past `n_index_comp`, and
the raw target SWA rows. The legacy already exploits this correctly — the
comment at [ds4.c:14105](../ds4.c#L14105) and the `n_comp` rewind in
`commit_prefix` are the right design and the harness confirms A, B and C agree
on every visible row while disagreeing on garbage. **Keep this.**

**Recurrent state that must represent the accepted prefix exactly.** The
per-layer `attn_state_kv` / `attn_state_score` window, the ratio-4
`index_state_kv` / `index_state_score` window, the GVR hint plus its validity
flag, and `n_comp` / `n_index_comp`. This is the only state that genuinely needs
a rollback mechanism — and it is a *sparse, bounded, statically known* set of
rows per token, not a set of whole tensors.

Concretely, for Athena:

1. Keep append-only caches append-only, with a visible-row counter. Garbage past
   the counter costs nothing.
2. Do **not** snapshot whole recurrent windows. Journal the pre-image of the
   rows a speculative token overwrites — the write pattern is fully determined
   by `pos % ratio`, so the row list can be computed ahead of the verifier and
   the journal can be a fixed-size pre-allocated arena, not a `malloc` per row.
3. Publish on accept, undo the journal tail on partial accept, undo the whole
   journal on failure. Full accept costs zero undo — as it already does today.
4. Batch the undo into one row-list kernel (the existing
   `ring_rows_restore_kernel` shape), so C's higher operation count does not
   become launch-bound.
5. This makes the cost proportional to **K × rows-actually-written**, i.e.
   O(K · 346 KiB) instead of O((K+1) · 11.68 MiB) — a 15–20× reduction that
   grows with K and with lane count.

If the future architecture keeps a canonical/speculative split, the split should
be at *row* granularity, not tensor granularity. The legacy's real mistake is
not that it rolls back; it is that its rollback unit is a whole 128-row
compressor window when the transaction only ever touches K ≤ 15 rows of it.

---

## 12. IMPORTANT LIMITATION

**NO CUDA/GB10 benchmark was performed.** Every timing in this document is a
host `memcpy` measurement on a Mac, using buffers sized from the legacy shapes.
They quantify *copy amplification only*.

Do not claim any decode tokens/s improvement from these measurements. The GB10
cost of these copies depends on unified-memory bandwidth, kernel launch
overhead, CUDA-graph capture behaviour (`DS4_CUDA_DSPARK_GRAPH=1` in the
launcher folds the verifier into a graph, which changes launch accounting) and
stream overlap — none of which is modelled here. A GB10 A/B is required before
any throughput claim.

Also not covered: draft-token, output-token and RNG equivalence, which live in
CUDA kernels and could not be exercised host-side.

---

## 13. FILES CHANGED

| file | change |
|---|---|
| `tests/test_dspark_rollback.c` | **new** — standalone host-only audit harness (modes A/B/C, correctness gate, microbenchmark, break-even, R=n) |
| `Makefile` | **new target** `test-dspark-rollback` (not wired into `make test`); added the binary to `clean` |
| `.gitignore` | ignore `tests/test_dspark_rollback` |
| `docs/DSPARK_ROLLBACK_AUDIT.md` | **new** — this report |

**`ds4.c`, `ds4_cuda.cu`, `ds4_server.c` and every production path are
unmodified.** No experimental flag was added to the engine; mode B and mode C
exist only inside the standalone harness, so the legacy default behaviour is
unchanged and unchangeable from the engine's configuration surface.

Pre-existing uncommitted changes in the working tree (`ds4.c`,
`tests/ds4_test.c`, streaming-expert-hotlist range rejection) are unrelated to
this audit and were left untouched.

Reproduce with:

```
make test-dspark-rollback
./tests/test_dspark_rollback --k=15 --iters=400
```
