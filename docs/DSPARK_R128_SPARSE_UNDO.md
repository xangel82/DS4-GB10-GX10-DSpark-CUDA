# DSpark ratio-128 sparse undo optimization

Milestone: replace the ratio-128 half of the speculative frontier machinery
with a sparse per-row undo journal, behind an experimental flag that is OFF by
default. Ratio-4, the indexer, GVR and the DSpark ring are untouched.

No CUDA was compiled or run; no GB10 was used. Host verification is on macOS.

---

## 1. PRODUCTIVE FLOW VERIFIED

Re-checked against the current tree, not against the previous audit.

```
ds4_server.c:11427   ds4_sessions_eval_speculative_sample_rn(cohort, selected_count)
  ds4.c:39038          per lane: DSpark neural draft + HybridLC retrieval suffix
  ds4.c:40512          ds4_sessions_verify_suffix_rn_begin(verify, active_count, &txn)
    ds4.c:33624          spec_frontier_snapshot_ex(&txn->frontier[r], session, true)
    ds4.c:33627          dspark_spec_ring_save(&session->graph, start, rows)
    ds4.c:33637          metal_graph_verify_suffix_rn(...)
      ds4.c:25081          metal_graph_spec_r128_journal_begin(state, start, rows)  [NEW]
      ds4.c:20080/20488    per-token compressor loop + prefix captures
  ds4.c:40520          ds4_sessions_verify_suffix_rn_reject(txn, rejection)
  ds4.c:40525          keep_rows[a] = 1 + committed_drafts
  ds4.c:40528          ds4_sessions_verify_suffix_rn_finish(txn, keep_rows)
    ds4.c:33937          keep == rows -> continue (nothing undone)
    ds4.c:33938          keep <  rows -> dspark_spec_ring_restore + spec_frontier_commit_prefix
  ds4.c:33494          error paths -> physical_verify_txn_restore -> spec_frontier_restore
```

The single-session twin `ds4_session_eval_dspark_cycle` (ds4.c:36730, snapshot
at 37299, commit at 37440, restore at 37599) is structurally identical and gets
the same treatment.

Launcher flags that select this path are unchanged:
`--dspark-draft 5`, `DS4_DSPARK_HYBRID_LC=1` (exported by the default branch,
raising `spec_prefix_cap` and `dspark_verify_cap` to 15/16),
`DS4_DSPARK_NIGHTJAR=1`, `DS4_DSPARK_NIGHTJAR_MAX_R=3`,
`DS4_SERVER_DSPARK_LANES=3` / `HOT_LANES=2`, `DS4_CUDA_DSPARK_GRAPH=1`,
with `DS4_CUDA_MTP_GRAPH` and `DS4_CUDA_TOKEN_GRAPH_PIPELINE` left **unset**
(presence-tested, so `unset` and `=0` are not interchangeable).

### The per-token write pattern, re-derived from the current code

During a speculative verify `g->spec_capture_prefixes != 0`, which disables
both bulk-replay branches (ds4.c:19774 and ds4.c:19892) and forces the
**per-token** compressor loop. That loop calls
`ds4_gpu_compressor_update_tensor` once per token; for ratio 128 it falls
through to `ds4_gpu_compressor_store_batch_tensor(..., n_tokens = 1)` and
`compressor_store_kernel` (ds4_cuda.cu:11659):

```c
uint32_t pos_mod = (pos0 + t) % ratio;
uint32_t dst_row = ratio == 4u ? ratio + pos_mod : pos_mod;
state_kv   [dst_row * width + j] = kv[t * width + j];
state_score[dst_row * width + j] = sc[t * width + j] + ape(pos_mod, j);
```

Confirmed for ratio 128: `coff = 1`, `width = 512`, `state_rows = 128`,
`dst_row = pos % 128`, and **no** `compressor_shift_ratio4_kernel` (that shift
is guarded by `ratio == 4u`, ds4_cuda.cu:16546).

Components actually mutated by a ratio-128 token:

| component | mutated | note |
|---|---|---|
| `layer_attn_state_kv` row `pos % 128` | yes | 512 f32 = 2 048 B |
| `layer_attn_state_score` row `pos % 128` | yes | 512 f32 = 2 048 B |
| `layer_n_comp` | on emit only | `(pos+1) % 128 == 0`; **derivable** |
| `layer_attn_comp_cache[n_comp]` | on emit only | append-only, invisible past `n_comp` |
| `layer_index_state_*`, `layer_gvr_hint`, `n_index_comp` | **no** | ratio-4 only |

This matches what `spec_frontier_snapshot` itself copies for ratio 128 (only
the two attention state tensors plus the counters), so the mutable set is
complete.

---

## 2. RATIO-128 MUTATION UNIT

```
row bytes            : 2 048 B state_kv + 2 048 B state_score = 4 096 B
rows touched / token : 1 per ratio-128 layer, at (pos0 + t) % 128
layers               : 20 (odd layers 3..41 of DeepSeek-V4-Flash)
total real delta/tok : 81 920 B
legacy copy / token  : 10 485 760 B  ->  128x write amplification
```

Speculative rows are `(pos0 + t) % 128` for `t < n_rows <= 16`, therefore
**pairwise distinct**. No row is written twice inside a cycle, so the journal
never aliases and a single batched suffix restore is exact regardless of order.

---

## 3. LEGACY A COST

Per cycle, per lane, measured by the harness at the real shapes:

```
                    K=5                          K=15
ratio-128 bytes     62 914 560  (full)           167 772 160  (full)
                    73 400 320  (partial)        178 257 920  (partial)
ratio-4   bytes     10 579 968 / 12 343 296       28 213 248 / 29 976 576
ring      bytes         36 864 /     55 325           98 304 /    147 712
------------------------------------------------------------------------
total bytes         73 531 392 / 85 798 941      196 083 712 / 208 382 208
total MiB                70.12 /      81.82           187.00 /     198.73
copies                     873 /       1 021             2 323 /      2 471
```

Ratio-128 is **85.6 %** of the K=5 cycle and **85.5 %** at K=15.

---

## 4. SPARSE JOURNAL DESIGN

**Descriptor.** There is no per-row descriptor array. The touched row set is a
closed form of `(pos0, n_rows, ratio)`, so the journal needs only three scalars
plus the bound layer list — the descriptor collapses into kernel arguments.
That is what keeps this to one launch instead of a descriptor sweep.

**Payload.** One preallocated device tensor, row-major and fully coalesced:

```
journal[(((l * n_rows) + t) * 2 + half) * width + j]
  l    dense ratio-128 layer index (0..19)
  t    speculative row (0..n_rows-1)
  half 0 = state_kv before-image, 1 = state_score before-image
  j    0..511
```

Sized once at graph allocation for `dspark_verify_cap` rows:
`20 x 16 x 2 x 512 x 4 = 1 310 720 B (1.25 MiB)`. Live use at K=5 is
`20 x 6 x 2 x 512 x 4 = 491 520 B`. **No hot-path allocation.**

A second preallocated tensor holds the device pointer table
(`table[l]` = layer `l` `state_kv`, `table[n_layers + l]` = `state_score`),
bound once by `ds4_gpu_spec_r128_journal_bind` at graph and lane allocation.

**Capture.** `metal_graph_spec_r128_journal_begin` (ds4.c:25948) issues **one**
kernel, `spec_r128_journal_capture_kernel`, over
`n_layers x n_rows x 2 x width` threads, before the verifier touches any state.
It is encoded inside the caller's existing command region, so it adds **no
synchronisation** and is recorded into the DSpark CUDA graph like every other
verifier node. It is re-issued in the graph-capture-abort fallback
(ds4.c:26105), because an aborted capture never executed its recorded work.

**Undo kernel.**

```
spec_r128_journal_undo_kernel(state_table, journal, n_layers,
                              pos0, keep_rows, n_rows, ratio, width)
```

restores rows `[keep_rows, n_rows)` in **one** launch;
`ds4_gpu_spec_r128_journal_undo` returns immediately when `keep == n_rows`, so
a full accept issues nothing at all. Called from
`spec_frontier_commit_prefix` (rejected suffix) and `spec_frontier_restore`
(whole suffix). No per-row, per-token or per-layer launches anywhere.

---

## 5. PRODUCTION CHANGES

**Experimental flag:** `DS4_DSPARK_R128_UNDO_JOURNAL` — presence-tested, so
`unset` and `=0` both select the legacy path. **OFF by default.**
Optional accounting: `DS4_DSPARK_R128_JOURNAL_STATS`, also OFF by default.
Launcher wires both with the same presence semantics as the surrounding flags
(`run-dspark-server.sh:96,219,391`) and prints the selected mode.

The flag is read once and cached, so the mode cannot change between a snapshot
and its matching commit/restore, and a captured CUDA graph keeps a stable node
topology.

| file | function | change |
|---|---|---|
| `ds4_cuda.cu` | `spec_r128_journal_capture_kernel` | **new** — batched before-image capture |
| | `spec_r128_journal_undo_kernel` | **new** — batched suffix restore |
| | `ds4_gpu_spec_r128_journal_bind` | **new** — one-time device pointer table |
| | `ds4_gpu_spec_r128_journal_capture` / `_undo` | **new** — bounds-checked launchers |
| `ds4_gpu.h` | — | declarations |
| `ds4.c` | `ds4_gpu_graph` | **new fields** journal, table, layer list, rows/pos0, base `n_comp`, valid flag |
| | `dspark_r128_journal_enabled` / `_stats_enabled` | **new** — cached presence tests |
| | `dspark_r128_derived_n_comp` | **new** — counter reconstruction |
| | `metal_graph_alloc_raw_cap` | alloc + bind (CUDA only) |
| | `metal_graph_alloc_shared_lane` | **per-lane** alloc + bind |
| | `metal_graph_spec_r128_journal_begin` | **new** — one capture per lane per cycle |
| | `metal_graph_verify_suffix_tops` | hook + graph-fallback re-issue |
| | `metal_graph_verify_suffix_rn` | hook, one per request |
| | `spec_frontier_snapshot_ex` | **new** — opt-in skip of the ratio-128 window |
| | `spec_frontier_snapshot` | thin legacy wrapper (unchanged behaviour) |
| | `spec_frontier_restore` | batched full undo when armed |
| | `spec_frontier_commit_prefix` | batched suffix undo + derived `n_comp` |
| | `metal_graph_capture_prefix_attn_state` | skip the ratio-128 window when journalling |
| `run-dspark-server.sh` | — | flag, validation, banner |

### Safety property: mode D is opt-in per call site

`spec_frontier_snapshot` keeps the legacy signature and **always** does the
full copy. Only `spec_frontier_snapshot_ex(..., allow_r128_journal = true)`
may skip the ratio-128 window, and it is called from exactly two places:
`ds4_sessions_verify_suffix_rn_begin` and `ds4_session_eval_dspark_cycle` —
the only two paths guaranteed to run a journalling verifier. The MTP
decode-2-exact path (ds4.c:41313) and the argmax speculative paths keep mode A
unconditionally, so a caller can never arm a rollback it has no journal for.

The decision travels with the frontier (`ds4_spec_frontier.r128_journal`), and
the journal is only usable once `spec_r128_journal_valid` is set by a
**successful** capture that runs before any mutation. If the verifier fails
earlier than that, nothing was mutated and skipping the ratio-128 restore is
correct. The journal is retired only after its undo has actually been issued,
so a failed commit still falls through to a working full restore.

### Counters (task item 7)

`n_comp` for a ratio-128 layer is a pure function of the baseline counter, the
block start and the accepted prefix:

```c
n_comp(keep) = base + #{ t in [0, keep) : (pos0 + t + 1) % 128 == 0 }
```

Mode D therefore journals **no counters at all** — not one copy, let alone K.
`dspark_r128_derived_n_comp` computes it in `spec_frontier_commit_prefix`.
`spec_prefix_n_comp` is still written host-side (a `uint32_t` store, no copy
operation) purely so `DS4_DSPARK_R128_JOURNAL_STATS` can cross-check derived
against captured and log any divergence. No other ratio-128 metadata exists
that is not reconstructible, so nothing else is journalled.

---

## 6. CORRECTNESS A/D

`tests/test_dspark_rollback.c`, extended from the previous audit harness.
Outcomes are scheduled exhaustively rather than sampled: every cycle class
(full accept, partial with A = 0..K-1 accepted drafts, complete-draft reject,
hard failure) is exercised in rotation.

```
K=1   600 fixtures   full=200 partial=200 failure=200
                     keep coverage: 0:200  1:200  2:200
K=5   600 fixtures   full=86  partial=429 failure=85
                     keep coverage: 0:85 1:86 2:86 3:86 4:86 5:85 6:86
K=15  600 fixtures   full=36  partial=529 failure=35
                     keep coverage: 0:35 ... 16:36  (all 17 classes hit)
```

Compared byte-for-byte after **every** cycle:

```
ratio-128 compressor KV        EQUAL
ratio-128 compressor score     EQUAL
n_comp                         EQUAL   (derived, never journalled)
ratio-4 attn state             EQUAL
indexer state                  EQUAL
n_index_comp                   EQUAL
GVR hints + validity           EQUAL
DSpark ring                    EQUAL
visible compressed rows        EQUAL   (garbage past n_comp ignored, as legacy)
whole-graph FNV digest         EQUAL

accept count                   identical -- computed from verifier logits
correction-token decision      identical -- mode D is downstream of it
RNG                            not exercised (CUDA-side, see section 13)
```

**PASS** at K=1, K=5 and K=15. Modes B and C from the previous audit were
re-run at K=5 and still match, so the earlier result did not regress.

---

## 7. COPY REDUCTION

Per cycle, per lane, mean over the fixture classes.

```
                          LEGACY A          JOURNAL D        REDUCTION
K=5  full accept
  ratio-128 bytes         62 914 560            491 520          128.0x
  cycle total bytes       73 531 392         11 108 352            6.6x
  cycle total MiB              70.12              10.59
K=5  partial / reject
  ratio-128 bytes         73 400 320            737 662           99.5x
  cycle total bytes       85 798 941         13 136 283            6.5x
  cycle total MiB              81.82              12.53
K=15 full accept
  ratio-128 bytes        167 772 160          1 310 720          128.0x
  cycle total bytes      196 083 712         29 622 272            6.6x
  cycle total MiB             187.00              28.25
K=15 partial / reject
  ratio-128 bytes        178 257 920          1 969 487           90.5x
  cycle total bytes      208 382 208         32 093 774            6.5x
  cycle total MiB             198.73              30.61
```

Of the 11.682 MiB frontier, the **10.000 MiB ratio-128 half is eliminated** as
a copy unit and replaced by 0.469 MiB of journal at K=5 (1.25 MiB at K=15),
paid once per cycle instead of `1 + K` times.

Hard failure: A 81.84 MiB -> D 12.78 MiB (ratio-128 74.7x less at K=5).

---

## 8. COPY / KERNEL COUNT

```
                       K=5                        K=15
                    A        D                 A        D
full accept       873      634              2 323    1 684
partial         1 021      743              2 471    1 793
hard failure    1 021      743              2 471    1 793

ratio-128 copy operations per cycle:
  A  full     : 145 x ... -> 40 per snapshot/capture, 240 total at K=5
  D  full     : 1 capture kernel, 0 undo kernels
  D  partial  : 1 capture kernel + 1 undo kernel
  D  failure  : 1 capture kernel + 1 undo kernel
journal entries (rows recorded, one batched kernel): 120 at K=5, 320 at K=15
restore launches for ratio-128:  A 20 layers x 2 tensors ; D exactly 1
```

Demonstrating the invariants the task asked for explicitly:

```
full accept : 0 ratio-128 prefix copies          [ratio-128 bytes = capture only]
partial     : 0 ratio-128 prefix commit copies   [undo kernels = 1]
              1 batched undo of the rejected suffix
failure     : 1 batched undo of the full speculative suffix
```

The remaining D copy count is entirely ratio-4: 21 layers x 5 copies x
(1 snapshot + K prefixes) + 3 ring + 1 journal.

---

## 9. NIGHTJAR / R=n

`ds4_sessions_verify_suffix_rn_begin` (ds4.c:33534) still snapshots and
`ds4_sessions_verify_suffix_rn_finish` (ds4.c:33875) still commits/restores
**per request**; that structure is unchanged. What changed is that each
lane now carries its **own** journal payload and its **own** device pointer
table, allocated and bound in `metal_graph_alloc_shared_lane` against the
lane's own `layer_attn_state_*`. Nothing is shared with the owner, and the
capture in `metal_graph_verify_suffix_rn` runs once per request, so mode D does
not reintroduce a per-lane full snapshot by another route.

Nightjar scheduling, lane selection, coalescing and acceptance were not
touched.

```
R    A bytes/cycle    D bytes/cycle    A/D
1        84 043 199       12 882 982   6.52x
2       168 086 398       25 765 963   6.52x
3       252 129 597       38 648 945   6.52x
```

At the configured `hot-lanes=2` the engine goes from ~160 MiB/cycle of frontier
traffic to ~25 MiB/cycle.

---

## 10. HOST MICROBENCHMARK

Mac host, 400 repetitions per cell, real buffer sizes, `CLOCK_MONOTONIC`.

```
K    outcome         median ms   p95 ms      bytes moved      ops   mode
5    full accept        2.0040   2.4220       73 531 392      873   A
                        0.6570   0.9060       11 108 352      634   D
5    partial            2.7130   2.9160       85 798 912    1 021   A
                        0.8600   0.9810       13 135 872      743   D
5    hard failure       2.6490   2.9750       85 817 344    1 021   A
                        0.7650   1.0440       13 400 064      743   D
15   full accept        5.4080   6.2180      196 083 712    2 323   A
                        1.6620   2.0210       29 622 272    1 684   D
15   partial            5.8280   6.3830      208 381 952    2 471   A
                        1.7510   2.0750       32 090 112    1 793   D
15   hard failure       5.8290   6.4530      208 431 104    2 471   A
                        1.7910   2.0360       32 794 624    1 793   D
```

3.0x-3.3x less host time, 6.5x fewer bytes. **This is copy amplification only.
It is not a GB10 decode tokens/s estimate and must not be reported as one.**

---

## 11. EXPECTED REMAINING COST

K=5, mode D, mean bytes per cycle per lane:

```
ratio-4 legacy snapshot + K prefix captures + commit : 12 090 552 B  (93.8 %)
DSpark ring save/restore (legacy)                    :     55 286 B  ( 0.4 %)
ratio-128 journal (capture + undo)                   :    737 143 B  ( 5.7 %)
```

Ratio-4 is now the whole problem. Its structure is the same 8x-per-token
amplification (a token writes row `4 + pos%4` of an 8-row window, 8 192 B of a
65 536 B attn window plus 2 048 B of an 8 192 B index window), with the extra
wrinkle that the emitting token also runs `compressor_shift_ratio4_kernel`,
which rewrites the whole window — so a ratio-4 journal needs an emit-aware row
set rather than a single row. That is deliberately out of scope here.

---

## 12. RECOMMENDATION

**Conditionally PROMOTE D, pending GB10 validation.**

The host evidence is unambiguous on the axis it can measure: byte-exact
equivalence across 1 800 fixtures spanning every accept class at K=1/5/15,
128x less ratio-128 traffic, 6.5x less total frontier traffic, and a strictly
*lower* launch count than mode A (634 vs 873 at K=5) — mode D does not trade
bandwidth for launches, which was the main risk the design had to avoid.

What the Mac cannot answer, and what the GB10 A/B must:

1. Whether the extra capture node changes DSpark CUDA-graph behaviour —
   specifically whether `cudaGraphExecUpdate` still succeeds for each family,
   or whether the added node forces re-instantiation.
2. Whether the capture kernel's gather (`row = (pos0 + t) % 128`, coalesced
   along `width`) reaches the same effective bandwidth as the straight-line
   `ds4_gpu_tensor_copy` it replaces.
3. Whether removing ~60 MiB/cycle of unified-memory traffic actually converts
   into decode throughput, or whether the verifier is compute-bound at K=5.

Keep the flag OFF by default until those three are measured. Do not promote on
host numbers alone.

---

## 13. CUDA/GB10

**NOT EXECUTED / PENDING.**

The CUDA kernels and launchers were written and syntax-verified with a
standalone `clang++` parse of the new block, and the full non-Apple branch of
`ds4.c` was syntax-checked with `-U__APPLE__` (the only 4 errors are the
pre-existing `st_mtim` macOS/Linux difference at ds4.c:38246, unrelated to this
change). **`nvcc` was never invoked and no kernel has ever run.**

Not verified host-side, and therefore still open:

- draft-token, output-token and RNG equivalence (all CUDA-side)
- CUDA-graph capture/update behaviour with the added node
- real kernel bandwidth and launch overhead on GB10
- the multi-lane `metal_graph_verify_suffix_rn` recursion at `total_rows >= 6`,
  which re-enters per lane and therefore re-journals per lane — correct by
  construction, but unexercised

Instrumentation is already in place for that run, OFF by default and
sync-free: NVTX ranges `ds4/decode/dspark/r128-journal-capture`,
`.../r128-journal-undo-suffix` and `.../r128-journal-undo-full` (visible under
the existing `DS4_CUDA_NVTX` profiling gate), plus the
`DS4_DSPARK_R128_JOURNAL_STATS` counter dump every 256 cycles reporting cycle
counts by outcome, capture/undo kernel counts, journal bytes and the legacy
ratio-128 bytes/copies avoided.

---

## 14. NEXT POSSIBLE STEP

Only if the GB10 A/B justifies it: **ratio-4 sparse journal**. It is now 93.8 %
of the remaining traffic, but it is the harder half — the emit-time
`compressor_shift_ratio4_kernel` rewrites the entire 8-row window, the indexer
window and the GVR hint are also live, and the GVR hint is produced inside the
Top-K kernel rather than copied. Not started, as instructed.

---

## 15. FILES CHANGED

| file | change |
|---|---|
| `ds4_cuda.cu` | **+152** — two kernels and three launchers for the ratio-128 journal |
| `ds4_gpu.h` | **+37** — declarations and contract |
| `ds4.c` | **+467/-6** — graph state, alloc/lane bind, capture hooks, mode-aware snapshot/restore/commit/prefix-capture, derived `n_comp`, NVTX + stats |
| `run-dspark-server.sh` | **+26** — `DS4_DSPARK_R128_UNDO_JOURNAL` and `..._STATS`, validation, banner |
| `tests/test_dspark_rollback.c` | mode D, component-split accounting, exhaustive outcome schedule, A/D gate at K=1/5/15, p95 microbenchmark |
| `Makefile` | `test-dspark-rollback` now runs the A/D gate |
| `docs/DSPARK_R128_SPARSE_UNDO.md` | this report |

Untouched: Athena's Engine, draft generation, sampler/RNG, p/q acceptance,
correction-token selection, HybridLC and Nightjar policy, ratio-4 compressor,
indexer, GVR, DSpark ring save/restore, and every other kernel.

Pre-existing uncommitted changes in `ds4.c` (streaming-expert-hotlist range
rejection) and `tests/ds4_test.c` are unrelated and were left alone. Note that
the `tests/ds4_test.c` hunk does not currently compile (`make ds4_test` fails
on three errors in that pre-existing test), which is why the A/D gate is a
standalone target.

Reproduce:

```
make test-dspark-rollback
DS4_DSPARK_R128_UNDO_JOURNAL=1 DS4_DSPARK_R128_JOURNAL_STATS=1 ./run-dspark-server.sh   # GB10 only
```
