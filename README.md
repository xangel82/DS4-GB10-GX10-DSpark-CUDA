# DS4 GB10/GX10 DSpark CUDA

This repository is an experimental GB10/GX10-oriented fork of Salvatore
Sanfilippo's `antirez/ds4` project:

```text
https://github.com/antirez/ds4
```

Fork repository:

```text
https://github.com/xangel82/DS4-GB10-GX10-DSpark-CUDA
```

This checkout contains a GB10-oriented DS4 runtime for DeepSeek-V4-Flash with a
DSpark sidecar.  The goal is single-machine inference on NVIDIA GB10 with
speculative decoding enabled, preserving target-model sampling semantics.

## Why this fork exists

This fork documents a practical optimization effort on NVIDIA GB10/GX10 for
DeepSeek-V4-Flash local inference.  The work focused on measuring real
bottlenecks, implementing CUDA-side speculative decoding paths, and validating
performance with reproducible logs, scripts and analyzer tools.

The goal is not to be a generic inference framework.  It is a focused
engineering branch for one concrete target: making DeepSeek-V4-Flash run faster
on a single GB10/GX10 machine without changing the target model distribution.

## Engineering highlights

- Implemented DSpark sidecar loading and GGUF conversion support.
- Added true p/q speculative rejection sampling for DSpark drafts.
- Moved verifier-side p/q acceptance logic to CUDA.
- Added DSpark-specific CUDA Graph variants for drafter and verifier paths.
- Added GB10-oriented Tensor Core tiny-batch experiments.
- Added Q8 tiny-batch reuse and Q8/F16 hot-cache launch profiles.
- Added raw-GGUF routed-MoE MMQ prefill with paired IQ2_XXS gate/up, Q2_K
  down projection and a token-bounded stream-K schedule.
- Added token-tile HMMA attention for exact Top-K indexed and dense raw/mixed
  prefill batches, structurally excluding decode and the DSpark verifier.
- Added a compact 68-byte MXFP4 indexer cache and native SM121a block-scaled
  scoring, using token tiles for prefill and head tiles for 1-6 row verifier
  batches.
- Added exact Radix Top-512 for large prefill batches, an exact parallel
  4096-column chunk tree for small batches and exact one-row GVR dispatch.
- Added byte-neutral in-place SoA replacement for target routed-MoE weights,
  with a numerically equivalent Q8_K small-batch path for decode/verification
  and D2R/MMQ tiers for large prefill batches.
- Stored FP8-rounded compressed attention KV directly as F16 and consumed it
  from the stage-32 token-tile path without a persistent F32 duplicate.
- Added reproducible cold/append GB10 sweeps with DSpark decode, process-memory
  high-water marks and deterministic token hashes. The complete path has been
  validated end to end on Athena through 84k context.
- Added pipelined direct-I/O model upload and release of copied GGUF source
  pages to reduce startup time and host page residency.
- Added long-prefix KV reuse so repeated tool turns can prefill only the
  appended suffix when the canonical token prefix still matches.
- Added a client-visible context guard so frontends compact before the physical
  DS4 context is exhausted.
- Added reproducible run scripts, benchmark analyzers and release-oriented
  installation notes.

## What changed

- DSpark sidecar support for `DeepSeek-V4-Flash-DSpark`.
- Lossless speculative rejection sampling: drafts are sampled from `q`, target
  rows define `p`, acceptance is `min(1, p(x) / q(x))`, and rejected drafts are
  replaced from the positive residual `max(p-q, 0)`.
- GPU-side p/q verification: target logits, draft probabilities, acceptance
  uniforms and residual uniforms stay on CUDA; the host reads only compact
  verifier results and the continuation logits row.
- DSpark-specific CUDA Graph families for drafter and verifier K variants.
- GB10 launch profile with:
  - copied target model in device memory;
  - 12 GiB Q8->F16 hot cache;
  - Q8 tiny-batch reuse;
  - DSpark Tensor Core tiny-batch path enabled by default;
  - always-on DSpark drafting for a single active GB10 decode stream;
  - `balanced` memory profile, 8192-token prefill chunks and copied sidecar;
  - 131k physical context with an 85% advertised context guard;
  - 16 GiB default disk budget for persisted KV checkpoints.
- Append-prefill optimization for long chats: canonical KV checkpoints are
  retained near long stable prompt boundaries, so subsequent requests with the
  same exact prefix can resume from disk and process only the new tail. On tool
  canonicalization mismatches, eviction now protects the longest checkpoint
  reusable by the incoming prompt instead of falling back to an older anchor.
  Direct post-tool RAM continuation remains a follow-up in `README-GB10.md`.
- `/v1/models` now advertises both `context_length` and `max_input_tokens`;
  `max_input_tokens` reserves the configured completion budget so clients can
  compact before generation runs into the physical context ceiling.

The latest measured profile reached 787.06 prefill token/s for a cold
13,597-token prompt and 724.69 token/s while appending 41,674 tokens up to 55k
context. Full central 8192-token chunks remained between 790.72 and 761.57
token/s during that long append. DSpark decode reached 25.13 token/s at 55k
context and 24.39 token/s at 70k. Exact numbers vary with prompt, seed,
sampling and chunk-boundary tails.

## Measured GB10 results

These numbers are from the development machine used for this fork: one NVIDIA
GB10 running `DeepSeek-V4-Flash` plus the `DeepSeek-V4-Flash-DSpark` sidecar.
They are useful as a sanity check, not as a guaranteed benchmark.

| Profile | Result | Notes |
| --- | ---: | --- |
| Original CUDA path, before the GB10 work | ~13 t/s | Starting point observed during the first Athena runs. |
| CUDA Graph + fused compressor + Q8/F16 cache path | ~14.5-14.7 t/s | Stable non-speculative baseline. |
| MTP sidecar experiments | ~15.1 t/s | Worked, but verifier cost limited the gain. |
| DSpark exact-match verifier, early versions | ~13-14.5 t/s | Too many fallback/bypass cycles; not the final algorithm. |
| DSpark p/q rejection sampling, always drafting | ~16.8-17.6 t/s | First correct speculative sampling path with consistent gain. |
| DSpark p/q rejection + GPU verifier + Tensor Core tiny batches | ~18.2 t/s weighted decode | Earlier release milestone; chunks reached about 19 t/s. |
| Raw-GGUF MMQ MoE + token-bounded stream-K | 404.46 t/s prefill | Weighted baseline over `ctx=24576..81920`; decode remained 23.00 t/s at 83k. |
| Token-tile HMMA indexed + raw/mixed attention | 509.14 t/s prefill | Same 57,344-token interval, **+25.88%**; full 61,214-token request averaged 496.57 t/s. |
| Current full pipeline, cold 13.6k prompt | 787.06 t/s prefill | First 8192-token chunk reached 854.26 t/s; average through 12,288 tokens was 834.49 t/s. The total includes the smaller final tail. |
| Current full pipeline, append 13.6k -> 55.3k | 724.69 t/s prefill | 41,674 appended tokens; the four complete central chunks measured 790.72, 780.88, 777.00 and 761.57 t/s. |
| Current full pipeline, append 69.7k -> 84.6k | 664.92 t/s prefill | Main complete chunk reached 725.18 t/s; partial boundary chunks reduce the request average. |
| Current DSpark decode at 55k / 70k | 25.13 / 24.39 t/s | Measured over 454 and 501 generated tokens respectively, with the small-batch recovery paths active. |

Representative earlier DSpark analyzer output (kept as scheduler history):

```text
Fused verifier cycles:     1029
Ordinary/fallback cycles:  0
P/Q rejection cycles:      1029
Verifier acceptance:       53.18%
Mean verifier target rows: 4.956
Mean verifier draft time:  20.572 ms
Mean verifier target time: 148.339 ms
Mean fused cycle:          168.911 ms
Verifier-cycle throughput: 18.376 t/s
Weighted request decode:   18.274 t/s
```

The important qualitative result is not only the raw t/s number: the final path
uses true speculative rejection sampling, so accepted drafts preserve the target
model sampling distribution instead of using a lossy exact-match shortcut.

The earlier 509.14 t/s prefill comparison was position matched. Across its seven complete
8192-token chunks, every chunk improved by 23.25-26.52%; elapsed time for the
57,344-token interval fell from 141.78 to 112.63 seconds. The primary 80.76 GiB
model upload completed in 19.076 seconds at 4.23 GiB/s; the separate 10.70 GiB
sidecar copy took 69.999 seconds in that run. The newer figures above include
the MXFP4 scorer, exact shape-specific Top-K dispatch and split target-MoE
execution: large batches use the prefill tiers, while decode and speculative
verification use the Q8_K small-batch path.

## Install on GB10/GX10

Choose a stable checkout path.  The examples below use the `athena` user and
keep source code and model files under `/home/athena`, not under `/tmp`.

Recommended layout:

```text
/home/athena/DS4-GB10-GX10-DSpark-CUDA   # source checkout
/home/athena/ds4                         # model files and logs
/tmp/ds4-gb10-dspark-kv                  # disposable KV disk cache, default 16 GiB budget
```

### 1. Clone and compile

```bash
cd /home/athena
git clone https://github.com/xangel82/DS4-GB10-GX10-DSpark-CUDA.git
cd /home/athena/DS4-GB10-GX10-DSpark-CUDA
```

Install/build dependencies expected by this fork:

```bash
sudo apt update
sudo apt install -y build-essential git curl wget rsync python3
```

CUDA must already be installed and visible at `/usr/local/cuda`. Verify the
native GB10 toolchain, run the CUDA regression, then build the server:

```bash
cd /home/athena/DS4-GB10-GX10-DSpark-CUDA && /usr/local/cuda/bin/nvcc --version && make -B cuda-regression CUDA_ARCH=sm_121a && make -B cuda-spark-graph-sm121
```

The regression must end with `cuda long-context regression: OK`. It validates
large Radix Top-K, the exact small-batch chunk tree and GVR dispatch, packed
MXFP4 scoring, aligned-SoA D2R and Q8_K MoE parity, and both token-tile
attention variants. Stop an already running server first if memory is close to
the GB10 limit.

### 2. Prepare model directory

The release launcher expects this model directory by default:

```bash
mkdir -p /home/athena/ds4
```

The two model files expected by the default launcher are:

```text
/home/athena/ds4/ds4flash.gguf
/home/athena/ds4/DeepSeek-V4-Flash-DSpark-Q4K-Q8.gguf
```

The recommended Q2/imatrix target can be downloaded with the repository
helper:

```bash
cd /home/athena/DS4-GB10-GX10-DSpark-CUDA && DS4_GGUF_DIR=/home/athena/ds4 ./download_model.sh q2-imatrix && ln -sfn /home/athena/ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf /home/athena/ds4/ds4flash.gguf
```

Alternatively, copy or build a compatible DeepSeek-V4-Flash GGUF as:

```text
/home/athena/ds4/ds4flash.gguf
```

If your main model has a different name or path, either rename/symlink it to
`/home/athena/ds4/ds4flash.gguf`, or launch with:

```bash
DS4_MODEL=/path/to/your/main-model.gguf ./run-dspark-server.sh
```

### 3. Download and build the DSpark sidecar

The DSpark sidecar is much smaller than the main model.  For a first install,
download only the DSpark module files used by DeepSeek's
`DeepSeek-V4-Flash-DSpark` repository:

```bash
mkdir -p /home/athena/ds4/dspark-v4flash-hf
cd /home/athena/ds4/dspark-v4flash-hf

for f in \
  config.json \
  model.safetensors.index.json \
  model-00046-of-00048.safetensors \
  model-00047-of-00048.safetensors \
  model-00048-of-00048.safetensors
do
  wget -c "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/resolve/main/$f"
done
```

Then convert those HF shards into the GGUF sidecar used by this fork:

```bash
cd /home/athena/DS4-GB10-GX10-DSpark-CUDA
./build-dspark-sidecar.sh 2>&1 | tee /home/athena/ds4/ds4-dspark-convert.log
```

The output should be:

```bash
/home/athena/ds4/DeepSeek-V4-Flash-DSpark-Q4K-Q8.gguf
```

After conversion, the temporary Hugging Face shard directory can be kept for
future rebuilds or removed to save disk space:

```bash
du -sh /home/athena/ds4/dspark-v4flash-hf
du -sh /home/athena/ds4/DeepSeek-V4-Flash-DSpark-Q4K-Q8.gguf
```

Custom paths are supported:

```bash
DS4_DSPARK_HF_DIR=/path/to/hf-shards \
DS4_DSPARK_GGUF=/path/to/DeepSeek-V4-Flash-DSpark-Q4K-Q8.gguf \
./build-dspark-sidecar.sh
```

### 4. Run the server

Default release profile, port `30007`:

```bash
cd /home/athena/DS4-GB10-GX10-DSpark-CUDA
./run-dspark-server.sh 2>&1 | tee /home/athena/ds4/ds4-dspark-release.log
```

Telemetry profile for benchmarking:

```bash
cd /home/athena/DS4-GB10-GX10-DSpark-CUDA
DS4_TELEMETRY=1 ./run-dspark-server.sh 2>&1 | tee /home/athena/ds4/ds4-dspark-release.log
```

Analyze:

```bash
cd /home/athena/DS4-GB10-GX10-DSpark-CUDA
./analyze-dspark-log.sh /home/athena/ds4/ds4-dspark-release.log 1
```

Quick API check:

```bash
curl http://127.0.0.1:30007/v1/models
```

Current GB10 release defaults in `run-dspark-server.sh`:

```text
DS4_CTX=131072
DS4_ADVERTISE_CONTEXT_PCT=85
DS4_MAX_TOKENS=2200
DS4_KV_DISK_SPACE_MB=16384
DS4_MEMORY_PROFILE=balanced
DS4_PREFILL_CHUNK=8192
DS4_CUDA_Q8_F16_CACHE_MB=12288
DS4_CUDA_COPY_SECONDARY_MODEL=1
DS4_CUDA_DROP_COPIED_MODEL_PAGES=1
DS4_KV_PREFILL_CHECKPOINT_POLICY=canonical-only
DS4_KV_LONG_COLD_ANCHOR_MIN_TOKENS=$((DS4_CTX / 2))
DS4_KV_LONG_COLD_ANCHOR_TRIM_TOKENS=$((DS4_CTX / 16))
```

With the default completion budget, the server advertises about 111k total
context and about 109k input tokens.  The remaining physical context is kept as
a safety margin for generation and for clients such as Claude Code to trigger
their own compaction before DS4 reaches the hard 131k limit.

The long-anchor values intentionally scale from `DS4_CTX`: at 131k context they
resolve to 65536 and 8192 tokens.  This preserves the append-prefill behavior
when the physical context is changed for A/B tests, instead of hardcoding one
specific checkpoint boundary.

To test a different guard or disk budget:

```bash
DS4_ADVERTISE_CONTEXT_PCT=95 ./run-dspark-server.sh
DS4_KV_DISK_SPACE_MB=65536 ./run-dspark-server.sh
```

The detailed lab notes, memory accounting and longer A/B history live in
`README-GB10.md`.

## Useful rollback switches

The routed-MoE MMQ and token-tile HMMA prefill paths are selected by structural
shape guards and intentionally have no launcher flag. For a full rollback,
keep the previous binary or build stable commit `59a5614` in a separate
checkout; do not restore only `ds4_cuda.cu` because MMQ also depends on
`cuda/mmq`, `ds4_gpu.h` and the CUDA regression test.

Disable Tensor Core tiny-batch completely:

```bash
DS4_CUDA_DSPARK_TENSOR_CORES=0 ./run-dspark-server.sh
```

Keep Tensor Core for F16 GEMMs but leave Q8 on the native GB10 reuse kernel:

```bash
DS4_CUDA_DSPARK_TENSOR_CORES=1 DS4_CUDA_DSPARK_TENSOR_CORES_Q8=0 ./run-dspark-server.sh
```

Disable DSpark p/q rejection sampling and return to exact-match verification:

```bash
DS4_DSPARK_REJECTION_DISABLE=1 ./run-dspark-server.sh
```

Re-enable the historical pre-draft performance gate:

```bash
DS4_DSPARK_ALWAYS_DRAFT=0 DS4_DSPARK_CIRCUIT_BREAKER=1 ./run-dspark-server.sh
```

## What to inspect in logs

Healthy DSpark release runs should show:

```text
Fused verifier cycles == DSpark cycles timed
Ordinary/fallback cycles: 0
P/Q rejection cycles close to fused verifier cycles
Pre-draft history bypasses: 0
K4 or K5 as dominant scheduler choice
```

The first eligible long prefill should also show:

```text
ds4: CUDA Entrpi batched MMQ MoE prefill enabled (... token-bound stream-K; decode excluded)
ds4: CUDA in-place aligned MoE execution active (Q8_K small-batch + D2R/MMQ prefill tiers)
ds4: CUDA packed MXFP4 indexer scorer enabled (68-byte rows, native block-scaled MMA; token-tile prefill + head-tile verifier on sm_121a)
ds4: CUDA exact radix Top-512 enabled (...)
ds4: CUDA exact parallel Top-512 enabled for small batches (4096-column chunk tree, low-index tie break)
ds4: CUDA Blackwell exact GVR Top-512 enabled (...)
ds4: CUDA token-tile HMMA raw/mixed prefill enabled (tile=16, heads=2)
ds4: CUDA token-tile HMMA indexed prefill enabled (tile=16, heads=2, exact-topk=512)
```

These lines are shape dependent and may appear only after the first request
that exercises the corresponding prefill or verifier path.

For Tensor Core confirmation:

```bash
grep -E 'tiny-batch Tensor Core|Tensor Core gemms|tiny-TC|GB10 verifier' /home/athena/ds4/ds4-dspark-release.log
```

Expected startup includes `tiny-TC=1` and `tiny-TC-Q8=1` in the GB10 verifier
line when the default release profile is active.

## About the fork maintainer

This GB10/GX10 DSpark CUDA fork is maintained as an independent experimental
work by Marco Palaferri.

Website: [www.palaferri.com](https://www.palaferri.com)

The project started as a practical effort to make DeepSeek-V4-Flash run faster
on a single NVIDIA GB10 machine, while keeping the original `antirez/ds4`
spirit: small codebase, direct experimentation, and measurable local inference
improvements.

## License and attribution

This repository keeps the original `ds4` MIT license.

The GB10/GX10 DSpark CUDA modifications in this fork are:

```text
Copyright (c) 2026 Marco Palaferri
Licensed under the MIT License
```

The MIT License allows use, copy, modification, publication, distribution,
sublicensing and sale of the software, provided that the copyright notice and
license text are preserved in copies or substantial portions of the software.
In practice: if you reuse the GB10/GX10 DSpark CUDA work from this fork, keep
the Marco Palaferri attribution together with the MIT license notice.
