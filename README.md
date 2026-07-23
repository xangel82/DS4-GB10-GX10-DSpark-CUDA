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
- Fused wide-prefill HC expansion, RMS, inverse RoPE packing and routed-MoE
  reduction epilogues while preserving the canonical FP32 state and reusing
  existing scratch buffers.
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
- Extended the direct-F16 FlashMLA-style indexed attention path to the actual
  compressed-KV tensor capacity, removing the artificial 131k fast-path cliff
  without adding a score matrix, mirror or persistent scratch.
- Added reproducible cold/append GB10 sweeps with DSpark decode, process-memory
  high-water marks and deterministic token hashes. The complete path has been
  validated end to end on Athena through 180.8k context.
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
  - 256k physical context with an 85% advertised context guard;
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

The current default allocates a 256k physical context and advertises 85% of it.
A real long-chat run sustained 913.15 prefill token/s from 27.7k to 95.1k,
859.77 token/s from 95.1k to 125.3k, and 836.16 token/s from 127.8k to 180.8k.
Before the dynamic direct-F16 capacity fix, the first chunk beyond 131k fell
from about 860 to 648 token/s and the 147.2k to 207.1k append averaged 624.92
token/s. The updated path kept its six complete post-threshold chunks between
858.48 and 828.20 token/s, removing that artificial cliff without allocating
additional buffers. DSpark decode after the 180.8k prefill measured 19.89
token/s over 290 generated tokens. Exact decode numbers vary with prompt,
sampling, draft acceptance and chunk-boundary tails.

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
| Pre-epilogue pipeline, cold 13.6k prompt | 787.06 t/s prefill | Historical same-machine baseline; first 8192-token chunk reached 854.26 t/s. |
| Fused epilogue pipeline, cold 25.3k prompt | 902.67 t/s prefill | Three complete chunks measured 835.81, 991.09 and 977.94 t/s; the 776-token tail reduced the request average. |
| Fused epilogue pipeline, cold 13.4k prompt | 952.97 t/s prefill | First 8192-token chunk reached 1009.78 t/s; the next 4096-token chunk reached 952.66 t/s. |
| Fused epilogue pipeline, append 57.8k -> 78.2k | 730.56 t/s prefill | 20,397 appended tokens; the central 8192-token chunk reached 898.45 t/s. |
| Fused epilogue pipeline, append 77.2k -> 90.5k | 760.77 t/s prefill | 13,348 appended tokens; the complete central chunk reached 891.13 t/s. |
| Current DSpark decode at 90.5k / 93.5k | 24.00 / 23.46 t/s | Measured over 284 and 222 generated tokens respectively; tool-call generation and canonical KV reuse remained active. |
| 256k profile, append 27.7k -> 95.1k | 913.15 t/s prefill | 67,316 appended tokens; complete chunks declined gradually from 940.58 to 896.36 t/s. |
| 256k profile, old path beyond 131k | 624.92 t/s prefill | 59,856 appended tokens from 147.2k to 207.1k; complete chunks measured about 641 to 618 t/s after the fast-path cutoff. |
| 256k profile, dynamic direct-F16 beyond 131k | 836.16 t/s prefill | 53,017 appended tokens from 127.8k to 180.8k, **+33.80%** versus the earlier deep append; complete post-threshold chunks measured 858.48 to 828.20 t/s. |
| DSpark decode after 180.8k prefill | 19.89 t/s | 290 generated tokens in the same operational run; no CUDA error, OOM or attention fallback. |

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
the MXFP4 scorer, exact shape-specific Top-K dispatch, FlashMLA-style exact
sparse attention and fused HC/RMS/RoPE/MoE epilogues. Large batches use the
prefill tiers, while decode and speculative verification remain on the Q8_K
small-batch path. The CUDA regression permits at most one F16 ULP between the
materialized and fused epilogues and rejects FP32 differences above `2e-6`.

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
MXFP4 scoring, aligned-SoA D2R and Q8_K MoE parity, both attention variants,
and the fused HC/RMS/RoPE/MoE prefill epilogues. Stop an already running server
first if memory is close to the GB10 limit.

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
DS4_CTX=262144
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

With the default completion budget, the server advertises about 222k total
context and about 220k input tokens.  The remaining physical context is kept as
a safety margin for generation and for clients such as Claude Code to trigger
their own compaction before DS4 reaches the hard 256k limit.

The long-anchor values intentionally scale from `DS4_CTX`: at 256k context they
resolve to 131072 and 16384 tokens.  This preserves the append-prefill behavior
when the physical context is changed for A/B tests, instead of hardcoding one
specific checkpoint boundary.

An experimental capacity-first launcher is also included for a 1M physical
context. It uses a 4096-token chunk, an isolated disk-KV directory and the same
85% advertised guard:

```bash
cd /home/athena/DS4-GB10-GX10-DSpark-CUDA && ./run-dspark-server-1m.sh 2>&1 | tee /tmp/ds4-1m.log
```

The 1M profile is not the default throughput configuration. Do not raise its
chunk to 8192 on the measured GB10 setup: that combination exceeded the
available unified-memory budget. See `README-GB10.md` for the measured memory
limits and rollback procedure.

To test a different guard or disk budget:

```bash
DS4_ADVERTISE_CONTEXT_PCT=95 ./run-dspark-server.sh
DS4_KV_DISK_SPACE_MB=65536 ./run-dspark-server.sh
```

The detailed lab notes, memory accounting and longer A/B history live in
`README-GB10.md`.

## Useful rollback switches

The routed-MoE MMQ, sparse-attention and fused-epilogue prefill paths are
selected by structural shape guards and intentionally have no launcher flag.
For a full rollback, keep the previous binary or build a known stable commit
in a separate checkout; do not restore only `ds4_cuda.cu` because the complete
path also depends on `cuda/mmq`, `cuda/indexer`, `ds4.c`, `ds4_gpu.h` and the
CUDA regression test.

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
ds4: CUDA complete fused MoE D2R prefill enabled (preallocated workspace, register gate/up, direct SwiGLU Q8 down)
ds4: CUDA in-place aligned MoE execution active (Q8_K small-batch + D2R/MMQ prefill tiers)
ds4: CUDA packed MXFP4 indexer scorer enabled (68-byte rows, native block-scaled MMA; token-tile prefill + head-tile verifier on sm_121a)
ds4: CUDA exact radix Top-512 enabled (...)
ds4: CUDA exact parallel Top-512 enabled for small batches (4096-column chunk tree, low-index tie break)
ds4: CUDA Blackwell exact GVR Top-512 enabled (...)
ds4: CUDA token-tile HMMA raw/mixed prefill enabled (tile=16, heads=2)
ds4: CUDA FlashMLA-style exact sparse prefill enabled (token=1, heads=32, stage=32, direct-topk=512, comp-kv=direct-f16)
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
