# Contributors

This file records external contributions that materially improved the project.
It complements the code-lineage and license information in the main README; it
does not replace per-file copyright notices or commit attribution.

## Runtime and CUDA

### avisual

- GitHub: [@avisual](https://github.com/avisual)
- Contribution: investigated long-running host RSS growth on GB10 and provided
  the measurements that identified accumulation of live CUDA Graph executables
  across variants as the underlying cause.
- The follow-up analysis demonstrated that `cudaGraphExecDestroy` releases the
  associated memory and corrected the initial driver-leak hypothesis. It also
  proposed a global cross-variant budget and supplied long-context soak data
  used to validate the direction of the fix.
- Tracking issue:
  [#3 - Per-request host RSS leak](https://github.com/xangel82/DS4-GB10-GX10-DSpark-CUDA/issues/3)

The project implementation builds on that diagnosis with a globally bounded,
segmented LRU cache, probation/protected states, a reserve for small graphs,
telemetry, and explicit A/B rollback controls.

We thank `avisual` for the careful investigation, for publicly correcting the
initial hypothesis when new evidence contradicted it, and for sharing the
measurements that made the final fix stronger.

## NVIDIA Developer Forum community

The following people contributed independent measurements, operational
testing, problem reports, priorities, or constructive discussion in the
[GB10/GX10 DSpark development thread](https://forums.developer.nvidia.com/t/optimizing-deepseek-v4-flash-on-a-single-nvidia-gb10-gx10-with-dspark-speculative-decoding/376830).

### Independent testing and technical feedback

- [`agupta30`](https://forums.developer.nvidia.com/u/agupta30) published an
  extensive same-machine comparison covering prefill depth, decode,
  concurrency, installation behavior, and long-context operation. Those
  measurements made multi-request scaling a concrete development priority.
- [`btvd`](https://forums.developer.nvidia.com/u/btvd) tested multiple releases,
  reported practical context constraints, exercised the 1M profile, and
  confirmed multi-day stability on an independent system.
- [`szymon-walczak`](https://forums.developer.nvidia.com/u/szymon-walczak)
  shared a complete 69-scenario `tool-eval-bench` run, providing independent
  quality, responsiveness, and tool-calling evidence.
- [`styles01`](https://forums.developer.nvidia.com/u/styles01) reported the
  single-node DS4-0731 OOM and max-context/concurrency installation problem,
  helping focus the upgrade recipe and memory guidance.
- [`foogitiff`](https://forums.developer.nvidia.com/u/foogitiff) shared early
  GB10 prompt-processing and long-context benchmark data from the upstream
  runtime, giving the prefill work an independent reference point.
- [`m0l0`](https://forums.developer.nvidia.com/u/m0l0) highlighted prompt
  processing in established codebases as the more important interactive
  bottleneck, helping establish prefill as a primary optimization target.

### Community discussion and release feedback

- [`Zambonilli`](https://forums.developer.nvidia.com/u/zambonilli) encouraged
  cross-project collaboration and raised useful questions about comparison and
  upstream integration.
- [`henning.firman`](https://forums.developer.nvidia.com/u/henning.firman)
  provided an independent deployment confirmation.
- [`VCR`](https://forums.developer.nvidia.com/u/vcr) provided welcome feedback
  on the pace and maturity of the project.

Thank you to everyone who spent time testing, measuring, questioning, and
improving the project in public.
