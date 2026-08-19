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
