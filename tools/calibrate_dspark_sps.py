#!/usr/bin/env python3
"""Build an immutable DSpark SPS(B) profile from verifier telemetry."""

from __future__ import annotations

import argparse
import math
import os
import re
import statistics
import sys
from collections import defaultdict
from itertools import combinations_with_replacement
from pathlib import Path

PROFILE_VERSION = 1
MAX_REQUESTS = 4
MAX_PREFIX = 5
CONTEXT_BUCKETS = 32
FINGERPRINT_RE = re.compile(
    r"DSpark offline SPS fingerprint=([0-9a-fA-F]{16})"
)


def fields(line: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in line.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        result[key] = value.rstrip(",")
    return result


def robust_samples(values: list[float], skip_first: int) -> list[float]:
    work = values[skip_first:]
    if len(work) < 3:
        return work
    center = statistics.median(work)
    deviations = [abs(value - center) for value in work]
    mad = statistics.median(deviations)
    if mad <= 0.0:
        return work
    limit = 3.5 * 1.4826 * mad
    trimmed = [value for value in work if abs(value - center) <= limit]
    return trimmed if trimmed else work


def parse_logs(
    paths: list[Path],
    allow_operational: bool,
) -> tuple[
    int | None,
    dict[
        tuple[int, int, int, int, int],
        dict[tuple[int, ...], list[float]],
    ],
]:
    fingerprint: int | None = None
    samples: dict[
        tuple[int, int, int, int, int],
        dict[tuple[int, ...], list[float]],
    ] = defaultdict(lambda: defaultdict(list))
    executor_id = {"physical": 1, "serial": 2}
    path_id = {"neural": 0, "hybrid": 1}

    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as source:
            for line in source:
                match = FINGERPRINT_RE.search(line)
                if match:
                    found = int(match.group(1), 16)
                    if fingerprint is not None and found != fingerprint:
                        raise ValueError(
                            f"incompatible fingerprints across logs: "
                            f"{fingerprint:016x} and {found:016x}"
                        )
                    fingerprint = found
                cohort_record = "ds4: dspark cohort timing " in line
                bench_record = "ds4-bench: SPS sample " in line
                if not cohort_record and not bench_record:
                    continue
                if cohort_record and not allow_operational:
                    continue
                item = fields(line)
                executor = executor_id.get(item.get("executor", ""))
                path_class = path_id.get(item.get("path", ""))
                if (
                    executor is None
                    or path_class is None
                    or (cohort_record and item.get("fallback") != "none")
                ):
                    continue
                try:
                    request_count = int(
                        item["requested_r"] if cohort_record else item["R"]
                    )
                    bucket = int(item["bucket"])
                    rows = int(item["batch"])
                    verify_ms = float(item["verify"])
                except (KeyError, ValueError):
                    continue
                if (
                    request_count < 1
                    or request_count > MAX_REQUESTS
                    or bucket < 0
                    or bucket >= CONTEXT_BUCKETS
                    or rows < request_count
                    or rows > request_count * (MAX_PREFIX + 1)
                    or not math.isfinite(verify_ms)
                    or verify_ms <= 0.0
                ):
                    continue
                shape: tuple[int, ...] = ()
                if bench_record:
                    try:
                        shape = tuple(
                            int(value)
                            for value in item["shape"].split(",")
                        )
                    except (KeyError, ValueError):
                        continue
                    if (
                        len(shape) != request_count
                        or shape != tuple(sorted(shape))
                        or any(value < 1 or value > 6 for value in shape)
                        or sum(shape) != rows
                    ):
                        continue
                key = (executor, path_class, bucket, request_count, rows)
                samples[key][shape].append(verify_ms / 1000.0)
    return fingerprint, samples


def expected_shapes(
    request_count: int,
    rows: int,
) -> set[tuple[int, ...]]:
    return {
        shape
        for shape in combinations_with_replacement(
            range(1, MAX_PREFIX + 2), request_count
        )
        if sum(shape) == rows
    }


def calibrate(
    raw: dict[
        tuple[int, int, int, int, int],
        dict[tuple[int, ...], list[float]],
    ],
    min_samples: int,
    skip_first: int,
) -> dict[tuple[int, int, int, int, int], tuple[float, int]]:
    result: dict[tuple[int, int, int, int, int], tuple[float, int]] = {}
    for key, shape_samples in raw.items():
        request_count = key[3]
        rows = key[4]
        dedicated = {
            shape: values
            for shape, values in shape_samples.items()
            if shape
        }
        if dedicated:
            required = expected_shapes(request_count, rows)
            if set(dedicated) != required:
                continue
            medians: list[float] = []
            retained = 0
            for shape in sorted(required):
                available = dedicated[shape][skip_first:]
                if len(available) < min_samples:
                    medians = []
                    break
                kept = robust_samples(dedicated[shape], skip_first)
                if not kept:
                    medians = []
                    break
                medians.append(statistics.median(kept))
                retained += len(kept)
            if medians:
                # SPS(B) cannot distinguish lane partitions. Use the slowest
                # measured partition so the scheduler never overestimates a
                # legal physical shape.
                result[key] = (max(medians), retained)
            continue

        operational = shape_samples.get((), [])
        available = operational[skip_first:]
        kept = robust_samples(operational, skip_first)
        if len(available) >= min_samples and kept:
            result[key] = (statistics.median(kept), len(kept))
    return result


def complete_groups(
    records: dict[tuple[int, int, int, int, int], tuple[float, int]],
) -> set[tuple[int, int, int, int]]:
    groups: set[tuple[int, int, int, int]] = set()
    candidates = {key[:4] for key in records}
    for group in candidates:
        request_count = group[3]
        required = range(request_count, request_count * (MAX_PREFIX + 1) + 1)
        if all((*group, rows) in records for rows in required):
            groups.add(group)
    return groups


def context_cross_validation(
    records: dict[tuple[int, int, int, int, int], tuple[float, int]],
) -> tuple[float, float, float, int, tuple[int, int, int, int, int] | None]:
    """Measure sparse-anchor interpolation error without using the held-out anchor."""
    groups = complete_groups(records)
    series: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for executor, path_class, bucket, request_count in groups:
        series[(executor, path_class, request_count)].append(bucket)

    errors: list[float] = []
    worst_error = 0.0
    worst_key: tuple[int, int, int, int, int] | None = None
    for (executor, path_class, request_count), raw_buckets in series.items():
        buckets = sorted(set(raw_buckets))
        if len(buckets) < 3:
            continue
        max_rows = request_count * (MAX_PREFIX + 1)
        for index in range(1, len(buckets) - 1):
            left = buckets[index - 1]
            middle = buckets[index]
            right = buckets[index + 1]
            if right == left:
                continue
            position = (middle - left) / (right - left)
            for rows in range(request_count, max_rows + 1):
                left_seconds = records[
                    (executor, path_class, left, request_count, rows)
                ][0]
                actual_seconds = records[
                    (executor, path_class, middle, request_count, rows)
                ][0]
                right_seconds = records[
                    (executor, path_class, right, request_count, rows)
                ][0]
                predicted = left_seconds + (
                    right_seconds - left_seconds
                ) * position
                relative = abs(predicted - actual_seconds) / actual_seconds
                errors.append(relative)
                if relative > worst_error:
                    worst_error = relative
                    worst_key = (
                        executor,
                        path_class,
                        middle,
                        request_count,
                        rows,
                    )
    median_error = statistics.median(errors) if errors else 0.0
    p95_error = (
        statistics.quantiles(errors, n=100, method="inclusive")[94]
        if len(errors) >= 2
        else (errors[0] if errors else 0.0)
    )
    return worst_error, p95_error, median_error, len(errors), worst_key


def interpolate_context(
    records: dict[tuple[int, int, int, int, int], tuple[float, int]],
) -> tuple[
    dict[tuple[int, int, int, int, int], tuple[float, int]],
    int,
]:
    """Fill only buckets bounded by complete measured groups."""
    result = dict(records)
    groups = complete_groups(records)
    series: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for executor, path_class, bucket, request_count in groups:
        series[(executor, path_class, request_count)].append(bucket)

    inserted = 0
    for (executor, path_class, request_count), raw_buckets in series.items():
        buckets = sorted(set(raw_buckets))
        max_rows = request_count * (MAX_PREFIX + 1)
        for left, right in zip(buckets, buckets[1:]):
            if right <= left + 1:
                continue
            for bucket in range(left + 1, right):
                position = (bucket - left) / (right - left)
                for rows in range(request_count, max_rows + 1):
                    key = (
                        executor,
                        path_class,
                        bucket,
                        request_count,
                        rows,
                    )
                    if key in result:
                        continue
                    left_seconds, left_samples = records[
                        (executor, path_class, left, request_count, rows)
                    ]
                    right_seconds, right_samples = records[
                        (executor, path_class, right, request_count, rows)
                    ]
                    seconds = left_seconds + (
                        right_seconds - left_seconds
                    ) * position
                    result[key] = (
                        seconds,
                        min(left_samples, right_samples),
                    )
                    inserted += 1
    return result, inserted


def write_profile(
    output: Path,
    fingerprint: int,
    records: dict[tuple[int, int, int, int, int], tuple[float, int]],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="ascii", newline="\n") as target:
        target.write(
            f"DS4_DSPARK_SPS_V{PROFILE_VERSION} {fingerprint:016x}\n"
        )
        target.write(
            "# C executor path bucket request_count rows "
            "verify_seconds samples\n"
        )
        for key in sorted(records):
            seconds, count = records[key]
            target.write(
                "C "
                + " ".join(str(value) for value in key)
                + f" {seconds:.17g} {count}\n"
            )
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate an immutable, fingerprinted DSpark SPS(B) profile "
            "from DS4_DSPARK_TIMING/DS4_DSPARK_LOG output."
        )
    )
    parser.add_argument("--log", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--fingerprint",
        help="16-digit runtime fingerprint; normally discovered in the log",
    )
    parser.add_argument("--min-samples", type=int, default=8)
    parser.add_argument(
        "--skip-first",
        type=int,
        default=2,
        help="warm-up samples discarded independently for each curve row",
    )
    parser.add_argument(
        "--allow-operational",
        action="store_true",
        help=(
            "accept ordinary cohort telemetry without complete partition "
            "coverage; dedicated SPS samples are required by default"
        ),
    )
    parser.add_argument(
        "--allow-no-complete-group",
        action="store_true",
        help="write analysis-only partial records even if runtime cannot use them",
    )
    parser.add_argument(
        "--interpolate-context",
        action="store_true",
        help=(
            "linearly fill context buckets bounded by complete measured "
            "groups; extrapolation beyond measured anchors is never used"
        ),
    )
    parser.add_argument(
        "--max-context-cv-error",
        type=float,
        default=0.0,
        help=(
            "maximum leave-one-anchor-out relative interpolation error; "
            "zero reports the error without enforcing a limit"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        args.min_samples < 1
        or args.skip_first < 0
        or args.max_context_cv_error < 0.0
        or not math.isfinite(args.max_context_cv_error)
    ):
        print("min-samples must be positive and skip-first non-negative", file=sys.stderr)
        return 2
    try:
        discovered, raw = parse_logs(args.log, args.allow_operational)
    except (OSError, ValueError) as exc:
        print(f"SPS calibration failed: {exc}", file=sys.stderr)
        return 2
    fingerprint = (
        int(args.fingerprint, 16) if args.fingerprint else discovered
    )
    if fingerprint is None or fingerprint == 0:
        print(
            "SPS calibration failed: no runtime fingerprint in logs; "
            "pass --fingerprint",
            file=sys.stderr,
        )
        return 2
    records = calibrate(raw, args.min_samples, args.skip_first)
    measured_records = len(records)
    (
        worst_cv,
        p95_cv,
        median_cv,
        cv_count,
        worst_key,
    ) = context_cross_validation(records)
    if cv_count:
        print(
            "context interpolation cross-validation: "
            f"median={median_cv:.3%} p95={p95_cv:.3%} "
            f"max={worst_cv:.3%} "
            f"comparisons={cv_count} worst={worst_key}"
        )
    if (
        args.max_context_cv_error > 0.0
        and cv_count
        and p95_cv > args.max_context_cv_error
    ):
        print(
            "SPS calibration failed: context interpolation p95 error "
            f"{p95_cv:.3%} exceeds "
            f"{args.max_context_cv_error:.3%}; add an anchor near "
            f"bucket {worst_key[2] if worst_key else 'unknown'}",
            file=sys.stderr,
        )
        return 2
    interpolated_records = 0
    if args.interpolate_context:
        records, interpolated_records = interpolate_context(records)
    groups = complete_groups(records)
    if not groups and not args.allow_no_complete_group:
        print(
            "SPS calibration produced no complete curve. Collect every "
            "B=R..6R row for at least one executor/path/context/R group.",
            file=sys.stderr,
        )
        return 2
    write_profile(args.output, fingerprint, records)
    print(
        f"wrote {args.output}: fingerprint={fingerprint:016x} "
        f"records={len(records)} measured={measured_records} "
        f"interpolated={interpolated_records} "
        f"complete_groups={len(groups)}"
    )
    for executor, path_class, bucket, request_count in sorted(groups):
        print(
            f"  complete executor={executor} path={path_class} "
            f"bucket={bucket} R={request_count} "
            f"rows={request_count}..{request_count * (MAX_PREFIX + 1)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
