#!/usr/bin/env python3
"""Sequential Temperature Scaling for DSpark held-out verifier samples."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

POSITIONS = 5


@dataclass(frozen=True)
class Sample:
    logits: tuple[float, ...]
    accepted_prefix: int
    group: str | None = None


def sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def expected_calibration_error(
    probabilities: list[float], labels: list[int], bins: int
) -> float:
    if not probabilities or len(probabilities) != len(labels) or bins <= 0:
        raise ValueError("invalid ECE inputs")
    counts = [0] * bins
    probability_sum = [0.0] * bins
    label_sum = [0.0] * bins
    for probability, label in zip(probabilities, labels, strict=True):
        bounded = min(max(probability, 0.0), 1.0)
        bucket = min(int(bounded * bins), bins - 1)
        counts[bucket] += 1
        probability_sum[bucket] += bounded
        label_sum[bucket] += label
    total = len(probabilities)
    ece = 0.0
    for bucket, count in enumerate(counts):
        if count:
            confidence = probability_sum[bucket] / count
            accuracy = label_sum[bucket] / count
            ece += count / total * abs(confidence - accuracy)
    return ece


def temperature_grid(minimum: float, maximum: float, steps: int) -> list[float]:
    if minimum <= 0.0 or maximum < minimum or steps < 2:
        raise ValueError("invalid temperature grid")
    log_min = math.log(minimum)
    log_span = math.log(maximum) - log_min
    return [
        math.exp(log_min + log_span * index / (steps - 1))
        for index in range(steps)
    ]


def calibrate(
    samples: list[Sample],
    *,
    bins: int = 20,
    minimum: float = 0.05,
    maximum: float = 20.0,
    steps: int = 2001,
) -> tuple[list[float], list[float]]:
    if not samples:
        raise ValueError("the held-out set is empty")
    grid = temperature_grid(minimum, maximum, steps)
    cumulative = [1.0] * len(samples)
    temperatures: list[float] = []
    eces: list[float] = []
    for position in range(POSITIONS):
        labels = [
            int(sample.accepted_prefix > position) for sample in samples
        ]
        best_temperature = 1.0
        best_ece = math.inf
        for temperature in grid:
            probabilities = [
                cumulative[index]
                * sigmoid(sample.logits[position] / temperature)
                for index, sample in enumerate(samples)
            ]
            ece = expected_calibration_error(probabilities, labels, bins)
            if ece < best_ece - 1.0e-15 or (
                abs(ece - best_ece) <= 1.0e-15
                and abs(math.log(temperature))
                < abs(math.log(best_temperature))
            ):
                best_ece = ece
                best_temperature = temperature
        temperatures.append(best_temperature)
        eces.append(best_ece)
        cumulative = [
            cumulative[index]
            * sigmoid(sample.logits[position] / best_temperature)
            for index, sample in enumerate(samples)
        ]
    return temperatures, eces


def evaluate(
    samples: list[Sample],
    temperatures: list[float],
    *,
    bins: int = 20,
) -> list[float]:
    if not samples:
        raise ValueError("the evaluation set is empty")
    if len(temperatures) != POSITIONS:
        raise ValueError("invalid temperature count")
    cumulative = [1.0] * len(samples)
    eces: list[float] = []
    for position, temperature in enumerate(temperatures):
        cumulative = [
            cumulative[index]
            * sigmoid(sample.logits[position] / temperature)
            for index, sample in enumerate(samples)
        ]
        labels = [
            int(sample.accepted_prefix > position) for sample in samples
        ]
        eces.append(expected_calibration_error(cumulative, labels, bins))
    return eces


def grouped_cross_validate(
    samples: list[Sample],
    folds: int,
    seed: int,
    *,
    bins: int = 20,
    minimum: float = 0.05,
    maximum: float = 20.0,
    steps: int = 2001,
    min_training_samples: int = 1,
) -> tuple[list[float], list[float], list[dict[str, object]]]:
    groups = sorted(
        {sample.group for sample in samples if sample.group is not None}
    )
    if (
        len(groups) < 2
        or len(groups) != len({sample.group for sample in samples})
        or any(sample.group is None for sample in samples)
    ):
        raise ValueError("grouped cross-validation needs grouped samples")
    if folds < 2 or folds > len(groups):
        raise ValueError("validation folds must be between 2 and group count")

    random.Random(seed).shuffle(groups)
    fold_groups = [set(groups[index::folds]) for index in range(folds)]
    candidate_probabilities = [[] for _ in range(POSITIONS)]
    baseline_probabilities = [[] for _ in range(POSITIONS)]
    labels = [[] for _ in range(POSITIONS)]
    reports: list[dict[str, object]] = []
    for fold, held_out_groups in enumerate(fold_groups):
        training = [
            sample for sample in samples
            if sample.group not in held_out_groups
        ]
        validation = [
            sample for sample in samples
            if sample.group in held_out_groups
        ]
        if len(training) < min_training_samples or not validation:
            raise ValueError(
                f"fold {fold} has training={len(training)} "
                f"validation={len(validation)}"
            )
        temperatures, _ = calibrate(
            training,
            bins=bins,
            minimum=minimum,
            maximum=maximum,
            steps=steps,
        )
        candidate_cumulative = [1.0] * len(validation)
        baseline_cumulative = [1.0] * len(validation)
        for position in range(POSITIONS):
            for index, sample in enumerate(validation):
                candidate_cumulative[index] *= sigmoid(
                    sample.logits[position] / temperatures[position]
                )
                baseline_cumulative[index] *= sigmoid(
                    sample.logits[position]
                )
                candidate_probabilities[position].append(
                    candidate_cumulative[index]
                )
                baseline_probabilities[position].append(
                    baseline_cumulative[index]
                )
                labels[position].append(
                    int(sample.accepted_prefix > position)
                )
        reports.append(
            {
                "fold": fold,
                "training_samples": len(training),
                "validation_samples": len(validation),
                "validation_groups": sorted(held_out_groups),
                "temperatures": temperatures,
            }
        )

    candidate_eces = [
        expected_calibration_error(
            candidate_probabilities[position], labels[position], bins
        )
        for position in range(POSITIONS)
    ]
    baseline_eces = [
        expected_calibration_error(
            baseline_probabilities[position], labels[position], bins
        )
        for position in range(POSITIONS)
    ]
    return candidate_eces, baseline_eces, reports


def profile_is_promotable(
    validation_eces: list[float],
    baseline_validation_eces: list[float],
    *,
    min_mean_improvement: float = 0.0,
    max_position_regression: float = 0.01,
) -> bool:
    if (
        len(validation_eces) != POSITIONS
        or len(baseline_validation_eces) != POSITIONS
        or min_mean_improvement < 0.0
        or max_position_regression < 0.0
    ):
        raise ValueError("invalid STS promotion inputs")
    candidate_mean = sum(validation_eces) / POSITIONS
    baseline_mean = sum(baseline_validation_eces) / POSITIONS
    if candidate_mean >= baseline_mean - min_mean_improvement:
        return False
    return all(
        candidate <= baseline + max_position_regression
        for candidate, baseline in zip(
            validation_eces, baseline_validation_eces, strict=True
        )
    )


def split_samples(
    samples: list[Sample],
    validation_fraction: float,
    seed: int,
) -> tuple[list[Sample], list[Sample]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between zero and one")
    if len(samples) < 2:
        raise ValueError("at least two samples are required for a split")
    grouped = all(sample.group is not None for sample in samples)
    groups = sorted({sample.group for sample in samples}) if grouped else []
    if grouped and len(groups) >= 2:
        random.Random(seed).shuffle(groups)
        validation_group_count = max(
            1, int(round(len(groups) * validation_fraction))
        )
        validation_group_count = min(
            validation_group_count, len(groups) - 1
        )
        validation_groups = set(groups[:validation_group_count])
        calibration = [
            sample for sample in samples
            if sample.group not in validation_groups
        ]
        validation = [
            sample for sample in samples
            if sample.group in validation_groups
        ]
        return calibration, validation

    indices = list(range(len(samples)))
    random.Random(seed).shuffle(indices)
    validation_count = max(
        1, int(round(len(samples) * validation_fraction))
    )
    validation_count = min(validation_count, len(samples) - 1)
    validation_indices = set(indices[:validation_count])
    calibration = [
        sample for index, sample in enumerate(samples)
        if index not in validation_indices
    ]
    validation = [
        sample for index, sample in enumerate(samples)
        if index in validation_indices
    ]
    return calibration, validation


def load_samples(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        expected = {
            *(f"logit_{position}" for position in range(1, POSITIONS + 1)),
            "accepted_prefix",
        }
        if not reader.fieldnames or not expected.issubset(reader.fieldnames):
            raise ValueError(
                "capture must contain logit_1..logit_5 and accepted_prefix"
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                logits = tuple(
                    float(row[f"logit_{position}"])
                    for position in range(1, POSITIONS + 1)
                )
                accepted_prefix = int(row["accepted_prefix"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid capture row {line_number}"
                ) from exc
            if (
                any(not math.isfinite(value) for value in logits)
                or accepted_prefix < 0
                or accepted_prefix > POSITIONS
            ):
                raise ValueError(f"invalid capture row {line_number}")
            group = row.get("group")
            samples.append(
                Sample(logits, accepted_prefix, group if group else None)
            )
    return samples


def write_profile(
    path: Path,
    samples: int,
    temperatures: list[float],
    eces: list[float],
) -> None:
    payload = (
        "DS4_DSPARK_STS_V1\n"
        "positions=5\n"
        f"samples={samples}\n"
        "temperatures="
        + ",".join(f"{value:.9g}" for value in temperatures)
        + "\n"
        "cumulative_ece="
        + ",".join(f"{value:.9g}" for value in eces)
        + "\n"
    )
    path.write_text(payload, encoding="ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit DSpark STS left-to-right on a held-out full-K verifier capture."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--grid-steps", type=int, default=2001)
    parser.add_argument(
        "--min-samples",
        type=int,
        default=512,
        help="reject captures smaller than this held-out sample count",
    )
    parser.add_argument("--min-temperature", type=float, default=0.05)
    parser.add_argument("--max-temperature", type=float, default=20.0)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
        help="deterministic out-of-sample validation fraction",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=260705147,
        help="seed used only to partition capture rows",
    )
    parser.add_argument(
        "--validation-folds",
        type=int,
        default=5,
        help=(
            "grouped out-of-fold validation count; falls back to the "
            "single held-out split when the capture has too few groups"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="optional JSON calibration/validation report",
    )
    parser.add_argument(
        "--min-mean-improvement",
        type=float,
        default=0.0,
        help=(
            "minimum held-out mean-ECE improvement required before writing "
            "the deployable profile"
        ),
    )
    parser.add_argument(
        "--max-position-regression",
        type=float,
        default=0.01,
        help=(
            "largest held-out ECE regression allowed at any prefix position"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_samples(args.input)
    if args.min_samples <= 0:
        raise ValueError("--min-samples must be positive")
    if len(samples) < args.min_samples:
        raise ValueError(
            f"held-out capture has {len(samples)} samples; "
            f"at least {args.min_samples} are required"
        )
    grouped = all(sample.group is not None for sample in samples)
    group_count = len({sample.group for sample in samples}) if grouped else 0
    validation_mode = "held-out-split"
    validation_fraction: float | None = args.validation_fraction
    fold_reports: list[dict[str, object]] = []
    if grouped and args.validation_folds >= 2 and \
            group_count >= args.validation_folds:
        validation_mode = "grouped-out-of-fold"
        validation_fraction = None
        validation_eces, baseline_validation_eces, fold_reports = \
            grouped_cross_validate(
                samples,
                args.validation_folds,
                args.split_seed,
                bins=args.bins,
                minimum=args.min_temperature,
                maximum=args.max_temperature,
                steps=args.grid_steps,
                min_training_samples=args.min_samples,
            )
        calibration = samples
        validation = samples
    else:
        calibration, validation = split_samples(
            samples, args.validation_fraction, args.split_seed
        )
        if len(calibration) < args.min_samples:
            raise ValueError(
                f"calibration split has {len(calibration)} samples; "
                f"at least {args.min_samples} are required after reserving "
                "validation rows"
            )
        validation_eces = []
        baseline_validation_eces = []

    temperatures, calibration_eces = calibrate(
        samples if validation_mode == "grouped-out-of-fold" else calibration,
        bins=args.bins,
        minimum=args.min_temperature,
        maximum=args.max_temperature,
        steps=args.grid_steps,
    )
    if validation_mode == "held-out-split":
        validation_eces = evaluate(
            validation, temperatures, bins=args.bins
        )
        baseline_validation_eces = evaluate(
            validation, [1.0] * POSITIONS, bins=args.bins
        )
    promotable = profile_is_promotable(
        validation_eces,
        baseline_validation_eces,
        min_mean_improvement=args.min_mean_improvement,
        max_position_regression=args.max_position_regression,
    )
    report = {
        "capture_samples": len(samples),
        "calibration_samples": len(calibration),
        "validation_samples": len(validation),
        "validation_mode": validation_mode,
        "validation_fraction": validation_fraction,
        "validation_folds":
            args.validation_folds
            if validation_mode == "grouped-out-of-fold" else 1,
        "group_count": group_count,
        "folds": fold_reports,
        "split_seed": args.split_seed,
        "temperatures": temperatures,
        "calibration_cumulative_ece": calibration_eces,
        "validation_cumulative_ece": validation_eces,
        "baseline_validation_cumulative_ece": baseline_validation_eces,
        "validation_mean_ece": sum(validation_eces) / POSITIONS,
        "baseline_validation_mean_ece":
            sum(baseline_validation_eces) / POSITIONS,
        "min_mean_improvement": args.min_mean_improvement,
        "max_position_regression": args.max_position_regression,
        "promotable": promotable,
    }
    if args.report:
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
    if not promotable:
        args.output.unlink(missing_ok=True)
        print(
            "rejected STS profile: held-out calibration did not beat the "
            "identity confidence safely "
            f"(validation_ece={report['validation_mean_ece']:.6f}, "
            "baseline_validation_ece="
            f"{report['baseline_validation_mean_ece']:.6f})"
        )
        raise SystemExit(2)
    write_profile(
        args.output, len(calibration), temperatures, validation_eces
    )
    print(
        f"wrote {args.output} calibration={len(calibration)} "
        f"validation={len(validation)} "
        f"validation_ece={report['validation_mean_ece']:.6f} "
        f"baseline_validation_ece="
        f"{report['baseline_validation_mean_ece']:.6f} "
        f"temperatures={','.join(f'{value:.5f}' for value in temperatures)}"
    )


if __name__ == "__main__":
    main()
