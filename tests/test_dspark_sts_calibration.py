#!/usr/bin/env python3

import importlib.util
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "calibrate_dspark_sts", ROOT / "tools" / "calibrate_dspark_sts.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def main() -> None:
    samples = []
    for index in range(400):
        accepted = index % 6
        logits = tuple(
            3.0 - 0.9 * position + ((index * (position + 3)) % 11 - 5) * 0.2
            for position in range(5)
        )
        samples.append(MODULE.Sample(logits, accepted))
    temperatures, eces = MODULE.calibrate(samples, steps=301)
    assert len(temperatures) == 5 and len(eces) == 5
    assert all(0.05 <= value <= 20.0 for value in temperatures)
    assert all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in eces)

    cumulative = [1.0] * len(samples)
    for position, temperature in enumerate(temperatures):
        labels = [int(sample.accepted_prefix > position) for sample in samples]
        identity = [
            cumulative[i] * MODULE.sigmoid(sample.logits[position])
            for i, sample in enumerate(samples)
        ]
        calibrated = [
            cumulative[i]
            * MODULE.sigmoid(sample.logits[position] / temperature)
            for i, sample in enumerate(samples)
        ]
        assert MODULE.expected_calibration_error(
            calibrated, labels, 20
        ) <= MODULE.expected_calibration_error(identity, labels, 20) + 1.0e-12
        cumulative = calibrated

    calibration_a, validation_a = MODULE.split_samples(samples, 0.2, 42)
    calibration_b, validation_b = MODULE.split_samples(samples, 0.2, 42)
    assert calibration_a == calibration_b
    assert validation_a == validation_b
    assert len(calibration_a) == 320
    assert len(validation_a) == 80
    assert {id(sample) for sample in calibration_a}.isdisjoint(
        id(sample) for sample in validation_a
    )

    validation_eces = MODULE.evaluate(
        validation_a, temperatures, bins=20
    )
    assert len(validation_eces) == 5
    assert all(
        math.isfinite(value) and 0.0 <= value <= 1.0
        for value in validation_eces
    )
    assert MODULE.profile_is_promotable(
        [0.01, 0.02, 0.03, 0.04, 0.05],
        [0.02, 0.03, 0.04, 0.05, 0.06],
    )
    assert not MODULE.profile_is_promotable(
        [0.03, 0.02, 0.03, 0.04, 0.05],
        [0.02, 0.03, 0.04, 0.05, 0.06],
        max_position_regression=0.005,
    )
    assert not MODULE.profile_is_promotable(
        [0.02, 0.03, 0.04, 0.05, 0.06],
        [0.02, 0.03, 0.04, 0.05, 0.06],
    )

    grouped = [
        MODULE.Sample(sample.logits, sample.accepted_prefix, f"g-{i // 40}")
        for i, sample in enumerate(samples)
    ]
    grouped_calibration, grouped_validation = MODULE.split_samples(
        grouped, 0.2, 42
    )
    calibration_groups = {
        sample.group for sample in grouped_calibration
    }
    validation_groups = {sample.group for sample in grouped_validation}
    assert calibration_groups.isdisjoint(validation_groups)
    assert len(validation_groups) == 2

    cross_a = MODULE.grouped_cross_validate(
        grouped, 5, 42, steps=101, min_training_samples=300
    )
    cross_b = MODULE.grouped_cross_validate(
        grouped, 5, 42, steps=101, min_training_samples=300
    )
    assert cross_a == cross_b
    candidate_eces, baseline_eces, reports = cross_a
    assert len(candidate_eces) == len(baseline_eces) == 5
    assert len(reports) == 5
    validation_group_union = set()
    for report in reports:
        validation_group_union.update(report["validation_groups"])
        assert report["training_samples"] == 320
        assert report["validation_samples"] == 80
    assert validation_group_union == {f"g-{index}" for index in range(10)}
    print("dspark STS calibration regression: OK")


if __name__ == "__main__":
    main()
