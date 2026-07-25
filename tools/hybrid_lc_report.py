#!/usr/bin/env python3
"""Summarize and gate an opt-in HybridLC DSpark run."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


FIELD_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")


def fields(line: str) -> dict[str, str]:
    return {key: value.rstrip(",") for key, value in FIELD_RE.findall(line)}


def number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except ValueError:
        return default


def parse_log(path: Path) -> dict:
    result = {
        "path": str(path),
        "cycles": 0,
        "hybrid_cycles": 0,
        "production_cycles": 0,
        "production_rejection_cycles": 0,
        "shadow_cycles": 0,
        "hybrid_rejection_cycles": 0,
        "blockv_cycles": 0,
        "retrieval_drafted": 0,
        "retrieval_committed": 0,
        "suffix_drafted": 0,
        "suffix_committed": 0,
        "transition_drafted": 0,
        "transition_committed": 0,
        "shadow_queries": 0,
        "shadow_attempts": 0,
        "shadow_match8": 0,
        "shadow_match16": 0,
        "shadow_full_matches": 0,
        "shadow_matches": 0,
        "shadow_tokens": 0,
        "oracle_rebuilds": 0,
        "self_checks": 0,
        "self_check_failures": 0,
        "summary_tokens": 0,
        "summary_seconds": 0.0,
        "widths": defaultdict(
            lambda: {
                "cycles": 0,
                "emitted": 0,
                "committed": 0,
                "verify_ms": 0.0,
                "total_ms": 0.0,
            }
        ),
    }
    for line in path.read_text(errors="replace").splitlines():
        if "ds4: dspark timing " in line:
            row = fields(line)
            result["cycles"] += 1
            hybrid = int(number(row, "hybrid"))
            blockv = int(number(row, "blockv"))
            rejection = int(number(row, "rejection"))
            result["hybrid_cycles"] += hybrid
            production = int(number(row, "hybrid_enabled"))
            result["production_cycles"] += production
            result["production_rejection_cycles"] += int(
                bool(production) and bool(rejection)
            )
            result["hybrid_rejection_cycles"] += int(
                bool(hybrid) and bool(rejection)
            )
            result["blockv_cycles"] += blockv
            result["shadow_cycles"] += int(number(row, "shadow") != 0.0)
            result["retrieval_drafted"] += int(number(row, "retrieval"))
            result["retrieval_committed"] += int(
                number(row, "retrieval_committed")
            )
            result["suffix_drafted"] += int(number(row, "suffix"))
            result["suffix_committed"] += int(
                number(row, "suffix_committed")
            )
            result["transition_drafted"] += int(number(row, "transition"))
            result["transition_committed"] += int(
                number(row, "transition_committed")
            )
            for key in (
                "shadow_queries",
                "shadow_attempts",
                "shadow_match8",
                "shadow_match16",
                "shadow_full",
                "oracle_rebuilds",
            ):
                result_key = (
                    "shadow_full_matches" if key == "shadow_full" else key
                )
                result[result_key] = max(
                    result[result_key], int(number(row, key))
                )
            result["shadow_matches"] = max(
                result["shadow_matches"], int(number(row, "shadow_matches"))
            )
            result["shadow_tokens"] = max(
                result["shadow_tokens"], int(number(row, "shadow_tokens"))
            )
            result["self_checks"] = max(
                result["self_checks"], int(number(row, "self_checks"))
            )
            result["self_check_failures"] = max(
                result["self_check_failures"],
                int(number(row, "self_check_failures")),
            )
            width = int(number(row, "width"))
            if width in (3, 6, 8, 12, 16):
                item = result["widths"][str(width)]
                item["cycles"] += 1
                item["emitted"] += int(number(row, "emitted"))
                item["committed"] += int(number(row, "committed"))
                item["verify_ms"] += number(row, "verify")
                item["total_ms"] += number(row, "total")
        elif "ds4-server: decode summary " in line:
            row = fields(line)
            result["summary_tokens"] += int(number(row, "gen"))
            result["summary_seconds"] += number(row, "seconds")

    result["widths"] = dict(result["widths"])
    for item in result["widths"].values():
        item["rate_tps"] = (
            1000.0 * item["emitted"] / item["total_ms"]
            if item["total_ms"] > 0.0
            else 0.0
        )
        item["mean_verify_ms"] = (
            item["verify_ms"] / item["cycles"] if item["cycles"] else 0.0
        )
    result["decode_tps"] = (
        result["summary_tokens"] / result["summary_seconds"]
        if result["summary_seconds"] > 0.0
        else 0.0
    )
    result["retrieval_acceptance"] = (
        result["retrieval_committed"] / result["retrieval_drafted"]
        if result["retrieval_drafted"]
        else 0.0
    )
    result["shadow_accuracy"] = (
        result["shadow_matches"] / result["shadow_tokens"]
        if result["shadow_tokens"]
        else 0.0
    )
    result["shadow_hit_rate"] = (
        result["shadow_attempts"] / result["shadow_queries"]
        if result["shadow_queries"]
        else 0.0
    )
    result["shadow_match8_coverage"] = (
        result["shadow_match8"] / result["shadow_queries"]
        if result["shadow_queries"]
        else 0.0
    )
    result["shadow_match16_coverage"] = (
        result["shadow_match16"] / result["shadow_queries"]
        if result["shadow_queries"]
        else 0.0
    )
    result["shadow_full_rate"] = (
        result["shadow_full_matches"] / result["shadow_attempts"]
        if result["shadow_attempts"]
        else 0.0
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report HybridLC cost curves and promotion gates"
    )
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--max-decode-regression", type=float, default=0.03)
    parser.add_argument("--min-shadow-accuracy", type=float, default=0.50)
    parser.add_argument("--min-shadow-match8-coverage", type=float, default=0.20)
    parser.add_argument("--min-shadow-match16-coverage", type=float, default=0.10)
    parser.add_argument(
        "--require-all-widths",
        action="store_true",
        help="Require measured N=3, N=6, N=8, N=12 and N=16 cost points",
    )
    args = parser.parse_args()
    for name in (
        "max_decode_regression",
        "min_shadow_accuracy",
        "min_shadow_match8_coverage",
        "min_shadow_match16_coverage",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")

    candidate = parse_log(args.candidate)
    baseline = parse_log(args.baseline) if args.baseline else None
    failures: list[str] = []
    if not candidate["production_cycles"] and not candidate["shadow_cycles"]:
        failures.append("log contains no HybridLC production or shadow cycles")
    if candidate["shadow_cycles"] and not candidate["shadow_queries"]:
        failures.append("shadow run completed without retrieval queries")
    if candidate["self_check_failures"]:
        failures.append("BlockV CPU/GPU self-check failed")
    if candidate["production_rejection_cycles"] and (
        candidate["blockv_cycles"] != candidate["production_rejection_cycles"]
    ):
        failures.append("not every HybridLC production cycle used BlockV")
    if candidate["blockv_cycles"] and not candidate["self_checks"]:
        failures.append("BlockV ran without a completed CPU/GPU self-check")
    if candidate["shadow_tokens"] and (
        candidate["shadow_accuracy"] < args.min_shadow_accuracy
    ):
        failures.append(
            f"shadow accuracy {candidate['shadow_accuracy']:.1%} is below "
            f"{args.min_shadow_accuracy:.1%}"
        )
    if candidate["shadow_queries"] and not (
        candidate["shadow_match8_coverage"]
        >= args.min_shadow_match8_coverage
        or candidate["shadow_match16_coverage"]
        >= args.min_shadow_match16_coverage
    ):
        failures.append(
            "retrieval oracle coverage is below both gates: "
            f"match>=8 {candidate['shadow_match8_coverage']:.1%} "
            f"(need {args.min_shadow_match8_coverage:.1%}) and "
            f"match>=16 {candidate['shadow_match16_coverage']:.1%} "
            f"(need {args.min_shadow_match16_coverage:.1%})"
        )
    if candidate["shadow_attempts"] > candidate["shadow_queries"]:
        failures.append("shadow tail attempts exceed retrieval queries")
    if candidate["shadow_match8"] > candidate["shadow_queries"] or (
        candidate["shadow_match16"] > candidate["shadow_match8"]
    ):
        failures.append("shadow retrieval coverage counters are inconsistent")
    if candidate["shadow_full_matches"] > candidate["shadow_attempts"]:
        failures.append("shadow full-tail matches exceed observed tails")
    if args.require_all_widths:
        missing = sorted(
            {"3", "6", "8", "12", "16"} - set(candidate["widths"]),
            key=int,
        )
        if missing:
            failures.append("missing cost-curve widths: " + ", ".join(missing))
    if baseline and baseline["decode_tps"] > 0.0:
        floor = baseline["decode_tps"] * (1.0 - args.max_decode_regression)
        if candidate["decode_tps"] < floor:
            failures.append(
                f"decode {candidate['decode_tps']:.3f} t/s is below "
                f"baseline gate {floor:.3f} t/s"
            )

    payload = {
        "candidate": candidate,
        "baseline": baseline,
        "passed": not failures,
        "failures": failures,
    }
    print(
        f"HybridLC cycles: {candidate['hybrid_cycles']} "
        f"(production {candidate['production_cycles']}, "
        f"BlockV {candidate['blockv_cycles']}, "
        f"shadow {candidate['shadow_cycles']})"
    )
    print(
        "Tail acceptance: "
        f"{candidate['retrieval_committed']}/"
        f"{candidate['retrieval_drafted']} "
        f"({candidate['retrieval_acceptance']:.2%})"
    )
    print(
        "Suffix/transition: "
        f"{candidate['suffix_committed']}/{candidate['suffix_drafted']} | "
        f"{candidate['transition_committed']}/"
        f"{candidate['transition_drafted']}"
    )
    for width, item in sorted(
        candidate["widths"].items(), key=lambda pair: int(pair[0])
    ):
        print(
            f"N={width}: cycles={item['cycles']} "
            f"verify={item['mean_verify_ms']:.3f} ms "
            f"rate={item['rate_tps']:.3f} t/s"
        )
    if candidate["shadow_tokens"]:
        print(
            f"Shadow: {candidate['shadow_matches']}/"
            f"{candidate['shadow_tokens']} "
            f"({candidate['shadow_accuracy']:.2%})"
        )
    if candidate["shadow_queries"]:
        print(
            "Shadow oracle: "
            f"queries={candidate['shadow_queries']} "
            f"tails={candidate['shadow_attempts']} "
            f"hit={candidate['shadow_hit_rate']:.2%} "
            f"match>=8={candidate['shadow_match8_coverage']:.2%} "
            f"match>=16={candidate['shadow_match16_coverage']:.2%} "
            f"full-tail={candidate['shadow_full_rate']:.2%}"
        )
    print(f"Oracle rebuilds: {candidate['oracle_rebuilds']}")
    print(
        f"Self-checks: {candidate['self_checks']} "
        f"(failures {candidate['self_check_failures']})"
    )
    print(f"Request decode: {candidate['decode_tps']:.3f} t/s")
    print("Gate: " + ("PASS" if not failures else "FAIL"))
    for failure in failures:
        print(f"  - {failure}")
    if args.json:
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
