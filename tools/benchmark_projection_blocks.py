#!/usr/bin/env python3
"""Measure complete tiny-batch projection blocks before CUDA Graph capture."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any


POLICIES = ("legacy", "f16-real", "f16-pad8", "f16-out", "q8-reuse", "auto")
BLOCKS = (
    "attn_q_a_kv",
    "attn_q_b_rms_rope",
    "attn_output_hc",
    "shared_gate_up_swiglu",
    "shared_down_hc",
    "indexer_q_b_rope",
    "indexer_compressor_pair",
    "hc_16384x24_f16",
    "router_4096x256_f16",
    "indexer_4096x64_f16",
    "vocab",
)
NARROW_F16_BLOCKS = {
    "hc_16384x24_f16",
    "router_4096x256_f16",
    "indexer_4096x64_f16",
}
NUMERIC_FIELDS = (
    "n",
    "iterations",
    "total_ms",
    "us_per_call",
    "fallbacks",
    "f16_resident_bytes",
    "token_graph_rebuilds",
    "speculative_graph_rebuilds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pre-capture DSpark projection microbenchmark and select "
            "only block/N variants at least 3% faster than legacy."
        )
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dspark", required=True, type=Path)
    parser.add_argument("--binary", type=Path, default=Path("./ds4-bench"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("benchmark-results/gb10-projection-blocks"),
    )
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--min-block-gain", type=float, default=3.0)
    parser.add_argument(
        "--policies", default=",".join(POLICIES),
        help="comma-separated diagnostic policies",
    )
    args = parser.parse_args()
    if args.iterations < 1 or args.repeats < 1:
        parser.error("--iterations and --repeats must be positive")
    if args.min_block_gain < 0.0:
        parser.error("--min-block-gain must be non-negative")
    policies = tuple(item.strip() for item in args.policies.split(",") if item.strip())
    if not policies or len(policies) != len(set(policies)):
        parser.error("--policies must contain unique values")
    invalid = sorted(set(policies) - set(POLICIES))
    if invalid:
        parser.error(f"invalid policies: {', '.join(invalid)}")
    if "legacy" not in policies:
        parser.error("--policies must include legacy")
    args.policy_values = policies
    for path, label in (
        (args.model, "model"),
        (args.dspark, "DSpark sidecar"),
        (args.binary, "benchmark binary"),
    ):
        if not path.is_file():
            parser.error(f"{label} does not exist: {path}")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=repo, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def git_worktree_diff(repo: Path) -> str:
    tracked = subprocess.run(
        ["git", "diff", "--binary", "--no-ext-diff"], cwd=repo, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    ).stdout
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    ).stdout
    chunks = [tracked]
    for raw_path in untracked.split(b"\0"):
        if not raw_path:
            continue
        proc = subprocess.run(
            ["git", "diff", "--no-index", "--binary", "--no-ext-diff",
             "--", "/dev/null", os.fsdecode(raw_path)],
            cwd=repo, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, check=False,
        )
        if proc.returncode in (0, 1):
            chunks.append(proc.stdout)
    return "".join(chunks)


def benchmark_env(policy: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "DS4_CUDA_PROJECTION_POLICY": "legacy",
        "DS4_CUDA_TARGET_PROJECTION_POLICY": policy,
        "DS4_CUDA_DSPARK_PROJECTION_POLICY": policy,
        "DS4_CUDA_MTP_PROJECTION_POLICY": "legacy",
        "DS4_CUDA_COPY_MODEL": "1",
        "DS4_CUDA_WEIGHT_CACHE_LIMIT_GB": "112",
        "DS4_CUDA_DROP_COPIED_MODEL_PAGES": "1",
        "DS4_CUDA_Q8_F16_CACHE_MB": "12288",
        "DS4_CUDA_COPY_SECONDARY_MODEL": "1",
        "DS4_CUDA_DSPARK_CACHE_PRIORITY": "1",
        "DS4_CUDA_DEFER_END_SYNC": "1",
        "DS4_CUDA_COALESCED_F16_MATMUL": "1",
        "DS4_CUDA_Q8_U16_LOADS": "1",
        "DS4_CUDA_Q8_BATCH_REUSE": "1",
        "DS4_CUDA_DSPARK_TENSOR_CORES": "1",
        "DS4_CUDA_DSPARK_TENSOR_CORES_Q8": "1",
        "DS4_CUDA_DSPARK_TC_PAD_N": "8",
    })
    env.pop("DS4_CUDA_PROJECTION_VERBOSE", None)
    env.pop("DS4_CUDA_TOKEN_GRAPH", None)
    env.pop("DS4_CUDA_DSPARK_GRAPH", None)
    return env


def run_one(
    args: argparse.Namespace,
    repo: Path,
    policy: str,
    repeat: int,
) -> list[dict[str, Any]]:
    stem = f"{policy}-r{repeat}"
    csv_path = args.output_dir / f"{stem}.csv"
    log_path = args.output_dir / f"{stem}.log"
    command = [
        str(args.binary.resolve()),
        "--cuda",
        "--model", str(args.model.resolve()),
        "--dspark", str(args.dspark.resolve()),
        "--projection-microbench",
        "--projection-microbench-iters", str(args.iterations),
        "--csv", str(csv_path.resolve()),
    ]
    started = time.monotonic()
    with log_path.open("wb") as log:
        proc = subprocess.run(
            command, cwd=repo, env=benchmark_env(policy),
            stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    wall_sec = time.monotonic() - started
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        raise RuntimeError(
            f"projection microbenchmark {stem} failed ({proc.returncode}):\n"
            + "\n".join(tail)
        )
    rows: list[dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as src:
        for row in csv.DictReader(src):
            parsed: dict[str, Any] = dict(row)
            for field in NUMERIC_FIELDS:
                parsed[field] = float(parsed[field])
            parsed["run"] = repeat
            parsed["wall_sec"] = wall_sec
            rows.append(parsed)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def medians(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["policy"]), str(row["block"]), int(row["n"]))
        grouped.setdefault(key, []).append(row)
    result: list[dict[str, Any]] = []
    for (policy, block, n), group in sorted(grouped.items()):
        item: dict[str, Any] = {
            "policy": policy,
            "block": block,
            "n": n,
            "runs": len(group),
        }
        for field in NUMERIC_FIELDS:
            if field == "n":
                continue
            item[field] = statistics.median(float(row[field]) for row in group)
        result.append(item)
    return result


def validate(
    summary: list[dict[str, Any]],
    policies: tuple[str, ...],
    repeats: int,
) -> list[str]:
    errors: list[str] = []
    indexed = {
        (str(row["policy"]), str(row["block"]), int(row["n"])): row
        for row in summary
    }
    expected = {
        (policy, block, n)
        for policy in policies for block in BLOCKS for n in range(2, 7)
    }
    missing = sorted(expected - set(indexed))
    if missing:
        errors.append(f"missing policy/block/N rows: {missing}")
    for key, row in sorted(indexed.items()):
        if int(row["runs"]) != repeats:
            errors.append(f"{key} has {row['runs']} runs, expected {repeats}")
        # Legacy may probe its historical tiny-TC/cache path and then recover
        # through the native Q8 kernel.  That is baseline behaviour, not a
        # failed candidate dispatch.  Candidate policies must remain
        # fallback-free so their timing measures the selected path itself.
        if key[0] != "legacy" and float(row["fallbacks"]) != 0.0:
            errors.append(f"{key} recorded {row['fallbacks']:g} fallbacks")
        if float(row["token_graph_rebuilds"]) != 0.0 or \
           float(row["speculative_graph_rebuilds"]) != 0.0:
            errors.append(f"{key} unexpectedly constructed a CUDA Graph")
        if float(row["us_per_call"]) <= 0.0:
            errors.append(f"{key} has a non-positive duration")
    return errors


def select_paths(
    summary: list[dict[str, Any]],
    min_gain: float,
) -> list[dict[str, Any]]:
    indexed = {
        (str(row["policy"]), str(row["block"]), int(row["n"])): row
        for row in summary
    }
    selections: list[dict[str, Any]] = []
    for block in BLOCKS:
        candidates = (
            ("auto",)
            if block in NARROW_F16_BLOCKS
            else tuple(
                policy for policy in POLICIES
                if policy != "legacy" and
                   (block == "indexer_compressor_pair" or policy != "auto")
            )
        )
        for n in range(2, 7):
            legacy = indexed.get(("legacy", block, n))
            if legacy is None:
                continue
            legacy_us = float(legacy["us_per_call"])
            ranked = []
            for policy in candidates:
                row = indexed.get((policy, block, n))
                if row is None:
                    continue
                us = float(row["us_per_call"])
                gain = 100.0 * (legacy_us / us - 1.0)
                ranked.append((us, policy, gain))
            ranked.sort()
            if ranked and ranked[0][2] >= min_gain:
                selected = ranked[0][1]
                selected_us = ranked[0][0]
                gain_pct = ranked[0][2]
            else:
                selected = "legacy"
                selected_us = legacy_us
                gain_pct = 0.0
            selections.append({
                "block": block,
                "n": n,
                "selected_policy": selected,
                "legacy_us": legacy_us,
                "selected_us": selected_us,
                "gain_pct": gain_pct,
                "passes_3pct_gate": gain_pct >= min_gain,
            })
    return selections


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parent.parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    worktree_diff_path = args.output_dir / "worktree.diff"
    worktree_diff_path.write_text(git_worktree_diff(repo), encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for repeat in range(1, args.repeats + 1):
        for policy in args.policy_values:
            rows.extend(run_one(args, repo, policy, repeat))
    summary = medians(rows)
    errors = validate(summary, args.policy_values, args.repeats)
    selections = select_paths(summary, args.min_block_gain)
    write_csv(args.output_dir / "raw.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "selection.csv", selections)
    metadata = {
        "schema": 1,
        "created_unix": int(time.time()),
        "git_revision": git_value(repo, "rev-parse", "HEAD"),
        "git_status": git_value(repo, "status", "--short"),
        "worktree_diff": str(worktree_diff_path),
        "worktree_diff_sha256": sha256(worktree_diff_path),
        "binary": str(args.binary.resolve()),
        "binary_sha256": sha256(args.binary),
        "model": str(args.model.resolve()),
        "model_bytes": args.model.stat().st_size,
        "model_mtime_ns": args.model.stat().st_mtime_ns,
        "dspark": str(args.dspark.resolve()),
        "dspark_bytes": args.dspark.stat().st_size,
        "dspark_mtime_ns": args.dspark.stat().st_mtime_ns,
        "iterations": args.iterations,
        "n5_iterations": args.iterations * 2,
        "repeats": args.repeats,
        "policies": args.policy_values,
        "min_block_gain_pct": args.min_block_gain,
        "validation": "failed" if errors else "passed",
        "validation_errors": errors,
        "selection": selections,
        "median": summary,
    }
    result_path = args.output_dir / "summary.json"
    result_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {result_path}")
    for error in errors:
        print(f"benchmark_projection_blocks: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
