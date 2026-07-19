#!/usr/bin/env python3
"""Reproducible GB10 prefill/decode benchmark with median aggregation."""

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


DEFAULT_FRONTIERS = (12288, 32768, 65536, 81920, 98304)
NUMERIC_COLUMNS = (
    "ctx_tokens",
    "prefill_tokens",
    "prefill_tps",
    "gen_tokens",
    "gen_tps",
    "kvcache_bytes",
    "decode_cycles",
    "tokens_per_cycle",
    "startup_sec",
    "startup_rss_bytes",
    "startup_hwm_bytes",
    "ready_rss_bytes",
    "ready_hwm_bytes",
    "prefill_rss_bytes",
    "prefill_hwm_bytes",
    "decode_rss_bytes",
    "decode_hwm_bytes",
    "wall_sec",
    "polled_peak_rss_bytes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cold and append DS4 GB10 sweeps and emit raw/median CSV plus JSON."
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dspark", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--binary", type=Path, default=Path("./ds4-bench"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-results/gb10"))
    parser.add_argument("--frontiers", default=",".join(str(v) for v in DEFAULT_FRONTIERS))
    parser.add_argument("--ctx", type=int, default=131072)
    parser.add_argument("--prefill-chunk", type=int, default=8192)
    parser.add_argument("--gen-tokens", type=int, default=256)
    parser.add_argument("--dspark-draft", type=int, default=5)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cold-only", action="store_true")
    parser.add_argument("--append-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.cold_only and args.append_only:
        parser.error("--cold-only and --append-only are mutually exclusive")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.ctx <= args.gen_tokens + 1:
        parser.error("--ctx must leave room for generated tokens")
    frontiers = []
    for item in args.frontiers.split(","):
        try:
            value = int(item)
        except ValueError:
            parser.error(f"invalid frontier: {item!r}")
        if value <= 0 or (frontiers and value <= frontiers[-1]):
            parser.error("--frontiers must be strictly increasing positive integers")
        if value + args.gen_tokens >= args.ctx:
            parser.error(f"frontier {value} leaves no decode room in ctx={args.ctx}")
        frontiers.append(value)
    args.frontier_values = tuple(frontiers)
    return args


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(cwd: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def process_rss_bytes(pid: int) -> int:
    try:
        text = Path(f"/proc/{pid}/status").read_text(encoding="ascii", errors="replace")
    except OSError:
        return 0
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            fields = line.split()
            if len(fields) >= 2:
                return int(fields[1]) * 1024
    return 0


def benchmark_env() -> dict[str, str]:
    env = os.environ.copy()
    defaults = {
        "DS4_CUDA_COPY_MODEL": "1",
        "DS4_CUDA_WEIGHT_CACHE_LIMIT_GB": "112",
        "DS4_CUDA_DROP_COPIED_MODEL_PAGES": "1",
        "DS4_CUDA_Q8_F16_CACHE_MB": "12288",
        "DS4_CUDA_COPY_SECONDARY_MODEL": "1",
        "DS4_CUDA_DSPARK_CACHE_PRIORITY": "1",
        "DS4_CUDA_DEFER_END_SYNC": "1",
        "DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS": "0",
        "DS4_CUDA_FUSED_COMPRESSOR_UPDATE": "1",
        "DS4_CUDA_TOKEN_GRAPH": "1",
        "DS4_CUDA_DSPARK_GRAPH": "1",
        "DS4_CUDA_COALESCED_F16_MATMUL": "1",
        "DS4_CUDA_Q8_U16_LOADS": "1",
        "DS4_CUDA_Q8_BATCH_REUSE": "1",
        "DS4_CUDA_MOE_TINY_DIRECT": "1",
        "DS4_CUDA_MOE_TINY_DIRECT_Q4_ONLY": "1",
        "DS4_CUDA_DSPARK_TENSOR_CORES": "1",
        "DS4_CUDA_DSPARK_TENSOR_CORES_Q8": "1",
        "DS4_CUDA_DSPARK_TC_PAD_N": "8",
        "DS4_DSPARK_ALWAYS_DRAFT": "1",
        "DS4_DSPARK_NO_CIRCUIT_BREAKER": "1",
        "DS4_PREFILL_FINAL_LOGITS_ONLY": "1",
    }
    for name, value in defaults.items():
        env.setdefault(name, value)
    for name in ("DS4_CUDA_TOKEN_GRAPH_PIPELINE", "DS4_CUDA_MTP_GRAPH",
                 "DS4_CUDA_MTP_TENSOR_CORES"):
        env.pop(name, None)
    return env


def run_one(
    args: argparse.Namespace,
    mode: str,
    repeat: int,
    repo: Path,
) -> list[dict[str, Any]]:
    run_dir = args.output_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{mode}-run-{repeat:02d}"
    csv_path = run_dir / f"{stem}.csv"
    log_path = run_dir / f"{stem}.log"
    command = [
        str(args.binary),
        "--cuda",
        "--model",
        str(args.model),
        "--dspark",
        str(args.dspark),
        "--dspark-draft",
        str(args.dspark_draft),
        "--prompt-file",
        str(args.prompt),
        "--frontiers",
        ",".join(str(v) for v in args.frontier_values),
        "--ctx-alloc",
        str(args.ctx),
        "--prefill-chunk",
        str(args.prefill_chunk),
        "--gen-tokens",
        str(args.gen_tokens),
        "--threads",
        str(args.threads),
        "--csv",
        str(csv_path),
    ]
    if mode == "cold":
        command.append("--cold-sweep")
    print(" ".join(command), flush=True)
    if args.dry_run:
        return []

    started = time.monotonic()
    peak_rss = 0
    with log_path.open("wb") as log:
        proc = subprocess.Popen(
            command,
            cwd=repo,
            env=benchmark_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        while proc.poll() is None:
            peak_rss = max(peak_rss, process_rss_bytes(proc.pid))
            time.sleep(0.1)
        peak_rss = max(peak_rss, process_rss_bytes(proc.pid))
    wall_sec = time.monotonic() - started
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        raise RuntimeError(
            f"{mode} run {repeat} failed with status {proc.returncode}:\n" + "\n".join(tail)
        )

    rows: list[dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as src:
        for row in csv.DictReader(src):
            parsed: dict[str, Any] = dict(row)
            for name in NUMERIC_COLUMNS:
                if name in parsed:
                    parsed[name] = float(parsed[name])
            parsed["run"] = repeat
            parsed["wall_sec"] = wall_sec
            parsed["polled_peak_rss_bytes"] = peak_rss
            rows.append(parsed)
    return rows


def write_raw_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def median_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["mode"]), int(float(row["ctx_tokens"])))
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (mode, ctx), group in sorted(groups.items()):
        item: dict[str, Any] = {"mode": mode, "ctx_tokens": ctx, "runs": len(group)}
        for name in NUMERIC_COLUMNS:
            if name == "ctx_tokens" or name not in group[0]:
                continue
            item[name] = statistics.median(float(row[name]) for row in group)
        token_hashes = sorted({str(row.get("greedy_token_hash", "")) for row in group})
        item["greedy_token_hash"] = (
            token_hashes[0] if len(token_hashes) == 1
            else "MISMATCH:" + ";".join(token_hashes)
        )
        out.append(item)
    return out


def validate_results(
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    modes: list[str],
    frontiers: tuple[int, ...],
    repeats: int,
) -> list[str]:
    errors: list[str] = []
    expected = {(mode, frontier) for mode in modes for frontier in frontiers}
    actual = {(str(row["mode"]), int(float(row["ctx_tokens"]))) for row in rows}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        errors.append(f"missing mode/frontier groups: {missing}")
    if extra:
        errors.append(f"unexpected mode/frontier groups: {extra}")

    for item in summary:
        key = (str(item["mode"]), int(item["ctx_tokens"]))
        if int(item["runs"]) != repeats:
            errors.append(f"{key} has {item['runs']} runs, expected {repeats}")
        token_hash = str(item.get("greedy_token_hash", ""))
        if not token_hash or token_hash.startswith("MISMATCH:"):
            errors.append(f"{key} greedy token hash is not deterministic: {token_hash}")
    return errors


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parent.parent
    for path, label in (
        (args.binary, "benchmark binary"),
        (args.model, "target model"),
        (args.dspark, "DSpark sidecar"),
        (args.prompt, "prompt"),
    ):
        candidate = path if path.is_absolute() else repo / path
        if not candidate.is_file() and not args.dry_run:
            print(f"benchmark_gb10: missing {label}: {candidate}", file=sys.stderr)
            return 2
    args.binary = args.binary if args.binary.is_absolute() else repo / args.binary
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    args.model = args.model if args.model.is_absolute() else repo / args.model
    args.dspark = args.dspark if args.dspark.is_absolute() else repo / args.dspark
    args.prompt = args.prompt if args.prompt.is_absolute() else repo / args.prompt
    args.output_dir.mkdir(parents=True, exist_ok=True)

    modes = ["cold", "append"]
    if args.cold_only:
        modes = ["cold"]
    elif args.append_only:
        modes = ["append"]

    rows: list[dict[str, Any]] = []
    for repeat in range(1, args.repeats + 1):
        for mode in modes:
            rows.extend(run_one(args, mode, repeat, repo))
    if args.dry_run:
        return 0

    summary = median_rows(rows)
    validation_errors = validate_results(
        rows, summary, modes, args.frontier_values, args.repeats
    )
    write_raw_csv(args.output_dir / "raw.csv", rows)
    write_raw_csv(args.output_dir / "summary.csv", summary)
    metadata = {
        "schema": 2,
        "created_unix": int(time.time()),
        "git_revision": git_revision(repo),
        "prompt": str(args.prompt),
        "prompt_sha256": file_sha256(args.prompt),
        "model": str(args.model),
        "dspark": str(args.dspark),
        "ctx": args.ctx,
        "prefill_chunk": args.prefill_chunk,
        "frontiers": args.frontier_values,
        "gen_tokens": args.gen_tokens,
        "dspark_draft": args.dspark_draft,
        "repeats": args.repeats,
        "validation": "failed" if validation_errors else "passed",
        "validation_errors": validation_errors,
        "median": summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output_dir / 'summary.json'}")
    if validation_errors:
        for error in validation_errors:
            print(f"benchmark_gb10: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
