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
PROJECTION_POLICIES = (
    "legacy", "auto", "f16-real", "f16-pad8", "f16-out", "q8-reuse"
)
AUTO_COMPONENTS = ("q8-reuse", "f16-gemm", "f16-out", "narrow-f16")
NUMERIC_COLUMNS = (
    "ctx_tokens",
    "prefill_tokens",
    "prefill_tps",
    "gen_tokens",
    "gen_tps",
    "kvcache_bytes",
    "decode_cycles",
    "tokens_per_cycle",
    "spec_cycles",
    "drafted_tokens",
    "target_rows",
    "committed_tokens",
    "emitted_tokens",
    "draft_ms",
    "target_ms",
    "total_ms",
    "drafts_per_cycle",
    "target_rows_per_cycle",
    "emitted_per_cycle",
    "acceptance",
    "startup_sec",
    "startup_rss_bytes",
    "startup_hwm_bytes",
    "ready_rss_bytes",
    "ready_hwm_bytes",
    "prefill_rss_bytes",
    "prefill_hwm_bytes",
    "decode_rss_bytes",
    "decode_hwm_bytes",
    "token_graph_rebuilds",
    "speculative_graph_rebuilds",
    "dspark_draft_graph_launches",
    "dspark_verifier_graph_launches",
    "projection_fallbacks",
    "projection_target_fallbacks",
    "projection_dspark_fallbacks",
    "projection_mtp_fallbacks",
    "projection_f16_resident_bytes",
    "token_graph_nodes",
    "speculative_graph_nodes",
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
    parser.add_argument(
        "--schedules",
        default="adaptive",
        help="comma-separated scheduler modes: adaptive and/or fixed-k1..fixed-k5",
    )
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--projection-policies",
        default="legacy",
        help="comma-separated CUDA projection policies to compare",
    )
    parser.add_argument(
        "--projection-scope",
        choices=("target", "dspark", "both"),
        default="target",
        help=(
            "apply each candidate policy only to the target verifier, only "
            "to DSpark decode, or to both (default: target)"
        ),
    )
    parser.add_argument(
        "--auto-components",
        default="all",
        help=(
            "diagnostic AUTO component list: all, none, or a comma-separated "
            "subset of q8-reuse,f16-gemm,f16-out,narrow-f16"
        ),
    )
    parser.add_argument(
        "--min-auto-decode-gain",
        type=float,
        default=0.0,
        help="fail unless auto improves aggregate decode throughput by this percent",
    )
    parser.add_argument(
        "--min-fixed-target-ms-saved",
        type=float,
        default=0.0,
        help="fail unless auto saves this many average target milliseconds on every fixed-K schedule",
    )
    parser.add_argument(
        "--min-auto-target-gain",
        type=float,
        default=0.0,
        help="minimum aggregate auto verifier target-time gain in percent",
    )
    parser.add_argument(
        "--min-auto-draft-gain",
        type=float,
        default=0.0,
        help="minimum aggregate AUTO DSpark draft-time gain in percent",
    )
    parser.add_argument(
        "--max-auto-target-regression",
        type=float,
        default=0.5,
        help="maximum auto target_ms regression at any measured frontier in percent",
    )
    parser.add_argument(
        "--projection-telemetry",
        action="store_true",
        help="enable verbose projection path/cache counters in per-run logs",
    )
    parser.add_argument(
        "--max-fixed-acceptance-delta",
        type=float,
        default=0.001,
        help="maximum absolute legacy/auto acceptance difference on fixed-K schedules",
    )
    parser.add_argument(
        "--max-auto-prefill-regression",
        type=float,
        default=1.0,
        help="maximum allowed auto prefill throughput regression in percent",
    )
    parser.add_argument(
        "--max-auto-startup-regression",
        type=float,
        default=5.0,
        help="maximum allowed auto startup-time regression in percent",
    )
    parser.add_argument(
        "--max-auto-memory-growth-mib",
        type=float,
        default=256.0,
        help="maximum allowed auto process peak-memory growth in MiB",
    )
    parser.add_argument(
        "--max-auto-projection-fallbacks",
        type=int,
        default=0,
        help="maximum allowed median projection fallbacks per auto decode frontier",
    )
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
    if args.dspark_draft < 1 or args.dspark_draft > 5:
        parser.error("--dspark-draft must be in 1..5")
    schedules = tuple(item.strip() for item in args.schedules.split(",") if item.strip())
    if not schedules or len(set(schedules)) != len(schedules):
        parser.error("--schedules must contain unique scheduler modes")
    for schedule in schedules:
        if schedule == "adaptive":
            continue
        if not schedule.startswith("fixed-k") or not schedule[7:].isdigit():
            parser.error(f"invalid scheduler mode: {schedule}")
        fixed_k = int(schedule[7:])
        if fixed_k < 1 or fixed_k > 5:
            parser.error(f"fixed scheduler K must be in 1..5: {schedule}")
    args.schedule_values = schedules
    policies = tuple(item.strip() for item in args.projection_policies.split(",") if item.strip())
    if not policies or len(set(policies)) != len(policies):
        parser.error("--projection-policies must contain unique policy names")
    invalid_policies = sorted(set(policies) - set(PROJECTION_POLICIES))
    if invalid_policies:
        parser.error(f"invalid projection policies: {', '.join(invalid_policies)}")
    if args.auto_components not in {"all", "none"}:
        auto_components = tuple(
            item.strip() for item in args.auto_components.split(",") if item.strip()
        )
        if not auto_components or len(set(auto_components)) != len(auto_components):
            parser.error("--auto-components must contain unique component names")
        invalid_components = sorted(set(auto_components) - set(AUTO_COMPONENTS))
        if invalid_components:
            parser.error(
                f"invalid AUTO components: {', '.join(invalid_components)}"
            )
        args.auto_components = ",".join(auto_components)
    if args.min_auto_decode_gain < 0.0:
        parser.error("--min-auto-decode-gain must be non-negative")
    if args.min_fixed_target_ms_saved < 0.0:
        parser.error("--min-fixed-target-ms-saved must be non-negative")
    if args.min_auto_target_gain < 0.0:
        parser.error("--min-auto-target-gain must be non-negative")
    if args.min_auto_draft_gain < 0.0:
        parser.error("--min-auto-draft-gain must be non-negative")
    if args.max_auto_target_regression < 0.0:
        parser.error("--max-auto-target-regression must be non-negative")
    if args.max_fixed_acceptance_delta < 0.0 or args.max_fixed_acceptance_delta > 1.0:
        parser.error("--max-fixed-acceptance-delta must be in 0..1")
    if args.max_auto_prefill_regression < 0.0:
        parser.error("--max-auto-prefill-regression must be non-negative")
    if args.max_auto_startup_regression < 0.0:
        parser.error("--max-auto-startup-regression must be non-negative")
    if args.max_auto_memory_growth_mib < 0.0:
        parser.error("--max-auto-memory-growth-mib must be non-negative")
    if args.max_auto_projection_fallbacks < 0:
        parser.error("--max-auto-projection-fallbacks must be non-negative")
    if args.min_auto_decode_gain and not {"legacy", "auto"}.issubset(policies):
        parser.error("--min-auto-decode-gain requires legacy and auto policies")
    if args.min_auto_target_gain and not {"legacy", "auto"}.issubset(policies):
        parser.error("--min-auto-target-gain requires legacy and auto policies")
    if args.min_auto_draft_gain:
        if not {"legacy", "auto"}.issubset(policies):
            parser.error("--min-auto-draft-gain requires legacy and auto policies")
        if args.projection_scope not in {"dspark", "both"}:
            parser.error("--min-auto-draft-gain requires dspark or both scope")
    if args.min_fixed_target_ms_saved:
        if not {"legacy", "auto"}.issubset(policies):
            parser.error("--min-fixed-target-ms-saved requires legacy and auto policies")
        if not any(schedule.startswith("fixed-k") for schedule in schedules):
            parser.error("--min-fixed-target-ms-saved requires a fixed-k schedule")
    args.projection_policy_values = policies
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


def git_status(cwd: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--short"], cwd=cwd, text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown\n"


def git_worktree_diff(cwd: Path) -> str:
    try:
        tracked = subprocess.check_output(
            ["git", "diff", "--binary", "--no-ext-diff"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        untracked_raw = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
        )
        chunks = [tracked]
        for raw_path in untracked_raw.split(b"\0"):
            if not raw_path:
                continue
            rel_path = os.fsdecode(raw_path)
            proc = subprocess.run(
                ["git", "diff", "--no-index", "--binary", "--no-ext-diff",
                 "--", "/dev/null", rel_path],
                cwd=cwd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if proc.returncode in (0, 1):
                chunks.append(proc.stdout)
        return "".join(chunks)
    except (OSError, subprocess.CalledProcessError):
        return ""


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


def benchmark_env(
    projection_policy: str,
    schedule: str,
    projection_telemetry: bool,
    auto_components: str,
    projection_scope: str,
) -> dict[str, str]:
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
    env["DS4_CUDA_PROJECTION_POLICY"] = "legacy"
    env["DS4_CUDA_TARGET_PROJECTION_POLICY"] = (
        projection_policy if projection_scope in {"target", "both"} else "legacy"
    )
    env["DS4_CUDA_DSPARK_PROJECTION_POLICY"] = (
        projection_policy if projection_scope in {"dspark", "both"} else "legacy"
    )
    env["DS4_CUDA_MTP_PROJECTION_POLICY"] = "legacy"
    env["DS4_CUDA_PROJECTION_AUTO_COMPONENTS"] = auto_components
    if schedule.startswith("fixed-k"):
        env["DS4_DSPARK_FIXED_VERIFY"] = "1"
    else:
        env.pop("DS4_DSPARK_FIXED_VERIFY", None)
    if projection_telemetry:
        env["DS4_CUDA_PROJECTION_VERBOSE"] = "1"
    else:
        env.pop("DS4_CUDA_PROJECTION_VERBOSE", None)
    for name in ("DS4_CUDA_TOKEN_GRAPH_PIPELINE", "DS4_CUDA_MTP_GRAPH",
                 "DS4_CUDA_MTP_TENSOR_CORES"):
        env.pop(name, None)
    return env


def run_one(
    args: argparse.Namespace,
    projection_policy: str,
    schedule: str,
    mode: str,
    repeat: int,
    repo: Path,
) -> list[dict[str, Any]]:
    run_dir = args.output_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{projection_policy}-{schedule}-{mode}-run-{repeat:02d}"
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
        str(int(schedule[7:]) if schedule.startswith("fixed-k") else args.dspark_draft),
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
            env=benchmark_env(
                projection_policy,
                schedule,
                args.projection_telemetry,
                args.auto_components,
                args.projection_scope,
            ),
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
            parsed["projection_policy"] = projection_policy
            parsed["schedule"] = schedule
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
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["projection_policy"]),
            str(row["schedule"]),
            str(row["mode"]),
            int(float(row["ctx_tokens"])),
        )
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (policy, schedule, mode, ctx), group in sorted(groups.items()):
        item: dict[str, Any] = {
            "projection_policy": policy,
            "schedule": schedule,
            "mode": mode,
            "ctx_tokens": ctx,
            "runs": len(group),
        }
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
    policies: tuple[str, ...],
    schedules: tuple[str, ...],
    modes: list[str],
    frontiers: tuple[int, ...],
    repeats: int,
) -> list[str]:
    errors: list[str] = []
    expected = {
        (policy, schedule, mode, frontier)
        for policy in policies
        for schedule in schedules
        for mode in modes
        for frontier in frontiers
    }
    actual = {
        (
            str(row["projection_policy"]),
            str(row["schedule"]),
            str(row["mode"]),
            int(float(row["ctx_tokens"])),
        )
        for row in rows
    }
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        errors.append(f"missing mode/frontier groups: {missing}")
    if extra:
        errors.append(f"unexpected mode/frontier groups: {extra}")

    for item in summary:
        key = (
            str(item["projection_policy"]),
            str(item["schedule"]),
            str(item["mode"]),
            int(item["ctx_tokens"]),
        )
        if int(item["runs"]) != repeats:
            errors.append(f"{key} has {item['runs']} runs, expected {repeats}")
        token_hash = str(item.get("greedy_token_hash", ""))
        if not token_hash or token_hash.startswith("MISMATCH:"):
            errors.append(f"{key} greedy token hash is not deterministic: {token_hash}")
    cross_policy_hashes: dict[tuple[str, int], set[str]] = {}
    for item in summary:
        key = (str(item["mode"]), int(item["ctx_tokens"]))
        cross_policy_hashes.setdefault(key, set()).add(str(item.get("greedy_token_hash", "")))
    for key, hashes in sorted(cross_policy_hashes.items()):
        if len(hashes) != 1:
            errors.append(f"{key} greedy token hash differs across policies: {sorted(hashes)}")
    return errors


def validate_auto_resource_gates(
    summary: list[dict[str, Any]],
    projection_scope: str,
    max_prefill_regression_pct: float,
    max_startup_regression_pct: float,
    max_memory_growth_mib: float,
    max_projection_fallbacks: int,
    max_target_regression_pct: float,
) -> list[str]:
    indexed = {
        (
            str(item["projection_policy"]),
            str(item["schedule"]),
            str(item["mode"]),
            int(item["ctx_tokens"]),
        ): item
        for item in summary
    }
    errors: list[str] = []
    compared_startup: set[tuple[str, str]] = set()
    for (policy, schedule, mode, ctx), legacy in sorted(indexed.items()):
        if policy != "legacy":
            continue
        auto = indexed.get(("auto", schedule, mode, ctx))
        if auto is None:
            continue
        legacy_prefill = float(legacy.get("prefill_tps", 0.0))
        auto_prefill = float(auto.get("prefill_tps", 0.0))
        if legacy_prefill > 0.0 and auto_prefill > 0.0:
            regression = 100.0 * (1.0 - auto_prefill / legacy_prefill)
            if regression > max_prefill_regression_pct:
                errors.append(
                    f"{schedule}/{mode}/{ctx} auto prefill regression "
                    f"{regression:.2f}% exceeds {max_prefill_regression_pct:.2f}%"
                )

        startup_key = (schedule, mode)
        if startup_key not in compared_startup:
            compared_startup.add(startup_key)
            legacy_startup = float(legacy.get("startup_sec", 0.0))
            auto_startup = float(auto.get("startup_sec", 0.0))
            if legacy_startup > 0.0 and auto_startup > 0.0:
                regression = 100.0 * (auto_startup / legacy_startup - 1.0)
                if regression > max_startup_regression_pct:
                    errors.append(
                        f"{schedule}/{mode} auto startup regression "
                        f"{regression:.2f}% exceeds {max_startup_regression_pct:.2f}%"
                    )

        memory_fields = (
            "ready_hwm_bytes",
            "decode_hwm_bytes",
            "polled_peak_rss_bytes",
        )
        legacy_peak = max(float(legacy.get(name, 0.0)) for name in memory_fields)
        auto_peak = max(float(auto.get(name, 0.0)) for name in memory_fields)
        growth_mib = (auto_peak - legacy_peak) / 1048576.0
        if growth_mib > max_memory_growth_mib:
            errors.append(
                f"{schedule}/{mode}/{ctx} auto peak-memory growth "
                f"{growth_mib:.1f} MiB exceeds {max_memory_growth_mib:.1f} MiB"
            )

        if projection_scope == "target":
            fallback_fields = ("projection_target_fallbacks",)
        elif projection_scope == "dspark":
            fallback_fields = ("projection_dspark_fallbacks",)
        else:
            fallback_fields = (
                "projection_target_fallbacks",
                "projection_dspark_fallbacks",
            )
        legacy_fallbacks = sum(
            float(legacy.get(field, 0.0)) for field in fallback_fields
        )
        auto_fallbacks = sum(
            float(auto.get(field, 0.0)) for field in fallback_fields
        )
        fallback_growth = auto_fallbacks - legacy_fallbacks
        if fallback_growth > float(max_projection_fallbacks):
            errors.append(
                f"{schedule}/{mode}/{ctx} auto {projection_scope} projection "
                f"fallback growth {fallback_growth:g} exceeds "
                f"{max_projection_fallbacks} "
                f"(legacy={legacy_fallbacks:g}, auto={auto_fallbacks:g})"
            )

        legacy_target_ms = float(legacy.get("target_ms", 0.0))
        auto_target_ms = float(auto.get("target_ms", 0.0))
        if legacy_target_ms > 0.0 and auto_target_ms > 0.0:
            target_regression = 100.0 * (
                auto_target_ms / legacy_target_ms - 1.0
            )
            if target_regression > max_target_regression_pct:
                errors.append(
                    f"{schedule}/{mode}/{ctx} auto target_ms regression "
                    f"{target_regression:.3f}% exceeds "
                    f"{max_target_regression_pct:.3f}%"
                )

        for field, label in (
            ("token_graph_rebuilds", "token-graph rebuilds"),
            ("speculative_graph_rebuilds", "speculative-graph rebuilds"),
        ):
            legacy_value = float(legacy.get(field, 0.0))
            auto_value = float(auto.get(field, 0.0))
            if auto_value > legacy_value:
                errors.append(
                    f"{schedule}/{mode}/{ctx} auto {label} {auto_value:g} "
                    f"exceed legacy {legacy_value:g}"
                )

        for field, label in (
            ("token_graph_nodes", "token-graph nodes"),
            ("speculative_graph_nodes", "speculative-graph nodes"),
        ):
            legacy_value = float(legacy.get(field, 0.0))
            auto_value = float(auto.get(field, 0.0))
            if auto_value > legacy_value:
                errors.append(
                    f"{schedule}/{mode}/{ctx} auto {label} {auto_value:g} "
                    f"exceed legacy {legacy_value:g}"
                )

        legacy_f16_bytes = float(
            legacy.get("projection_f16_resident_bytes", 0.0)
        )
        auto_f16_bytes = float(auto.get("projection_f16_resident_bytes", 0.0))
        if auto_f16_bytes > legacy_f16_bytes:
            errors.append(
                f"{schedule}/{mode}/{ctx} auto resident FP16 cache "
                f"{auto_f16_bytes / 1048576.0:.1f} MiB exceeds legacy "
                f"{legacy_f16_bytes / 1048576.0:.1f} MiB"
            )
    return errors


def aggregate_decode_tps(summary: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    totals: dict[tuple[str, str], tuple[float, float]] = {}
    for item in summary:
        policy = str(item["projection_policy"])
        schedule = str(item["schedule"])
        tokens = float(item.get("gen_tokens", 0.0))
        tps = float(item.get("gen_tps", 0.0))
        if tokens <= 0.0 or tps <= 0.0:
            continue
        key = (schedule, policy)
        total_tokens, total_seconds = totals.get(key, (0.0, 0.0))
        totals[key] = (total_tokens + tokens, total_seconds + tokens / tps)
    out: dict[str, dict[str, float]] = {}
    for (schedule, policy), (tokens, seconds) in totals.items():
        if tokens > 0.0 and seconds > 0.0:
            out.setdefault(schedule, {})[policy] = tokens / seconds
    return out


def aggregate_target_ms(summary: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    totals: dict[tuple[str, str], tuple[float, float]] = {}
    for item in summary:
        cycles = float(item.get("spec_cycles", 0.0))
        target_ms = float(item.get("target_ms", 0.0))
        if cycles <= 0.0 or target_ms <= 0.0:
            continue
        key = (str(item["schedule"]), str(item["projection_policy"]))
        weighted_ms, total_cycles = totals.get(key, (0.0, 0.0))
        totals[key] = (weighted_ms + target_ms * cycles, total_cycles + cycles)
    out: dict[str, dict[str, float]] = {}
    for (schedule, policy), (weighted_ms, cycles) in totals.items():
        if cycles > 0.0:
            out.setdefault(schedule, {})[policy] = weighted_ms / cycles
    return out


def aggregate_draft_ms(summary: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    totals: dict[tuple[str, str], tuple[float, float]] = {}
    for item in summary:
        cycles = float(item.get("spec_cycles", 0.0))
        draft_ms = float(item.get("draft_ms", 0.0))
        if cycles <= 0.0 or draft_ms <= 0.0:
            continue
        key = (str(item["schedule"]), str(item["projection_policy"]))
        weighted_ms, total_cycles = totals.get(key, (0.0, 0.0))
        totals[key] = (weighted_ms + draft_ms * cycles, total_cycles + cycles)
    out: dict[str, dict[str, float]] = {}
    for (schedule, policy), (weighted_ms, cycles) in totals.items():
        if cycles > 0.0:
            out.setdefault(schedule, {})[policy] = weighted_ms / cycles
    return out


def aggregate_acceptance(summary: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    totals: dict[tuple[str, str], tuple[float, float]] = {}
    for item in summary:
        drafted = float(item.get("drafted_tokens", 0.0))
        committed = float(item.get("committed_tokens", 0.0))
        if drafted <= 0.0:
            continue
        key = (str(item["schedule"]), str(item["projection_policy"]))
        total_committed, total_drafted = totals.get(key, (0.0, 0.0))
        totals[key] = (total_committed + committed, total_drafted + drafted)
    out: dict[str, dict[str, float]] = {}
    for (schedule, policy), (committed, drafted) in totals.items():
        if drafted > 0.0:
            out.setdefault(schedule, {})[policy] = committed / drafted
    return out


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
    worktree_diff_path = args.output_dir / "worktree.diff"
    if not args.dry_run:
        worktree_diff_path.write_text(git_worktree_diff(repo), encoding="utf-8")

    modes = ["cold", "append"]
    if args.cold_only:
        modes = ["cold"]
    elif args.append_only:
        modes = ["append"]

    rows: list[dict[str, Any]] = []
    for repeat in range(1, args.repeats + 1):
        for projection_policy in args.projection_policy_values:
            for schedule in args.schedule_values:
                for mode in modes:
                    rows.extend(run_one(
                        args, projection_policy, schedule, mode, repeat, repo
                    ))
    if args.dry_run:
        return 0

    summary = median_rows(rows)
    validation_errors = validate_results(
        rows, summary, args.projection_policy_values,
        args.schedule_values, modes, args.frontier_values, args.repeats
    )
    if {"legacy", "auto"}.issubset(args.projection_policy_values):
        validation_errors.extend(validate_auto_resource_gates(
            summary,
            args.projection_scope,
            args.max_auto_prefill_regression,
            args.max_auto_startup_regression,
            args.max_auto_memory_growth_mib,
            args.max_auto_projection_fallbacks,
            args.max_auto_target_regression,
        ))
    aggregate_tps = aggregate_decode_tps(summary)
    aggregate_target = aggregate_target_ms(summary)
    aggregate_draft = aggregate_draft_ms(summary)
    acceptance = aggregate_acceptance(summary)
    auto_gain_pct: dict[str, float] = {}
    auto_target_gain_pct: dict[str, float] = {}
    auto_target_ms_saved: dict[str, float] = {}
    auto_draft_gain_pct: dict[str, float] = {}
    for schedule, schedule_tps in aggregate_tps.items():
        if "legacy" in schedule_tps and "auto" in schedule_tps:
            auto_gain_pct[schedule] = 100.0 * (
                schedule_tps["auto"] / schedule_tps["legacy"] - 1.0
            )
    for schedule, schedule_target in aggregate_target.items():
        if "legacy" in schedule_target and "auto" in schedule_target:
            auto_target_ms_saved[schedule] = (
                schedule_target["legacy"] - schedule_target["auto"]
            )
            auto_target_gain_pct[schedule] = 100.0 * (
                1.0 - schedule_target["auto"] / schedule_target["legacy"]
            )
    for schedule, schedule_draft in aggregate_draft.items():
        if "legacy" in schedule_draft and "auto" in schedule_draft:
            auto_draft_gain_pct[schedule] = 100.0 * (
                1.0 - schedule_draft["auto"] / schedule_draft["legacy"]
            )
    if args.min_auto_decode_gain:
        for schedule in args.schedule_values:
            gain = auto_gain_pct.get(schedule)
            if gain is None:
                validation_errors.append(
                    f"cannot compute legacy-to-auto decode gain for {schedule}"
                )
            elif gain < args.min_auto_decode_gain:
                validation_errors.append(
                    f"{schedule} auto decode gain {gain:.2f}% is below "
                    f"{args.min_auto_decode_gain:.2f}%"
                )
    if args.min_auto_target_gain:
        for schedule in args.schedule_values:
            gain = auto_target_gain_pct.get(schedule)
            if gain is None:
                validation_errors.append(
                    f"cannot compute legacy-to-auto verifier target gain for {schedule}"
                )
            elif gain < args.min_auto_target_gain:
                validation_errors.append(
                    f"{schedule} auto verifier target gain {gain:.2f}% is below "
                    f"{args.min_auto_target_gain:.2f}%"
                )
    if args.min_auto_draft_gain:
        for schedule in args.schedule_values:
            gain = auto_draft_gain_pct.get(schedule)
            if gain is None:
                validation_errors.append(
                    f"cannot compute legacy-to-auto draft gain for {schedule}"
                )
            elif gain < args.min_auto_draft_gain:
                validation_errors.append(
                    f"{schedule} auto draft gain {gain:.2f}% is below "
                    f"{args.min_auto_draft_gain:.2f}%"
                )
    if args.min_fixed_target_ms_saved:
        for schedule in args.schedule_values:
            if not schedule.startswith("fixed-k"):
                continue
            saved = auto_target_ms_saved.get(schedule)
            if saved is None:
                validation_errors.append(
                    f"cannot compute legacy-to-auto target saving for {schedule}"
                )
            elif saved < args.min_fixed_target_ms_saved:
                validation_errors.append(
                    f"{schedule} auto target saving {saved:.3f} ms is below "
                    f"{args.min_fixed_target_ms_saved:.3f} ms"
                )
    if {"legacy", "auto"}.issubset(args.projection_policy_values):
        for schedule in args.schedule_values:
            if not schedule.startswith("fixed-k"):
                continue
            schedule_acceptance = acceptance.get(schedule, {})
            if "legacy" not in schedule_acceptance or "auto" not in schedule_acceptance:
                validation_errors.append(
                    f"cannot compare legacy/auto acceptance for {schedule}"
                )
                continue
            delta = abs(
                schedule_acceptance["auto"] - schedule_acceptance["legacy"]
            )
            if delta > args.max_fixed_acceptance_delta:
                validation_errors.append(
                    f"{schedule} acceptance delta {delta:.6f} exceeds "
                    f"{args.max_fixed_acceptance_delta:.6f}"
                )
    write_raw_csv(args.output_dir / "raw.csv", rows)
    write_raw_csv(args.output_dir / "summary.csv", summary)
    metadata = {
        "schema": 4,
        "created_unix": int(time.time()),
        "git_revision": git_revision(repo),
        "git_status": git_status(repo),
        "worktree_diff": str(worktree_diff_path),
        "worktree_diff_sha256": file_sha256(worktree_diff_path),
        "binary": str(args.binary),
        "binary_sha256": file_sha256(args.binary),
        "binary_bytes": args.binary.stat().st_size,
        "prompt": str(args.prompt),
        "prompt_sha256": file_sha256(args.prompt),
        "model": str(args.model),
        "model_bytes": args.model.stat().st_size,
        "model_mtime_ns": args.model.stat().st_mtime_ns,
        "dspark": str(args.dspark),
        "dspark_bytes": args.dspark.stat().st_size,
        "dspark_mtime_ns": args.dspark.stat().st_mtime_ns,
        "ctx": args.ctx,
        "prefill_chunk": args.prefill_chunk,
        "frontiers": args.frontier_values,
        "gen_tokens": args.gen_tokens,
        "dspark_draft": args.dspark_draft,
        "schedules": args.schedule_values,
        "repeats": args.repeats,
        "projection_policies": args.projection_policy_values,
        "projection_scope": args.projection_scope,
        "auto_components": args.auto_components,
        "projection_telemetry": args.projection_telemetry,
        "ds4_environment": {
            name: value
            for name, value in sorted(
                benchmark_env(
                    "legacy", "adaptive", False, args.auto_components,
                    args.projection_scope
                ).items()
            )
            if name.startswith("DS4_") and
               name not in {
                   "DS4_CUDA_PROJECTION_POLICY",
                   "DS4_CUDA_TARGET_PROJECTION_POLICY",
                   "DS4_CUDA_DSPARK_PROJECTION_POLICY",
                   "DS4_CUDA_MTP_PROJECTION_POLICY",
               }
        },
        "aggregate_decode_tps": aggregate_tps,
        "aggregate_target_ms": aggregate_target,
        "aggregate_draft_ms": aggregate_draft,
        "aggregate_acceptance": acceptance,
        "auto_decode_gain_pct": auto_gain_pct,
        "auto_target_gain_pct": auto_target_gain_pct,
        "auto_draft_gain_pct": auto_draft_gain_pct,
        "auto_target_ms_saved": auto_target_ms_saved,
        "min_auto_decode_gain_pct": args.min_auto_decode_gain,
        "min_auto_target_gain_pct": args.min_auto_target_gain,
        "min_auto_draft_gain_pct": args.min_auto_draft_gain,
        "max_auto_target_regression_pct": args.max_auto_target_regression,
        "min_fixed_target_ms_saved": args.min_fixed_target_ms_saved,
        "max_fixed_acceptance_delta": args.max_fixed_acceptance_delta,
        "max_auto_prefill_regression_pct": args.max_auto_prefill_regression,
        "max_auto_startup_regression_pct": args.max_auto_startup_regression,
        "max_auto_memory_growth_mib": args.max_auto_memory_growth_mib,
        "max_auto_projection_fallbacks": args.max_auto_projection_fallbacks,
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
