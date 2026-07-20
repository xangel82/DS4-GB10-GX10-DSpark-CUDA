#!/usr/bin/env python3
"""Compare deterministic responses and DSpark acceptance across two binaries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_PROMPTS = [
    "Explain Redis in one sentence.",
    "What is 17 times 23? Answer with only the number.",
    "Write a Python function that reverses a string.",
    "Complete this C function: int add(int a, int b) {",
    "In one sentence, explain why a binary search is logarithmic.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Greedy DSpark baseline-versus-candidate acceptance fixture"
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dspark", required=True, type=Path)
    parser.add_argument("--binary", type=Path, default=Path("./ds4-server"))
    parser.add_argument(
        "--baseline-binary",
        required=True,
        type=Path,
        help="Stable ds4-server used as the ASIS quality/acceptance baseline",
    )
    parser.add_argument("--port", type=int, default=30009)
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--startup-timeout", type=float, default=240.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/tmp/ds4-dspark-acceptance"))
    parser.add_argument("--prompt", action="append", dest="prompts")
    return parser.parse_args()


def fixture_environment(dspark: bool) -> dict[str, str]:
    env = os.environ.copy()
    defaults = {
        "DS4_CUDA_COPY_MODEL": "1",
        "DS4_CUDA_DROP_COPIED_MODEL_PAGES": "1",
        "DS4_CUDA_WEIGHT_CACHE_LIMIT_GB": "112",
        "DS4_CUDA_Q8_F16_CACHE_MB": "12288",
        "DS4_CUDA_DEFER_END_SYNC": "1",
        "DS4_CUDA_TOKEN_GRAPH": "1",
        "DS4_CUDA_COALESCED_F16_MATMUL": "1",
        "DS4_CUDA_Q8_U16_LOADS": "1",
    }
    if dspark:
        defaults.update({
            "DS4_CUDA_COPY_SECONDARY_MODEL": "1",
            "DS4_CUDA_DSPARK_CACHE_PRIORITY": "1",
            "DS4_CUDA_DSPARK_GRAPH": "1",
            "DS4_CUDA_DSPARK_TENSOR_CORES": "1",
            "DS4_CUDA_DSPARK_TENSOR_CORES_Q8": "1",
            "DS4_CUDA_DSPARK_TC_PAD_N": "8",
            "DS4_CUDA_Q8_BATCH_REUSE": "1",
            "DS4_CUDA_MOE_TINY_DIRECT": "1",
            "DS4_CUDA_MOE_TINY_DIRECT_Q4_ONLY": "1",
            "DS4_DSPARK_ALWAYS_DRAFT": "1",
            "DS4_DSPARK_NO_CIRCUIT_BREAKER": "1",
            "DS4_DSPARK_TIMING": "1",
        })
    for key, value in defaults.items():
        env.setdefault(key, value)
    return env


def server_command(args: argparse.Namespace, binary: Path) -> list[str]:
    command = [
        str(binary.resolve()),
        "--cuda",
        "-m", str(args.model.resolve()),
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--ctx", str(args.ctx),
        "--tokens", str(args.tokens),
        "--threads", str(args.threads),
        "--prefill-chunk", "1024",
        "--dspark", str(args.dspark.resolve()),
        "--dspark-draft", "5",
    ]
    return command


def wait_ready(process: subprocess.Popen[bytes], port: int,
               timeout: float, log_path: Path) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"server exited with {process.returncode}; inspect {log_path}"
            )
        try:
            with urlopen(url, timeout=2.0) as response:
                if response.status == 200:
                    return
        except (HTTPError, URLError, TimeoutError):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"server did not become ready; inspect {log_path}")


def request_completion(port: int, prompt: str, tokens: int,
                       timeout: float) -> dict[str, object]:
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": tokens,
        "stream": False,
        "thinking": {"type": "disabled"},
    }
    request = Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError(f"completion has no choices: {body}")
    choice = choices[0]
    message = choice.get("message") or {}
    return {
        "content": message.get("content") or "",
        "reasoning_content": message.get("reasoning_content") or "",
        "finish_reason": choice.get("finish_reason"),
        "usage": body.get("usage") or {},
    }


def stop_server(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10.0)


def run_mode(args: argparse.Namespace, prompts: list[str], binary: Path,
             label: str) -> tuple[list[dict[str, object]], Path]:
    log_path = args.output_dir / f"{label}.log"
    results: list[dict[str, object]] = []
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            server_command(args, binary),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=fixture_environment(True),
            start_new_session=True,
        )
        try:
            wait_ready(process, args.port, args.startup_timeout, log_path)
            for index, prompt in enumerate(prompts):
                started = time.monotonic()
                result = request_completion(
                    args.port, prompt, args.tokens, args.request_timeout
                )
                result["id"] = index
                result["prompt"] = prompt
                result["wall_sec"] = time.monotonic() - started
                results.append(result)
        finally:
            stop_server(process)
    return results, log_path


def dspark_stats(log_path: Path) -> dict[str, int]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    drafted = sum(int(value) for value in
                  re.findall(r"dspark timing drafted=(\d+)", text))
    committed = sum(int(value) for value in
                    re.findall(r"dspark timing drafted=\d+[^\n]*committed=(\d+)",
                               text))
    return {
        "cycles": len(re.findall(r"dspark timing drafted=", text)),
        "drafted": drafted,
        "committed": committed,
    }


def main() -> int:
    args = parse_args()
    prompts = args.prompts or DEFAULT_PROMPTS
    for label, binary in (("baseline", args.baseline_binary),
                          ("candidate", args.binary)):
        if not binary.is_file() or not os.access(binary, os.X_OK):
            print(f"fixture: {label} executable not found: {binary}",
                  file=sys.stderr)
            return 2
    if not args.model.is_file() or not args.dspark.is_file():
        print("fixture: model or DSpark sidecar not found", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline, baseline_log = run_mode(
        args, prompts, args.baseline_binary, "baseline"
    )
    candidate, candidate_log = run_mode(
        args, prompts, args.binary, "candidate"
    )
    mismatches = 0
    rows = []
    for reference, result in zip(baseline, candidate):
        match = (
            reference["content"] == result["content"] and
            reference["reasoning_content"] == result["reasoning_content"] and
            reference["finish_reason"] == result["finish_reason"]
        )
        if not match:
            mismatches += 1
        rows.append({
            "id": reference["id"],
            "match": match,
            "baseline_sec": reference["wall_sec"],
            "candidate_sec": result["wall_sec"],
            "prompt": reference["prompt"],
            "baseline": reference,
            "candidate": result,
        })
        print(
            f"case={reference['id']} match={int(match)} "
            f"baseline={reference['wall_sec']:.3f}s "
            f"candidate={result['wall_sec']:.3f}s "
            f"prompt={reference['prompt']!r}"
        )

    baseline_stats = dspark_stats(baseline_log)
    candidate_stats = dspark_stats(candidate_log)
    summary = {
        "cases": len(rows),
        "mismatches": mismatches,
        "baseline_stats": baseline_stats,
        "candidate_stats": candidate_stats,
        "baseline_log": str(baseline_log),
        "candidate_log": str(candidate_log),
        "results": rows,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"fixture: cases={len(rows)} mismatches={mismatches} "
        f"baseline_cycles={baseline_stats['cycles']} "
        f"baseline_drafted={baseline_stats['drafted']} "
        f"baseline_committed={baseline_stats['committed']} "
        f"candidate_cycles={candidate_stats['cycles']} "
        f"candidate_drafted={candidate_stats['drafted']} "
        f"candidate_committed={candidate_stats['committed']} "
        f"summary={summary_path}"
    )
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
