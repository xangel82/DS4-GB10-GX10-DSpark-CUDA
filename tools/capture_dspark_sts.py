#!/usr/bin/env python3
"""Collect a diverse, lossless DSpark STS calibration capture."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROMPTS = (
    "Explain how a B-tree index handles inserts and page splits. Include a "
    "small worked example and the trade-offs for database workloads.",
    "Review this hypothetical bug: a lock-free queue occasionally loses a "
    "node after wraparound. Develop three plausible root causes and a careful "
    "debugging plan without assuming which cause is correct.",
    "Write a compact Python implementation of Dijkstra's algorithm, then "
    "explain its invariants, complexity, and behavior on disconnected graphs.",
    "Compare optimistic and pessimistic concurrency control for a reservation "
    "system. Give concrete failure scenarios and a recommendation.",
    "Derive the closed form of the sum of the first n squares and verify it "
    "by induction. Keep every algebraic step explicit.",
    "Translate the following idea into clear technical Italian and then "
    "summarize it in English: asynchronous scheduling can hide host latency "
    "while preserving a causal decision boundary.",
    "Design a JSON API for resumable multipart uploads. Specify endpoints, "
    "idempotency behavior, validation errors, and cleanup semantics.",
    "Explain why floating-point reduction order can change an argmax even "
    "when relative error is small. Give a numerical example and mitigations.",
    "Create a migration plan from a monolith to three services while keeping "
    "the public API stable. Include observability and rollback checkpoints.",
    "Analyze the security of a local AI gateway that should contact only one "
    "LAN host. Separate application controls from operating-system controls.",
    "Write a C function that parses an unsigned decimal integer with overflow "
    "detection. Explain each edge case and provide a focused test table.",
    "Compare radix top-k, heap top-k, and full sort for selecting 512 elements "
    "from 32768 scores on a GPU. Discuss exact tie handling.",
    "Explain compressed sparse attention to a senior software engineer. Trace "
    "one token through indexing, exact top-k, softmax, and value aggregation.",
    "Propose a benchmark for speculative decoding that separates drafter "
    "quality, verifier cost, end-to-end throughput, and output correctness.",
    "Diagnose a Linux service that is fast after startup but slow after hours "
    "of tool calls. Build a hypothesis tree using memory, I/O, and cache data.",
    "Describe how to make a deterministic scheduler from noisy throughput "
    "measurements. Include held-out validation and a safe production canary.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:30007/v1"
    )
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument(
        "--grouped-capture",
        type=Path,
        help=(
            "optional copy with a prompt group column for leakage-free "
            "calibration/validation splitting"
        ),
    )
    parser.add_argument("--target-samples", type=int, default=800)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


def capture_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as source:
        return max(sum(1 for line in source if line.strip()) - 1, 0)


def request_completion(args: argparse.Namespace, prompt: str, seed: int) -> int:
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Answer thoroughly and directly. Use enough detail to "
                    "make the reasoning and result independently useful."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": args.max_tokens,
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "min_p": 0.05,
        "seed": seed,
        "stream": False,
    }
    request = Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=args.timeout) as response:
            result = json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"request failed: {exc}") from exc
    usage = result.get("usage") or {}
    return int(usage.get("completion_tokens") or 0)


def append_grouped_rows(
    source: Path,
    destination: Path,
    first_row: int,
    last_row: int,
    group: str,
) -> None:
    with source.open(newline="", encoding="utf-8") as capture:
        rows = list(csv.DictReader(capture))
    if first_row < 0 or last_row < first_row or last_row > len(rows):
        raise RuntimeError("capture changed unexpectedly while grouping rows")
    fieldnames = [
        *(f"logit_{position}" for position in range(1, 6)),
        "accepted_prefix",
        "group",
    ]
    write_header = not destination.exists() or destination.stat().st_size == 0
    with destination.open("a", newline="", encoding="utf-8") as grouped:
        writer = csv.DictWriter(grouped, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows[first_row:last_row]:
            row["group"] = group
            writer.writerow({name: row[name] for name in fieldnames})


def main() -> int:
    args = parse_args()
    if args.target_samples <= 0 or args.max_tokens <= 0:
        raise ValueError("sample and token targets must be positive")
    initial = capture_rows(args.capture)
    if args.grouped_capture:
        args.grouped_capture.write_text("", encoding="ascii")
    prompt_index = 0
    generated = 0
    started = time.monotonic()
    while capture_rows(args.capture) - initial < args.target_samples:
        prompt = PROMPTS[prompt_index % len(PROMPTS)]
        round_index = prompt_index // len(PROMPTS)
        if round_index:
            prompt += (
                f"\nThis is validation variant {round_index}; use a different "
                "example and organization from earlier variants."
            )
        before = capture_rows(args.capture)
        generated += request_completion(
            args, prompt, 0xD5400000 + prompt_index
        )
        after = capture_rows(args.capture)
        if args.grouped_capture:
            append_grouped_rows(
                args.capture,
                args.grouped_capture,
                before,
                after,
                f"prompt-family-{prompt_index % len(PROMPTS) + 1}",
            )
        prompt_index += 1
        rows = capture_rows(args.capture) - initial
        print(
            f"capture prompts={prompt_index} rows={rows}/"
            f"{args.target_samples} completion_tokens={generated}",
            flush=True,
        )
    print(
        f"capture complete rows={capture_rows(args.capture) - initial} "
        f"prompts={prompt_index} elapsed={time.monotonic() - started:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
