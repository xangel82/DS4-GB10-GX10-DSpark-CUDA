#!/usr/bin/env python3
"""Run deterministic concurrent chat requests against a DS4 server."""

import argparse
import concurrent.futures
import hashlib
import json
import threading
import time
import urllib.error
import urllib.request


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:30007/v1")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument(
        "--prompt-mode",
        choices=("technical", "repetitive"),
        default="technical",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def run_request(args, lane, barrier):
    if args.prompt_mode == "repetitive":
        cycle = (
            "alpha beta gamma delta epsilon zeta eta theta "
            "iota kappa lambda mu "
        )
        prompt = (
            f"Lane {lane}: continue the exact cyclic sequence below. "
            "Output only sequence words separated by spaces and keep "
            f"repeating until {args.max_tokens} output tokens.\n"
            + cycle * 96
            + "alpha beta gamma delta"
        )
    else:
        prompt = (
            f"Lane {lane}: write a continuous technical analysis of CUDA "
            f"scheduling and speculative decoding. Produce exactly "
            f"{args.max_tokens} output tokens. Do not conclude early, do not "
            "summarize, and keep writing until the token limit."
        )
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_k": 0,
        "top_p": 1.0,
        "min_p": 0.0,
        "seed": args.seed + lane,
        "stream": False,
    }
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    barrier.wait()
    started = time.monotonic()
    try:
        with urllib.request.urlopen(
                request, timeout=args.timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read()
        raise RuntimeError(
            f"lane {lane} HTTP {exc.code}: "
            f"{body.decode('utf-8', 'replace')}") from exc
    elapsed = time.monotonic() - started
    decoded = json.loads(body)
    usage = decoded.get("usage") or {}
    completion_tokens = int(usage.get("completion_tokens") or 0)
    choices = decoded.get("choices") or []
    finish_reason = choices[0].get("finish_reason") if choices else None
    content = ""
    if choices:
        message = choices[0].get("message") or {}
        content = (
            message.get("reasoning_content") or "") + (
            message.get("content") or "")
    return {
        "lane": lane,
        "tokens": completion_tokens,
        "seconds": elapsed,
        "tps": completion_tokens / elapsed if elapsed > 0.0 else 0.0,
        "finish": finish_reason,
        "sha256": hashlib.sha256(
            content.encode("utf-8")).hexdigest()[:16],
    }


def main():
    args = parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")
    barrier = threading.Barrier(args.concurrency)
    wall_started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(run_request, args, lane, barrier)
            for lane in range(args.concurrency)
        ]
        results = [future.result() for future in futures]
    wall_seconds = time.monotonic() - wall_started
    total_tokens = sum(result["tokens"] for result in results)
    for result in sorted(results, key=lambda item: item["lane"]):
        print(
            f"lane={result['lane']} tokens={result['tokens']} "
            f"seconds={result['seconds']:.3f} "
            f"tps={result['tps']:.2f} finish={result['finish']} "
            f"sha256={result['sha256']}"
        )
    print(
        f"R={args.concurrency} total_tokens={total_tokens} "
        f"wall_seconds={wall_seconds:.3f} "
        f"aggregate_tps={total_tokens / wall_seconds:.2f}"
    )


if __name__ == "__main__":
    main()
