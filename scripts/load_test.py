#!/usr/bin/env python3
"""Lightweight local load test for the Agentic CV platform API.

Targets a running server (default: http://127.0.0.1:8000) and reports basic
latency/error statistics for a small mixed request workload. Intended for local
MVP load smoke/perf sanity checks, not for production-grade benchmarking.

Examples:
  python scripts/load_test.py
  python scripts/load_test.py --base-url http://127.0.0.1:8000 --requests 200 --concurrency 20
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from collections import Counter

import httpx


DEFAULT_TENANT = "00000000-0000-0000-0000-000000000001"


async def _hit(client: httpx.AsyncClient, i: int) -> tuple[bool, float, int | None, str]:
    # Mixed read-heavy workload against default/local-safe endpoints.
    if i % 10 == 0:
        method, path = "GET", "/v1/metrics"
    elif i % 5 == 0:
        method, path = "GET", "/api-info"
    else:
        method, path = "GET", "/health/ready"
    start = time.perf_counter()
    try:
        response = await client.request(method, path)
        elapsed_ms = (time.perf_counter() - start) * 1000
        ok = 200 <= response.status_code < 300
        return ok, elapsed_ms, response.status_code, f"{method} {path}"
    except Exception as exc:  # pragma: no cover - exercised in manual runs
        elapsed_ms = (time.perf_counter() - start) * 1000
        return False, elapsed_ms, None, f"{method} {path} ({type(exc).__name__})"


async def run(base_url: str, total_requests: int, concurrency: int, timeout: float) -> int:
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    headers = {"X-Tenant-ID": DEFAULT_TENANT}
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    statuses: Counter[str] = Counter()
    errors = 0

    async with httpx.AsyncClient(base_url=base_url, timeout=timeout, limits=limits, headers=headers) as client:
        async def worker(i: int) -> None:
            nonlocal errors
            async with semaphore:
                ok, elapsed_ms, status_code, label = await _hit(client, i)
            latencies.append(elapsed_ms)
            statuses[str(status_code) if status_code is not None else label] += 1
            if not ok:
                errors += 1

        started = time.perf_counter()
        await asyncio.gather(*(worker(i) for i in range(total_requests)))
        duration_s = time.perf_counter() - started

    latencies_sorted = sorted(latencies)
    def pct(p: float) -> float:
        if not latencies_sorted:
            return 0.0
        idx = min(len(latencies_sorted) - 1, int(round((p / 100.0) * (len(latencies_sorted) - 1))))
        return latencies_sorted[idx]

    rps = (total_requests / duration_s) if duration_s > 0 else 0.0
    print(f"base_url={base_url}")
    print(f"requests={total_requests} concurrency={concurrency} duration_s={duration_s:.3f} rps={rps:.2f}")
    print(f"errors={errors} error_rate={(errors / total_requests * 100) if total_requests else 0:.2f}%")
    if latencies:
        print(
            "latency_ms "
            f"min={min(latencies):.2f} "
            f"p50={statistics.median(latencies):.2f} "
            f"p95={pct(95):.2f} "
            f"p99={pct(99):.2f} "
            f"max={max(latencies):.2f} "
            f"avg={statistics.mean(latencies):.2f}"
        )
    print("status_counts=" + ", ".join(f"{k}:{v}" for k, v in sorted(statuses.items())))
    return 1 if errors else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight local API load test.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1:
        raise SystemExit("--requests and --concurrency must be >= 1")
    raise SystemExit(asyncio.run(run(args.base_url, args.requests, args.concurrency, args.timeout)))


if __name__ == "__main__":
    main()
