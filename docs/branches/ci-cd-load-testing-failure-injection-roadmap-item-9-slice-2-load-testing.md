# Agentic CV Platform: CI/CD, Load Testing, Failure-Injection (Roadmap Item 9, Slice 2 — Load Testing)

Adds a lightweight local load-test script for basic API throughput/latency/error smoke checks. Builds on slice 1 (CI); defaults and runtime behavior unchanged.

## What this branch provides

- **`scripts/load_test.py`**
  - Async `httpx` load generator targeting a running server (`--base-url`, default `http://127.0.0.1:8000`).
  - Configurable `--requests`, `--concurrency`, `--timeout`.
  - Mixed read-heavy workload over local-safe endpoints (`/health/ready`, `/api-info`, `/v1/metrics`) and reports duration, RPS, error rate, latency stats (min/p50/p95/p99/max/avg), and status counts. Exits non-zero on request errors.
- Intended for local MVP load smoke/perf sanity checks (not production-grade benchmarking).

## Validation

- Workload sanity-verified in-process via `httpx.ASGITransport` against `create_local_app(...)` (40 concurrent mixed requests, all 200) with proper lifespan/resource cleanup.
- `python -m compileall -q src/agentic_cv tests` → OK.
- Full default suite (no services): **168 passed, 10 skipped, 1 warning**.
- `git diff --check` → clean.

## Usage

```bash
# with the app running locally (agentic-cv serve --data-dir .agentic-cv)
.\.venv\Scripts\python scripts/load_test.py --base-url http://127.0.0.1:8000 --requests 200 --concurrency 20
```

## Limitations / next boundary

- Manual/scripted smoke tool (not integrated into default CI, no long-soak/distributed/load-shape scenarios).
- Further item 9 slice: failure-injection testing.

## Roadmap status

- **Items 6, 6b, 6c, 7, 8: complete.**
- **Item 9:** slice 1 (CI) and slice 2 (load testing) delivered.
- **Next:** Item 9 slice 3 (failure-injection testing), then Item 10.

---
