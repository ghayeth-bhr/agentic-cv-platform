# Agentic CV Platform: Observability and GPU Metrics (Roadmap Item 8, Slice 1)

Adds a lightweight observability foundation focused on runtime and GPU metrics, exposed through the composed platform app. This is the first slice of roadmap item 8 (observability and GPU metrics); SQLite/local defaults and existing APIs remain unchanged.

## What this branch provides

- **`src/agentic_cv/observability.py`**
  - `collect_gpu_metrics(torch_module=None)`: reports CUDA/GPU availability and memory metrics (`total/allocated/reserved/free`) when `torch.cuda` is available; returns an explicit unavailable reason when torch or CUDA is not present. Accepts an injected torch-like module for deterministic testing.
  - `collect_runtime_metrics()`: basic process/runtime info (Python version, platform, PID, Unix timestamp, and `resource` usage where available).
  - `collect_metrics()`: combined `status`, `runtime`, and `gpu` payload.
- **Platform endpoint and discovery** (`application.py`)
  - `GET /v1/metrics` returns the collected observability payload as JSON.
  - `/api-info` now includes `observability.metrics` → `/v1/metrics`.
- **Tests:** `tests/test_observability.py` covers GPU-metrics branches (mocked torch/CUDA and unavailable cases), payload shape, and the `/v1/metrics` + `/api-info` integration.

## Validation

- Focused: `tests/test_observability.py tests/test_application.py` → **8 passed, 1 warning**.
- `python -m compileall -q src/agentic_cv tests` → OK.
- Full default suite (no services): **165 passed, 10 skipped, 1 warning**.
- `git diff --check` → clean.

## Limitations / next boundary

- JSON metrics endpoint only (no Prometheus/OpenTelemetry exporter, no request/job tracing middleware, no persistent metrics store in this slice).
- GPU metrics depend on `torch`/CUDA being available in the runtime; environments without them receive an explicit unavailable report (no hard dependency added).
- Further item 8 slices can add structured logging, request/worker instrumentation, and richer GPU/process telemetry as needed.

## Roadmap status

- **Items 6, 6b, 6c, and Frontend/gallery integration: complete** (prior slices).
- **Item 8 (Observability and GPU metrics): slice 1 delivered in this branch.**

---
