# Agentic CV Platform: Observability and GPU Metrics (Roadmap Item 8, Slice 2)

Adds structured logging utilities and HTTP request observability on top of slice 1. SQLite/local defaults and existing APIs remain unchanged.

## What this branch provides

- **Structured logging** (`src/agentic_cv/observability.py`)
  - `JsonLogFormatter`: single-line JSON log records (`timestamp`, `level`, `logger`, `message`, optional `exception`, and structured extras such as `method`/`path`/`status_code`/`duration_ms`).
  - `configure_logging(level=None, fmt=None)`: configures the `agentic_cv` logger namespace (scoped, no uvicorn/root override) with `text` or `json` formatting. Reads `AGENTIC_CV_LOG_LEVEL` and `AGENTIC_CV_LOG_FORMAT` when explicit args are not provided; avoids duplicate handlers on repeated calls.
- **Request observability middleware** (`src/agentic_cv/application.py`)
  - Logs request completion/failure via `agentic_cv.http` with `method`, `path`, `status_code`, and `duration_ms`.
- **CLI integration** (`src/agentic_cv/cli.py`)
  - `agentic-cv serve` and `agentic-cv worker` call `configure_logging()` at startup so env-based log config applies.
- **Tests:** extended `tests/test_observability.py` for JSON formatter fields, scoped/idempotent logger configuration, and middleware request-completion logging.

## Validation

- Focused: `tests/test_observability.py tests/test_application.py` → **11 passed, 1 warning**.
- `python -m compileall -q src/agentic_cv tests` → OK.
- Full default suite (no services): **168 passed, 10 skipped, 1 warning**.
- `git diff --check` → clean.

## Limitations / next boundary

- Structured logs and request middleware are in-process/stdlib-based (no OpenTelemetry tracing, no log shipping/aggregation config, no Prometheus exporter in this slice).
- Middleware records per-request completion/failure; deeper job/worker stage instrumentation and trace correlation can be added in later item 8 slices.

## Roadmap status

- **Items 6, 6b, 6c, and Frontend/gallery integration: complete** (prior slices).
- **Item 8 (Observability and GPU metrics): slice 2 delivered in this branch** (builds on slice 1 metrics/`/v1/metrics`).

---
