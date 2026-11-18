# Agentic CV Platform: Observability and GPU Metrics (Roadmap Item 8, Slice 3)

Adds worker instrumentation to the dataset export worker and makes structured JSON logs carry arbitrary record extras. SQLite/local defaults and behavior remain unchanged.

## What this branch provides

- **Export worker instrumentation** (`src/agentic_cv/dataset_export_worker.py`)
  - Logs on `agentic_cv.worker`: expired lease requeue count, per-job attempt start/finish/failure (worker/job/tenant/kind/status/duration_ms/error as applicable), and worker loop start/stop.
  - Adds timing around job attempts and reports post-attempt job status; exceptions are still caught so one bad export cannot starve the queue (behavior preserved).
- **Structured logging enhancement** (`src/agentic_cv/observability.py`)
  - `JsonLogFormatter` now serializes non-reserved extra `LogRecord` fields (with string fallback for non-JSON values), so worker/request/job context fields are preserved in JSON logs.
- **Tests:** `tests/test_dataset_export_worker.py` asserts worker log emission (start/finish and loop lifecycle) while preserving existing behavior checks.

## Validation

- Focused: `tests/test_observability.py tests/test_dataset_export_worker.py tests/test_application.py` → **14 passed, 1 warning**.
- `python -m compileall -q src/agentic_cv tests` → OK.
- Full default suite (no services): **168 passed, 10 skipped, 1 warning**.
- `git diff --check` → clean.

## Limitations / next boundary

- Instrumentation currently targets the dataset export worker loop/attempts; other workers/workflows (e.g. label/batch execution paths) and job-stage tracing/correlation can be instrumented in later item 8 slices.
- Still stdlib logging/JSON metrics (no OpenTelemetry tracing, log shipping/aggregation config, or Prometheus exporter).

## Roadmap status

- **Items 6, 6b, 6c, and Frontend/gallery integration: complete** (prior slices).
- **Item 8 (Observability and GPU metrics): slice 3 delivered in this branch** (builds on slices 1–2).

---
