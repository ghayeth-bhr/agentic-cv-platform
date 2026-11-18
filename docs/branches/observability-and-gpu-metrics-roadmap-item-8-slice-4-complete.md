# Agentic CV Platform: Observability and GPU Metrics (Roadmap Item 8, Slice 4 — Complete)

Completes the item 8 MVP observability/GPU-metrics baseline by instrumenting the durable label workflow in addition to slices 1–3. SQLite/local defaults and behavior remain unchanged.

## What this branch provides

- **Label workflow instrumentation** (`src/agentic_cv/label_workflow.py`)
  - Logs on `agentic_cv.workflow`: workflow start, progress checkpoints (debug), completion/cancellation/failure with `worker_id`, `job_id`, `tenant_id`/counts/status, and `duration_ms`.
  - Preserves existing behavior (claim/heartbeat/preview/checkpoint/cancel/outbox/annotation persistence semantics unchanged).
- **Tests:** `tests/test_label_workflow.py` asserts workflow observability events (start/completed/failed/cancelled) while preserving existing behavior checks.

## Validation

- Focused: `tests/test_label_workflow.py tests/test_observability.py tests/test_dataset_export_worker.py` → **15 passed, 1 warning**.
- `python -m compileall -q src/agentic_cv tests` → OK.
- Full default suite (no services): **168 passed, 10 skipped, 1 warning**.
- `git diff --check` → clean.

## Item 8 summary (baseline complete)

- **Slice 1:** runtime/GPU metrics collection + `GET /v1/metrics` (`e47f0c0`).
- **Slice 2:** structured logging (text/JSON, scoped `agentic_cv` config) + HTTP request observability + CLI integration (`4adc0e6`).
- **Slice 3:** export worker instrumentation + richer JSON log extras (`2ce9065`).
- **Slice 4:** label workflow instrumentation (**this slice**).

## Limitations (out of scope for this baseline)

- Stdlib/JSON logging and in-process metrics only (no OpenTelemetry tracing, log/metric shipping, or Prometheus exporter).
- Instrumentation covers HTTP requests, dataset export worker, and durable label workflow; additional job/workflow stages and trace correlation can be extended later if needed.

## Roadmap status

- **Items 6, 6b, 6c, 7 (frontend + gallery integration), and 8 (observability/GPU metrics baseline): complete.**
- **Next roadmap items:** 9 (CI/CD, load testing, failure-injection testing), 10 (SAM 3 licensing and hosted-product review) per agreed order.

---
