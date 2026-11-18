# Agentic CV Platform: Orchestrator and Workflow API (Roadmap Item 6b)

Wires the chat→TaskSpec orchestrator and the existing calibration and batch label workflows into the platform HTTP API so they are reachable through the composed application.

## What this branch provides

- **Chat → TaskSpec orchestration:** `src/agentic_cv/orchestrator.py` (`TaskOrchestrator`, deterministic extraction of `task_type`/targets/attributes/defaults) and `POST /v1/orchestrate` returning a `TaskSpec` payload (authz-gated when an authorizer is configured; `labels:write`).
- **Batch label workflow API:** `LabelApiDependencies` + endpoints over `DurableLabelWorkflow` / job store — `POST /v1/label-jobs` (submit), `GET /v1/label-jobs/{job_id}` (status), `POST /v1/label-jobs/{job_id}/cancel` — tenant-scoped, kind `batch-label` (`labels:write`/`labels:read`).
- **Calibration workflow API:** `CalibrationApiDependencies` + `POST /v1/calibration/run` (multipart images + prompt/thresholds/agreement/iterations) running `Sam3Calibrator` against the configured SAM3 provider with built-in validators (`has_prediction`, `score_at_least:<value>`), returning attempts/status/selected threshold.
- **Composition wiring:** `application.py` now composes the annotation store (SQLite default, Postgres opt-in), `DurableLabelWorkflow` (`Sam3BatchLabeler` + preview/outbox/jobs/annotations), and exposes orchestrator/label/calibration dependencies; `/api-info` advertises `orchestrate`, `labels`, and `calibration` routes. `LocalAppResources` owns/closes annotations.

## Validation

- Focused: `tests/test_orchestrator_and_workflow_api.py tests/test_api.py tests/test_dataset_export_api.py tests/test_application.py tests/test_label_workflow.py tests/test_calibration.py tests/test_vlm_calibration.py tests/test_contracts.py` → **32 passed, 1 warning**.
- `python -m compileall -q src/agentic_cv tests` → OK.
- Full default suite (no services): **160 passed, 10 skipped, 1 warning**.
- `git diff --check` → clean.

## Limitations

- Orchestrator is deterministic/heuristic (no live LLM dependency); richer chat extraction can be layered on later without changing the endpoint contract.
- Batch label execution remains the responsibility of a label executor/worker path; this slice exposes submit/status/cancel (job lifecycle) via API, mirroring export workflow exposure.
- Calibration API runs synchronously with server-configured SAM3 provider and simple validators; VLM-judged calibration (overlay builder + Ollama/client budget) is not exposed as an HTTP mode in this slice.
- Default `pytest` requires no live services; Postgres/live-provider checks remain opt-in.

## Roadmap status

- **Item 6 (PostgreSQL/pgvector): complete** (prior slices).
- **Item 6b (orchestrator + calibration/label workflow API wiring): delivered in this slice.**
- **Next: Item 6c** — end-to-end smoke test: upload → search/label → VLM calibration → export, kept as a permanent regression test.
- **Then: Frontend/gallery integration** (after 6c), per roadmap order.

---
