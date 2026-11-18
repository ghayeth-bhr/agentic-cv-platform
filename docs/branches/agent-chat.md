# feature/agent-chat (P3)

## Scope

Make the Agent section a real product path instead of a placeholder:

1. Choose an asset in the browser, describe a dataset in natural language, and review the extracted task specification before anything runs.
2. Ask the local Ollama text model for strict `TaskSpec` JSON; fall back to the deterministic rules planner with a visible note when the model is unavailable or returns invalid JSON.
3. Calibrate a threshold on a representative sample with a configurable timeout, then label the whole asset through the existing durable label workflow.
4. Build an immutable dataset version from the resulting annotation revisions, package a COCO archive, and download it from the chat.
5. Persist sessions and messages in SQLite so a page reload restores the active or finished run.
6. Allow cancellation during labeling/export and fixed-threshold confirmation when calibration does not reach agreement.

## What changed

- `agent.py`: new planner/service layer. It creates a durable session, records chat messages, extracts a prompt, calibrates through `Sam3Calibrator` on asset-derived samples, submits an idempotent whole-asset label job, advances a succeeded job into `DatasetBuilder` + `DatasetExportWorkflow`, and maps job states back to the session.
- `agent_api.py`: `POST/GET /v1/agent/sessions`, `GET /v1/agent/sessions/{id}`, `POST .../confirm`, and `POST .../cancel`. Conflicts are 409, missing sessions/assets are 404, calibration timeout is 504, model unavailability preserves the P0 503 path, and validation errors are 422.
- `agent_store.py`: SQLite `agent_sessions` table with messages, status, spec, planner mode/note, threshold, calibration result, label/export job IDs, dataset version ID, and errors. `list_open()` is what the worker advances.
- `agent_worker.py`: polling worker that advances `labeling → dataset/export` and `exporting → complete`, with expired-lease recovery inherited from the label/export job stores.
- `ollama_text.py`: strict JSON task-spec client for Ollama `/api/chat`, with schema validation before a spec is accepted.
- `application.py`: wires the session store, planner, timeout, service, API router, and agent worker into the local composition root. `agentic-cv serve` starts the agent worker. New config: `agent_planner`, `calibration_timeout_sec`, plus environment variables documented in the README.
- `web/index.html`, `web/app.js`, `web/app.css`: the Agent placeholder is replaced by a chat/compose panel and a live run panel with specification, calibration, labeling, dataset, export progress, clear errors, retry at a fixed threshold, cancel, and download. The active session is stored in `localStorage`; UUIDs never appear in rendered text.
- `web/app.js`: dataset and Agent downloads now use an authenticated `fetch` + blob download instead of navigating to a header-protected URL. This fixes the P2 dataset export link as well.
- Tests: `tests/test_agent_planner.py`, `tests/test_agent_chat_e2e.py`, and `tests/test_agent_ui_playwright.py`.

## Validation (real output)

```text
node --check src/agentic_cv/web/app.js
.venv/Scripts/python -m compileall -q src tests
git diff --check
uv lock --check

.venv/Scripts/python -m pytest -q
235 passed, 10 skipped, 1 warning in 26.56s
```

The 10 skips are the existing opt-in PostgreSQL tests. The warning is the existing Starlette/httpx deprecation.

Focused P3 gates:

- `tests/test_agent_chat_e2e.py` uses a fake SAM 3 provider and real HTTP routes to run plan → confirm → calibrate → label → dataset → COCO zip download, then validates the archive with `pycocotools`. It also covers calibration timeout (504 + failed session message), cancellation of a queued label job, fixed-threshold retry after `needs_attention`, and Ollama-down planner fallback.
- `tests/test_agent_ui_playwright.py` starts a real uvicorn server, uploads through the shipped UI, opens Agent, plans, confirms, waits for `Dataset ready`, downloads the zip through the browser download event, validates COCO, reloads the page, and confirms session restore. It also asserts no UUID appears in rendered text and no horizontal overflow.

### Real-model manual run

Executed on the owner's machine: RTX 4050 Laptop, 6 GB VRAM, `torch 2.6.0+cu124`, cached `facebook/sam3`, Ollama `0.31.2` running at `127.0.0.1:11434`.

- Real Ollama planner (`qwen3-vl:4b`): returned a valid spec in **9.37 s**. Ollama placed the structured answer in `message.thinking` with an empty `message.content`; `ollama_text.py` now accepts that narrow fallback (locked by a unit test).
- Real 20-image run: 20 generated 512×384 images uploaded through the real HTTP API, agent session created with the rules planner (to avoid loading the VLM alongside SAM 3), calibration at **0.30** with **5/5** sample agreement, whole-asset label job `succeeded`.
- Timings: upload **1.15 s**, calibration including initial SAM 3 load **38.49 s**, label job **13.75 s**, end-to-end **53.97 s**.
- Result: **20/20** COCO images and annotations, **10,975-byte** zip, peak allocated VRAM **1.96 GB**.
- This validates real-model mechanics and memory headroom on generated images. It does not validate segmentation quality on the owner's real photo set.

## Limitations

- Calibration's default acceptance test is “SAM 3 returned at least one prediction.” The VLM validator adapter exists and is injectable, but the default Agent path does not call the VLM yet.
- The chat exports COCO only. YOLO remains available through the dataset-export API; a format switch is not in this phase.
- The planner is a local-model call with a deterministic fallback, not a multi-turn conversational agent. Each session is one task.
- Cancellation is cooperative. A label job cancelled mid-run can leave partial revisions, but only a succeeded job is packaged into a dataset.
- The browser tests use fake SAM 3/embedding providers and `RulesTaskPlanner`, so they validate wiring, not real model quality or latency. The separate real-model run measured mechanics and VRAM on generated images, not quality on real photos.
- `agentic-cv worker` still runs only the export worker; label, index, and agent-advance workers run inside `agentic-cv serve`.

## Next boundary

P4 `feature/manual-refinement-ui`: open a bad image from a completed dataset, click positive/negative points, preview and accept a mask revision, and make the next dataset build include that revision. The P3 session/dataset artifacts and the existing click-refinement WebSocket are the inputs.
