# Phase 2b — background calibration and the live calibration board

Branch: `feature/calibration-live`
Base: `feature/calibration-engine` (Phase 2a)
Plan: [`docs/OPTIMIZATION_PLAN.md`](../OPTIMIZATION_PLAN.md), Phase 2 (defect B9 and the "4-tile live calibration board").

## Why

After Phase 2a, calibration was correct but took about 142 s **inside `POST /confirm`**: the browser waited on one request, nothing showed progress, and a restart lost the run. The brief asks for the Roboflow-style step (see SAM 3 on a few samples, vary the threshold, pick one) done agentically, with a modern live view.

## Part A — background calibration

- `POST /v1/agent/sessions/{id}/confirm` validates, stores the request on the session (`spec.calibration_request`), and **returns at once with status `calibrating`**.
- Calibration runs on a single-worker executor, so runs queue on the shared GPU.
- Progress is saved after each threshold (`calibration.status = "running"`, `calibration.progress[]`) without overwriting a concurrent cancel.
- **Resume:** `calibrating` sessions are part of the agent worker's open set. One that no thread in this process owns (for example after a restart) is resubmitted with its stored request.
- `wait: true` runs inline and keeps the old outcomes and error codes (504 on timeout, 503 when a model is unavailable). Scripts and outcome-focused HTTP tests use it. The UI never does.
- **`class_thresholds: {name: t}`** skips calibration and labels each class at the user's threshold. It reuses the prompts calibration chose and keeps the previous calibration under `calibration.previous`.

## Part B — live calibration board (Agent run panel)

- **Data:**
  - After SAM 3's pass for a class (before any VLM call), its raw predictions are written to `calibration-overlays/<tenant>/<session>/preview.json`: per sample, the size and each prediction's score, box, and simplified polygons.
  - `GET /v1/agent/sessions/{id}/calibration-preview` serves that file.
  - `GET /v1/agent/sessions/{id}/calibration-samples/{image_id}/image` serves a sample image. It is restricted to images in the preview and to the session's tenant.
- **UI** (`#agent-board`), one panel per class:
  - Sample tiles drawn on canvases, colour-coded per class, with score labels.
  - **A threshold slider** that refilters and redraws the masks in the browser with no model call.
  - Threshold chips (`0.45 · 5/5`) appear as verdicts arrive; the chosen one is highlighted, and clicking a chip moves the slider.
  - Per-tile verdict badges at tried thresholds (✓ accepted, ✓ no target, ✗ missed, ✗ rejected), with the VLM's reasoning on hover.
  - A **Live** badge while calibrating.
  - When the session needs attention: **Label with these thresholds** (sends `class_thresholds`).
- The older verdict log stays hidden while calibration runs. During a run it misread the in-progress payload as "Fallback validator · no attempts".

## Real-model validation

Driven through the real browser UI with real SAM 3, `qwen3-vl:4b`, the Ollama planner and SigLIP, on the 9 local wildlife frames (`scripts/real_ui_calibration_board.py`):

- Confirm returned immediately.
- The board showed masks on every sample before the first VLM verdict, then filled in the chips.
- Calibrated `elephant` at 0.45 (5/5) and `tusk` at 0.60 (5/5).
- The YOLO-seg dataset (21 instances) was ready 211 s after the page opened, about 150 s from confirm.

See [`docs/PERFORMANCE.md`](../PERFORMANCE.md).

## Validation

```text
$ .venv/Scripts/python -m pytest -q
345 passed, 10 skipped, 1 warning
$ compileall / node --check / git diff --check   # ok
```

New `tests/test_calibration_live.py`:
- Confirm returns in under 2 s and `progress` appears mid-run.
- A simulated crash mid-calibration resumes after restart.
- `class_thresholds` validation and override.
- Preview and sample-image endpoints, including tenant isolation.
- A Playwright board journey: the slider changes 2 → 1 → 0 masks, a verdict badge appears at a tried threshold, and labeling runs at the user's threshold.

The UI contract test covers the new routes and DOM ids. Thirteen outcome-focused HTTP tests now send `wait: true`.

## Limitations

- The slider previews what SAM 3 returned at the lowest calibration threshold (0.30 by default). Below that, nothing extra exists to show.
- Verdict badges exist only at thresholds the VLM judged.
- The board covers calibration samples, not the whole asset. Phase 3 brings the live labeling view and health monitoring to the same standard.
- One calibration at a time per server process, by design (one GPU).
