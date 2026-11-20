# Phase 3 — faster labeling and the monitoring agent

Branch: `feature/label-monitor`
Base: `feature/review-approval` (Phase 4)
Plan: [`docs/OPTIMIZATION_PLAN.md`](../OPTIMIZATION_PLAN.md), Phase 3.

## Why

The brief asks for an agent that monitors the auto-labeling process and makes sure it runs well. Calibration sees five samples per class, but the whole asset can behave differently, for example a prompt that does not fit other scenes or a threshold that floods crowded frames. Before this phase nothing watched the run, and multi-class labeling ran SAM 3's full model once per class per image.

## Changes

**Throughput**
- `LocalSam3Provider.predict_many(image, [(prompt, threshold), ...])` computes the vision features once and runs each class prompt on them. Measured on an RTX 4050 Laptop, two classes, warm model, 720×1280: **1.11 → 0.63 s per image** with identical detections on all 9 test frames.
- `Sam3BatchLabeler` uses it for multi-class runs, falling back to per-class `predict` for providers without it. It decodes the next image on a CPU thread while the GPU works, and takes `skip_image_ids` for resumed runs.

**Monitoring agent** (`label_monitor.py`)
- Rolling metrics per job: processed/total, images/s, ETA, images with no detections, and per class the detections, images with detections, and mean score.
- Deterministic rules, checked after `max(8, 15%)` images on assets of 12 or more:
  - `mostly_empty`: more than 70% of images have no detections at all.
  - `class_missing:<name>`: one class never found while others are.
  - `crowded:<name>`: more than 25 detections per image, which usually means the threshold is labeling background.
- When a rule fires, the job is **paused**: a new job status `paused` (running → paused → queued | cancelled), an outbox event `LabelRunPaused`, and the alert saved with the health snapshot. Health is a small JSON file per job (`<data>/label-health/<job>.json`). Writes are best effort and retried, because Windows refuses to replace a file another thread is reading; a health write can never fail labeling.
- **Resume** skips the images already done (from the job's preview events), restores the counters, and does not fire an acknowledged alert again.

**Agent**
- New session status `paused`. The agent says what it saw and asks: *"I paused labeling to check with you. … Should I continue with the same settings, or stop so you can adjust the task?"*, with **Continue labeling / Stop** quick replies. Typed answers work too ("keep going", "stop"); anything else gets a short reminder of the two options.
- The session response includes `health`.

**UI**
- A **Labeling health** panel in the run panel: progress bar, images, speed, time left, no-detection count, per-class detections and mean score, and the alert banner. Its state reads Watching / Paused / Resuming / Finished.
- The paused question uses the chat composer and chips, and polling restarts after you answer.

## Bugs found and fixed

- **Windows file replacement:** publishing health while the API read it raised `WinError 5` and failed the label job. Writes now retry and are non-fatal; reads retry around a replace in progress.
- **Duplicate alert:** after "continue" on an all-empty single-class run, `class_missing` fired for the same cause. It now fires only when other classes are being found.
- **Shutdown during background calibration** (from Phase 2b): the calibration thread kept writing to closed stores, and a cancelled run was recorded as `failed`. Shutdown now cancels, waits up to 15 s, and leaves the session `calibrating`, so the next start resumes it.
- `scripts/real_agent_journey.py` confirms with `wait: true` (it predated background calibration).

## Real-model validation

Same 9 wildlife frames, elephants and tusks, production wiring (see [`docs/PERFORMANCE.md`](../PERFORMANCE.md)):
- Calibration unchanged: 142 s, 5/5 and 5/5 at 0.45.
- Labeling, build and export: **24.2 s** (27.2 s before). The gain is small here because reloading SAM 3 after calibration (about 14 s) dominates 9 images.
- Identical output: 26 instances (8 elephants, 18 tusks).
- Final health: 9/9, 2 images with no detections, elephants mean score 0.88, tusks 0.68, no alert (9 < 12 images).
- No errors and a clean shutdown.

## Validation

```text
$ .venv/Scripts/python -m pytest -q
362 passed, 10 skipped, 1 warning
$ compileall / node --check / git diff --check   # ok
```

New:
- `tests/test_label_monitor.py`:
  - Rules, thresholds and acknowledgement.
  - One shared vision call per image, and skip on resume.
  - HTTP: pause → unclear answer → "Continue labeling" → complete, with SAM 3 called once per image in total (no relabeling).
  - "Stop" cancels a paused run.
  - Playwright: health panel, alert, chip, finish.
- `tests/test_calibration_live.py`: shutdown mid-calibration, then resume to completion.

## Limitations

- The speed (and therefore the ETA) includes the first model load, so early ETAs are pessimistic.
- Rules are fixed thresholds. They catch gross problems, not subtle label quality. That is Phase 7 (evaluation harness).
- Health files assume one host (local-first). A multi-host deployment would move them into the job store.
- The live per-image grid still loads server-rendered PNGs over the replayable SSE endpoint. Client-side drawing was not needed for the monitor and is left as polish.
