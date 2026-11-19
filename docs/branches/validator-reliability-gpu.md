# P8 — validator reliability and GPU release

Branch: `feature/validator-reliability-gpu`  
Base: `09cbeb8` (pushed P7 tip)  
Status: implementation and real runs complete; pending owner review.

## Scope

P8 proves the VLM validator with real models and known ground truth, releases
the Ollama VLM after calibration, and records per-phase timings. It does not
change the shipped whole-asset labeling, correction, or dataset version journey.

## Changes

- `OllamaVlmClient.unload()` sends `POST /api/generate` with `keep_alive: 0` and
  an empty prompt. `num_predict` is bounded so calibration answers stay small.
- `VlmValidator`, `VlmCalibrationValidator`, and `ValidatorSelection` expose an
  idempotent `release()` hook. Injected validators need no Ollama.
- `AgentService.confirm` releases the validator after calibration, before
  `labels.submit()`, and appends a chat message with the unload result.
- A deterministic `minimum_prediction_score` guard rejects implausibly
  low-score mask sets before a VLM call. Default `0.2`, configurable with
  `AGENTIC_CV_VLM_MIN_PREDICTION_SCORE`.
- An absent-target calibration failure now ends with
  `I can't segment reliably ...` in both the chat and the error field.
- Provider status exposes `load_seconds` for SigLIP2 and SAM 3; calibration
  payloads carry calibration/unload timings.
- `scripts/validate_calibration_reliability.py` is a manual real-model harness;
  it writes every case, including failures, to JSON.
- `scripts/measure_full_journey.py` reports model-load, calibration, unload,
  labeling, export, and confirm-to-complete phases, plus the unload signal.

## Real reliability run

```powershell
..\frames-extractor\.venv\Scripts\python.exe scripts/validate_calibration_reliability.py `
  --images 30 --json-out ..\p8-reliability.json
```

First run: 30/30 IoU agreement and absent prompt passed, but the too-low
threshold check failed. At threshold `0.02` SAM 3 produced 12 masks, best IoU
`0.2767`, 2 false-positive masks, and `qwen3-vl:4b` accepted with confidence
`0.55`. The VLM reasoned that the extra masks outlined parts of the same
object. This failure is recorded in `docs/PERFORMANCE.md` and led to the
low-score guard.

Re-run after the guard:

```json
{
  "case_count": 30,
  "vlm_calls": 51,
  "agreement_count": 30,
  "agreement_rate": 1.0,
  "false_accepts": 0,
  "false_rejects": 0,
  "wall_sec": 332.73,
  "low_threshold_rejected": true,
  "absent_prompt_message_ok": true
}
```

The low threshold produced 11 masks below `0.20` and was rejected before a VLM
call. The absent prompt ended in `needs_attention` with the "can't segment
reliably" message. VLM unload at harness end succeeded.

## Real full-journey run

```powershell
$env:AGENTIC_CV_VLM_TIMEOUT_SEC = "180"
..\frames-extractor\.venv\Scripts\python.exe scripts/measure_full_journey.py `
  --images 20 --source .agentic-cv/derived/semantic-frames `
  --message "Create a segmentation dataset of animals." --prompt "animals" `
  --require-vlm --timeout 900 --calibration-timeout 600 `
  --json-out ..\p8-full-journey.json
```

- Threshold `0.3`, `5/5` accepted VLM verdicts, `5` overlay ids.
- `vlm_unloaded_after_calibration: true`; unload took `2.06 s`.
- 20 images, 34 annotations, COCO `3,602,151` bytes, refined `3,602,149` bytes.
- Model load `48.38 s` (`siglip2 30.00`, `sam3 18.38`).
- Calibration `351.39 s`, labeling `13.91 s`, export `0.54 s`,
  confirm-to-complete `371.24 s`.
- PyTorch peak `4,202.8 MB` allocated / `4,390.0 MB` reserved.
- `nvidia-smi` whole-GPU peak `5,914 MB` of `6,141 MB`; calibration window
  `2,375 MB` to `5,914 MB` (`+3,539 MB`).

P7's `383.21 s` checkpoint was the same shape: VLM calibration dominated it.
P8's phase instrumentation attributes `351.39 s` of `369.05 s` to calibration,
not to labeling or export. The unload happens after the peak; it removes the
VLM from labeling but does not lower the recorded whole-GPU peak.

Full output and caveats are in [`docs/PERFORMANCE.md`](../PERFORMANCE.md).

## Deterministic gates

```text
$ .venv\Scripts\python -m pytest -q
279 passed, 10 skipped, 1 warning in 41.03s

$ .venv\Scripts\python -m pytest tests/test_agent_ui_playwright.py tests/test_manual_refinement_ui_playwright.py tests/test_assets_ui_playwright.py -q
3 passed in 21.86s

$ node --check src/agentic_cv/web/app.js
(no output; exit 0)

$ .venv\Scripts\python -m compileall -q src tests scripts
(no output; exit 0)

$ uv lock --check
Resolved 90 packages in 8ms

$ git diff --check
(no output; exit 0)

$ ..\frames-extractor\.venv\Scripts\agentic-cv.exe doctor --no-network --port 8125
Summary: 11 ok, 0 warn, 0 fail
```

The 10 skips are the opt-in PostgreSQL tests.

## Fresh-clone gate

To be recorded after the clean clone run.

## Limits

- The IoU set uses synthetic rectangles with known masks. Real production-photo
  quality remains a human section-7 check.
- The low-score guard rejects over-permissive thresholds deterministically;
  it does not prove the VLM's visual judgment on every object class.
- Ollama is external to the app's model manager. An external process holding
  several GB before calibration can still cause an OOM.
- No pushed history was rewritten and no default-branch change was made.
