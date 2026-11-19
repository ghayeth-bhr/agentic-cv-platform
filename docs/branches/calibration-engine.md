# Phase 2a — calibration engine (correctness and speed)

Branch: `feature/calibration-engine`
Base: `feature/search-selection`
Plan: [`docs/OPTIMIZATION_PLAN.md`](../OPTIMIZATION_PLAN.md), Phase 2 (2b, background calibration plus the live calibration board, is next).

## Why

The first real end-to-end run (Phase 1) calibrated `elephants` and `tusks` on 9 wildlife frames and ended in `needs_attention`, although SAM 3 was right on every sample:

- An empty SAM 3 result was rejected automatically, so the frames without an elephant (dik-dik, buffalo, gerenuk) counted as failures.
- The diversity sampler picked exactly those outlier frames.
- The ladder chose the first passing threshold, which is the most permissive one.
- SAM 3 ran once per threshold.
- The whole calibration took 459 s.

## Changes

| Change | Where |
|---|---|
| **One SAM 3 forward pass per sample** at the lowest threshold. Higher thresholds filter by score, which is exact because SAM 3 applies the threshold in post-processing. 25 passes → 5 per class. | `calibration.py` (`Sam3Calibrator.infer`) |
| **Every threshold is scored; the best agreement wins**, and ties go to the middle of the best run. A pass also needs at least one sample where the target was found. New status `no_target_in_samples`. | `calibration.py` (`score`, `_decide`) |
| **Correct negatives.** With no masks, the VLM judges the plain image ("is a … visible?"). Absent means accepted (`target_present=false`); present means "Missed". The boolean decides: qwen3-vl:4b answers "absent" correctly but with low confidence. One call per sample (cached across thresholds). | `vlm_validator.py` |
| **Target-aware samples.** `ceil(0.6·n)` samples are the images most similar to the class text (SigLIP); the rest are chosen farthest-first to test false positives. Falls back to the old selector without vectors. | `calibration_samples.py` (`select_samples_for_target`) |
| **Prompt alternatives.** A class that fails with its first phrase is calibrated with the prompt agent's next phrase (up to 3), and labeling uses the phrase that passed. | `agent.py` |
| **Phased rounds: SAM 3 and the VLM are never resident together.** Each round runs SAM 3 for every pending class, unloads SAM 3, then validates. A retry round unloads the VLM first. SigLIP is unloaded after sampling. Measured: a VLM verdict takes about 4 s alone and about 45 s next to SAM 3 on a 6 GB GPU (Windows pages GPU memory). | `agent.py` (`calibrate_all`), `application.py` (`release_models`) |
| VLM images are downscaled to 768 px on the longest side (`AGENTIC_CV_VLM_MAX_SIDE`, 0 disables). | `ollama_vlm.py` |
| Calibration progress is saved after each threshold (`calibration.progress`) without overwriting a concurrent cancel. Phase 2b's live view will render it. | `agent.py` (`_save_progress`) |
| Payload: `forward_passes`, `prompts_tried`, and per verdict `prediction_count` and `target_present`. UI: "Correct · no target" / "Missed" labels and a "Prompts tried" line. | `agent.py`, `web/app.js` |

## Real-model result

Full table in [`docs/PERFORMANCE.md`](../PERFORMANCE.md#phase-2a-calibration-engine-real-models-2026-11-20). Same 9 frames, same chat, four runs:

| Run | Elephants / tusks agreement at the chosen threshold | Calibration | Dataset |
|---|---|---|---|
| Phase 1 | 2/5 / 2/5 (failed) | 459 s | none |
| Phase 2a final | **5/5 / 5/5**, both at 0.45 | **142 s** | 9 images, 26 instances, YOLO-seg `nc: 2` |

## Validation

```text
$ .venv/Scripts/python -m pytest -q
340 passed, 10 skipped, 1 warning
$ compileall / node --check / git diff --check   # ok
```

New or changed tests:
- **Calibrator:** one pass per sample, best-run choice, correct negatives with a required positive, `no_target_in_samples`.
- **VLM validator:** presence check, cache reuse, a missed target, absence accepted at low confidence, downscaling.
- **Sampler:** prefers target images but still includes negatives (the old selector picks all 3 outliers on the same data).
- **Agent:** prompt fallback end to end, and the load order "SAM 3 for all classes → unload SAM 3 → validate → unload VLM → retry".

Seven older tests asserted the old contract (first passing threshold, one SAM 3 call per threshold, empty results rejected without a VLM call) and were updated.

## Limitations

- Calibration still runs inside `POST /confirm`. It takes 142 s here, which is too long for one HTTP request. Phase 2b moves it to a background job and renders `calibration.progress` live.
- Unloading and reloading SAM 3 costs about 10–15 s per round. A retry round (second prompt) pays it again.
- Without SigLIP vectors (assets uploaded without indexing), sampling falls back to diversity and can still pick negatives. The presence check now makes that harmless for correctness, but a sample set can end up with fewer positives.
- The fallback validator (`at least one prediction`, used when Ollama is down) cannot tell a correct empty result from a miss.
- One small real set. Label quality needs the Phase 7 evaluation harness.
