# Measured performance

All numbers below were measured locally on the owner's laptop. They are
observations, not targets or production capacity claims.

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU, 6140.5 MB reported total |
| Device / precision | `cuda` / `float16` |
| Python | 3.12.13 |
| PyTorch | `2.6.0+cu124` |
| Transformers | `5.14.1` |
| ML environment | `..\frames-extractor\.venv` |
| App commit for baseline | P4 tip `8505a7f` |
| App commit for post-change run | P5 worktree (resource manager + box clipping) |
| App branch for P7 run | `feature/real-validation-loop` (20-image run on 2026-11-19) |
| App branch for P8 runs | `feature/validator-reliability-gpu` (30-image reliability + 20-image journey on 2026-11-19) |

Windows reports physical VRAM through `torch.cuda.mem_get_info()`; the discrete
6 GB figure and the reported 6140.5 MB differ because the driver reserves some
memory.

## Baseline before P5 (P4 commit)

Command:

```powershell
..\frames-extractor\.venv\Scripts\python.exe scripts/measure_model_runtime.py `
  --images 20 --json-out ..\baseline-p5.json
```

This harness loads the embedding model, embeds 20 synthetic 640x480 images and
one text query, closes the embedding model, then runs SAM 3 on the same 20
images. The SAM 3 phase therefore ran with the embedding model unloaded.

| Phase | Wall time | Peak allocated VRAM | Peak reserved VRAM |
|---|---:|---:|---:|
| SigLIP2 embeddings, 20 images + 1 text query | 39.54 s | 2217.4 MB | 2270.0 MB |
| SAM 3 labeling, 20 images | 30.30 s | 2005.4 MB | 2182.0 MB |

The embedding model stayed at 2197.4 MB allocated / 2270.0 MB reserved after
the embedding phase. The SAM 3 model stayed at 1708.4 MB allocated / 2182.0 MB
reserved after its phase.

These numbers show why a naive full journey matters: each model fits alone, but
the app needs both, and their live models together are the risk on a 6 GB card.

## Post-P5 pressure policy

Command:

```powershell
..\frames-extractor\.venv\Scripts\python.exe scripts/measure_model_runtime.py `
  --images 20 --keep-embedding-loaded --headroom-mb 4096 `
  --json-out ..\after-p5-pressure.json
```

With `AGENTIC_CV_MODEL_HEADROOM_MB`-equivalent headroom forced above the free
VRAM, the shared resource manager closed SigLIP2 before SAM 3 loaded:

| Phase | Wall time | Peak allocated VRAM | Peak reserved VRAM |
|---|---:|---:|---:|
| SigLIP2 embeddings, 20 images + 1 text query | 56.92 s | 2217.4 MB | 2268.0 MB |
| SAM 3 labeling, 20 images, after unloading SigLIP2 | 36.70 s | 2197.4 MB | 2250.0 MB |

The recorded event was `siglip2: closed (make room for sam3)`. The SAM 3 model
ended at 1708.4 MB allocated / 2200.0 MB reserved. This is the intended
low-memory behavior: one model resident rather than two.

## Post-P5 full HTTP journey

Command:

```powershell
..\frames-extractor\.venv\Scripts\python.exe scripts/measure_full_journey.py `
  --images 20 --source .agentic-cv/derived/semantic-frames `
  --json-out ..\after-p5-full-journey.json
```

The harness starts the real local app with real SigLIP2 and SAM 3 providers and
real workers, then performs the shipped HTTP journey:

1. upload 20 images into one asset,
2. queue and finish durable search indexing,
3. plan the task with the deterministic rules planner,
4. confirm and calibrate at 0.30,
5. label all 20 images with SAM 3,
6. build the dataset and download its COCO archive,
7. click-refine one annotation,
8. accept the refinement, rebuild the dataset, export and download the refined
   COCO archive.

The source images were real 720x1280 video frames already cached in the
workspace; copies were made byte-distinct so the registry would not deduplicate
them. The run produced 34 annotations.

| Checkpoint | Wall time | Peak allocated VRAM at checkpoint |
|---|---:|---:|
| Upload 20 files (HTTP transfer only) | 1.28 s | 0.0 MB |
| Durable index job, including SigLIP2 first load and 20 image embeddings | 39.33 s | 2217.4 MB |
| Agent confirm + calibration + labeling 20 images + dataset build + COCO export | 40.29 s | 4202.8 MB |
| Click refinement, including mask preview | 1.08 s | 4202.8 MB |
| Accept refinement + rebuild + refined export download | 2.84 s | 4202.8 MB |
| Whole run peak | — | 4202.8 MB allocated / 4390.0 MB reserved |

End state from `/api-info`: SigLIP2 unloaded, SAM 3 loaded, 2855.0 MB free of
6140.5 MB. The resource manager event was
`siglip2: closed (make room for sam3)`.

The dataset download was 3,602,103 bytes; the refined download was 3,602,102
bytes. Both archives contained annotations and images. The run completed
without CUDA OOM on the 6 GB laptop.

### Notes on this run

- The agent UI/HTTP path calibrates at 0.30, labels 20 images, and builds the
  dataset in 40.29 s after indexing. The first model load dominates the first
  run; later runs reuse the warm model.
- The resource manager serializes embedding and SAM 3 inference. During this
  journey the two live models coexisted at a 4.2 GB peak because there was
  enough free VRAM; under forced pressure it unloads the other model instead.
- Label boxes are clipped to image bounds. The first real-image run exposed an
  out-of-bounds SAM 3 box, which now fails closed by clipping or dropping the
  empty box rather than failing the whole job.
- Numbers will vary with driver state, desktop GPU usage, image resolution, and
  whether weights are cached.

## Post-P7 real VLM validation

P7 changes the Agent's default calibration validator from "SAM 3 returned at
least one prediction" to the local Ollama VLM, and records the verdict evidence
and sample overlays in the calibration result. This run measures the real
20-image HTTP journey with SAM 3, SigLIP2, and the VLM in the same run.

Environment additions:

- Ollama `qwen3-vl:4b`, Q4_K_M, `size_vram` 3,364.1 MB.
- `AGENTIC_CV_VLM_TIMEOUT_SEC=180`; calibration budget 600 s.
- `doctor --no-network` on this machine: `11 ok, 0 warn, 0 fail`.

Command (PowerShell; the harness starts the real local app, real providers, and
real workers):

```powershell
$env:AGENTIC_CV_VLM_TIMEOUT_SEC = "180"
..\frames-extractor\.venv\Scripts\python.exe scripts/measure_full_journey.py `
  --images 20 --source .agentic-cv/derived/semantic-frames `
  --message "Create a segmentation dataset of animals." --prompt "animals" `
  --require-vlm --timeout 900 --calibration-timeout 600 `
  --json-out ..\p7-vram-measurement.json
```

The harness printed:

```json
{
  "calibration": {
    "status": "calibrated",
    "threshold": 0.3,
    "validator_mode": "vlm",
    "validator_model": "qwen3-vl:4b",
    "notice": null,
    "thresholds_tried": [0.3],
    "attempts": 1,
    "verdicts": 5,
    "accepted_verdicts": 5,
    "overlay_ids": [
      "07e37e9c901e4bfa5c104c07.png",
      "7e8d501972fd08ec5c3d2049.png",
      "9d45db17a85200676c5b1be7.png",
      "7cf30f3fbba8da707fedf461.png",
      "88be4e4768e9f832dc817a88.png"
    ]
  },
  "agent": {
    "image_count": 20,
    "annotation_count": 34,
    "download_bytes": 3602141
  },
  "refinement": {
    "status": "accepted_rebuilt_exported",
    "revision": 2,
    "annotations": 2,
    "download_bytes": 3602143
  },
  "peak": {
    "cuda_available": true,
    "peak_allocated_mb": 4202.8,
    "peak_reserved_mb": 4390.0
  },
  "nvidia_smi": {
    "available": true,
    "peak_used_mb": 5822.0,
    "total_mb": 6141.0,
    "window_baseline_mb": 3437.0,
    "window_peak_mb": 5822.0,
    "window_growth_mb": 2385.0,
    "calibration_baseline_mb": 5620.0,
    "calibration_peak_mb": 5822.0,
    "calibration_growth_mb": 202.0
  }
}
```

Checkpoint timings from the same report:

| Checkpoint | Wall time | Peak allocated VRAM at checkpoint |
|---|---:|---:|
| Upload 20 files (HTTP transfer only) | 1.57 s | 0.0 MB |
| Durable index job, including SigLIP2 first load and 20 image embeddings | 47.03 s | 2217.4 MB |
| Agent confirm + VLM calibration + labeling 20 images + dataset build + COCO export | 383.21 s | 4202.8 MB |
| Click refinement, including mask preview | 1.24 s | 4202.8 MB |
| Accept refinement + rebuild + refined export download | 3.11 s | 4202.8 MB |

The calibration card received five real sample overlays and five VLM verdicts;
all five accepted at threshold 0.30 on the first attempt. There was no fallback
notice. `nvidia-smi` sampled the whole GPU while Ollama was already resident:
the run started at 3,437 MB used and peaked at 5,822 MB of 6,141 MB, while the
calibration window itself moved from 5,620 MB to 5,822 MB (+202 MB).

After the run, the Ollama process still reported:

```text
qwen3-vl:4b  3364.1 MB VRAM  Q4_K_M
```

These counters are not additive. `torch.cuda.max_memory_allocated()` reports the
PyTorch allocator's peak (4,202.8 MB); `nvidia-smi` reports dedicated GPU memory
used by every process (5,822 MB peak). On Windows the driver can back part of an
allocation with shared system memory, so the allocator peak and the dedicated
counter describe different things. The run completed without a CUDA OOM, but the
6 GB figure is not a hard partition and the VLM is an external Ollama process
that the app's model manager does not control.

The VLM path is materially slower than the P5 no-VLM path: the
calibrate/label/export checkpoint was 383.21 s versus 40.29 s. The extra time is
real VLM image validation on the same 6 GB GPU, not a fixed overhead target.

### Notes on this run

- The VLM `qwen3-vl:4b` returned its JSON in Ollama's `message.thinking` field,
  not `message.content`. `OllamaVlmClient` now accepts either; without that
  compatibility the first probe failed with `message.content was not valid JSON`.
- The five calibration samples were selected by farthest-point diversity from
  the stored SigLIP vectors, not by taking the first five images.
- The default VLM validator falls back to the explicitly labeled "at least one
  prediction" check if Ollama is unreachable at confirmation or if a VLM call
  fails during calibration. In the latter case the chat and calibration card
  carry the failure reason. This run used the VLM path end to end.
- Numbers will vary with driver state, desktop GPU usage, image resolution,
  model cache state, and whether Ollama has already loaded the VLM.

## Post-P8 validator reliability, unload, and phase timings

P8 measures the validator instead of assuming it is correct, and releases the
Ollama VLM before whole-asset labeling. The raw reports are
`..\p8-reliability.json`, `..\p8-reliability-after-guard.json`, and
`..\p8-full-journey.json`.

### Reliability harness

Synthetic 320x320 images with known orange-rectangle masks give IoU ground
truth without hand-labeling production photos. The harness also drives the real
HTTP confirm path for an absent prompt.

```powershell
..\frames-extractor\.venv\Scripts\python.exe scripts/validate_calibration_reliability.py `
  --images 30 --json-out ..\p8-reliability.json
```

The first run found a real failure. It agreed with IoU on 30/30 cases and the
absent-prompt chat check passed, but the too-low threshold check **failed**:

```text
threshold=0.02, prediction_count=12, best_iou=0.2767,
false_positive_masks=2, vlm_accepted=True, vlm_confidence=0.55
reasoning: The orange shape is completely covered by the masks. Extra masks
appear to outline different parts of the same object rather than separate
background objects. The overlay matches the target.
```

The VLM was not reliable at rejecting an obviously over-permissive threshold on
its own. `VlmCalibrationValidator` now applies a deterministic
`minimum_prediction_score=0.2` guard before the VLM call, configurable with
`AGENTIC_CV_VLM_MIN_PREDICTION_SCORE`. Thresholds at or above 0.3 are unchanged;
low-score mask sets are rejected without spending a VLM call.

The same command was re-run to `..\p8-reliability-after-guard.json`. Exact
summary:

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

The low-score check now reports `prediction_count=11`,
`best_iou=0.2845`, `false_positive_masks=4`, `vlm_accepted=False`, and this
reason: `Rejected 11 low-score mask(s) below 0.20; this threshold is too
permissive.` The absent-prompt check ended in `needs_attention` with the exact
message: `I can't segment reliably from these images for “purple unicorn”. I
can retry with different samples, or you can run at a fixed threshold such as
0.50.` The VLM was unloaded at the end of the harness. This is still a
synthetic-target measurement; production-photo quality remains a human check.

### Full journey with VLM unload

```powershell
$env:AGENTIC_CV_VLM_TIMEOUT_SEC = "180"
..\frames-extractor\.venv\Scripts\python.exe scripts/measure_full_journey.py `
  --images 20 --source .agentic-cv/derived/semantic-frames `
  --message "Create a segmentation dataset of animals." --prompt "animals" `
  --require-vlm --timeout 900 --calibration-timeout 600 `
  --json-out ..\p8-full-journey.json
```

The run calibrated at 0.30 on the first attempt with five VLM verdicts, all
five accepted, and five overlay ids. The calibration card recorded
`vlm_unloaded_after_calibration: true`; the unload call took 2.06 s and the chat
received `Unloaded the vision validator after calibration; whole-asset labeling
continues with SAM 3 only.` before the labeling message. The agent produced 34
annotations; the COCO download was 3,602,151 bytes and the refined download
3,602,149 bytes.

| Checkpoint | Wall time | Peak allocated VRAM |
|---|---:|---:|
| Upload 20 files | 1.09 s | 0.0 MB |
| Durable index job | 32.69 s | 2217.4 MB |
| Agent confirm + VLM calibration + labeling + export | 369.05 s | 4202.8 MB |
| Click refinement | 0.74 s | 4202.8 MB |
| Accept + rebuild + refined export | 1.74 s | 4202.8 MB |

| Phase inside the confirm checkpoint | Seconds |
|---|---:|
| SigLIP2 model load (during index) | 30.00 |
| SAM 3 model load (during calibration) | 18.38 |
| Total first model load | 48.38 |
| VLM calibration, including SAM 3 setup | 351.39 |
| VLM unload after calibration | 2.06 |
| Label 20 images | 13.91 |
| Dataset/COCO export | 0.54 |
| Confirm to complete | 371.24 |

These phase numbers explain P7's 383.21 s checkpoint. It was dominated by real
VLM calibration, not by labeling or export; P8's equivalent checkpoint was
369.05 s, of which 351.39 s was calibration, 13.91 s labeling, and 0.54 s
export. The model-load total of 48.38 s is charged to the index and calibration
windows, not added again. P7 had no phase instrumentation, so its number cannot
be decomposed further after the fact; the difference from P8 is run variance.

Whole-GPU measurement, re-run after the unload change:

```json
{
  "peak": {
    "peak_allocated_mb": 4202.8,
    "peak_reserved_mb": 4390.0
  },
  "nvidia_smi": {
    "available": true,
    "peak_used_mb": 5914.0,
    "total_mb": 6141.0,
    "calibration_baseline_mb": 2375.0,
    "calibration_peak_mb": 5914.0,
    "calibration_growth_mb": 3539.0
  }
}
```

The app's PyTorch peak is unchanged at 4,202.8 MB allocated / 4,390.0 MB
reserved. `nvidia-smi` saw 5,914 MB of 6,141 MB at its peak while SAM 3 and the
VLM were both active during calibration. After the 2.06 s unload, labeling ran
without the Ollama model resident. These counters are not additive; on Windows
WDDM allocations can be backed by shared system memory. The unload is evidenced
by the timing and the recorded signal, not by a lower peak: the peak happens
before the unload. Another process loading several GB before calibration can
still cause an OOM because Ollama is outside the app's model manager.

### Fresh-clone gate

A clean clone of commit `7fb079d` ran `uv sync --extra ml` (90 packages, 69
installed, CUDA wheels `torch==2.6.0+cu124` / `torchvision==0.21.0+cu124`). Its
`doctor --no-network --port 8126` reported `device=cuda` and `11 ok, 0 warn, 0
fail`. The same clone then produced 3 real `/v1/search` results and a complete
agent run with 3 images, 3 annotations, and a 2,483-byte COCO zip. The smoke
script first failed with `start_workers=False` (queued index job) and then with
a wrong response key; both were harness-script errors and were corrected before
the passing run. Full output is in
[`docs/branches/validator-reliability-gpu.md`](branches/validator-reliability-gpu.md).

## Phase 2a calibration engine (real models, 2026-11-20)

Same machine (RTX 4050 Laptop 6 GB), `qwen3-vl:4b` on Ollama 0.31.2 for
planning and validation, `facebook/sam3`, SigLIP2 indexing. Input: 9 real
720x1280 wildlife frames in two assets; 6 contain elephants, 3 do not (dik-dik,
buffalo, gerenuk). Task from chat: segmentation of `elephants` and `tusks`,
YOLO export. Harness: `scripts/real_agent_journey.py`.

| Run | Calibration change | Elephants | Tusks | Calibration time | Result |
|---|---|---|---|---|---|
| 1 (Phase 1) | ladder, empty result = rejection | 2/5 at every threshold | 2/5 | 459 s | `needs_attention`, no dataset |
| 2 | one SAM 3 pass/sample, best threshold, VLM presence check, target-aware samples | 3/5 → 0.45 | 4/5 → 0.45 | 1101 s | dataset built |
| 3 | + presence decided by the boolean (model answered "absent" with low confidence), 768 px VLM images, SigLIP unloaded | **5/5** → 0.45 | **5/5** → 0.45 | 995 s | dataset built |
| 4 | + **phased rounds: SAM 3 never resident with the VLM** | **5/5** → 0.45 | **5/5** → 0.45 | **142 s** | dataset built |

Run 4 detail: 5 SAM 3 forward passes per class (previously 25), 20 VLM
verdicts in 90 s (~4.5 s each; ~4 s measured for a VLM call on an idle GPU vs
~45 s while SAM 3 was co-resident and Windows paged GPU memory), labeling +
build + export 27 s (includes reloading SAM 3), whole-GPU peak 5,868 MB.
Output: 9 images, 26 instances (8 elephants, 18 tusks), YOLO segmentation with
`nc: 2`, train/val/test folders. Correct negatives were recorded as such, e.g.
"No elephants in this image, so returning no masks is correct. The image shows
a buffalo...".

These are single local runs on one small set, not benchmarks; they show the
order of magnitude and the failure modes, not label quality.

## How to reproduce

The two harnesses are not collected by pytest; they load real model weights and
write JSON. Use them with the CUDA environment:

```powershell
..\frames-extractor\.venv\Scripts\python.exe scripts/measure_model_runtime.py --images 20
..\frames-extractor\.venv\Scripts\python.exe scripts/measure_full_journey.py --images 20 --source .agentic-cv/derived/semantic-frames
```

For the P8 validator reliability checks, run the separate harness with real
SAM 3 and Ollama. It deliberately writes failed cases into the JSON report:

```powershell
..\frames-extractor\.venv\Scripts\python.exe scripts/validate_calibration_reliability.py `
  --images 30 --json-out ..\p8-reliability-after-guard.json
```

For the P7/P8 VLM journey, add `--require-vlm` and set
`AGENTIC_CV_VLM_TIMEOUT_SEC` high enough for the first model load:

```powershell
$env:AGENTIC_CV_VLM_TIMEOUT_SEC = "180"
..\frames-extractor\.venv\Scripts\python.exe scripts/measure_full_journey.py `
  --images 20 --source .agentic-cv/derived/semantic-frames --require-vlm `
  --calibration-timeout 600 --json-out ..\p7-vram-measurement.json
```

Re-run with `--json-out` to keep the raw measurement next to this document.
