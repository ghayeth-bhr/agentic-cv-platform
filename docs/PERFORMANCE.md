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

## How to reproduce

The two harnesses are not collected by pytest; they load real model weights and
write JSON. Use them with the CUDA environment:

```powershell
..\frames-extractor\.venv\Scripts\python.exe scripts/measure_model_runtime.py --images 20
..\frames-extractor\.venv\Scripts\python.exe scripts/measure_full_journey.py --images 20 --source .agentic-cv/derived/semantic-frames
```

For the P7 validator path, add `--require-vlm` and set
`AGENTIC_CV_VLM_TIMEOUT_SEC` high enough for the first model load:

```powershell
$env:AGENTIC_CV_VLM_TIMEOUT_SEC = "180"
..\frames-extractor\.venv\Scripts\python.exe scripts/measure_full_journey.py `
  --images 20 --source .agentic-cv/derived/semantic-frames --require-vlm `
  --calibration-timeout 600 --json-out ..\p7-vram-measurement.json
```

Re-run with `--json-out` to keep the raw measurement next to this document.
