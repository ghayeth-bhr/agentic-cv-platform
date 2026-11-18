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

## How to reproduce

The two harnesses are not collected by pytest; they load real model weights and
write JSON. Use them with the CUDA environment:

```powershell
..\frames-extractor\.venv\Scripts\python.exe scripts/measure_model_runtime.py --images 20
..\frames-extractor\.venv\Scripts\python.exe scripts/measure_full_journey.py --images 20 --source .agentic-cv/derived/semantic-frames
```

Re-run with `--json-out` to keep the raw measurement next to this document.
