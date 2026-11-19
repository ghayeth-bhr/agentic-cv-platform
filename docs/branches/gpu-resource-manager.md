# P5 — GPU resource manager

Branch: `feature/gpu-resource-manager` (stacked on pushed P4 `feature/manual-refinement-ui`).

## Goal

Make the real SigLIP2 + SAM 3 journey survive on the owner's 6 GB laptop without
CUDA OOM, while keeping the existing provider injection and zero-config local
mode. Measure first; publish the observed numbers instead of targets.

## What changed

### Shared model resource manager

- Added `src/agentic_cv/model_resources.py`:
  - one process-wide `RLock` shared by local embedding and SAM 3 providers, so
    CUDA inference is serialized instead of racing or stacking temporary
    allocations;
  - registration with an estimated VRAM footprint;
  - before a model infers, compare free device memory with
    `estimate + AGENTIC_CV_MODEL_HEADROOM_MB` and close other resident models if
    it will not fit;
  - idle-TTL eviction (`AGENTIC_CV_MODEL_IDLE_UNLOAD_SEC`, default 300 s);
  - short event log and live status (loaded models, free/total MB, busy state).
- `embeddings.py` and `sam3.py` accept an optional manager. Standalone providers
  still use their own lock, so tests and other callers keep working.
- `/api-info` now reports `model_runtime.resources` with loaded models and memory.
- `LocalAppResources.close()` shuts down the manager after the providers.

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `AGENTIC_CV_MODEL_HEADROOM_MB` | `512` | Extra free VRAM required before loading a model |
| `AGENTIC_CV_MODEL_IDLE_UNLOAD_SEC` | `300` | Idle TTL; next lease closes a stale model |
| `AGENTIC_CV_EMBEDDING_VRAM_MB` | `2304` | SigLIP2 estimate used by the policy |
| `AGENTIC_CV_SAM3_VRAM_MB` | `2048` | SAM 3 estimate used by the policy |

The estimates are policy inputs, not claims about the exact allocator footprint.
`docs/PERFORMANCE.md` records the actual measured peaks.

### Real-image stability fix

The first real full journey exposed an out-of-bounds SAM 3 box
(`bbox must fit within image dimensions`), which failed the whole label job.
`clip_box(...)` now clamps xyxy boxes to image bounds in both the local SAM 3
provider and the batch labeler; an empty clipped box is dropped. Raw predictions
are still preserved for audit.

### Measurement harnesses

- `scripts/measure_model_runtime.py`: embedding + SAM 3 timing and real CUDA
  peaks, with managed/unmanaged and forced-headroom modes.
- `scripts/measure_full_journey.py`: real HTTP journey through upload, durable
  indexing, agent plan/confirm/calibration/label, COCO download, click
  refinement, rebuild, and refined export download.

Neither script runs under pytest.

## Acceptance evidence

Measured on the RTX 4050 Laptop (6 GB), CUDA `torch 2.6.0+cu124`:

```text
baseline (P4, 20 synthetic images)
  embeddings: 39.54 s, 2217.4 MB peak
  SAM 3:      30.30 s, 2005.4 MB peak (embedding model unloaded)

after P5 full HTTP journey (20 real 720x1280 frames, 34 annotations)
  upload:                  1.28 s
  durable index job:      39.33 s
  agent calibrate+label:  40.29 s
  click refinement:        1.08 s
  rebuild + export:        2.84 s
  peak VRAM:            4202.8 MB allocated / 4390.0 MB reserved
  result:                 no OOM; COCO archives downloaded
```

Forced-pressure run (`--headroom-mb 4096`) unloaded SigLIP2 before SAM 3 and
kept the SAM 3 peak at 2197.4 MB, proving the unload path. Exact commands and
tables are in [`docs/PERFORMANCE.md`](../PERFORMANCE.md).

Test suite:

```text
.\.venv\Scripts\python -m pytest -q
246 passed, 10 skipped, 1 warning in 35.14s
```

The new `tests/test_model_resources.py` covers serialization, pressure unload,
idle unload, CPU no-torch behavior, and provider reload after an unload. The
SAM 3/batch-label tests cover box clipping.

## Limitations

- The manager does not preload models or reserve VRAM in the driver; it reacts to
  free memory at each lease.
- The default 4.2 GB co-resident peak fits this 6 GB laptop with the desktop in
  its measured state. Another process using several GB of VRAM can still cause
  an OOM before the manager can react.
- Inference remains one image at a time. SAM 3 batching was not added because it
  raises peak VRAM on a 6 GB card; the measured goal was stable whole-asset
  execution.
- `/api-info` opens the CUDA context to read memory. On a CPU-only install it
  reports `null` memory without importing torch.
- The measurement harness is local and manual; timings vary with cache state,
  driver state, desktop GPU usage, and image size.
