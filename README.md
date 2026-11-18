# Agentic CV Platform

A local-first platform for managing image/video assets, searching them semantically, and building detection/segmentation datasets with prompted SAM 3 labeling.

This repository currently contains a **strong infrastructure core and a growing product surface**. P0 (`fix/runtime-foundation`) made the local environment honest and diagnosable. P1 (`feature/label-execution-engine`) now executes queued label jobs, builds dataset versions from annotation revisions, and produces downloadable COCO/YOLO archives. The browser-level Assets and Agent experience is the next phase.

## Current status

| Area | Status |
|---|---|
| Asset registry, resumable uploads, local/S3 object storage | Implemented |
| SQLite and opt-in PostgreSQL stores (assets, jobs, outbox, preview, sub-assets, annotations, dataset versions, pgvector) | Implemented |
| Semantic indexing with SigLIP + persistent SQLite/pgvector index | Implemented |
| Search results rendered as images; sub-assets materialized into the gallery | Implemented |
| Prompted SAM 3 on a selected asset source/frame (`/v1/assets/{id}/predict`) | Implemented |
| Auth, RBAC, OIDC/JWT, metrics, worker leases, CI/load/failure tests | Implemented |
| `agentic-cv doctor` environment diagnostics and typed `503 model_unavailable` errors | Added in P0 |
| Background label worker + asset→image/video-frame resolver | Added in P1 |
| Dataset builder over latest annotation revisions | Added in P1 |
| COCO/YOLO export archives with images and `zip` download | Added in P1 |
| **Agent chat** (task spec → confirm → calibrate → label → dataset) | **Not built yet** |
| **Browser-level label/dataset flow** | **Next (P2/P3)** |
| **Manual refinement UI**, dataset gallery, live step cards | **Not built yet** |

The original root README was a 950-line concatenation of 23 branch READMEs. Those are preserved verbatim in [`docs/branches/`](docs/branches/README.md).

## Quickstart (Windows)

Prerequisites: Python 3.11+, [`uv`](https://docs.astral.sh/uv/), Git. The owner's machine uses Python 3.12.13 on an RTX 4050 Laptop (6 GB VRAM).

```powershell
cd C:\Users\bhrga\OneDrive\Desktop\Agentic-CV\agentic-cv-platform

# Base app + tests (no GPU, no torch, no live services)
uv sync

# Optional: ML runtime for semantic search and SAM 3
uv sync --extra ml

# Optional: PostgreSQL stores/vectors
uv sync --extra postgres
```

`uv sync --extra ml` installs the validated ML stack (`torch 2.6.x`, `torchvision 0.21.x`, `transformers 5.14.x`, `accelerate`). PyPI wheels may be CPU-only on Windows; for CUDA, install the matching CUDA wheels explicitly:

```powershell
uv pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 `
  --index-url https://download.pytorch.org/whl/cu124
```

Keep `torch` and `torchvision` versions matched. The default device is `cuda` when torch reports a usable CUDA device and `cpu` otherwise (precision is forced to `float32` on CPU). Set `AGENTIC_CV_DEVICE` to override the automatic choice.

### Check the environment

```powershell
agentic-cv doctor
agentic-cv doctor --no-network
```

The doctor checks Python, the selected device, PyTorch/CUDA, torchvision, transformers, accelerate, Hugging Face cache/access for `facebook/sam3` and SigLIP, Ollama reachability and model presence, free disk, and the API port. Every failure includes a fix. It exits non-zero if any check fails.

### Run

```powershell
agentic-cv serve --data-dir .agentic-cv --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. Useful endpoints: `/api-info`, `/docs`, `/health/live`, `/health/ready`, `/health`, `/v1/metrics`.

`agentic-cv serve` starts a label worker and an export worker in the app process. To run dataset-export jobs in a separate process instead:

```powershell
agentic-cv worker --data-dir .agentic-cv --worker-id export-worker
```

Restart the server after Python changes; hard-refresh (`Ctrl+F5`) after JS/CSS changes.

### Existing local ML environment

The owner already has a working ML environment at `..\frames-extractor\.venv` (CUDA-enabled torch/torchvision, transformers, app dependencies). It can run the app without reinstalling:

```powershell
..\frames-extractor\.venv\Scripts\agentic-cv.exe serve --data-dir .agentic-cv --host 127.0.0.1 --port 8000
```

Do not run `ollama serve` if Ollama is already running on `127.0.0.1:11434`.

## What works today (browser)

1. **Assets → Gallery**: list assets, thumbnails, source counts, "Index for search".
2. **Assets → Upload**: create an asset and upload an image or video. Uploaded content is auto-indexed for search.
3. **Assets → Search library**: semantic search across indexed frames. Results show images. Select results and **Create sub-asset**; the sub-asset is materialized as a real gallery asset.
4. **Tools → Quick test**: run prompted SAM 3 on an existing asset source/frame or an uploaded image. Bounding boxes are scaled to the preview image.
5. **Jobs & Exports → Label jobs / Exports**: inspect durable preview events and queue/inspect dataset-export jobs by dataset version.

The P1 backend path is available over HTTP even though the browser does not expose it yet: `POST /v1/label-jobs` → poll `GET /v1/label-jobs/{id}` → `POST /v1/datasets` → `POST /v1/dataset-exports` → `GET /v1/dataset-exports/{id}/download`. `agentic-cv serve` starts both the label and export workers by default.

This is a single-user local workbench, not yet the agentic product described below.

## Target product (the plan)

Two primary sections:

- **Assets**: manage resources; search inside an asset with SigLIP; create sub-assets from results.
- **Agent**: chat with an agent that extracts a task spec (objects, visual attributes, detection vs segmentation), calibrates SAM 3 on a stratified sample with a VLM validator, labels the whole asset, builds a dataset, shows a preview gallery, and offers COCO/YOLO download.

The execution plan is tracked in the audit document supplied by the owner (P0–P6). P0 is this branch. The next phases are:

- **P1** `feature/label-execution-engine`: background runner, asset→image resolver, dataset builder, export download, job progress. Implemented in this branch; HTTP e2e and crash/cancel recovery are covered by tests.
- **P2** `feature/assets-experience`: single-workspace default, multi-file/chunked uploads, auto-index jobs, asset detail page, in-asset search, sub-asset lineage.
- **P3** `feature/agent-chat`: session persistence, LLM task-spec agent, VLM-calibrated pipeline, chat UI, dataset gallery.
- **P4** `feature/manual-refinement-ui`: point-click refinement wired to revisions.
- **P5** `feature/gpu-resource-manager`: model serialization, lazy load/unload, batching, measured performance.
- **P6** `feature/ux-e2e-quality`: Playwright journeys, empty/loading/error states, contract tests.

No phase counts as complete until its feature is reachable from the browser and covered by an HTTP-level test.

## Architecture

```
FastAPI app (application.py)
├── api.py                 assets, uploads, orchestration, label jobs, calibration, SAM3 predict
├── semantic_api.py        search, record images, asset indexing, sub-asset materialization
├── preview_api.py         replayable label preview events
├── sam3_service.py        standalone multipart /v1/predict
├── web/                   sidebar workbench (HTML/CSS/JS, no build step)
├── asset_registry.py      SQLite asset/source registry
├── uploads.py             resumable uploads + local/S3 object storage
├── semantic_search.py     provider-neutral cosine search
├── semantic_indexing.py   images + sampled video frames → SigLIP vectors
├── persistent_vectors.py  SQLite vector index
├── postgres_pgvector.py   opt-in pgvector index
├── sam3.py                lazy local SAM 3 provider
├── embeddings.py          lazy SigLIP embeddings
├── label_workflow.py      durable label-job state machine
├── label_worker.py        queued label-job polling + expired-lease recovery
├── label_images.py        asset sources/video frames → stable LabelImage records
├── batch_labeling.py      SAM 3 batch labeling + annotation persistence
├── dataset_builder.py     latest revisions → immutable dataset version + image files
├── dataset.py             COCO/YOLO serialization
├── dataset_store.py       tenant dataset versions
├── dataset_export.py      COCO/YOLO files, images, manifest, zip archive
├── dataset_export_workflow.py / dataset_export_worker.py
├── job_store.py           durable jobs, leases, heartbeat, cancel
├── outbox.py              durable event outbox
├── auth.py                opt-in static API key / OIDC + RBAC
└── observability.py       request logs + metrics
```

Heavy dependencies are lazy or injectable. Default `pytest` runs with fakes and needs no GPU, network, or live services.

## API map

| Method | Path | Purpose |
|---|---|---|
| POST/GET | `/v1/assets` | create/list assets |
| POST | `/v1/assets/{id}/uploads` | create a resumable upload |
| PUT | `/v1/uploads/{id}/parts/{n}` | upload a part |
| POST | `/v1/uploads/{id}/complete` | finalize and register a source |
| GET | `/v1/assets/{id}/thumbnail` | representative thumbnail |
| GET | `/v1/assets/{id}/image` | full representative image/frame (`source_id` optional) |
| POST | `/v1/assets/{id}/index` | index an asset for semantic search |
| POST | `/v1/search` | semantic search |
| GET | `/v1/search/records/{id}/image` | search-result image |
| POST | `/v1/sub-assets/from-search` | create a sub-asset (and gallery asset) |
| POST | `/v1/assets/{id}/predict` | prompted SAM 3 on one source |
| POST | `/v1/predict` | multipart SAM 3 smoke endpoint |
| POST | `/v1/orchestrate` | deterministic message → `TaskSpec` (regex fallback; not the agent) |
| POST/GET | `/v1/label-jobs` | submit/read durable label jobs; executed by the label worker |
| POST | `/v1/label-jobs/{id}/cancel` | request cancellation of a queued/running label job |
| POST/GET | `/v1/datasets` | build/list dataset versions from succeeded label jobs |
| GET | `/v1/datasets/{id}` | read dataset-version summary |
| POST | `/v1/calibration/run` | threshold calibration on uploaded images |
| POST/GET | `/v1/dataset-exports` | submit/read dataset-export jobs |
| GET | `/v1/dataset-exports/{id}/download` | download the packaged COCO/YOLO zip |
| GET | `/v1/preview/...` | replay label preview events |

## Testing

```powershell
.venv\Scripts\python -m pytest -q
```

Current local result: **214 passed, 10 skipped, 1 warning** (Starlette/httpx deprecation). The 10 skips are the opt-in PostgreSQL tests. The P1 tests include an HTTP-only fake-SAM journey through upload, labeling, dataset creation, COCO/YOLO export, zip download, cancellation, and expired-lease recovery.

Real PostgreSQL validation is opt-in:

```powershell
$env:AGENTIC_CV_TEST_POSTGRES_DSN = "postgresql://postgres@localhost/cv"
.venv\Scripts\python -m pytest -q -m postgres
```

Default runtime uses SQLite; `AGENTIC_CV_POSTGRES_DSN` enables PostgreSQL stores/vectors.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `AGENTIC_CV_DEVICE` | `auto` (`cuda` when available, else `cpu`) | `cpu`, `cuda`, or `cuda:<index>` |
| `AGENTIC_CV_PRECISION` | `float16` (`float32` on CPU) | `float16`, `bfloat16`, `float32` |
| `AGENTIC_CV_POSTGRES_DSN` | unset | opt-in PostgreSQL + pgvector |
| `AGENTIC_CV_OIDC_ISSUER/AUDIENCE/JWKS_URL` | unset | opt-in OIDC/JWT auth |
| `AGENTIC_CV_S3_BUCKET/ENDPOINT_URL` + `AWS_REGION` | unset | opt-in S3/MinIO storage |
| `AGENTIC_CV_RABBITMQ_HOST/PORT/EXCHANGE` | unset | opt-in RabbitMQ publishing |

Model access: `facebook/sam3` is gated. Request access on Hugging Face and set `HF_TOKEN`; the doctor verifies cache/access without downloading weights.

## Demo script (current capabilities)

1. `agentic-cv doctor` and `agentic-cv serve`.
2. Upload one image and one short video to a new asset.
3. Wait for auto-indexing; open **Search library** and search for an object.
4. Select results and create a sub-asset; confirm it appears in the gallery.
5. Open **Quick test**, select the sub-asset and a source, run a prompt, check that boxes align with the image.
6. Open **Jobs & Exports** and inspect a label job's preview events or an export job.

## Limitations / safety

- Interactive prediction is one image/frame per request. Whole-asset autolabeling is available through the HTTP label-job → dataset → export flow, but the browser does not expose it until P2/P3.
- `agentic-cv serve` executes label jobs and export jobs. The separate `agentic-cv worker` command still runs only the export worker.
- Label jobs that are cancelled mid-run can leave partial annotation revisions; dataset creation requires a succeeded job, so partial runs are not packaged.
- The agent chat, task-spec LLM, VLM calibration bridge, dataset gallery, and refinement UI are not reachable from the UI yet.
- SAM 3 is subject to Meta's SAM License and gated weights; this repository is not legal advice or a hosted-product clearance. See `docs/branches/sam-3-licensing-and-hosted-product-review-roadmap-item-10-complete.md`.
- Do not commit `.agentic-cv/`, model weights, tokens, or local exports.

## Branch documentation

- Current P1 branch: [`docs/branches/label-execution-engine.md`](docs/branches/label-execution-engine.md)
- P0 branch record: [`docs/branches/fix-runtime-foundation.md`](docs/branches/fix-runtime-foundation.md)
- Historical branch READMEs: [`docs/branches/README.md`](docs/branches/README.md)
- Repository hygiene proposal: [`docs/REPO_HYGIENE.md`](docs/REPO_HYGIENE.md)

## License

No license file is currently present in the repository. SAM 3 and model weights have separate upstream terms; review before distribution or hosted use.
