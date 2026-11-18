# Agentic CV Platform

A local-first platform for managing image/video assets, searching them semantically, and building detection/segmentation datasets with prompted SAM 3 labeling.

This repository now contains a working local product path: **Assets** for upload/index/search/sub-assets and an **Agent** chat that plans a task, calibrates a threshold, labels the whole asset, builds a dataset version, and offers a downloadable COCO archive. P0 (`fix/runtime-foundation`) made the runtime honest and diagnosable. P1 (`feature/label-execution-engine`) added durable label execution, dataset builds, and COCO/YOLO archives. P2 (`feature/assets-experience`) replaced the UUID-driven workbench with one workspace. P3 (`feature/agent-chat`) connects the chat to the existing execution engine and adds durable sessions, a planner with a visible local fallback, calibration timeouts, cancellation, and browser download. Manual refinement is P4.

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
| Single-workspace Assets UI with no tenant/UUID controls | Added in P2 |
| Multi-file chunked uploads + durable auto-index jobs | Added in P2 |
| Asset detail, in-asset search, and sub-asset lineage | Added in P2 |
| Datasets and Jobs lists readable from the workbench | Added in P2 |
| Browser upload → index → search → sub-asset journey covered by Playwright | Added in P2 |
| Agent chat: plan → confirm → calibrate → label whole asset → dataset → COCO download | Added in P3 |
| Durable agent sessions/messages with browser restore and cancel/timeout handling | Added in P3 |
| Ollama task planner with a visible deterministic rules fallback | Added in P3 |
| Agent chat covered by HTTP and Playwright end-to-end tests | Added in P3 |
| Real 20-image SAM 3 run: 20/20 annotations, 1.96 GB peak VRAM, 54 s total | Measured in P3 |
| **Manual refinement UI** | **Next (P4)** |

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

`agentic-cv serve` starts label, search-index, export, and agent-advance workers in the app process. To run dataset-export jobs in a separate process instead:

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

The workbench has two primary sections, **Assets** and **Agent**, plus a small **Datasets / Jobs** library area. It uses one local workspace; there is no tenant input and no UUID shown.

1. **Assets**: list assets by name and thumbnail. Open a card to inspect its sources, media type, and lineage.
2. **Upload media**: create one asset from several images/videos. Files are hashed and uploaded in 5 MB chunks; a durable search-index job is queued automatically when upload finishes.
3. **Search the library**: search all assets or only the selected asset with a natural-language description. Results show images and scores.
4. **Save results as sub-asset**: materializes the matches as a real gallery asset with parent-asset query lineage shown in its detail panel.
5. **Agent**: choose an asset, describe the task, inspect the extracted specification, and confirm. The agent calibrates a threshold on a sample, labels the whole asset in the background, builds a dataset version, packages a COCO archive, and exposes a download button. Sessions survive a page reload; failures and timeouts are shown in the chat.
6. **Datasets**: list dataset versions built from label revisions and export a COCO archive.
7. **Jobs**: inspect background search-index, labeling, and export jobs, including failures.

The Agent planner asks your local Ollama model (`qwen3-vl:4b` by default, configurable) for a strict JSON task specification. If Ollama is unavailable, it falls back to the deterministic local planner and says so in the chat instead of failing. The P1 label path remains available over HTTP: `POST /v1/label-jobs` → poll `GET /v1/label-jobs/{id}` → `POST /v1/datasets` → `POST /v1/dataset-exports` → `GET /v1/dataset-exports/{id}/download`.

This is a single-user local workbench, not a hosted service.

## Target product (the plan)

Two primary sections:

- **Assets**: manage resources; search inside an asset with SigLIP; create sub-assets from results.
- **Agent**: chat with an agent that extracts a task spec (objects, visual attributes, detection vs segmentation), calibrates SAM 3 on a stratified sample with a VLM validator, labels the whole asset, builds a dataset, shows a preview gallery, and offers COCO/YOLO download.

The execution plan is tracked in the audit document supplied by the owner (P0–P6). P0 is the runtime-foundation branch. The next phases are:

- **P1** `feature/label-execution-engine`: background runner, asset→image resolver, dataset builder, export download, job progress. Implemented in this branch; HTTP e2e and crash/cancel recovery are covered by tests.
- **P2** `feature/assets-experience`: single-workspace default, multi-file/chunked uploads, auto-index jobs, asset detail page, in-asset search, sub-asset lineage. Implemented in this branch; covered by a Playwright upload → index → search → sub-asset journey.
- **P3** `feature/agent-chat`: session persistence, LLM task-spec agent, calibrated pipeline, chat UI, dataset download. Implemented in this branch; covered by HTTP and browser end-to-end tests.
- **P4** `feature/manual-refinement-ui`: point-click refinement wired to revisions.
- **P5** `feature/gpu-resource-manager`: model serialization, lazy load/unload, batching, measured performance.
- **P6** `feature/ux-e2e-quality`: Playwright journeys, empty/loading/error states, contract tests.

No phase counts as complete until its feature is reachable from the browser and covered by an HTTP-level test.

## Architecture

```
FastAPI app (application.py)
├── api.py                 assets, uploads, orchestration, label jobs, calibration, SAM3 predict
├── agent.py               plan/confirm/calibrate/label/dataset orchestration
├── agent_api.py           /v1/agent/sessions HTTP surface
├── agent_store.py         durable SQLite agent sessions + messages
├── agent_worker.py        advances labeling → dataset build → export
├── ollama_text.py         strict JSON task-spec client
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
├── index_worker.py        durable auto-index jobs + expired-lease recovery
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
| POST | `/v1/assets/{id}/index` | index an asset for semantic search (synchronous) |
| POST | `/v1/assets/{id}/index-jobs` | queue a durable auto-index job |
| GET | `/v1/index-jobs/{id}` | read index-job status |
| POST | `/v1/search` | semantic search |
| GET | `/v1/search/records/{id}/image` | search-result image |
| POST | `/v1/sub-assets/from-search` | create a sub-asset (and gallery asset) |
| GET | `/v1/sub-assets` | list sub-assets with parent names and lineage |
| GET | `/v1/jobs` | list background jobs for the tenant |
| POST/GET | `/v1/agent/sessions` | plan an agent task / list sessions |
| GET | `/v1/agent/sessions/{id}` | read a durable agent session |
| POST | `/v1/agent/sessions/{id}/confirm` | calibrate (or use a fixed threshold) and start whole-asset labeling |
| POST | `/v1/agent/sessions/{id}/cancel` | cancel the active labeling/export run |
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

Current local result: **235 passed, 10 skipped, 1 warning** (Starlette/httpx deprecation). The 10 skips are the opt-in PostgreSQL tests. The P1 tests include an HTTP-only fake-SAM journey through upload, labeling, dataset creation, COCO/YOLO export, zip download, cancellation, and expired-lease recovery. The P3 tests add a chat → confirm → calibration → label → dataset → browser-download journey, plus calibration-timeout, cancellation, fixed-threshold fallback, planner fallback, Ollama thinking-field compatibility, and durable-session coverage.

The browser tests drive a real uvicorn server with fake providers. They skip if the Playwright browser binary is not installed:

```powershell
.venv\Scripts\python -m playwright install chromium
.venv\Scripts\python -m pytest tests/test_assets_ui_playwright.py tests/test_agent_ui_playwright.py -q
```

The P3 real-model manual run was executed on the owner's RTX 4050 Laptop (6 GB VRAM) with CUDA `torch 2.6.0+cu124` and cached `facebook/sam3`. It uploaded 20 generated images through the real HTTP API, calibrated at threshold 0.30 with 5/5 sample agreement, labeled all 20 images, and downloaded a COCO archive with 20 images and 20 annotations. Calibration plus the first model load took 38.49 s, the label job took 13.75 s, total wall time was 53.97 s, and peak allocated VRAM was 1.96 GB. The real Ollama planner (`qwen3-vl:4b`) returned a valid `TaskSpec` in 9.37 s. These are local measurements, not production benchmarks.

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
| `AGENTIC_CV_AGENT_PLANNER` | `ollama` | set `rules` to skip the text model entirely |
| `AGENTIC_CV_TEXT_MODEL` | `qwen3-vl:4b` | Ollama model used for task-spec JSON |
| `AGENTIC_CV_OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `AGENTIC_CV_OLLAMA_TIMEOUT_SEC` | `30` | task-spec request timeout |
| `AGENTIC_CV_CALIBRATION_TIMEOUT_SEC` | `120` | agent calibration budget; a timeout returns HTTP 504 |
| `AGENTIC_CV_OIDC_ISSUER/AUDIENCE/JWKS_URL` | unset | opt-in OIDC/JWT auth |
| `AGENTIC_CV_S3_BUCKET/ENDPOINT_URL` + `AWS_REGION` | unset | opt-in S3/MinIO storage |
| `AGENTIC_CV_RABBITMQ_HOST/PORT/EXCHANGE` | unset | opt-in RabbitMQ publishing |

Model access: `facebook/sam3` is gated. Request access on Hugging Face and set `HF_TOKEN`; the doctor verifies cache/access without downloading weights.

## Demo script (current capabilities)

1. `agentic-cv doctor` and `agentic-cv serve`.
2. Open **Assets → Upload media**, name the asset, and select several photos plus a short video.
3. Wait for the upload to finish and the auto-index job to report `indexed for search`.
4. Open the asset card; confirm its sources and detail panel.
5. In **Search the library**, switch scope to **Selected asset**, search for an object, and inspect the returned images.
6. Click **Save results as sub-asset**; open the new card and confirm the query and parent-asset lineage.
7. Open **Agent**, choose the asset, and enter `Create a segmentation dataset of workers wearing yellow vests.`
8. Inspect the extracted task/prompt/sample specification, then click **Confirm and calibrate**.
9. Watch calibration, whole-asset labeling, dataset build, and packaging progress; download the COCO archive from the run panel.
10. Open **Jobs** to inspect the label/export jobs, or **Datasets** to list the new version and export it again.
11. Reload the page and reopen **Agent** to confirm the finished run is restored from the local session.

The release demo in the owner's section 7 also calls for manual refinement of one bad image and a YOLO download from the chat; refinement is P4, and the Agent currently downloads COCO (YOLO remains available through the dataset-export API).

## Limitations / safety

- The browser now covers upload, indexing, search, sub-assets, Agent labeling, dataset listing, and downloads. Manual correction of an individual bad mask is still P4.
- The Agent's default calibration validator is “SAM 3 returned at least one prediction.” The existing VLM validator adapter is injectable but is not wired into the default Agent path yet.
- The Agent chat downloads COCO. YOLO exports exist behind `/v1/dataset-exports`; the chat does not yet offer a format switch.
- `agentic-cv serve` executes label, index, export, and agent-advance jobs. The separate `agentic-cv worker` command still runs only the export worker.
- Label jobs cancelled mid-run can leave partial annotation revisions; dataset creation requires a succeeded job, so partial runs are not packaged.
- The Agent and Assets browser tests use fake SAM 3/embedding providers and a deterministic planner. The separate P3 manual run exercised real SAM 3 and the real Ollama planner; it does not establish quality on your production photos or a full 6 GB laptop UI session.
- The P3 planner asks Ollama for strict JSON. If the model is down or returns invalid JSON, the chat visibly falls back to deterministic rules.
- Sub-assets copy the matched search images into a new asset; they are snapshots, not live filters over the parent asset.
- SAM 3 is subject to Meta's SAM License and gated weights; this repository is not legal advice or a hosted-product clearance. See `docs/branches/sam-3-licensing-and-hosted-product-review-roadmap-item-10-complete.md`.
- Do not commit `.agentic-cv/`, model weights, tokens, or local exports.

## Branch documentation

- Current P3 branch: [`docs/branches/agent-chat.md`](docs/branches/agent-chat.md)
- P2 branch record: [`docs/branches/assets-experience.md`](docs/branches/assets-experience.md)
- P1 branch record: [`docs/branches/label-execution-engine.md`](docs/branches/label-execution-engine.md)
- P0 branch record: [`docs/branches/fix-runtime-foundation.md`](docs/branches/fix-runtime-foundation.md)
- Historical branch READMEs: [`docs/branches/README.md`](docs/branches/README.md)
- Repository hygiene proposal: [`docs/REPO_HYGIENE.md`](docs/REPO_HYGIENE.md)

## License

No license file is currently present in the repository. SAM 3 and model weights have separate upstream terms; review before distribution or hosted use.
