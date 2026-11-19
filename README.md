# Agentic CV Platform

A local-first platform for managing image/video assets, searching them semantically, and building detection/segmentation datasets with prompted SAM 3 labeling.

This repository now contains a working local product path: **Assets** for upload/index/search/sub-assets, an **Agent** chat that plans a task, calibrates a threshold with the real local VLM, labels the whole asset, builds a dataset version, and offers a downloadable COCO archive, and a **Review labels** flow that turns a positive/negative click into an annotation revision and a fresh dataset version. P0 (`fix/runtime-foundation`) made the runtime honest and diagnosable. P1 (`feature/label-execution-engine`) added durable label execution, dataset builds, and COCO/YOLO archives. P2 (`feature/assets-experience`) replaced the UUID-driven workbench with one workspace. P3 (`feature/agent-chat`) connects the chat to the existing execution engine and adds durable sessions, a planner with a visible local fallback, calibration timeouts, cancellation, and browser download. P4 (`feature/manual-refinement-ui`) wires the click refinement service to dataset images, annotation revisions, and a rebuilt/exportable dataset version. P5 (`feature/gpu-resource-manager`) serializes local model inference, unloads models under memory pressure, and records a real 20-image whole-journey measurement on the 6 GB laptop. P6 (`feature/ux-e2e-quality`) makes loading, empty, and recoverable error states explicit across the workspace and adds browser-state and UI/API contract tests. P7 (`feature/real-validation-loop`) makes the VLM the default calibration validator, stores sample overlays and verdict evidence, makes calibration cancellable, selects diverse samples from SigLIP vectors, and adds a COCO/YOLO selector to dataset export.

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
| Review labels: click a dataset image, preview a mask, accept a revision, rebuild and export | Added in P4 |
| Manual refinement covered by HTTP and Playwright end-to-end tests | Added in P4 |
| Shared GPU model lock, memory-pressure unload, idle unload, `/api-info` model status | Added in P5 |
| Real 20-image HTTP journey: upload → index → agent label → COCO download → refine → rebuilt export, 4.20 GB peak VRAM, no OOM | Measured in P5 |
| Explicit loading/empty/error states with retries across Assets, Datasets, and Jobs | Added in P6 |
| UI/API contract tests for routes, response envelopes, DOM ids, and navigation | Added in P6 |
| Real VLM calibration: `qwen3-vl:4b` verdicts, sample overlays, and a labeled fallback | Added in P7 |
| Diversity-based calibration samples from stored SigLIP vectors | Added in P7 |
| Dataset-card COCO / YOLO detection / YOLO segmentation export selector | Added in P7 |
| Doctor warning for CPU-only torch on an NVIDIA machine + Windows cu124 lock config | Added in P7 |
| Real 20-image VLM journey: 5/5 verdicts, 5.82 GB whole-GPU peak, 383 s calibration/label/export | Measured in P7 |
| Run-scoped annotation sets, tombstone deletes, persisted source-aware splits, 0-based YOLO with split folders, integer-id COCO, time-based shared video frames, SQLite WAL | Fixed in Phase 0 ([`docs/branches/data-correctness.md`](docs/branches/data-correctness.md)) |
| Conversational agent: asks for missing details, several assets and classes per run, SAM 3 prompt agent, COCO/YOLO chosen in chat | Added in Phase 1 ([`docs/branches/agent-dialogue.md`](docs/branches/agent-dialogue.md)) |
| Select individual search results (checkboxes, min-score slider) and save only the selection as a named asset; tenant-scoped search | Added in [`docs/branches/search-selection.md`](docs/branches/search-selection.md) |
| Calibration engine: one SAM 3 pass per sample, best threshold, correct negatives via VLM presence check, target-aware samples, prompt alternatives, SAM 3/VLM never co-resident (real run: 459 s failed → 142 s calibrated) | Added in Phase 2a ([`docs/branches/calibration-engine.md`](docs/branches/calibration-engine.md)) |

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

`uv sync --extra ml` installs the validated ML stack (`torch 2.6.x`, `torchvision 0.21.x`, `transformers 5.14.x`, `accelerate`). On Windows the lock now routes `torch`/`torchvision` through the `pytorch-cu124` index, so the sync installs CUDA wheels instead of the CPU-only PyPI wheels. Verify with `agentic-cv doctor`: if it sees an NVIDIA GPU and `torch.version.cuda` is empty, it warns and prints the fix. For a different CUDA version or platform, install the matching wheels explicitly:

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

The doctor checks Python, the selected device, PyTorch/CUDA, torchvision, transformers, accelerate, Hugging Face cache/access for `facebook/sam3` and SigLIP, Ollama reachability and model presence, free disk, and the API port. Every failure includes a fix; an NVIDIA GPU with a CPU-only torch build is a warning with the Windows wheel command. It exits non-zero if any check fails.

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
4. **Save selected as asset**: every result starts selected; untick the images that do not belong (or use the min-score slider), name the asset, and only the selection is copied into a new gallery asset with query lineage.
5. **Agent**: choose an asset, describe the task, inspect the extracted specification, and confirm. The agent selects a diverse sample using stored SigLIP vectors, calibrates a threshold with the real VLM, and shows the validator mode, thresholds tried, sample overlays, and each verdict. It then labels the whole asset in the background, builds a dataset version, packages a COCO archive, and exposes a download button. Sessions survive a page reload; failures, fallbacks, and timeouts are shown in the chat.
6. **Datasets**: list dataset versions built from label revisions and export them as COCO, YOLO detection, or YOLO segmentation.
7. **Review labels**: in a dataset card, open **Review labels**, click an image, place positive/negative points, accept the previewed mask, and the app writes a new annotation revision, rebuilds a fresh dataset version, and leaves it ready to export.
8. **Jobs**: inspect background search-index, labeling, and export jobs, including failures.

The Agent planner asks your local Ollama model (`qwen3-vl:4b` by default, configurable) for a strict JSON task specification. If Ollama is unavailable, it falls back to the deterministic local planner and says so in the chat instead of failing. The same model is the default calibration validator: if Ollama and the model are reachable, calibration sends each sample overlay to it and stores accepted/confidence/reasoning per sample. If it is unreachable, calibration uses the explicitly labeled `at least one prediction` fallback; `AGENTIC_CV_AGENT_VALIDATOR=any-prediction` forces that fallback. The P1 label path remains available over HTTP: `POST /v1/label-jobs` → poll `GET /v1/label-jobs/{id}` → `POST /v1/datasets` → `POST /v1/dataset-exports` → `GET /v1/dataset-exports/{id}/download`.

This is a single-user local workbench, not a hosted service.

## Target product (the plan)

Two primary sections:

- **Assets**: manage resources; search inside an asset with SigLIP; create sub-assets from results.
- **Agent**: chat with an agent that extracts a task spec (objects, visual attributes, detection vs segmentation), calibrates SAM 3 on a diverse sample with a VLM validator, labels the whole asset, builds a dataset, shows a preview gallery, and offers COCO/YOLO download.

The execution plan is tracked in the audit document supplied by the owner (P0–P7). P0 is the runtime-foundation branch. The next phases are:

- **P1** `feature/label-execution-engine`: background runner, asset→image resolver, dataset builder, export download, job progress. Implemented in this branch; HTTP e2e and crash/cancel recovery are covered by tests.
- **P2** `feature/assets-experience`: single-workspace default, multi-file/chunked uploads, auto-index jobs, asset detail page, in-asset search, sub-asset lineage. Implemented in this branch; covered by a Playwright upload → index → search → sub-asset journey.
- **P3** `feature/agent-chat`: session persistence, LLM task-spec agent, calibrated pipeline, chat UI, dataset download. Implemented in this branch; covered by HTTP and browser end-to-end tests.
- **P4** `feature/manual-refinement-ui`: point-click refinement wired to revisions. Implemented in this branch; a Playwright journey proves mask preview, acceptance, dataset rebuild, and a corrected COCO export.
- **P5** `feature/gpu-resource-manager`: shared model lock, lazy load/unload under memory pressure, measured performance. Implemented in this branch; the real 20-image HTTP journey completed on the 6 GB laptop without OOM. See [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).
- **P6** `feature/ux-e2e-quality`: browser loading/empty/error states, retry actions, contract tests, and the hidden-panel CSS fix. Implemented in this branch; see [`docs/branches/ux-e2e-quality.md`](docs/branches/ux-e2e-quality.md).
- **P7** `feature/real-validation-loop`: real VLM calibration with traceable verdicts and sample overlays, a labeled fallback, cancellable calibration, diversity sampling, dataset-format selection, and the torch/CUDA doctor warning. Implemented in this branch; see [`docs/branches/real-validation-loop.md`](docs/branches/real-validation-loop.md) and the measured run in [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

No phase counts as complete until its feature is reachable from the browser and covered by an HTTP-level test.

## Architecture

```
FastAPI app (application.py)
├── api.py                 assets, uploads, orchestration, label jobs, calibration, SAM3 predict
├── agent.py               plan/confirm/calibrate/label/dataset orchestration
├── agent_api.py           /v1/agent/sessions HTTP surface + calibration overlays
├── agent_store.py         durable SQLite agent sessions + messages
├── agent_worker.py        advances labeling → dataset build → export
├── calibration.py         SAM 3 threshold attempts + cancellation/traces
├── calibration_samples.py diversity-first sample selection from SigLIP vectors
├── calibration_validation.py default VLM/fallback choice + overlay artifacts
├── vlm_validator.py       structured VLM acceptance rubric + verdict evidence
├── ollama_vlm.py          local Ollama vision client (content/thinking JSON)
├── overlay.py             source + SAM 3 masks/boxes → reviewable PNG
├── ollama_text.py         strict JSON task-spec client
├── click_refinement.py    click-point sessions + SAM 3 mask selection
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
├── model_resources.py     shared GPU lock, memory-pressure/idle unload, status
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
| POST | `/v1/sub-assets/from-search` | create an asset from the selected `record_ids` (or, without them, from every result) |
| GET | `/v1/sub-assets` | list sub-assets with parent names and lineage |
| GET | `/v1/jobs` | list background jobs for the tenant |
| POST/GET | `/v1/agent/sessions` | start an agent conversation (one or more `asset_ids`) / list sessions |
| POST | `/v1/agent/sessions/{id}/messages` | answer the agent or change the plan before confirming |
| GET | `/v1/agent/sessions/{id}` | read a durable agent session |
| POST | `/v1/agent/sessions/{id}/confirm` | calibrate (or use a fixed threshold) and start whole-asset labeling |
| POST | `/v1/agent/sessions/{id}/cancel` | cancel the active calibration/labeling/export run |
| GET | `/v1/agent/sessions/{id}/calibration-overlays/{overlay_id}` | authenticated calibration sample overlay PNG |
| POST | `/v1/assets/{id}/predict` | prompted SAM 3 on one source |
| POST | `/v1/predict` | multipart SAM 3 smoke endpoint |
| POST | `/v1/orchestrate` | deterministic message → `TaskSpec` (regex fallback; not the agent) |
| POST/GET | `/v1/label-jobs` | submit/read durable label jobs; executed by the label worker |
| POST | `/v1/label-jobs/{id}/cancel` | request cancellation of a queued/running label job |
| POST/GET | `/v1/datasets` | build/list dataset versions from succeeded label jobs |
| GET | `/v1/datasets/{id}` | read dataset-version summary |
| GET | `/v1/datasets/{id}/images` | list dataset images and their annotations |
| GET | `/v1/datasets/{id}/images/{image_id}/file` | fetch the image bytes for review |
| POST | `/v1/datasets/{id}/rebuild` | rebuild a version from its latest annotation revisions |
| DELETE | `/v1/datasets/{id}/images/{image_id}/annotations/{annotation_id}` | retire an annotation (applied by the next rebuild) |
| WS | `/v1/refinement/{version_id}/{image_id}` | click-point mask preview (tenant query parameter) |
| POST | `/v1/refinements/{session_id}/accept` | save the previewed mask as an annotation revision |
| POST | `/v1/calibration/run` | threshold calibration on uploaded images |
| POST/GET | `/v1/dataset-exports` | submit/read dataset-export jobs |
| GET | `/v1/dataset-exports/{id}/download` | download the packaged COCO/YOLO zip |
| GET | `/v1/preview/...` | replay label preview events |

## Testing

```powershell
.venv\Scripts\python -m pytest -q
```

Current local result: **275 passed, 10 skipped, 1 warning** (Starlette/httpx deprecation). The 10 skips are the opt-in PostgreSQL tests. The P1 tests include an HTTP-only fake-SAM journey through upload, labeling, dataset creation, COCO/YOLO export, zip download, cancellation, and expired-lease recovery. The P3 tests add a chat → confirm → calibration → label → dataset → browser-download journey, plus calibration-timeout, cancellation, fixed-threshold fallback, planner fallback, Ollama thinking-field compatibility, and durable-session coverage. The P4 tests add an HTTP refinement → revision → rebuild → corrected COCO export check and a browser click → mask preview → accept → rebuilt export journey. The P5 tests add shared-lock serialization, memory-pressure/idle unload, CPU no-torch status, provider reload, and SAM 3 box clipping. The P6 tests add skeleton/loading/empty/error/retry browser journeys and UI/API contract checks. The P7 tests add VLM selection/fallback/normalization, authenticated overlay serving, calibration cancellation and timeout recovery, diversity sampling, `get_embedding` across vector stores, the doctor CPU-torch warning, and the dataset format selector.

The browser tests drive a real uvicorn server with fake providers. They skip if the Playwright browser binary is not installed:

```powershell
.venv\Scripts\python -m playwright install chromium
.venv\Scripts\python -m pytest tests/test_assets_ui_playwright.py tests/test_agent_ui_playwright.py tests/test_manual_refinement_ui_playwright.py -q
```

The P4 browser journey runs the Agent, opens **Review labels**, checks that the mask overlay draws, accepts the correction, rebuilds the dataset, downloads the corrected COCO archive, verifies the new bounding box with `pycocotools`, and asserts that no UUID appears in visible text.

The P5 unit/HTTP tests cover shared-model serialization, memory-pressure unload, idle unload, CPU/no-torch behavior, provider reload after unload, and clipping of out-of-bounds SAM 3 boxes. The real measurement is manual: [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) records the commands and observed numbers. On the RTX 4050 Laptop (6 GB), a 20-image real-image HTTP journey took 39.33 s to index, 40.29 s to calibrate/label/build/export, peaked at 4202.8 MB allocated / 4390.0 MB reserved, and completed without OOM.

The P6 browser tests hold library requests to assert the gallery skeletons and Agent loading option, release them to assert every empty state, and fail library requests to assert error panels and working retries. Contract tests check every route/method the UI calls, the empty response envelopes, the `detail` error field, and that each `app.js` element id exists in `index.html`.

The P7 real run used the same 20 real cached video frames with SigLIP2, SAM 3, and `qwen3-vl:4b` together. Calibration selected five diverse samples, tried threshold 0.30 once, and recorded five accepted VLM verdicts and five overlay ids. The run produced 34 annotations, downloaded a 3,602,141-byte COCO archive, then refined/rebuild/exported successfully. PyTorch peaked at 4,202.8 MB allocated / 4,390.0 MB reserved; the whole-GPU `nvidia-smi` counter peaked at 5,822 MB of 6,141 MB while Ollama was resident. Calibration/label/export took 383.21 s, versus 40.29 s on the P5 no-VLM path. Exact output is in [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md); these are local observations, not production benchmarks.

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
| `AGENTIC_CV_MODEL_HEADROOM_MB` | `512` | Extra free VRAM required before loading a model |
| `AGENTIC_CV_MODEL_IDLE_UNLOAD_SEC` | `300` | Idle seconds before a model is evicted |
| `AGENTIC_CV_EMBEDDING_VRAM_MB` | `2304` | SigLIP2 footprint used by the unload policy |
| `AGENTIC_CV_SAM3_VRAM_MB` | `2048` | SAM 3 footprint used by the unload policy |
| `AGENTIC_CV_POSTGRES_DSN` | unset | opt-in PostgreSQL + pgvector |
| `AGENTIC_CV_VIDEO_SAMPLE_FPS` | `1.0` | video frames sampled per second (search and labeling share them) |
| `AGENTIC_CV_VIDEO_MAX_FRAMES` | `1000` | per-video frame cap; `0` = unlimited (a warning is logged when reached) |
| `AGENTIC_CV_AGENT_PLANNER` | `ollama` | set `rules` to skip the text model entirely |
| `AGENTIC_CV_AGENT_PROMPTS` | `ollama` | set `rules` to keep the LLM planner but skip SAM 3 prompt rewriting |
| `AGENTIC_CV_TEXT_MODEL` | `qwen3-vl:4b` | Ollama model used for task-spec JSON |
| `AGENTIC_CV_OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `AGENTIC_CV_OLLAMA_TIMEOUT_SEC` | `30` | task-spec request timeout |
| `AGENTIC_CV_CALIBRATION_TIMEOUT_SEC` | `120` | agent calibration budget; a timeout returns HTTP 504 |
| `AGENTIC_CV_AGENT_VALIDATOR` | unset (VLM when reachable) | set `any-prediction` (or `rules`) to force the labeled fallback |
| `AGENTIC_CV_VLM_MODEL` | `qwen3-vl:4b` (or `AGENTIC_CV_TEXT_MODEL`) | Ollama vision model used for calibration |
| `AGENTIC_CV_VLM_TIMEOUT_SEC` | `100` | per-VLM-request timeout; raise for first-load/warm-up on a 6 GB GPU |
| `AGENTIC_CV_VLM_MAX_CALLS` | `50` | hard cap on VLM calls per calibration run |
| `AGENTIC_CV_VLM_MIN_CONFIDENCE` | `0.7` | minimum VLM confidence for an accepted verdict |
| `AGENTIC_CV_VLM_MAX_SIDE` | `768` | longest side of images sent to the VLM; `0` sends full size |
| `AGENTIC_CV_VLM_PROBE_TIMEOUT_SEC` | `1.0` | Ollama `/api/tags` reachability probe timeout |
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
9. In the calibration card, confirm the validator says **Vision (VLM)** with `qwen3-vl:4b`, and inspect the thresholds tried, sample overlays, and accepted/rejected verdicts. If it says **at least one prediction**, read the notice and check Ollama before treating the run as VLM-validated.
10. Watch whole-asset labeling, dataset build, and packaging progress; download the COCO archive from the run panel.
11. Open **Datasets**, choose **YOLO detection** or **YOLO segmentation** in the version's format selector, click **Export**, and verify the downloaded archive. Then click **Review labels**, select an image, click a positive point (and a negative point if needed), and accept the previewed mask.
12. The app writes a revision, rebuilds a refined dataset version, and reports that it is ready; export the refined version and verify the corrected annotation.
13. Open **Jobs** to inspect the label/export jobs.
14. Reload the page and reopen **Agent** to confirm the finished run and its calibration verdicts/overlays are restored from the local session.

The owner's section-7 demo calls for both COCO and YOLO downloads. The Agent run panel downloads COCO; the Datasets card now offers COCO, YOLO detection, and YOLO segmentation for the same version.

## Limitations / safety

- The browser covers upload, indexing, search, sub-assets, Agent labeling, dataset listing, downloads, and per-image review. Manual correction is a click-guided selection of SAM 3 text-prompt masks for the dataset category, not a point-conditioned segmentation model; a click that no candidate mask contains cannot create a new object out of nothing.
- Accepting a correction updates one existing annotation (or adds one when the image has none) and rebuilds a new immutable dataset version. It does not edit the old version in place.
- The Agent's default calibration validator is the local Ollama VLM. If Ollama or `qwen3-vl:4b` is unavailable at confirmation, calibration uses the labeled `at least one prediction` fallback and shows the reason. If a VLM request fails mid-run, the remaining samples use the same labeled fallback; the calibration is not silently presented as VLM-validated. Set `AGENTIC_CV_AGENT_VALIDATOR=any-prediction` to force the fallback.
- VLM validation sends a rendered sample overlay, not the original full-resolution image. Very small objects can be harder to judge, and the validator's confidence/reasoning is evidence for a human, not a guarantee of annotation quality.
- VLM calibration is materially slower than the fallback and shares the 6 GB GPU with SAM 3. The measured 20-image VLM journey spent 383.21 s in calibrate/label/export versus 40.29 s on the no-VLM path.
- The VLM runs as an external Ollama process. The app's shared model manager controls SigLIP2 and SAM 3, not Ollama; on Windows the driver can back allocations with shared system memory, so the 6 GB figure is not a hard partition.
- The Agent run panel downloads COCO. Dataset cards offer COCO, YOLO detection, and YOLO segmentation through `/v1/dataset-exports`; the review flow rebuilds a version that can be exported in any of those formats.
- `agentic-cv serve` executes label, index, export, and agent-advance jobs. The separate `agentic-cv worker` command still runs only the export worker.
- Label jobs cancelled mid-run can leave partial annotation revisions; dataset creation requires a succeeded job, so partial runs are not packaged.
- The Agent and Assets browser tests use fake SAM 3/embedding providers and a deterministic planner. Real-model evidence is a separate HTTP journey, not a full browser journey: the P7 run used 20 real cached video frames, real SigLIP2/SAM 3/`qwen3-vl:4b`, real workers, click refinement, and export, and peaked at 5.82 GB whole-GPU / 4.20 GB torch-allocated on the 6 GB laptop. It does not establish label quality on your production photos.
- P5 serializes model inference and unloads under pressure, but it does not reserve VRAM in the driver. Another GPU-heavy process can still cause an OOM, and inference remains one image at a time rather than batched.
- P6 tests loading with held browser requests and errors with failed browser responses. They do not simulate a real slow disk, a crashing backend process, or every payload shape; the contract tests are intentionally shallow.
- The P3 planner asks Ollama for strict JSON. If the model is down or returns invalid JSON, the chat visibly falls back to deterministic rules.
- Sub-assets copy the matched search images into a new asset; they are snapshots, not live filters over the parent asset.
- SAM 3 is subject to Meta's SAM License and gated weights; this repository is not legal advice or a hosted-product clearance. See `docs/branches/sam-3-licensing-and-hosted-product-review-roadmap-item-10-complete.md`.
- Do not commit `.agentic-cv/`, model weights, tokens, or local exports.

## Branch documentation

- Current P7 branch: [`docs/branches/real-validation-loop.md`](docs/branches/real-validation-loop.md)
- P6 branch record: [`docs/branches/ux-e2e-quality.md`](docs/branches/ux-e2e-quality.md)
- P5 branch record: [`docs/branches/gpu-resource-manager.md`](docs/branches/gpu-resource-manager.md)
- Measurements: [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md)
- P4 branch record: [`docs/branches/manual-refinement-ui.md`](docs/branches/manual-refinement-ui.md)
- P3 branch record: [`docs/branches/agent-chat.md`](docs/branches/agent-chat.md)
- P2 branch record: [`docs/branches/assets-experience.md`](docs/branches/assets-experience.md)
- P1 branch record: [`docs/branches/label-execution-engine.md`](docs/branches/label-execution-engine.md)
- P0 branch record: [`docs/branches/fix-runtime-foundation.md`](docs/branches/fix-runtime-foundation.md)
- Historical branch READMEs: [`docs/branches/README.md`](docs/branches/README.md)
- Repository hygiene proposal: [`docs/REPO_HYGIENE.md`](docs/REPO_HYGIENE.md)

## License

No license file is currently present in the repository. SAM 3 and model weights have separate upstream terms; review before distribution or hosted use.
