# feature/label-execution-engine (P1)

## Scope

Turn the existing durable label-job records into a real local execution path:

1. Run queued label jobs in the app/worker without a GPU in the default test suite.
2. Resolve an asset's image/video sources into deterministic `LabelImage` records.
3. Build an immutable dataset version from the latest annotation revisions.
4. Package COCO/YOLO output with images and expose a zip download.
5. Cover success, cancellation, and expired-lease crash recovery through the real HTTP API.

## What changed

- `label_images.py`: `LabelImageResolver` maps an asset's local image/video sources to stable image IDs and unique export filenames; video sources use the existing frame extractor (`stride=30`, `max_frames=120`).
- `label_worker.py`: `LabelWorker` polls `batch-label` jobs, requeues expired leases, and delegates to `DurableLabelWorkflow`.
- `label_workflow.py`: `execute` can resolve images after claiming the job, so resolver failures are persisted as failed jobs; annotation saves use `expected_revision=None` so a retried job is not blocked by revisions written before a crash.
- `annotation_store.py` / `postgres_annotation_store.py`: `list_latest` returns the current revision per annotation; `save` accepts `expected_revision=None` for recovery-safe appends.
- `dataset_builder.py`: `DatasetBuilder` requires a succeeded label job, joins the latest revisions for the resolved images, creates a `DatasetVersion`, and copies the source images into a per-version image directory.
- `dataset_export.py`: exports now copy images when present, write YOLO `data.yaml`, and produce a deterministic zip archive; `archive_path` is recorded on the result.
- `batch_labeling.py`: `LabelImage.file_name` allows unique filenames for multi-source assets while preserving the old `path.name` default.
- `api.py`: added `POST/GET /v1/datasets`, `GET /v1/datasets/{id}`, and `GET /v1/dataset-exports/{id}/download`.
- `application.py`: constructs the resolver, label worker, dataset builder, and export worker; `agentic-cv serve` starts both workers in the app lifespan and stops them before closing stores. `LocalAppConfig.start_workers` defaults to `True`; `worker_poll_interval` defaults to `0.25`.
- `pyproject.toml` / `uv.lock`: added `pycocotools` to the dev group for COCO validation.

## Validation (real output)

```text
.venv/Scripts/python -m pytest -q
214 passed, 10 skipped, 1 warning in 10.33s
```

The 10 skips are the existing opt-in PostgreSQL tests.

The new HTTP test (`tests/test_label_execution_e2e.py`) uses a fake SAM provider and no network/GPU:

- create asset → upload PNG → `POST /v1/label-jobs`
- worker executes the job → poll to `succeeded`
- `POST /v1/datasets` → dataset summary has 1 image/1 annotation
- `POST /v1/dataset-exports` (`coco`) → poll → download zip → `pycocotools.COCO` loads it and the referenced image exists in the archive
- `POST /v1/dataset-exports` (`yolo-detection`) → poll → download zip → `data.yaml`, `images/`, and a 5-field normalized label file are present

Additional coverage:

- `tests/test_label_worker.py`: queued execution, expired-lease requeue, pre-cancelled job, unresolvable asset → failed.
- `tests/test_label_execution_e2e.py::test_http_cancel_stops_running_label_job`: a blocking fake provider keeps the job running while `POST /v1/label-jobs/{id}/cancel` is sent; the job finishes as `cancelled`.
- `tests/test_label_execution_e2e.py::test_http_restart_recovers_expired_label_lease`: first app claims the job with a short lease and closes; a second app on the same data dir recovers the expired job and reports `succeeded` over HTTP.
- `tests/test_dataset_builder.py`: latest manual revision wins, images are copied, and the COCO zip contains annotations, images, and manifest.
- `tests/test_annotation_store.py`: latest-per-annotation selection and recovery-safe append behavior.

## Limitations

- The P1 label flow is HTTP-only. P2 added the Assets upload/index/search experience; P3 will expose labeling through the Agent chat.
- `agentic-cv serve` runs both workers. The separate `agentic-cv worker` command still runs only the export worker.
- Cancellation after some images have been labeled leaves those annotation revisions in place. Dataset creation is gated on `succeeded`, so a cancelled run is not packaged.
- The fake-SAM HTTP journey proves orchestration and persistence, not SAM 3 quality. Real-model validation remains a P3/owner-machine gate.
- Dataset images are copied at build time. A dataset version is immutable, but rebuilding after source changes requires a new version/key.
- `pycocotools` is a dev-only dependency; runtime exports use only the standard library and existing project dependencies.

## Next boundary

P2 `feature/assets-experience`: make the workspace usable without UUIDs, with multi-file upload, asset detail, in-asset search, and sub-asset lineage. P3 then wires the label/dataset/download path into the Agent chat.
