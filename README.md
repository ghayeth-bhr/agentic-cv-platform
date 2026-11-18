# Agentic CV Platform: Deterministic Outbox Ordering

This branch fixes a nondeterministic delivery order in the durable local outbox. It is a hygiene slice, not a feature: no roadmap item is reordered.

## The defect

`Outbox.pending()` ordered work with `ORDER BY occurred_at, event_id`. Events enqueued in the same clock tick share an identical `occurred_at`, so the tie was broken by `event_id` — a random UUID. Delivery order (and `tests/test_outbox_dispatcher.py::test_dispatcher_records_failure_and_continues`, which asserts insertion order) could therefore flip between runs.

## The fix

- An explicit monotonic `sequence` column, assigned inside the existing write lock as `MAX(sequence) + 1`.
- Pending reads now use `ORDER BY occurred_at, sequence`, so equal timestamps preserve insertion order.
- `OutboxRecord.sequence` is exposed for inspection; existing fields, payloads, and public methods are unchanged.
- Tables created before this branch are migrated on open: the column is added and backfilled from SQLite `rowid`, which is insertion order for this append-only table.

## Validation

- `tests/test_outbox.py::test_pending_uses_insertion_order_when_timestamps_tie` — three events with one shared timestamp and deliberately descending UUIDs. Verified to fail against the previous implementation (`[1, 2, 3]` returned as `[3, 2, 1]`) and pass after the fix.
- `tests/test_outbox.py::test_legacy_outbox_without_sequence_column_is_migrated` — old-schema database opens, backfills, and keeps the legacy event ahead of a newly enqueued one.
- Focused suites (`test_outbox.py`, `test_outbox_dispatcher.py`) green; the previously flaky dispatcher test passed 10 consecutive runs.
- Full suite: **128 passed, 1 warning** (Starlette/httpx deprecation).

## Limitations

- `sequence` is per outbox database. It gives a total order for this single-writer, single-file local design, not across shards or Postgres-backed implementations.
- Migration backfills from `rowid`; if rows were ever deleted from the table, historical ordering follows the surviving rowids.

## Next boundary

Roadmap item 6 — PostgreSQL/pgvector behind the existing store and vector-index boundaries, starting with schema and migrations. That work ships on its own stacked branch.

---

# Agentic CV Platform: VLM Validation Adapter

This branch adds structured VLM validation for SAM 3 calibration. It validates the response schema, applies an explicit acceptance rubric, caps calls per run, and caches judgments by overlay, prompt, and model revision.

## What this branch provides

- Typed target-presence, completeness, false-positive, and mask-tightness judgments.
- Confidence and reasoning validation.
- Explicit acceptance criteria for calibration.
- Maximum VLM call budgets with a dedicated exception.
- Deterministic cache keys including model revision.
- Client/provider injection with a local Ollama implementation.

## Local Ollama

The adapter targets Ollama at `http://localhost:11434` and defaults to `qwen3-vl:4b`.
Start Ollama and pull the model before wiring it into calibration:

```text
ollama pull qwen3-vl:4b
```

`render_prediction_overlay` writes the source image, SAM3 masks, and score-labeled
boxes to a local PNG. `OllamaVlmClient` sends that artifact as base64 to `/api/chat`
and requests the structured validation schema. Its timeout, context size,
temperature, and keep-alive are configurable; tests use a mocked HTTP boundary
and do not require Ollama or GPU access.

`OllamaPromptEnhancer` uses the same local model to produce one to three concise
SAM3 noun phrases. On Ollama/Qwen responses where structured JSON is returned in
`message.thinking`, the adapter accepts that field when `message.content` is empty.

## Interactive refinement

When configured with an image loader and `ClickRefinementService`, the API exposes
`/v1/refinement/{image_id}` as a WebSocket. It sends an initial snapshot, accepts
revisioned `click` messages with positive or negative points, and returns updated
SAM3 predictions and masks.

`JobStore` persists worker lease ownership, heartbeat expiry, and cancellation
intent. Claiming and heartbeat operations reject stale or competing workers;
`DurableLabelWorkflow` claims a lease, renews it at image checkpoints, observes
cancellation, and emits a dedicated cancellation event.

API authentication is opt-in through `StaticApiKeyAuthorizer`. When configured,
asset and upload write operations require `X-API-Key`, bind the key to
`X-Tenant-ID`, and enforce the `writer` role. Semantic-search and preview reads
use the `reader` role when their optional authorizer is configured. The default
local APIs remain unauthenticated until an authorizer is supplied.

Production OIDC/JWT validation is available through `OidcJwtAuthorizer`. The
`serve` command enables it when `AGENTIC_CV_OIDC_ISSUER`,
`AGENTIC_CV_OIDC_AUDIENCE`, and `AGENTIC_CV_OIDC_JWKS_URL` are set. Tokens must
include `exp`, `iat`, `sub`, a UUID tenant claim, and the configured roles claim.

`RolePolicy` centralizes resource permissions: readers can search, inspect
previews, and read exports; writers can mutate assets, uploads, sub-assets, and
exports; admins receive the wildcard permission. API routes enforce these
resource-action permissions rather than trusting a coarse role label.

Cloud-compatible uploads use `S3ObjectStorage`. Set `AGENTIC_CV_S3_BUCKET` to
select it, optionally set `AGENTIC_CV_S3_ENDPOINT_URL` for MinIO and `AWS_REGION`
for AWS. When the bucket variable is absent, the local filesystem adapter remains
the default.

`OutboxDispatcher` delivers pending integration events through an injected
publisher, marks successful deliveries, and records retryable failures without
blocking later events in the same polling pass.

`RabbitMqPublisher` is the concrete broker publisher. It declares a durable topic
exchange and publishes persistent JSON event envelopes; set
`AGENTIC_CV_RABBITMQ_HOST` and the optional RabbitMQ connection variables to
activate it in the composition root. `RabbitMqConsumer` declares durable queues,
binds routing keys, limits in-flight messages with QoS, acknowledges successful
handlers, and requeues failures.

`split_dataset` creates deterministic, image-disjoint train, validation, and test
`DatasetVersion` objects from configurable ratios and a seed. Annotations remain
with their source images, so each partition can be passed directly to the existing
COCO or YOLO exporters.

`DatasetVersionStore` persists validated dataset snapshots in tenant-scoped SQLite
storage with reopen-safe reads, ordered listing, and idempotent creation.

`DatasetExportService` materializes persisted versions into tenant/version-scoped
COCO or YOLO directories and writes a manifest for each export.

`DatasetExportWorkflow` wraps those exports in durable leased jobs with idempotent
submission, cancellation handling, and completion/failure events.

`DatasetExportWorker.run_once()` polls queued export jobs and processes each through
the lease-aware workflow; one failed job does not prevent the remaining queue from
being inspected. `run_forever()` adds a stoppable polling loop for local worker
processes, and each polling pass recovers expired dataset-export leases before
claiming queued work.

When configured, the API exposes authenticated `POST /v1/dataset-exports` for job
submission, `GET /v1/dataset-exports/{job_id}` for tenant-scoped status reads, and
`POST /v1/dataset-exports/{job_id}/cancel` for cancellation requests.

## Local development

```text
uv run pytest -q
```

The validator tests use a fake VLM client. No multimodal model, GPU, network call, or cloud service is required.

## Run the assembled local MVP

Start the unified API with:

```text
uv run agentic-cv serve --data-dir .agentic-cv
```

It assembles the core asset/upload API, semantic-search routes, preview SSE,
dataset-export jobs, and lazy SAM3 service under one FastAPI process. The health
checks are available at `/health/live` and `/health/ready`; model weights are
loaded only when their inference or embedding endpoints are used.

The root URL serves the local browser dashboard. It provides asset upload, SAM3
prediction, semantic search, and export-job controls. JSON runtime discovery is
available at `/api-info`, and interactive API documentation remains at `/docs`.

For a containerized local stack, use:

```text
docker compose -f docker-compose.local.yml up --build
```

This starts the API, export worker, RabbitMQ, and MinIO services with persistent
local volumes. The host-based `agentic-cv serve` and `agentic-cv worker` commands
remain the lighter development path.

## Branch boundary

This branch builds on `feature/persistent-vector-index`. It provides a local
Ollama client, overlay renderer, prompt enhancer, calibration bridge, and optional
click-refinement WebSocket, plus opt-in tenant authorization, but does not yet
implement hosted deployment configuration, health-based orchestration, or worker
autoscaling.

The older `frames-extractor` repository is reference material only and is not part of this Git history.
