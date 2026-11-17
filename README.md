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
`X-Tenant-ID`, and enforce the `writer` role. The default local API remains
unauthenticated until an authorizer is supplied.

`split_dataset` creates deterministic, image-disjoint train, validation, and test
`DatasetVersion` objects from configurable ratios and a seed. Annotations remain
with their source images, so each partition can be passed directly to the existing
COCO or YOLO exporters.

`DatasetVersionStore` persists validated dataset snapshots in tenant-scoped SQLite
storage with reopen-safe reads, ordered listing, and idempotent creation.

`DatasetExportService` materializes persisted versions into tenant/version-scoped
COCO or YOLO directories and writes a manifest for each export.

## Local development

```text
uv run pytest -q
```

The validator tests use a fake VLM client. No multimodal model, GPU, network call, or cloud service is required.

## Branch boundary

This branch builds on `feature/persistent-vector-index`. It provides a local
Ollama client, overlay renderer, prompt enhancer, calibration bridge, and optional
click-refinement WebSocket, plus opt-in tenant authorization, but does not yet
implement production identity-provider integration, full RBAC policy management,
cloud object storage, or asynchronous export job orchestration.

The older `frames-extractor` repository is reference material only and is not part of this Git history.
