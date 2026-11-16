# Agentic CV Platform: SAM 3 Inference Service

This branch exposes the local SAM 3 provider through a small, bounded FastAPI service. It is the first process boundary around model inference and is designed to run without loading gated weights during ordinary tests.

## What this branch provides

- `GET /health/live` for process liveness.
- `GET /health/ready` for service readiness.
- `POST /v1/predict` for an image, text prompt, and optional thresholds.
- Upload-size limits and invalid-image rejection before model execution.
- Typed JSON responses containing scores, boxes, and boolean masks.
- Provider injection for deterministic API tests and future hosted implementations.

## Local development

Run the complete test suite:

```text
uv run pytest -q
```

The service tests use a fake provider and do not download SAM 3 weights. A real deployment must run with the SAM 3 environment, gated checkpoint access, CUDA, and a process supervisor. The endpoint currently handles one request at a time through the injected provider; queueing, GPU bulkheads, and batch workers are later slices.

## Branch boundary

This branch depends on `feature/sam3-inference-adapter`. It does not yet include authentication, object-storage URLs, workflow persistence, dynamic batching, or separate interactive and batch queues.

The older `frames-extractor` repository is reference material only and is not part of this Git history.
