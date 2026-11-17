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

## Local development

```text
uv run pytest -q
```

The validator tests use a fake VLM client. No multimodal model, GPU, network call, or cloud service is required.

## Branch boundary

This branch builds on `feature/persistent-vector-index`. It provides a local
Ollama client, overlay renderer, and calibration bridge but does not yet implement
prompt enhancement or a production worker integration.

The older `frames-extractor` repository is reference material only and is not part of this Git history.
