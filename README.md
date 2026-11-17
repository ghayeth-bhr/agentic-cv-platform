# Agentic CV Platform: VLM Validation Adapter

This branch adds structured VLM validation for SAM 3 calibration. It validates the response schema, applies an explicit acceptance rubric, caps calls per run, and caches judgments by overlay, prompt, and model revision.

## What this branch provides

- Typed target-presence, completeness, false-positive, and mask-tightness judgments.
- Confidence and reasoning validation.
- Explicit acceptance criteria for calibration.
- Maximum VLM call budgets with a dedicated exception.
- Deterministic cache keys including model revision.
- Client/provider injection for local Ollama or hosted VLM adapters.

## Local development

```text
uv run pytest -q
```

The validator tests use a fake VLM client. No multimodal model, GPU, network call, or cloud service is required.

## Branch boundary

This branch builds on `feature/persistent-vector-index`. It validates VLM judgments but does not yet implement a real Ollama/hosted client, overlay artifact generation, prompt enhancement, or direct integration into the calibration sample validator.

The older `frames-extractor` repository is reference material only and is not part of this Git history.
