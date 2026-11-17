# Agentic CV Platform: SAM 3 Calibration Loop

This branch adds bounded threshold calibration for SAM 3 over a representative sample set. It prevents a single lucky image from authorizing a full asset run and returns an explicit result when no threshold meets the required agreement.

## What this branch provides

- Multiple `CalibrationSample` inputs instead of single-image validation.
- Configurable threshold candidates and maximum iteration count.
- Injected validators that can later call a VLM rubric.
- Per-attempt sample counts and agreement scores.
- `calibrated`, `max_iterations`, and `no_acceptable_threshold` outcomes.
- No hidden success: failed calibration returns no production threshold.

## Local development

```text
uv run pytest -q
```

The calibration tests use a fake SAM 3 provider and CPU images. No gated model weights, GPU, VLM, broker, or cloud service is required.

## Branch boundary

This branch builds on `feature/local-asset-upload-api`. It does not yet persist calibration runs, render overlay artifacts, call a real VLM, adjust prompts automatically, or launch a whole-asset batch workflow.

The older `frames-extractor` repository is reference material only and is not part of this Git history.
