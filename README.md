# Agentic CV Platform: SAM 3 Adapter

This branch adds the provider-neutral SAM 3 inference adapter to the fresh platform implementation. It wraps the tested Transformers API while keeping model imports lazy, so API code and CI can run without SAM 3 weights or a CUDA runtime.

## What this branch provides

- `Sam3Config` for checkpoint, device, precision, and threshold settings.
- `Sam3Provider` as the inference boundary that future hosted providers can implement.
- `LocalSam3Provider` for the local `facebook/sam3` checkpoint.
- Normalized prediction results containing confidence, box coordinates, and masks.
- RGB image loading and explicit provider cleanup for GPU memory release.

## Local development

Run the complete test suite:

```text
uv run pytest -q
```

The adapter tests inject a fake model and do not download weights. To run real inference, use the existing SAM 3 environment, ensure the gated checkpoint is available, and configure a CUDA device. The provider defaults to FP16 and one image per call; batching and request scheduling belong to later service work.

## Branch boundary

This branch depends on `feature/local-resumable-upload` and contains no HTTP service, workflow engine, or batch scheduler yet. The next slice should expose this provider through a bounded inference API with liveness/readiness checks and separate interactive and batch paths.

The older `frames-extractor` repository is reference material only and is not part of this Git history.
