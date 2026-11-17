# Agentic CV Platform: Hugging Face Embeddings

This branch adds the real lazy Hugging Face image/text embedding provider behind the semantic-search interface. It is compatible with SigLIP-style checkpoints and keeps heavy ML imports out of the base test process.

## What this branch provides

- Configurable checkpoint, device, and FP16/BF16 precision.
- Lazy `AutoProcessor` and `AutoModel` loading.
- Image and text feature extraction with one vector per request.
- Query padding with `max_length` for stable text embeddings.
- Explicit CUDA checks and model cleanup for local GPU operation.
- Injectable loader tests without Torch, Transformers, or model downloads.

## Local development

```text
uv run pytest -q
```

The provider tests use fake Torch/model objects. Real semantic search requires installing the compatible Torch and Transformers versions in the local SAM/ML environment and granting access to the selected checkpoint.

## Branch boundary

This branch builds on `feature/semantic-search-index`. It provides model integration but does not yet persist vectors, batch embeddings, expose semantic search over HTTP, or create sub-assets from search results.

The older `frames-extractor` repository is reference material only and is not part of this Git history.
