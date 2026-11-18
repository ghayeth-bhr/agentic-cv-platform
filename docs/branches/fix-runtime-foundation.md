# fix/runtime-foundation (P0)

## Scope

Make the local project runnable and honest before adding product features:

1. Optional `ml` dependency extra with the validated torch/torchvision/transformers/accelerate stack.
2. `agentic-cv doctor` with actionable checks (Python, device, CUDA, packages, HF access, Ollama, disk, port).
3. `ModelUnavailableError` → HTTP `503 model_unavailable` instead of a 500 traceback.
4. `AGENTIC_CV_DEVICE` / `AGENTIC_CV_PRECISION` with a documented CPU fallback.
5. Root README rewrite; historical branch READMEs moved to `docs/branches/`.
6. UI honesty: “Quick test” moved out of a fake “Agent” group until the agent chat exists.

## What changed

- `pyproject.toml`: added `ml` extra (`torch>=2.6,<2.7`, `torchvision>=0.21,<0.22`, `transformers>=5.14,<5.15`, `accelerate>=1,<2`); `uv.lock` updated.
- `model_runtime.py`: device/precision resolution (`cpu`, `cuda`, `cuda:<index>`), CPU forces `float32`.
- `errors.py`: typed `ModelUnavailableError` and a shared FastAPI 503 handler.
- `embeddings.py`, `sam3.py`: lazy imports wrapped in `ModelUnavailableError`; CUDA-unavailable and model-load failures are actionable; `float32` accepted for CPU.
- `application.py`: model settings from the environment; `/api-info` exposes device/precision; handler installed on the app.
- `api.py`, `semantic_api.py`, `sam3_service.py`: install the 503 handler and do not swallow `ModelUnavailableError` into a 500.
- `doctor.py` + `cli.py`: `agentic-cv doctor [--no-network] [--json]`.
- `web/index.html`, `web/app.js`: navigation now groups **Assets** (Gallery, Upload, Search library), **Tools** (Quick test), and **Jobs & Exports**; the Quick test copy states it is not the agent.
- `README.md`: rewritten for the current product; 23 historical branch READMEs preserved in `docs/branches/`.
- `docs/REPO_HYGIENE.md`: proposal only; no merge, prune, or default-branch change was executed.

## Validation (real output)

```text
.venv/Scripts/python -m pytest -q
202 passed, 10 skipped, 1 warning in 7.43s
```

`agentic-cv doctor --no-network --port 8123` in the owner's ML environment:

```text
[ok] Python: Python 3.12.13
[ok] Model runtime: device=cuda, precision=float16
[ok] PyTorch: torch 2.6.0+cu124; CUDA: NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GiB)
[ok] torchvision: torchvision 0.21.0+cu124
[ok] transformers: transformers 5.14.1
[ok] accelerate: accelerate 1.14.0
[ok] HF access: facebook/sam3 cached ...
[ok] HF access: google/siglip2-so400m-patch14-384 cached ...
[fail] Ollama: Ollama is not reachable at http://localhost:11434 ...
[ok] Disk: 59.2 GiB free ...
[ok] Port: 127.0.0.1:8123 is free
```

Ollama was not running during the check; that is an environment issue, not a code failure. Start Ollama and rerun `agentic-cv doctor`.

`uv sync --extra ml --dry-run` resolved the validated stack and reported the packages it would install; the actual fresh-clone sync remains an owner-machine gate.

## Limitations

- The P0 gate’s **fresh clone → `uv sync --extra ml` → doctor all green on the owner’s laptop** has not been executed end-to-end by the agent. The lockfile and dry-run are verified; the owner must run the real fresh-clone check.
- `uv sync --extra ml` selects PyPI wheels. Windows CUDA users must install the `+cu124` wheels explicitly (documented in the root README).
- Ollama is still required for the future agent/VLM phases; P0 only diagnoses it.
- No agent chat, label execution, dataset builder, export download, or refinement UI exists yet.
- Repository hygiene (merge to `main`, default branch, branch pruning) is **proposed and waiting for owner approval**.

## Next boundary

P1 `feature/label-execution-engine`: run queued label jobs in the app/worker, resolve assets to `LabelImage`s, build a dataset version from annotation revisions, and add export download.
