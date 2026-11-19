"""Lazy Hugging Face image/text embeddings for semantic search."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from PIL import Image

from .errors import ModelUnavailableError
from .model_resources import ModelResourceManager


@dataclass(frozen=True, kw_only=True)
class HuggingFaceEmbeddingConfig:
    checkpoint_id: str = "google/siglip2-so400m-patch14-384"
    device: str = "cuda"
    precision: str = "float16"

    def __post_init__(self) -> None:
        if self.precision not in ("float16", "bfloat16", "float32"):
            raise ValueError("precision must be float16, bfloat16, or float32")


class HuggingFaceEmbeddingProvider:
    """Lazy provider compatible with SigLIP-style Transformers checkpoints."""

    def __init__(
        self,
        config: HuggingFaceEmbeddingConfig | None = None,
        *,
        loader: Callable[[HuggingFaceEmbeddingConfig], tuple[Any, Any, Any]] | None = None,
        resources: ModelResourceManager | None = None,
        resource_name: str = "siglip2",
        estimated_vram_mb: int = 2304,
    ) -> None:
        self.config = config or HuggingFaceEmbeddingConfig()
        self._loader = loader
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self._resources = resources
        self._resource_name = resource_name
        self._local_lock = threading.RLock()
        self._load_seconds = 0.0
        if resources is not None:
            resources.register(
                resource_name, self, estimated_mb=estimated_vram_mb
            )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def load_seconds(self) -> float:
        return self._load_seconds

    def _guard(self):
        if self._resources is not None:
            return self._resources.lease(self._resource_name)
        return self._local_lock

    def _load(self) -> None:
        if self._model is not None:
            return
        started = time.perf_counter()
        try:
            self._load_model()
        finally:
            self._load_seconds += time.perf_counter() - started

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if self._loader is not None:
            self._model, self._processor, self._torch = self._loader(self.config)
            return

        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise ModelUnavailableError(_install_hint()) from exc

        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise ModelUnavailableError(
                "The embedding provider is configured for CUDA but CUDA is not available. "
                "Set AGENTIC_CV_DEVICE=cpu for CPU inference, or install CUDA-enabled PyTorch. "
                "Run `agentic-cv doctor` for details."
            )
        try:
            dtype = getattr(torch, self.config.precision)
        except AttributeError as exc:
            raise ModelUnavailableError(
                f"PyTorch does not expose precision '{self.config.precision}'."
            ) from exc
        try:
            self._processor = AutoProcessor.from_pretrained(self.config.checkpoint_id)
            self._model = AutoModel.from_pretrained(
                self.config.checkpoint_id, dtype=dtype, low_cpu_mem_usage=True
            ).to(self.config.device)
        except Exception as exc:
            raise ModelUnavailableError(
                f"Could not load embedding model '{self.config.checkpoint_id}': {exc}. "
                "Check Hugging Face access/token and network connectivity, then run "
                "`agentic-cv doctor`."
            ) from exc
        self._model.eval()
        self._torch = torch

    def embed_image(self, image: Image.Image) -> np.ndarray:
        with self._guard():
            self._load()
            inputs = self._processor(
                images=[image.convert("RGB")], return_tensors="pt"
            ).to(self.config.device)
            with self._torch.inference_mode():
                output = self._model.get_image_features(**inputs)
            return self._to_numpy(output)

    def embed_text(self, query: str) -> np.ndarray:
        if not query.strip():
            raise ValueError("query must not be empty")
        with self._guard():
            self._load()
            inputs = self._processor(
                text=[query], padding="max_length", return_tensors="pt"
            ).to(self.config.device)
            with self._torch.inference_mode():
                output = self._model.get_text_features(**inputs)
            return self._to_numpy(output)

    @staticmethod
    def _to_numpy(output: Any) -> np.ndarray:
        vector = getattr(output, "pooler_output", output)
        if hasattr(vector, "detach"):
            vector = vector.detach().float().cpu().numpy()
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim == 2 and vector.shape[0] == 1:
            vector = vector[0]
        if vector.ndim != 1:
            raise ValueError("embedding model must return one vector per request")
        return vector

    def close(self) -> None:
        guard = (
            self._resources.lock
            if self._resources is not None
            else self._local_lock
        )
        with guard:
            self._model = None
            self._processor = None
            if self._torch is not None and self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
            self._torch = None


def _install_hint() -> str:
    return (
        "Image/text embeddings are unavailable because PyTorch/Transformers are not "
        "installed. Install the optional ML extra with `uv sync --extra ml` "
        "(Windows CUDA: install torch/torchvision from "
        "https://download.pytorch.org/whl/cu124), set AGENTIC_CV_DEVICE=cpu for CPU "
        "inference, then restart the server. Run `agentic-cv doctor` for details."
    )
