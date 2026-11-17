"""Lazy Hugging Face image/text embeddings for semantic search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from PIL import Image


@dataclass(frozen=True, kw_only=True)
class HuggingFaceEmbeddingConfig:
    checkpoint_id: str = "google/siglip2-so400m-patch14-384"
    device: str = "cuda"
    precision: str = "float16"

    def __post_init__(self) -> None:
        if self.precision not in ("float16", "bfloat16"):
            raise ValueError("precision must be float16 or bfloat16")


class HuggingFaceEmbeddingProvider:
    """Lazy provider compatible with SigLIP-style Transformers checkpoints."""

    def __init__(
        self,
        config: HuggingFaceEmbeddingConfig | None = None,
        *,
        loader: Callable[[HuggingFaceEmbeddingConfig], tuple[Any, Any, Any]] | None = None,
    ) -> None:
        self.config = config or HuggingFaceEmbeddingConfig()
        self._loader = loader
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if self._loader is not None:
            self._model, self._processor, self._torch = self._loader(self.config)
            return

        import torch
        from transformers import AutoModel, AutoProcessor

        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("embedding provider requires CUDA for the configured device")
        dtype = getattr(torch, self.config.precision)
        self._processor = AutoProcessor.from_pretrained(self.config.checkpoint_id)
        self._model = AutoModel.from_pretrained(
            self.config.checkpoint_id, dtype=dtype, low_cpu_mem_usage=True
        ).to(self.config.device)
        self._model.eval()
        self._torch = torch

    def embed_image(self, image: Image.Image) -> np.ndarray:
        self._load()
        inputs = self._processor(images=[image.convert("RGB")], return_tensors="pt").to(
            self.config.device
        )
        with self._torch.inference_mode():
            output = self._model.get_image_features(**inputs)
        return self._to_numpy(output)

    def embed_text(self, query: str) -> np.ndarray:
        if not query.strip():
            raise ValueError("query must not be empty")
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
        self._model = None
        self._processor = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        self._torch = None
