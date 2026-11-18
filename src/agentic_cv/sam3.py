"""Provider-neutral SAM 3 inference with lazy local model loading."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
from PIL import Image

from .errors import ModelUnavailableError


@dataclass(frozen=True, kw_only=True)
class Sam3Config:
    checkpoint_id: str = "facebook/sam3"
    device: str = "cuda"
    precision: str = "float16"
    default_threshold: float = 0.5
    default_mask_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.precision not in ("float16", "bfloat16", "float32"):
            raise ValueError("precision must be float16, bfloat16, or float32")
        for name in ("default_threshold", "default_mask_threshold"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True, kw_only=True)
class Sam3Prediction:
    score: float
    box: tuple[float, float, float, float]
    mask: np.ndarray


class Sam3Provider(Protocol):
    def predict(
        self,
        image: Image.Image,
        prompt: str,
        *,
        threshold: float | None = None,
        mask_threshold: float | None = None,
    ) -> list[Sam3Prediction]: ...


class LocalSam3Provider:
    """Lazy local provider using the tested Transformers SAM3 API."""

    def __init__(
        self,
        config: Sam3Config | None = None,
        *,
        loader: Callable[[Sam3Config], tuple[Any, Any, Any]] | None = None,
    ) -> None:
        self.config = config or Sam3Config()
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

        try:
            import torch
            from transformers import Sam3Model, Sam3Processor
        except ImportError as exc:
            raise ModelUnavailableError(_install_hint()) from exc

        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise ModelUnavailableError(
                "SAM 3 is configured for CUDA but CUDA is not available. "
                "Set AGENTIC_CV_DEVICE=cpu for CPU inference, or install CUDA-enabled "
                "PyTorch. Run `agentic-cv doctor` for details."
            )
        try:
            dtype = getattr(torch, self.config.precision)
        except AttributeError as exc:
            raise ModelUnavailableError(
                f"PyTorch does not expose precision '{self.config.precision}'."
            ) from exc
        try:
            self._processor = Sam3Processor.from_pretrained(self.config.checkpoint_id)
            self._model = Sam3Model.from_pretrained(
                self.config.checkpoint_id,
                dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(self.config.device)
        except Exception as exc:
            raise ModelUnavailableError(
                f"Could not load SAM 3 model '{self.config.checkpoint_id}': {exc}. "
                "The checkpoint is gated: request access at "
                "https://huggingface.co/facebook/sam3, set HF_TOKEN, and verify network "
                "connectivity. Run `agentic-cv doctor`."
            ) from exc
        self._model.eval()
        self._torch = torch

    def predict(
        self,
        image: Image.Image,
        prompt: str,
        *,
        threshold: float | None = None,
        mask_threshold: float | None = None,
    ) -> list[Sam3Prediction]:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        threshold = self.config.default_threshold if threshold is None else threshold
        mask_threshold = self.config.default_mask_threshold if mask_threshold is None else mask_threshold
        if not 0 <= threshold <= 1 or not 0 <= mask_threshold <= 1:
            raise ValueError("thresholds must be between 0 and 1")

        self._load()
        inputs = self._processor(images=image.convert("RGB"), text=prompt, return_tensors="pt").to(
            self.config.device
        )
        if self.config.precision == "float16" and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].half()
        context = self._torch.inference_mode() if self._torch is not None else nullcontext()
        with context:
            outputs = self._model(**inputs)
        results = self._processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs["original_sizes"].tolist(),
        )[0]
        return [
            Sam3Prediction(
                score=float(score),
                box=tuple(float(value) for value in box),
                mask=mask.cpu().numpy(),
            )
            for mask, box, score in zip(results["masks"], results["boxes"], results["scores"])
        ]

    def close(self) -> None:
        self._model = None
        self._processor = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        self._torch = None


def _install_hint() -> str:
    return (
        "SAM 3 is unavailable because PyTorch/Transformers/torchvision are not all "
        "installed. Install the optional ML extra with `uv sync --extra ml` "
        "(Windows CUDA: install torch/torchvision from "
        "https://download.pytorch.org/whl/cu124), set AGENTIC_CV_DEVICE=cpu for CPU "
        "inference, then restart the server. Run `agentic-cv doctor` for details."
    )


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")
