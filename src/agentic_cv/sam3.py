"""Provider-neutral SAM 3 inference with lazy local model loading."""

from __future__ import annotations

import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
from PIL import Image

from .errors import ModelUnavailableError
from .model_resources import ModelResourceManager


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


def clip_box(
    box: tuple[float, float, float, float], *, width: int, height: int
) -> tuple[float, float, float, float] | None:
    """Clip an xyxy box to image bounds, returning ``None`` if it is empty."""

    x1 = max(0.0, min(float(box[0]), float(width)))
    y1 = max(0.0, min(float(box[1]), float(height)))
    x2 = max(0.0, min(float(box[2]), float(width)))
    y2 = max(0.0, min(float(box[3]), float(height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


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
        resources: ModelResourceManager | None = None,
        resource_name: str = "sam3",
        estimated_vram_mb: int = 2048,
    ) -> None:
        self.config = config or Sam3Config()
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

        with self._guard():
            self._load()
            inputs = self._processor(
                images=image.convert("RGB"), text=prompt, return_tensors="pt"
            ).to(self.config.device)
            if self.config.precision == "float16" and "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].half()
            context = (
                self._torch.inference_mode()
                if self._torch is not None
                else nullcontext()
            )
            with context:
                outputs = self._model(**inputs)
            results = self._processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=mask_threshold,
                target_sizes=inputs["original_sizes"].tolist(),
            )[0]
            predictions = []
            for mask, box, score in zip(
                results["masks"], results["boxes"], results["scores"]
            ):
                clipped = clip_box(
                    tuple(float(value) for value in box),
                    width=image.width,
                    height=image.height,
                )
                if clipped is None:
                    continue
                predictions.append(
                    Sam3Prediction(
                        score=float(score),
                        box=clipped,
                        mask=mask.cpu().numpy(),
                    )
                )
            return predictions

    def predict_many(
        self,
        image: Image.Image,
        prompts: list[tuple[str, float]],
        *,
        mask_threshold: float | None = None,
    ) -> list[list[Sam3Prediction]]:
        """Several (prompt, threshold) pairs on one image with one vision pass.

        The vision encoder dominates SAM 3's cost; its features are computed
        once and reused for every class prompt. Measured on an RTX 4050
        Laptop with two classes: 1.11 s -> 0.63 s per 720x1280 image, with
        identical detections.
        """

        if not prompts:
            return []
        if len(prompts) == 1:
            prompt, threshold = prompts[0]
            return [self.predict(image, prompt, threshold=threshold, mask_threshold=mask_threshold)]
        mask_threshold = self.config.default_mask_threshold if mask_threshold is None else mask_threshold
        for prompt, threshold in prompts:
            if not prompt.strip():
                raise ValueError("prompt must not be empty")
            if not 0 <= threshold <= 1:
                raise ValueError("thresholds must be between 0 and 1")
        with self._guard():
            self._load()
            rgb = image.convert("RGB")
            inputs = self._processor(images=rgb, return_tensors="pt").to(self.config.device)
            pixel_values = inputs["pixel_values"]
            if self.config.precision == "float16":
                pixel_values = pixel_values.half()
            context = self._torch.inference_mode() if self._torch is not None else nullcontext()
            results: list[list[Sam3Prediction]] = []
            with context:
                vision = self._model.get_vision_features(pixel_values=pixel_values)
                for prompt, threshold in prompts:
                    text = self._processor(text=prompt, return_tensors="pt").to(self.config.device)
                    outputs = self._model(
                        vision_embeds=vision,
                        input_ids=text["input_ids"],
                        attention_mask=text["attention_mask"],
                    )
                    processed = self._processor.post_process_instance_segmentation(
                        outputs,
                        threshold=threshold,
                        mask_threshold=mask_threshold,
                        target_sizes=inputs["original_sizes"].tolist(),
                    )[0]
                    results.append(self._predictions(processed, image))
            return results

    def _predictions(self, results: Any, image: Image.Image) -> list[Sam3Prediction]:
        predictions = []
        for mask, box, score in zip(results["masks"], results["boxes"], results["scores"]):
            clipped = clip_box(
                tuple(float(value) for value in box), width=image.width, height=image.height
            )
            if clipped is None:
                continue
            predictions.append(Sam3Prediction(score=float(score), box=clipped, mask=mask.cpu().numpy()))
        return predictions

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
        "SAM 3 is unavailable because PyTorch/Transformers/torchvision are not all "
        "installed. Install the optional ML extra with `uv sync --extra ml` "
        "(Windows CUDA: install torch/torchvision from "
        "https://download.pytorch.org/whl/cu124), set AGENTIC_CV_DEVICE=cpu for CPU "
        "inference, then restart the server. Run `agentic-cv doctor` for details."
    )


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")
