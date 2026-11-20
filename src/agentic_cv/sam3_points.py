"""Point-prompted SAM 3 segmentation for click refinement.

The older refinement re-ran SAM 3's *text* prompt and picked the returned mask
nearest the clicks, so it could not add an object the text prompt missed.
``Sam3TrackerModel`` (SAM 3's promptable visual segmentation head) segments
from positive/negative points directly.

Measured on an RTX 4050 Laptop (6 GB), 720x1280 frame: load 5.8 s and
~0.9 GB, image embedding 0.78 s once per image, each click 0.03-0.15 s.
The embedding is cached per image, so only the first click pays for it.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Callable

import numpy as np
from PIL import Image

from .click_refinement import ClickPoint
from .errors import ModelUnavailableError
from .model_resources import ModelResourceManager
from .sam3 import Sam3Config, Sam3Prediction, clip_box

_EMBEDDING_CACHE = 4


class Sam3PointProvider:
    """Lazy SAM 3 tracker with a small per-image embedding cache."""

    def __init__(
        self,
        config: Sam3Config | None = None,
        *,
        loader: Callable[[Sam3Config], tuple[Any, Any, Any]] | None = None,
        resources: ModelResourceManager | None = None,
        resource_name: str = "sam3-tracker",
        estimated_vram_mb: int = 1100,
    ) -> None:
        self.config = config or Sam3Config()
        self._loader = loader
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self._resources = resources
        self._resource_name = resource_name
        self._local_lock = threading.RLock()
        self._embeddings: OrderedDict[tuple[int, tuple[int, int]], Any] = OrderedDict()
        if resources is not None:
            resources.register(resource_name, self, estimated_mb=estimated_vram_mb)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _guard(self):
        if self._resources is not None:
            return self._resources.lease(self._resource_name)
        return self._local_lock

    def _load(self) -> None:
        if self._model is not None:
            return
        if self._loader is not None:
            self._model, self._processor, self._torch = self._loader(self.config)
            return
        try:
            import torch
            from transformers import Sam3TrackerModel, Sam3TrackerProcessor
        except ImportError as exc:
            raise ModelUnavailableError(
                "Point refinement needs the ML extra (torch, transformers>=5.14 with Sam3Tracker). "
                "Run `uv sync --extra ml` and `agentic-cv doctor`."
            ) from exc
        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise ModelUnavailableError(
                "Point refinement is configured for CUDA but CUDA is not available."
            )
        try:
            self._processor = Sam3TrackerProcessor.from_pretrained(self.config.checkpoint_id)
            self._model = (
                Sam3TrackerModel.from_pretrained(
                    self.config.checkpoint_id, dtype=getattr(torch, self.config.precision)
                )
                .to(self.config.device)
                .eval()
            )
        except Exception as exc:
            raise ModelUnavailableError(
                f"Could not load the SAM 3 tracker from '{self.config.checkpoint_id}': {exc}"
            ) from exc
        self._torch = torch

    def _embedding(self, image: Image.Image) -> Any:
        # Refinement sessions keep their PIL image alive, so its id is a stable
        # key for the session's lifetime; the size guards against id reuse.
        key = (id(image), image.size)
        cached = self._embeddings.get(key)
        if cached is not None:
            self._embeddings.move_to_end(key)
            return cached
        inputs = self._processor(images=image, return_tensors="pt").to(self.config.device)
        pixel_values = inputs["pixel_values"]
        if self._torch is not None and self.config.precision in ("float16", "bfloat16"):
            pixel_values = pixel_values.to(getattr(self._torch, self.config.precision))
        embedding = self._model.get_image_embeddings(pixel_values)
        self._embeddings[key] = embedding
        while len(self._embeddings) > _EMBEDDING_CACHE:
            self._embeddings.popitem(last=False)
        return embedding

    def segment(self, image: Image.Image, points: tuple[ClickPoint, ...]) -> list[Sam3Prediction]:
        """Best mask for one object described by positive/negative clicks."""

        if not points:
            return []
        image = image if image.mode == "RGB" else image.convert("RGB")
        coordinates = [[float(point.x), float(point.y)] for point in points]
        labels = [1 if point.label == "positive" else 0 for point in points]
        with self._guard():
            self._load()
            context = self._torch.inference_mode() if self._torch is not None else _null()
            with context:
                embedding = self._embedding(image)
                prompt = self._processor(
                    images=image,
                    input_points=[[coordinates]],
                    input_labels=[[labels]],
                    return_tensors="pt",
                ).to(self.config.device)
                output = self._model(
                    input_points=prompt["input_points"],
                    input_labels=prompt["input_labels"],
                    image_embeddings=embedding,
                    multimask_output=True,
                )
                masks = self._processor.post_process_masks(
                    output.pred_masks.cpu(), prompt["original_sizes"]
                )[0][0]
                scores = output.iou_scores[0, 0].float().cpu().numpy()
        best = int(np.argmax(scores))
        mask = np.asarray(masks[best].numpy() if hasattr(masks[best], "numpy") else masks[best], dtype=bool)
        if not mask.any():
            return []
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        box = clip_box(
            (float(cols[0]), float(rows[0]), float(cols[-1] + 1), float(rows[-1] + 1)),
            width=image.width,
            height=image.height,
        )
        if box is None:
            return []
        return [Sam3Prediction(score=float(min(1.0, max(0.0, scores[best]))), box=box, mask=mask)]

    def close(self) -> None:
        guard = self._resources.lock if self._resources is not None else self._local_lock
        with guard:
            self._embeddings.clear()
            self._model = None
            self._processor = None
            if self._torch is not None and self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
            self._torch = None


class Sam3PointRefinementProvider:
    """Click refinement backed by point prompts, with the text-ranked fallback.

    If the tracker cannot load (no ML extra, missing weights), the provider
    falls back to ranking SAM 3's text-prompt masks by the clicks, and says so
    in ``mode``.
    """

    def __init__(self, points: Sam3PointProvider, fallback: Any | None = None) -> None:
        self.points = points
        self.fallback = fallback
        self.mode = "points"

    def refine(self, image: Image.Image, points: tuple[ClickPoint, ...]) -> list[Sam3Prediction]:
        return self.refine_with_prompt(image, points, "object")

    def refine_with_prompt(
        self, image: Image.Image, points: tuple[ClickPoint, ...], prompt: str
    ) -> list[Sam3Prediction]:
        if self.mode == "points":
            try:
                return self.points.segment(image, points)
            except ModelUnavailableError:
                if self.fallback is None:
                    raise
                self.mode = "ranked"
        return self.fallback.refine_with_prompt(image, points, prompt)


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
