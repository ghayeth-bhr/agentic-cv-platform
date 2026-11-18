"""Batch SAM3 labeling with raw-output preservation and dataset conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import cv2
import numpy as np
from PIL import Image

from .dataset import Annotation, DatasetCategory, DatasetImage, DatasetVersion
from .sam3 import Sam3Prediction, Sam3Provider


BatchTask = Literal["detection", "segmentation"]
ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True, kw_only=True)
class LabelImage:
    image_id: str
    path: Path
    file_name: str | None = None


@dataclass(frozen=True, kw_only=True)
class RawPrediction:
    image_id: str
    score: float
    box: tuple[float, float, float, float]
    mask: np.ndarray


@dataclass(frozen=True, kw_only=True)
class BatchLabelResult:
    raw_predictions: tuple[RawPrediction, ...]
    dataset: DatasetVersion


def mask_polygons(mask: np.ndarray) -> tuple[tuple[float, ...], ...]:
    binary = (mask.astype(np.uint8) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        polygon = tuple(float(value) for point in contour.reshape(-1, 2) for value in point)
        if len(polygon) >= 6:
            polygons.append(polygon)
    return tuple(polygons)


class Sam3BatchLabeler:
    def __init__(self, provider: Sam3Provider) -> None:
        self.provider = provider

    def label(
        self,
        images: list[LabelImage],
        *,
        prompt: str,
        threshold: float,
        category: DatasetCategory,
        task: BatchTask,
        on_progress: ProgressCallback | None = None,
    ) -> BatchLabelResult:
        if not images:
            raise ValueError("at least one image is required")
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if task not in ("detection", "segmentation"):
            raise ValueError("task must be detection or segmentation")

        dataset_images: list[DatasetImage] = []
        annotations: list[Annotation] = []
        raw_predictions: list[RawPrediction] = []
        for image in images:
            with Image.open(image.path) as source:
                source = source.convert("RGB")
                dataset_images.append(
                    DatasetImage(
                        image_id=image.image_id,
                        file_name=image.file_name or image.path.name,
                        width=source.width,
                        height=source.height,
                    )
                )
                predictions = self.provider.predict(source, prompt, threshold=threshold)
            for index, prediction in enumerate(predictions):
                raw_predictions.append(
                    RawPrediction(
                        image_id=image.image_id, score=prediction.score,
                        box=prediction.box, mask=prediction.mask,
                    )
                )
                x1, y1, x2, y2 = prediction.box
                segmentation = mask_polygons(prediction.mask) if task == "segmentation" else ()
                annotations.append(
                    Annotation(
                        annotation_id=f"{image.image_id}:{index}",
                        image_id=image.image_id,
                        category_id=category.category_id,
                        bbox=(x1, y1, x2 - x1, y2 - y1),
                        segmentation=segmentation,
                        score=prediction.score,
                    )
                )
            if on_progress is not None:
                on_progress(image.image_id, len(dataset_images), len(images))

        dataset = DatasetVersion(
            images=tuple(dataset_images), categories=(category,), annotations=tuple(annotations)
        )
        return BatchLabelResult(raw_predictions=tuple(raw_predictions), dataset=dataset)
