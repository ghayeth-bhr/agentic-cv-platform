"""Batch SAM3 labeling with raw-output preservation and dataset conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import cv2
import numpy as np
from PIL import Image

from .dataset import Annotation, DatasetCategory, DatasetImage, DatasetVersion
from .sam3 import Sam3Prediction, Sam3Provider, clip_box


BatchTask = Literal["detection", "segmentation"]
ProgressCallback = Callable[[str, int, int], None]
ImageCallback = Callable[[str, DatasetImage, tuple[Annotation, ...]], None]


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


POLYGON_EPSILON_RATIO = 0.002
MIN_PART_AREA_PX = 9.0
MIN_PART_AREA_RATIO = 0.01


def mask_polygons(
    mask: np.ndarray,
    *,
    epsilon_ratio: float = POLYGON_EPSILON_RATIO,
    min_area_px: float = MIN_PART_AREA_PX,
    min_area_ratio: float = MIN_PART_AREA_RATIO,
) -> tuple[tuple[float, ...], ...]:
    """Mask → simplified polygons, largest first.

    Raw contours carry one vertex per boundary pixel step, which bloats
    labels; ``approxPolyDP`` with a perimeter-relative epsilon keeps the shape
    (mask IoU stays ~0.98) at a fraction of the size. Speckles smaller than
    ``min_area_px`` or ``min_area_ratio`` of the largest part are dropped.
    """

    binary = (np.asarray(mask).astype(np.uint8) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        if len(contour) < 3:
            continue
        candidates.append((float(cv2.contourArea(contour)), contour))
    if not candidates:
        return ()
    largest = max(area for area, _ in candidates)
    polygons: list[tuple[float, tuple[float, ...]]] = []
    for area, contour in candidates:
        is_largest = area == largest
        if not is_largest and (area < min_area_px or area < largest * min_area_ratio):
            continue
        epsilon = max(0.5, epsilon_ratio * cv2.arcLength(contour, True))
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        if len(simplified) < 3:
            simplified = contour
        polygon = tuple(float(value) for point in simplified.reshape(-1, 2) for value in point)
        if len(polygon) >= 6:
            polygons.append((area, polygon))
    polygons.sort(key=lambda item: -item[0])
    return tuple(polygon for _, polygon in polygons)


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
        on_image: ImageCallback | None = None,
        annotation_namespace: str | None = None,
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
                width, height = source.width, source.height
            image_annotations: list[Annotation] = []
            for index, prediction in enumerate(predictions):
                raw_predictions.append(
                    RawPrediction(
                        image_id=image.image_id, score=prediction.score,
                        box=prediction.box, mask=prediction.mask,
                    )
                )
                clipped = clip_box(
                    prediction.box, width=width, height=height
                )
                if clipped is None:
                    continue
                x1, y1, x2, y2 = clipped
                segmentation = mask_polygons(prediction.mask) if task == "segmentation" else ()
                local_id = f"{image.image_id}:{index}"
                annotation = Annotation(
                    annotation_id=(
                        f"{annotation_namespace}/{local_id}"
                        if annotation_namespace
                        else local_id
                    ),
                    image_id=image.image_id,
                    category_id=category.category_id,
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                    segmentation=segmentation,
                    score=prediction.score,
                )
                annotations.append(annotation)
                image_annotations.append(annotation)
            if on_image is not None:
                on_image(image.image_id, dataset_images[-1], tuple(image_annotations))
            if on_progress is not None:
                on_progress(image.image_id, len(dataset_images), len(images))

        dataset = DatasetVersion(
            images=tuple(dataset_images), categories=(category,), annotations=tuple(annotations)
        )
        return BatchLabelResult(raw_predictions=tuple(raw_predictions), dataset=dataset)
