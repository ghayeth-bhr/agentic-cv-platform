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


@dataclass(frozen=True, kw_only=True)
class LabelClass:
    """One class of a labeling run: its SAM 3 prompt, threshold, and category."""

    prompt: str
    threshold: float
    category: DatasetCategory


class Sam3BatchLabeler:
    def __init__(self, provider: Sam3Provider) -> None:
        self.provider = provider

    def label(
        self,
        images: list[LabelImage],
        *,
        prompt: str | None = None,
        threshold: float | None = None,
        category: DatasetCategory | None = None,
        task: BatchTask,
        classes: list[LabelClass] | None = None,
        on_progress: ProgressCallback | None = None,
        on_image: ImageCallback | None = None,
        annotation_namespace: str | None = None,
        keep_raw: bool = True,
    ) -> BatchLabelResult:
        """Label every image for every class.

        A single-class run keeps the historical annotation ids
        (``<image_id>:<n>``); multi-class runs use ``<image_id>:<category_id>:<n>``.
        ``keep_raw=False`` drops the per-prediction masks the durable workflow
        never reads, so memory stays flat on long runs.
        """

        if classes is None:
            if prompt is None or threshold is None or category is None:
                raise ValueError("prompt, threshold, and category are required without classes")
            classes = [LabelClass(prompt=prompt, threshold=threshold, category=category)]
        if not images:
            raise ValueError("at least one image is required")
        if not classes:
            raise ValueError("at least one class is required")
        for item in classes:
            if not item.prompt.strip():
                raise ValueError("prompt must not be empty")
            if not 0 <= item.threshold <= 1:
                raise ValueError("threshold must be between 0 and 1")
        if len({item.category.category_id for item in classes}) != len(classes):
            raise ValueError("class category ids must be unique")
        if task not in ("detection", "segmentation"):
            raise ValueError("task must be detection or segmentation")
        multi = len(classes) > 1

        dataset_images: list[DatasetImage] = []
        annotations: list[Annotation] = []
        raw_predictions: list[RawPrediction] = []
        for image in images:
            image_annotations: list[Annotation] = []
            with Image.open(image.path) as source:
                source = source.convert("RGB")
                width, height = source.width, source.height
                dataset_images.append(
                    DatasetImage(
                        image_id=image.image_id,
                        file_name=image.file_name or image.path.name,
                        width=width,
                        height=height,
                    )
                )
                per_class = [
                    (item, self.provider.predict(source, item.prompt, threshold=item.threshold))
                    for item in classes
                ]
            for item, predictions in per_class:
                for index, prediction in enumerate(predictions):
                    if keep_raw:
                        raw_predictions.append(
                            RawPrediction(
                                image_id=image.image_id, score=prediction.score,
                                box=prediction.box, mask=prediction.mask,
                            )
                        )
                    clipped = clip_box(prediction.box, width=width, height=height)
                    if clipped is None:
                        continue
                    x1, y1, x2, y2 = clipped
                    segmentation = mask_polygons(prediction.mask) if task == "segmentation" else ()
                    local_id = (
                        f"{image.image_id}:{item.category.category_id}:{index}"
                        if multi
                        else f"{image.image_id}:{index}"
                    )
                    annotation = Annotation(
                        annotation_id=(
                            f"{annotation_namespace}/{local_id}"
                            if annotation_namespace
                            else local_id
                        ),
                        image_id=image.image_id,
                        category_id=item.category.category_id,
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
            images=tuple(dataset_images),
            categories=tuple(item.category for item in classes),
            annotations=tuple(annotations),
        )
        return BatchLabelResult(raw_predictions=tuple(raw_predictions), dataset=dataset)
