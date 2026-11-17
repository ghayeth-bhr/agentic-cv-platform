"""Validated annotation records and deterministic dataset exporters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DatasetTask = Literal["detection", "segmentation"]


@dataclass(frozen=True, kw_only=True)
class DatasetImage:
    image_id: str
    file_name: str
    width: int
    height: int

    def __post_init__(self) -> None:
        if not self.image_id.strip() or not self.file_name.strip():
            raise ValueError("image_id and file_name must not be empty")
        if self.width < 1 or self.height < 1:
            raise ValueError("image dimensions must be positive")


@dataclass(frozen=True, kw_only=True)
class DatasetCategory:
    category_id: int
    name: str

    def __post_init__(self) -> None:
        if self.category_id < 0 or not self.name.strip():
            raise ValueError("category_id must be non-negative and name must not be empty")


@dataclass(frozen=True, kw_only=True)
class Annotation:
    annotation_id: str
    image_id: str
    category_id: int
    bbox: tuple[float, float, float, float]
    segmentation: tuple[tuple[float, ...], ...] = ()
    score: float | None = None

    def validate(self, image: DatasetImage) -> None:
        if len(self.bbox) != 4:
            raise ValueError("bbox must contain positive x, y, width, and height")
        x, y, width, height = self.bbox
        if width <= 0 or height <= 0:
            raise ValueError("bbox must contain positive x, y, width, and height")
        if x < 0 or y < 0 or x + width > image.width or y + height > image.height:
            raise ValueError("bbox must fit within image dimensions")
        if self.score is not None and not 0 <= self.score <= 1:
            raise ValueError("score must be between 0 and 1")
        for polygon in self.segmentation:
            if len(polygon) < 6 or len(polygon) % 2:
                raise ValueError("segmentation polygons need at least three coordinate pairs")
            if any(value < 0 for value in polygon):
                raise ValueError("segmentation coordinates must be non-negative")


@dataclass(frozen=True, kw_only=True)
class DatasetVersion:
    images: tuple[DatasetImage, ...]
    categories: tuple[DatasetCategory, ...]
    annotations: tuple[Annotation, ...]

    def __post_init__(self) -> None:
        images = {image.image_id: image for image in self.images}
        categories = {category.category_id for category in self.categories}
        if len(images) != len(self.images) or len(categories) != len(self.categories):
            raise ValueError("image IDs and category IDs must be unique")
        for annotation in self.annotations:
            if annotation.image_id not in images or annotation.category_id not in categories:
                raise ValueError("annotation references an unknown image or category")
            annotation.validate(images[annotation.image_id])


def export_coco(dataset: DatasetVersion) -> dict:
    return {
        "images": [
            {"id": image.image_id, "file_name": image.file_name, "width": image.width, "height": image.height}
            for image in dataset.images
        ],
        "categories": [
            {"id": category.category_id, "name": category.name, "supercategory": "object"}
            for category in dataset.categories
        ],
        "annotations": [
            {
                "id": annotation.annotation_id,
                "image_id": annotation.image_id,
                "category_id": annotation.category_id,
                "bbox": list(annotation.bbox),
                "area": annotation.bbox[2] * annotation.bbox[3],
                "segmentation": [list(polygon) for polygon in annotation.segmentation],
                "iscrowd": 0,
            }
            for annotation in dataset.annotations
        ],
    }


def write_coco(dataset: DatasetVersion, path: Path) -> None:
    path.write_text(json.dumps(export_coco(dataset), indent=2), encoding="utf-8")


def export_yolo(dataset: DatasetVersion, task: DatasetTask) -> dict[str, str]:
    if task not in ("detection", "segmentation"):
        raise ValueError("task must be detection or segmentation")
    category_ids = {category.category_id for category in dataset.categories}
    annotations_by_image: dict[str, list[Annotation]] = {image.image_id: [] for image in dataset.images}
    for annotation in dataset.annotations:
        annotations_by_image[annotation.image_id].append(annotation)
    files: dict[str, str] = {}
    for image in dataset.images:
        lines: list[str] = []
        for annotation in annotations_by_image[image.image_id]:
            if annotation.category_id not in category_ids:
                raise ValueError("annotation references an unknown category")
            x, y, width, height = annotation.bbox
            center_x = (x + width / 2) / image.width
            center_y = (y + height / 2) / image.height
            if task == "detection":
                values = (annotation.category_id, center_x, center_y, width / image.width, height / image.height)
            else:
                if not annotation.segmentation:
                    raise ValueError("segmentation export requires polygon coordinates")
                polygon_values = tuple(
                    coordinate / (image.width if index % 2 == 0 else image.height)
                    for polygon in annotation.segmentation
                    for index, coordinate in enumerate(polygon)
                )
                values = (annotation.category_id, *polygon_values)
            lines.append(" ".join(str(value) for value in values))
        files[f"{Path(image.file_name).stem}.txt"] = "\n".join(lines) + ("\n" if lines else "")
    return files
