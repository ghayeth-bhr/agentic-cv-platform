"""Validated annotation records and deterministic dataset exporters."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

DatasetTask = Literal["detection", "segmentation"]
DatasetSplit = Literal["train", "validation", "test"]
SPLITS: tuple[DatasetSplit, ...] = ("train", "validation", "test")
DEFAULT_SPLIT_RATIOS = (0.8, 0.1, 0.1)


@dataclass(frozen=True, kw_only=True)
class DatasetImage:
    image_id: str
    file_name: str
    width: int
    height: int
    split: DatasetSplit | None = None

    def __post_init__(self) -> None:
        if not self.image_id.strip() or not self.file_name.strip():
            raise ValueError("image_id and file_name must not be empty")
        if self.width < 1 or self.height < 1:
            raise ValueError("image dimensions must be positive")
        if self.split is not None and self.split not in SPLITS:
            raise ValueError("split must be train, validation, or test")


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
    deleted: bool = False

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
    # Annotation set this version reads from (see ``annotation_sets``); None
    # marks a version created before sets existed.
    annotation_set: str | None = None
    source_asset_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        images = {image.image_id: image for image in self.images}
        categories = {category.category_id for category in self.categories}
        if len(images) != len(self.images) or len(categories) != len(self.categories):
            raise ValueError("image IDs and category IDs must be unique")
        for annotation in self.annotations:
            if annotation.image_id not in images or annotation.category_id not in categories:
                raise ValueError("annotation references an unknown image or category")
            if annotation.deleted:
                raise ValueError("deleted annotations cannot be part of a dataset version")
            annotation.validate(images[annotation.image_id])

    def split_of(self, image: DatasetImage) -> DatasetSplit:
        """The persisted split, or the deterministic assignment for legacy versions."""

        if image.split is not None:
            return image.split
        return assign_splits(self.images)[image.image_id]

    def with_splits(self) -> "DatasetVersion":
        """Return a copy where every image carries an explicit split."""

        if all(image.split is not None for image in self.images):
            return self
        assignments = assign_splits(self.images)
        return replace(
            self,
            images=tuple(
                image if image.split is not None else replace(image, split=assignments[image.image_id])
                for image in self.images
            ),
        )


def split_group(image_id: str) -> str:
    """Images from the same source share a split so video frames cannot leak.

    Label image ids look like ``asset:source:image`` or
    ``asset:source:frame:N``; the group is ``asset:source``. Other ids are
    their own group.
    """

    parts = image_id.split(":")
    if len(parts) >= 3:
        return ":".join(parts[:2])
    return image_id


def assign_splits(
    images: tuple[DatasetImage, ...] | list[DatasetImage],
    *,
    ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    seed: int = 0,
) -> dict[str, DatasetSplit]:
    """Deterministic, source-aware train/validation/test assignment.

    Whole source groups are assigned, in hash order, to the split whose image
    count is furthest below its target. Small datasets keep everything in
    train until another split is owed at least one image.
    """

    if any(ratio < 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("split ratios must be non-negative and sum to 1")
    groups: dict[str, list[str]] = {}
    for image in images:
        groups.setdefault(split_group(image.image_id), []).append(image.image_id)
    total = sum(len(members) for members in groups.values())
    targets = [total * ratio for ratio in ratios]
    assigned = [0, 0, 0]
    result: dict[str, DatasetSplit] = {}
    ordered = sorted(
        groups,
        key=lambda key: hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest(),
    )
    for key in ordered:
        size = len(groups[key])
        deficits = [targets[index] - assigned[index] for index in range(3)]
        index = max(range(3), key=lambda item: (deficits[item], -item))
        if index != 0 and deficits[index] < size / 2:
            index = 0
        assigned[index] += size
        for image_id in groups[key]:
            result[image_id] = SPLITS[index]
    return result


def split_dataset(
    dataset: DatasetVersion,
    *,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 0,
) -> dict[DatasetSplit, DatasetVersion]:
    """Create deterministic, image-disjoint train/validation/test versions."""

    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(ratio < 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("split ratios must be non-negative and sum to 1")
    if not dataset.images:
        raise ValueError("dataset must contain at least one image")

    ordered = sorted(
        dataset.images,
        key=lambda image: hashlib.sha256(f"{seed}:{image.image_id}".encode("utf-8")).digest(),
    )
    exact_counts = [len(ordered) * ratio for ratio in ratios]
    counts = [int(value) for value in exact_counts]
    for index in sorted(
        range(3), key=lambda index: (-(exact_counts[index] - counts[index]), index)
    )[: len(ordered) - sum(counts)]:
        counts[index] += 1

    names: tuple[DatasetSplit, ...] = ("train", "validation", "test")
    assignments: dict[DatasetSplit, tuple[DatasetImage, ...]] = {}
    offset = 0
    for name, count in zip(names, counts):
        assignments[name] = tuple(ordered[offset : offset + count])
        offset += count

    annotations_by_image = {
        image_id: tuple(annotation for annotation in dataset.annotations if annotation.image_id == image_id)
        for image_id in (image.image_id for image in dataset.images)
    }
    return {
        name: DatasetVersion(
            images=images,
            categories=dataset.categories,
            annotations=tuple(
                annotation
                for image in images
                for annotation in annotations_by_image[image.image_id]
            ),
        )
        for name, images in assignments.items()
    }


def export_coco(dataset: DatasetVersion) -> dict:
    """COCO with integer ids (what Detectron2/MMDetection/pycocotools expect).

    The durable string ids are kept as ``image_uid``/``annotation_uid``.
    """

    image_numbers = {image.image_id: index for index, image in enumerate(dataset.images, start=1)}
    return {
        "images": [
            {
                "id": image_numbers[image.image_id],
                "image_uid": image.image_id,
                "file_name": image.file_name,
                "width": image.width,
                "height": image.height,
                "split": dataset.split_of(image),
            }
            for image in dataset.images
        ],
        "categories": [
            {"id": category.category_id, "name": category.name, "supercategory": "object"}
            for category in dataset.categories
        ],
        "annotations": [
            {
                "id": index,
                "annotation_uid": annotation.annotation_id,
                "image_id": image_numbers[annotation.image_id],
                "category_id": annotation.category_id,
                "bbox": list(annotation.bbox),
                "area": _annotation_area(annotation),
                "segmentation": [list(polygon) for polygon in annotation.segmentation],
                "iscrowd": 0,
                **({"score": annotation.score} if annotation.score is not None else {}),
            }
            for index, annotation in enumerate(dataset.annotations, start=1)
        ],
    }


def _polygon_area(polygon: tuple[float, ...]) -> float:
    xs = polygon[0::2]
    ys = polygon[1::2]
    count = len(xs)
    return abs(
        sum(xs[index] * ys[(index + 1) % count] - xs[(index + 1) % count] * ys[index] for index in range(count))
    ) / 2.0


def _annotation_area(annotation: Annotation) -> float:
    if annotation.segmentation:
        return round(sum(_polygon_area(polygon) for polygon in annotation.segmentation), 3)
    return annotation.bbox[2] * annotation.bbox[3]


def subset(dataset: DatasetVersion, split: DatasetSplit) -> DatasetVersion:
    images = tuple(image for image in dataset.images if dataset.split_of(image) == split)
    image_ids = {image.image_id for image in images}
    return replace(
        dataset,
        images=images,
        annotations=tuple(item for item in dataset.annotations if item.image_id in image_ids),
    )


def write_coco(dataset: DatasetVersion, path: Path) -> None:
    path.write_text(json.dumps(export_coco(dataset), indent=2), encoding="utf-8")


def yolo_class_indices(dataset: DatasetVersion) -> dict[int, int]:
    """YOLO needs contiguous 0-based class indices; map category ids in order."""

    return {
        category.category_id: index
        for index, category in enumerate(sorted(dataset.categories, key=lambda item: item.category_id))
    }


def merge_polygons(polygons: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
    """Join an instance's polygons into one YOLO-seg outline.

    YOLO segmentation has one polygon per line. Concatenating disjoint parts
    draws spurious edges between them; instead connect each part to the next
    through their closest vertices and walk back the same way (the approach
    Ultralytics uses for COCO conversion), so the bridge has zero area.
    """

    parts = [
        [(polygon[index], polygon[index + 1]) for index in range(0, len(polygon), 2)]
        for polygon in polygons
        if len(polygon) >= 6
    ]
    if not parts:
        return ()
    if len(parts) == 1:
        return tuple(value for point in parts[0] for value in point)

    def closest(first, second) -> tuple[int, int]:
        best = (float("inf"), 0, 0)
        for i, (x1, y1) in enumerate(first):
            for j, (x2, y2) in enumerate(second):
                distance = (x1 - x2) ** 2 + (y1 - y2) ** 2
                if distance < best[0]:
                    best = (distance, i, j)
        return best[1], best[2]

    links = [closest(parts[index], parts[index + 1]) for index in range(len(parts) - 1)]
    # Rotate each part so it starts at its incoming link vertex.
    path: list[tuple[float, float]] = []
    entry = [0] + [link[1] for link in links]
    exit_ = [link[0] for link in links] + [None]
    forward: list[list[tuple[float, float]]] = []
    for index, part in enumerate(parts):
        start = entry[index]
        rotated = part[start:] + part[:start]
        forward.append(rotated)
    # Walk out through each part to its exit vertex, then back the remainder.
    for index, part in enumerate(forward):
        if exit_[index] is None:
            path.extend(part + [part[0]])
            continue
        exit_offset = (exit_[index] - entry[index]) % len(part)
        path.extend(part[: exit_offset + 1])
    for index in range(len(forward) - 2, -1, -1):
        part = forward[index]
        exit_offset = (exit_[index] - entry[index]) % len(part)
        path.extend(part[exit_offset:] + [part[0]])
    return tuple(value for point in path for value in point)


def _fmt(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)


def export_yolo(dataset: DatasetVersion, task: DatasetTask) -> dict[str, str]:
    if task not in ("detection", "segmentation"):
        raise ValueError("task must be detection or segmentation")
    class_index = yolo_class_indices(dataset)
    annotations_by_image: dict[str, list[Annotation]] = {image.image_id: [] for image in dataset.images}
    for annotation in dataset.annotations:
        annotations_by_image[annotation.image_id].append(annotation)
    files: dict[str, str] = {}
    for image in dataset.images:
        lines: list[str] = []
        for annotation in annotations_by_image[image.image_id]:
            if annotation.category_id not in class_index:
                raise ValueError("annotation references an unknown category")
            label = class_index[annotation.category_id]
            x, y, width, height = annotation.bbox
            center_x = (x + width / 2) / image.width
            center_y = (y + height / 2) / image.height
            if task == "detection":
                values = (label, center_x, center_y, width / image.width, height / image.height)
            else:
                if not annotation.segmentation:
                    raise ValueError(
                        "segmentation export requires polygon coordinates; "
                        "this dataset was labeled for detection"
                    )
                merged = merge_polygons(annotation.segmentation)
                polygon_values = tuple(
                    min(1.0, max(0.0, coordinate / (image.width if index % 2 == 0 else image.height)))
                    for index, coordinate in enumerate(merged)
                )
                values = (label, *polygon_values)
            lines.append(" ".join(_fmt(value) for value in values))
        files[f"{Path(image.file_name).stem}.txt"] = "\n".join(lines) + ("\n" if lines else "")
    return files
