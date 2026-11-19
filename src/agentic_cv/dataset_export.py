"""Local export orchestration for persisted dataset versions."""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import UUID

from .dataset import (
    SPLITS,
    DatasetVersion,
    export_coco,
    export_yolo,
    subset,
    yolo_class_indices,
)
from .dataset_store import DatasetVersionBackend

ExportFormat = Literal["coco", "yolo-detection", "yolo-segmentation"]
YOLO_SPLIT_DIRS = {"train": "train", "validation": "val", "test": "test"}


@dataclass(frozen=True, kw_only=True)
class DatasetExportResult:
    version_id: UUID
    tenant_id: UUID
    format: ExportFormat
    output_dir: Path
    files: tuple[Path, ...]
    archive_path: Path | None = None


class DatasetExportService:
    def __init__(
        self,
        versions: DatasetVersionBackend,
        output_root: Path,
        *,
        images_root: Path | None = None,
    ) -> None:
        self.versions = versions
        self.output_root = output_root
        self.images_root = images_root or (output_root.parent / "dataset-images")

    def export(
        self,
        *,
        tenant_id: UUID,
        version_id: UUID,
        format: ExportFormat,
    ) -> DatasetExportResult:
        if format not in ("coco", "yolo-detection", "yolo-segmentation"):
            raise ValueError("format must be coco, yolo-detection, or yolo-segmentation")
        record = self.versions.get(tenant_id=tenant_id, version_id=version_id)
        dataset = record.dataset.with_splits()
        output_dir = self.output_root / str(tenant_id) / str(version_id) / format
        if output_dir.exists():
            # Re-exports must not keep files from an older layout.
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        source_images = self.images_root / str(tenant_id) / str(version_id) / "images"

        if format == "coco":
            files = self._write_coco(dataset, output_dir)
            image_files = self._copy_images(source_images, dataset, output_dir, by_split=False)
        else:
            task = "detection" if format == "yolo-detection" else "segmentation"
            files = self._write_yolo(dataset, output_dir, task)
            image_files = self._copy_images(source_images, dataset, output_dir, by_split=True)
        files = (*files, *image_files)

        manifest = output_dir / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "tenant_id": str(tenant_id),
                    "version_id": str(version_id),
                    "format": format,
                    "splits": {
                        split: sum(1 for image in dataset.images if image.split == split)
                        for split in SPLITS
                    },
                    "files": [str(path.relative_to(output_dir).as_posix()) for path in files],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        archive = self.package(
            tenant_id=tenant_id, version_id=version_id, format=format
        )
        return DatasetExportResult(
            version_id=version_id, tenant_id=tenant_id, format=format,
            output_dir=output_dir, files=files + (manifest,), archive_path=archive,
        )

    @staticmethod
    def _write_coco(dataset: DatasetVersion, output_dir: Path) -> tuple[Path, ...]:
        """``annotations.json`` (all images, each tagged with its split) plus
        ``annotations/instances_<split>.json`` for every non-empty split."""

        combined = output_dir / "annotations.json"
        combined.write_text(json.dumps(export_coco(dataset), indent=2), encoding="utf-8")
        written = [combined]
        split_dir = output_dir / "annotations"
        for split in SPLITS:
            part = subset(dataset, split)
            if not part.images:
                continue
            split_dir.mkdir(exist_ok=True)
            path = split_dir / f"instances_{split}.json"
            path.write_text(json.dumps(export_coco(part), indent=2), encoding="utf-8")
            written.append(path)
        return tuple(written)

    @staticmethod
    def _write_yolo(
        dataset: DatasetVersion, output_dir: Path, task: str
    ) -> tuple[Path, ...]:
        written: list[Path] = []
        for split in SPLITS:
            part = subset(dataset, split)
            if not part.images:
                continue
            labels_dir = output_dir / "labels" / YOLO_SPLIT_DIRS[split]
            labels_dir.mkdir(parents=True, exist_ok=True)
            exported = export_yolo(part, task)  # type: ignore[arg-type]
            for name in sorted(exported):
                path = labels_dir / name
                path.write_text(exported[name], encoding="utf-8")
                written.append(path)
        data_path = output_dir / "data.yaml"
        data_path.write_text(_yolo_data_yaml(dataset), encoding="utf-8")
        written.append(data_path)
        return tuple(written)

    @staticmethod
    def _copy_images(
        source_images: Path,
        dataset: DatasetVersion,
        output_dir: Path,
        *,
        by_split: bool,
    ) -> tuple[Path, ...]:
        if not source_images.is_dir():
            return ()
        copied: list[Path] = []
        for image in dataset.images:
            name = Path(image.file_name).name
            source = source_images / name
            if not source.is_file():
                continue
            if by_split:
                target = output_dir / "images" / YOLO_SPLIT_DIRS[image.split or "train"] / name
            else:
                target = output_dir / "images" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(target)
        return tuple(sorted(copied))

    def archive_path_for(
        self, *, tenant_id: UUID, version_id: UUID, format: ExportFormat
    ) -> Path:
        return self.output_root / str(tenant_id) / str(version_id) / f"{format}.zip"

    def package(
        self, *, tenant_id: UUID, version_id: UUID, format: ExportFormat
    ) -> Path:
        output_dir = self.output_root / str(tenant_id) / str(version_id) / format
        if not output_dir.is_dir():
            raise FileNotFoundError(f"export output does not exist: {output_dir}")
        archive = self.archive_path_for(
            tenant_id=tenant_id, version_id=version_id, format=format
        )
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(output_dir.rglob("*")):
                if path.is_file():
                    bundle.write(path, path.relative_to(output_dir))
        return archive


def _yolo_data_yaml(dataset: DatasetVersion) -> str:
    """Ultralytics data.yaml: 0-based contiguous names and real split folders.

    Ultralytics requires a ``val`` entry; a dataset too small to own a
    validation image validates on train, and the file says so.
    """

    present = {image.split for image in dataset.images}
    lines = ["path: .", "train: images/train"]
    if "validation" in present:
        lines.append("val: images/val")
    else:
        lines.append("val: images/train  # too few images for a separate validation split")
    if "test" in present:
        lines.append("test: images/test")
    indices = yolo_class_indices(dataset)
    lines.append(f"nc: {len(indices)}")
    lines.append("names:")
    for category in sorted(dataset.categories, key=lambda item: indices[item.category_id]):
        lines.append(f"  {indices[category.category_id]}: {json.dumps(category.name)}")
    return "\n".join(lines) + "\n"
