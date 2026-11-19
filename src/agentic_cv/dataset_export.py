"""Local export orchestration for persisted dataset versions."""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import UUID

from .dataset import export_coco, export_yolo
from .dataset_store import DatasetVersionBackend

ExportFormat = Literal["coco", "yolo-detection", "yolo-segmentation"]


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
        output_dir = self.output_root / str(tenant_id) / str(version_id) / format
        output_dir.mkdir(parents=True, exist_ok=True)

        if format == "coco":
            output_path = output_dir / "annotations.json"
            output_path.write_text(json.dumps(export_coco(record.dataset), indent=2), encoding="utf-8")
            files = (output_path,)
        else:
            labels_dir = output_dir / "labels"
            labels_dir.mkdir(exist_ok=True)
            task = "detection" if format == "yolo-detection" else "segmentation"
            exported = export_yolo(record.dataset, task)
            label_files = tuple(labels_dir / filename for filename in sorted(exported))
            for path, content in zip(label_files, (exported[name] for name in sorted(exported))):
                path.write_text(content, encoding="utf-8")
            data_path = output_dir / "data.yaml"
            data_path.write_text(_yolo_data_yaml(record.dataset), encoding="utf-8")
            files = (*label_files, data_path)

        image_files = self._copy_dataset_images(
            tenant_id=tenant_id, version_id=version_id, output_dir=output_dir
        )
        files = (*files, *image_files)

        manifest = output_dir / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "tenant_id": str(tenant_id),
                    "version_id": str(version_id),
                    "format": format,
                    "files": [str(path.relative_to(output_dir)) for path in files],
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

    def _copy_dataset_images(
        self, *, tenant_id: UUID, version_id: UUID, output_dir: Path
    ) -> tuple[Path, ...]:
        source_images = self.images_root / str(tenant_id) / str(version_id) / "images"
        if not source_images.is_dir():
            return ()
        target_images = output_dir / "images"
        shutil.copytree(source_images, target_images, dirs_exist_ok=True)
        return tuple(sorted(path for path in target_images.rglob("*") if path.is_file()))

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


def _yolo_data_yaml(dataset) -> str:
    lines = ["path: .", "train: images", "val: images", "names:"]
    for category in dataset.categories:
        lines.append(f"  {category.category_id}: {json.dumps(category.name)}")
    return "\n".join(lines) + "\n"