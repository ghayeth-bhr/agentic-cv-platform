"""Local export orchestration for persisted dataset versions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import UUID

from .dataset import export_coco, export_yolo
from .dataset_store import DatasetVersionStore

ExportFormat = Literal["coco", "yolo-detection", "yolo-segmentation"]


@dataclass(frozen=True, kw_only=True)
class DatasetExportResult:
    version_id: UUID
    tenant_id: UUID
    format: ExportFormat
    output_dir: Path
    files: tuple[Path, ...]


class DatasetExportService:
    def __init__(self, versions: DatasetVersionStore, output_root: Path) -> None:
        self.versions = versions
        self.output_root = output_root

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
            files = tuple(labels_dir / filename for filename in sorted(exported))
            for path, content in zip(files, (exported[name] for name in sorted(exported))):
                path.write_text(content, encoding="utf-8")

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
        return DatasetExportResult(
            version_id=version_id, tenant_id=tenant_id, format=format,
            output_dir=output_dir, files=files + (manifest,),
        )