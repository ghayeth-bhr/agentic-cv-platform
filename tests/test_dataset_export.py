import json
from uuid import uuid4

import pytest

from agentic_cv.dataset import Annotation, DatasetCategory, DatasetImage, DatasetVersion
from agentic_cv.dataset_export import DatasetExportService
from agentic_cv.dataset_store import DatasetVersionStore


def make_dataset() -> DatasetVersion:
    return DatasetVersion(
        images=(DatasetImage(image_id="image-1", file_name="frame.jpg", width=10, height=10),),
        categories=(DatasetCategory(category_id=0, name="object"),),
        annotations=(Annotation(annotation_id="a-1", image_id="image-1", category_id=0, bbox=(1, 1, 4, 4)),),
    )


def test_export_service_writes_coco_and_manifest(tmp_path):
    tenant_id = uuid4()
    versions = DatasetVersionStore(str(tmp_path / "versions.sqlite3"))
    version = versions.create(
        tenant_id=tenant_id, name="v1", dataset=make_dataset(), idempotency_key="v1"
    )
    result = DatasetExportService(versions, tmp_path / "exports").export(
        tenant_id=tenant_id, version_id=version.version_id, format="coco"
    )

    payload = json.loads((result.output_dir / "annotations.json").read_text())
    manifest = json.loads((result.output_dir / "manifest.json").read_text())
    assert payload["images"][0]["file_name"] == "frame.jpg"
    assert manifest["format"] == "coco"
    assert result.files == (result.output_dir / "annotations.json", result.output_dir / "manifest.json")
    versions.close()


def test_export_service_writes_yolo_and_enforces_tenant_scope(tmp_path):
    owner = uuid4()
    versions = DatasetVersionStore(str(tmp_path / "versions.sqlite3"))
    version = versions.create(
        tenant_id=owner, name="v1", dataset=make_dataset(), idempotency_key="v1"
    )
    service = DatasetExportService(versions, tmp_path / "exports")
    result = service.export(tenant_id=owner, version_id=version.version_id, format="yolo-detection")

    assert (result.output_dir / "labels" / "frame.txt").read_text() == "0 0.3 0.3 0.4 0.4\n"
    with pytest.raises(KeyError, match="unknown"):
        service.export(tenant_id=uuid4(), version_id=version.version_id, format="coco")
    versions.close()


def test_export_service_rejects_unknown_format(tmp_path):
    versions = DatasetVersionStore(str(tmp_path / "versions.sqlite3"))
    with pytest.raises(ValueError, match="format"):
        DatasetExportService(versions, tmp_path / "exports").export(
            tenant_id=uuid4(), version_id=uuid4(), format="xml"
        )
    versions.close()