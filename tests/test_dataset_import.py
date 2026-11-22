"""Importing an external COCO dataset as a version the gallery can edit."""

from __future__ import annotations

import io
import json
import time
import zipfile
from uuid import UUID

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pycocotools import mask as mask_utils

from agentic_cv.annotation_store import AnnotationRevisionStore
from agentic_cv.application import LocalAppConfig, create_local_app
from agentic_cv.dataset_import import (
    DatasetImporter,
    DatasetImportError,
    _decode_rle,
)
from agentic_cv.dataset_store import DatasetVersionStore

TENANT = UUID("00000000-0000-0000-0000-000000000001")


def png(width=40, height=30, color=(200, 180, 20)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def rle_for(width, height, box):
    mask = np.zeros((height, width), dtype=np.uint8)
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = 1
    encoded = mask_utils.encode(np.asfortranarray(mask))
    return {"size": [height, width], "counts": encoded["counts"].decode("ascii")}


def coco_zip(coco: dict, files: dict[str, bytes]) -> io.BytesIO:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("export/annotations.json", json.dumps(coco))
        for name, data in files.items():
            bundle.writestr(name, data)
    buffer.seek(0)
    return buffer


def sample_coco():
    return {
        "images": [
            {"id": 7, "file_name": "street/a.png", "width": 40, "height": 30},
            {"id": 8, "file_name": "b.png", "width": 40, "height": 30},
            {"id": 9, "file_name": "not-in-zip.png", "width": 40, "height": 30},
        ],
        "categories": [{"id": 3, "name": "taxi"}, {"id": 11, "name": "bus"}],
        "annotations": [
            {"id": 1, "image_id": 7, "category_id": 3, "bbox": [1, 1, 10, 8],
             "segmentation": [[1, 1, 11, 1, 11, 9, 1, 9]], "iscrowd": 0},
            {"id": 2, "image_id": 7, "category_id": 11, "bbox": [20, 5, 15, 20], "iscrowd": 0},
            {"id": 3, "image_id": 8, "category_id": 3, "segmentation": rle_for(40, 30, (5, 4, 25, 20)),
             "bbox": [5, 4, 20, 16], "iscrowd": 0},
            {"id": 4, "image_id": 8, "category_id": 3, "bbox": [0, 0, 5, 5], "iscrowd": 1},
            {"id": 5, "image_id": 8, "category_id": 99, "bbox": [0, 0, 5, 5], "iscrowd": 0},
            {"id": 6, "image_id": 9, "category_id": 3, "bbox": [0, 0, 5, 5], "iscrowd": 0},
        ],
    }


def importer(tmp_path):
    annotations = AnnotationRevisionStore(tmp_path / "annotations.db")
    versions = DatasetVersionStore(tmp_path / "versions.db")
    return DatasetImporter(
        annotations=annotations, versions=versions, images_root=tmp_path / "dataset-images"
    ), annotations


def test_import_builds_an_editable_version(tmp_path):
    service, annotations = importer(tmp_path)
    archive = coco_zip(sample_coco(), {"street/a.png": png(), "b.png": png()})

    result = service.import_coco_zip(tenant_id=TENANT, archive=archive, name="Open Images taxis")

    dataset = result.record.dataset
    assert result.image_count == 2
    assert result.annotation_count == 3
    assert result.skipped_annotations == 3  # crowd, unknown category, image not in the ZIP
    assert result.missing_images == ("not-in-zip.png",)
    assert [category.name for category in dataset.categories] == ["taxi", "bus"]
    assert [category.category_id for category in dataset.categories] == [0, 1]
    assert {image.file_name for image in dataset.images} == {"images/a.png", "images/b.png"}
    assert all(item.annotation_id.startswith(f"{dataset.annotation_set}/") for item in dataset.annotations)
    bus = next(item for item in dataset.annotations if item.category_id == 1)
    assert bus.bbox == (20.0, 5.0, 15.0, 20.0) and bus.segmentation == ()
    rle = next(item for item in dataset.annotations if item.image_id.endswith("-2"))
    assert rle.segmentation and rle.bbox == (5.0, 4.0, 19.0, 15.0)
    folder = tmp_path / "dataset-images" / str(TENANT) / str(result.record.version_id) / "images"
    assert sorted(path.name for path in folder.iterdir()) == ["a.png", "b.png"]
    stored = annotations.list_latest(
        tenant_id=TENANT, image_ids=[image.image_id for image in dataset.images]
    )
    assert len(stored) == 3 and all(item.author_id == "dataset-import" for item in stored)


def test_two_imports_of_the_same_zip_stay_separate(tmp_path):
    service, _ = importer(tmp_path)
    files = {"street/a.png": png(), "b.png": png()}
    first = service.import_coco_zip(tenant_id=TENANT, archive=coco_zip(sample_coco(), files), name="One")
    second = service.import_coco_zip(tenant_id=TENANT, archive=coco_zip(sample_coco(), files), name="Two")

    assert first.record.version_id != second.record.version_id
    assert first.record.dataset.annotation_set != second.record.dataset.annotation_set
    assert not {image.image_id for image in first.record.dataset.images} & {
        image.image_id for image in second.record.dataset.images
    }


def test_rle_decoding_matches_pycocotools():
    rng = np.random.default_rng(4)
    mask = (rng.random((23, 31)) > 0.6).astype(np.uint8)
    encoded = mask_utils.encode(np.asfortranarray(mask))
    compressed = {"size": [23, 31], "counts": encoded["counts"].decode("ascii")}
    uncompressed = {"size": [23, 31], "counts": [int(v) for v in _runs(mask)]}

    assert np.array_equal(_decode_rle(compressed), mask.astype(bool))
    assert np.array_equal(_decode_rle(uncompressed), mask.astype(bool))


def _runs(mask):
    flat = mask.T.reshape(-1)
    runs, value, count = [], 0, 0
    for pixel in flat:
        if pixel == value:
            count += 1
        else:
            runs.append(count)
            value, count = pixel, 1
    runs.append(count)
    return runs


def test_paths_inside_the_zip_never_escape_the_version(tmp_path):
    service, _ = importer(tmp_path)
    coco = {
        "images": [{"id": 1, "file_name": "../../evil.png"}],
        "categories": [{"id": 1, "name": "taxi"}],
        "annotations": [],
    }
    result = service.import_coco_zip(
        tenant_id=TENANT, archive=coco_zip(coco, {"../../evil.png": png()}), name="Paths"
    )

    assert [image.file_name for image in result.record.dataset.images] == ["images/evil.png"]
    assert not (tmp_path / "evil.png").exists()
    assert not (tmp_path / "dataset-images" / "evil.png").exists()


def test_unusable_uploads_explain_why(tmp_path):
    service, _ = importer(tmp_path)

    def zipped(**files):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as bundle:
            for name, data in files.items():
                bundle.writestr(name.replace("__", "/"), data)
        buffer.seek(0)
        return buffer

    cases = [
        (io.BytesIO(b"not a zip"), "not a ZIP"),
        (zipped(**{"a.png": png()}), "no COCO .json"),
        (zipped(**{"notes.json": b"{}"}), "no .json file"),
        (zipped(**{"c.json": json.dumps({"images": [], "categories": []})}), "no categories"),
        (
            zipped(**{"c.json": json.dumps({"images": [{"id": 1, "file_name": "x.png"}],
                                            "categories": [{"id": 1, "name": "taxi"}]})}),
            "none of the images",
        ),
    ]
    for archive, message in cases:
        with pytest.raises(DatasetImportError, match=message):
            service.import_coco_zip(tenant_id=TENANT, archive=archive, name="Broken")


def test_imported_dataset_works_in_gallery_review_edit_and_export(tmp_path):
    app = create_local_app(LocalAppConfig(data_dir=tmp_path / "data", start_workers=True, worker_poll_interval=0.05))
    headers = {"X-Tenant-ID": str(TENANT)}
    with TestClient(app) as client:
        archive = coco_zip(sample_coco(), {"street/a.png": png(), "b.png": png()})
        imported = client.post(
            "/v1/datasets/import",
            headers=headers,
            data={"name": "Open Images taxis"},
            files={"file": ("taxis.zip", archive, "application/zip")},
        )
        assert imported.status_code == 201, imported.text
        body = imported.json()
        version_id = body["dataset"]["version_id"]
        assert body["image_count"] == 2 and body["annotation_count"] == 3
        assert body["missing_images"] == ["not-in-zip.png"]
        assert body["dataset"]["name"] == "Open Images taxis"

        listed = client.get("/v1/datasets", headers=headers).json()
        assert [item["version_id"] for item in listed] == [version_id]

        review = client.get(f"/v1/datasets/{version_id}/review", headers=headers).json()
        assert review["total"] == 2
        first = next(image for image in review["images"] if image["annotations"])
        assert client.get(
            f"/v1/datasets/{version_id}/images/{first['image_id']}/file", headers=headers
        ).status_code == 200

        target = first["annotations"][0]
        deleted = client.delete(
            f"/v1/datasets/{version_id}/images/{first['image_id']}/annotations/{target['annotation_id']}",
            headers=headers,
            params={"expected_revision": target["revision"]},
        )
        assert deleted.status_code == 200, deleted.text
        after = client.get(f"/v1/datasets/{version_id}/review", headers=headers).json()
        assert sum(len(image["annotations"]) for image in after["images"]) == 2

        export = client.post(
            "/v1/dataset-exports",
            headers=headers,
            json={"version_id": version_id, "format": "coco", "idempotency_key": "import-export"},
        )
        assert export.status_code == 202, export.text
        job_id = export.json()["job_id"]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            job = client.get(f"/v1/dataset-exports/{job_id}", headers=headers).json()
            if job["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.05)
        assert job["status"] == "succeeded", job
        download = client.get(f"/v1/dataset-exports/{job_id}/download", headers=headers)
        assert download.status_code == 200
        with zipfile.ZipFile(io.BytesIO(download.content)) as bundle:
            assert any(name.endswith(".png") for name in bundle.namelist())


def test_import_rejects_a_bad_upload_with_422(tmp_path):
    app = create_local_app(LocalAppConfig(data_dir=tmp_path / "data", start_workers=False))
    with TestClient(app) as client:
        response = client.post(
            "/v1/datasets/import",
            headers={"X-Tenant-ID": str(TENANT)},
            data={"name": "Broken"},
            files={"file": ("x.zip", io.BytesIO(b"nope"), "application/zip")},
        )
    assert response.status_code == 422
    assert "not a ZIP" in response.json()["detail"]
