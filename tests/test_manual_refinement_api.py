import json
from uuid import uuid4

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from agentic_cv.annotation_store import AnnotationRevisionStore
from agentic_cv.api import (
    ApiDependencies,
    DatasetApiDependencies,
    RefinementApiDependencies,
    create_app,
)
from agentic_cv.asset_registry import AssetRegistry
from agentic_cv.click_refinement import (
    ClickRefinementService,
    RefinementTarget,
)
from agentic_cv.dataset import Annotation, DatasetCategory, DatasetImage, DatasetVersion
from agentic_cv.dataset_builder import DatasetBuilder
from agentic_cv.dataset_export import DatasetExportService
from agentic_cv.dataset_store import DatasetVersionStore
from agentic_cv.job_store import JobStore
from agentic_cv.label_images import LabelImageResolver
from agentic_cv.object_storage import LocalObjectStorage
from agentic_cv.outbox import Outbox
from agentic_cv.sam3 import Sam3Prediction
from agentic_cv.uploads import ResumableUploadService


class FakePromptProvider:
    def refine_with_prompt(self, image, points, prompt):
        assert prompt == "worker"
        assert points[0].label == "positive"
        return [
            Sam3Prediction(
                score=0.8,
                box=(0, 0, 6, 6),
                mask=np.ones((6, 6), dtype=bool),
            )
        ]


def test_manual_refinement_accepts_revision_and_export_includes_it(tmp_path):
    tenant_id = uuid4()
    registry = AssetRegistry(str(tmp_path / "assets.sqlite3"))
    outbox = Outbox(str(tmp_path / "events.sqlite3"))
    annotations = AnnotationRevisionStore(str(tmp_path / "annotations.sqlite3"))
    versions = DatasetVersionStore(str(tmp_path / "versions.sqlite3"))
    jobs = JobStore(str(tmp_path / "jobs.sqlite3"))
    resolver = LabelImageResolver(registry=registry, data_dir=tmp_path / "data")
    image_root = tmp_path / "dataset-images"

    image = DatasetImage(
        image_id="frame-1",
        file_name="images/frame.png",
        width=6,
        height=6,
    )
    original = Annotation(
        annotation_id="auto-1",
        image_id="frame-1",
        category_id=0,
        bbox=(1, 1, 2, 2),
        segmentation=((1, 1, 3, 1, 3, 3, 1, 3),),
        score=0.9,
    )
    record = versions.create(
        tenant_id=tenant_id,
        name="workers-v1",
        dataset=DatasetVersion(
            images=(image,),
            categories=(DatasetCategory(category_id=0, name="worker"),),
            annotations=(original,),
        ),
        idempotency_key="dataset-1",
    )
    version_dir = image_root / str(tenant_id) / str(record.version_id) / "images"
    version_dir.mkdir(parents=True)
    Image.new("RGB", (6, 6), (255, 255, 255)).save(version_dir / "frame.png")
    annotations.save(
        tenant_id=tenant_id,
        annotation=original,
        author_id="auto-labeler",
        expected_revision=0,
    )

    service = ClickRefinementService(FakePromptProvider())

    def load_target(target_tenant, version_id, image_id):
        assert target_tenant == tenant_id
        assert version_id == record.version_id
        assert image_id == "frame-1"
        return RefinementTarget(
            image=Image.new("RGB", (6, 6)),
            prompt="worker",
            category_id=0,
        )

    builder = DatasetBuilder(
        jobs=jobs,
        annotations=annotations,
        versions=versions,
        images=resolver,
        images_root=image_root,
    )
    dependencies = ApiDependencies(
        registry=registry,
        uploads=ResumableUploadService(registry, LocalObjectStorage(tmp_path / "objects")),
        outbox=outbox,
        refinement=RefinementApiDependencies(
            service=service,
            annotations=annotations,
            dataset_loader=load_target,
        ),
        datasets=DatasetApiDependencies(
            builder=builder,
            versions=versions,
            images_root=image_root,
        ),
    )
    client = TestClient(create_app(dependencies))
    headers = {"X-Tenant-ID": str(tenant_id)}
    try:
        listing = client.get(f"/v1/datasets/{record.version_id}/images", headers=headers)
        assert listing.status_code == 200
        assert listing.json()["images"][0]["annotations"][0]["annotation_id"] == "auto-1"
        image_response = client.get(
            f"/v1/datasets/{record.version_id}/images/frame-1/file", headers=headers
        )
        assert image_response.status_code == 200
        assert image_response.headers["content-type"].startswith("image/png")

        with client.websocket_connect(
            f"/v1/refinement/{record.version_id}/frame-1?tenant_id={tenant_id}"
        ) as socket:
            initial = socket.receive_json()
            assert initial["category_id"] == 0
            assert initial["prompt"] == "worker"
            socket.send_json(
                {
                    "type": "click",
                    "expected_revision": 0,
                    "x": 2,
                    "y": 2,
                    "label": "positive",
                }
            )
            snapshot = socket.receive_json()
        assert snapshot["revision"] == 1
        assert snapshot["predictions"][0]["box"] == [0, 0, 6, 6]

        accepted = client.post(
            f"/v1/refinements/{snapshot['session_id']}/accept",
            headers=headers,
            json={
                "annotation_id": "auto-1",
                "prediction_index": 0,
                "refinement_revision": 1,
            },
        )
        assert accepted.status_code == 200, accepted.text
        assert accepted.json()["revision"] >= 2

        rebuilt = client.post(
            f"/v1/datasets/{record.version_id}/rebuild",
            headers=headers,
            json={"name": "workers-refined", "idempotency_key": "refine-1"},
        )
        assert rebuilt.status_code == 201, rebuilt.text
        refined_id = rebuilt.json()["version_id"]
        refined = versions.get(tenant_id=tenant_id, version_id=refined_id)
        assert refined.dataset.annotations[0].bbox == (0, 0, 6, 6)
        assert refined.dataset.annotations[0].score is None
        assert refined.dataset.annotations[0].segmentation

        export = DatasetExportService(
            versions,
            tmp_path / "exports",
            images_root=image_root,
        ).export(tenant_id=tenant_id, version_id=refined_id, format="coco")
        payload = json.loads((export.output_dir / "annotations.json").read_text())
        assert payload["annotations"][0]["bbox"] == [0, 0, 6, 6]
    finally:
        registry.close()
        outbox.close()
        annotations.close()
        versions.close()
        jobs.close()
