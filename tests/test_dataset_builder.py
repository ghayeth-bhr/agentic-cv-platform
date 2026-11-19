import hashlib
import json
import zipfile
from uuid import uuid4

import numpy as np
from PIL import Image

from agentic_cv.annotation_store import AnnotationRevisionStore
from agentic_cv.asset_registry import AssetRegistry
from agentic_cv.batch_labeling import Sam3BatchLabeler
from agentic_cv.dataset import Annotation, DatasetCategory
from agentic_cv.dataset_builder import DatasetBuilder
from agentic_cv.dataset_export import DatasetExportService
from agentic_cv.dataset_store import DatasetVersionStore
from agentic_cv.job_store import JobStore
from agentic_cv.label_images import LabelImageResolver
from agentic_cv.label_workflow import DurableLabelWorkflow, LabelWorkflowDependencies
from agentic_cv.outbox import Outbox
from agentic_cv.preview import PreviewStore
from agentic_cv.sam3 import Sam3Prediction
from agentic_cv.annotation_sets import job_set_id, scoped_id


class FakeProvider:
    def predict(self, image, prompt, *, threshold=None, mask_threshold=None):
        return [
            Sam3Prediction(
                score=0.9,
                box=(1, 1, 5, 5),
                mask=np.ones((6, 6), dtype=bool),
            )
        ]


def test_builder_uses_latest_revisions_and_exports_images(tmp_path):
    tenant_id = uuid4()
    registry = AssetRegistry(str(tmp_path / "assets.sqlite3"))
    annotations = AnnotationRevisionStore(str(tmp_path / "annotations.sqlite3"))
    jobs = JobStore(str(tmp_path / "jobs.sqlite3"))
    versions = DatasetVersionStore(str(tmp_path / "versions.sqlite3"))
    outbox = Outbox(str(tmp_path / "events.sqlite3"))
    preview = PreviewStore(str(tmp_path / "preview.sqlite3"))
    workflow = DurableLabelWorkflow(
        LabelWorkflowDependencies(
            jobs=jobs,
            annotations=annotations,
            outbox=outbox,
            labeler=Sam3BatchLabeler(FakeProvider()),
            preview=preview,
        )
    )
    resolver = LabelImageResolver(registry=registry, data_dir=tmp_path / "data")
    asset = registry.create_asset(tenant_id=tenant_id, name="photos")
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (6, 6), (255, 255, 255)).save(image_path)
    registry.add_source(
        tenant_id=tenant_id,
        asset_id=asset.asset_id,
        filename=image_path.name,
        media_type="image/png",
        object_uri=image_path.resolve().as_uri(),
        content_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
    )
    images = resolver.resolve(tenant_id=tenant_id, asset_id=str(asset.asset_id))
    job = workflow.submit(
        tenant_id=tenant_id,
        asset_id=str(asset.asset_id),
        prompt="worker",
        threshold=0.5,
        task="detection",
        category_id=0,
        category_name="worker",
        idempotency_key="run-1",
    )
    workflow.execute(
        job.job_id,
        images=images,
        tenant_id=tenant_id,
        prompt="worker",
        threshold=0.5,
        task="detection",
        category=DatasetCategory(category_id=0, name="worker"),
    )
    annotation = Annotation(
        annotation_id=scoped_id(job_set_id(job.job_id), f"{images[0].image_id}:0"),
        image_id=images[0].image_id,
        category_id=0,
        bbox=(1, 1, 4, 4),
        score=0.95,
    )
    annotations.save(
        tenant_id=tenant_id,
        annotation=annotation,
        author_id="reviewer",
        expected_revision=1,
    )
    builder = DatasetBuilder(
        jobs=jobs,
        annotations=annotations,
        versions=versions,
        images=resolver,
        images_root=tmp_path / "dataset-images",
    )

    result = builder.build_from_job(
        tenant_id=tenant_id,
        job_id=job.job_id,
        name="workers-v1",
        idempotency_key="dataset-1",
    )

    assert result.image_count == 1
    assert result.annotation_count == 1
    assert result.record.dataset.images[0].file_name.startswith("images/")
    assert result.record.dataset.annotations[0].bbox == (1, 1, 4, 4)
    assert (tmp_path / "dataset-images" / str(tenant_id) / str(result.record.version_id) / "images").is_dir()

    export = DatasetExportService(
        versions,
        tmp_path / "exports",
        images_root=tmp_path / "dataset-images",
    ).export(tenant_id=tenant_id, version_id=result.record.version_id, format="coco")
    payload = json.loads((export.output_dir / "annotations.json").read_text())
    assert payload["annotations"][0]["bbox"] == [1, 1, 4, 4]
    assert any(path.suffix == ".png" for path in export.files)
    with zipfile.ZipFile(export.archive_path) as bundle:
        names = set(bundle.namelist())
    assert "annotations.json" in names
    assert any(name.startswith("images/") for name in names)
    assert "manifest.json" in names
    registry.close(); annotations.close(); jobs.close(); versions.close(); outbox.close(); preview.close()
