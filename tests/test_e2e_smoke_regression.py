"""Permanent end-to-end smoke regression: upload -> search/label -> VLM calibration -> export.

Exercises the main local workflow chain in-process with SQLite/local stores and
fake providers (no live services): resumable upload, semantic search indexing/query,
durable batch labeling, VLM-backed calibration, and dataset export.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from agentic_cv.annotation_store import AnnotationRevisionStore
from agentic_cv.asset_registry import AssetRegistry
from agentic_cv.batch_labeling import LabelImage, Sam3BatchLabeler
from agentic_cv.calibration import CalibrationSample, Sam3Calibrator
from agentic_cv.dataset import DatasetCategory
from agentic_cv.dataset_export import DatasetExportService
from agentic_cv.dataset_export_workflow import DatasetExportWorkflow, DatasetExportWorkflowDependencies
from agentic_cv.dataset_store import DatasetVersionStore
from agentic_cv.job_store import JobStore
from agentic_cv.label_workflow import DurableLabelWorkflow, LabelWorkflowDependencies
from agentic_cv.object_storage import LocalObjectStorage
from agentic_cv.outbox import Outbox
from agentic_cv.preview import PreviewStore
from agentic_cv.sam3 import Sam3Prediction
from agentic_cv.semantic_search import ImageRecord, LocalVectorIndex, SemanticSearchService
from agentic_cv.uploads import ResumableUploadService
from agentic_cv.vlm_validator import VlmCalibrationValidator, VlmValidator
from agentic_cv.annotation_sets import job_set_id, scoped_id


class FakeEmbeddingProvider:
    def embed_image(self, image):
        # Distinguish by dominant red channel so "red" queries find red images.
        arr = np.asarray(image)
        return [1.0, 0.0] if arr[:, :, 0].mean() > arr[:, :, 2].mean() else [0.0, 1.0]

    def embed_text(self, query):
        return [1.0, 0.0] if "red" in query.lower() else [0.0, 1.0]


class FakeSam3Provider:
    def predict(self, image, prompt, *, threshold=None, mask_threshold=None):
        score = float(threshold) if threshold is not None else 0.9
        mask = np.array([
            [False, False, False, False, False],
            [False, True, True, True, False],
            [False, True, True, True, False],
            [False, True, True, True, False],
            [False, False, False, False, False],
        ], dtype=bool)
        return [Sam3Prediction(score=min(score, 0.99), box=(1.0, 1.0, 4.0, 4.0), mask=mask)]


class FakeVlmClient:
    def __init__(self):
        self.calls = []

    def judge(self, *, overlay_uri, prompt):
        self.calls.append((overlay_uri, prompt))
        return {
            "target_present": True,
            "fully_detected": True,
            "false_positive": False,
            "mask_tight": True,
            "confidence": 0.95,
            "reasoning": "Target is present and well segmented.",
        }


def _save_image(path: Path, color=(220, 40, 40)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path)
    return path


def test_end_to_end_smoke_upload_search_label_vlm_calibration_export(tmp_path):
    tenant_id = uuid4()
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()

    # --- Upload ---
    registry = AssetRegistry(str(metadata_dir / "assets.sqlite3"))
    storage = LocalObjectStorage(tmp_path / "objects")
    uploads = ResumableUploadService(registry, storage)
    asset = registry.create_asset(tenant_id=tenant_id, name="smoke-asset")

    seed = tmp_path / "seed.png"
    _save_image(seed, color=(220, 40, 40))
    content = seed.read_bytes()
    digest = hashlib.sha256(content).hexdigest()

    upload = uploads.start(
        tenant_id=tenant_id,
        asset_id=asset.asset_id,
        filename="frame.png",
        media_type="image/png",
        expected_sha256=digest,
    )
    uploads.put_part(upload.upload_id, 1, content)
    completed = uploads.complete(upload.upload_id)
    assert completed.source.asset_id == asset.asset_id
    assert completed.source.content_sha256 == digest

    # Use the stored frame for downstream search/label steps.
    frame_path = tmp_path / "frame.png"
    frame_path.write_bytes(content)

    # --- Search ---
    search_service = SemanticSearchService(FakeEmbeddingProvider(), LocalVectorIndex())
    search_service.index_images([
        ImageRecord(record_id="frame-1", asset_id=str(asset.asset_id), image_path=frame_path)
    ])
    search_results = search_service.search("red object", asset_ids={str(asset.asset_id)}, limit=5)
    assert [r.record_id for r in search_results] == ["frame-1"]
    assert search_results[0].score > 0.9

    # --- Label (durable batch workflow) ---
    jobs = JobStore(str(metadata_dir / "jobs.sqlite3"))
    annotations = AnnotationRevisionStore(str(metadata_dir / "annotations.sqlite3"))
    outbox = Outbox(str(metadata_dir / "events.sqlite3"))
    preview = PreviewStore(str(metadata_dir / "preview.sqlite3"))
    versions = DatasetVersionStore(str(metadata_dir / "versions.sqlite3"))
    try:
        sam3 = FakeSam3Provider()
        label_workflow = DurableLabelWorkflow(LabelWorkflowDependencies(
            jobs=jobs,
            annotations=annotations,
            outbox=outbox,
            labeler=Sam3BatchLabeler(sam3),
            preview=preview,
        ))
        label_job = label_workflow.submit(
            tenant_id=tenant_id,
            asset_id=str(asset.asset_id),
            prompt="red object",
            threshold=0.6,
            task="detection",
            category_id=1,
            category_name="object",
            idempotency_key="smoke-label",
        )
        label_result = label_workflow.execute(
            label_job.job_id,
            images=[LabelImage(image_id="frame-1", path=frame_path)],
            tenant_id=tenant_id,
            prompt="red object",
            threshold=0.6,
            task="detection",
            category=DatasetCategory(category_id=1, name="object"),
        )
        assert jobs.get(label_job.job_id).status == "succeeded"
        assert len(label_result.dataset.annotations) == 1
        assert annotations.latest(tenant_id=tenant_id, image_id="frame-1", annotation_id=scoped_id(job_set_id(label_job.job_id), "frame-1:0")).revision == 1
        assert any(event.event_type == "LabelRunCompleted" for event in outbox.pending())

        # --- VLM calibration ---
        vlm_client = FakeVlmClient()
        vlm_validator = VlmCalibrationValidator(
            VlmValidator(vlm_client),
            lambda sample, predictions: f"overlay://{sample.image_id}/{len(predictions)}",
        )
        calibration_samples = [
            CalibrationSample(image_id="frame-1", image=Image.open(frame_path).convert("RGB"))
        ]
        calibration = Sam3Calibrator(
            sam3, vlm_validator, minimum_agreement=1.0, max_iterations=3
        ).calibrate(calibration_samples, prompt="red object", thresholds=[0.4, 0.6, 0.8])
        assert calibration.status == "calibrated"
        # One forward pass; the provider scores at the lowest threshold.
        assert calibration.forward_passes == 1
        assert calibration.threshold in (0.4, 0.6, 0.8)
        assert vlm_client.calls  # VLM judgments were actually requested

        # --- Export ---
        version = versions.create(
            tenant_id=tenant_id,
            name="smoke-export-v1",
            dataset=label_result.dataset,
            idempotency_key="smoke-version-1",
        )
        export_workflow = DatasetExportWorkflow(DatasetExportWorkflowDependencies(
            jobs=jobs,
            versions=versions,
            exports=DatasetExportService(versions, tmp_path / "exports"),
            outbox=outbox,
        ))
        export_job = export_workflow.submit(
            tenant_id=tenant_id,
            version_id=version.version_id,
            format="coco",
            idempotency_key="smoke-export-1",
        )
        export_result = export_workflow.execute(export_job.job_id, tenant_id=tenant_id)
        assert jobs.get(export_job.job_id).status == "succeeded"
        assert export_result.output_dir.joinpath("annotations.json").exists()
        assert any(event.event_type == "DatasetExportCompleted" for event in outbox.pending())
    finally:
        versions.close()
        preview.close()
        outbox.close()
        annotations.close()
        jobs.close()
        registry.close()
