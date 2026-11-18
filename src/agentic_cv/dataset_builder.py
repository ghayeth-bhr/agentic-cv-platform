"""Build immutable dataset versions from durable annotation revisions."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from PIL import Image

from .annotation_store import AnnotationRevisionBackend
from .batch_labeling import LabelImage
from .dataset import DatasetCategory, DatasetImage, DatasetVersion
from .dataset_store import DatasetVersionBackend, DatasetVersionRecord
from .job_store import JobStoreBackend
from .label_images import LabelImageResolver


@dataclass(frozen=True, kw_only=True)
class DatasetBuildResult:
    record: DatasetVersionRecord
    image_count: int
    annotation_count: int


class DatasetBuilder:
    """Build a dataset from the latest annotations for a succeeded label job.

    The label workflow persists annotations as revisions. Building from those
    revisions (instead of a transient in-memory result) means a restarted
    worker and later manual corrections both feed the same dataset path.
    """

    def __init__(
        self,
        *,
        jobs: JobStoreBackend,
        annotations: AnnotationRevisionBackend,
        versions: DatasetVersionBackend,
        images: LabelImageResolver,
        images_root: Path,
    ) -> None:
        self.jobs = jobs
        self.annotations = annotations
        self.versions = versions
        self.images = images
        self.images_root = images_root

    def build_from_job(
        self,
        *,
        tenant_id: UUID,
        job_id: UUID,
        name: str = "",
        idempotency_key: str,
    ) -> DatasetBuildResult:
        if not idempotency_key.strip():
            raise ValueError("idempotency_key must not be empty")
        job = self.jobs.get(job_id)
        if job.tenant_id != tenant_id:
            raise PermissionError("job does not belong to tenant")
        if job.kind != "batch-label":
            raise ValueError("dataset can only be built from a batch-label job")
        if job.status != "succeeded":
            raise ValueError(
                f"label job must be succeeded before building a dataset; current status is {job.status}"
            )

        payload = job.payload
        asset_id = str(payload["asset_id"])
        category_id = int(payload["category_id"])
        category = DatasetCategory(
            category_id=category_id,
            name=str(payload["category_name"]),
        )
        label_images = self.images.resolve(tenant_id=tenant_id, asset_id=asset_id)
        revisions = self.annotations.list_latest(
            tenant_id=tenant_id,
            image_ids=[image.image_id for image in label_images],
        )
        annotations = tuple(
            revision.annotation
            for revision in revisions
            if revision.annotation.category_id == category_id
        )

        dataset_images: list[DatasetImage] = []
        for label_image in label_images:
            with Image.open(label_image.path) as source:
                width, height = source.size
            file_name = Path(label_image.file_name or label_image.path.name).name
            dataset_images.append(
                DatasetImage(
                    image_id=label_image.image_id,
                    file_name=f"images/{file_name}",
                    width=width,
                    height=height,
                )
            )

        dataset = DatasetVersion(
            images=tuple(dataset_images),
            categories=(category,),
            annotations=annotations,
        )
        record = self.versions.create(
            tenant_id=tenant_id,
            name=name.strip() or f"Label run {job_id}",
            dataset=dataset,
            idempotency_key=idempotency_key,
        )
        self._copy_images(
            tenant_id=tenant_id,
            version_id=record.version_id,
            label_images=label_images,
        )
        return DatasetBuildResult(
            record=record,
            image_count=len(dataset_images),
            annotation_count=len(annotations),
        )

    def _copy_images(
        self,
        *,
        tenant_id: UUID,
        version_id: UUID,
        label_images: list[LabelImage],
    ) -> None:
        target_dir = self.images_root / str(tenant_id) / str(version_id) / "images"
        target_dir.mkdir(parents=True, exist_ok=True)
        for label_image in label_images:
            file_name = Path(label_image.file_name or label_image.path.name).name
            shutil.copy2(label_image.path, target_dir / file_name)
