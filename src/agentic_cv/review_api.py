"""Dataset review: live annotations, approval status, class edits, approved-only rebuilds.

A dataset version is immutable. Review works on its *annotation set*: the
latest live revisions (corrections, additions, tombstones), plus a review
status per image. ``rebuild-approved`` turns the reviewed set into a new
version containing only approved images.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

from .annotation_sets import in_set, live_in_set
from .annotation_store import AnnotationRevisionBackend, RevisionConflictError
from .api import _authorize, _credential, _tenant
from .auth import Authorizer
from .asset_registry import AssetRegistryBackend
from .dataset_builder import DatasetBuilder
from .label_images import image_display_names
from .dataset_store import DatasetVersionBackend, DatasetVersionRecord
from .review_store import REVIEW_STATUSES, ReviewBackend, ReviewStatus, review_set_key

StatusFilter = Literal["all", "pending", "approved", "rejected"]


class ReviewAnnotation(BaseModel):
    annotation_id: str
    category_id: int
    bbox: list[float]
    segmentation: list[list[float]]
    score: float | None
    revision: int
    author_id: str


class ReviewImage(BaseModel):
    image_id: str
    file_name: str
    source_name: str | None = None
    width: int
    height: int
    split: str
    status: ReviewStatus
    annotations: list[ReviewAnnotation]


class ReviewCounts(BaseModel):
    total: int
    pending: int
    approved: int
    rejected: int


class ReviewPage(BaseModel):
    version_id: UUID
    images: list[ReviewImage]
    counts: ReviewCounts
    total: int
    offset: int
    limit: int
    has_more: bool


class ReviewUpdateRequest(BaseModel):
    status: ReviewStatus
    image_ids: list[str] | None = Field(default=None, max_length=10000)
    # Apply to every image of the version (optionally only those currently
    # in ``only_status``), e.g. "approve all pending".
    all: bool = False
    only_status: ReviewStatus | None = None


class ReviewUpdateResponse(BaseModel):
    updated: int
    counts: ReviewCounts


class ClassChangeRequest(BaseModel):
    category_id: int
    expected_revision: int | None = Field(default=None, ge=0)


class RebuildApprovedRequest(BaseModel):
    idempotency_key: str = Field(min_length=1)
    name: str = ""
    include_pending: bool = False


class RebuildApprovedResponse(BaseModel):
    version_id: UUID
    name: str
    image_count: int
    annotation_count: int


@dataclass(frozen=True, kw_only=True)
class ReviewApiDependencies:
    versions: DatasetVersionBackend
    annotations: AnnotationRevisionBackend
    reviews: ReviewBackend
    builder: DatasetBuilder
    authorizer: Authorizer | None = None
    registry: AssetRegistryBackend | None = None


def create_app(dependencies: ReviewApiDependencies) -> FastAPI:
    app = FastAPI(title="BotFlow dataset review API", version="1")
    router = APIRouter()

    def authorized(x_tenant_id: str, authorization: str | None, x_api_key: str | None, role: str) -> UUID:
        tenant_id = _tenant(x_tenant_id)
        _authorize(
            dependencies.authorizer,
            credential=_credential(authorization, x_api_key),
            tenant_id=tenant_id,
            role=role,
        )
        return tenant_id

    def version(tenant_id: UUID, version_id: UUID) -> DatasetVersionRecord:
        try:
            return dependencies.versions.get(tenant_id=tenant_id, version_id=version_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    def set_key(record: DatasetVersionRecord) -> str:
        return review_set_key(record.dataset.annotation_set, record.version_id)

    def counts(record: DatasetVersionRecord, statuses: dict[str, ReviewStatus]) -> ReviewCounts:
        values = [statuses.get(image.image_id, "pending") for image in record.dataset.images]
        return ReviewCounts(
            total=len(values),
            pending=values.count("pending"),
            approved=values.count("approved"),
            rejected=values.count("rejected"),
        )

    @router.get("/v1/datasets/{version_id}/review", response_model=ReviewPage)
    def review_page(
        version_id: UUID,
        status: StatusFilter = Query(default="all"),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=200, ge=1, le=1000),
        x_tenant_id: str = Header(...),
        x_api_key: str | None = Header(None),
        authorization: str | None = Header(None),
    ) -> ReviewPage:
        """Images with their *live* annotations (corrections included) and review status."""

        tenant_id = authorized(x_tenant_id, authorization, x_api_key, "labels:read")
        record = version(tenant_id, version_id)
        statuses = dependencies.reviews.statuses(tenant_id=tenant_id, set_key=set_key(record))
        images = [
            image for image in record.dataset.images
            if status == "all" or statuses.get(image.image_id, "pending") == status
        ]
        page = images[offset : offset + limit]
        names = (
            image_display_names(
                dependencies.registry, tenant_id=tenant_id, image_ids=[image.image_id for image in page]
            )
            if dependencies.registry is not None
            else {}
        )
        category_ids = {category.category_id for category in record.dataset.categories}
        revisions = live_in_set(
            dependencies.annotations.list_latest(
                tenant_id=tenant_id, image_ids=[image.image_id for image in page]
            ),
            record.dataset.annotation_set,
            category_ids=category_ids,
        ) if page else []
        by_image: dict[str, list[ReviewAnnotation]] = {}
        for revision in revisions:
            annotation = revision.annotation
            by_image.setdefault(annotation.image_id, []).append(
                ReviewAnnotation(
                    annotation_id=annotation.annotation_id,
                    category_id=annotation.category_id,
                    bbox=list(annotation.bbox),
                    segmentation=[list(polygon) for polygon in annotation.segmentation],
                    score=annotation.score,
                    revision=revision.revision,
                    author_id=revision.author_id,
                )
            )
        return ReviewPage(
            version_id=version_id,
            images=[
                ReviewImage(
                    image_id=image.image_id,
                    file_name=image.file_name,
                    source_name=names.get(image.image_id),
                    width=image.width,
                    height=image.height,
                    split=record.dataset.split_of(image),
                    status=statuses.get(image.image_id, "pending"),
                    annotations=by_image.get(image.image_id, []),
                )
                for image in page
            ],
            counts=counts(record, statuses),
            total=len(images),
            offset=offset,
            limit=limit,
            has_more=offset + len(page) < len(images),
        )

    @router.post("/v1/datasets/{version_id}/review", response_model=ReviewUpdateResponse)
    def update_review(
        version_id: UUID,
        request: ReviewUpdateRequest,
        x_tenant_id: str = Header(...),
        x_api_key: str | None = Header(None),
        authorization: str | None = Header(None),
    ) -> ReviewUpdateResponse:
        """Approve, reject, or reset images: a list, or all (optionally one status)."""

        tenant_id = authorized(x_tenant_id, authorization, x_api_key, "labels:write")
        record = version(tenant_id, version_id)
        key = set_key(record)
        known = [image.image_id for image in record.dataset.images]
        if request.all:
            targets = known
            if request.only_status is not None:
                current = dependencies.reviews.statuses(tenant_id=tenant_id, set_key=key)
                targets = [
                    image_id for image_id in known
                    if current.get(image_id, "pending") == request.only_status
                ]
        else:
            if not request.image_ids:
                raise HTTPException(status_code=422, detail="give image_ids or set all=true")
            unknown = set(request.image_ids) - set(known)
            if unknown:
                raise HTTPException(status_code=404, detail="image is not part of this dataset")
            targets = request.image_ids
        updated = dependencies.reviews.set_status(
            tenant_id=tenant_id,
            set_key=key,
            image_ids=targets,
            status=request.status,
            author_id="reviewer",
        )
        statuses = dependencies.reviews.statuses(tenant_id=tenant_id, set_key=key)
        return ReviewUpdateResponse(updated=updated, counts=counts(record, statuses))

    @router.patch(
        "/v1/datasets/{version_id}/images/{image_id}/annotations/{annotation_id:path}",
        response_model=ReviewAnnotation,
    )
    def change_class(
        version_id: UUID,
        image_id: str,
        annotation_id: str,
        request: ClassChangeRequest,
        x_tenant_id: str = Header(...),
        x_api_key: str | None = Header(None),
        authorization: str | None = Header(None),
    ) -> ReviewAnnotation:
        """Assign an annotation to another class of the dataset (a new revision)."""

        tenant_id = authorized(x_tenant_id, authorization, x_api_key, "labels:write")
        record = version(tenant_id, version_id)
        if request.category_id not in {category.category_id for category in record.dataset.categories}:
            raise HTTPException(status_code=422, detail="category is not part of this dataset")
        if not any(image.image_id == image_id for image in record.dataset.images):
            raise HTTPException(status_code=404, detail="dataset image not found")
        if not in_set(annotation_id, record.dataset.annotation_set):
            raise HTTPException(status_code=404, detail="annotation not found")
        try:
            latest = dependencies.annotations.latest(
                tenant_id=tenant_id, image_id=image_id, annotation_id=annotation_id
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="annotation not found") from exc
        if latest.annotation.deleted:
            raise HTTPException(status_code=409, detail="annotation is deleted")
        try:
            saved = dependencies.annotations.save(
                tenant_id=tenant_id,
                annotation=replace(latest.annotation, category_id=request.category_id),
                author_id="manual-review",
                expected_revision=(
                    latest.revision if request.expected_revision is None else request.expected_revision
                ),
            )
        except RevisionConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        annotation = saved.annotation
        return ReviewAnnotation(
            annotation_id=annotation.annotation_id,
            category_id=annotation.category_id,
            bbox=list(annotation.bbox),
            segmentation=[list(polygon) for polygon in annotation.segmentation],
            score=annotation.score,
            revision=saved.revision,
            author_id=saved.author_id,
        )

    @router.post(
        "/v1/datasets/{version_id}/rebuild-approved",
        response_model=RebuildApprovedResponse,
        status_code=201,
    )
    def rebuild_approved(
        version_id: UUID,
        request: RebuildApprovedRequest,
        x_tenant_id: str = Header(...),
        x_api_key: str | None = Header(None),
        authorization: str | None = Header(None),
    ) -> RebuildApprovedResponse:
        """New version with the reviewed annotations of approved images only."""

        tenant_id = authorized(x_tenant_id, authorization, x_api_key, "labels:write")
        record = version(tenant_id, version_id)
        statuses = dependencies.reviews.statuses(tenant_id=tenant_id, set_key=set_key(record))
        allowed = {"approved", "pending"} if request.include_pending else {"approved"}
        keep = {
            image.image_id for image in record.dataset.images
            if statuses.get(image.image_id, "pending") in allowed
        }
        try:
            result = dependencies.builder.rebuild_from_version(
                tenant_id=tenant_id,
                version_id=version_id,
                name=request.name.strip() or f"{record.name} — approved",
                idempotency_key=request.idempotency_key,
                image_ids=keep,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return RebuildApprovedResponse(
            version_id=result.record.version_id,
            name=result.record.name,
            image_count=result.image_count,
            annotation_count=result.annotation_count,
        )

    app.include_router(router)
    return app


__all__ = ["REVIEW_STATUSES", "ReviewApiDependencies", "create_app"]
