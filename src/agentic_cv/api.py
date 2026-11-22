"""Local HTTP API composing assets, uploads, workflows, and orchestration."""

from __future__ import annotations

import io
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Literal
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname
from uuid import UUID

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, Field

from .annotation_sets import in_set, job_set_id, live_in_set, manual_annotation_id
from .annotation_store import AnnotationRevisionBackend, RevisionConflictError
from .asset_registry import AssetRegistry
from .auth import Authorizer
from .batch_labeling import mask_polygons
from .calibration import CalibrationSample, CalibrationStatus, Sam3Calibrator
from .click_refinement import (
    ClickPoint,
    ClickRefinementService,
    RefinementSnapshot,
    RefinementTarget,
)
from .contracts import EventEnvelope, TaskSpec
from .dataset import Annotation
from .dataset_builder import DatasetBuilder
from .dataset_import import DatasetImportError, DatasetImporter
from .dataset_export import ExportFormat
from .dataset_export_workflow import DatasetExportWorkflow
from .dataset_store import DatasetVersionBackend, DatasetVersionRecord
from .errors import ModelUnavailableError, install_model_unavailable_handler
from .job_store import JobRecord, JobStoreBackend
from .label_images import LabelImageResolver, image_display_names
from .label_workflow import DurableLabelWorkflow, label_plan
from .orchestrator import TaskOrchestrationRequest, TaskOrchestrator
from .outbox import Outbox
from .overlay import class_color, render_annotation_overlay
from .sam3 import Sam3Prediction, Sam3Provider
from .uploads import ResumableUploadService


class AssetCreateRequest(BaseModel):
    name: str


class AssetResponse(BaseModel):
    asset_id: UUID
    tenant_id: UUID
    name: str
    created_at: str


class UploadCreateRequest(BaseModel):
    filename: str
    media_type: str
    expected_sha256: str


class UploadResponse(BaseModel):
    upload_id: UUID
    asset_id: UUID
    object_key: str
    status: str


class SourceResponse(BaseModel):
    source_id: UUID
    asset_id: UUID
    object_uri: str
    content_sha256: str


class SourceInfo(BaseModel):
    source_id: UUID
    asset_id: UUID
    filename: str
    media_type: str
    content_sha256: str
    created_at: str


class AssetSummaryResponse(BaseModel):
    asset_id: UUID
    tenant_id: UUID
    name: str
    created_at: str
    sources: list[SourceInfo]
    thumbnail_url: str | None = None


class AssetListResponse(BaseModel):
    assets: list[AssetSummaryResponse]


class DatasetExportRequest(BaseModel):
    version_id: UUID
    format: ExportFormat
    idempotency_key: str


class DatasetExportJobResponse(BaseModel):
    job_id: UUID
    tenant_id: UUID
    status: str
    kind: str
    attempts: int
    error: str | None


class JobSummaryResponse(BaseModel):
    job_id: UUID
    kind: str
    status: str
    attempts: int
    error: str | None
    created_at: str
    updated_at: str


class JobListResponse(BaseModel):
    jobs: list[JobSummaryResponse]


@dataclass(frozen=True, kw_only=True)
class RefinementApiDependencies:
    service: ClickRefinementService
    image_loader: Callable[[str], Image.Image] | None = None
    annotations: AnnotationRevisionBackend | None = None
    dataset_loader: Callable[[UUID, UUID, str], RefinementTarget] | None = None


@dataclass(frozen=True, kw_only=True)
class ExportApiDependencies:
    workflow: DatasetExportWorkflow


@dataclass(frozen=True, kw_only=True)
class LabelApiDependencies:
    workflow: DurableLabelWorkflow


@dataclass(frozen=True, kw_only=True)
class DatasetApiDependencies:
    builder: DatasetBuilder
    versions: DatasetVersionBackend
    images_root: Path | None = None
    importer: DatasetImporter | None = None


@dataclass(frozen=True, kw_only=True)
class CalibrationApiDependencies:
    provider: Sam3Provider


def _tenant(value: str) -> UUID:
    try:
        return UUID(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="X-Tenant-ID must be a UUID") from exc


@dataclass(frozen=True, kw_only=True)
class ApiDependencies:
    registry: AssetRegistry
    uploads: ResumableUploadService
    outbox: Outbox
    refinement: RefinementApiDependencies | None = None
    authorizer: Authorizer | None = None
    exports: ExportApiDependencies | None = None
    labels: LabelApiDependencies | None = None
    datasets: DatasetApiDependencies | None = None
    calibration: CalibrationApiDependencies | None = None
    orchestrator: TaskOrchestrator | None = None
    sam3: Sam3Provider | None = None
    jobs: JobStoreBackend | None = None
    label_images: LabelImageResolver | None = None
    annotations: AnnotationRevisionBackend | None = None
    # Directory for generated thumbnails (sources are immutable, so a
    # thumbnail is keyed by source id). None regenerates on every request.
    thumbnail_cache: Path | None = None


def _authorize(
    authorizer: Authorizer | None,
    *,
    credential: str | None,
    tenant_id: UUID,
    role: str,
) -> None:
    if authorizer is None:
        return
    try:
        authorizer.authorize(credential=credential, tenant_id=tenant_id, role=role)
    except PermissionError as exc:
        status = 401 if credential is None else 403
        raise HTTPException(status_code=status, detail=str(exc)) from exc


class OrchestrateRequest(BaseModel):
    message: str = ""
    task_type: Literal["detection", "segmentation"] | None = None
    target_objects: list[str] = Field(default_factory=list)
    visual_attributes: list[str] = Field(default_factory=list)
    sample_size: int = 5
    max_iterations: int = 5


class TaskSpecResponse(BaseModel):
    target_objects: tuple[str, ...]
    task_type: Literal["detection", "segmentation"]
    visual_attributes: tuple[str, ...]
    sample_size: int
    max_iterations: int


class LabelJobRequest(BaseModel):
    asset_id: str
    prompt: str
    threshold: float = 0.5
    task: Literal["detection", "segmentation"] = "detection"
    category_id: int = 1
    category_name: str = "object"
    idempotency_key: str


class LabelJobResponse(BaseModel):
    job_id: UUID
    kind: str
    status: str
    created_at: str
    updated_at: str
    lease_owner: str | None
    cancel_requested: bool
    payload: dict


class DatasetCreateRequest(BaseModel):
    job_id: UUID
    name: str = ""
    idempotency_key: str


class DatasetCategoryResponse(BaseModel):
    category_id: int
    name: str


class DatasetClassCountResponse(BaseModel):
    category_id: int
    name: str
    count: int
    color: str


class DatasetSummaryResponse(BaseModel):
    version_id: UUID
    name: str
    image_count: int
    annotation_count: int
    categories: list[DatasetCategoryResponse]
    created_at: str
    source_asset: str = ""
    status: str = "ready"
    class_counts: list[DatasetClassCountResponse] = Field(default_factory=list)


class DatasetAnnotationResponse(BaseModel):
    annotation_id: str
    category_id: int
    bbox: list[float]
    segmentation: list[list[float]]
    score: float | None


class DatasetImageResponse(BaseModel):
    image_id: str
    file_name: str
    source_name: str | None = None
    width: int
    height: int
    annotations: list[DatasetAnnotationResponse]
    split: str = "train"
    file_size: int = 0


class DatasetImagesResponse(BaseModel):
    images: list[DatasetImageResponse]
    total: int = 0
    limit: int = 0
    offset: int = 0
    has_more: bool = False


class DatasetImportResponse(BaseModel):
    dataset: DatasetSummaryResponse
    image_count: int
    annotation_count: int
    skipped_annotations: int
    missing_images: list[str]


class DatasetRebuildRequest(BaseModel):
    name: str = ""
    idempotency_key: str


class RefinementAcceptRequest(BaseModel):
    annotation_id: str | None = None
    # Class for the accepted object; defaults to the session's class.
    category_id: int | None = None
    prediction_index: int = 0
    refinement_revision: int | None = None
    expected_revision: int | None = None


class AnnotationDeleteResponse(BaseModel):
    image_id: str
    annotation_id: str
    revision: int
    deleted: bool = True


class RefinementAcceptResponse(BaseModel):
    session_id: UUID
    image_id: str
    annotation_id: str
    revision: int


class CalibrationAttemptResponse(BaseModel):
    threshold: float
    sample_count: int
    accepted_count: int
    agreement: float


class CalibrationResponse(BaseModel):
    status: CalibrationStatus
    prompt: str
    threshold: float | None
    attempts: list[CalibrationAttemptResponse]
    reason: str | None = None


class AssetPredictRequest(BaseModel):
    prompt: str
    threshold: float | None = None
    source_id: UUID | None = None


class Sam3PredictionResponse(BaseModel):
    score: float
    box: tuple[float, float, float, float]
    mask: list[list[bool]]


class Sam3PredictResponse(BaseModel):
    prompt: str
    predictions: list[Sam3PredictionResponse]
    image_url: str | None = None
    image_width: int | None = None
    image_height: int | None = None


def _task_spec_response(spec: TaskSpec) -> TaskSpecResponse:
    return TaskSpecResponse(
        target_objects=spec.target_objects,
        task_type=spec.task_type,
        visual_attributes=spec.visual_attributes,
        sample_size=spec.sample_size,
        max_iterations=spec.max_iterations,
    )


def _label_job_response(job: JobRecord) -> LabelJobResponse:
    return LabelJobResponse(
        job_id=job.job_id,
        kind=job.kind,
        status=job.status,
        created_at=job.created_at.isoformat() if hasattr(job.created_at, "isoformat") else str(job.created_at),
        updated_at=job.updated_at.isoformat() if hasattr(job.updated_at, "isoformat") else str(job.updated_at),
        lease_owner=job.lease_owner,
        cancel_requested=job.cancel_requested,
        payload=job.payload,
    )


def _color_hex(category_id: int) -> str:
    red, green, blue = class_color(category_id)
    return f"#{red:02x}{green:02x}{blue:02x}"


def _dataset_source(
    record: DatasetVersionRecord, asset_names: dict[str, str] | None
) -> str:
    """Name the source asset(s) from recorded ids; parse the name for old versions."""

    ids = record.dataset.source_asset_ids
    if ids and asset_names is not None:
        names = [asset_names[asset_id] for asset_id in ids if asset_id in asset_names]
        if names:
            return ", ".join(names)
    return _source_asset_name(record.name)


def _source_asset_name(name: str) -> str:
    for separator in (" — ", " – ", " - "):
        if separator in name:
            return name.split(separator, 1)[0].strip()
    return name


def _dataset_summary(
    record: DatasetVersionRecord, asset_names: dict[str, str] | None = None
) -> DatasetSummaryResponse:
    categories = {
        category.category_id: category.name for category in record.dataset.categories
    }
    counts: dict[int, int] = {}
    for annotation in record.dataset.annotations:
        counts[annotation.category_id] = counts.get(annotation.category_id, 0) + 1
    class_counts = [
        DatasetClassCountResponse(
            category_id=category_id,
            name=categories.get(category_id, f"class {category_id}"),
            count=count,
            color=_color_hex(category_id),
        )
        for category_id, count in sorted(counts.items())
    ]
    return DatasetSummaryResponse(
        version_id=record.version_id,
        name=record.name,
        image_count=len(record.dataset.images),
        annotation_count=len(record.dataset.annotations),
        categories=[
            DatasetCategoryResponse(category_id=category.category_id, name=category.name)
            for category in record.dataset.categories
        ],
        created_at=record.created_at,
        source_asset=_dataset_source(record, asset_names),
        status="ready",
        class_counts=class_counts,
    )



def _dataset_image_path(
    images_root: Path,
    *,
    tenant_id: UUID,
    version_id: UUID,
    file_name: str,
) -> Path:
    root = (images_root / str(tenant_id) / str(version_id)).resolve()
    path = (root / file_name).resolve()
    if path != root and root not in path.parents:
        raise HTTPException(status_code=400, detail="invalid dataset image path")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="dataset image not found")
    return path


def _snapshot_payload(snapshot: RefinementSnapshot, *, polygons: bool = False) -> dict:
    """Refinement state for the browser.

    ``polygons=True`` sends each mask as simplified polygons (a few KB)
    instead of a full boolean grid (several MB for a 720x1280 frame), so a
    0.03 s point-prompt click is not dominated by JSON transfer.
    """

    if polygons:
        return {
            "type": "snapshot",
            "session_id": str(snapshot.session_id),
            "image_id": snapshot.image_id,
            "revision": snapshot.revision,
            "prompt": snapshot.prompt,
            "category_id": snapshot.category_id,
            "points": [
                {"x": point.x, "y": point.y, "label": point.label}
                for point in snapshot.points
            ],
            "predictions": [
                {
                    "score": prediction.score,
                    "box": prediction.box,
                    "polygons": [list(polygon) for polygon in mask_polygons(prediction.mask)],
                }
                for prediction in snapshot.predictions
            ],
        }
    return {
        "type": "snapshot",
        "session_id": str(snapshot.session_id),
        "image_id": snapshot.image_id,
        "revision": snapshot.revision,
        "prompt": snapshot.prompt,
        "category_id": snapshot.category_id,
        "points": [
            {"x": point.x, "y": point.y, "label": point.label}
            for point in snapshot.points
        ],
        "predictions": [
            {
                "score": prediction.score,
                "box": prediction.box,
                "mask": prediction.mask.astype(bool).tolist(),
            }
            for prediction in snapshot.predictions
        ],
    }


def _asset_names(registry, tenant_id: UUID) -> dict[str, str]:
    try:
        return {
            str(asset.asset_id): asset.name
            for asset in registry.list_assets(tenant_id=tenant_id)
        }
    except Exception:
        return {}


def _credential(authorization: str | None, api_key: str | None) -> str | None:
    return authorization or api_key


def _job_response(job: JobRecord) -> DatasetExportJobResponse:
    return DatasetExportJobResponse(
        job_id=job.job_id, tenant_id=job.tenant_id, status=job.status,
        kind=job.kind, attempts=job.attempts, error=job.error,
    )


def _path_from_file_uri(uri: str) -> Path | None:
    try:
        parsed = urlparse(uri)
    except Exception:
        return None
    if parsed.scheme != "file":
        return None
    return Path(url2pathname(unquote(parsed.path)))


def _representative_source(sources):
    if not sources:
        return None
    for source in sources:
        if (source.media_type or "").startswith("image/"):
            return source
    for source in sources:
        if (source.media_type or "").startswith("video/"):
            return source
    return sources[0]


def _thumbnail_bytes(source) -> bytes | None:
    path = _path_from_file_uri(source.object_uri)
    if path is None or not path.exists():
        return None
    media_type = (source.media_type or "").lower()
    try:
        if media_type.startswith("image/"):
            with Image.open(path) as img:
                img = img.convert("RGB")
                img.thumbnail((512, 512))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=82)
                return buf.getvalue()
        if media_type.startswith("video/"):
            import cv2  # type: ignore

            capture = cv2.VideoCapture(str(path))
            try:
                ok, frame = capture.read()
            finally:
                capture.release()
            if not ok or frame is None:
                return None
            frame = frame[:, :, ::-1]
            img = Image.fromarray(frame)
            img.thumbnail((512, 512))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=82)
            return buf.getvalue()
    except Exception:
        return None
    return None


def _cached_thumbnail(cache_dir: Path | None, source) -> bytes | None:
    """Thumbnail bytes, generated once per source (videos are decoded once)."""

    if cache_dir is None:
        return _thumbnail_bytes(source)
    path = cache_dir / f"{source.source_id}.jpg"
    try:
        return path.read_bytes()
    except OSError:
        pass
    data = _thumbnail_bytes(source)
    if data is not None:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        except OSError:
            pass
    return data


def _load_source_image(source) -> Image.Image | None:
    path = _path_from_file_uri(source.object_uri)
    if path is None or not path.exists():
        return None
    media_type = (source.media_type or "").lower()
    try:
        if media_type.startswith("image/"):
            with Image.open(path) as img:
                return img.convert("RGB")
        if media_type.startswith("video/"):
            import cv2  # type: ignore

            capture = cv2.VideoCapture(str(path))
            try:
                ok, frame = capture.read()
            finally:
                capture.release()
            if not ok or frame is None:
                return None
            frame = frame[:, :, ::-1]
            return Image.fromarray(frame).convert("RGB")
    except Exception:
        return None
    return None


def _source_image_bytes(source) -> tuple[bytes, str] | None:
    path = _path_from_file_uri(source.object_uri)
    if path is None or not path.exists():
        return None
    media_type = (source.media_type or "").lower()
    try:
        if media_type.startswith("image/"):
            data = path.read_bytes()
            if data:
                return data, (source.media_type or "image/jpeg")
            return None
        if media_type.startswith("video/"):
            import cv2  # type: ignore

            capture = cv2.VideoCapture(str(path))
            try:
                ok, frame = capture.read()
            finally:
                capture.release()
            if not ok or frame is None:
                return None
            frame = frame[:, :, ::-1]
            img = Image.fromarray(frame)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=92)
            return buf.getvalue(), "image/jpeg"
    except Exception:
        return None
    return None


def _prediction_response(
    prompt: str,
    predictions: list[Sam3Prediction],
    *,
    image_url: str | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
) -> Sam3PredictResponse:
    return Sam3PredictResponse(
        prompt=prompt,
        predictions=[
            Sam3PredictionResponse(score=prediction.score, box=prediction.box, mask=prediction.mask.astype(bool).tolist())
            for prediction in predictions
        ],
        image_url=image_url,
        image_width=image_width,
        image_height=image_height,
    )


def create_app(dependencies: ApiDependencies) -> FastAPI:
    app = FastAPI(title="BotFlow API", version="1")
    install_model_unavailable_handler(app)

    @app.post("/v1/assets", response_model=AssetResponse, status_code=201)
    def create_asset(
        request: AssetCreateRequest,
        x_tenant_id: str = Header(...),
        x_api_key: str | None = Header(None),
        authorization: str | None = Header(None),
    ) -> AssetResponse:
        tenant_id = _tenant(x_tenant_id)
        _authorize(dependencies.authorizer, credential=_credential(authorization, x_api_key), tenant_id=tenant_id, role="assets:write")
        asset = dependencies.registry.create_asset(
            tenant_id=tenant_id, name=request.name
        )
        return AssetResponse(
            asset_id=asset.asset_id,
            tenant_id=asset.tenant_id,
            name=asset.name,
            created_at=asset.created_at,
        )

    @app.get("/v1/assets", response_model=AssetListResponse)
    def list_assets(
        x_tenant_id: str | None = Header(None),
        tenant_id_q: str | None = Query(None, alias="tenant_id"),
        x_api_key: str | None = Header(None),
        authorization: str | None = Header(None),
    ) -> AssetListResponse:
        raw_tenant = x_tenant_id or tenant_id_q
        if raw_tenant is None:
            raise HTTPException(status_code=400, detail="X-Tenant-ID or tenant_id is required")
        tenant_id = _tenant(raw_tenant)
        _authorize(dependencies.authorizer, credential=_credential(authorization, x_api_key), tenant_id=tenant_id, role="assets:read")
        summaries = dependencies.registry.list_assets(tenant_id=tenant_id)
        assets: list[AssetSummaryResponse] = []
        for summary in summaries:
            representative = _representative_source(summary.sources)
            thumbnail_url = (
                f"/v1/assets/{summary.asset_id}/thumbnail?tenant_id={tenant_id}"
                if representative is not None
                else None
            )
            assets.append(AssetSummaryResponse(
                asset_id=summary.asset_id,
                tenant_id=summary.tenant_id,
                name=summary.name,
                created_at=summary.created_at,
                sources=[SourceInfo(
                    source_id=source.source_id,
                    asset_id=source.asset_id,
                    filename=source.filename,
                    media_type=source.media_type,
                    content_sha256=source.content_sha256,
                    created_at=source.created_at,
                ) for source in summary.sources],
                thumbnail_url=thumbnail_url,
            ))
        return AssetListResponse(assets=assets)

    @app.get("/v1/assets/{asset_id}/thumbnail")
    def asset_thumbnail(
        asset_id: UUID,
        x_tenant_id: str | None = Header(None),
        tenant_id_q: str | None = Query(None, alias="tenant_id"),
        x_api_key: str | None = Header(None),
        authorization: str | None = Header(None),
    ) -> Response:
        raw_tenant = x_tenant_id or tenant_id_q
        if raw_tenant is None:
            raise HTTPException(status_code=400, detail="X-Tenant-ID or tenant_id is required")
        tenant_id = _tenant(raw_tenant)
        _authorize(dependencies.authorizer, credential=_credential(authorization, x_api_key), tenant_id=tenant_id, role="assets:read")
        summaries = dependencies.registry.list_assets(tenant_id=tenant_id)
        summary = next((item for item in summaries if item.asset_id == asset_id), None)
        if summary is None:
            raise HTTPException(status_code=404, detail="unknown asset")
        representative = _representative_source(summary.sources)
        if representative is None:
            raise HTTPException(status_code=404, detail="asset has no source")
        data = _cached_thumbnail(dependencies.thumbnail_cache, representative)
        if data is None:
            raise HTTPException(status_code=404, detail="thumbnail unavailable")
        return Response(
            content=data,
            media_type="image/jpeg",
            headers={"Cache-Control": "private, max-age=86400", "ETag": f'"{representative.source_id}"'},
        )

    @app.post("/v1/assets/{asset_id}/uploads", response_model=UploadResponse, status_code=201)
    def start_upload(
        asset_id: UUID,
        request: UploadCreateRequest,
        x_tenant_id: str = Header(...),
        x_api_key: str | None = Header(None),
        authorization: str | None = Header(None),
    ) -> UploadResponse:
        try:
            tenant_id = _tenant(x_tenant_id)
            _authorize(dependencies.authorizer, credential=_credential(authorization, x_api_key), tenant_id=tenant_id, role="uploads:write")
            upload = dependencies.uploads.start(
                tenant_id=tenant_id, asset_id=asset_id,
                filename=request.filename, media_type=request.media_type,
                expected_sha256=request.expected_sha256,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return UploadResponse(
            upload_id=upload.upload_id, asset_id=upload.asset_id,
            object_key=upload.object_key, status=upload.status,
        )

    @app.put("/v1/uploads/{upload_id}/parts/{part_number}")
    async def put_part(
        upload_id: UUID,
        part_number: int,
        part: UploadFile = File(...),
        x_tenant_id: str | None = Header(None),
        x_api_key: str | None = Header(None),
        authorization: str | None = Header(None),
    ) -> dict[str, int]:
        try:
            upload = dependencies.uploads.registry.get_upload(upload_id)
            if dependencies.authorizer is not None:
                if x_tenant_id is None:
                    raise HTTPException(status_code=400, detail="X-Tenant-ID is required")
                tenant_id = _tenant(x_tenant_id)
                _authorize(dependencies.authorizer, credential=_credential(authorization, x_api_key), tenant_id=tenant_id, role="uploads:write")
                if upload.tenant_id != tenant_id:
                    raise PermissionError("upload does not belong to tenant")
            dependencies.uploads.put_part(upload_id, part_number, await part.read())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"part_number": part_number}

    @app.post("/v1/uploads/{upload_id}/complete", response_model=SourceResponse)
    def complete_upload(
        upload_id: UUID,
        x_tenant_id: str | None = Header(None),
        x_api_key: str | None = Header(None),
        authorization: str | None = Header(None),
    ) -> SourceResponse:
        try:
            upload = dependencies.uploads.registry.get_upload(upload_id)
            if dependencies.authorizer is not None:
                if x_tenant_id is None:
                    raise HTTPException(status_code=400, detail="X-Tenant-ID is required")
                tenant_id = _tenant(x_tenant_id)
                _authorize(dependencies.authorizer, credential=_credential(authorization, x_api_key), tenant_id=tenant_id, role="uploads:write")
                if upload.tenant_id != tenant_id:
                    raise PermissionError("upload does not belong to tenant")
            completed = dependencies.uploads.complete(upload_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        source = completed.source
        dependencies.outbox.enqueue(
            EventEnvelope(
                event_type="SourceReady",
                tenant_id=source.tenant_id,
                aggregate_type="source",
                aggregate_id=source.source_id,
                payload={
                    "asset_id": str(source.asset_id),
                    "object_uri": source.object_uri,
                    "content_sha256": source.content_sha256,
                },
                idempotency_key=f"source-ready:{source.source_id}",
            )
        )
        return SourceResponse(
            source_id=source.source_id,
            asset_id=source.asset_id,
            object_uri=source.object_uri,
            content_sha256=source.content_sha256,
        )

    if dependencies.jobs is not None:
        @app.get("/v1/jobs", response_model=JobListResponse)
        def list_jobs(
            status: str | None = Query(None),
            kind: str | None = Query(None),
            x_tenant_id: str = Header(...),
            x_api_key: str | None = Header(None),
            authorization: str | None = Header(None),
        ) -> JobListResponse:
            tenant_id = _tenant(x_tenant_id)
            _authorize(
                dependencies.authorizer,
                credential=_credential(authorization, x_api_key),
                tenant_id=tenant_id,
                role="jobs:read",
            )
            if status is not None and status not in {
                "queued", "running", "succeeded", "failed", "cancelled"
            }:
                raise HTTPException(status_code=422, detail="invalid job status")
            records = dependencies.jobs.list(status=status, kind=kind)  # type: ignore[arg-type]
            return JobListResponse(jobs=[
                JobSummaryResponse(
                    job_id=job.job_id,
                    kind=job.kind,
                    status=job.status,
                    attempts=job.attempts,
                    error=job.error,
                    created_at=(
                        job.created_at.isoformat()
                        if hasattr(job.created_at, "isoformat")
                        else str(job.created_at)
                    ),
                    updated_at=(
                        job.updated_at.isoformat()
                        if hasattr(job.updated_at, "isoformat")
                        else str(job.updated_at)
                    ),
                )
                for job in records
                if job.tenant_id == tenant_id
            ])

    if dependencies.exports is not None:
        export_workflow = dependencies.exports.workflow

        @app.post("/v1/dataset-exports", response_model=DatasetExportJobResponse, status_code=202)
        def submit_dataset_export(
            request: DatasetExportRequest,
            x_tenant_id: str = Header(...),
            x_api_key: str | None = Header(None),
            authorization: str | None = Header(None),
        ) -> DatasetExportJobResponse:
            tenant_id = _tenant(x_tenant_id)
            _authorize(dependencies.authorizer, credential=_credential(authorization, x_api_key), tenant_id=tenant_id, role="exports:write")
            try:
                job = export_workflow.submit(
                    tenant_id=tenant_id, version_id=request.version_id,
                    format=request.format, idempotency_key=request.idempotency_key,
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            return _job_response(job)

        @app.get("/v1/dataset-exports/{job_id}", response_model=DatasetExportJobResponse)
        def get_dataset_export(
            job_id: UUID,
            x_tenant_id: str = Header(...),
            x_api_key: str | None = Header(None),
            authorization: str | None = Header(None),
        ) -> DatasetExportJobResponse:
            tenant_id = _tenant(x_tenant_id)
            _authorize(dependencies.authorizer, credential=_credential(authorization, x_api_key), tenant_id=tenant_id, role="exports:read")
            try:
                job = export_workflow.dependencies.jobs.get(job_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            if job.tenant_id != tenant_id:
                raise HTTPException(status_code=404, detail="unknown export job")
            return _job_response(job)

        @app.post("/v1/dataset-exports/{job_id}/cancel", response_model=DatasetExportJobResponse)
        def cancel_dataset_export(
            job_id: UUID,
            x_tenant_id: str = Header(...),
            x_api_key: str | None = Header(None),
            authorization: str | None = Header(None),
        ) -> DatasetExportJobResponse:
            tenant_id = _tenant(x_tenant_id)
            _authorize(dependencies.authorizer, credential=_credential(authorization, x_api_key), tenant_id=tenant_id, role="exports:write")
            try:
                job = export_workflow.dependencies.jobs.get(job_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            if job.tenant_id != tenant_id:
                raise HTTPException(status_code=404, detail="unknown export job")
            try:
                job = export_workflow.dependencies.jobs.request_cancel(job_id)
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            return _job_response(job)

        @app.get("/v1/dataset-exports/{job_id}/download")
        def download_dataset_export(
            job_id: UUID,
            x_tenant_id: str = Header(...),
            x_api_key: str | None = Header(None),
            authorization: str | None = Header(None),
        ) -> FileResponse:
            tenant_id = _tenant(x_tenant_id)
            _authorize(dependencies.authorizer, credential=_credential(authorization, x_api_key), tenant_id=tenant_id, role="exports:read")
            try:
                job = export_workflow.dependencies.jobs.get(job_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            if job.tenant_id != tenant_id or job.kind != "dataset-export":
                raise HTTPException(status_code=404, detail="unknown export job")
            if job.status != "succeeded":
                raise HTTPException(
                    status_code=409,
                    detail=f"export job is not ready for download; current status is {job.status}",
                )
            version_id = UUID(job.payload["version_id"])
            format = job.payload["format"]
            try:
                archive = export_workflow.dependencies.exports.package(
                    tenant_id=tenant_id, version_id=version_id, format=format
                )
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            return FileResponse(
                archive,
                media_type="application/zip",
                filename=f"{version_id}-{format}.zip",
            )

    if dependencies.refinement is not None:
        refinement = dependencies.refinement

        if refinement.image_loader is not None:
            @app.websocket("/v1/refinement/{image_id}")
            async def refinement_socket(websocket: WebSocket, image_id: str) -> None:
                await websocket.accept()
                try:
                    assert refinement.image_loader is not None
                    image = refinement.image_loader(image_id)
                    snapshot = refinement.service.create(image_id=image_id, image=image)
                except (KeyError, ValueError) as exc:
                    await websocket.close(code=1008, reason=str(exc))
                    return
                await websocket.send_json(_snapshot_payload(snapshot))
                try:
                    while True:
                        message = await websocket.receive_json()
                        message_type = message.get("type")
                        if message_type == "click":
                            try:
                                snapshot = refinement.service.click(
                                    snapshot.session_id,
                                    expected_revision=message["expected_revision"],
                                    point=ClickPoint(
                                        x=message["x"],
                                        y=message["y"],
                                        label=message["label"],
                                    ),
                                )
                            except (KeyError, TypeError, ValueError) as exc:
                                await websocket.send_json(
                                    {"type": "error", "detail": str(exc)}
                                )
                                continue
                            await websocket.send_json(_snapshot_payload(snapshot))
                        elif message_type == "get":
                            await websocket.send_json(
                                _snapshot_payload(
                                    refinement.service.get(snapshot.session_id)
                                )
                            )
                        else:
                            await websocket.send_json(
                                {
                                    "type": "error",
                                    "detail": "unsupported message type",
                                }
                            )
                except WebSocketDisconnect:
                    return

        if refinement.dataset_loader is not None:
            dataset_loader = refinement.dataset_loader

            @app.websocket("/v1/refinement/{version_id}/{image_id}")
            async def dataset_refinement_socket(
                websocket: WebSocket, version_id: str, image_id: str
            ) -> None:
                await websocket.accept()
                as_polygons = websocket.query_params.get("format") == "polygons"
                tenant_raw = websocket.query_params.get("tenant_id")
                if not tenant_raw:
                    await websocket.close(
                        code=1008, reason="tenant_id query parameter is required"
                    )
                    return
                try:
                    tenant_id = _tenant(tenant_raw)
                    _authorize(
                        dependencies.authorizer,
                        credential=_credential(
                            None, websocket.query_params.get("api_key")
                        ),
                        tenant_id=tenant_id,
                        role="labels:write",
                    )
                    target = dataset_loader(tenant_id, UUID(version_id), image_id)
                    snapshot = refinement.service.create(
                        image_id=image_id,
                        image=target.image,
                        tenant_id=tenant_id,
                        version_id=UUID(version_id),
                        prompt=target.prompt,
                        category_id=target.category_id,
                    )
                except (HTTPException, KeyError, ValueError, FileNotFoundError) as exc:
                    detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
                    await websocket.close(code=1008, reason=str(detail))
                    return
                await websocket.send_json(_snapshot_payload(snapshot, polygons=as_polygons))
                try:
                    while True:
                        message = await websocket.receive_json()
                        message_type = message.get("type")
                        if message_type == "click":
                            try:
                                snapshot = refinement.service.click(
                                    snapshot.session_id,
                                    expected_revision=message["expected_revision"],
                                    point=ClickPoint(
                                        x=message["x"],
                                        y=message["y"],
                                        label=message["label"],
                                    ),
                                )
                            except ModelUnavailableError as exc:
                                await websocket.send_json(
                                    {"type": "error", "detail": str(exc)}
                                )
                                continue
                            except (KeyError, TypeError, ValueError) as exc:
                                await websocket.send_json(
                                    {"type": "error", "detail": str(exc)}
                                )
                                continue
                            await websocket.send_json(_snapshot_payload(snapshot, polygons=as_polygons))
                        elif message_type == "get":
                            await websocket.send_json(
                                _snapshot_payload(
                                    refinement.service.get(snapshot.session_id)
                                )
                            )
                        else:
                            await websocket.send_json(
                                {
                                    "type": "error",
                                    "detail": "unsupported message type",
                                }
                            )
                except WebSocketDisconnect:
                    return

    orchestrator = dependencies.orchestrator or TaskOrchestrator()

    @app.post("/v1/orchestrate", response_model=TaskSpecResponse)
    def orchestrate(
        request: OrchestrateRequest,
        x_tenant_id: str | None = Header(None),
        x_api_key: str | None = Header(None),
        authorization: str | None = Header(None),
    ) -> TaskSpecResponse:
        tenant_id = _tenant(x_tenant_id) if x_tenant_id is not None else None
        _authorize(dependencies.authorizer, credential=_credential(authorization, x_api_key), tenant_id=tenant_id, role="labels:write")
        try:
            spec = orchestrator.orchestrate(TaskOrchestrationRequest(
                message=request.message,
                task_type=request.task_type,
                target_objects=tuple(request.target_objects),
                visual_attributes=tuple(request.visual_attributes),
                sample_size=request.sample_size,
                max_iterations=request.max_iterations,
            ))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _task_spec_response(spec)

    if dependencies.labels is not None:
        label_workflow = dependencies.labels.workflow

        @app.post("/v1/label-jobs", response_model=LabelJobResponse, status_code=202)
        def submit_label_job(
            request: LabelJobRequest,
            x_tenant_id: str = Header(...),
            x_api_key: str | None = Header(None),
            authorization: str | None = Header(None),
        ) -> LabelJobResponse:
            tenant_id = _tenant(x_tenant_id)
            _authorize(dependencies.authorizer, credential=_credential(authorization, x_api_key), tenant_id=tenant_id, role="labels:write")
            if not request.prompt.strip():
                raise HTTPException(status_code=422, detail="prompt must not be empty")
            if request.task not in ("detection", "segmentation"):
                raise HTTPException(status_code=422, detail="task must be detection or segmentation")
            if not 0 <= request.threshold <= 1:
                raise HTTPException(status_code=422, detail="threshold must be between 0 and 1")
            try:
                job = label_workflow.submit(
                    tenant_id=tenant_id,
                    asset_id=request.asset_id,
                    prompt=request.prompt,
                    threshold=request.threshold,
                    task=request.task,
                    category_id=request.category_id,
                    category_name=request.category_name,
                    idempotency_key=request.idempotency_key,
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            return _label_job_response(job)

        @app.get("/v1/label-jobs/{job_id}", response_model=LabelJobResponse)
        def get_label_job(
            job_id: UUID,
            x_tenant_id: str = Header(...),
            x_api_key: str | None = Header(None),
            authorization: str | None = Header(None),
        ) -> LabelJobResponse:
            tenant_id = _tenant(x_tenant_id)
            _authorize(dependencies.authorizer, credential=_credential(authorization, x_api_key), tenant_id=tenant_id, role="labels:read")
            try:
                job = label_workflow.dependencies.jobs.get(job_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            if job.tenant_id != tenant_id or job.kind != "batch-label":
                raise HTTPException(status_code=404, detail="unknown label job")
            return _label_job_response(job)

        @app.post("/v1/label-jobs/{job_id}/cancel", response_model=LabelJobResponse)
        def cancel_label_job(
            job_id: UUID,
            x_tenant_id: str = Header(...),
            x_api_key: str | None = Header(None),
            authorization: str | None = Header(None),
        ) -> LabelJobResponse:
            tenant_id = _tenant(x_tenant_id)
            _authorize(dependencies.authorizer, credential=_credential(authorization, x_api_key), tenant_id=tenant_id, role="labels:write")
            try:
                job = label_workflow.dependencies.jobs.get(job_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            if job.tenant_id != tenant_id or job.kind != "batch-label":
                raise HTTPException(status_code=404, detail="unknown label job")
            try:
                job = label_workflow.dependencies.jobs.request_cancel(job_id)
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            return _label_job_response(job)

        if dependencies.label_images is not None and dependencies.annotations is not None:
            @app.get("/v1/label-jobs/{job_id}/preview/{image_id}/image")
            def label_preview_image(
                job_id: UUID,
                image_id: str,
                x_tenant_id: str | None = Header(None),
                tenant_id_q: str | None = Query(None, alias="tenant_id"),
                x_api_key: str | None = Header(None),
                authorization: str | None = Header(None),
            ) -> Response:
                raw_tenant = x_tenant_id or tenant_id_q
                if raw_tenant is None:
                    raise HTTPException(status_code=400, detail="X-Tenant-ID or tenant_id is required")
                tenant_id = _tenant(raw_tenant)
                _authorize(
                    dependencies.authorizer,
                    credential=_credential(authorization, x_api_key),
                    tenant_id=tenant_id,
                    role="labels:read",
                )
                try:
                    job = label_workflow.dependencies.jobs.get(job_id)
                except KeyError as exc:
                    raise HTTPException(status_code=404, detail=str(exc)) from exc
                if job.tenant_id != tenant_id or job.kind != "batch-label":
                    raise HTTPException(status_code=404, detail="unknown label job")
                try:
                    plan = label_plan(job.payload)
                except (KeyError, TypeError, ValueError) as exc:
                    raise HTTPException(status_code=404, detail="label job has no source asset") from exc
                try:
                    images = dependencies.label_images.resolve_many(
                        tenant_id=tenant_id, asset_ids=plan.asset_ids
                    )
                except (KeyError, ValueError) as exc:
                    raise HTTPException(status_code=404, detail=str(exc)) from exc
                match = next((item for item in images if item.image_id == image_id), None)
                if match is None:
                    raise HTTPException(status_code=404, detail="unknown preview image")
                revisions = dependencies.annotations.list_latest(
                    tenant_id=tenant_id, image_ids=[image_id]
                )
                class_names = {
                    category.category_id: category.name for category in plan.categories
                }
                revisions = live_in_set(
                    revisions, job_set_id(job_id), category_ids=set(class_names)
                )
                try:
                    with Image.open(match.path) as source:
                        overlay = render_annotation_overlay(
                            source.convert("RGB"),
                            [revision.annotation for revision in revisions],
                            class_names=class_names,
                        )
                except OSError as exc:
                    raise HTTPException(status_code=404, detail="preview image unavailable") from exc
                buffer = io.BytesIO()
                overlay.save(buffer, format="PNG")
                return Response(content=buffer.getvalue(), media_type="image/png")

    if dependencies.datasets is not None:
        dataset_builder = dependencies.datasets.builder
        versions = dependencies.datasets.versions

        @app.post("/v1/datasets", response_model=DatasetSummaryResponse, status_code=201)
        def create_dataset(
            request: DatasetCreateRequest,
            x_tenant_id: str = Header(...),
            x_api_key: str | None = Header(None),
            authorization: str | None = Header(None),
        ) -> DatasetSummaryResponse:
            tenant_id = _tenant(x_tenant_id)
            _authorize(dependencies.authorizer, credential=_credential(authorization, x_api_key), tenant_id=tenant_id, role="labels:write")
            if not request.idempotency_key.strip():
                raise HTTPException(status_code=422, detail="idempotency_key must not be empty")
            try:
                result = dataset_builder.build_from_job(
                    tenant_id=tenant_id,
                    job_id=request.job_id,
                    name=request.name,
                    idempotency_key=request.idempotency_key,
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except PermissionError as exc:
                raise HTTPException(status_code=403, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            return _dataset_summary(
                result.record, _asset_names(dependencies.registry, tenant_id)
            )

        if dependencies.datasets.importer is not None:
            importer = dependencies.datasets.importer

            @app.post(
                "/v1/datasets/import",
                response_model=DatasetImportResponse,
                status_code=201,
            )
            def import_dataset(
                file: UploadFile = File(...),
                name: str = Form(...),
                x_tenant_id: str = Header(...),
                x_api_key: str | None = Header(None),
                authorization: str | None = Header(None),
            ) -> DatasetImportResponse:
                tenant_id = _tenant(x_tenant_id)
                _authorize(dependencies.authorizer, credential=_credential(authorization, x_api_key), tenant_id=tenant_id, role="labels:write")
                try:
                    result = importer.import_coco_zip(
                        tenant_id=tenant_id, archive=file.file, name=name
                    )
                except DatasetImportError as exc:
                    raise HTTPException(status_code=422, detail=str(exc)) from exc
                return DatasetImportResponse(
                    dataset=_dataset_summary(
                        result.record, _asset_names(dependencies.registry, tenant_id)
                    ),
                    image_count=result.image_count,
                    annotation_count=result.annotation_count,
                    skipped_annotations=result.skipped_annotations,
                    missing_images=list(result.missing_images[:50]),
                )

        @app.get("/v1/datasets", response_model=list[DatasetSummaryResponse])
        def list_datasets(
            x_tenant_id: str = Header(...),
            x_api_key: str | None = Header(None),
            authorization: str | None = Header(None),
        ) -> list[DatasetSummaryResponse]:
            tenant_id = _tenant(x_tenant_id)
            _authorize(dependencies.authorizer, credential=_credential(authorization, x_api_key), tenant_id=tenant_id, role="labels:read")
            names = _asset_names(dependencies.registry, tenant_id)
            return [
                _dataset_summary(record, names)
                for record in versions.list(tenant_id=tenant_id)
            ]

        @app.get("/v1/datasets/{version_id}", response_model=DatasetSummaryResponse)
        def get_dataset(
            version_id: UUID,
            x_tenant_id: str = Header(...),
            x_api_key: str | None = Header(None),
            authorization: str | None = Header(None),
        ) -> DatasetSummaryResponse:
            tenant_id = _tenant(x_tenant_id)
            _authorize(dependencies.authorizer, credential=_credential(authorization, x_api_key), tenant_id=tenant_id, role="labels:read")
            try:
                record = versions.get(tenant_id=tenant_id, version_id=version_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            return _dataset_summary(record, _asset_names(dependencies.registry, tenant_id))

        @app.get(
            "/v1/datasets/{version_id}/images",
            response_model=DatasetImagesResponse,
        )
        def list_dataset_images(
            version_id: UUID,
            x_tenant_id: str = Header(...),
            x_api_key: str | None = Header(None),
            authorization: str | None = Header(None),
            limit: int = Query(default=200, ge=1, le=1000),
            offset: int = Query(default=0, ge=0),
        ) -> DatasetImagesResponse:
            tenant_id = _tenant(x_tenant_id)
            _authorize(
                dependencies.authorizer,
                credential=_credential(authorization, x_api_key),
                tenant_id=tenant_id,
                role="labels:read",
            )
            try:
                record = versions.get(tenant_id=tenant_id, version_id=version_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            by_image: dict[str, list[Annotation]] = {
                image.image_id: [] for image in record.dataset.images
            }
            for annotation in record.dataset.annotations:
                by_image.setdefault(annotation.image_id, []).append(annotation)
            page = list(record.dataset.images[offset : offset + limit])
            splits = {
                image.image_id: record.dataset.split_of(image) for image in page
            }
            names = image_display_names(
                dependencies.registry, tenant_id=tenant_id, image_ids=[image.image_id for image in page]
            )

            def image_size(image: object) -> int:
                if dependencies.datasets is None or dependencies.datasets.images_root is None:
                    return 0
                try:
                    return _dataset_image_path(
                        dependencies.datasets.images_root,
                        tenant_id=tenant_id,
                        version_id=version_id,
                        file_name=image.file_name,
                    ).stat().st_size
                except (OSError, ValueError):
                    return 0

            return DatasetImagesResponse(
                images=[
                    DatasetImageResponse(
                        image_id=image.image_id,
                        file_name=image.file_name,
                        width=image.width,
                        height=image.height,
                        split=splits[image.image_id],
                        source_name=names.get(image.image_id),
                        file_size=image_size(image),
                        annotations=[
                            DatasetAnnotationResponse(
                                annotation_id=annotation.annotation_id,
                                category_id=annotation.category_id,
                                bbox=list(annotation.bbox),
                                segmentation=[
                                    list(polygon) for polygon in annotation.segmentation
                                ],
                                score=annotation.score,
                            )
                            for annotation in by_image[image.image_id]
                        ],
                    )
                    for image in page
                ],
                total=len(record.dataset.images),
                limit=limit,
                offset=offset,
                has_more=offset + len(page) < len(record.dataset.images),
            )

        @app.get("/v1/datasets/{version_id}/images/{image_id}/file")
        def get_dataset_image_file(
            version_id: UUID,
            image_id: str,
            x_tenant_id: str = Header(...),
            x_api_key: str | None = Header(None),
            authorization: str | None = Header(None),
        ) -> FileResponse:
            tenant_id = _tenant(x_tenant_id)
            _authorize(
                dependencies.authorizer,
                credential=_credential(authorization, x_api_key),
                tenant_id=tenant_id,
                role="labels:read",
            )
            if dependencies.datasets.images_root is None:
                raise HTTPException(status_code=404, detail="dataset images are unavailable")
            try:
                record = versions.get(tenant_id=tenant_id, version_id=version_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            image = next(
                (item for item in record.dataset.images if item.image_id == image_id),
                None,
            )
            if image is None:
                raise HTTPException(status_code=404, detail="dataset image not found")
            path = _dataset_image_path(
                dependencies.datasets.images_root,
                tenant_id=tenant_id,
                version_id=version_id,
                file_name=image.file_name,
            )
            return FileResponse(path)

        @app.post(
            "/v1/datasets/{version_id}/rebuild",
            response_model=DatasetSummaryResponse,
            status_code=201,
        )
        def rebuild_dataset(
            version_id: UUID,
            request: DatasetRebuildRequest,
            x_tenant_id: str = Header(...),
            x_api_key: str | None = Header(None),
            authorization: str | None = Header(None),
        ) -> DatasetSummaryResponse:
            tenant_id = _tenant(x_tenant_id)
            _authorize(
                dependencies.authorizer,
                credential=_credential(authorization, x_api_key),
                tenant_id=tenant_id,
                role="labels:write",
            )
            if not request.idempotency_key.strip():
                raise HTTPException(status_code=422, detail="idempotency_key must not be empty")
            try:
                result = dataset_builder.rebuild_from_version(
                    tenant_id=tenant_id,
                    version_id=version_id,
                    name=request.name,
                    idempotency_key=request.idempotency_key,
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            return _dataset_summary(
                result.record, _asset_names(dependencies.registry, tenant_id)
            )

    if (
        dependencies.refinement is not None
        and dependencies.refinement.annotations is not None
        and dependencies.datasets is not None
    ):
        refinement = dependencies.refinement
        annotations = refinement.annotations
        versions = dependencies.datasets.versions

        @app.post(
            "/v1/refinements/{session_id}/accept",
            response_model=RefinementAcceptResponse,
        )
        def accept_refinement(
            session_id: UUID,
            request: RefinementAcceptRequest,
            x_tenant_id: str = Header(...),
            x_api_key: str | None = Header(None),
            authorization: str | None = Header(None),
        ) -> RefinementAcceptResponse:
            tenant_id = _tenant(x_tenant_id)
            _authorize(
                dependencies.authorizer,
                credential=_credential(authorization, x_api_key),
                tenant_id=tenant_id,
                role="labels:write",
            )
            try:
                snapshot = refinement.service.get(session_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            if (
                snapshot.tenant_id != tenant_id
                or snapshot.version_id is None
            ):
                raise HTTPException(status_code=404, detail="refinement session not found")
            if (
                request.refinement_revision is not None
                and request.refinement_revision != snapshot.revision
            ):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"refinement revision moved to {snapshot.revision}; "
                        "click again to refresh the mask"
                    ),
                )
            if not 0 <= request.prediction_index < len(snapshot.predictions):
                raise HTTPException(
                    status_code=409,
                    detail="there is no mask preview to accept; click the image first",
                )

            try:
                version_record = versions.get(
                    tenant_id=tenant_id, version_id=snapshot.version_id
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            dataset_image = next(
                (
                    image
                    for image in version_record.dataset.images
                    if image.image_id == snapshot.image_id
                ),
                None,
            )
            if dataset_image is None:
                raise HTTPException(status_code=404, detail="dataset image not found")
            category_id = (
                request.category_id if request.category_id is not None else snapshot.category_id
            )
            if category_id not in {
                category.category_id for category in version_record.dataset.categories
            }:
                raise HTTPException(
                    status_code=422,
                    detail="the refinement category is not part of this dataset",
                )

            latest = None
            set_id = version_record.dataset.annotation_set
            annotation_id = request.annotation_id
            if annotation_id is not None and not in_set(annotation_id, set_id):
                raise HTTPException(
                    status_code=422,
                    detail="annotation does not belong to this dataset",
                )
            if annotation_id is not None:
                try:
                    latest = annotations.latest(
                        tenant_id=tenant_id,
                        image_id=snapshot.image_id,
                        annotation_id=annotation_id,
                    )
                except KeyError as exc:
                    raise HTTPException(
                        status_code=404, detail="annotation not found"
                    ) from exc
                if request.category_id is None and latest.annotation.category_id != snapshot.category_id:
                    raise HTTPException(
                        status_code=422,
                        detail="annotation belongs to a different category",
                    )
            else:
                # No target annotation means "add a new object". Replacing an
                # existing mask must name it explicitly, so a click on a missed
                # object never overwrites a correct one.
                annotation_id = manual_annotation_id(set_id, snapshot.image_id)

            current_revision = latest.revision if latest is not None else 0
            if (
                request.expected_revision is not None
                and request.expected_revision != current_revision
            ):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"annotation revision moved to {current_revision}; "
                        "reload the dataset and try again"
                    ),
                )

            prediction = snapshot.predictions[request.prediction_index]
            x1, y1, x2, y2 = (float(value) for value in prediction.box)
            annotation = Annotation(
                annotation_id=str(annotation_id),
                image_id=snapshot.image_id,
                category_id=category_id,
                bbox=(x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)),
                segmentation=mask_polygons(prediction.mask),
                score=None,
            )
            try:
                annotation.validate(dataset_image)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            saved = annotations.save(
                tenant_id=tenant_id,
                annotation=annotation,
                author_id="manual-refinement",
                expected_revision=current_revision,
            )
            return RefinementAcceptResponse(
                session_id=snapshot.session_id,
                image_id=snapshot.image_id,
                annotation_id=saved.annotation.annotation_id,
                revision=saved.revision,
            )

        @app.delete(
            "/v1/datasets/{version_id}/images/{image_id}/annotations/{annotation_id:path}",
            response_model=AnnotationDeleteResponse,
        )
        def delete_annotation(
            version_id: UUID,
            image_id: str,
            annotation_id: str,
            expected_revision: int | None = Query(default=None, ge=0),
            x_tenant_id: str = Header(...),
            x_api_key: str | None = Header(None),
            authorization: str | None = Header(None),
        ) -> AnnotationDeleteResponse:
            """Retire an annotation (e.g. a false positive) with a tombstone revision.

            The version itself is immutable: the deletion shows up in the next
            ``/rebuild`` of this version, like any other correction.
            """

            tenant_id = _tenant(x_tenant_id)
            _authorize(
                dependencies.authorizer,
                credential=_credential(authorization, x_api_key),
                tenant_id=tenant_id,
                role="labels:write",
            )
            try:
                version_record = versions.get(tenant_id=tenant_id, version_id=version_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            if not any(image.image_id == image_id for image in version_record.dataset.images):
                raise HTTPException(status_code=404, detail="dataset image not found")
            if not in_set(annotation_id, version_record.dataset.annotation_set):
                raise HTTPException(status_code=404, detail="annotation not found")
            try:
                latest = annotations.latest(
                    tenant_id=tenant_id, image_id=image_id, annotation_id=annotation_id
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="annotation not found") from exc
            if latest.annotation.deleted:
                raise HTTPException(status_code=409, detail="annotation is already deleted")
            try:
                saved = annotations.save(
                    tenant_id=tenant_id,
                    annotation=replace(latest.annotation, deleted=True),
                    author_id="manual-review",
                    expected_revision=(
                        latest.revision if expected_revision is None else expected_revision
                    ),
                )
            except RevisionConflictError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            return AnnotationDeleteResponse(
                image_id=image_id,
                annotation_id=annotation_id,
                revision=saved.revision,
            )

    if dependencies.calibration is not None:
        @app.post("/v1/calibration/run", response_model=CalibrationResponse)
        async def run_calibration(
            prompt: str = Form(...),
            thresholds: str = Form(...),
            minimum_agreement: float = Form(1.0),
            max_iterations: int = Form(5),
            validator_mode: str = Form("has_prediction"),
            images: list[UploadFile] = File(...),
            x_tenant_id: str | None = Header(None),
            x_api_key: str | None = Header(None),
            authorization: str | None = Header(None),
        ) -> CalibrationResponse:
            tenant_id = _tenant(x_tenant_id) if x_tenant_id is not None else None
            _authorize(dependencies.authorizer, credential=_credential(authorization, x_api_key), tenant_id=tenant_id, role="labels:write")
            if not prompt.strip():
                raise HTTPException(status_code=422, detail="prompt must not be empty")
            try:
                parsed_thresholds = [float(value.strip()) for value in thresholds.split(",") if value.strip()]
            except ValueError as exc:
                raise HTTPException(status_code=422, detail="thresholds must be comma-separated numbers") from exc
            if not parsed_thresholds:
                raise HTTPException(status_code=422, detail="at least one threshold is required")
            if not images:
                raise HTTPException(status_code=422, detail="at least one calibration image is required")
            samples: list[CalibrationSample] = []
            for upload in images:
                data = await upload.read()
                try:
                    image = Image.open(io.BytesIO(data)).convert("RGB")
                except Exception as exc:
                    raise HTTPException(status_code=422, detail=f"invalid image: {upload.filename}") from exc
                samples.append(CalibrationSample(image_id=upload.filename or "image", image=image))

            mode = validator_mode.strip().lower()
            if mode == "has_prediction":
                validator = lambda sample, predictions: len(predictions) > 0
            elif mode.startswith("score_at_least:"):
                try:
                    min_score = float(mode.split(":", 1)[1])
                except ValueError as exc:
                    raise HTTPException(status_code=422, detail="invalid score_at_least validator mode") from exc
                validator = lambda sample, predictions, min_score=min_score: any(p.score >= min_score for p in predictions)
            else:
                raise HTTPException(status_code=422, detail="unsupported validator_mode")

            try:
                calibrator = Sam3Calibrator(
                    dependencies.calibration.provider,
                    validator,
                    minimum_agreement=minimum_agreement,
                    max_iterations=max_iterations,
                )
                result = calibrator.calibrate(samples, prompt=prompt, thresholds=parsed_thresholds)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            return CalibrationResponse(
                status=result.status,
                prompt=result.prompt,
                threshold=result.threshold,
                attempts=[
                    CalibrationAttemptResponse(
                        threshold=a.threshold,
                        sample_count=a.sample_count,
                        accepted_count=a.accepted_count,
                        agreement=a.agreement,
                    )
                    for a in result.attempts
                ],
                reason=result.reason,
            )

    @app.post("/v1/assets/{asset_id}/predict", response_model=Sam3PredictResponse)
    def predict_asset(
        asset_id: UUID,
        request: AssetPredictRequest,
        x_tenant_id: str = Header(...),
        x_api_key: str | None = Header(None),
        authorization: str | None = Header(None),
    ) -> Sam3PredictResponse:
        tenant_id = _tenant(x_tenant_id)
        _authorize(dependencies.authorizer, credential=_credential(authorization, x_api_key), tenant_id=tenant_id, role="assets:read")
        provider = dependencies.sam3 or (dependencies.calibration.provider if dependencies.calibration else None)
        if provider is None:
            raise HTTPException(status_code=503, detail="SAM3 prediction is not configured")
        prompt = (request.prompt or "").strip()
        if not prompt:
            raise HTTPException(status_code=422, detail="prompt must not be empty")
        summaries = dependencies.registry.list_assets(tenant_id=tenant_id)
        summary = next((item for item in summaries if item.asset_id == asset_id), None)
        if summary is None:
            raise HTTPException(status_code=404, detail="unknown asset")
        source = None
        if request.source_id is not None:
            source = next((item for item in summary.sources if item.source_id == request.source_id), None)
            if source is None:
                raise HTTPException(status_code=404, detail="unknown source")
        else:
            source = _representative_source(summary.sources)
        if source is None:
            raise HTTPException(status_code=422, detail="asset has no usable source")
        image = _load_source_image(source)
        if image is None:
            raise HTTPException(status_code=422, detail="asset source image could not be loaded")
        try:
            predictions = provider.predict(image, prompt, threshold=request.threshold)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ModelUnavailableError:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"prediction failed: {exc}") from exc
        width, height = image.size
        image_url = f"/v1/assets/{asset_id}/image?tenant_id={tenant_id}"
        if request.source_id is not None:
            image_url += f"&source_id={request.source_id}"
        return _prediction_response(prompt, predictions, image_url=image_url, image_width=width, image_height=height)

    @app.get("/v1/assets/{asset_id}/image")
    def asset_image(
        asset_id: UUID,
        x_tenant_id: str | None = Header(None),
        tenant_id_q: str | None = Query(None, alias="tenant_id"),
        source_id_q: UUID | None = Query(None, alias="source_id"),
        x_api_key: str | None = Header(None),
        authorization: str | None = Header(None),
    ) -> Response:
        raw_tenant = x_tenant_id or tenant_id_q
        if raw_tenant is None:
            raise HTTPException(status_code=400, detail="X-Tenant-ID or tenant_id is required")
        tenant_id = _tenant(raw_tenant)
        _authorize(dependencies.authorizer, credential=_credential(authorization, x_api_key), tenant_id=tenant_id, role="assets:read")
        summaries = dependencies.registry.list_assets(tenant_id=tenant_id)
        summary = next((item for item in summaries if item.asset_id == asset_id), None)
        if summary is None:
            raise HTTPException(status_code=404, detail="unknown asset")
        if source_id_q is not None:
            source = next((item for item in summary.sources if item.source_id == source_id_q), None)
            if source is None:
                raise HTTPException(status_code=404, detail="unknown source")
        else:
            source = _representative_source(summary.sources)
        if source is None:
            raise HTTPException(status_code=404, detail="asset has no source")
        payload = _source_image_bytes(source)
        if payload is None:
            raise HTTPException(status_code=404, detail="asset image unavailable")
        data, media_type = payload
        return Response(content=data, media_type=media_type, headers={"Cache-Control": "private, max-age=300"})

    return app
