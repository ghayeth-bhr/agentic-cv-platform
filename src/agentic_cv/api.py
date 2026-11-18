"""Local HTTP API composing assets, uploads, workflows, and orchestration."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname
from uuid import UUID

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Response, UploadFile, WebSocket, WebSocketDisconnect
from PIL import Image
from pydantic import BaseModel, Field

from .asset_registry import AssetRegistry
from .auth import Authorizer
from .calibration import CalibrationSample, CalibrationStatus, Sam3Calibrator
from .click_refinement import ClickPoint, ClickRefinementService, RefinementSnapshot
from .contracts import EventEnvelope, TaskSpec
from .dataset_export import ExportFormat
from .dataset_export_workflow import DatasetExportWorkflow
from .job_store import JobRecord
from .label_workflow import DurableLabelWorkflow
from .orchestrator import TaskOrchestrationRequest, TaskOrchestrator
from .outbox import Outbox
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


@dataclass(frozen=True, kw_only=True)
class RefinementApiDependencies:
    service: ClickRefinementService
    image_loader: Callable[[str], Image.Image]


@dataclass(frozen=True, kw_only=True)
class ExportApiDependencies:
    workflow: DatasetExportWorkflow


@dataclass(frozen=True, kw_only=True)
class LabelApiDependencies:
    workflow: DurableLabelWorkflow


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
    calibration: CalibrationApiDependencies | None = None
    orchestrator: TaskOrchestrator | None = None
    sam3: Sam3Provider | None = None


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



def _snapshot_payload(snapshot: RefinementSnapshot) -> dict:
    return {
        "type": "snapshot",
        "session_id": str(snapshot.session_id),
        "image_id": snapshot.image_id,
        "revision": snapshot.revision,
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


def _prediction_response(prompt: str, predictions: list[Sam3Prediction]) -> Sam3PredictResponse:
    return Sam3PredictResponse(
        prompt=prompt,
        predictions=[
            Sam3PredictionResponse(score=prediction.score, box=prediction.box, mask=prediction.mask.astype(bool).tolist())
            for prediction in predictions
        ],
    )


def create_app(dependencies: ApiDependencies) -> FastAPI:
    app = FastAPI(title="Agentic CV platform API", version="1")

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
        data = _thumbnail_bytes(representative)
        if data is None:
            raise HTTPException(status_code=404, detail="thumbnail unavailable")
        return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=300"})

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

    if dependencies.refinement is not None:
        refinement = dependencies.refinement

        @app.websocket("/v1/refinement/{image_id}")
        async def refinement_socket(websocket: WebSocket, image_id: str) -> None:
            await websocket.accept()
            try:
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
                                    x=message["x"], y=message["y"], label=message["label"]
                                ),
                            )
                        except (KeyError, TypeError, ValueError) as exc:
                            await websocket.send_json({"type": "error", "detail": str(exc)})
                            continue
                        await websocket.send_json(_snapshot_payload(snapshot))
                    elif message_type == "get":
                        await websocket.send_json(_snapshot_payload(refinement.service.get(snapshot.session_id)))
                    else:
                        await websocket.send_json({"type": "error", "detail": "unsupported message type"})
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
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"prediction failed: {exc}") from exc
        return _prediction_response(prompt, predictions)

    return app
