"""Local HTTP API composing assets, resumable uploads, and the outbox."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID
from typing import Callable

from fastapi import FastAPI, File, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from PIL import Image
from pydantic import BaseModel

from .asset_registry import AssetRegistry
from .click_refinement import ClickPoint, ClickRefinementService, RefinementSnapshot
from .contracts import EventEnvelope
from .outbox import Outbox
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


@dataclass(frozen=True, kw_only=True)
class RefinementApiDependencies:
    service: ClickRefinementService
    image_loader: Callable[[str], Image.Image]


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


def create_app(dependencies: ApiDependencies) -> FastAPI:
    app = FastAPI(title="Agentic CV platform API", version="1")

    @app.post("/v1/assets", response_model=AssetResponse, status_code=201)
    def create_asset(
        request: AssetCreateRequest,
        x_tenant_id: str = Header(...),
    ) -> AssetResponse:
        asset = dependencies.registry.create_asset(
            tenant_id=_tenant(x_tenant_id), name=request.name
        )
        return AssetResponse(
            asset_id=asset.asset_id,
            tenant_id=asset.tenant_id,
            name=asset.name,
            created_at=asset.created_at,
        )

    @app.post("/v1/assets/{asset_id}/uploads", response_model=UploadResponse, status_code=201)
    def start_upload(
        asset_id: UUID,
        request: UploadCreateRequest,
        x_tenant_id: str = Header(...),
    ) -> UploadResponse:
        try:
            upload = dependencies.uploads.start(
                tenant_id=_tenant(x_tenant_id), asset_id=asset_id,
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
    ) -> dict[str, int]:
        try:
            dependencies.uploads.put_part(upload_id, part_number, await part.read())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"part_number": part_number}

    @app.post("/v1/uploads/{upload_id}/complete", response_model=SourceResponse)
    def complete_upload(upload_id: UUID) -> SourceResponse:
        try:
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

    return app
