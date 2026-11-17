"""Local HTTP API composing assets, resumable uploads, and the outbox."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel

from .asset_registry import AssetRegistry
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

    return app
