"""Replayable SSE endpoint for durable batch preview events."""

from __future__ import annotations

import json
from dataclasses import dataclass
from uuid import UUID

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import StreamingResponse

from .preview import PreviewEvent, PreviewStore
from .auth import StaticApiKeyAuthorizer


@dataclass(frozen=True, kw_only=True)
class PreviewApiDependencies:
    store: PreviewStore
    authorizer: StaticApiKeyAuthorizer | None = None


def _tenant(value: str) -> UUID:
    try:
        return UUID(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="X-Tenant-ID must be a UUID") from exc


def _format_event(event: PreviewEvent) -> str:
    payload = {
        "event_id": event.event_id,
        "job_id": str(event.job_id),
        "image_id": event.image_id,
        "processed": event.processed,
        "total": event.total,
        "status": event.status,
        "created_at": event.created_at,
    }
    return f"id: {event.event_id}\nevent: preview\ndata: {json.dumps(payload, sort_keys=True)}\n\n"


def _authorize(
    authorizer: StaticApiKeyAuthorizer | None,
    *,
    api_key: str | None,
    tenant_id: UUID,
) -> None:
    if authorizer is None:
        return
    try:
        authorizer.authorize(api_key=api_key, tenant_id=tenant_id, role="reader")
    except PermissionError as exc:
        raise HTTPException(status_code=401 if api_key is None else 403, detail=str(exc)) from exc


def create_app(dependencies: PreviewApiDependencies) -> FastAPI:
    app = FastAPI(title="Agentic CV preview API", version="1")

    @app.get("/v1/jobs/{job_id}/preview")
    def preview_stream(
        job_id: UUID,
        x_tenant_id: str = Header(...),
        x_api_key: str | None = Header(None),
        after_event_id: int = Query(default=0, ge=0),
    ) -> StreamingResponse:
        tenant_id = _tenant(x_tenant_id)
        _authorize(dependencies.authorizer, api_key=x_api_key, tenant_id=tenant_id)
        events = dependencies.store.list_for_job(
            tenant_id=tenant_id, job_id=job_id, after_event_id=after_event_id
        )
        return StreamingResponse(
            (_format_event(event) for event in events),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
