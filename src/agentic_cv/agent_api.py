"""HTTP API for the agent chat workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .agent import AgentConflict, AgentService, AgentSessionView, CalibrationTimeoutError
from .agent_dialogue import missing_slots
from .api import _authorize, _credential, _tenant
from .auth import Authorizer
from .contracts import TaskSpec
from .errors import ModelUnavailableError


class AgentSessionCreateRequest(BaseModel):
    # One or more assets; ``asset_id`` is the v1 single-asset field.
    asset_id: UUID | None = None
    asset_ids: list[UUID] = Field(default_factory=list, max_length=50)
    message: str = Field(min_length=1)
    task_type: Literal["detection", "segmentation"] | None = None
    target_objects: list[str] = Field(default_factory=list)
    visual_attributes: list[str] = Field(default_factory=list)
    sample_size: int | None = Field(default=None, ge=1, le=20)
    max_iterations: int | None = Field(default=None, ge=1, le=10)


class AgentMessageRequest(BaseModel):
    message: str = Field(min_length=1)
    # When present, replaces the session's asset selection before this turn.
    asset_ids: list[UUID] | None = Field(default=None, max_length=50)


class AgentClassResponse(BaseModel):
    name: str
    attributes: list[str]
    prompts: list[str]


class AgentConfirmRequest(BaseModel):
    threshold: float | None = Field(default=None, ge=0, le=1)
    thresholds: list[float] | None = None
    minimum_agreement: float | None = Field(default=None, gt=0, le=1)


class AgentTaskSpecResponse(BaseModel):
    target_objects: tuple[str, ...]
    task_type: Literal["detection", "segmentation"]
    visual_attributes: tuple[str, ...]
    sample_size: int
    max_iterations: int


class AgentMessageResponse(BaseModel):
    role: str
    content: str
    created_at: str


class AgentDatasetResponse(BaseModel):
    name: str
    image_count: int
    annotation_count: int
    categories: list[dict[str, object]]


class AgentSessionResponse(BaseModel):
    session_id: UUID
    tenant_id: UUID
    asset_id: UUID
    asset_ids: list[str]
    asset_name: str
    message: str
    status: str
    spec: AgentTaskSpecResponse | None
    task_type: str | None
    classes: list[AgentClassResponse]
    export_format: str | None
    missing_slots: list[str]
    pending_slot: str | None
    quick_replies: list[str]
    assumptions: list[str]
    planner_mode: str
    planner_note: str | None
    prompt: str | None
    threshold: float | None
    calibration: dict | None
    label_job_id: UUID | None
    label_status: str | None
    dataset_version_id: UUID | None
    dataset: AgentDatasetResponse | None
    export_job_id: UUID | None
    export_status: str | None
    download_path: str | None
    error: str | None
    messages: list[AgentMessageResponse]
    created_at: str
    updated_at: str


class AgentSessionListResponse(BaseModel):
    sessions: list[AgentSessionResponse]


@dataclass(frozen=True, kw_only=True)
class AgentApiDependencies:
    service: AgentService
    authorizer: Authorizer | None = None


def _session_response(view: AgentSessionView) -> AgentSessionResponse:
    record = view.record
    spec = (
        TaskSpec.from_dict(record.spec)
        if record.spec is not None and "target_objects" in record.spec
        else None
    )
    draft = view.draft
    return AgentSessionResponse(
        session_id=record.session_id,
        tenant_id=record.tenant_id,
        asset_id=record.asset_id,
        asset_ids=list(draft.asset_ids),
        asset_name=record.asset_name,
        task_type=draft.task_type,
        classes=[
            AgentClassResponse(
                name=item.name,
                attributes=list(item.attributes),
                prompts=list(item.prompts),
            )
            for item in draft.classes
        ],
        export_format=draft.export_format,
        missing_slots=list(missing_slots(draft)),
        pending_slot=draft.pending_slot,
        quick_replies=list(view.quick_replies),
        assumptions=list(draft.assumptions),
        message=record.message,
        status=record.status,
        spec=(
            AgentTaskSpecResponse(
                target_objects=spec.target_objects,
                task_type=spec.task_type,
                visual_attributes=spec.visual_attributes,
                sample_size=spec.sample_size,
                max_iterations=spec.max_iterations,
            )
            if spec is not None
            else None
        ),
        planner_mode=record.planner_mode,
        planner_note=record.planner_note,
        prompt=record.prompt,
        threshold=record.threshold,
        calibration=record.calibration,
        label_job_id=record.label_job_id,
        label_status=view.label_status,
        dataset_version_id=record.dataset_version_id,
        dataset=(
            AgentDatasetResponse(**view.dataset) if view.dataset is not None else None
        ),
        export_job_id=record.export_job_id,
        export_status=view.export_status,
        download_path=view.download_path,
        error=record.error,
        messages=[
            AgentMessageResponse(
                role=message.role, content=message.content, created_at=message.created_at
            )
            for message in record.messages
        ],
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _raise_http(exc: Exception) -> None:
    if isinstance(exc, AgentConflict):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, CalibrationTimeoutError):
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    if isinstance(exc, ModelUnavailableError):
        raise
    if isinstance(exc, KeyError):
        raise HTTPException(status_code=404, detail="Agent session or asset not found") from exc
    if isinstance(exc, PermissionError):
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    raise exc


def create_app(dependencies: AgentApiDependencies) -> FastAPI:
    app = FastAPI(title="Agentic CV agent API", version="1")
    router = APIRouter()

    @router.post("/v1/agent/sessions", response_model=AgentSessionResponse)
    def create_session(
        request: AgentSessionCreateRequest,
        x_tenant_id: str = Header(...),
        x_api_key: str | None = Header(None),
        authorization: str | None = Header(None),
    ) -> AgentSessionResponse:
        tenant_id = _tenant(x_tenant_id)
        _authorize(
            dependencies.authorizer,
            credential=_credential(authorization, x_api_key),
            tenant_id=tenant_id,
            role="labels:write",
        )
        try:
            view = dependencies.service.create_session(
                tenant_id=tenant_id,
                asset_id=request.asset_id,
                asset_ids=tuple(request.asset_ids),
                message=request.message,
                task_type=request.task_type,
                target_objects=tuple(request.target_objects),
                visual_attributes=tuple(request.visual_attributes),
                sample_size=request.sample_size,
                max_iterations=request.max_iterations,
            )
        except Exception as exc:
            _raise_http(exc)
            raise
        return _session_response(view)

    @router.post(
        "/v1/agent/sessions/{session_id}/messages", response_model=AgentSessionResponse
    )
    def post_message(
        session_id: UUID,
        request: AgentMessageRequest,
        x_tenant_id: str = Header(...),
        x_api_key: str | None = Header(None),
        authorization: str | None = Header(None),
    ) -> AgentSessionResponse:
        """Continue the conversation: answer the agent's question or change the plan."""

        tenant_id = _tenant(x_tenant_id)
        _authorize(
            dependencies.authorizer,
            credential=_credential(authorization, x_api_key),
            tenant_id=tenant_id,
            role="labels:write",
        )
        try:
            view = dependencies.service.add_message(
                tenant_id=tenant_id,
                session_id=session_id,
                message=request.message,
                asset_ids=(
                    tuple(request.asset_ids) if request.asset_ids is not None else None
                ),
            )
        except Exception as exc:
            _raise_http(exc)
            raise
        return _session_response(view)

    @router.get("/v1/agent/sessions", response_model=AgentSessionListResponse)
    def list_sessions(
        limit: int = 50,
        x_tenant_id: str = Header(...),
        x_api_key: str | None = Header(None),
        authorization: str | None = Header(None),
    ) -> AgentSessionListResponse:
        tenant_id = _tenant(x_tenant_id)
        _authorize(
            dependencies.authorizer,
            credential=_credential(authorization, x_api_key),
            tenant_id=tenant_id,
            role="labels:read",
        )
        try:
            sessions = dependencies.service.list_sessions(
                tenant_id=tenant_id, limit=max(1, min(200, limit))
            )
        except Exception as exc:
            _raise_http(exc)
            raise
        return AgentSessionListResponse(
            sessions=[_session_response(view) for view in sessions]
        )

    @router.get("/v1/agent/sessions/{session_id}", response_model=AgentSessionResponse)
    def get_session(
        session_id: UUID,
        x_tenant_id: str = Header(...),
        x_api_key: str | None = Header(None),
        authorization: str | None = Header(None),
    ) -> AgentSessionResponse:
        tenant_id = _tenant(x_tenant_id)
        _authorize(
            dependencies.authorizer,
            credential=_credential(authorization, x_api_key),
            tenant_id=tenant_id,
            role="labels:read",
        )
        try:
            view = dependencies.service.get_session(
                tenant_id=tenant_id, session_id=session_id
            )
        except Exception as exc:
            _raise_http(exc)
            raise
        return _session_response(view)

    @router.get(
        "/v1/agent/sessions/{session_id}/calibration-overlays/{overlay_id}"
    )
    def get_calibration_overlay(
        session_id: UUID,
        overlay_id: str,
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
        try:
            path = dependencies.service.overlay_path(
                tenant_id=tenant_id,
                session_id=session_id,
                overlay_id=overlay_id,
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail="calibration overlay not found"
            ) from exc
        return FileResponse(path, media_type="image/png")

    @router.post(
        "/v1/agent/sessions/{session_id}/confirm", response_model=AgentSessionResponse
    )
    def confirm_session(
        session_id: UUID,
        request: AgentConfirmRequest | None = None,
        x_tenant_id: str = Header(...),
        x_api_key: str | None = Header(None),
        authorization: str | None = Header(None),
    ) -> AgentSessionResponse:
        tenant_id = _tenant(x_tenant_id)
        _authorize(
            dependencies.authorizer,
            credential=_credential(authorization, x_api_key),
            tenant_id=tenant_id,
            role="labels:write",
        )
        body = request or AgentConfirmRequest()
        try:
            view = dependencies.service.confirm(
                tenant_id=tenant_id,
                session_id=session_id,
                threshold=body.threshold,
                thresholds=(
                    tuple(body.thresholds) if body.thresholds is not None else None
                ),
                minimum_agreement=body.minimum_agreement,
            )
        except Exception as exc:
            _raise_http(exc)
            raise
        return _session_response(view)

    @router.post(
        "/v1/agent/sessions/{session_id}/cancel", response_model=AgentSessionResponse
    )
    def cancel_session(
        session_id: UUID,
        x_tenant_id: str = Header(...),
        x_api_key: str | None = Header(None),
        authorization: str | None = Header(None),
    ) -> AgentSessionResponse:
        tenant_id = _tenant(x_tenant_id)
        _authorize(
            dependencies.authorizer,
            credential=_credential(authorization, x_api_key),
            tenant_id=tenant_id,
            role="labels:write",
        )
        try:
            view = dependencies.service.cancel(
                tenant_id=tenant_id, session_id=session_id
            )
        except Exception as exc:
            _raise_http(exc)
            raise
        return _session_response(view)

    app.include_router(router)
    return app
