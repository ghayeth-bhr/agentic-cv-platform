"""HTTP API for semantic search and logical sub-asset creation."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from .semantic_search import SearchResult, SemanticSearchService
from .sub_assets import SubAssetStore
from .auth import StaticApiKeyAuthorizer


class SearchRequest(BaseModel):
    query: str
    asset_ids: list[str] | None = None
    limit: int = 20


class SearchResultResponse(BaseModel):
    record_id: str
    asset_id: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResultResponse]


class SubAssetResponse(BaseModel):
    sub_asset_id: UUID
    parent_asset_ids: tuple[str, ...]
    query: str
    result_ids: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class SemanticApiDependencies:
    search: SemanticSearchService
    sub_assets: SubAssetStore
    authorizer: StaticApiKeyAuthorizer | None = None


def _tenant(value: str) -> UUID:
    try:
        return UUID(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="X-Tenant-ID must be a UUID") from exc


def _result_response(results: list[SearchResult]) -> SearchResponse:
    return SearchResponse(
        results=[
            SearchResultResponse(record_id=result.record_id, asset_id=result.asset_id, score=result.score)
            for result in results
        ]
    )


def _validate_request(request: SearchRequest) -> None:
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")
    if request.limit < 1:
        raise HTTPException(status_code=422, detail="limit must be at least 1")


def _authorize(
    authorizer: StaticApiKeyAuthorizer | None,
    *,
    api_key: str | None,
    tenant_id: UUID | None,
    role: str,
) -> None:
    if authorizer is None:
        return
    if tenant_id is None:
        raise HTTPException(status_code=400, detail="X-Tenant-ID is required")
    try:
        authorizer.authorize(api_key=api_key, tenant_id=tenant_id, role=role)
    except PermissionError as exc:
        raise HTTPException(status_code=401 if api_key is None else 403, detail=str(exc)) from exc


def create_app(dependencies: SemanticApiDependencies) -> FastAPI:
    app = FastAPI(title="Agentic CV semantic search API", version="1")

    @app.post("/v1/search", response_model=SearchResponse)
    def search(
        request: SearchRequest,
        x_tenant_id: str | None = Header(None),
        x_api_key: str | None = Header(None),
    ) -> SearchResponse:
        tenant_id = _tenant(x_tenant_id) if x_tenant_id is not None else None
        _authorize(dependencies.authorizer, api_key=x_api_key, tenant_id=tenant_id, role="reader")
        _validate_request(request)
        try:
            results = dependencies.search.search(
                request.query,
                asset_ids=set(request.asset_ids) if request.asset_ids is not None else None,
                limit=request.limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _result_response(results)

    @app.post("/v1/sub-assets/from-search", response_model=SubAssetResponse, status_code=201)
    def create_sub_asset(
        request: SearchRequest,
        x_tenant_id: str = Header(...),
        x_api_key: str | None = Header(None),
    ) -> SubAssetResponse:
        tenant_id = _tenant(x_tenant_id)
        _authorize(dependencies.authorizer, api_key=x_api_key, tenant_id=tenant_id, role="writer")
        _validate_request(request)
        try:
            results = dependencies.search.search(
                request.query,
                asset_ids=set(request.asset_ids) if request.asset_ids is not None else None,
                limit=request.limit,
            )
            record = dependencies.sub_assets.create_from_results(
                tenant_id=tenant_id, query=request.query, results=results
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return SubAssetResponse(
            sub_asset_id=record.sub_asset_id,
            parent_asset_ids=record.parent_asset_ids,
            query=record.query,
            result_ids=record.result_ids,
        )

    return app
