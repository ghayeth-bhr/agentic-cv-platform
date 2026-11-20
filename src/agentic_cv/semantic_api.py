"""HTTP API for semantic search, indexing, and logical sub-asset creation."""

from __future__ import annotations

import hashlib
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
from uuid import UUID

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .asset_registry import AssetRegistryBackend
from .auth import Authorizer
from .errors import install_model_unavailable_handler
from .index_worker import INDEX_JOB_KIND, IndexWorker
from .job_store import JobRecord
from .semantic_indexing import AssetIndexResult, AssetSemanticIndexer
from .semantic_search import SearchResult, SemanticSearchService
from .sub_assets import SubAssetBackend, SubAssetRecord


class SearchRequest(BaseModel):
    query: str
    asset_ids: list[str] | None = None
    limit: int = Field(default=20, le=500)
    # Drop matches below this cosine similarity (search and legacy save).
    min_score: float | None = Field(default=None, ge=-1, le=1)
    # Save exactly these results (the user's selection) instead of
    # re-running the query and saving every match.
    record_ids: list[str] | None = Field(default=None, max_length=1000)
    # Name for the asset created from the selection.
    name: str | None = Field(default=None, max_length=120)


class SearchResultResponse(BaseModel):
    record_id: str
    asset_id: str
    score: float
    image_url: str | None = None


class SearchResponse(BaseModel):
    results: list[SearchResultResponse]


class SubAssetResponse(BaseModel):
    sub_asset_id: UUID
    parent_asset_ids: tuple[str, ...]
    parent_asset_names: list[str] = Field(default_factory=list)
    query: str
    result_ids: tuple[str, ...]
    asset_id: UUID | None = None
    asset_name: str | None = None
    source_count: int = 0
    created_at: str | None = None


class IndexAssetResponse(BaseModel):
    asset_id: UUID
    indexed_records: int
    sources_indexed: int
    detail: str


class IndexJobResponse(BaseModel):
    job_id: UUID
    asset_id: UUID
    kind: str
    status: str
    attempts: int
    error: str | None = None


@dataclass(frozen=True, kw_only=True)
class SemanticApiDependencies:
    search: SemanticSearchService
    sub_assets: SubAssetBackend
    indexer: AssetSemanticIndexer | None = None
    index_worker: IndexWorker | None = None
    registry: AssetRegistryBackend | None = None
    authorizer: Authorizer | None = None


def _tenant(value: str) -> UUID:
    try:
        return UUID(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="X-Tenant-ID must be a UUID") from exc


def _image_url(record_id: str, tenant_id: UUID | None) -> str | None:
    if tenant_id is None:
        return None
    return f"/v1/search/records/{quote(record_id, safe='')}/image?tenant_id={tenant_id}"


def _result_response(results: list[SearchResult], *, tenant_id: UUID | None) -> SearchResponse:
    return SearchResponse(
        results=[
            SearchResultResponse(
                record_id=result.record_id,
                asset_id=result.asset_id,
                score=result.score,
                image_url=_image_url(result.record_id, tenant_id),
            )
            for result in results
        ]
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _materialize_search_asset(
    *,
    registry: AssetRegistryBackend,
    tenant_id: UUID,
    query: str,
    results: list[SearchResult],
    name: str | None = None,
) -> tuple[UUID, str, int]:
    if not results:
        raise ValueError("search returned no results to save as a sub-asset")
    name = (name or "").strip() or f"Sub-asset: {query.strip()[:80] or 'search results'}"
    asset = registry.create_asset(tenant_id=tenant_id, name=name)
    added: list[str] = []
    for index, result in enumerate(results, start=1):
        image_path = Path(result.image_path)
        if not image_path.exists():
            continue
        media_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        digest = _sha256_file(image_path)
        suffix = image_path.suffix or ".jpg"
        filename = f"result-{index:03d}{suffix}"
        registry.add_source(
            tenant_id=tenant_id,
            asset_id=asset.asset_id,
            filename=filename,
            media_type=media_type,
            object_uri=image_path.resolve().as_uri(),
            content_sha256=digest,
        )
        added.append(result.record_id)
    if not added:
        raise ValueError("search result images are unavailable on disk")
    return asset.asset_id, asset.name, len(added)


def _tenant_asset_ids(dependencies: "SemanticApiDependencies", tenant_id: UUID | None) -> set[str] | None:
    """The tenant's assets, or None when no registry/tenant is available."""

    registry = dependencies.registry or (dependencies.indexer.registry if dependencies.indexer else None)
    if registry is None or tenant_id is None:
        return None
    return {str(item.asset_id) for item in registry.list_assets(tenant_id=tenant_id)}


def _scoped_asset_ids(
    dependencies: "SemanticApiDependencies", tenant_id: UUID | None, requested: list[str] | None
) -> set[str] | None:
    """Search only the tenant's assets; requested ids outside it match nothing."""

    owned = _tenant_asset_ids(dependencies, tenant_id)
    if requested is None:
        return owned
    if owned is None:
        return set(requested)
    return set(requested) & owned


def _apply_min_score(results: list[SearchResult], min_score: float | None) -> list[SearchResult]:
    if min_score is None:
        return results
    return [result for result in results if result.score >= min_score]


def _validate_request(request: SearchRequest) -> None:
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")
    if request.limit < 1:
        raise HTTPException(status_code=422, detail="limit must be at least 1")


def _authorize(
    authorizer: Authorizer | None,
    *,
    credential: str | None,
    tenant_id: UUID | None,
    role: str,
) -> None:
    if authorizer is None:
        return
    if tenant_id is None:
        raise HTTPException(status_code=400, detail="X-Tenant-ID is required")
    try:
        authorizer.authorize(credential=credential, tenant_id=tenant_id, role=role)
    except PermissionError as exc:
        raise HTTPException(status_code=401 if credential is None else 403, detail=str(exc)) from exc


def _credential(authorization: str | None, api_key: str | None) -> str | None:
    return authorization or api_key


def _index_job_response(job: JobRecord) -> IndexJobResponse:
    return IndexJobResponse(
        job_id=job.job_id,
        asset_id=UUID(str(job.payload.get("asset_id"))),
        kind=job.kind,
        status=job.status,
        attempts=job.attempts,
        error=job.error,
    )


def _sub_asset_response(
    record: SubAssetRecord,
    *,
    registry: AssetRegistryBackend | None,
    tenant_id: UUID,
    source_count: int | None = None,
) -> SubAssetResponse:
    names: dict[str, str] = {}
    if registry is not None:
        names = {
            str(asset.asset_id): asset.name
            for asset in registry.list_assets(tenant_id=tenant_id)
        }
    return SubAssetResponse(
        sub_asset_id=record.sub_asset_id,
        parent_asset_ids=record.parent_asset_ids,
        parent_asset_names=[
            names.get(parent_id, "Source asset") for parent_id in record.parent_asset_ids
        ],
        query=record.query,
        result_ids=record.result_ids,
        asset_id=record.asset_id,
        asset_name=names.get(str(record.asset_id)) if record.asset_id else None,
        source_count=(
            len(record.result_ids) if source_count is None else source_count
        ),
        created_at=record.created_at,
    )


def create_app(dependencies: SemanticApiDependencies) -> FastAPI:
    app = FastAPI(title="BotFlow semantic search API", version="1")
    install_model_unavailable_handler(app)

    @app.post("/v1/search", response_model=SearchResponse)
    def search(
        request: SearchRequest,
        x_tenant_id: str | None = Header(None),
        tenant_id_q: str | None = Query(None, alias="tenant_id"),
        x_api_key: str | None = Header(None),
        authorization: str | None = Header(None),
    ) -> SearchResponse:
        raw_tenant = x_tenant_id or tenant_id_q
        tenant_id = _tenant(raw_tenant) if raw_tenant is not None else None
        _authorize(dependencies.authorizer, credential=_credential(authorization, x_api_key), tenant_id=tenant_id, role="search:read")
        _validate_request(request)
        try:
            results = dependencies.search.search(
                request.query,
                asset_ids=_scoped_asset_ids(dependencies, tenant_id, request.asset_ids),
                limit=request.limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _result_response(_apply_min_score(results, request.min_score), tenant_id=tenant_id)

    @app.get("/v1/search/records/{record_id}/image")
    def search_record_image(
        record_id: str,
        x_tenant_id: str | None = Header(None),
        tenant_id_q: str | None = Query(None, alias="tenant_id"),
        x_api_key: str | None = Header(None),
        authorization: str | None = Header(None),
    ) -> FileResponse:
        raw_tenant = x_tenant_id or tenant_id_q
        if raw_tenant is None:
            raise HTTPException(status_code=400, detail="X-Tenant-ID or tenant_id is required")
        tenant_id = _tenant(raw_tenant)
        _authorize(dependencies.authorizer, credential=_credential(authorization, x_api_key), tenant_id=tenant_id, role="search:read")
        record = dependencies.search.get_record(record_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown search record")
        registry = dependencies.registry or (dependencies.indexer.registry if dependencies.indexer else None)
        if registry is None:
            raise HTTPException(status_code=503, detail="asset registry is not configured")
        try:
            tenant_asset_ids = {str(item.asset_id) for item in registry.list_assets(tenant_id=tenant_id)}
        except Exception as exc:
            raise HTTPException(status_code=500, detail="failed to verify asset ownership") from exc
        if str(record.asset_id) not in tenant_asset_ids:
            raise HTTPException(status_code=404, detail="unknown search record")
        if not record.image_path.exists():
            raise HTTPException(status_code=404, detail="search result image unavailable")
        return FileResponse(record.image_path)

    @app.post("/v1/assets/{asset_id}/index", response_model=IndexAssetResponse)
    def index_asset(
        asset_id: UUID,
        x_tenant_id: str = Header(...),
        x_api_key: str | None = Header(None),
        authorization: str | None = Header(None),
    ) -> IndexAssetResponse:
        tenant_id = _tenant(x_tenant_id)
        _authorize(dependencies.authorizer, credential=_credential(authorization, x_api_key), tenant_id=tenant_id, role="assets:write")
        if dependencies.indexer is None:
            raise HTTPException(status_code=503, detail="semantic indexing is not configured")
        try:
            result = dependencies.indexer.index_asset(tenant_id=tenant_id, asset_id=asset_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return IndexAssetResponse(
            asset_id=result.asset_id,
            indexed_records=result.indexed_records,
            sources_indexed=result.sources_indexed,
            detail=result.detail,
        )

    if dependencies.index_worker is not None:
        @app.post("/v1/assets/{asset_id}/index-jobs", response_model=IndexJobResponse, status_code=202)
        def submit_index_job(
            asset_id: UUID,
            x_tenant_id: str = Header(...),
            x_api_key: str | None = Header(None),
            authorization: str | None = Header(None),
        ) -> IndexJobResponse:
            tenant_id = _tenant(x_tenant_id)
            _authorize(
                dependencies.authorizer,
                credential=_credential(authorization, x_api_key),
                tenant_id=tenant_id,
                role="assets:write",
            )
            registry = dependencies.registry or (
                dependencies.indexer.registry if dependencies.indexer else None
            )
            if registry is not None:
                known = {
                    str(asset.asset_id)
                    for asset in registry.list_assets(tenant_id=tenant_id)
                }
                if str(asset_id) not in known:
                    raise HTTPException(status_code=404, detail="unknown asset")
            job = dependencies.index_worker.submit(
                tenant_id=tenant_id, asset_id=asset_id
            )
            return _index_job_response(job)

        @app.get("/v1/index-jobs/{job_id}", response_model=IndexJobResponse)
        def get_index_job(
            job_id: UUID,
            x_tenant_id: str = Header(...),
            x_api_key: str | None = Header(None),
            authorization: str | None = Header(None),
        ) -> IndexJobResponse:
            tenant_id = _tenant(x_tenant_id)
            _authorize(
                dependencies.authorizer,
                credential=_credential(authorization, x_api_key),
                tenant_id=tenant_id,
                role="assets:read",
            )
            try:
                job = dependencies.index_worker.jobs.get(job_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            if job.tenant_id != tenant_id or job.kind != INDEX_JOB_KIND:
                raise HTTPException(status_code=404, detail="unknown index job")
            return _index_job_response(job)

    @app.get("/v1/sub-assets", response_model=list[SubAssetResponse])
    def list_sub_assets(
        x_tenant_id: str = Header(...),
        x_api_key: str | None = Header(None),
        authorization: str | None = Header(None),
    ) -> list[SubAssetResponse]:
        tenant_id = _tenant(x_tenant_id)
        _authorize(
            dependencies.authorizer,
            credential=_credential(authorization, x_api_key),
            tenant_id=tenant_id,
            role="assets:read",
        )
        registry = dependencies.registry or (
            dependencies.indexer.registry if dependencies.indexer else None
        )
        return [
            _sub_asset_response(record, registry=registry, tenant_id=tenant_id)
            for record in dependencies.sub_assets.list_for_tenant(tenant_id)
        ]

    @app.post("/v1/sub-assets/from-search", response_model=SubAssetResponse, status_code=201)
    def create_sub_asset(
        request: SearchRequest,
        x_tenant_id: str = Header(...),
        x_api_key: str | None = Header(None),
        authorization: str | None = Header(None),
    ) -> SubAssetResponse:
        tenant_id = _tenant(x_tenant_id)
        _authorize(dependencies.authorizer, credential=_credential(authorization, x_api_key), tenant_id=tenant_id, role="subassets:write")
        _validate_request(request)
        try:
            if request.record_ids is not None:
                selected = list(dict.fromkeys(item for item in request.record_ids if item))
                if not selected:
                    raise ValueError("select at least one search result")
                try:
                    results = dependencies.search.score_records(request.query, selected)
                except KeyError as exc:
                    raise HTTPException(status_code=404, detail="unknown search record") from exc
                owned = _tenant_asset_ids(dependencies, tenant_id)
                if owned is not None and any(result.asset_id not in owned for result in results):
                    raise HTTPException(status_code=404, detail="unknown search record")
            else:
                results = _apply_min_score(
                    dependencies.search.search(
                        request.query,
                        asset_ids=_scoped_asset_ids(dependencies, tenant_id, request.asset_ids),
                        limit=request.limit,
                    ),
                    request.min_score,
                )
            record = dependencies.sub_assets.create_from_results(
                tenant_id=tenant_id, query=request.query, results=results
            )
            source_count = 0
            registry = dependencies.registry or (dependencies.indexer.registry if dependencies.indexer else None)
            if registry is not None:
                asset_id, _, source_count = _materialize_search_asset(
                    registry=registry,
                    tenant_id=tenant_id,
                    query=request.query,
                    results=results,
                    name=request.name,
                )
                record = dependencies.sub_assets.link_asset(
                    record.sub_asset_id, asset_id=asset_id
                )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _sub_asset_response(
            record,
            registry=registry,
            tenant_id=tenant_id,
            source_count=source_count,
        )

    return app
