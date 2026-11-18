"""Local MVP composition root for the Agentic CV platform."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping

import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api import ApiDependencies, CalibrationApiDependencies, ExportApiDependencies, LabelApiDependencies, create_app as create_core_app
from .observability import collect_metrics
from .annotation_store import AnnotationRevisionBackend, AnnotationRevisionStore
from .asset_registry import AssetRegistry, AssetRegistryBackend
from .auth import Authorizer, OidcJwtAuthorizer, Principal, StaticApiKeyAuthorizer
from .batch_labeling import Sam3BatchLabeler
from .broker import RabbitMqConfig, RabbitMqPublisher
from .dataset_export import DatasetExportService
from .dataset_export_workflow import DatasetExportWorkflow, DatasetExportWorkflowDependencies
from .dataset_store import DatasetVersionBackend, DatasetVersionStore
from .embeddings import HuggingFaceEmbeddingConfig, HuggingFaceEmbeddingProvider
from .job_store import JobStore, JobStoreBackend
from .label_workflow import DurableLabelWorkflow, LabelWorkflowDependencies
from .object_storage import LocalObjectStorage, ObjectStorage, S3ObjectStorage
from .orchestrator import TaskOrchestrator
from .outbox import Outbox, OutboxBackend
from .outbox_dispatcher import EventPublisher, OutboxDispatcher
from .postgres import PostgresConfig
from .preview import PreviewBackend, PreviewStore
from .preview_api import PreviewApiDependencies, create_app as create_preview_app
from .sam3 import LocalSam3Provider, Sam3Config, Sam3Provider
from .sam3_service import create_app as create_sam3_app
from .semantic_api import SemanticApiDependencies, create_app as create_semantic_app
from .semantic_search import EmbeddingProvider, LocalVectorIndex, SemanticSearchService, VectorIndex
from .sub_assets import SubAssetBackend, SubAssetStore
from .uploads import ResumableUploadService


@dataclass(frozen=True, kw_only=True)
class LocalAppConfig:
    data_dir: Path
    api_keys: Mapping[str, Principal] | None = None
    authorizer: Authorizer | None = None
    object_storage: ObjectStorage | None = None
    event_publisher: EventPublisher | None = None
    embedding_provider: EmbeddingProvider | None = None
    sam3_provider: Sam3Provider | None = None
    postgres: PostgresConfig | None = None

    @classmethod
    def from_environment(cls, data_dir: Path) -> "LocalAppConfig":
        issuer = os.getenv("AGENTIC_CV_OIDC_ISSUER")
        audience = os.getenv("AGENTIC_CV_OIDC_AUDIENCE")
        jwks_url = os.getenv("AGENTIC_CV_OIDC_JWKS_URL")
        authorizer = None
        if issuer and audience and jwks_url:
            authorizer = OidcJwtAuthorizer(
                issuer=issuer, audience=audience, jwks_url=jwks_url,
                tenant_claim=os.getenv("AGENTIC_CV_OIDC_TENANT_CLAIM", "tenant_id"),
                roles_claim=os.getenv("AGENTIC_CV_OIDC_ROLES_CLAIM", "roles"),
            )
        bucket = os.getenv("AGENTIC_CV_S3_BUCKET")
        object_storage = None
        if bucket:
            object_storage = S3ObjectStorage.from_environment(
                bucket=bucket,
                endpoint_url=os.getenv("AGENTIC_CV_S3_ENDPOINT_URL"),
                region_name=os.getenv("AWS_REGION"),
            )
        event_publisher = None
        if os.getenv("AGENTIC_CV_RABBITMQ_HOST"):
            event_publisher = RabbitMqPublisher(RabbitMqConfig.from_environment())
        return cls(
            data_dir=data_dir, authorizer=authorizer, object_storage=object_storage,
            event_publisher=event_publisher, postgres=PostgresConfig.from_environment(),
        )


@dataclass
class LocalAppResources:
    registry: AssetRegistryBackend
    outbox: OutboxBackend
    versions: DatasetVersionBackend
    jobs: JobStoreBackend
    preview: PreviewBackend
    sub_assets: SubAssetBackend
    annotations: AnnotationRevisionBackend
    vector_index: VectorIndex
    embedding_provider: EmbeddingProvider
    sam3_provider: Sam3Provider
    outbox_dispatcher: OutboxDispatcher | None = None

    def close(self) -> None:
        for resource in (
            self.registry, self.outbox, self.versions, self.jobs, self.preview, self.sub_assets, self.annotations,
        ):
            resource.close()
        for component in (self.vector_index, self.embedding_provider, self.sam3_provider):
            close = getattr(component, "close", None)
            if close is not None:
                close()
        if self.outbox_dispatcher is not None:
            close = getattr(self.outbox_dispatcher.publisher, "close", None)
            if close is not None:
                close()


def create_local_app(config: LocalAppConfig) -> FastAPI:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = config.data_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    if config.postgres is not None:
        from .postgres_asset_registry import open_postgres_asset_registry

        registry = open_postgres_asset_registry(config.postgres)
    else:
        registry = AssetRegistry(str(metadata_dir / "assets.sqlite3"))
    if config.postgres is not None:
        from .postgres_outbox import open_postgres_outbox

        outbox = open_postgres_outbox(config.postgres)
    else:
        outbox = Outbox(str(metadata_dir / "outbox.sqlite3"))
    if config.postgres is not None:
        from .postgres_dataset_store import open_postgres_dataset_version_store

        versions = open_postgres_dataset_version_store(config.postgres)
    else:
        versions = DatasetVersionStore(str(metadata_dir / "versions.sqlite3"))
    if config.postgres is not None:
        from .postgres_job_store import open_postgres_job_store

        jobs = open_postgres_job_store(config.postgres)
    else:
        jobs = JobStore(str(metadata_dir / "jobs.sqlite3"))
    if config.postgres is not None:
        from .postgres_preview import open_postgres_preview_store

        preview = open_postgres_preview_store(config.postgres)
    else:
        preview = PreviewStore(str(metadata_dir / "preview.sqlite3"))
    if config.postgres is not None:
        from .postgres_sub_assets import open_postgres_sub_asset_store

        sub_assets = open_postgres_sub_asset_store(config.postgres)
    else:
        sub_assets = SubAssetStore(str(metadata_dir / "sub-assets.sqlite3"))
    if config.postgres is not None:
        from .postgres_annotation_store import open_postgres_annotation_store

        annotations = open_postgres_annotation_store(config.postgres)
    else:
        annotations = AnnotationRevisionStore(str(metadata_dir / "annotations.sqlite3"))
    storage = config.object_storage or LocalObjectStorage(config.data_dir / "objects")
    uploads = ResumableUploadService(registry, storage)
    authorizer = config.authorizer or (
        StaticApiKeyAuthorizer(config.api_keys) if config.api_keys else None
    )
    embedding_provider = config.embedding_provider or HuggingFaceEmbeddingProvider(
        HuggingFaceEmbeddingConfig()
    )
    sam3_provider = config.sam3_provider or LocalSam3Provider(Sam3Config())
    if config.postgres is not None:
        from .postgres_pgvector import open_postgres_vector_index

        vector_index = open_postgres_vector_index(config.postgres)
    else:
        vector_index = LocalVectorIndex()
    resources = LocalAppResources(
        registry=registry, outbox=outbox, versions=versions, jobs=jobs, preview=preview,
        sub_assets=sub_assets, annotations=annotations, vector_index=vector_index,
        embedding_provider=embedding_provider, sam3_provider=sam3_provider,
    )
    export_service = DatasetExportService(versions, config.data_dir / "exports")
    export_workflow = DatasetExportWorkflow(
        DatasetExportWorkflowDependencies(
            jobs=jobs, versions=versions, exports=export_service, outbox=outbox
        )
    )
    label_workflow = DurableLabelWorkflow(
        LabelWorkflowDependencies(
            jobs=jobs, annotations=annotations, outbox=outbox,
            labeler=Sam3BatchLabeler(sam3_provider), preview=preview,
        )
    )
    outbox_dispatcher = (
        OutboxDispatcher(outbox=outbox, publisher=config.event_publisher)
        if config.event_publisher is not None
        else None
    )
    resources.outbox_dispatcher = outbox_dispatcher

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            resources.close()

    app = FastAPI(title="Agentic CV Platform", version="1", lifespan=lifespan)
    http_logger = logging.getLogger("agentic_cv.http")

    @app.middleware("http")
    async def observability_middleware(request: Request, call_next):
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((time.perf_counter() - start) * 1000, 3)
            http_logger.exception(
                "request failed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": 500,
                    "duration_ms": duration_ms,
                },
            )
            raise
        duration_ms = round((time.perf_counter() - start) * 1000, 3)
        http_logger.info(
            "request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response

    web_dir = Path(__file__).parent / "web"
    app.mount("/static", StaticFiles(directory=web_dir), name="static")
    app.state.outbox_dispatcher = outbox_dispatcher
    app.include_router(
        create_core_app(ApiDependencies(
            registry=registry, uploads=uploads, outbox=outbox,
            authorizer=authorizer,
            exports=ExportApiDependencies(workflow=export_workflow),
            labels=LabelApiDependencies(workflow=label_workflow),
            calibration=CalibrationApiDependencies(provider=sam3_provider),
            orchestrator=TaskOrchestrator(),
        )).router
    )
    app.include_router(
        create_semantic_app(SemanticApiDependencies(
            search=SemanticSearchService(embedding_provider, vector_index),
            sub_assets=sub_assets, authorizer=authorizer,
        )).router
    )
    app.include_router(create_preview_app(PreviewApiDependencies(store=preview, authorizer=authorizer)).router)
    app.include_router(create_sam3_app(sam3_provider).router)

    @app.get("/api-info")
    def application_info() -> dict[str, object]:
        return {
            "name": "Agentic CV Platform",
            "status": "ok",
            "docs": "/docs",
            "health": {"live": "/health/live", "ready": "/health/ready", "alias": "/health"},
            "observability": {"metrics": "/v1/metrics"},
            "api": {
                "assets": "/v1/assets",
                "search": "/v1/search",
                "exports": "/v1/dataset-exports",
                "labels": "/v1/label-jobs",
                "calibration": "/v1/calibration/run",
                "orchestrate": "/v1/orchestrate",
                "sam3": "/v1/predict",
            },
        }

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(web_dir / "index.html")

    @app.get("/health/live")
    def liveness() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    def readiness() -> dict[str, str]:
        return {"status": "ready"}

    @app.get("/health")
    def health() -> dict[str, str]:
        """Generic health alias for probes expecting /health."""
        return {"status": "ok"}

    @app.get("/v1/metrics")
    def metrics() -> dict[str, object]:
        return collect_metrics()

    app.state.resources = resources
    return app