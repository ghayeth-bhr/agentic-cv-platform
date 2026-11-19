"""Local MVP composition root for the Agentic CV platform."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from threading import Event, Thread
from typing import Callable, Mapping
from uuid import UUID

import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .agent import (
    AgentService,
    AgentServiceDependencies,
    TaskPlanner,
    planner_from_environment,
)
from .agent_api import AgentApiDependencies, create_app as create_agent_app
from .agent_store import AgentSessionBackend, AgentSessionStore
from .agent_worker import AgentWorker
from .api import ApiDependencies, CalibrationApiDependencies, DatasetApiDependencies, ExportApiDependencies, LabelApiDependencies, RefinementApiDependencies, create_app as create_core_app
from .observability import collect_metrics
from .annotation_store import AnnotationRevisionBackend, AnnotationRevisionStore
from .asset_registry import AssetRegistry, AssetRegistryBackend
from .auth import Authorizer, OidcJwtAuthorizer, Principal, StaticApiKeyAuthorizer
from .batch_labeling import Sam3BatchLabeler
from .broker import RabbitMqConfig, RabbitMqPublisher
from .calibration_validation import (
    ValidatorSelection,
    default_validator_factory,
)
from .click_refinement import (
    ClickRefinementProvider,
    ClickRefinementService,
    RefinementTarget,
    Sam3ClickRefinementProvider,
)
from .dataset_builder import DatasetBuilder
from .dataset_export import DatasetExportService
from .dataset_export_workflow import DatasetExportWorkflow, DatasetExportWorkflowDependencies
from .dataset_export_worker import DatasetExportWorker
from .dataset_store import DatasetVersionBackend, DatasetVersionStore
from .embeddings import HuggingFaceEmbeddingProvider
from .errors import install_model_unavailable_handler
from .index_worker import IndexWorker
from .job_store import JobStore, JobStoreBackend
from .label_images import LabelImageResolver
from .label_worker import LabelWorker
from .label_workflow import DurableLabelWorkflow, LabelWorkflowDependencies
from .model_resources import ModelResourceManager
from .model_runtime import ModelRuntimeSettings
from .object_storage import LocalObjectStorage, ObjectStorage, S3ObjectStorage
from .orchestrator import TaskOrchestrator
from .outbox import Outbox, OutboxBackend
from .outbox_dispatcher import EventPublisher, OutboxDispatcher
from .postgres import PostgresConfig
from .persistent_vectors import SqliteVectorIndex
from .preview import PreviewBackend, PreviewStore
from .preview_api import PreviewApiDependencies, create_app as create_preview_app
from .sam3 import LocalSam3Provider, Sam3Provider, load_image
from .sam3_service import create_app as create_sam3_app
from .semantic_api import SemanticApiDependencies, create_app as create_semantic_app
from .semantic_indexing import AssetSemanticIndexer
from .semantic_search import EmbeddingProvider, SemanticSearchService, VectorIndex
from .sub_assets import SubAssetBackend, SubAssetStore
from .uploads import ResumableUploadService


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, value)


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(minimum, value)


@dataclass(frozen=True, kw_only=True)
class LocalAppConfig:
    data_dir: Path
    api_keys: Mapping[str, Principal] | None = None
    authorizer: Authorizer | None = None
    object_storage: ObjectStorage | None = None
    event_publisher: EventPublisher | None = None
    embedding_provider: EmbeddingProvider | None = None
    sam3_provider: Sam3Provider | None = None
    refinement_provider: ClickRefinementProvider | None = None
    postgres: PostgresConfig | None = None
    agent_planner: TaskPlanner | None = None
    calibration_validator_factory: (
        Callable[[UUID, UUID], ValidatorSelection] | None
    ) = None
    calibration_timeout_sec: float = 120.0
    start_workers: bool = True
    worker_poll_interval: float = 0.25

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
        try:
            calibration_timeout = float(
                os.getenv("AGENTIC_CV_CALIBRATION_TIMEOUT_SEC", "120")
            )
        except ValueError:
            calibration_timeout = 120.0
        return cls(
            data_dir=data_dir, authorizer=authorizer, object_storage=object_storage,
            event_publisher=event_publisher, postgres=PostgresConfig.from_environment(),
            calibration_timeout_sec=calibration_timeout,
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
    agent_sessions: AgentSessionBackend
    model_resources: ModelResourceManager
    outbox_dispatcher: OutboxDispatcher | None = None

    def close(self) -> None:
        for resource in (
            self.registry, self.outbox, self.versions, self.jobs, self.preview, self.sub_assets, self.annotations,
            self.agent_sessions,
        ):
            resource.close()
        for component in (self.vector_index, self.embedding_provider, self.sam3_provider):
            close = getattr(component, "close", None)
            if close is not None:
                close()
        self.model_resources.close_all()
        if self.outbox_dispatcher is not None:
            close = getattr(self.outbox_dispatcher.publisher, "close", None)
            if close is not None:
                close()


def create_local_app(config: LocalAppConfig) -> FastAPI:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = config.data_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    model_settings = ModelRuntimeSettings.from_environment()
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
    agent_sessions = AgentSessionStore(str(metadata_dir / "agent-sessions.sqlite3"))
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
    model_resources = ModelResourceManager(
        device=model_settings.device,
        headroom_mb=_env_int("AGENTIC_CV_MODEL_HEADROOM_MB", 512),
        idle_unload_sec=_env_float("AGENTIC_CV_MODEL_IDLE_UNLOAD_SEC", 300.0),
    )
    embedding_provider = config.embedding_provider or HuggingFaceEmbeddingProvider(
        model_settings.embedding_config(),
        resources=model_resources,
        resource_name="siglip2",
        estimated_vram_mb=_env_int("AGENTIC_CV_EMBEDDING_VRAM_MB", 2304, minimum=1),
    )
    sam3_provider = config.sam3_provider or LocalSam3Provider(
        model_settings.sam3_config(),
        resources=model_resources,
        resource_name="sam3",
        estimated_vram_mb=_env_int("AGENTIC_CV_SAM3_VRAM_MB", 2048, minimum=1),
    )
    if config.postgres is not None:
        from .postgres_pgvector import open_postgres_vector_index

        vector_index = open_postgres_vector_index(config.postgres)
    else:
        vector_index = SqliteVectorIndex(str(metadata_dir / "vectors.sqlite3"))
    semantic_search = SemanticSearchService(embedding_provider, vector_index)
    semantic_indexer = AssetSemanticIndexer(
        registry=registry, search=semantic_search, data_dir=config.data_dir,
    )
    index_worker = IndexWorker(jobs=jobs, indexer=semantic_indexer)
    resources = LocalAppResources(
        registry=registry, outbox=outbox, versions=versions, jobs=jobs, preview=preview,
        sub_assets=sub_assets, annotations=annotations, vector_index=vector_index,
        embedding_provider=embedding_provider, sam3_provider=sam3_provider,
        agent_sessions=agent_sessions, model_resources=model_resources,
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
    label_images = LabelImageResolver(registry=registry, data_dir=config.data_dir)
    label_worker = LabelWorker(
        jobs=jobs, workflow=label_workflow, images=label_images
    )
    dataset_builder = DatasetBuilder(
        jobs=jobs,
        annotations=annotations,
        versions=versions,
        images=label_images,
        images_root=config.data_dir / "dataset-images",
    )
    dataset_images_root = config.data_dir / "dataset-images"
    refinement_provider = config.refinement_provider or Sam3ClickRefinementProvider(
        sam3_provider
    )
    refinement_service = ClickRefinementService(refinement_provider)

    def load_refinement_target(
        tenant_id, version_id, image_id
    ) -> RefinementTarget:
        record = versions.get(tenant_id=tenant_id, version_id=version_id)
        image = next(
            (item for item in record.dataset.images if item.image_id == image_id),
            None,
        )
        if image is None:
            raise KeyError("dataset image not found")
        categories = {
            category.category_id: category for category in record.dataset.categories
        }
        category = None
        for annotation in record.dataset.annotations:
            if annotation.image_id == image_id:
                category = categories.get(annotation.category_id)
                if category is not None:
                    break
        if category is None and record.dataset.categories:
            category = record.dataset.categories[0]
        if category is None:
            raise ValueError("dataset has no category to refine")
        root = (
            dataset_images_root / str(tenant_id) / str(version_id)
        ).resolve()
        path = (root / image.file_name).resolve()
        if root not in path.parents:
            raise ValueError("dataset image path is outside its version")
        return RefinementTarget(
            image=load_image(path),
            prompt=category.name,
            category_id=category.category_id,
        )
    export_worker = DatasetExportWorker(jobs=jobs, workflow=export_workflow)
    agent_service = AgentService(
        AgentServiceDependencies(
            store=agent_sessions,
            registry=registry,
            labels=label_workflow,
            jobs=jobs,
            dataset_builder=dataset_builder,
            export_workflow=export_workflow,
            images=label_images,
            sam3_provider=sam3_provider,
            planner=config.agent_planner or planner_from_environment(),
            calibration_timeout_sec=config.calibration_timeout_sec,
            validator_factory=(
                config.calibration_validator_factory
                or default_validator_factory(config.data_dir)
            ),
            vectors=vector_index,
            overlay_root=config.data_dir / "calibration-overlays",
        )
    )
    agent_worker = AgentWorker(service=agent_service)
    outbox_dispatcher = (
        OutboxDispatcher(outbox=outbox, publisher=config.event_publisher)
        if config.event_publisher is not None
        else None
    )
    resources.outbox_dispatcher = outbox_dispatcher
    worker_stop = Event()
    workers: list[Thread] = []

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            if config.start_workers:
                if config.worker_poll_interval < 0:
                    raise ValueError("worker_poll_interval must not be negative")
                for target, name in (
                    (label_worker.run_forever, "label-worker"),
                    (index_worker.run_forever, "index-worker"),
                    (export_worker.run_forever, "export-worker"),
                    (agent_worker.run_forever, "agent-worker"),
                ):
                    thread = Thread(
                        target=target,
                        kwargs={
                            "poll_interval": config.worker_poll_interval,
                            "stop_event": worker_stop,
                            "sleep_fn": worker_stop.wait,
                        },
                        name=name,
                        daemon=True,
                    )
                    thread.start()
                    workers.append(thread)
            yield
        finally:
            worker_stop.set()
            for thread in workers:
                thread.join(timeout=5)
                if thread.is_alive():
                    logging.getLogger("agentic_cv.worker").warning(
                        "worker thread did not stop before shutdown",
                        extra={"worker_thread": thread.name},
                    )
            resources.close()

    app = FastAPI(title="Agentic CV Platform", version="1", lifespan=lifespan)
    install_model_unavailable_handler(app)
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
            datasets=DatasetApiDependencies(
                builder=dataset_builder,
                versions=versions,
                images_root=dataset_images_root,
            ),
            refinement=RefinementApiDependencies(
                service=refinement_service,
                annotations=annotations,
                dataset_loader=load_refinement_target,
            ),
            calibration=CalibrationApiDependencies(provider=sam3_provider),
            orchestrator=TaskOrchestrator(),
            sam3=sam3_provider,
            label_images=label_images,
            annotations=annotations,
            jobs=jobs,
        )).router
    )
    app.include_router(
        create_semantic_app(SemanticApiDependencies(
            search=semantic_search,
            sub_assets=sub_assets,
            indexer=semantic_indexer,
            index_worker=index_worker,
            registry=registry,
            authorizer=authorizer,
        )).router
    )
    app.include_router(
        create_agent_app(
            AgentApiDependencies(service=agent_service, authorizer=authorizer)
        ).router
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
            "model_runtime": {
                "device": model_settings.device,
                "precision": model_settings.precision,
                "doctor": "agentic-cv doctor",
                "resources": model_resources.status().as_dict(),
            },
            "api": {
                "assets": "/v1/assets",
                "search": "/v1/search",
                "datasets": "/v1/datasets",
                "exports": "/v1/dataset-exports",
                "labels": "/v1/label-jobs",
                "calibration": "/v1/calibration/run",
                "orchestrate": "/v1/orchestrate",
                "agent": "/v1/agent/sessions",
                "sam3": "/v1/predict",
                "asset_predict": "/v1/assets/{asset_id}/predict",
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