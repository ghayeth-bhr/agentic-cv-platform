"""Local MVP composition root for the Agentic CV platform."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from fastapi import FastAPI

from .api import ApiDependencies, ExportApiDependencies, create_app as create_core_app
from .asset_registry import AssetRegistry
from .auth import Principal, StaticApiKeyAuthorizer
from .dataset_export import DatasetExportService
from .dataset_export_workflow import DatasetExportWorkflow, DatasetExportWorkflowDependencies
from .dataset_store import DatasetVersionStore
from .embeddings import HuggingFaceEmbeddingConfig, HuggingFaceEmbeddingProvider
from .job_store import JobStore
from .object_storage import LocalObjectStorage
from .outbox import Outbox
from .preview import PreviewStore
from .preview_api import PreviewApiDependencies, create_app as create_preview_app
from .sam3 import LocalSam3Provider, Sam3Config, Sam3Provider
from .sam3_service import create_app as create_sam3_app
from .semantic_api import SemanticApiDependencies, create_app as create_semantic_app
from .semantic_search import EmbeddingProvider, LocalVectorIndex, SemanticSearchService
from .sub_assets import SubAssetStore
from .uploads import ResumableUploadService


@dataclass(frozen=True, kw_only=True)
class LocalAppConfig:
    data_dir: Path
    api_keys: Mapping[str, Principal] | None = None
    embedding_provider: EmbeddingProvider | None = None
    sam3_provider: Sam3Provider | None = None


@dataclass
class LocalAppResources:
    registry: AssetRegistry
    outbox: Outbox
    versions: DatasetVersionStore
    jobs: JobStore
    preview: PreviewStore
    sub_assets: SubAssetStore
    embedding_provider: EmbeddingProvider
    sam3_provider: Sam3Provider

    def close(self) -> None:
        for resource in (
            self.registry, self.outbox, self.versions, self.jobs, self.preview, self.sub_assets,
        ):
            resource.close()
        for provider in (self.embedding_provider, self.sam3_provider):
            close = getattr(provider, "close", None)
            if close is not None:
                close()


def create_local_app(config: LocalAppConfig) -> FastAPI:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = config.data_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    registry = AssetRegistry(str(metadata_dir / "assets.sqlite3"))
    outbox = Outbox(str(metadata_dir / "outbox.sqlite3"))
    versions = DatasetVersionStore(str(metadata_dir / "versions.sqlite3"))
    jobs = JobStore(str(metadata_dir / "jobs.sqlite3"))
    preview = PreviewStore(str(metadata_dir / "preview.sqlite3"))
    sub_assets = SubAssetStore(str(metadata_dir / "sub-assets.sqlite3"))
    storage = LocalObjectStorage(config.data_dir / "objects")
    uploads = ResumableUploadService(registry, storage)
    authorizer = StaticApiKeyAuthorizer(config.api_keys) if config.api_keys else None
    embedding_provider = config.embedding_provider or HuggingFaceEmbeddingProvider(
        HuggingFaceEmbeddingConfig()
    )
    sam3_provider = config.sam3_provider or LocalSam3Provider(Sam3Config())
    resources = LocalAppResources(
        registry=registry, outbox=outbox, versions=versions, jobs=jobs, preview=preview,
        sub_assets=sub_assets, embedding_provider=embedding_provider, sam3_provider=sam3_provider,
    )
    export_service = DatasetExportService(versions, config.data_dir / "exports")
    export_workflow = DatasetExportWorkflow(
        DatasetExportWorkflowDependencies(
            jobs=jobs, versions=versions, exports=export_service, outbox=outbox
        )
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            resources.close()

    app = FastAPI(title="Agentic CV Platform", version="1", lifespan=lifespan)
    app.include_router(
        create_core_app(ApiDependencies(
            registry=registry, uploads=uploads, outbox=outbox,
            authorizer=authorizer,
            exports=ExportApiDependencies(workflow=export_workflow),
        )).router
    )
    app.include_router(
        create_semantic_app(SemanticApiDependencies(
            search=SemanticSearchService(embedding_provider, LocalVectorIndex()),
            sub_assets=sub_assets, authorizer=authorizer,
        )).router
    )
    app.include_router(create_preview_app(PreviewApiDependencies(store=preview, authorizer=authorizer)).router)
    app.include_router(create_sam3_app(sam3_provider).router)

    @app.get("/")
    def application_info() -> dict[str, object]:
        return {
            "name": "Agentic CV Platform",
            "status": "ok",
            "docs": "/docs",
            "health": {"live": "/health/live", "ready": "/health/ready"},
            "api": {
                "assets": "/v1/assets",
                "search": "/v1/search",
                "exports": "/v1/dataset-exports",
                "sam3": "/v1/predict",
            },
        }

    @app.get("/health/live")
    def liveness() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    def readiness() -> dict[str, str]:
        return {"status": "ready"}

    app.state.resources = resources
    return app