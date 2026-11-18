"""Composition and lifecycle helpers for local worker processes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .dataset_export import DatasetExportService
from .dataset_export_worker import DatasetExportWorker
from .dataset_export_workflow import DatasetExportWorkflow, DatasetExportWorkflowDependencies
from .dataset_store import DatasetVersionStore
from .job_store import JobStore, JobStoreBackend
from .outbox import Outbox, OutboxBackend
from .postgres import PostgresConfig


@dataclass
class ExportWorkerRuntime:
    worker: DatasetExportWorker
    versions: DatasetVersionStore
    jobs: JobStoreBackend
    outbox: OutboxBackend

    def close(self) -> None:
        self.versions.close()
        self.jobs.close()
        self.outbox.close()


def create_export_worker(data_dir: Path) -> ExportWorkerRuntime:
    data_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = data_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    versions = DatasetVersionStore(str(metadata_dir / "versions.sqlite3"))
    postgres = PostgresConfig.from_environment()
    if postgres is not None:
        from .postgres_job_store import open_postgres_job_store
        from .postgres_outbox import open_postgres_outbox

        jobs = open_postgres_job_store(postgres)
        outbox = open_postgres_outbox(postgres)
    else:
        jobs = JobStore(str(metadata_dir / "jobs.sqlite3"))
        outbox = Outbox(str(metadata_dir / "outbox.sqlite3"))
    workflow = DatasetExportWorkflow(
        DatasetExportWorkflowDependencies(
            jobs=jobs,
            versions=versions,
            exports=DatasetExportService(versions, data_dir / "exports"),
            outbox=outbox,
        )
    )
    return ExportWorkerRuntime(
        worker=DatasetExportWorker(jobs=jobs, workflow=workflow),
        versions=versions,
        jobs=jobs,
        outbox=outbox,
    )
