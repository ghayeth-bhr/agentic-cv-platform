from uuid import uuid4

from agentic_cv.dataset import DatasetCategory, DatasetImage, DatasetVersion
from agentic_cv.dataset_export import DatasetExportService
from agentic_cv.dataset_export_workflow import DatasetExportWorkflow, DatasetExportWorkflowDependencies
from agentic_cv.dataset_export_worker import DatasetExportWorker
from agentic_cv.dataset_store import DatasetVersionStore
from agentic_cv.job_store import JobStore
from agentic_cv.outbox import Outbox


def test_worker_processes_all_queued_export_jobs(tmp_path):
    versions = DatasetVersionStore(str(tmp_path / "versions.sqlite3"))
    jobs = JobStore(str(tmp_path / "jobs.sqlite3"))
    outbox = Outbox(str(tmp_path / "events.sqlite3"))
    workflow = DatasetExportWorkflow(DatasetExportWorkflowDependencies(
        jobs=jobs, versions=versions,
        exports=DatasetExportService(versions, tmp_path / "exports"), outbox=outbox,
    ))
    worker = DatasetExportWorker(jobs=jobs, workflow=workflow)
    tenant_id = uuid4()
    dataset = DatasetVersion(
        images=(DatasetImage(image_id="image-1", file_name="frame.jpg", width=2, height=2),),
        categories=(DatasetCategory(category_id=0, name="object"),), annotations=(),
    )
    version = versions.create(tenant_id=tenant_id, name="v1", dataset=dataset, idempotency_key="version")
    jobs_created = [
        workflow.submit(tenant_id=tenant_id, version_id=version.version_id, format="coco", idempotency_key=f"export-{index}")
        for index in range(2)
    ]

    inspected = worker.run_once(worker_id="worker-a")

    assert inspected == tuple(job.job_id for job in jobs_created)
    assert all(jobs.get(job.job_id).status == "succeeded" for job in jobs_created)
    assert len(outbox.pending()) == 2
    versions.close(); jobs.close(); outbox.close()