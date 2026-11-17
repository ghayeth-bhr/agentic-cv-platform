from uuid import uuid4

import pytest

from agentic_cv.dataset import DatasetCategory, DatasetImage, DatasetVersion
from agentic_cv.dataset_export import DatasetExportService
from agentic_cv.dataset_export_workflow import (
    DatasetExportWorkflow,
    DatasetExportWorkflowDependencies,
)
from agentic_cv.dataset_store import DatasetVersionStore
from agentic_cv.job_store import JobStore
from agentic_cv.label_workflow import JobCancelled
from agentic_cv.outbox import Outbox


def make_workflow(tmp_path):
    versions = DatasetVersionStore(str(tmp_path / "versions.sqlite3"))
    jobs = JobStore(str(tmp_path / "jobs.sqlite3"))
    outbox = Outbox(str(tmp_path / "events.sqlite3"))
    exports = DatasetExportService(versions, tmp_path / "exports")
    workflow = DatasetExportWorkflow(
        DatasetExportWorkflowDependencies(jobs=jobs, versions=versions, exports=exports, outbox=outbox)
    )
    return workflow, versions, jobs, outbox


def test_export_workflow_is_durable_and_emits_completion(tmp_path):
    workflow, versions, jobs, outbox = make_workflow(tmp_path)
    tenant_id = uuid4()
    dataset = DatasetVersion(
        images=(DatasetImage(image_id="image-1", file_name="frame.jpg", width=2, height=2),),
        categories=(DatasetCategory(category_id=0, name="object"),), annotations=(),
    )
    version = versions.create(tenant_id=tenant_id, name="v1", dataset=dataset, idempotency_key="v1")
    job = workflow.submit(
        tenant_id=tenant_id, version_id=version.version_id, format="coco", idempotency_key="export-1"
    )

    result = workflow.execute(job.job_id, tenant_id=tenant_id, worker_id="worker-a")

    assert result.output_dir.joinpath("annotations.json").exists()
    assert jobs.get(job.job_id).status == "succeeded"
    assert outbox.pending()[0].event_type == "DatasetExportCompleted"
    versions.close(); jobs.close(); outbox.close()


def test_export_workflow_rejects_pre_cancelled_job(tmp_path):
    workflow, versions, jobs, outbox = make_workflow(tmp_path)
    tenant_id = uuid4()
    dataset = DatasetVersion(
        images=(DatasetImage(image_id="image-1", file_name="frame.jpg", width=2, height=2),),
        categories=(DatasetCategory(category_id=0, name="object"),), annotations=(),
    )
    version = versions.create(tenant_id=tenant_id, name="v1", dataset=dataset, idempotency_key="v1")
    job = workflow.submit(
        tenant_id=tenant_id, version_id=version.version_id, format="coco", idempotency_key="export-1"
    )
    jobs.request_cancel(job.job_id)

    with pytest.raises(JobCancelled, match="before execution"):
        workflow.execute(job.job_id, tenant_id=tenant_id)
    assert jobs.get(job.job_id).status == "cancelled"
    versions.close(); jobs.close(); outbox.close()