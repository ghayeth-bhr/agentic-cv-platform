from uuid import uuid4

from fastapi.testclient import TestClient

from agentic_cv.api import ApiDependencies, ExportApiDependencies, create_app
from agentic_cv.asset_registry import AssetRegistry
from agentic_cv.auth import Principal, StaticApiKeyAuthorizer
from agentic_cv.dataset import DatasetCategory, DatasetImage, DatasetVersion
from agentic_cv.dataset_export import DatasetExportService
from agentic_cv.dataset_export_workflow import DatasetExportWorkflow, DatasetExportWorkflowDependencies
from agentic_cv.dataset_store import DatasetVersionStore
from agentic_cv.job_store import JobStore
from agentic_cv.object_storage import LocalObjectStorage
from agentic_cv.outbox import Outbox
from agentic_cv.uploads import ResumableUploadService


def test_dataset_export_api_submits_and_reads_tenant_job(tmp_path):
    tenant_id = uuid4()
    registry = AssetRegistry(str(tmp_path / "assets.sqlite3"))
    versions = DatasetVersionStore(str(tmp_path / "versions.sqlite3"))
    jobs = JobStore(str(tmp_path / "jobs.sqlite3"))
    outbox = Outbox(str(tmp_path / "events.sqlite3"))
    workflow = DatasetExportWorkflow(DatasetExportWorkflowDependencies(
        jobs=jobs, versions=versions,
        exports=DatasetExportService(versions, tmp_path / "exports"), outbox=outbox,
    ))
    version = versions.create(
        tenant_id=tenant_id, name="v1",
        dataset=DatasetVersion(
            images=(DatasetImage(image_id="image-1", file_name="frame.jpg", width=2, height=2),),
            categories=(DatasetCategory(category_id=0, name="object"),), annotations=(),
        ),
        idempotency_key="version-1",
    )
    authorizer = StaticApiKeyAuthorizer({
        "writer-key": Principal(tenant_id=tenant_id, roles=frozenset({"writer"})),
    })
    app = create_app(ApiDependencies(
        registry=registry,
        uploads=ResumableUploadService(registry, LocalObjectStorage(tmp_path / "objects")),
        outbox=outbox,
        authorizer=authorizer,
        exports=ExportApiDependencies(workflow=workflow),
    ))
    client = TestClient(app)
    headers = {"X-Tenant-ID": str(tenant_id), "X-API-Key": "writer-key"}

    submitted = client.post(
        "/v1/dataset-exports", headers=headers,
        json={"version_id": str(version.version_id), "format": "coco", "idempotency_key": "export-1"},
    )
    status = client.get(f"/v1/dataset-exports/{submitted.json()['job_id']}", headers=headers)

    assert submitted.status_code == 202
    assert submitted.json()["status"] == "queued"
    assert status.status_code == 200
    assert status.json()["kind"] == "dataset-export"
    assert status.json()["attempts"] == 0
    registry.close(); versions.close(); jobs.close(); outbox.close()