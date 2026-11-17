import hashlib
from io import BytesIO
from uuid import uuid4

from fastapi.testclient import TestClient

from agentic_cv.api import ApiDependencies, create_app
from agentic_cv.asset_registry import AssetRegistry
from agentic_cv.auth import Principal, StaticApiKeyAuthorizer
from agentic_cv.object_storage import LocalObjectStorage
from agentic_cv.outbox import Outbox
from agentic_cv.uploads import ResumableUploadService


def make_client(tmp_path, *, authorizer=None):
    registry = AssetRegistry(str(tmp_path / "metadata.sqlite3"))
    storage = LocalObjectStorage(tmp_path / "objects")
    outbox = Outbox(str(tmp_path / "events.sqlite3"))
    service = ResumableUploadService(registry, storage)
    client = TestClient(create_app(ApiDependencies(
        registry=registry, uploads=service, outbox=outbox, authorizer=authorizer
    )))
    return client, registry, outbox


def test_asset_upload_flow_registers_source_and_event(tmp_path):
    client, registry, outbox = make_client(tmp_path)
    tenant_id = uuid4()
    content = b"frame-content"
    digest = hashlib.sha256(content).hexdigest()

    asset_response = client.post(
        "/v1/assets", headers={"X-Tenant-ID": str(tenant_id)}, json={"name": "Wildlife"}
    )
    assert asset_response.status_code == 201
    asset_id = asset_response.json()["asset_id"]
    upload_response = client.post(
        f"/v1/assets/{asset_id}/uploads",
        headers={"X-Tenant-ID": str(tenant_id)},
        json={"filename": "frame.jpg", "media_type": "image/jpeg", "expected_sha256": digest},
    )
    upload_id = upload_response.json()["upload_id"]

    part_response = client.put(
        f"/v1/uploads/{upload_id}/parts/1",
        files={"part": ("part.bin", BytesIO(content), "application/octet-stream")},
    )
    completed = client.post(f"/v1/uploads/{upload_id}/complete")

    assert upload_response.status_code == 201
    assert part_response.status_code == 200
    assert completed.status_code == 200
    assert completed.json()["content_sha256"] == digest
    pending = outbox.pending()
    assert len(pending) == 1
    assert pending[0].event_type == "SourceReady"
    registry.close()
    outbox.close()


def test_asset_upload_rejects_wrong_tenant_and_bad_header(tmp_path):
    client, registry, outbox = make_client(tmp_path)
    owner = uuid4()
    asset = client.post(
        "/v1/assets", headers={"X-Tenant-ID": str(owner)}, json={"name": "Private"}
    ).json()

    wrong_tenant = client.post(
        f"/v1/assets/{asset['asset_id']}/uploads",
        headers={"X-Tenant-ID": str(uuid4())},
        json={"filename": "x.jpg", "media_type": "image/jpeg", "expected_sha256": "a" * 64},
    )
    bad_header = client.post(
        "/v1/assets", headers={"X-Tenant-ID": "not-a-uuid"}, json={"name": "Invalid"}
    )

    assert wrong_tenant.status_code == 403
    assert bad_header.status_code == 400
    registry.close()
    outbox.close()


def test_optional_api_key_auth_enforces_tenant_and_writer_role(tmp_path):
    tenant_id = uuid4()
    authorizer = StaticApiKeyAuthorizer({
        "writer-key": Principal(tenant_id=tenant_id, roles=frozenset({"writer"})),
    })
    client, registry, outbox = make_client(tmp_path, authorizer=authorizer)

    missing_key = client.post(
        "/v1/assets", headers={"X-Tenant-ID": str(tenant_id)}, json={"name": "Private"}
    )
    valid = client.post(
        "/v1/assets",
        headers={"X-Tenant-ID": str(tenant_id), "X-API-Key": "writer-key"},
        json={"name": "Private"},
    )
    bearer_valid = client.post(
        "/v1/assets",
        headers={"X-Tenant-ID": str(tenant_id), "Authorization": "Bearer writer-key"},
        json={"name": "Bearer asset"},
    )
    wrong_tenant = client.post(
        "/v1/assets",
        headers={"X-Tenant-ID": str(uuid4()), "X-API-Key": "writer-key"},
        json={"name": "Private"},
    )

    assert missing_key.status_code == 401
    assert valid.status_code == 201
    assert bearer_valid.status_code == 201
    assert wrong_tenant.status_code == 403
    registry.close()
    outbox.close()
