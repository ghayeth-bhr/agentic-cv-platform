from uuid import uuid4

from fastapi.testclient import TestClient

from agentic_cv.preview import PreviewStore
from agentic_cv.preview_api import PreviewApiDependencies, create_app
from agentic_cv.auth import Principal, StaticApiKeyAuthorizer


def test_preview_endpoint_replays_events_and_supports_cursor(tmp_path):
    store = PreviewStore(str(tmp_path / "preview.sqlite3"))
    tenant_id = uuid4()
    job_id = uuid4()
    first = store.publish(tenant_id=tenant_id, job_id=job_id, image_id="frame-1", processed=1, total=2)
    store.publish(tenant_id=tenant_id, job_id=job_id, image_id="frame-2", processed=2, total=2)
    client = TestClient(create_app(PreviewApiDependencies(store=store)))

    response = client.get(
        f"/v1/jobs/{job_id}/preview", headers={"X-Tenant-ID": str(tenant_id)},
    )
    resumed = client.get(
        f"/v1/jobs/{job_id}/preview?after_event_id={first.event_id}",
        headers={"X-Tenant-ID": str(tenant_id)},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert f"id: {first.event_id}" in response.text
    assert "frame-2" in resumed.text
    assert f"id: {first.event_id}" not in resumed.text
    store.close()


def test_preview_endpoint_is_tenant_scoped_and_validates_header(tmp_path):
    store = PreviewStore(str(tmp_path / "preview.sqlite3"))
    job_id = uuid4()
    store.publish(tenant_id=uuid4(), job_id=job_id, image_id="frame", processed=1, total=1)
    client = TestClient(create_app(PreviewApiDependencies(store=store)))

    wrong_tenant = client.get(f"/v1/jobs/{job_id}/preview", headers={"X-Tenant-ID": str(uuid4())})
    bad_header = client.get(f"/v1/jobs/{job_id}/preview", headers={"X-Tenant-ID": "bad"})

    assert wrong_tenant.status_code == 200
    assert wrong_tenant.text == ""
    assert bad_header.status_code == 400
    store.close()


def test_preview_api_optional_reader_auth(tmp_path):
    store = PreviewStore(str(tmp_path / "preview.sqlite3"))
    tenant_id = uuid4()
    job_id = uuid4()
    store.publish(tenant_id=tenant_id, job_id=job_id, image_id="frame", processed=1, total=1)
    authorizer = StaticApiKeyAuthorizer({
        "reader-key": Principal(tenant_id=tenant_id, roles=frozenset({"reader"})),
    })
    client = TestClient(create_app(PreviewApiDependencies(store=store, authorizer=authorizer)))

    missing = client.get(f"/v1/jobs/{job_id}/preview", headers={"X-Tenant-ID": str(tenant_id)})
    valid = client.get(
        f"/v1/jobs/{job_id}/preview",
        headers={"X-Tenant-ID": str(tenant_id), "X-API-Key": "reader-key"},
    )
    wrong = client.get(
        f"/v1/jobs/{job_id}/preview",
        headers={"X-Tenant-ID": str(uuid4()), "X-API-Key": "reader-key"},
    )

    assert missing.status_code == 401
    assert valid.status_code == 200
    assert wrong.status_code == 403
    store.close()
