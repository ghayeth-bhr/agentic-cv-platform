from fastapi.testclient import TestClient

from agentic_cv.application import LocalAppConfig, create_local_app


def test_local_application_assembles_health_and_core_routes(tmp_path):
    app = create_local_app(LocalAppConfig(data_dir=tmp_path / "data"))

    with TestClient(app) as client:
        info = client.get("/")
        api_info = client.get("/api-info")
        assert client.get("/health/live").json() == {"status": "ok"}
        assert client.get("/health/ready").json() == {"status": "ready"}
        assert info.status_code == 200
        assert "Agentic CV" in info.text
        assert api_info.json()["docs"] == "/docs"

        response = client.post(
            "/v1/assets", headers={"X-Tenant-ID": "00000000-0000-0000-0000-000000000001"},
            json={"name": "MVP asset"},
        )

    assert response.status_code == 201