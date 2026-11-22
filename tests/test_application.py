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
        assert "BotFlow" in info.text
        assert api_info.json()["docs"] == "/docs"

        response = client.post(
            "/v1/assets", headers={"X-Tenant-ID": "00000000-0000-0000-0000-000000000001"},
            json={"name": "MVP asset"},
        )

    assert response.status_code == 201


def test_local_application_uses_cpu_model_runtime_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTIC_CV_DEVICE", "cpu")
    monkeypatch.delenv("AGENTIC_CV_PRECISION", raising=False)

    app = create_local_app(LocalAppConfig(data_dir=tmp_path / "data"))

    with TestClient(app) as client:
        runtime = client.get("/api-info").json()["model_runtime"]

    assert runtime["device"] == "cpu"
    assert runtime["precision"] == "float32"
    assert app.state.resources.embedding_provider.config.device == "cpu"
    assert app.state.resources.embedding_provider.config.precision == "float32"
    assert app.state.resources.sam3_provider.config.device == "cpu"


def test_local_application_reads_oidc_configuration_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTIC_CV_OIDC_ISSUER", "https://issuer.example")
    monkeypatch.setenv("AGENTIC_CV_OIDC_AUDIENCE", "agentic-cv")
    monkeypatch.setenv("AGENTIC_CV_OIDC_JWKS_URL", "https://issuer.example/jwks")

    config = LocalAppConfig.from_environment(tmp_path / "data")

    assert config.authorizer is not None

def test_application_config_keeps_sqlite_registry_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENTIC_CV_POSTGRES_DSN", raising=False)

    config = LocalAppConfig.from_environment(tmp_path / "data")

    assert config.postgres is None


def test_application_config_reads_postgres_opt_in(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTIC_CV_POSTGRES_DSN", "postgresql://postgres@localhost/cv")

    config = LocalAppConfig.from_environment(tmp_path / "data")

    assert config.postgres is not None
    assert config.postgres.dsn == "postgresql://postgres@localhost/cv"


def test_jobs_endpoint_lists_queued_index_job_without_other_tenants(tmp_path):
    app = create_local_app(
        LocalAppConfig(data_dir=tmp_path / "data", start_workers=False)
    )
    tenant_id = "00000000-0000-0000-0000-000000000001"
    headers = {"X-Tenant-ID": tenant_id}

    with TestClient(app) as client:
        asset = client.post("/v1/assets", headers=headers, json={"name": "Job asset"})
        assert asset.status_code == 201
        asset_id = asset.json()["asset_id"]

        queued = client.post(f"/v1/assets/{asset_id}/index-jobs", headers=headers)
        assert queued.status_code == 202

        jobs = client.get("/v1/jobs", headers=headers)

    assert jobs.status_code == 200
    values = jobs.json()["jobs"]
    assert len(values) == 1
    assert values[0]["kind"] == "index-asset"
    assert values[0]["status"] == "queued"
    assert "job_id" in values[0]


def test_ui_files_are_revalidated_so_updates_are_never_stale(tmp_path):
    from fastapi.testclient import TestClient

    from agentic_cv.application import LocalAppConfig, create_local_app

    app = create_local_app(LocalAppConfig(data_dir=tmp_path / "data", start_workers=False))
    with TestClient(app) as client:
        for path in ("/", "/static/app.js", "/static/app.css"):
            response = client.get(path)
            assert response.status_code == 200, path
            assert response.headers["cache-control"] == "no-cache", path
        script = client.get("/static/app.js")
        unchanged = client.get(
            "/static/app.js", headers={"If-None-Match": script.headers["etag"]}
        )
        assert unchanged.status_code == 304
