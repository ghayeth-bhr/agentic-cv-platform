"""Contracts between the shipped browser UI and the FastAPI surface.

The tests are intentionally shallow: they lock the routes, response envelopes,
and DOM ids the UI depends on so a backend rename or a view rename fails here
before it becomes a browser-only regression.
"""

from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from starlette.routing import WebSocketRoute

from agentic_cv.application import LocalAppConfig, create_local_app

TENANT_HEADERS = {"X-Tenant-ID": "00000000-0000-0000-0000-000000000001"}

# Methods and paths called by src/agentic_cv/web/app.js. Template parameter names
# must match too because the UI interpolates the same names.
UI_ROUTES = {
    "/api-info": {"get"},
    "/v1/assets": {"get", "post"},
    "/v1/assets/{asset_id}/image": {"get"},
    "/v1/assets/{asset_id}/uploads": {"post"},
    "/v1/assets/{asset_id}/index-jobs": {"post"},
    "/v1/uploads/{upload_id}/parts/{part_number}": {"put"},
    "/v1/uploads/{upload_id}/complete": {"post"},
    "/v1/index-jobs/{job_id}": {"get"},
    "/v1/sub-assets": {"get"},
    "/v1/sub-assets/from-search": {"post"},
    "/v1/search": {"post"},
    "/v1/agent/sessions": {"get", "post"},
    "/v1/agent/sessions/{session_id}": {"get"},
    "/v1/agent/sessions/{session_id}/confirm": {"post"},
    "/v1/agent/sessions/{session_id}/cancel": {"post"},
    "/v1/datasets": {"get"},
    "/v1/datasets/{version_id}/images": {"get"},
    "/v1/datasets/{version_id}/images/{image_id}/file": {"get"},
    "/v1/datasets/{version_id}/rebuild": {"post"},
    "/v1/refinements/{session_id}/accept": {"post"},
    "/v1/dataset-exports": {"post"},
    "/v1/dataset-exports/{job_id}": {"get"},
    "/v1/dataset-exports/{job_id}/download": {"get"},
    "/v1/jobs": {"get"},
}


def _app(tmp_path):
    return create_local_app(
        LocalAppConfig(data_dir=tmp_path / "data", start_workers=False)
    )


def _walk_routes(routes):
    for route in routes:
        nested = getattr(route, "routes", None)
        if nested is None:
            original = getattr(route, "original_router", None)
            nested = getattr(original, "routes", None) if original else None
        if nested:
            yield from _walk_routes(nested)
        else:
            yield route


def test_every_ui_http_route_exists_with_the_expected_method(tmp_path):
    app = _app(tmp_path)
    routes = {}
    websocket_paths = set()
    for route in _walk_routes(app.routes):
        if isinstance(route, APIRoute):
            routes.setdefault(route.path, set()).update(
                method.lower() for method in route.methods
            )
        elif isinstance(route, WebSocketRoute):
            websocket_paths.add(route.path)

    missing = {}
    for path, methods in UI_ROUTES.items():
        actual = routes.get(path, set())
        if not methods <= actual:
            missing[path] = {"expected": sorted(methods), "actual": sorted(actual)}
    assert not missing, missing

    assert "/v1/refinement/{version_id}/{image_id}" in websocket_paths


def test_empty_library_response_envelopes_are_stable(tmp_path):
    with TestClient(_app(tmp_path)) as client:
        info = client.get("/api-info")
        assert info.status_code == 200
        info_body = info.json()
        assert {"name", "status", "health", "observability", "model_runtime"} <= set(info_body)
        assert info_body["model_runtime"]["resources"]["models"].keys() == {
            "siglip2",
            "sam3",
        }

        assets = client.get("/v1/assets", headers=TENANT_HEADERS)
        assert assets.status_code == 200
        assert assets.json() == {"assets": []}

        sub_assets = client.get("/v1/sub-assets", headers=TENANT_HEADERS)
        assert sub_assets.status_code == 200
        assert isinstance(sub_assets.json(), list)

        jobs = client.get("/v1/jobs", headers=TENANT_HEADERS)
        assert jobs.status_code == 200
        assert jobs.json() == {"jobs": []}

        datasets = client.get("/v1/datasets", headers=TENANT_HEADERS)
        assert datasets.status_code == 200
        assert datasets.json() == []

        sessions = client.get("/v1/agent/sessions", headers=TENANT_HEADERS)
        assert sessions.status_code == 200
        assert sessions.json() == {"sessions": []}


def test_ui_error_envelope_uses_detail(tmp_path):
    with TestClient(_app(tmp_path)) as client:
        response = client.post(
            "/v1/agent/sessions",
            headers=TENANT_HEADERS,
            json={"asset_id": str(uuid4()), "message": "label workers"},
        )

        assert response.status_code == 404
        assert isinstance(response.json().get("detail"), str)


def test_ui_selector_and_navigation_contract():
    root = Path(__file__).resolve().parents[1]
    html = (root / "src" / "agentic_cv" / "web" / "index.html").read_text(encoding="utf-8")
    javascript = (root / "src" / "agentic_cv" / "web" / "app.js").read_text(encoding="utf-8")

    html_ids = set(re.findall(r'id="([^"]+)"', html))
    js_ids = set(re.findall(r'\$\("([^"]+)"\)', javascript))
    assert not (js_ids - html_ids), f"app.js references missing ids: {sorted(js_ids - html_ids)}"

    assert re.findall(r'data-view="([^"]+)"', html) == [
        "assets",
        "agent",
        "datasets",
        "jobs",
    ]
    assert "tenant-id" not in html
