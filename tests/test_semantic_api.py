from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from agentic_cv.semantic_api import SemanticApiDependencies, create_app
from agentic_cv.semantic_search import SearchResult
from agentic_cv.sub_assets import SubAssetStore
from agentic_cv.auth import Principal, StaticApiKeyAuthorizer


class FakeSearch:
    def search(self, query, *, asset_ids=None, limit=20):
        results = [
            SearchResult(record_id="frame-1", asset_id="asset-a", image_path=Path("secret/a.jpg"), score=0.95),
            SearchResult(record_id="frame-2", asset_id="asset-b", image_path=Path("secret/b.jpg"), score=0.8),
        ]
        if asset_ids is not None:
            results = [result for result in results if result.asset_id in asset_ids]
        return results[:limit]


def make_client(tmp_path, *, authorizer=None):
    store = SubAssetStore(str(tmp_path / "sub-assets.sqlite3"))
    client = TestClient(create_app(SemanticApiDependencies(
        search=FakeSearch(), sub_assets=store, authorizer=authorizer
    )))
    return client, store


def test_search_returns_safe_metadata_without_local_paths(tmp_path):
    client, store = make_client(tmp_path)

    response = client.post("/v1/search", json={"query": "bear", "asset_ids": ["asset-a"]})

    assert response.status_code == 200
    assert response.json()["results"] == [{"record_id": "frame-1", "asset_id": "asset-a", "score": 0.95}]
    assert "secret" not in response.text
    store.close()


def test_search_results_create_tenant_scoped_sub_asset(tmp_path):
    client, store = make_client(tmp_path)
    tenant_id = uuid4()

    response = client.post(
        "/v1/sub-assets/from-search",
        headers={"X-Tenant-ID": str(tenant_id)},
        json={"query": "bear", "limit": 2},
    )

    assert response.status_code == 201
    value = response.json()
    assert value["parent_asset_ids"] == ["asset-a", "asset-b"]
    assert value["result_ids"] == ["frame-1", "frame-2"]
    assert store.get_for_tenant(tenant_id, value["sub_asset_id"]).query == "bear"
    store.close()


def test_search_api_validates_empty_query_and_tenant_header(tmp_path):
    client, store = make_client(tmp_path)

    empty = client.post("/v1/search", json={"query": " "})
    bad_tenant = client.post(
        "/v1/sub-assets/from-search",
        headers={"X-Tenant-ID": "bad"},
        json={"query": "bear"},
    )

    assert empty.status_code == 422
    assert bad_tenant.status_code == 400
    store.close()


def test_semantic_api_optional_reader_auth(tmp_path):
    tenant_id = uuid4()
    authorizer = StaticApiKeyAuthorizer({
        "reader-key": Principal(tenant_id=tenant_id, roles=frozenset({"reader"})),
    })
    client, store = make_client(tmp_path, authorizer=authorizer)

    missing = client.post("/v1/search", json={"query": "bear"})
    valid = client.post(
        "/v1/search", headers={"X-Tenant-ID": str(tenant_id), "X-API-Key": "reader-key"},
        json={"query": "bear"},
    )
    wrong = client.post(
        "/v1/search", headers={"X-Tenant-ID": str(uuid4()), "X-API-Key": "reader-key"},
        json={"query": "bear"},
    )

    assert missing.status_code == 400
    assert valid.status_code == 200
    assert wrong.status_code == 403
    store.close()
