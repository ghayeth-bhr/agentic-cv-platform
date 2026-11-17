from pathlib import Path
from uuid import uuid4

import pytest

from agentic_cv.semantic_search import SearchResult
from agentic_cv.sub_assets import SubAssetStore


def result(record_id, asset_id):
    return SearchResult(record_id=record_id, asset_id=asset_id, image_path=Path(f"{record_id}.jpg"), score=0.9)


def test_sub_asset_persists_query_parents_and_result_order(tmp_path):
    store = SubAssetStore(str(tmp_path / "sub-assets.sqlite3"))
    tenant_id = uuid4()
    record = store.create_from_results(
        tenant_id=tenant_id,
        query="red object",
        results=[result("frame-2", "asset-b"), result("frame-1", "asset-a"), result("frame-3", "asset-b")],
    )
    store.close()

    reopened = SubAssetStore(str(tmp_path / "sub-assets.sqlite3"))
    restored = reopened.get_for_tenant(tenant_id, record.sub_asset_id)

    assert restored.parent_asset_ids == ("asset-b", "asset-a")
    assert restored.result_ids == ("frame-2", "frame-1", "frame-3")
    assert restored.query == "red object"
    reopened.close()


def test_sub_asset_rejects_empty_or_duplicate_results(tmp_path):
    store = SubAssetStore(str(tmp_path / "sub-assets.sqlite3"))
    with pytest.raises(ValueError, match="at least one"):
        store.create_from_results(tenant_id=uuid4(), query="objects", results=[])
    with pytest.raises(ValueError, match="unique"):
        store.create_from_results(
            tenant_id=uuid4(), query="objects", results=[result("same", "asset"), result("same", "asset")]
        )
    store.close()


def test_sub_asset_tenant_access_is_enforced(tmp_path):
    store = SubAssetStore(str(tmp_path / "sub-assets.sqlite3"))
    record = store.create_from_results(
        tenant_id=uuid4(), query="object", results=[result("frame", "asset")]
    )

    with pytest.raises(PermissionError):
        store.get_for_tenant(uuid4(), record.sub_asset_id)
    store.close()
