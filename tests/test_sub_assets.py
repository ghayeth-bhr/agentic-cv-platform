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


def test_sub_asset_links_materialized_asset_and_lists_lineage(tmp_path):
    store = SubAssetStore(str(tmp_path / "sub-assets.sqlite3"))
    tenant_id = uuid4()
    asset_id = uuid4()
    record = store.create_from_results(
        tenant_id=tenant_id,
        query="red object",
        results=[result("frame-1", "parent-a"), result("frame-2", "parent-b")],
    )

    linked = store.link_asset(record.sub_asset_id, asset_id=asset_id)

    assert linked.asset_id == asset_id
    assert store.find_by_asset(tenant_id, asset_id).sub_asset_id == record.sub_asset_id
    assert store.find_by_asset(uuid4(), asset_id) is None
    listed = store.list_for_tenant(tenant_id)
    assert [item.sub_asset_id for item in listed] == [record.sub_asset_id]
    assert listed[0].parent_asset_ids == ("parent-a", "parent-b")
    store.close()


def test_sub_asset_store_migrates_old_table_without_asset_id(tmp_path):
    import sqlite3

    path = tmp_path / "sub-assets.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE sub_assets (
            sub_asset_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            parent_asset_ids TEXT NOT NULL,
            query TEXT NOT NULL,
            result_ids TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.commit()
    connection.close()

    store = SubAssetStore(str(path))
    record = store.create_from_results(
        tenant_id=uuid4(), query="object", results=[result("frame", "asset")]
    )
    assert record.asset_id is None
    assert store.find_by_asset(record.tenant_id, uuid4()) is None
    store.close()
