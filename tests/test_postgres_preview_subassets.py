"""Contract tests shared by SQLite and PostgreSQL preview/sub-asset stores."""

import os
from pathlib import Path
from uuid import uuid4

import pytest

from agentic_cv.preview import PreviewStore
from agentic_cv.semantic_search import SearchResult
from agentic_cv.sub_assets import SubAssetStore


def _result(record_id: str, asset_id: str) -> SearchResult:
    return SearchResult(
        record_id=record_id, asset_id=asset_id,
        image_path=Path(f"{record_id}.jpg"), score=0.9,
    )


def exercise_preview_contract(store):
    tenant_id = uuid4()
    job_id = uuid4()
    first = store.publish(
        tenant_id=tenant_id, job_id=job_id, image_id="frame-1", processed=1, total=2
    )
    second = store.publish(
        tenant_id=tenant_id, job_id=job_id, image_id="frame-2", processed=2, total=2
    )

    assert first.event_id < second.event_id
    assert [event.image_id for event in store.list_for_job(tenant_id=tenant_id, job_id=job_id)] == [
        "frame-1", "frame-2",
    ]
    assert [event.image_id for event in store.list_for_job(
        tenant_id=tenant_id, job_id=job_id, after_event_id=first.event_id
    )] == ["frame-2"]
    assert store.list_for_job(tenant_id=uuid4(), job_id=job_id) == []
    assert store.get(second.event_id).processed == 2
    with pytest.raises(ValueError, match="progress"):
        store.publish(
            tenant_id=tenant_id, job_id=job_id, image_id="frame", processed=2, total=1
        )
    with pytest.raises(ValueError, match="after_event_id"):
        store.list_for_job(tenant_id=tenant_id, job_id=job_id, after_event_id=-1)
    with pytest.raises(KeyError, match="unknown preview event"):
        store.get(999999)


def exercise_sub_asset_contract(store):
    tenant_id = uuid4()
    record = store.create_from_results(
        tenant_id=tenant_id,
        query="red object",
        results=[
            _result("frame-2", "asset-b"),
            _result("frame-1", "asset-a"),
            _result("frame-3", "asset-b"),
        ],
    )
    restored = store.get_for_tenant(tenant_id, record.sub_asset_id)
    assert restored.parent_asset_ids == ("asset-b", "asset-a")
    assert restored.result_ids == ("frame-2", "frame-1", "frame-3")
    assert restored.query == "red object"
    assert store.get(record.sub_asset_id).sub_asset_id == record.sub_asset_id

    with pytest.raises(PermissionError):
        store.get_for_tenant(uuid4(), record.sub_asset_id)
    with pytest.raises(KeyError, match="unknown sub-asset"):
        store.get(uuid4())
    with pytest.raises(ValueError, match="at least one"):
        store.create_from_results(tenant_id=uuid4(), query="objects", results=[])
    with pytest.raises(ValueError, match="unique"):
        store.create_from_results(
            tenant_id=uuid4(), query="objects",
            results=[_result("same", "asset"), _result("same", "asset")],
        )
    with pytest.raises(ValueError, match="query"):
        store.create_from_results(
            tenant_id=uuid4(), query="  ", results=[_result("x", "a")]
        )


def test_sqlite_preview_contract(tmp_path):
    store = PreviewStore(str(tmp_path / "preview.sqlite3"))
    try:
        exercise_preview_contract(store)
    finally:
        store.close()


def test_sqlite_sub_asset_contract(tmp_path):
    store = SubAssetStore(str(tmp_path / "sub-assets.sqlite3"))
    try:
        exercise_sub_asset_contract(store)
    finally:
        store.close()


@pytest.mark.postgres
def test_postgres_preview_contract():
    dsn = os.environ.get("AGENTIC_CV_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("set AGENTIC_CV_TEST_POSTGRES_DSN to run against a live PostgreSQL")
    from agentic_cv.postgres import PostgresConfig, connect
    from agentic_cv.postgres_preview import open_postgres_preview_store

    schema = f"agentic_cv_test_{uuid4().hex[:8]}"
    store = None
    try:
        store = open_postgres_preview_store(PostgresConfig(dsn=dsn, schema=schema))
        exercise_preview_contract(store)
    finally:
        if store is not None:
            store.close()
        cleanup = connect(PostgresConfig(dsn=dsn))
        cleanup.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        cleanup.commit()
        cleanup.close()


@pytest.mark.postgres
def test_postgres_sub_asset_contract():
    dsn = os.environ.get("AGENTIC_CV_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("set AGENTIC_CV_TEST_POSTGRES_DSN to run against a live PostgreSQL")
    from agentic_cv.postgres import PostgresConfig, connect
    from agentic_cv.postgres_sub_assets import open_postgres_sub_asset_store

    schema = f"agentic_cv_test_{uuid4().hex[:8]}"
    store = None
    try:
        store = open_postgres_sub_asset_store(PostgresConfig(dsn=dsn, schema=schema))
        exercise_sub_asset_contract(store)
    finally:
        if store is not None:
            store.close()
        cleanup = connect(PostgresConfig(dsn=dsn))
        cleanup.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        cleanup.commit()
        cleanup.close()
