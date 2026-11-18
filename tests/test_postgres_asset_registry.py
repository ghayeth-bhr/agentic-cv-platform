"""Contract tests shared by the SQLite and PostgreSQL asset registries."""

import os
from uuid import UUID, uuid4

import pytest

from agentic_cv.asset_registry import AssetRegistry


def _digest(seed: str) -> str:
    return (seed * 64)[:64]


def exercise_asset_lifecycle(registry):
    tenant_id = uuid4()
    other_tenant = uuid4()
    asset = registry.create_asset(tenant_id=tenant_id, name="street-clip")

    assert asset.tenant_id == tenant_id
    assert asset.name == "street-clip"
    assert isinstance(asset.asset_id, UUID)
    assert asset.created_at

    with pytest.raises(ValueError, match="asset name"):
        registry.create_asset(tenant_id=tenant_id, name="  ")
    with pytest.raises(KeyError, match="unknown asset"):
        registry.start_upload(
            tenant_id=tenant_id,
            asset_id=uuid4(),
            filename="frame.jpg",
            media_type="image/jpeg",
            expected_sha256=_digest("a"),
        )
    with pytest.raises(PermissionError, match="tenant"):
        registry.start_upload(
            tenant_id=other_tenant,
            asset_id=asset.asset_id,
            filename="frame.jpg",
            media_type="image/jpeg",
            expected_sha256=_digest("a"),
        )

    upload = registry.start_upload(
        tenant_id=tenant_id,
        asset_id=asset.asset_id,
        filename="frame.jpg",
        media_type="image/jpeg",
        expected_sha256=_digest("a"),
    )
    assert upload.status == "active"
    assert registry.get_upload(upload.upload_id) == upload

    source = registry.complete_upload(upload.upload_id, object_uri="file:///tmp/frame.jpg")
    assert source.content_sha256 == upload.expected_sha256
    assert source.asset_id == asset.asset_id
    assert registry.get_upload(upload.upload_id).status == "completed"

    # Completing again returns the durable source instead of duplicating it.
    assert registry.complete_upload(upload.upload_id, object_uri="file:///ignored") == source

    duplicate = registry.start_upload(
        tenant_id=tenant_id,
        asset_id=asset.asset_id,
        filename="frame-again.jpg",
        media_type="image/jpeg",
        expected_sha256=_digest("a"),
    )
    assert registry.complete_upload(duplicate.upload_id, object_uri="file:///dup") == source
    assert registry.get_upload(duplicate.upload_id).status == "completed"


def test_sqlite_asset_registry_contract(tmp_path):
    registry = AssetRegistry(str(tmp_path / "assets.sqlite3"))
    try:
        exercise_asset_lifecycle(registry)
    finally:
        registry.close()


@pytest.mark.postgres
def test_postgres_asset_registry_contract():
    dsn = os.environ.get("AGENTIC_CV_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("set AGENTIC_CV_TEST_POSTGRES_DSN to run against a live PostgreSQL")
    from agentic_cv.postgres import PostgresConfig, connect
    from agentic_cv.postgres_asset_registry import open_postgres_asset_registry

    schema = f"agentic_cv_test_{uuid4().hex[:8]}"
    config = PostgresConfig(dsn=dsn, schema=schema)
    registry = None
    try:
        registry = open_postgres_asset_registry(config)
        exercise_asset_lifecycle(registry)
    finally:
        if registry is not None:
            registry.close()
        cleanup = connect(PostgresConfig(dsn=dsn))
        cleanup.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        cleanup.commit()
        cleanup.close()
