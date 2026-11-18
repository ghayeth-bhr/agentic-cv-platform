import sqlite3
from uuid import UUID, uuid4

import pytest

from agentic_cv.dataset import DatasetCategory, DatasetImage, DatasetVersion
from agentic_cv.dataset_store import DatasetVersionStore

FIXED_CREATED_AT = "2026-01-01T00:00:00+00:00"
DATASET_JSON = (
    '{"annotations": [], "categories": [{"category_id": 0, "name": "object"}], '
    '"images": [{"file_name": "frame.jpg", "height": 10, "image_id": "image-1", "width": 10}]}'
)


def dataset() -> DatasetVersion:
    return DatasetVersion(
        images=(DatasetImage(image_id="image-1", file_name="frame.jpg", width=10, height=10),),
        categories=(DatasetCategory(category_id=0, name="object"),),
        annotations=(),
    )


def test_dataset_versions_persist_and_list_by_tenant(tmp_path):
    path = str(tmp_path / "datasets.sqlite3")
    tenant_id = uuid4()
    store = DatasetVersionStore(path)
    first = store.create(tenant_id=tenant_id, name="train-v1", dataset=dataset(), idempotency_key="train-v1")
    second = store.create(tenant_id=tenant_id, name="test-v1", dataset=dataset(), idempotency_key="test-v1")
    store.close()

    reopened = DatasetVersionStore(path)
    restored = reopened.get(tenant_id=tenant_id, version_id=first.version_id)
    assert restored.dataset.images[0].file_name == "frame.jpg"
    assert [record.version_id for record in reopened.list(tenant_id=tenant_id)] == [first.version_id, second.version_id]
    with pytest.raises(KeyError, match="unknown"):
        reopened.get(tenant_id=uuid4(), version_id=first.version_id)
    reopened.close()


def test_dataset_version_creation_is_idempotent_and_conflicts_are_rejected(tmp_path):
    store = DatasetVersionStore(str(tmp_path / "datasets.sqlite3"))
    tenant_id = uuid4()
    first = store.create(tenant_id=tenant_id, name="v1", dataset=dataset(), idempotency_key="key")
    assert store.create(tenant_id=tenant_id, name="v1", dataset=dataset(), idempotency_key="key") == first
    with pytest.raises(ValueError, match="different version"):
        store.create(tenant_id=tenant_id, name="v2", dataset=dataset(), idempotency_key="key")
    store.close()

def _force_creation_order(path, first_id, second_id):
    connection = sqlite3.connect(path)
    connection.execute(
        "UPDATE dataset_versions SET version_id = ?, created_at = ? WHERE version_id = ?",
        (str(UUID(int=2)), FIXED_CREATED_AT, str(first_id)),
    )
    connection.execute(
        "UPDATE dataset_versions SET version_id = ?, created_at = ? WHERE version_id = ?",
        (str(UUID(int=1)), FIXED_CREATED_AT, str(second_id)),
    )
    connection.commit()
    connection.close()


def _legacy_dataset_database(path, tenant_id, rows):
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE dataset_versions (
            version_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            name TEXT NOT NULL,
            dataset TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE (tenant_id, idempotency_key)
        )
        """
    )
    for version_id, name, key in rows:
        connection.execute(
            """
            INSERT INTO dataset_versions (
                version_id, tenant_id, name, dataset, idempotency_key, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (str(version_id), str(tenant_id), name, DATASET_JSON, key, FIXED_CREATED_AT),
        )
    connection.commit()
    connection.close()


def test_list_preserves_creation_order_when_timestamps_tie(tmp_path):
    path = str(tmp_path / "datasets.sqlite3")
    tenant_id = uuid4()
    store = DatasetVersionStore(path)
    first = store.create(
        tenant_id=tenant_id, name="v1", dataset=dataset(), idempotency_key="v1"
    )
    second = store.create(
        tenant_id=tenant_id, name="v2", dataset=dataset(), idempotency_key="v2"
    )
    _force_creation_order(path, first.version_id, second.version_id)

    assert [record.version_id for record in store.list(tenant_id=tenant_id)] == [
        UUID(int=2),
        UUID(int=1),
    ]
    store.close()


def test_legacy_dataset_table_is_migrated_and_keeps_insertion_order(tmp_path):
    path = str(tmp_path / "datasets.sqlite3")
    tenant_id = uuid4()
    _legacy_dataset_database(
        path,
        tenant_id,
        [(UUID(int=2), "v1", "v1"), (UUID(int=1), "v2", "v2")],
    )

    store = DatasetVersionStore(path)

    assert [record.version_id for record in store.list(tenant_id=tenant_id)] == [
        UUID(int=2),
        UUID(int=1),
    ]
    assert store.get(tenant_id=tenant_id, version_id=UUID(int=1)).name == "v2"
    store.close()
