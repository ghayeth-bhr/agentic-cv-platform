from uuid import uuid4

import pytest

from agentic_cv.dataset import DatasetCategory, DatasetImage, DatasetVersion
from agentic_cv.dataset_store import DatasetVersionStore


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