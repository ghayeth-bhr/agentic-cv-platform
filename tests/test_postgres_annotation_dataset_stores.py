"""Contract tests shared by SQLite and PostgreSQL annotation/dataset stores."""

import json
import os
from uuid import uuid4

import pytest

from agentic_cv.annotation_store import AnnotationRevisionStore, RevisionConflictError
from agentic_cv.dataset import Annotation, DatasetCategory, DatasetImage, DatasetVersion
from agentic_cv.dataset_store import DatasetVersionStore


def _annotation(x: float) -> Annotation:
    return Annotation(
        annotation_id="annotation-1",
        image_id="image-1",
        category_id=1,
        bbox=(x, 1, 4, 4),
        score=0.8,
    )


def _dataset() -> DatasetVersion:
    return DatasetVersion(
        images=(DatasetImage(image_id="image-1", file_name="frame.jpg", width=10, height=10),),
        categories=(DatasetCategory(category_id=0, name="object"),),
        annotations=(),
    )


def exercise_annotation_contract(store):
    tenant_id = uuid4()
    first = store.save(tenant_id=tenant_id, annotation=_annotation(1), author_id="annotator")
    second = store.save(
        tenant_id=tenant_id, annotation=_annotation(2), author_id="reviewer", expected_revision=1
    )

    assert first.revision == 1
    assert second.revision == 2
    latest = store.latest(tenant_id=tenant_id, image_id="image-1", annotation_id="annotation-1")
    history = store.history(tenant_id=tenant_id, image_id="image-1", annotation_id="annotation-1")
    assert latest.annotation.bbox[0] == 2
    assert [item.author_id for item in history] == ["annotator", "reviewer"]

    with pytest.raises(RevisionConflictError, match="current revision is 2"):
        store.save(
            tenant_id=tenant_id, annotation=_annotation(3), author_id="stale", expected_revision=1
        )
    with pytest.raises(KeyError, match="unknown annotation"):
        store.latest(tenant_id=tenant_id, image_id="image-1", annotation_id="missing")
    with pytest.raises(ValueError, match="author_id"):
        store.save(tenant_id=tenant_id, annotation=_annotation(1), author_id=" ")
    with pytest.raises(ValueError, match="expected_revision"):
        store.save(tenant_id=tenant_id, annotation=_annotation(1), author_id="x", expected_revision=-1)


def exercise_dataset_version_contract(store):
    tenant_id = uuid4()
    first = store.create(
        tenant_id=tenant_id, name="train-v1", dataset=_dataset(), idempotency_key="train-v1"
    )
    second = store.create(
        tenant_id=tenant_id, name="test-v1", dataset=_dataset(), idempotency_key="test-v1"
    )

    restored = store.get(tenant_id=tenant_id, version_id=first.version_id)
    assert restored.dataset.images[0].file_name == "frame.jpg"
    assert [record.version_id for record in store.list(tenant_id=tenant_id)] == [
        first.version_id, second.version_id,
    ]
    assert store.create(
        tenant_id=tenant_id, name="train-v1", dataset=_dataset(), idempotency_key="train-v1"
    ) == first
    with pytest.raises(ValueError, match="different version"):
        store.create(
            tenant_id=tenant_id, name="other", dataset=_dataset(), idempotency_key="train-v1"
        )
    with pytest.raises(ValueError, match="name and idempotency_key"):
        store.create(
            tenant_id=tenant_id, name=" ", dataset=_dataset(), idempotency_key="k"
        )
    with pytest.raises(KeyError, match="unknown dataset version"):
        store.get(tenant_id=uuid4(), version_id=first.version_id)


def test_sqlite_annotation_contract(tmp_path):
    store = AnnotationRevisionStore(str(tmp_path / "annotations.sqlite3"))
    try:
        exercise_annotation_contract(store)
    finally:
        store.close()


def test_sqlite_dataset_version_contract(tmp_path):
    store = DatasetVersionStore(str(tmp_path / "datasets.sqlite3"))
    try:
        exercise_dataset_version_contract(store)
    finally:
        store.close()


@pytest.mark.postgres
def test_postgres_annotation_contract():
    dsn = os.environ.get("AGENTIC_CV_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("set AGENTIC_CV_TEST_POSTGRES_DSN to run against a live PostgreSQL")
    from agentic_cv.postgres import PostgresConfig, connect
    from agentic_cv.postgres_annotation_store import open_postgres_annotation_store

    schema = f"agentic_cv_test_{uuid4().hex[:8]}"
    store = None
    try:
        store = open_postgres_annotation_store(PostgresConfig(dsn=dsn, schema=schema))
        exercise_annotation_contract(store)
    finally:
        if store is not None:
            store.close()
        cleanup = connect(PostgresConfig(dsn=dsn))
        cleanup.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        cleanup.commit()
        cleanup.close()


@pytest.mark.postgres
def test_postgres_dataset_version_contract():
    dsn = os.environ.get("AGENTIC_CV_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("set AGENTIC_CV_TEST_POSTGRES_DSN to run against a live PostgreSQL")
    from agentic_cv.postgres import PostgresConfig, connect
    from agentic_cv.postgres_dataset_store import open_postgres_dataset_version_store

    schema = f"agentic_cv_test_{uuid4().hex[:8]}"
    store = None
    try:
        store = open_postgres_dataset_version_store(PostgresConfig(dsn=dsn, schema=schema))
        exercise_dataset_version_contract(store)
    finally:
        if store is not None:
            store.close()
        cleanup = connect(PostgresConfig(dsn=dsn))
        cleanup.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        cleanup.commit()
        cleanup.close()
