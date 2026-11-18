from uuid import uuid4

import pytest

from agentic_cv.annotation_store import AnnotationRevisionStore, RevisionConflictError
from agentic_cv.dataset import Annotation


def annotation(x: float) -> Annotation:
    return Annotation(
        annotation_id="annotation-1", image_id="image-1", category_id=1,
        bbox=(x, 1, 4, 4), score=0.8,
    )


def test_revisions_persist_and_keep_history(tmp_path):
    path = str(tmp_path / "annotations.sqlite3")
    tenant_id = uuid4()
    store = AnnotationRevisionStore(path)
    first = store.save(tenant_id=tenant_id, annotation=annotation(1), author_id="annotator")
    second = store.save(
        tenant_id=tenant_id, annotation=annotation(2), author_id="reviewer", expected_revision=1
    )
    store.close()

    reopened = AnnotationRevisionStore(path)
    latest = reopened.latest(tenant_id=tenant_id, image_id="image-1", annotation_id="annotation-1")
    history = reopened.history(tenant_id=tenant_id, image_id="image-1", annotation_id="annotation-1")

    assert first.revision == 1
    assert second.revision == 2
    assert latest.annotation.bbox[0] == 2
    assert [item.author_id for item in history] == ["annotator", "reviewer"]
    reopened.close()


def test_stale_revision_is_rejected(tmp_path):
    store = AnnotationRevisionStore(str(tmp_path / "annotations.sqlite3"))
    tenant_id = uuid4()
    store.save(tenant_id=tenant_id, annotation=annotation(1), author_id="one")

    with pytest.raises(RevisionConflictError, match="current revision is 1"):
        store.save(
            tenant_id=tenant_id, annotation=annotation(3), author_id="stale", expected_revision=0
        )
    store.close()


def test_revision_inputs_are_validated(tmp_path):
    store = AnnotationRevisionStore(str(tmp_path / "annotations.sqlite3"))
    with pytest.raises(ValueError, match="author_id"):
        store.save(tenant_id=uuid4(), annotation=annotation(1), author_id=" ")
    with pytest.raises(ValueError, match="expected_revision"):
        store.save(tenant_id=uuid4(), annotation=annotation(1), author_id="one", expected_revision=-1)
    store.close()


def test_list_latest_returns_one_current_revision_per_annotation(tmp_path):
    store = AnnotationRevisionStore(str(tmp_path / "annotations.sqlite3"))
    tenant_id = uuid4()
    store.save(tenant_id=tenant_id, annotation=annotation(1), author_id="first")
    store.save(
        tenant_id=tenant_id,
        annotation=annotation(2),
        author_id="second",
        expected_revision=1,
    )
    other = Annotation(
        annotation_id="annotation-2",
        image_id="image-2",
        category_id=1,
        bbox=(1, 1, 2, 2),
    )
    store.save(tenant_id=tenant_id, annotation=other, author_id="first")

    latest = store.list_latest(tenant_id=tenant_id)
    filtered = store.list_latest(tenant_id=tenant_id, image_ids=["image-1"])

    assert len(latest) == 2
    assert [item.annotation.bbox[0] for item in latest] == [2, 1]
    assert len(filtered) == 1
    assert filtered[0].annotation.bbox[0] == 2
    store.close()


def test_save_without_expected_revision_is_append_only_for_recovery(tmp_path):
    store = AnnotationRevisionStore(str(tmp_path / "annotations.sqlite3"))
    tenant_id = uuid4()
    first = store.save(
        tenant_id=tenant_id,
        annotation=annotation(1),
        author_id="first",
        expected_revision=None,
    )
    second = store.save(
        tenant_id=tenant_id,
        annotation=annotation(2),
        author_id="retry",
        expected_revision=None,
    )

    assert (first.revision, second.revision) == (1, 2)
    store.close()
