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
