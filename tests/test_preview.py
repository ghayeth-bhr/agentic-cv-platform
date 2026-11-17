from uuid import uuid4

import pytest

from agentic_cv.preview import PreviewStore


def test_preview_events_are_ordered_and_incrementally_readable(tmp_path):
    store = PreviewStore(str(tmp_path / "preview.sqlite3"))
    tenant_id = uuid4()
    job_id = uuid4()
    first = store.publish(tenant_id=tenant_id, job_id=job_id, image_id="frame-1", processed=1, total=2)
    second = store.publish(tenant_id=tenant_id, job_id=job_id, image_id="frame-2", processed=2, total=2)

    assert [event.image_id for event in store.list_for_job(tenant_id=tenant_id, job_id=job_id)] == [
        "frame-1", "frame-2"
    ]
    assert [event.image_id for event in store.list_for_job(
        tenant_id=tenant_id, job_id=job_id, after_event_id=first.event_id
    )] == ["frame-2"]
    assert second.processed == 2
    store.close()


def test_preview_events_are_tenant_scoped_and_validate_progress(tmp_path):
    store = PreviewStore(str(tmp_path / "preview.sqlite3"))
    job_id = uuid4()
    tenant_id = uuid4()
    store.publish(tenant_id=tenant_id, job_id=job_id, image_id="frame", processed=1, total=1)

    assert store.list_for_job(tenant_id=uuid4(), job_id=job_id) == []
    with pytest.raises(ValueError, match="progress"):
        store.publish(tenant_id=tenant_id, job_id=job_id, image_id="frame", processed=2, total=1)
    store.close()
