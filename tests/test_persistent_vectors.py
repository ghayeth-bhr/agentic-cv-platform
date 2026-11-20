from pathlib import Path

import pytest

from agentic_cv.persistent_vectors import SqliteVectorIndex
from agentic_cv.semantic_search import ImageRecord


def test_vectors_survive_reopen_and_support_asset_filter(tmp_path):
    path = str(tmp_path / "vectors.sqlite3")
    first = SqliteVectorIndex(path)
    first.upsert(ImageRecord(record_id="red", asset_id="asset-a", image_path=Path("red.jpg")), [1, 0])
    first.upsert(ImageRecord(record_id="blue", asset_id="asset-b", image_path=Path("blue.jpg")), [0, 1])
    first.close()

    reopened = SqliteVectorIndex(path)
    results = reopened.search([1, 0], asset_ids={"asset-a"})

    assert [result.record_id for result in results] == ["red"]
    assert results[0].score == pytest.approx(1.0)
    assert reopened.get_embedding("red") == pytest.approx([1.0, 0.0])
    assert reopened.get_embedding("missing") is None
    reopened.close()


def test_upsert_remove_and_dimension_validation(tmp_path):
    index = SqliteVectorIndex(str(tmp_path / "vectors.sqlite3"))
    record = ImageRecord(record_id="frame", asset_id="asset", image_path=Path("frame.jpg"))
    index.upsert(record, [1, 0])
    index.upsert(record, [0, 1])
    assert index.search([0, 1])[0].score == pytest.approx(1.0)

    with pytest.raises(ValueError, match="dimension"):
        index.upsert(ImageRecord(record_id="other", asset_id="asset", image_path=Path("other.jpg")), [1, 2, 3])
    index.remove("frame")
    assert index.search([0, 1]) == []
    index.close()


def test_zero_vectors_and_invalid_limits_fail(tmp_path):
    index = SqliteVectorIndex(str(tmp_path / "vectors.sqlite3"))
    with pytest.raises(ValueError, match="zero"):
        index.upsert(ImageRecord(record_id="x", asset_id="a", image_path=Path("x.jpg")), [0, 0])
    with pytest.raises(ValueError, match="limit"):
        index.search([1, 0], limit=0)
    index.close()


def test_search_cache_sees_writes_from_this_and_other_connections(tmp_path):
    from pathlib import Path as _Path

    from agentic_cv.persistent_vectors import SqliteVectorIndex
    from agentic_cv.semantic_search import ImageRecord

    path = str(tmp_path / "vectors.sqlite3")
    reader = SqliteVectorIndex(path)
    writer = SqliteVectorIndex(path)
    reader.upsert(ImageRecord(record_id="a", asset_id="x", image_path=_Path("a.jpg")), [1.0, 0.0])
    assert [item.record_id for item in reader.search([1.0, 0.0])] == ["a"]  # builds the cache

    reader.upsert(ImageRecord(record_id="b", asset_id="y", image_path=_Path("b.jpg")), [0.0, 1.0])
    assert [item.record_id for item in reader.search([0.0, 1.0], limit=1)] == ["b"]

    # Another connection (e.g. a separate worker process) writes; the cache notices.
    writer.upsert(ImageRecord(record_id="c", asset_id="y", image_path=_Path("c.jpg")), [0.6, 0.8])
    assert [item.record_id for item in reader.search([0.6, 0.8], limit=1)] == ["c"]
    writer.remove("c")
    assert "c" not in [item.record_id for item in reader.search([0.6, 0.8])]

    # Asset filter and ties ordered by record id, as before.
    reader.upsert(ImageRecord(record_id="d", asset_id="y", image_path=_Path("d.jpg")), [0.0, 1.0])
    assert [item.record_id for item in reader.search([0.0, 1.0], asset_ids={"y"}, limit=2)] == ["b", "d"]
    assert reader.search([1.0, 0.0], asset_ids={"nope"}) == []
    reader.close()
    writer.close()
