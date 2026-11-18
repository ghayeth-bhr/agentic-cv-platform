"""Contract tests for local and PostgreSQL/pgvector vector indexes."""

import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from agentic_cv.semantic_search import ImageRecord, LocalVectorIndex, SemanticSearchService


class FakeEmbeddingProvider:
    def embed_text(self, query: str) -> list[float]:
        if "red" in query:
            return [1.0, 0.0]
        return [0.0, 1.0]

    def embed_image(self, image) -> list[float]:
        arr = np.asarray(image)
        if arr[:, :, 0].mean() > arr[:, :, 2].mean():
            return [0.95, 0.05]
        return [0.05, 0.95]


def _record(record_id: str, asset_id: str = "asset-1") -> ImageRecord:
    return ImageRecord(record_id=record_id, asset_id=asset_id, image_path=Path(f"{record_id}.jpg"))


def exercise_vector_index(index):
    index.upsert(_record("red", "asset-a"), [1.0, 0.0])
    index.upsert(_record("blue", "asset-b"), [0.0, 1.0])

    results = index.search([0.9, 0.1])
    assert [result.record_id for result in results] == ["red", "blue"]
    assert results[0].asset_id == "asset-a"
    assert results[0].image_path == Path("red.jpg")
    assert results[0].score > results[1].score

    exact = index.search([1.0, 0.0], limit=1)
    assert exact[0].record_id == "red"
    assert exact[0].score == pytest.approx(1.0, abs=1e-5)

    filtered = index.search([0.9, 0.1], asset_ids={"asset-b"})
    assert [result.record_id for result in filtered] == ["blue"]

    limited = index.search([0.9, 0.1], limit=1)
    assert [result.record_id for result in limited] == ["red"]

    index.remove("red")
    assert [result.record_id for result in index.search([0.9, 0.1])] == ["blue"]

    with pytest.raises(ValueError, match="embedding dimension"):
        index.upsert(_record("bad"), [1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="embedding dimension"):
        index.search([1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="zero vector"):
        index.upsert(_record("zero"), [0.0, 0.0])
    with pytest.raises(ValueError, match="limit"):
        index.search([1.0, 0.0], limit=0)


def test_local_vector_index_contract():
    index = LocalVectorIndex()
    exercise_vector_index(index)


def test_local_semantic_search_service(tmp_path):
    red = tmp_path / "red.jpg"
    blue = tmp_path / "blue.jpg"
    from PIL import Image

    Image.new("RGB", (4, 4), (255, 0, 0)).save(red)
    Image.new("RGB", (4, 4), (0, 0, 255)).save(blue)

    service = SemanticSearchService(FakeEmbeddingProvider(), LocalVectorIndex())
    service.index_images([
        ImageRecord(record_id="red", asset_id="asset-1", image_path=red),
        ImageRecord(record_id="blue", asset_id="asset-2", image_path=blue),
    ])

    assert [result.record_id for result in service.search("red object")] == ["red", "blue"]
    assert [result.record_id for result in service.search("red", asset_ids={"asset-1"})] == ["red"]
    with pytest.raises(ValueError, match="query"):
        service.search(" ")


@pytest.mark.postgres
def test_postgres_pgvector_index_contract():
    dsn = os.environ.get("AGENTIC_CV_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("set AGENTIC_CV_TEST_POSTGRES_DSN to run against a live PostgreSQL")
    from agentic_cv.postgres import PostgresConfig, connect
    from agentic_cv.postgres_pgvector import open_postgres_vector_index

    schema = f"agentic_cv_test_{uuid4().hex[:8]}"
    index = None
    try:
        index = open_postgres_vector_index(PostgresConfig(dsn=dsn, schema=schema))
        exercise_vector_index(index)
    finally:
        if index is not None:
            index.close()
        cleanup = connect(PostgresConfig(dsn=dsn))
        cleanup.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        cleanup.commit()
        cleanup.close()


@pytest.mark.postgres
def test_postgres_pgvector_index_persists_across_reopen():
    dsn = os.environ.get("AGENTIC_CV_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("set AGENTIC_CV_TEST_POSTGRES_DSN to run against a live PostgreSQL")
    from agentic_cv.postgres import PostgresConfig, connect
    from agentic_cv.postgres_pgvector import open_postgres_vector_index

    schema = f"agentic_cv_test_{uuid4().hex[:8]}"
    config = PostgresConfig(dsn=dsn, schema=schema)
    try:
        index = open_postgres_vector_index(config)
        index.upsert(_record("red", "asset-a"), [1.0, 0.0])
        index.upsert(_record("blue", "asset-b"), [0.0, 1.0])
        index.close()

        reopened = open_postgres_vector_index(config)
        try:
            results = reopened.search([0.9, 0.1])
            assert [result.record_id for result in results] == ["red", "blue"]
            exact = reopened.search([1.0, 0.0], limit=1)
            assert exact[0].record_id == "red"
            assert exact[0].score == pytest.approx(1.0, abs=1e-5)
            with pytest.raises(ValueError, match="embedding dimension"):
                reopened.upsert(_record("bad"), [1.0, 0.0, 0.0])
        finally:
            reopened.close()
    finally:
        cleanup = connect(PostgresConfig(dsn=dsn))
        cleanup.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        cleanup.commit()
        cleanup.close()
