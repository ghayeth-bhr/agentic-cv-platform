from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from agentic_cv.semantic_search import ImageRecord, LocalVectorIndex, SemanticSearchService


class FakeEmbeddingProvider:
    def embed_image(self, image):
        return [image.getpixel((0, 0))[0], image.getpixel((0, 0))[1]]

    def embed_text(self, query):
        return [1, 0] if "red" in query else [0, 1]


def write_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (1, 1), color).save(path)


def test_index_and_asset_scoped_search_returns_cosine_order(tmp_path):
    red = tmp_path / "red.jpg"
    green = tmp_path / "green.jpg"
    write_image(red, (255, 0, 0))
    write_image(green, (0, 255, 0))
    service = SemanticSearchService(FakeEmbeddingProvider(), LocalVectorIndex())
    service.index_images([
        ImageRecord(record_id="red", asset_id="asset-a", image_path=red),
        ImageRecord(record_id="green", asset_id="asset-b", image_path=green),
    ])

    results = service.search("red object", asset_ids={"asset-a", "asset-b"})

    assert [result.record_id for result in results] == ["red", "green"]
    assert results[0].score == pytest.approx(1.0)


def test_upsert_replaces_record_and_dimensions_are_guarded(tmp_path):
    index = LocalVectorIndex()
    record = ImageRecord(record_id="frame", asset_id="asset", image_path=tmp_path / "frame.jpg")
    index.upsert(record, [1, 0])
    index.upsert(record, [0, 1])

    assert index.search([0, 1])[0].score == pytest.approx(1.0)
    with pytest.raises(ValueError, match="dimension"):
        index.upsert(ImageRecord(record_id="other", asset_id="asset", image_path=record.image_path), [1, 2, 3])


def test_invalid_query_and_zero_vectors_fail(tmp_path):
    index = LocalVectorIndex()
    with pytest.raises(ValueError, match="zero"):
        index.upsert(ImageRecord(record_id="x", asset_id="a", image_path=tmp_path / "x"), [0, 0])
    with pytest.raises(ValueError, match="query"):
        SemanticSearchService(FakeEmbeddingProvider(), index).search(" ")
