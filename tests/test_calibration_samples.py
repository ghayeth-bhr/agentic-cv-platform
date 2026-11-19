from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

from agentic_cv.agent import AgentService
from agentic_cv.batch_labeling import LabelImage
from agentic_cv.calibration_samples import select_calibration_images


def _images(count: int) -> list[LabelImage]:
    return [
        LabelImage(image_id=f"frame-{index}", path=Path(f"frame-{index}.png"))
        for index in range(count)
    ]


def test_diversity_prefers_visual_outliers_over_even_spacing():
    images = _images(6)
    embeddings = {
        "frame-0": [1.0, 0.0],
        "frame-1": [1.0, 0.02],
        "frame-2": [0.98, 0.0],
        "frame-3": [1.0, -0.02],
        "frame-4": [0.0, 1.0],
        "frame-5": [-1.0, 0.0],
    }

    selected = select_calibration_images(images, count=3, embeddings=embeddings)

    assert [image.image_id for image in selected] == [
        "frame-0",
        "frame-4",
        "frame-5",
    ]


def test_selection_falls_back_to_even_spacing_when_vectors_are_missing():
    images = _images(7)

    selected = select_calibration_images(
        images,
        count=4,
        embeddings={"frame-0": [1.0, 0.0]},
    )

    assert [image.image_id for image in selected] == [
        "frame-0",
        "frame-2",
        "frame-4",
        "frame-6",
    ]


def test_zero_and_mismatched_vectors_fall_back_to_even_spacing():
    images = _images(5)
    zero = select_calibration_images(
        images,
        count=3,
        embeddings={image.image_id: [0.0, 0.0] for image in images},
    )
    mismatched = select_calibration_images(
        images,
        count=3,
        embeddings={
            "frame-0": [1.0, 0.0],
            "frame-1": [1.0, 0.0],
            "frame-2": [1.0, 0.0, 0.0],
            "frame-3": [1.0, 0.0],
            "frame-4": [1.0, 0.0],
        },
    )

    assert [image.image_id for image in zero] == ["frame-0", "frame-2", "frame-4"]
    assert [image.image_id for image in mismatched] == [
        "frame-0",
        "frame-2",
        "frame-4",
    ]


def test_selection_returns_every_image_when_sample_covers_the_asset():
    images = _images(3)

    selected = select_calibration_images(images, count=10, embeddings={})

    assert selected == images


def test_agent_service_reads_vectors_from_the_index_before_sampling():
    images = _images(6)
    embeddings = {
        "frame-0": [1.0, 0.0],
        "frame-1": [1.0, 0.02],
        "frame-2": [0.98, 0.0],
        "frame-3": [1.0, -0.02],
        "frame-4": [0.0, 1.0],
        "frame-5": [-1.0, 0.0],
    }

    class Images:
        def resolve(self, *, tenant_id, asset_id):
            return images

    class Vectors:
        def get_embedding(self, record_id):
            return embeddings.get(record_id)

    service = object.__new__(AgentService)
    service.dependencies = SimpleNamespace(images=Images(), vectors=Vectors())

    selected = service._sample_images(
        tenant_id=UUID("00000000-0000-0000-0000-000000000001"),
        asset_id=UUID("00000000-0000-0000-0000-000000000002"),
        sample_size=3,
    )

    assert [image.image_id for image in selected] == [
        "frame-0",
        "frame-4",
        "frame-5",
    ]
