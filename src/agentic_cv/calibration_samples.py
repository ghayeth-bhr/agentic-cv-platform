"""Pick calibration samples that cover the asset's visual variation."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .batch_labeling import LabelImage


def select_calibration_images(
    images: list[LabelImage],
    *,
    count: int,
    embeddings: Mapping[str, Sequence[float]],
) -> list[LabelImage]:
    """Choose diverse images with stored SigLIP vectors, or evenly spaced fallback.

    The selector is deterministic. It starts from the image farthest from the
    embedding centroid, then repeatedly adds the image farthest from the already
    selected set. If any candidate is missing a usable vector, the caller gets
    the same evenly spaced sample as previous releases.
    """

    if not images:
        return []
    count = max(1, min(count, len(images)))
    if count >= len(images):
        return list(images)
    if count < 2:
        return _evenly_spaced(images, count)

    vectors = _vectors_for(images, embeddings)
    if vectors is None:
        return _evenly_spaced(images, count)

    centroid = vectors.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm == 0:
        return _evenly_spaced(images, count)
    centroid = centroid / centroid_norm

    selected: list[int] = [int(np.argmax(1.0 - vectors @ centroid))]
    remaining = set(range(len(images))) - set(selected)
    while len(selected) < count:
        similarities = vectors[list(selected)] @ vectors.T
        nearest = similarities.max(axis=0)
        distances = 1.0 - nearest
        candidates = [
            (float(distances[index]), index)
            for index in remaining
        ]
        # Farthest first; lowest original index breaks ties deterministically.
        _, next_index = max(candidates, key=lambda item: (item[0], -item[1]))
        selected.append(next_index)
        remaining.remove(next_index)
    return [images[index] for index in sorted(selected)]


def _vectors_for(
    images: list[LabelImage],
    embeddings: Mapping[str, Sequence[float]],
) -> np.ndarray | None:
    vectors: list[np.ndarray] = []
    dimension: int | None = None
    for image in images:
        raw = embeddings.get(image.image_id)
        if raw is None:
            return None
        try:
            vector = np.asarray(raw, dtype=np.float32)
        except (TypeError, ValueError):
            return None
        if vector.ndim != 1 or vector.size == 0:
            return None
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm == 0:
            return None
        if dimension is None:
            dimension = int(vector.size)
        elif vector.size != dimension:
            return None
        vectors.append(vector / norm)
    if not vectors:
        return None
    return np.stack(vectors)


def _evenly_spaced(images: list[LabelImage], count: int) -> list[LabelImage]:
    if count >= len(images):
        return list(images)
    if count == 1:
        return [images[0]]
    indices = [
        round(index * (len(images) - 1) / (count - 1)) for index in range(count)
    ]
    return [images[index] for index in indices]


def select_samples_for_target(
    images: list[LabelImage],
    *,
    count: int,
    embeddings: Mapping[str, Sequence[float]],
    text_embedding: Sequence[float] | None,
    positive_share: float = 0.6,
) -> list[LabelImage]:
    """Calibration samples that mostly contain the target, plus diverse negatives.

    Pure diversity sampling picks the visual outliers, which in a curated
    asset are often exactly the frames *without* the target (measured: 3 of 5
    samples had no elephant). Here ``ceil(count * positive_share)`` samples are
    the images most similar to the class text (SigLIP text-image similarity),
    and the rest are chosen farthest-first so a false-positive-prone threshold
    still gets tested. Falls back to :func:`select_calibration_images` when the
    text or any image vector is missing.
    """

    if not images:
        return []
    count = max(1, min(count, len(images)))
    if count >= len(images):
        return list(images)
    vectors = _vectors_for(images, embeddings)
    if text_embedding is None or vectors is None:
        return select_calibration_images(images, count=count, embeddings=embeddings)
    text = np.asarray(text_embedding, dtype=np.float32)
    norm = float(np.linalg.norm(text))
    if text.ndim != 1 or text.size != vectors.shape[1] or not np.isfinite(norm) or norm == 0:
        return select_calibration_images(images, count=count, embeddings=embeddings)
    similarity = vectors @ (text / norm)
    positives = max(1, min(count, int(np.ceil(count * positive_share))))
    # Highest similarity first; lowest index breaks ties deterministically.
    ranked = sorted(range(len(images)), key=lambda index: (-float(similarity[index]), index))
    selected = ranked[:positives]
    remaining = set(range(len(images))) - set(selected)
    while len(selected) < count and remaining:
        nearest = (vectors[selected] @ vectors.T).max(axis=0)
        _, next_index = max(
            ((1.0 - float(nearest[index]), index) for index in remaining),
            key=lambda item: (item[0], -item[1]),
        )
        selected.append(next_index)
        remaining.remove(next_index)
    return [images[index] for index in sorted(selected)]
