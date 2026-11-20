"""Annotation sets: scope annotation revisions to the run that produced them.

Revisions are keyed by ``(image_id, annotation_id)``. Without a scope, two
label runs on the same asset wrote ``"{image_id}:0"`` and both read "the latest
revision per id", so a second run inherited stale labels from the first. A set
id prefixes every annotation id (``"{set_id}/..."``); a dataset version records
its set and only ever reads annotations from it. Manual corrections write into
the version's set, so rebuilds see them and other runs never do.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar
from uuid import UUID, uuid4

from .annotation_store import AnnotationRevision

SEPARATOR = "/"
T = TypeVar("T", bound=AnnotationRevision)


def job_set_id(job_id: UUID | str) -> str:
    return f"job-{job_id}"


def scoped_id(set_id: str | None, local_id: str) -> str:
    if not set_id:
        return local_id
    return f"{set_id}{SEPARATOR}{local_id}"


def set_of(annotation_id: str) -> str | None:
    head, separator, _ = annotation_id.partition(SEPARATOR)
    return head if separator else None


def in_set(annotation_id: str, set_id: str | None) -> bool:
    """Legacy versions (set None) only see unscoped annotations."""

    return set_of(annotation_id) == set_id


def manual_annotation_id(set_id: str | None, image_id: str) -> str:
    return scoped_id(set_id, f"manual:{image_id}:{uuid4().hex[:8]}")


def live_in_set(
    revisions: Iterable[T],
    set_id: str | None,
    *,
    category_ids: set[int] | None = None,
) -> list[T]:
    """Latest, non-deleted revisions that belong to one set (and categories)."""

    return [
        revision
        for revision in revisions
        if not revision.annotation.deleted
        and in_set(revision.annotation.annotation_id, set_id)
        and (category_ids is None or revision.annotation.category_id in category_ids)
    ]
