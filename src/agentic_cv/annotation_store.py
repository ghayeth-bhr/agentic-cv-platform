"""SQLite-backed immutable annotation revisions with optimistic concurrency."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import UUID, uuid4

from .dataset import Annotation
from .sqlite_utils import connect_sqlite


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RevisionConflictError(ValueError):
    """Raised when a caller edits a stale annotation revision."""


@dataclass(frozen=True, kw_only=True)
class AnnotationRevision:
    revision_id: UUID
    tenant_id: UUID
    image_id: str
    annotation: Annotation
    revision: int
    author_id: str
    created_at: str


class AnnotationRevisionBackend(Protocol):
    def close(self) -> None: ...

    def save(
        self,
        *,
        tenant_id: UUID,
        annotation: Annotation,
        author_id: str,
        expected_revision: int | None = 0,
    ) -> AnnotationRevision: ...

    def latest(
        self, *, tenant_id: UUID, image_id: str, annotation_id: str
    ) -> AnnotationRevision: ...

    def list_latest(
        self,
        *,
        tenant_id: UUID,
        image_ids: list[str] | None = None,
    ) -> list[AnnotationRevision]: ...

    def history(
        self, *, tenant_id: UUID, image_id: str, annotation_id: str
    ) -> list[AnnotationRevision]: ...


class AnnotationRevisionStore:
    def __init__(self, path: str) -> None:
        self._connection = connect_sqlite(path)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS annotation_revisions (
                revision_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                image_id TEXT NOT NULL,
                annotation_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
                annotation TEXT NOT NULL,
                author_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE (tenant_id, image_id, annotation_id, revision)
            )
            """
        )
        self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def save(
        self,
        *,
        tenant_id: UUID,
        annotation: Annotation,
        author_id: str,
        expected_revision: int | None = 0,
    ) -> AnnotationRevision:
        if not author_id.strip():
            raise ValueError("author_id must not be empty")
        if expected_revision is not None and expected_revision < 0:
            raise ValueError("expected_revision must not be negative")
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM annotation_revisions
                WHERE tenant_id = ? AND image_id = ? AND annotation_id = ?
                ORDER BY revision DESC LIMIT 1
                """,
                (str(tenant_id), annotation.image_id, annotation.annotation_id),
            ).fetchone()
            current_revision = row["revision"] if row is not None else 0
            if expected_revision is not None and current_revision != expected_revision:
                raise RevisionConflictError(
                    f"expected revision {expected_revision}, current revision is {current_revision}"
                )
            revision = AnnotationRevision(
                revision_id=uuid4(), tenant_id=tenant_id,
                image_id=annotation.image_id, annotation=annotation,
                revision=current_revision + 1, author_id=author_id,
                created_at=_utc_now(),
            )
            self._connection.execute(
                "INSERT INTO annotation_revisions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(revision.revision_id), str(revision.tenant_id), revision.image_id,
                    revision.annotation.annotation_id, revision.revision,
                    json.dumps(self._annotation_dict(revision.annotation), sort_keys=True),
                    revision.author_id, revision.created_at,
                ),
            )
            self._connection.commit()
            return revision

    def latest(self, *, tenant_id: UUID, image_id: str, annotation_id: str) -> AnnotationRevision:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM annotation_revisions
                WHERE tenant_id = ? AND image_id = ? AND annotation_id = ?
                ORDER BY revision DESC LIMIT 1
                """,
                (str(tenant_id), image_id, annotation_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown annotation: {annotation_id}")
        return self._record(row)

    def list_latest(
        self,
        *,
        tenant_id: UUID,
        image_ids: list[str] | None = None,
    ) -> list[AnnotationRevision]:
        if image_ids is not None and not image_ids:
            return []
        clauses = [
            "r.tenant_id = ?",
            """
            r.revision = (
                SELECT MAX(r2.revision) FROM annotation_revisions r2
                WHERE r2.tenant_id = r.tenant_id
                  AND r2.image_id = r.image_id
                  AND r2.annotation_id = r.annotation_id
            )
            """,
        ]
        values: list[str] = [str(tenant_id)]
        if image_ids is not None:
            placeholders = ", ".join("?" for _ in image_ids)
            clauses.append(f"r.image_id IN ({placeholders})")
            values.extend(image_ids)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT r.* FROM annotation_revisions r
                WHERE {' AND '.join(clauses)}
                ORDER BY r.image_id, r.annotation_id
                """,
                values,
            ).fetchall()
        return [self._record(row) for row in rows]

    def history(self, *, tenant_id: UUID, image_id: str, annotation_id: str) -> list[AnnotationRevision]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM annotation_revisions
                WHERE tenant_id = ? AND image_id = ? AND annotation_id = ?
                ORDER BY revision
                """,
                (str(tenant_id), image_id, annotation_id),
            ).fetchall()
        return [self._record(row) for row in rows]

    @staticmethod
    def _annotation_dict(annotation: Annotation) -> dict:
        return {
            "annotation_id": annotation.annotation_id,
            "image_id": annotation.image_id,
            "category_id": annotation.category_id,
            "bbox": list(annotation.bbox),
            "segmentation": [list(polygon) for polygon in annotation.segmentation],
            "score": annotation.score,
            **({"deleted": True} if annotation.deleted else {}),
        }

    @classmethod
    def _record(cls, row: sqlite3.Row) -> AnnotationRevision:
        value = json.loads(row["annotation"])
        annotation = Annotation(
            annotation_id=value["annotation_id"], image_id=value["image_id"],
            category_id=value["category_id"], bbox=tuple(value["bbox"]),
            segmentation=tuple(tuple(polygon) for polygon in value["segmentation"]),
            score=value["score"],
            deleted=bool(value.get("deleted", False)),
        )
        return AnnotationRevision(
            revision_id=UUID(row["revision_id"]), tenant_id=UUID(row["tenant_id"]),
            image_id=row["image_id"], annotation=annotation,
            revision=row["revision"], author_id=row["author_id"],
            created_at=row["created_at"],
        )
