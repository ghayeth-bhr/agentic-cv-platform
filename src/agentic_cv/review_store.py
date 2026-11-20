"""Per-image review status for a dataset's working annotation set.

Status lives on ``(tenant, annotation set, image)`` rather than on an
immutable dataset version, so approvals survive rebuilds of the same set.
Versions created before annotation sets existed use ``version-<id>`` as their
set key. An image with no row is ``pending``.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from typing import Iterable, Literal, Protocol
from uuid import UUID

from .sqlite_utils import connect_sqlite

ReviewStatus = Literal["pending", "approved", "rejected"]
REVIEW_STATUSES: tuple[ReviewStatus, ...] = ("pending", "approved", "rejected")


def review_set_key(annotation_set: str | None, version_id: UUID) -> str:
    return annotation_set or f"version-{version_id}"


class ReviewBackend(Protocol):
    def set_status(
        self,
        *,
        tenant_id: UUID,
        set_key: str,
        image_ids: Iterable[str],
        status: ReviewStatus,
        author_id: str,
    ) -> int: ...

    def statuses(self, *, tenant_id: UUID, set_key: str) -> dict[str, ReviewStatus]: ...

    def close(self) -> None: ...


class ReviewStore:
    def __init__(self, path: str) -> None:
        self._connection = connect_sqlite(path)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS image_reviews (
                tenant_id TEXT NOT NULL,
                set_key TEXT NOT NULL,
                image_id TEXT NOT NULL,
                status TEXT NOT NULL,
                author_id TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (tenant_id, set_key, image_id)
            )
            """
        )
        self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def set_status(
        self,
        *,
        tenant_id: UUID,
        set_key: str,
        image_ids: Iterable[str],
        status: ReviewStatus,
        author_id: str,
    ) -> int:
        if status not in REVIEW_STATUSES:
            raise ValueError("status must be pending, approved, or rejected")
        ids = list(dict.fromkeys(image_ids))
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            if status == "pending":
                self._connection.executemany(
                    "DELETE FROM image_reviews WHERE tenant_id = ? AND set_key = ? AND image_id = ?",
                    [(str(tenant_id), set_key, image_id) for image_id in ids],
                )
            else:
                self._connection.executemany(
                    """
                    INSERT INTO image_reviews (tenant_id, set_key, image_id, status, author_id, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT (tenant_id, set_key, image_id)
                    DO UPDATE SET status = excluded.status, author_id = excluded.author_id,
                                  updated_at = excluded.updated_at
                    """,
                    [(str(tenant_id), set_key, image_id, status, author_id, now) for image_id in ids],
                )
            self._connection.commit()
        return len(ids)

    def statuses(self, *, tenant_id: UUID, set_key: str) -> dict[str, ReviewStatus]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT image_id, status FROM image_reviews WHERE tenant_id = ? AND set_key = ?",
                (str(tenant_id), set_key),
            ).fetchall()
        return {row["image_id"]: row["status"] for row in rows}
