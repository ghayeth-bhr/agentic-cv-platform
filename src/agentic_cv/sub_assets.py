"""Durable logical sub-assets created from semantic-search results."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Protocol
from uuid import UUID, uuid4

from .semantic_search import SearchResult
from .sqlite_utils import connect_sqlite


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, kw_only=True)
class SubAssetRecord:
    sub_asset_id: UUID
    tenant_id: UUID
    parent_asset_ids: tuple[str, ...]
    query: str
    result_ids: tuple[str, ...]
    created_at: str
    asset_id: UUID | None = None


class SubAssetBackend(Protocol):
    def close(self) -> None: ...

    def create_from_results(
        self, *, tenant_id: UUID, query: str, results: Iterable[SearchResult]
    ) -> SubAssetRecord: ...

    def get(self, sub_asset_id: UUID) -> SubAssetRecord: ...

    def get_for_tenant(self, tenant_id: UUID, sub_asset_id: UUID) -> SubAssetRecord: ...

    def link_asset(self, sub_asset_id: UUID, *, asset_id: UUID) -> SubAssetRecord: ...

    def list_for_tenant(self, tenant_id: UUID) -> list[SubAssetRecord]: ...

    def find_by_asset(self, tenant_id: UUID, asset_id: UUID) -> SubAssetRecord | None: ...


class SubAssetStore:
    def __init__(self, path: str) -> None:
        self._connection = connect_sqlite(path)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sub_assets (
                sub_asset_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                parent_asset_ids TEXT NOT NULL,
                query TEXT NOT NULL,
                result_ids TEXT NOT NULL,
                created_at TEXT NOT NULL,
                asset_id TEXT
            )
            """
        )
        columns = {
            row["name"] for row in self._connection.execute("PRAGMA table_info(sub_assets)")
        }
        if "asset_id" not in columns:
            self._connection.execute("ALTER TABLE sub_assets ADD COLUMN asset_id TEXT")
        self._connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS sub_assets_asset_idx ON sub_assets (asset_id)"
        )
        self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def create_from_results(
        self,
        *,
        tenant_id: UUID,
        query: str,
        results: Iterable[SearchResult],
    ) -> SubAssetRecord:
        if not query.strip():
            raise ValueError("query must not be empty")
        ordered_results = tuple(results)
        if not ordered_results:
            raise ValueError("at least one search result is required")
        result_ids = tuple(result.record_id for result in ordered_results)
        if len(set(result_ids)) != len(result_ids):
            raise ValueError("search result IDs must be unique")
        parent_asset_ids = tuple(dict.fromkeys(result.asset_id for result in ordered_results))
        record = SubAssetRecord(
            sub_asset_id=uuid4(), tenant_id=tenant_id,
            parent_asset_ids=parent_asset_ids, query=query,
            result_ids=result_ids, created_at=_utc_now(),
        )
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO sub_assets (
                    sub_asset_id, tenant_id, parent_asset_ids, query, result_ids,
                    created_at, asset_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(record.sub_asset_id), str(record.tenant_id),
                    json.dumps(record.parent_asset_ids), record.query,
                    json.dumps(record.result_ids), record.created_at, None,
                ),
            )
            self._connection.commit()
        return record

    def get(self, sub_asset_id: UUID) -> SubAssetRecord:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM sub_assets WHERE sub_asset_id = ?", (str(sub_asset_id),)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown sub-asset: {sub_asset_id}")
        return self._record(row)

    def get_for_tenant(self, tenant_id: UUID, sub_asset_id: UUID) -> SubAssetRecord:
        record = self.get(sub_asset_id)
        if record.tenant_id != tenant_id:
            raise PermissionError("sub-asset does not belong to tenant")
        return record

    def link_asset(self, sub_asset_id: UUID, *, asset_id: UUID) -> SubAssetRecord:
        with self._lock:
            record = self.get(sub_asset_id)
            self._connection.execute(
                "UPDATE sub_assets SET asset_id = ? WHERE sub_asset_id = ?",
                (str(asset_id), str(sub_asset_id)),
            )
            self._connection.commit()
        return SubAssetRecord(
            sub_asset_id=record.sub_asset_id,
            tenant_id=record.tenant_id,
            parent_asset_ids=record.parent_asset_ids,
            query=record.query,
            result_ids=record.result_ids,
            created_at=record.created_at,
            asset_id=asset_id,
        )

    def list_for_tenant(self, tenant_id: UUID) -> list[SubAssetRecord]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM sub_assets WHERE tenant_id = ? ORDER BY created_at",
                (str(tenant_id),),
            ).fetchall()
        return [self._record(row) for row in rows]

    def find_by_asset(self, tenant_id: UUID, asset_id: UUID) -> SubAssetRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM sub_assets WHERE tenant_id = ? AND asset_id = ?",
                (str(tenant_id), str(asset_id)),
            ).fetchone()
        return self._record(row) if row is not None else None

    @staticmethod
    def _record(row: sqlite3.Row) -> SubAssetRecord:
        raw_asset_id = row["asset_id"] if "asset_id" in row.keys() else None
        return SubAssetRecord(
            sub_asset_id=UUID(row["sub_asset_id"]),
            tenant_id=UUID(row["tenant_id"]),
            parent_asset_ids=tuple(json.loads(row["parent_asset_ids"])),
            query=row["query"],
            result_ids=tuple(json.loads(row["result_ids"])),
            created_at=row["created_at"],
            asset_id=UUID(raw_asset_id) if raw_asset_id else None,
        )
