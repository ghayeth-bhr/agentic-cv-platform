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


class SubAssetBackend(Protocol):
    def close(self) -> None: ...

    def create_from_results(
        self, *, tenant_id: UUID, query: str, results: Iterable[SearchResult]
    ) -> SubAssetRecord: ...

    def get(self, sub_asset_id: UUID) -> SubAssetRecord: ...

    def get_for_tenant(self, tenant_id: UUID, sub_asset_id: UUID) -> SubAssetRecord: ...


class SubAssetStore:
    def __init__(self, path: str) -> None:
        self._connection = sqlite3.connect(path, check_same_thread=False)
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
                created_at TEXT NOT NULL
            )
            """
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
                "INSERT INTO sub_assets VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(record.sub_asset_id), str(record.tenant_id),
                    json.dumps(record.parent_asset_ids), record.query,
                    json.dumps(record.result_ids), record.created_at,
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
        return SubAssetRecord(
            sub_asset_id=UUID(row["sub_asset_id"]), tenant_id=UUID(row["tenant_id"]),
            parent_asset_ids=tuple(json.loads(row["parent_asset_ids"])),
            query=row["query"], result_ids=tuple(json.loads(row["result_ids"])),
            created_at=row["created_at"],
        )

    def get_for_tenant(self, tenant_id: UUID, sub_asset_id: UUID) -> SubAssetRecord:
        record = self.get(sub_asset_id)
        if record.tenant_id != tenant_id:
            raise PermissionError("sub-asset does not belong to tenant")
        return record
