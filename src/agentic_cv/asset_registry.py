"""Durable local asset, source, and upload metadata."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4


UploadStatus = Literal["active", "completed", "aborted"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, kw_only=True)
class AssetRecord:
    asset_id: UUID
    tenant_id: UUID
    name: str
    created_at: str


@dataclass(frozen=True, kw_only=True)
class UploadRecord:
    upload_id: UUID
    tenant_id: UUID
    asset_id: UUID
    filename: str
    media_type: str
    expected_sha256: str
    object_key: str
    status: UploadStatus
    created_at: str


@dataclass(frozen=True, kw_only=True)
class SourceRecord:
    source_id: UUID
    tenant_id: UUID
    asset_id: UUID
    filename: str
    media_type: str
    object_uri: str
    content_sha256: str
    created_at: str


class AssetRegistry:
    def __init__(self, path: Path | str) -> None:
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.RLock()
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS assets (
                asset_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS uploads (
                upload_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                asset_id TEXT NOT NULL REFERENCES assets(asset_id),
                filename TEXT NOT NULL,
                media_type TEXT NOT NULL,
                expected_sha256 TEXT NOT NULL,
                object_key TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sources (
                source_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                asset_id TEXT NOT NULL REFERENCES assets(asset_id),
                filename TEXT NOT NULL,
                media_type TEXT NOT NULL,
                object_uri TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE (asset_id, content_sha256)
            );
            """
        )
        self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def create_asset(self, *, tenant_id: UUID, name: str) -> AssetRecord:
        if not name.strip():
            raise ValueError("asset name must not be empty")
        record = AssetRecord(asset_id=uuid4(), tenant_id=tenant_id, name=name, created_at=_utc_now())
        with self._lock:
            self._connection.execute(
                "INSERT INTO assets VALUES (?, ?, ?, ?)",
                (str(record.asset_id), str(record.tenant_id), record.name, record.created_at),
            )
            self._connection.commit()
        return record

    def start_upload(
        self,
        *,
        tenant_id: UUID,
        asset_id: UUID,
        filename: str,
        media_type: str,
        expected_sha256: str,
    ) -> UploadRecord:
        if not filename.strip() or not media_type.strip() or len(expected_sha256) != 64:
            raise ValueError("filename, media_type, and a SHA-256 digest are required")
        with self._lock:
            asset = self._connection.execute(
                "SELECT tenant_id FROM assets WHERE asset_id = ?", (str(asset_id),)
            ).fetchone()
            if asset is None:
                raise KeyError(f"unknown asset: {asset_id}")
            if asset["tenant_id"] != str(tenant_id):
                raise PermissionError("asset does not belong to tenant")
            record = UploadRecord(
                upload_id=uuid4(),
                tenant_id=tenant_id,
                asset_id=asset_id,
                filename=filename,
                media_type=media_type,
                expected_sha256=expected_sha256,
                object_key=f"tenants/{tenant_id}/assets/{asset_id}/uploads/{uuid4()}/{filename}",
                status="active",
                created_at=_utc_now(),
            )
            self._connection.execute(
                "INSERT INTO uploads VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(record.upload_id), str(record.tenant_id), str(record.asset_id),
                    record.filename, record.media_type, record.expected_sha256,
                    record.object_key, record.status, record.created_at,
                ),
            )
            self._connection.commit()
            return record

    def get_upload(self, upload_id: UUID) -> UploadRecord:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM uploads WHERE upload_id = ?", (str(upload_id),)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown upload: {upload_id}")
        return UploadRecord(
            upload_id=UUID(row["upload_id"]), tenant_id=UUID(row["tenant_id"]),
            asset_id=UUID(row["asset_id"]), filename=row["filename"],
            media_type=row["media_type"], expected_sha256=row["expected_sha256"],
            object_key=row["object_key"], status=row["status"], created_at=row["created_at"],
        )

    def complete_upload(self, upload_id: UUID, *, object_uri: str) -> SourceRecord:
        with self._lock:
            upload = self.get_upload(upload_id)
            if upload.status == "completed":
                row = self._connection.execute(
                    "SELECT * FROM sources WHERE asset_id = ? AND content_sha256 = ?",
                    (str(upload.asset_id), upload.expected_sha256),
                ).fetchone()
                if row is not None:
                    return self._source(row)
            if upload.status != "active":
                raise ValueError(f"upload is not active: {upload.status}")
            existing = self._connection.execute(
                "SELECT * FROM sources WHERE asset_id = ? AND content_sha256 = ?",
                (str(upload.asset_id), upload.expected_sha256),
            ).fetchone()
            if existing is not None:
                self._connection.execute(
                    "UPDATE uploads SET status = 'completed' WHERE upload_id = ?",
                    (str(upload_id),),
                )
                self._connection.commit()
                return self._source(existing)
            source = SourceRecord(
                source_id=uuid4(), tenant_id=upload.tenant_id, asset_id=upload.asset_id,
                filename=upload.filename, media_type=upload.media_type,
                object_uri=object_uri, content_sha256=upload.expected_sha256,
                created_at=_utc_now(),
            )
            self._connection.execute(
                "INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(source.source_id), str(source.tenant_id), str(source.asset_id),
                    source.filename, source.media_type, source.object_uri,
                    source.content_sha256, source.created_at,
                ),
            )
            self._connection.execute(
                "UPDATE uploads SET status = 'completed' WHERE upload_id = ?",
                (str(upload_id),),
            )
            self._connection.commit()
            return source

    @staticmethod
    def _source(row: sqlite3.Row) -> SourceRecord:
        return SourceRecord(
            source_id=UUID(row["source_id"]), tenant_id=UUID(row["tenant_id"]),
            asset_id=UUID(row["asset_id"]), filename=row["filename"],
            media_type=row["media_type"], object_uri=row["object_uri"],
            content_sha256=row["content_sha256"], created_at=row["created_at"],
        )