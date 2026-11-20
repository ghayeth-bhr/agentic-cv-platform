"""Durable local storage for agent conversations and labeling runs."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Literal, Protocol
from uuid import UUID, uuid4
from .sqlite_utils import connect_sqlite

AgentSessionStatus = Literal[
    "gathering",
    "awaiting_confirmation",
    "calibrating",
    "needs_attention",
    "labeling",
    "exporting",
    "complete",
    "failed",
    "cancelled",
]
AgentRole = Literal["user", "assistant", "system"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, kw_only=True)
class AgentMessage:
    role: AgentRole
    content: str
    created_at: str

    @classmethod
    def create(cls, role: AgentRole, content: str) -> "AgentMessage":
        return cls(role=role, content=content, created_at=_utc_now())

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content, "created_at": self.created_at}


@dataclass(frozen=True, kw_only=True)
class AgentSessionRecord:
    session_id: UUID
    tenant_id: UUID
    asset_id: UUID
    asset_name: str
    message: str
    status: AgentSessionStatus
    spec: dict[str, Any] | None
    planner_mode: str
    planner_note: str | None
    prompt: str | None
    threshold: float | None
    calibration: dict[str, Any] | None
    label_job_id: UUID | None
    dataset_version_id: UUID | None
    export_job_id: UUID | None
    error: str | None
    messages: tuple[AgentMessage, ...]
    created_at: str
    updated_at: str


class AgentSessionBackend(Protocol):
    def close(self) -> None: ...

    def create(self, record: AgentSessionRecord) -> AgentSessionRecord: ...

    def get(self, *, tenant_id: UUID, session_id: UUID) -> AgentSessionRecord: ...

    def get_any(self, session_id: UUID) -> AgentSessionRecord: ...

    def list(self, *, tenant_id: UUID, limit: int = 50) -> list[AgentSessionRecord]: ...

    def list_open(self) -> list[AgentSessionRecord]: ...

    def save(self, record: AgentSessionRecord) -> AgentSessionRecord: ...


class AgentSessionStore:
    def __init__(self, path: str) -> None:
        self._connection = connect_sqlite(path)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_sessions (
                session_id TEXT PRIMARY KEY,
                sequence INTEGER NOT NULL,
                tenant_id TEXT NOT NULL,
                asset_id TEXT NOT NULL,
                asset_name TEXT NOT NULL,
                message TEXT NOT NULL,
                status TEXT NOT NULL,
                spec TEXT,
                planner_mode TEXT NOT NULL,
                planner_note TEXT,
                prompt TEXT,
                threshold REAL,
                calibration TEXT,
                label_job_id TEXT,
                dataset_version_id TEXT,
                export_job_id TEXT,
                error TEXT,
                messages TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def create(self, record: AgentSessionRecord) -> AgentSessionRecord:
        with self._lock:
            sequence = self._connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM agent_sessions"
            ).fetchone()[0]
            self._connection.execute(
                """
                INSERT INTO agent_sessions (
                    session_id, sequence, tenant_id, asset_id, asset_name, message,
                    status, spec, planner_mode, planner_note, prompt, threshold,
                    calibration, label_job_id, dataset_version_id, export_job_id,
                    error, messages, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(record.session_id), sequence, str(record.tenant_id),
                    str(record.asset_id), record.asset_name, record.message,
                    record.status, _encode(record.spec), record.planner_mode,
                    record.planner_note, record.prompt, record.threshold,
                    _encode(record.calibration), _uuid(record.label_job_id),
                    _uuid(record.dataset_version_id), _uuid(record.export_job_id),
                    record.error, _encode([item.to_dict() for item in record.messages]),
                    record.created_at, record.updated_at,
                ),
            )
            self._connection.commit()
            return self.get_any(record.session_id)

    def get(self, *, tenant_id: UUID, session_id: UUID) -> AgentSessionRecord:
        record = self.get_any(session_id)
        if record.tenant_id != tenant_id:
            raise KeyError(f"unknown agent session: {session_id}")
        return record

    def get_any(self, session_id: UUID) -> AgentSessionRecord:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM agent_sessions WHERE session_id = ?", (str(session_id),)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown agent session: {session_id}")
        return self._record(row)

    def list(self, *, tenant_id: UUID, limit: int = 50) -> list[AgentSessionRecord]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM agent_sessions WHERE tenant_id = ?
                ORDER BY sequence DESC LIMIT ?
                """,
                (str(tenant_id), limit),
            ).fetchall()
        return [self._record(row) for row in rows]

    def list_open(self) -> list[AgentSessionRecord]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM agent_sessions
                WHERE status IN ('calibrating', 'labeling', 'exporting')
                ORDER BY sequence
                """
            ).fetchall()
        return [self._record(row) for row in rows]

    def save(self, record: AgentSessionRecord) -> AgentSessionRecord:
        with self._lock:
            existing = self._connection.execute(
                "SELECT 1 FROM agent_sessions WHERE session_id = ?",
                (str(record.session_id),),
            ).fetchone()
            if existing is None:
                raise KeyError(f"unknown agent session: {record.session_id}")
            updated = replace(record, updated_at=_utc_now())
            self._connection.execute(
                """
                UPDATE agent_sessions SET
                    status = ?, spec = ?, planner_mode = ?, planner_note = ?,
                    prompt = ?, threshold = ?, calibration = ?, label_job_id = ?,
                    dataset_version_id = ?, export_job_id = ?, error = ?,
                    messages = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (
                    updated.status, _encode(updated.spec), updated.planner_mode,
                    updated.planner_note, updated.prompt, updated.threshold,
                    _encode(updated.calibration), _uuid(updated.label_job_id),
                    _uuid(updated.dataset_version_id), _uuid(updated.export_job_id),
                    updated.error,
                    _encode([item.to_dict() for item in updated.messages]),
                    updated.updated_at, str(updated.session_id),
                ),
            )
            self._connection.commit()
        return self.get_any(record.session_id)

    @classmethod
    def _record(cls, row: sqlite3.Row) -> AgentSessionRecord:
        return AgentSessionRecord(
            session_id=UUID(row["session_id"]),
            tenant_id=UUID(row["tenant_id"]),
            asset_id=UUID(row["asset_id"]),
            asset_name=row["asset_name"],
            message=row["message"],
            status=row["status"],
            spec=json.loads(row["spec"]) if row["spec"] else None,
            planner_mode=row["planner_mode"],
            planner_note=row["planner_note"],
            prompt=row["prompt"],
            threshold=row["threshold"],
            calibration=json.loads(row["calibration"]) if row["calibration"] else None,
            label_job_id=UUID(row["label_job_id"]) if row["label_job_id"] else None,
            dataset_version_id=(
                UUID(row["dataset_version_id"]) if row["dataset_version_id"] else None
            ),
            export_job_id=UUID(row["export_job_id"]) if row["export_job_id"] else None,
            error=row["error"],
            messages=tuple(
                AgentMessage(
                    role=item["role"],
                    content=item["content"],
                    created_at=item["created_at"],
                )
                for item in json.loads(row["messages"])
            ),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


def new_session_record(
    *,
    tenant_id: UUID,
    asset_id: UUID,
    asset_name: str,
    message: str,
    spec: dict[str, Any],
    planner_mode: str,
    planner_note: str | None,
    prompt: str | None,
    messages: tuple[AgentMessage, ...],
) -> AgentSessionRecord:
    now = _utc_now()
    return AgentSessionRecord(
        session_id=uuid4(),
        tenant_id=tenant_id,
        asset_id=asset_id,
        asset_name=asset_name,
        message=message,
        status="awaiting_confirmation",
        spec=spec,
        planner_mode=planner_mode,
        planner_note=planner_note,
        prompt=prompt,
        threshold=None,
        calibration=None,
        label_job_id=None,
        dataset_version_id=None,
        export_job_id=None,
        error=None,
        messages=messages,
        created_at=now,
        updated_at=now,
    )


def _encode(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True)


def _uuid(value: UUID | None) -> str | None:
    return str(value) if value is not None else None
