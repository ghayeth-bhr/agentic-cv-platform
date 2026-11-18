"""Opt-in PostgreSQL/pgvector foundation: configuration, connections, and migrations.

The SQLite stores remain the default. Nothing in this module is imported by the
local runtime path, and the PostgreSQL driver is imported lazily so a default
install and the default test suite need neither ``psycopg`` nor a live server.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

DEFAULT_SCHEMA = "public"
_SCHEMA_PATTERN = re.compile(r"[a-z_][a-z0-9_]{0,62}")


class MigrationError(RuntimeError):
    """Raised when a schema migration fails and is rolled back."""


@dataclass(frozen=True, kw_only=True)
class PostgresConfig:
    """Connection settings for the opt-in PostgreSQL backend."""

    dsn: str
    schema: str = DEFAULT_SCHEMA

    def __post_init__(self) -> None:
        if not self.dsn.strip():
            raise ValueError("dsn must not be empty")
        if not _SCHEMA_PATTERN.fullmatch(self.schema):
            raise ValueError(
                "schema must match [a-z_][a-z0-9_]{0,62} to be safely interpolated"
            )

    @classmethod
    def from_environment(cls) -> "PostgresConfig | None":
        """Return a config when ``AGENTIC_CV_POSTGRES_DSN`` is set, else ``None``."""
        dsn = os.environ.get("AGENTIC_CV_POSTGRES_DSN", "").strip()
        if not dsn:
            return None
        return cls(
            dsn=dsn,
            schema=os.environ.get("AGENTIC_CV_POSTGRES_SCHEMA", DEFAULT_SCHEMA).strip()
            or DEFAULT_SCHEMA,
        )


@dataclass(frozen=True, kw_only=True)
class Migration:
    """One ordered, transactional schema change."""

    version: int
    name: str
    statements: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError("migration version must be at least 1")
        if not self.name.strip():
            raise ValueError("migration name must not be empty")
        if not self.statements:
            raise ValueError("migration must contain at least one statement")


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version=1,
        name="platform_schema",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS assets (
                asset_id UUID PRIMARY KEY,
                tenant_id UUID NOT NULL,
                name TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS uploads (
                upload_id UUID PRIMARY KEY,
                tenant_id UUID NOT NULL,
                asset_id UUID NOT NULL REFERENCES assets(asset_id),
                filename TEXT NOT NULL,
                media_type TEXT NOT NULL,
                expected_sha256 TEXT NOT NULL,
                object_key TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS sources (
                source_id UUID PRIMARY KEY,
                tenant_id UUID NOT NULL,
                asset_id UUID NOT NULL REFERENCES assets(asset_id),
                filename TEXT NOT NULL,
                media_type TEXT NOT NULL,
                object_uri TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                UNIQUE (asset_id, content_sha256)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id UUID PRIMARY KEY,
                tenant_id UUID NOT NULL,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                payload JSONB NOT NULL,
                idempotency_key TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                lease_owner TEXT,
                lease_expires_at TIMESTAMPTZ,
                cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                UNIQUE (tenant_id, idempotency_key)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS outbox_events (
                event_id UUID PRIMARY KEY,
                sequence BIGINT NOT NULL,
                event_type TEXT NOT NULL,
                tenant_id UUID NOT NULL,
                aggregate_type TEXT NOT NULL,
                aggregate_id UUID NOT NULL,
                payload JSONB NOT NULL,
                idempotency_key TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                occurred_at TIMESTAMPTZ NOT NULL,
                published_at TIMESTAMPTZ,
                next_attempt_at TIMESTAMPTZ NOT NULL,
                UNIQUE (tenant_id, idempotency_key)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS annotation_revisions (
                revision_id UUID PRIMARY KEY,
                tenant_id UUID NOT NULL,
                image_id TEXT NOT NULL,
                annotation_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
                annotation JSONB NOT NULL,
                author_id TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                UNIQUE (tenant_id, image_id, annotation_id, revision)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS sub_assets (
                sub_asset_id UUID PRIMARY KEY,
                tenant_id UUID NOT NULL,
                parent_asset_ids TEXT[] NOT NULL,
                query TEXT NOT NULL,
                result_ids TEXT[] NOT NULL,
                created_at TIMESTAMPTZ NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS dataset_versions (
                version_id UUID PRIMARY KEY,
                tenant_id UUID NOT NULL,
                name TEXT NOT NULL,
                dataset JSONB NOT NULL,
                idempotency_key TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                UNIQUE (tenant_id, idempotency_key)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS vector_records (
                record_id TEXT PRIMARY KEY,
                asset_id UUID NOT NULL,
                image_path TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                embedding VECTOR NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS assets_tenant_idx ON assets (tenant_id)",
            "CREATE INDEX IF NOT EXISTS sources_asset_idx ON sources (asset_id)",
            "CREATE INDEX IF NOT EXISTS jobs_tenant_status_idx ON jobs (tenant_id, status)",
            """
            CREATE INDEX IF NOT EXISTS outbox_pending_idx
                ON outbox_events (occurred_at, sequence)
                WHERE published_at IS NULL
            """,
            "CREATE INDEX IF NOT EXISTS vector_records_asset_idx ON vector_records (asset_id)",
        ),
    ),
    Migration(
        version=2,
        name="outbox_sequence",
        statements=(
            "CREATE SEQUENCE IF NOT EXISTS outbox_event_sequence",
            """
            ALTER TABLE outbox_events
                ALTER COLUMN sequence SET DEFAULT nextval('outbox_event_sequence')
            """,
            """
            SELECT setval(
                'outbox_event_sequence',
                COALESCE((SELECT MAX(sequence) FROM outbox_events), 0) + 1,
                false
            )
            """,
        ),
    ),
    Migration(
        version=3,
        name="jobs_sequence",
        statements=(
            "CREATE SEQUENCE IF NOT EXISTS jobs_sequence",
            "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS sequence BIGINT",
            """
            UPDATE jobs SET sequence = ordered.rn
            FROM (
                SELECT job_id, row_number() OVER (ORDER BY created_at, job_id) AS rn
                FROM jobs WHERE sequence IS NULL
            ) AS ordered
            WHERE jobs.job_id = ordered.job_id
            """,
            """
            ALTER TABLE jobs
                ALTER COLUMN sequence SET DEFAULT nextval('jobs_sequence')
            """,
            """
            SELECT setval(
                'jobs_sequence',
                COALESCE((SELECT MAX(sequence) FROM jobs), 0) + 1,
                false
            )
            """,
        ),
    ),
)


def _migration_table(schema: str) -> str:
    return f"{schema}.schema_migrations"


def apply_migrations(
    connection: Any,
    *,
    schema: str = DEFAULT_SCHEMA,
    migrations: Sequence[Migration] = MIGRATIONS,
) -> tuple[int, ...]:
    """Apply pending migrations in order and return the versions applied now.

    Each migration runs in its own transaction. A failed migration is rolled
    back and reported through :class:`MigrationError`, leaving earlier versions
    applied and the failing version unrecorded.
    """
    if not _SCHEMA_PATTERN.fullmatch(schema):
        raise ValueError("schema must match [a-z_][a-z0-9_]{0,62}")
    ordered = _validated(migrations)
    connection.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    # The vector extension is database-wide and must not be owned by a
    # per-tenant schema: dropping that schema would leave it dangling.
    connection.execute("SET search_path TO public")
    connection.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
    connection.execute(f"SET search_path TO {schema}, public")
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_migration_table(schema)} (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    applied = {
        row["version"] if hasattr(row, "keys") else row[0]
        for row in connection.execute(
            f"SELECT version FROM {_migration_table(schema)}"
        ).fetchall()
    }
    applied_now: list[int] = []
    for migration in ordered:
        if migration.version in applied:
            continue
        try:
            for statement in migration.statements:
                connection.execute(statement)
            connection.execute(
                f"INSERT INTO {_migration_table(schema)} (version, name) VALUES (%s, %s)",
                (migration.version, migration.name),
            )
            connection.commit()
        except Exception as exc:
            connection.rollback()
            raise MigrationError(
                f"migration {migration.version} ({migration.name}) failed: {exc}"
            ) from exc
        applied_now.append(migration.version)
    return tuple(applied_now)


def _validated(migrations: Iterable[Migration]) -> tuple[Migration, ...]:
    ordered = tuple(sorted(migrations, key=lambda item: item.version))
    versions = [item.version for item in ordered]
    if len(set(versions)) != len(versions):
        raise ValueError("migration versions must be unique")
    return ordered


def connect(config: PostgresConfig, *, row_factory: Any = None) -> Any:
    """Open a PostgreSQL connection scoped to the configured schema."""
    try:
        import psycopg
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PostgreSQL support requires the optional driver: "
            "install with `uv sync --extra postgres`"
        ) from exc
    connection = psycopg.connect(config.dsn, row_factory=row_factory)
    connection.execute(f"SET search_path TO {config.schema}")
    return connection
