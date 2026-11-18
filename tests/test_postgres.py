"""Tests for the opt-in PostgreSQL foundation (no live server required by default)."""

import os

import pytest

from agentic_cv.postgres import (
    DEFAULT_SCHEMA,
    MIGRATIONS,
    Migration,
    MigrationError,
    PostgresConfig,
    apply_migrations,
)


class FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class FakeConnection:
    """Minimal psycopg-like connection that records statements."""

    def __init__(self, applied=(), *, fail_on=None):
        self.applied = dict(applied)
        self.statements = []
        self.commits = 0
        self.rollbacks = 0
        self.fail_on = fail_on

    def execute(self, sql, params=None):
        self.statements.append((sql, params))
        if self.fail_on is not None and self.fail_on in sql:
            raise RuntimeError("simulated database failure")
        stripped = sql.strip()
        if stripped.startswith("SELECT version FROM"):
            return FakeCursor([(version,) for version in self.applied])
        if stripped.startswith("INSERT INTO") and "schema_migrations" in stripped:
            version, name = params
            self.applied[version] = name
        return FakeCursor([])

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def executed(self):
        return "\n".join(sql for sql, _ in self.statements)


def test_config_is_opt_in_through_the_environment(monkeypatch):
    monkeypatch.delenv("AGENTIC_CV_POSTGRES_DSN", raising=False)

    assert PostgresConfig.from_environment() is None

    monkeypatch.setenv("AGENTIC_CV_POSTGRES_DSN", "postgresql://user:pass@localhost/cv")
    config = PostgresConfig.from_environment()

    assert config is not None
    assert config.dsn == "postgresql://user:pass@localhost/cv"
    assert config.schema == DEFAULT_SCHEMA


def test_config_reads_schema_override(monkeypatch):
    monkeypatch.setenv("AGENTIC_CV_POSTGRES_DSN", "postgresql://localhost/cv")
    monkeypatch.setenv("AGENTIC_CV_POSTGRES_SCHEMA", "agentic_cv")

    config = PostgresConfig.from_environment()

    assert config is not None
    assert config.schema == "agentic_cv"


def test_config_rejects_empty_dsn_and_unsafe_schema():
    with pytest.raises(ValueError, match="dsn"):
        PostgresConfig(dsn="  ")
    with pytest.raises(ValueError, match="schema"):
        PostgresConfig(dsn="postgresql://localhost/cv", schema='public; DROP SCHEMA public')


def test_migration_definitions_are_ordered_and_unique():
    versions = [migration.version for migration in MIGRATIONS]

    assert versions == sorted(versions)
    assert len(set(versions)) == len(versions)
    assert all(migration.statements for migration in MIGRATIONS)


def test_apply_migrations_creates_schema_and_records_versions():
    connection = FakeConnection()

    applied = apply_migrations(connection)

    assert applied == tuple(migration.version for migration in MIGRATIONS)
    executed = connection.executed()
    assert "CREATE SCHEMA IF NOT EXISTS public" in executed
    assert "CREATE EXTENSION IF NOT EXISTS vector" in executed
    assert "CREATE TABLE IF NOT EXISTS assets" in executed
    assert "CREATE TABLE IF NOT EXISTS vector_records" in executed
    assert connection.applied == {
        migration.version: migration.name for migration in MIGRATIONS
    }
    assert connection.commits == len(MIGRATIONS)


def test_apply_migrations_is_idempotent():
    connection = FakeConnection()
    apply_migrations(connection)
    connection.statements.clear()

    applied = apply_migrations(connection)

    assert applied == ()
    assert "CREATE TABLE IF NOT EXISTS assets" not in connection.executed()
    assert connection.commits == len(MIGRATIONS)


def test_failed_migration_rolls_back_and_is_not_recorded():
    connection = FakeConnection(
        fail_on="CREATE TABLE IF NOT EXISTS broken", applied=()
    )
    migrations = (
        MIGRATIONS[0],
        Migration(
            version=99,
            name="broken",
            statements=(
                "CREATE TABLE IF NOT EXISTS broken (id INTEGER)",
            ),
        ),
    )

    with pytest.raises(MigrationError, match="99 \\(broken\\)"):
        apply_migrations(connection, migrations=migrations)

    assert connection.rollbacks == 1
    assert 99 not in connection.applied
    assert MIGRATIONS[0].version in connection.applied


def test_apply_migrations_sorts_and_rejects_duplicate_versions():
    ordered = (
        Migration(version=2, name="second", statements=("SELECT 2",)),
        Migration(version=1, name="first", statements=("SELECT 1",)),
    )
    connection = FakeConnection()

    assert apply_migrations(connection, migrations=ordered) == (1, 2)

    with pytest.raises(ValueError, match="unique"):
        apply_migrations(
            FakeConnection(),
            migrations=(
                Migration(version=1, name="a", statements=("SELECT 1",)),
                Migration(version=1, name="b", statements=("SELECT 1",)),
            ),
        )


def test_apply_migrations_rejects_unsafe_schema_identifier():
    with pytest.raises(ValueError, match="schema"):
        apply_migrations(FakeConnection(), schema="public; DROP SCHEMA public")


@pytest.mark.postgres
def test_apply_migrations_against_real_postgres():
    dsn = os.environ.get("AGENTIC_CV_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("set AGENTIC_CV_TEST_POSTGRES_DSN to run against a live PostgreSQL")
    from agentic_cv.postgres import connect

    connection = connect(PostgresConfig(dsn=dsn, schema="agentic_cv_test"))
    try:
        first = apply_migrations(connection, schema="agentic_cv_test")
        second = apply_migrations(connection, schema="agentic_cv_test")
        rows = connection.execute(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'agentic_cv_test'
            """
        ).fetchall()
        embedding_type = connection.execute(
            """
            SELECT udt_name FROM information_schema.columns
            WHERE table_schema = 'agentic_cv_test'
              AND table_name = 'vector_records'
              AND column_name = 'embedding'
            """
        ).fetchone()[0]
    finally:
        connection.execute("DROP SCHEMA IF EXISTS agentic_cv_test CASCADE")
        connection.commit()
        connection.close()

    assert first == tuple(migration.version for migration in MIGRATIONS)
    assert second == ()
    names = {row[0] for row in rows}
    assert {"assets", "jobs", "outbox_events", "vector_records"} <= names
    assert embedding_type == "vector"
