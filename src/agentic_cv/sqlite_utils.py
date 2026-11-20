"""Shared SQLite connection settings for the local stores."""

from __future__ import annotations

import sqlite3

BUSY_TIMEOUT_MS = 5000


def connect_sqlite(path) -> sqlite3.Connection:
    """Open a store connection that tolerates the app's worker threads.

    WAL lets readers (the live view, gallery polling) proceed while a worker
    writes, and ``busy_timeout`` waits briefly instead of raising
    ``database is locked`` when two processes touch the same file.
    """

    path = str(path)
    connection = sqlite3.connect(
        path, check_same_thread=False, timeout=BUSY_TIMEOUT_MS / 1000
    )
    connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    if path != ":memory:" and not path.startswith("file::memory:"):
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
    return connection
