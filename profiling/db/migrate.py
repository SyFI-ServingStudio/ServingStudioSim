"""L1 profile DB migration entry point."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 2
SCHEMA_HASH = "l1-python-metric-family-v2"


def migrate(db_path: Path | str, target_version: int = SCHEMA_VERSION) -> None:
    if target_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported profile DB schema version {target_version}")

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        migrate_connection(conn, target_version=target_version)


def migrate_connection(
    conn: sqlite3.Connection,
    target_version: int = SCHEMA_VERSION,
) -> None:
    """Migrate inside the caller's write transaction.

    Table writes use this form so schema preparation and row persistence share
    one transaction. Read paths never receive a writable connection and cannot
    invoke migration as a side effect.
    """
    if target_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported profile DB schema version {target_version}")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS _db_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO _db_metadata(key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        [
            ("schema_version", str(SCHEMA_VERSION)),
            ("schema_hash", SCHEMA_HASH),
        ],
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO _db_metadata(key, value)
        VALUES ('created_at', CURRENT_TIMESTAMP)
        """
    )
    conn.execute(
        """
        INSERT INTO _db_metadata(key, value)
        VALUES ('last_migrated_at', CURRENT_TIMESTAMP)
        ON CONFLICT(key) DO UPDATE SET value = CURRENT_TIMESTAMP
        """
    )
