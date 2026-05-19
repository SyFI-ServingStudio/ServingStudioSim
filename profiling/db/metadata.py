"""Manifest-facing profile DB metadata helpers."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from profiling.db.migrate import SCHEMA_HASH, SCHEMA_VERSION, migrate
from profiling.db.registry import iter_kernel_profiler_specs


@dataclass(frozen=True)
class DbMetadata:
    schema_version: int
    schema_hash: str
    created_at: str | None = None
    last_migrated_at: str | None = None


@dataclass(frozen=True)
class ProfilerVersion:
    op_family: str
    profiler_git_hash: str


def get_db_metadata(db_path: Path | str) -> DbMetadata:
    migrate(db_path)
    with sqlite3.connect(db_path) as conn:
        rows = dict(conn.execute("SELECT key, value FROM _db_metadata").fetchall())
    return DbMetadata(
        schema_version=int(rows.get("schema_version", SCHEMA_VERSION)),
        schema_hash=rows.get("schema_hash", SCHEMA_HASH),
        created_at=rows.get("created_at"),
        last_migrated_at=rows.get("last_migrated_at"),
    )


def get_profiler_versions(
    db_path: Path | str,
    used_op_families: list[str] | None = None,
) -> list[ProfilerVersion]:
    migrate(db_path)
    selected = set(used_op_families or [])
    versions: set[tuple[str, str]] = set()
    with sqlite3.connect(db_path) as conn:
        for profiler_spec in iter_kernel_profiler_specs():
            op_family = profiler_spec.kernel_kind.value
            if selected and op_family not in selected and profiler_spec.table_name not in selected:
                continue
            if not _table_exists(conn, profiler_spec.table_name):
                continue
            if not _column_exists(conn, profiler_spec.table_name, "profiler_git_hash"):
                continue
            # Table._row_values populates profiler_git_hash from ProfileRow or
            # the current profiler repo commit; this helper only reads it for
            # manifest/repro reporting.
            rows = conn.execute(
                f"""
                SELECT DISTINCT profiler_git_hash
                FROM {profiler_spec.table_name}
                WHERE profiler_git_hash IS NOT NULL AND profiler_git_hash != ''
                """
            ).fetchall()
            for (profiler_git_hash,) in rows:
                versions.add((op_family, str(profiler_git_hash)))
    return [
        ProfilerVersion(op_family=op_family, profiler_git_hash=profiler_git_hash)
        for op_family, profiler_git_hash in sorted(versions)
    ]


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        LIMIT 1
        """,
        (table_name,),
    ).fetchone()
    return row is not None


def _column_exists(conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(row[1] == column_name for row in rows)
