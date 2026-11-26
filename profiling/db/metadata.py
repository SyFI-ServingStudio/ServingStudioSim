"""Manifest-facing profile DB metadata helpers."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from profiling.db import layout
from profiling.db.migrate import SCHEMA_HASH, SCHEMA_VERSION, require_current
from profiling.db.registry import iter_kernel_profiler_specs
from profiling.db.storage import RUN_TABLE


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
    """The profile DB directory's metadata, over its kind files.

    Every file must carry the same schema; ``created_at`` is the earliest
    file's and ``last_migrated_at`` the latest write's.
    """
    schemas: set[tuple[int, str]] = set()
    created: list[str] = []
    migrated: list[str] = []
    for path in layout.kind_files(layout.require_directory(db_path)):
        conn = _connect_read_only(path)
        if conn is None:
            continue
        with closing(conn):
            if not _table_exists(conn, "_db_metadata"):
                continue
            rows = dict(conn.execute("SELECT key, value FROM _db_metadata").fetchall())
        schemas.add(
            (int(rows.get("schema_version", SCHEMA_VERSION)), rows.get("schema_hash", SCHEMA_HASH))
        )
        if rows.get("created_at"):
            created.append(rows["created_at"])
        if rows.get("last_migrated_at"):
            migrated.append(rows["last_migrated_at"])
    if len(schemas) > 1:
        raise RuntimeError(f"{db_path} mixes profile DB schemas {sorted(schemas)}")
    schema_version, schema_hash = schemas.pop() if schemas else (SCHEMA_VERSION, SCHEMA_HASH)
    return DbMetadata(
        schema_version=schema_version,
        schema_hash=schema_hash,
        created_at=min(created) if created else None,
        last_migrated_at=max(migrated) if migrated else None,
    )


def get_profiler_versions(
    db_path: Path | str,
    used_op_families: list[str] | None = None,
) -> list[ProfilerVersion]:
    root = layout.require_directory(db_path)
    selected = set(used_op_families or [])
    versions: set[tuple[str, str]] = set()
    for profiler_spec in iter_kernel_profiler_specs():
        op_family = profiler_spec.kernel_kind
        if selected and op_family not in selected and profiler_spec.table_name not in selected:
            continue
        path = layout.kind_file(root, profiler_spec.table_name)
        conn = _connect_read_only(path)
        if conn is None:
            continue
        with closing(conn):
            require_current(conn, str(path))
            if not _table_exists(conn, profiler_spec.table_name):
                continue
            if not _column_exists(conn, profiler_spec.table_name, "run_key"):
                continue
            # Table._row_values populates profiler_git_hash from ProfileRow or
            # the current profiler repo commit; this helper only reads it for
            # manifest/repro reporting.
            rows = conn.execute(
                f"""
                SELECT DISTINCT r.profiler_git_hash
                FROM {profiler_spec.table_name} t JOIN {RUN_TABLE} r ON r.run_key = t.run_key
                WHERE r.profiler_git_hash != ''
                """
            ).fetchall()
            for (profiler_git_hash,) in rows:
                versions.add((op_family, str(profiler_git_hash)))
    return [
        ProfilerVersion(op_family=op_family, profiler_git_hash=profiler_git_hash)
        for op_family, profiler_git_hash in sorted(versions)
    ]


def _connect_read_only(db_path: Path | str) -> sqlite3.Connection | None:
    path = Path(db_path)
    if not path.is_file():
        return None
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


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
