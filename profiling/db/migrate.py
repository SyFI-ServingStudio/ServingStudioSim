"""L1 profile DB migration entry point."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from profiling.db import storage
from profiling.db.storage import NON_ARG_COLUMNS, RUN_COLUMNS, quote

SCHEMA_VERSION = 3
SCHEMA_HASH = "l1-compact-keys-v3"

_METRIC_COLUMNS = frozenset(
    {"time_ms", "tflops", "memory_bandwidth_gbps", "algbw_gbps", "busbw_gbps", "energy_j"}
)
# Tables an older checkout wrote that this one no longer reads (the kernel-config
# registry). Migration drops them; `profiling.db.merge` refuses an input that
# still holds one.
RETIRED_TABLES = (
    "_kernel_config",
    "_kernel_config_source",
    "_kernel_config_role",
    "_kernel_config_use",
    "_kernel_config_blob",
)


def migrate(db_path: Path | str, target_version: int = SCHEMA_VERSION) -> bool:
    """Migrate the DB at ``db_path``; VACUUM it when rows were rewritten.

    Returns whether anything was upgraded. Writers migrate implicitly, but only
    this entry point reclaims the pages the upgraded v2 tables leave free.
    """
    if target_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported profile DB schema version {target_version}")

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        upgraded = migrate_connection(conn, target_version=target_version)
        conn.commit()
        if upgraded:
            conn.execute("VACUUM")
    finally:
        conn.close()
    return upgraded


def migrate_connection(
    conn: sqlite3.Connection,
    target_version: int = SCHEMA_VERSION,
) -> bool:
    """Migrate inside the caller's write transaction.

    Table writes use this form so schema preparation and row persistence share
    one transaction. Read paths never receive a writable connection and cannot
    invoke migration as a side effect. Returns whether a v2 table was upgraded
    or a retired table dropped.
    """
    if target_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported profile DB schema version {target_version}")

    if not conn.in_transaction:
        # The upgrade rewrites tables; another writer must not interleave.
        conn.execute("BEGIN IMMEDIATE")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS _db_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    upgraded = _upgrade_v2(conn)
    upgraded |= _drop_retired(conn)
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
    return upgraded


def require_current(conn: sqlite3.Connection, where: str) -> None:
    """Refuse to read a DB an older checkout wrote until it is migrated."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = '_db_metadata'"
    ).fetchone()
    if row is None:
        return
    version = conn.execute("SELECT value FROM _db_metadata WHERE key = 'schema_version'").fetchone()
    if version is not None and version[0] != str(SCHEMA_VERSION):
        raise RuntimeError(
            f"{where} has profile DB schema v{version[0]}; this checkout reads "
            f"v{SCHEMA_VERSION}. Run `uv run python -m launcher kernel-profile migrate-db {where}`."
        )


# ── v2 → v3 ──────────────────────────────────────────────────────────────────


def _upgrade_v2(conn: sqlite3.Connection) -> bool:
    """Rewrite every table still in its v2 layout (see ``profiling.db.storage``).

    Each table is recognised by its columns, not by the metadata version: a DB
    can hold kind tables written before the metadata table existed.
    """
    kinds = [name for name in _user_tables(conn) if _is_v2_kind_table(conn, name)]
    if not kinds:
        return False
    conn.execute(storage.RUN_SCHEMA)
    for name in kinds:
        _upgrade_kind_table(conn, name)
    return True


def _drop_retired(conn: sqlite3.Connection) -> bool:
    retired = [name for name in _user_tables(conn) if name in RETIRED_TABLES]
    for name in retired:
        conn.execute(f"DROP TABLE {quote(name)}")
    return bool(retired)


def _user_tables(conn: sqlite3.Connection) -> list[str]:
    return [
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
    ]


def _columns(conn: sqlite3.Connection, table: str) -> list[tuple]:
    return conn.execute(f"PRAGMA table_info({quote(table)})").fetchall()


def _is_v2_kind_table(conn: sqlite3.Connection, table: str) -> bool:
    if table.startswith("_"):
        return False
    return "profiler_git_hash" in {row[1] for row in _columns(conn, table)}


def _column_def(row: tuple) -> str:
    _, name, declared, notnull, default, _ = row
    out = f"{quote(name)} {declared}".rstrip()
    if notnull:
        out += " NOT NULL"
    if default is not None:
        out += f" DEFAULT {default}"
    return out


def _upgrade_kind_table(conn: sqlite3.Connection, table: str) -> None:
    explicit = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = ? AND sql IS NOT NULL",
        (table,),
    ).fetchall()
    if explicit:
        raise ValueError(f"{table} has explicit indexes {explicit}; migrate them by hand")
    info = _columns(conn, table)
    names = [str(row[1]) for row in info]
    declared = {str(row[1]): str(row[2]) for row in info}
    args = [name for name in names if name not in NON_ARG_COLUMNS]
    metrics = [name for name in names if name in _METRIC_COLUMNS]
    by_name = {str(row[1]): row for row in info}
    staging = f"_v3_{table}"
    conn.execute(
        storage.kind_table_schema(
            staging,
            [_column_def(by_name[name]) for name in args],
            [_column_def(by_name[name]) for name in metrics],
        )
    )
    carried = [
        "id",
        "gpu_name",
        "backend",
        *args,
        "verified",
        *metrics,
        "is_outlier",
        "retry_count",
        "outlier_reason",
    ]
    read = [*carried, *RUN_COLUMNS, "profiler_run_at", "created_at"]
    write = [*carried, "args_hash", "run_key", "profiler_run_at", "created_at"]
    insert = (
        f"INSERT INTO {quote(staging)} ({', '.join(quote(c) for c in write)}) "
        f"VALUES ({', '.join('?' for _ in write)})"
    )
    rows = conn.execute(f"SELECT {', '.join(quote(c) for c in read)} FROM {quote(table)}")
    width = len(carried)
    arg_slice = slice(3, 3 + len(args))
    for row in rows.fetchall():
        values = list(row[:width])
        provenance = tuple(row[width : width + len(RUN_COLUMNS)])
        run_at, created_at = row[width + len(RUN_COLUMNS) :]
        if provenance[0] is None or run_at is None or created_at is None:
            raise ValueError(f"{table} row {row[0]} lacks provenance or timestamps")
        key = storage.args_hash(dict(zip(args, values[arg_slice], strict=True)), declared)
        values += [
            key,
            storage.run_key(conn, provenance),
            storage.epoch(run_at),
            storage.epoch(created_at),
        ]
        conn.execute(insert, values)
    conn.execute(f"DROP TABLE {quote(table)}")
    conn.execute(f"ALTER TABLE {quote(staging)} RENAME TO {quote(table)}")
