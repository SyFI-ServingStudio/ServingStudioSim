"""L1 profile DB migration entry point."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from profiling.db import storage
from profiling.db.storage import (
    CONFIG_TABLE,
    NON_ARG_COLUMNS,
    RUN_COLUMNS,
    SOURCE_TABLE,
    USE_TABLE,
    quote,
)

SCHEMA_VERSION = 4
SCHEMA_HASH = "l1-identity-no-local-paths-v4"

_METRIC_COLUMNS = frozenset(
    {"time_ms", "tflops", "memory_bandwidth_gbps", "algbw_gbps", "busbw_gbps", "energy_j"}
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
    invoke migration as a side effect. Returns whether any table was rewritten.
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
    upgraded = _drop_identity_paths(conn) or upgraded
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
    registry = _is_v2_registry(conn)
    if not kinds and not registry:
        return False
    conn.execute(storage.RUN_SCHEMA)
    for name in kinds:
        _upgrade_kind_table(conn, name)
    if registry:
        _upgrade_registry(conn)
    return True


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


def _is_v2_registry(conn: sqlite3.Connection) -> bool:
    tables = set(_user_tables(conn))
    if USE_TABLE not in tables:
        return False
    return "kind" in {row[1] for row in _columns(conn, USE_TABLE)}


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


def _upgrade_registry(conn: sqlite3.Connection) -> None:
    for table in (CONFIG_TABLE, SOURCE_TABLE, USE_TABLE):
        conn.execute(f"ALTER TABLE {table} RENAME TO _v2{table}")
    for statement in storage.REGISTRY_SCHEMA:
        conn.execute(statement)

    for row in conn.execute(
        f"""
        SELECT id, kind, config_hash, gpu_name, profile_kind, identity, cache_coords,
            grid_axes, cells, infeasible, created_at
        FROM _v2{CONFIG_TABLE} ORDER BY id
        """
    ).fetchall():
        id_, kind, config_hash, gpu_name, profile_kind, identity, *grid, created_at = row
        conn.execute(
            f"""
            INSERT INTO {CONFIG_TABLE}
                (id, config_key, kind, config_hash, gpu_name, profile_kind, identity,
                 cache_coords, grid_axes, cells, infeasible, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                id_,
                storage.config_key(kind, config_hash, gpu_name),
                kind,
                config_hash,
                gpu_name,
                profile_kind,
                storage.pack_identity(conn, json.loads(identity)),
                *grid,
                storage.epoch(created_at),
            ),
        )
    for id_, source_hash, source, created_at in conn.execute(
        f"SELECT id, source_hash, source, created_at FROM _v2{SOURCE_TABLE} ORDER BY id"
    ).fetchall():
        conn.execute(
            f"""
            INSERT INTO {SOURCE_TABLE} (id, source_key, source_hash, source, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (id_, storage.source_key(source_hash), source_hash, source, storage.epoch(created_at)),
        )
    for id_, kind, config_hash, gpu_name, source_hash, pool, role, created_at in conn.execute(
        f"""
        SELECT id, kind, config_hash, gpu_name, source_hash, pool, role, created_at
        FROM _v2{USE_TABLE} ORDER BY id
        """
    ).fetchall():
        conn.execute(
            f"""
            INSERT INTO {USE_TABLE} (id, config_key, source_key, pool, role_key, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                id_,
                storage.config_key(kind, config_hash, gpu_name),
                storage.source_key(source_hash),
                pool,
                storage.ensure_role(conn, role),
                storage.epoch(created_at),
            ),
        )
    for table in (CONFIG_TABLE, SOURCE_TABLE, USE_TABLE):
        conn.execute(f"DROP TABLE _v2{table}")


# ── v3 → v4 ──────────────────────────────────────────────────────────────────


def _drop_identity_paths(conn: sqlite3.Connection) -> bool:
    """Take the payload path out of every corpus-routed config's identity.

    A token corpus binding named its payload by the absolute path the building
    machine resolved (``expert_demand.corpus.data_file``), so the config hash --
    a hash of the identity -- differed from machine to machine for the same
    corpus. The checksum and dimensions name the bytes; the simulator now leaves
    the path out of an identity, and this rewrites the configs stored before:
    the identity, its hash and key, and the uses that reference the key. Rows
    are keyed by their args, not by a config, so none moves.
    """
    if CONFIG_TABLE not in _user_tables(conn):
        return False
    blobs = storage.load_blobs(conn)
    rows = conn.execute(
        f"SELECT id, config_key, kind, gpu_name, identity FROM {CONFIG_TABLE} ORDER BY id"
    ).fetchall()
    changed = False
    for id_, old_key, kind, gpu_name, text in rows:
        identity = storage.unpack_identity(text, blobs)
        if not _strip_corpus_paths(identity):
            continue
        changed = True
        config_hash = hashlib.sha256(storage.canonical_json(identity).encode()).hexdigest()
        new_key = storage.config_key(kind, config_hash, gpu_name)
        grid = "cache_coords, grid_axes, cells, infeasible"
        twin = conn.execute(
            f"SELECT id, {grid} FROM {CONFIG_TABLE} WHERE config_key = ?", (new_key,)
        ).fetchone()
        if twin is not None:
            # The same config registered from two machines: one record.
            mine = conn.execute(
                f"SELECT {grid} FROM {CONFIG_TABLE} WHERE id = ?", (id_,)
            ).fetchone()
            if tuple(twin[1:]) != tuple(mine):
                raise ValueError(
                    f"{kind} config {config_hash} on {gpu_name} was registered with two grids"
                )
            conn.execute(f"DELETE FROM {CONFIG_TABLE} WHERE id = ?", (id_,))
        else:
            conn.execute(
                f"""
                UPDATE {CONFIG_TABLE} SET config_key = ?, config_hash = ?, identity = ?
                WHERE id = ?
                """,
                (new_key, config_hash, storage.pack_identity(conn, identity), id_),
            )
        conn.execute(
            f"UPDATE OR IGNORE {USE_TABLE} SET config_key = ? WHERE config_key = ?",
            (new_key, old_key),
        )
        conn.execute(f"DELETE FROM {USE_TABLE} WHERE config_key = ?", (old_key,))
    if changed:
        storage.prune_blobs(conn)
    return changed


def _strip_corpus_paths(value: object) -> bool:
    """Drop ``data_file`` from every token-corpus binding inside ``value``, in
    place; whether one was dropped."""
    dropped = False
    if isinstance(value, dict):
        corpus = value.get("corpus")
        if isinstance(corpus, dict) and "data_file" in corpus:
            del corpus["data_file"]
            dropped = True
        for item in value.values():
            dropped = _strip_corpus_paths(item) or dropped
    elif isinstance(value, list):
        for item in value:
            dropped = _strip_corpus_paths(item) or dropped
    return dropped
