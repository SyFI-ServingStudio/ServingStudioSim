"""How profile.db stores rows compactly (schema v3).

profile.db is tracked in git, so every byte of it is paid again in each
committed version. Three things made v2 large, and v3 stores each once:

- **Row keys.** A kind table's unique key was ``(gpu_name, backend, *args)``.
  SQLite keeps a copy of the whole key in its unique index, and list-valued
  args (per-expert batches, row positions) run to kilobytes, so the index was
  larger than the table. v3 keys a row by ``(gpu_name, backend, args_hash)``;
  the args stay in their columns. :func:`args_hash` hashes the args *as SQLite
  stores them* (see :func:`stored_value`), so a hash from a Python args value
  and one from a stored row agree.
- **Provenance.** Every row repeated its profiler git hash, CUDA, driver and
  backend versions: 172 distinct tuples over 113k rows. v3 keeps each tuple
  once in ``_profile_run`` and a row holds its ``run_key``. Timestamps are
  epoch seconds (v2 stored second-precision UTC text).
- **Registry strings.** ``_kernel_config_use`` repeated three 64-character
  hashes and a dotted role path per row; v3 references configs, sources and
  roles by content key. Large arrays inside a config's identity (an MoE
  routing's per-layer popularity) are shared between configs: v3 keeps each
  once in ``_kernel_config_blob`` and the identity names it by
  ``{"$blob": "<key>"}``.

Every reference is a *content key* (:func:`content_key`, a hash of what it
names), never a surrogate ``id``. Two databases that store the same run, role
or blob therefore give it the same key, and ``profiling.db.merge`` merges them
by their unique keys without remapping references.

Readers never see keys or epoch seconds: :func:`logical_select` and the
registry readers in ``profiling.db.kernel_config`` return the v2 columns.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

RUN_TABLE = "_profile_run"
CONFIG_TABLE = "_kernel_config"
SOURCE_TABLE = "_kernel_config_source"
ROLE_TABLE = "_kernel_config_role"
USE_TABLE = "_kernel_config_use"
BLOB_TABLE = "_kernel_config_blob"

# The provenance a run tuple holds, in `_profile_run` column order.
RUN_COLUMNS = ("profiler_git_hash", "cuda_version", "driver_version", "backend_version")
# A kind table's columns that are not args, in v2 (logical) naming.
NON_ARG_COLUMNS = frozenset(
    {
        "id",
        "gpu_name",
        "backend",
        "args_hash",
        "run_key",
        *RUN_COLUMNS,
        "profiler_run_at",
        "verified",
        "time_ms",
        "tflops",
        "memory_bandwidth_gbps",
        "algbw_gbps",
        "busbw_gbps",
        "energy_j",
        "is_outlier",
        "retry_count",
        "outlier_reason",
        "created_at",
    }
)
# An identity array whose canonical JSON reaches this many bytes is stored once
# in `_kernel_config_blob`. Below it a reference would not be much smaller.
BLOB_MIN_BYTES = 1024
BLOB_REF = "$blob"

# The v2 text forms, which readers still receive.
RUN_AT_FORMAT = "%Y-%m-%dT%H:%M:%S+00:00"
CREATED_AT_FORMAT = "%Y-%m-%d %H:%M:%S"


def canonical_json(value: Any) -> str:
    """The one JSON text a value hashes as: sorted keys, no whitespace."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def content_key(*parts: Any) -> int:
    """A signed 64-bit key for ``parts``: what a reference to them stores."""
    digest = hashlib.sha256(canonical_json(list(parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=True)


def stored_value(value: Any, declared_type: str) -> Any:
    """``value`` as SQLite stores it in a column of ``declared_type``.

    SQLite converts a bound value to the column's affinity: a bool arg in a
    TEXT column is stored as ``'1'``, an int in a REAL column as a float. The
    args hash must see the stored form, or a Python value and the row it was
    stored as would hash differently.
    """
    affinity = declared_type.upper()
    if affinity == "INTEGER":
        if isinstance(value, bool | int):
            return int(value)
        if isinstance(value, float) and value.is_integer():
            return int(value)
    elif affinity == "REAL":
        if isinstance(value, bool | int | float):
            return float(value)
    elif affinity == "TEXT":
        if isinstance(value, str):
            return value
        if isinstance(value, bool | int):
            return str(int(value))
    raise TypeError(f"cannot key {value!r} stored in a {declared_type or 'untyped'} column")


def args_hash(values: Mapping[str, Any], declared_types: Mapping[str, str]) -> bytes:
    """The 8-byte key of one row's args (``{column: value}``, every args column).

    Within one ``(gpu_name, backend)`` a table holds at most ~10^4 rows, so a
    64-bit hash collides with probability ~10^-11; readers still compare the
    args columns after the index lookup.
    """
    stored = {
        column: stored_value(value, declared_types[column]) for column, value in values.items()
    }
    return hashlib.sha256(canonical_json(stored).encode()).digest()[:8]


def epoch(timestamp: str) -> int:
    """Epoch seconds of a v2 timestamp (``RUN_AT_FORMAT`` or ``CREATED_AT_FORMAT``)."""
    parsed = datetime.fromisoformat(timestamp)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    if parsed.microsecond:
        raise ValueError(f"timestamp {timestamp!r} is finer than the second epoch storage keeps")
    return int(parsed.timestamp())


# ── provenance runs ──────────────────────────────────────────────────────────

RUN_SCHEMA = f"""
    CREATE TABLE IF NOT EXISTS {RUN_TABLE} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_key INTEGER NOT NULL,
        profiler_git_hash TEXT NOT NULL,
        cuda_version TEXT,
        driver_version TEXT,
        backend_version TEXT,
        UNIQUE(run_key)
    )
"""


def run_key(conn: sqlite3.Connection, provenance: tuple[Any, ...]) -> int:
    """The key of a ``RUN_COLUMNS`` tuple, stored in ``_profile_run`` if new."""
    key = content_key(*provenance)
    stored = conn.execute(
        f"SELECT {', '.join(RUN_COLUMNS)} FROM {RUN_TABLE} WHERE run_key = ?", (key,)
    ).fetchone()
    if stored is None:
        conn.execute(
            f"INSERT INTO {RUN_TABLE} (run_key, {', '.join(RUN_COLUMNS)}) VALUES (?, ?, ?, ?, ?)",
            (key, *provenance),
        )
    elif tuple(stored) != tuple(provenance):
        raise ValueError(f"run key {key} names both {tuple(stored)} and {provenance}")
    return key


# ── kind tables ──────────────────────────────────────────────────────────────


def kind_table_schema(name: str, arg_defs: list[str], metric_defs: list[str]) -> str:
    """``CREATE TABLE`` of a kind table; ``*_defs`` are ``"column TYPE ..."``."""
    body = ",\n        ".join(
        [
            "id INTEGER PRIMARY KEY AUTOINCREMENT",
            "gpu_name TEXT NOT NULL",
            "backend TEXT NOT NULL",
            *arg_defs,
            "args_hash BLOB NOT NULL",
            "run_key INTEGER NOT NULL",
            "profiler_run_at INTEGER NOT NULL",
            "verified INTEGER NOT NULL DEFAULT 0",
            *metric_defs,
            "is_outlier INTEGER NOT NULL DEFAULT 0",
            "retry_count INTEGER NOT NULL DEFAULT 0",
            "outlier_reason TEXT",
            "created_at INTEGER NOT NULL DEFAULT (unixepoch())",
            "UNIQUE(gpu_name, backend, args_hash)",
        ]
    )
    return f"CREATE TABLE IF NOT EXISTS {quote(name)} (\n        {body}\n    )"


def logical_select(conn: sqlite3.Connection, table: str, alias: str = "t") -> str:
    """``SELECT <v2 columns> FROM table alias JOIN _profile_run``: a kind table's
    rows as v2 stored them (provenance text, text timestamps, no ``args_hash``).
    Append ``WHERE`` / ``ORDER BY`` clauses over ``alias``'s columns."""
    columns = [str(row[1]) for row in conn.execute(f"PRAGMA table_info({quote(table)})")]
    select: list[str] = []
    for column in columns:
        if column == "args_hash":
            continue
        if column == "run_key":
            select.extend(f"r.{name} AS {name}" for name in RUN_COLUMNS)
        elif column == "profiler_run_at":
            select.append(
                f"{iso_sql(f'{alias}.profiler_run_at', RUN_AT_FORMAT)} AS profiler_run_at"
            )
        elif column == "created_at":
            select.append(f"{iso_sql(f'{alias}.created_at', CREATED_AT_FORMAT)} AS created_at")
        else:
            select.append(f"{alias}.{quote(column)} AS {quote(column)}")
    return (
        f"SELECT {', '.join(select)} FROM {quote(table)} {alias} "
        f"JOIN {RUN_TABLE} r ON r.run_key = {alias}.run_key"
    )


def iso_sql(expression: str, fmt: str) -> str:
    """SQL rendering epoch-seconds ``expression`` in a v2 text format."""
    return f"strftime('{fmt}', {expression}, 'unixepoch')"


# ── kernel-config registry ───────────────────────────────────────────────────

REGISTRY_SCHEMA = (
    f"""
    CREATE TABLE IF NOT EXISTS {CONFIG_TABLE} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        config_key INTEGER NOT NULL,
        kind TEXT NOT NULL,
        config_hash TEXT NOT NULL,
        gpu_name TEXT NOT NULL,
        profile_kind TEXT NOT NULL,
        identity TEXT NOT NULL,
        cache_coords TEXT NOT NULL,
        grid_axes TEXT NOT NULL,
        cells BLOB NOT NULL,
        infeasible TEXT NOT NULL,
        created_at INTEGER NOT NULL DEFAULT (unixepoch()),
        UNIQUE(kind, config_hash, gpu_name),
        UNIQUE(config_key)
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {SOURCE_TABLE} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_key INTEGER NOT NULL,
        source_hash TEXT NOT NULL,
        source TEXT NOT NULL,
        created_at INTEGER NOT NULL DEFAULT (unixepoch()),
        UNIQUE(source_hash),
        UNIQUE(source_key)
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {ROLE_TABLE} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role_key INTEGER NOT NULL,
        role TEXT NOT NULL,
        UNIQUE(role_key)
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {USE_TABLE} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        config_key INTEGER NOT NULL,
        source_key INTEGER NOT NULL,
        pool TEXT NOT NULL,
        role_key INTEGER NOT NULL,
        created_at INTEGER NOT NULL DEFAULT (unixepoch()),
        UNIQUE(config_key, source_key, pool, role_key)
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {BLOB_TABLE} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        blob_key INTEGER NOT NULL,
        body TEXT NOT NULL,
        UNIQUE(blob_key)
    )
    """,
)
# Semantic key of each registry and lookup table, for `profiling.db.merge`.
LOOKUP_KEYS = {
    RUN_TABLE: ("run_key",),
    CONFIG_TABLE: ("kind", "config_hash", "gpu_name"),
    SOURCE_TABLE: ("source_hash",),
    ROLE_TABLE: ("role_key",),
    USE_TABLE: ("config_key", "source_key", "pool", "role_key"),
    BLOB_TABLE: ("blob_key",),
}


def config_key(kind: str, config_hash: str, gpu_name: str) -> int:
    return content_key(kind, config_hash, gpu_name)


def source_key(source_hash: str) -> int:
    return content_key(source_hash)


def role_key(role: str) -> int:
    return content_key(role)


def ensure_role(conn: sqlite3.Connection, role: str) -> int:
    key = role_key(role)
    stored = conn.execute(f"SELECT role FROM {ROLE_TABLE} WHERE role_key = ?", (key,)).fetchone()
    if stored is None:
        conn.execute(f"INSERT INTO {ROLE_TABLE} (role_key, role) VALUES (?, ?)", (key, role))
    elif stored[0] != role:
        raise ValueError(f"role key {key} names both {stored[0]!r} and {role!r}")
    return key


def pack_identity(conn: sqlite3.Connection, identity: Any) -> str:
    """``identity``'s stored text: each large array moved to ``_kernel_config_blob``."""

    def pack(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: pack(item) for key, item in value.items()}
        if isinstance(value, list):
            body = canonical_json(value)
            if len(body) >= BLOB_MIN_BYTES:
                return {BLOB_REF: _store_blob(conn, body)}
        return value

    return canonical_json(pack(identity))


def unpack_identity(text: str, blobs: Mapping[str, Any]) -> Any:
    """Inverse of :func:`pack_identity`; ``blobs`` is :func:`load_blobs`."""

    def unpack(value: Any) -> Any:
        if isinstance(value, dict):
            if set(value) == {BLOB_REF}:
                return blobs[value[BLOB_REF]]
            return {key: unpack(item) for key, item in value.items()}
        return value

    return unpack(json.loads(text))


def load_blobs(conn: sqlite3.Connection) -> dict[str, Any]:
    """Every stored blob, by the reference an identity names it with."""
    return {
        _blob_ref(key): json.loads(body)
        for key, body in conn.execute(f"SELECT blob_key, body FROM {BLOB_TABLE}")
    }


def prune_blobs(conn: sqlite3.Connection) -> None:
    """Drop blobs no stored identity references (after configs are deleted)."""
    referenced: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if set(value) == {BLOB_REF}:
                referenced.add(value[BLOB_REF])
            for item in value.values():
                walk(item)

    for (text,) in conn.execute(f"SELECT identity FROM {CONFIG_TABLE}"):
        walk(json.loads(text))
    stale = [
        (key,)
        for (key,) in conn.execute(f"SELECT blob_key FROM {BLOB_TABLE}")
        if _blob_ref(key) not in referenced
    ]
    conn.executemany(f"DELETE FROM {BLOB_TABLE} WHERE blob_key = ?", stale)


def _store_blob(conn: sqlite3.Connection, body: str) -> str:
    key = content_key(body)
    stored = conn.execute(f"SELECT body FROM {BLOB_TABLE} WHERE blob_key = ?", (key,)).fetchone()
    if stored is None:
        conn.execute(f"INSERT INTO {BLOB_TABLE} (blob_key, body) VALUES (?, ?)", (key, body))
    elif stored[0] != body:
        raise ValueError(f"blob key {key} names two different arrays")
    return _blob_ref(key)


def _blob_ref(key: int) -> str:
    return f"{key & 0xFFFFFFFFFFFFFFFF:016x}"


def quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'
