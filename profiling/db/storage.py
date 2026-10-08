"""How profile.db stores rows compactly (schema v3).

profile.db is tracked in git, so every byte of it is paid again in each
committed version. Two things made v2 large, and v3 stores each once:

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

A row references its run by a *content key* (:func:`content_key`, a hash of
what it names), never a surrogate ``id``. Two databases that store the same run
therefore give it the same key, and ``profiling.db.merge`` merges them by their
unique keys without remapping references.

Readers never see keys or epoch seconds: :func:`logical_select` returns the v2
columns.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

RUN_TABLE = "_profile_run"

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


def now_epoch() -> int:
    """Current time as epoch seconds, computed in Python.

    Inserts supply ``created_at`` with this instead of leaning on SQLite's
    ``unixepoch()``: that function arrived in SQLite 3.38 (2022), and some
    runtimes we profile inside (the vLLM-ROCm container) ship an older SQLite,
    where an insert whose ``created_at`` defaulted to ``unixepoch()`` died with
    ``unknown function: unixepoch()``. The stored value is identical to what
    ``unixepoch()`` would have produced: integer UTC epoch seconds.
    """
    return int(time.time())


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
            # created_at is supplied explicitly on every insert (storage.now_epoch);
            # no SQL-side unixepoch() default, so inserts work on pre-3.38 SQLite.
            "created_at INTEGER NOT NULL",
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


# Semantic key of each lookup table, for `profiling.db.merge`.
LOOKUP_KEYS = {RUN_TABLE: ("run_key",)}


def quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'
