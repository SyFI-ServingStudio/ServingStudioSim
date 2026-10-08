"""SQLite-backed L1b table abstraction."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from enum import Enum
from functools import cache, lru_cache
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, get_origin

from profiling.db import storage
from profiling.db.args import KernelArgs, field_types
from profiling.db.kind import KernelKind
from profiling.db.migrate import SCHEMA_HASH, migrate_connection, require_current
from profiling.db.registry import KernelProfilerSpec, MetricFamily
from profiling.runners.metrics import CommMetrics, ComputeMetrics, Metrics

STANDARD_COLUMNS = [
    "gpu_name",
    "backend",
    "profiler_git_hash",
    "profiler_run_at",
    "cuda_version",
    "driver_version",
    "backend_version",
    "verified",
]
# How long a writer waits for the write lock before giving up. See
# `Table._write_transaction`.
_WRITE_LOCK_TIMEOUT_S = 120.0
# Bound parameters per lookup statement; SQLite allows 32766. See `Table._match_rows`.
_SQL_VARIABLES = 30000
# Each thread's last read-only connection. See `Table._connect_read_only`.
_read_only = threading.local()

COMPUTE_METRIC_COLUMNS = ["time_ms", "tflops", "memory_bandwidth_gbps", "energy_j"]
# message_size for comm kernels is an args/cache-key column, NOT a measured
# result — the simulator derives moved bytes from busbw × time. So it is not a
# metric column here (it lives in the args region of the table).
COMM_METRIC_COLUMNS = ["time_ms", "algbw_gbps", "busbw_gbps", "energy_j"]
# Metric columns are a table-level contract, not inferred from each row. Keep
# this in lockstep with KernelProfilerSpec.metric_family and L1 design §3.2.2.
METRIC_COLUMNS_BY_FAMILY = {
    MetricFamily.COMPUTE: COMPUTE_METRIC_COLUMNS,
    MetricFamily.COMM: COMM_METRIC_COLUMNS,
}
ALL_METRIC_COLUMNS = [
    "time_ms",
    "tflops",
    "memory_bandwidth_gbps",
    "algbw_gbps",
    "busbw_gbps",
    "energy_j",
]
METRIC_CREATE_DEFS_BY_FAMILY = {
    MetricFamily.COMPUTE: {
        "time_ms": "REAL NOT NULL",
        "tflops": "REAL NOT NULL",
        "memory_bandwidth_gbps": "REAL NOT NULL",
        "energy_j": "REAL NOT NULL DEFAULT 0.0",
    },
    MetricFamily.COMM: {
        "time_ms": "REAL NOT NULL",
        "algbw_gbps": "REAL NOT NULL",
        "busbw_gbps": "REAL NOT NULL",
        "energy_j": "REAL NOT NULL DEFAULT 0.0",
    },
}
METRIC_ALTER_DEFS_BY_FAMILY = {
    MetricFamily.COMPUTE: {
        "time_ms": "REAL",
        "tflops": "REAL",
        "memory_bandwidth_gbps": "REAL",
        "energy_j": "REAL NOT NULL DEFAULT 0.0",
    },
    MetricFamily.COMM: {
        "time_ms": "REAL",
        "algbw_gbps": "REAL",
        "busbw_gbps": "REAL",
        "energy_j": "REAL NOT NULL DEFAULT 0.0",
    },
}


@dataclass(frozen=True)
class MissingEntry:
    kernel_kind: KernelKind
    backend: str
    gpu_name: str
    args: KernelArgs


@dataclass(frozen=True)
class ProfileRow:
    args: KernelArgs
    metrics: Metrics
    gpu_name: str
    backend: str
    profiler_git_hash: str | None = None
    profiler_run_at: str | None = None
    cuda_version: str | None = None
    driver_version: str | None = None
    backend_version: str | None = None
    verified: bool = False


@dataclass(frozen=True)
class TableMetadata:
    name: str
    row_count: int
    schema_hash: str
    profiler_git_hashes: tuple[str, ...]


class Table:
    """One profiling table for one args schema.

    The table owns SQL and schema reflection. Runners only return ``Metrics``.
    """

    def __init__(self, profiler_spec: KernelProfilerSpec, db_path: Path | str):
        self.profiler_spec = profiler_spec
        self.name = profiler_spec.table_name
        self.db_path = Path(db_path)
        self.args_columns = [field.name for field in fields(profiler_spec.args_schema)]

    def insert(self, rows: list[ProfileRow]) -> None:
        if not rows:
            return
        metric_columns = self._metric_columns()
        self._validate_row_metric_family(rows)
        columns = [
            "gpu_name",
            "backend",
            *self.args_columns,
            "args_hash",
            "run_key",
            "profiler_run_at",
            "verified",
            *metric_columns,
            "created_at",
        ]
        placeholders = ", ".join("?" for _ in columns)
        # Replacement policy: same (gpu_name, backend, args) overwrites the
        # measurement/provenance columns and marks the row as freshly
        # verified/non-outlier. Key columns are not updated.
        replace_columns = ["run_key", "profiler_run_at", "verified", *metric_columns]
        updates = ", ".join(f"{column}=excluded.{column}" for column in replace_columns)
        sql = f"""
            INSERT INTO {self.name} ({", ".join(columns)})
            VALUES ({placeholders})
            ON CONFLICT(gpu_name, backend, args_hash)
            DO UPDATE SET {updates}, is_outlier=0, retry_count=0, outlier_reason=NULL
        """
        default_git_hash = (
            _current_git_hash() if any(not row.profiler_git_hash for row in rows) else ""
        )
        with self._write_transaction() as conn:
            conn.executemany(
                sql,
                [self._row_values(conn, row, metric_columns, default_git_hash) for row in rows],
            )

    def query(
        self,
        args_list: list[KernelArgs],
        *,
        backend: str | None = None,
        gpu_name: str,
        include_outliers: bool = False,
    ) -> list[Metrics | MissingEntry]:
        backend = backend or self.profiler_spec.backend
        conn = self._connect_read_only()
        if conn is None:
            return [self._missing_entry(args, backend, gpu_name) for args in args_list]
        with conn:
            if not self._table_exists(conn):
                return [self._missing_entry(args, backend, gpu_name) for args in args_list]
            rows = self._match_rows(
                conn, args_list, backend, gpu_name, include_outliers, select="t.*"
            )
        return [
            self._missing_entry(args, backend, gpu_name)
            if row is None
            else _metrics_from_row(row, self.profiler_spec.metric_family)
            for args, row in zip(args_list, rows, strict=True)
        ]

    def rows_for(
        self,
        args_list: list[KernelArgs],
        *,
        backend: str,
        gpu_name: str,
    ) -> list[dict[str, Any] | None]:
        """The stored row of each args (outliers included), or None. Read-only."""
        conn = self._connect_read_only()
        if conn is None:
            return [None for _ in args_list]
        with conn:
            if not self._table_exists(conn):
                return [None for _ in args_list]
            select = storage.logical_select(conn, self.name)
            out: list[dict[str, Any] | None] = []
            for args in args_list:
                where, values = self._where(args, backend, gpu_name)
                row = conn.execute(f"{select} WHERE {where} LIMIT 1", values).fetchone()
                out.append(dict(row) if row is not None else None)
            return out

    def db_key(self, args: KernelArgs) -> tuple[Any, ...]:
        """``args`` as this table stores them, in ``args_columns`` order."""
        return tuple(_to_db_value(getattr(args, column)) for column in self.args_columns)

    def args_hash(self, key: tuple[Any, ...]) -> bytes:
        """The ``args_hash`` column of a row whose args are ``key`` (:meth:`db_key`)."""
        return storage.args_hash(
            dict(zip(self.args_columns, key, strict=True)),
            _declared_types(self.profiler_spec.args_schema),
        )

    def args_match_sql(self, alias: str = "") -> str:
        """``args_hash = ? AND <each args column> = ?`` over ``alias``'s columns;
        bind :meth:`args_hash` then the :meth:`db_key` values. The hash finds the
        row through the unique index; the columns confirm it."""
        prefix = f"{alias}." if alias else ""
        return " AND ".join(
            [f"{prefix}args_hash = ?", *(f"{prefix}{column} = ?" for column in self.args_columns)]
        )

    def exists(
        self,
        args_list: list[KernelArgs],
        *,
        backend: str | None = None,
        gpu_name: str,
    ) -> list[bool]:
        backend = backend or self.profiler_spec.backend
        conn = self._connect_read_only()
        if conn is None:
            return [False for _ in args_list]
        with conn:
            if not self._table_exists(conn):
                return [False for _ in args_list]
            rows = self._match_rows(conn, args_list, backend, gpu_name, include_outliers=False)
        return [row is not None for row in rows]

    def metadata(self) -> TableMetadata:
        conn = self._connect_read_only()
        if conn is None:
            return self._empty_metadata()
        with conn:
            if not self._table_exists(conn):
                return self._empty_metadata()
            row = conn.execute(f"SELECT COUNT(*) AS n FROM {self.name}").fetchone()
            hashes = conn.execute(
                f"""
                SELECT DISTINCT r.profiler_git_hash
                FROM {self.name} t JOIN {storage.RUN_TABLE} r ON r.run_key = t.run_key
                WHERE r.profiler_git_hash != ''
                ORDER BY r.profiler_git_hash
                """
            ).fetchall()
        return TableMetadata(
            name=self.name,
            row_count=int(row["n"]),
            schema_hash=SCHEMA_HASH,
            profiler_git_hashes=tuple(str(hash_row["profiler_git_hash"]) for hash_row in hashes),
        )

    @contextmanager
    def _write_transaction(self) -> Iterator[sqlite3.Connection]:
        """Own schema preparation and persistence as one write transaction.

        The timeout is not the default 5 s because writers are now concurrent:
        ``profiling.plan.issue`` measures one unit per GPU on its own thread and
        each finishes with an insert, so a 1400-row insert can be holding the
        write lock while another unit's ``migrate_connection`` DDL arrives. Five
        seconds of that and the second unit raises `database is locked`, which
        discards GPU time already spent. Waiting is always cheaper than
        remeasuring.
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path, timeout=_WRITE_LOCK_TIMEOUT_S) as conn:
            conn.row_factory = sqlite3.Row
            migrate_connection(conn)
            self._ensure_schema(conn)
            yield conn

    def _connect_read_only(self) -> sqlite3.Connection | None:
        """A physically read-only connection, without creating the DB.

        A build queries once per kernel, and a new connection parses the whole
        schema before its first statement. So each thread reuses its last
        connection while the file is the same one, unchanged on disk; any write,
        replacement or fork opens (and ``require_current`` checks) a new one.
        """
        if not self.db_path.is_file():
            return None
        path = self.db_path.resolve()
        stat = path.stat()
        identity = (path, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, os.getpid())
        cached = getattr(_read_only, "connection", None)
        if cached is not None and cached[0] == identity:
            return cached[1]
        conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON")
        require_current(conn, str(self.db_path))
        _read_only.connection = (identity, conn)
        return conn

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        """Prepare this table inside an already-open write transaction."""
        conn.execute(storage.RUN_SCHEMA)
        conn.execute(
            storage.kind_table_schema(
                self.name,
                [
                    f"{column} {declared} NOT NULL"
                    for column, declared in _declared_types(self.profiler_spec.args_schema).items()
                ],
                [
                    f"{column} {column_def}"
                    for column, column_def in self._metric_create_defs().items()
                ],
            )
        )
        self._ensure_metric_columns(conn)

    def _table_exists(self, conn: sqlite3.Connection) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
            (self.name,),
        ).fetchone()
        return row is not None

    def _empty_metadata(self) -> TableMetadata:
        return TableMetadata(
            name=self.name,
            row_count=0,
            schema_hash=SCHEMA_HASH,
            profiler_git_hashes=(),
        )

    def _missing_entry(
        self,
        args: KernelArgs,
        backend: str,
        gpu_name: str,
    ) -> MissingEntry:
        return MissingEntry(
            kernel_kind=self.profiler_spec.kernel_kind,
            backend=backend,
            gpu_name=gpu_name,
            args=args,
        )

    def _ensure_metric_columns(self, conn: sqlite3.Connection) -> None:
        existing = {
            row["name"] for row in conn.execute(f"PRAGMA table_info({self.name})").fetchall()
        }
        metric_columns = set(self._metric_columns())
        for column, column_def in METRIC_ALTER_DEFS_BY_FAMILY[
            self.profiler_spec.metric_family
        ].items():
            if column not in existing:
                conn.execute(f"ALTER TABLE {self.name} ADD COLUMN {column} {column_def}")
        for column in ALL_METRIC_COLUMNS:
            if column in existing and column not in metric_columns:
                conn.execute(f"ALTER TABLE {self.name} DROP COLUMN {column}")

    def _match_rows(
        self,
        conn: sqlite3.Connection,
        args_list: list[KernelArgs],
        backend: str,
        gpu_name: str,
        include_outliers: bool,
        *,
        select: str = "",
    ) -> list[sqlite3.Row | None]:
        """The row of each args on ``(gpu_name, backend)``, or None.

        One statement per chunk of args rather than one per args: the chunk is a
        ``VALUES`` table ``c`` joined to this table's rows ``t``, and ``select``
        names further ``t`` columns to return. Each args still finds its row
        through the ``(gpu_name, backend, args_hash)`` unique index, and its args
        columns confirm it in SQL, where each column's declared type converts the
        ``VALUES`` value exactly as it converts a bound ``column = ?``.
        """
        columns = ["_i", "_hash", *self.args_columns]
        match = " AND ".join(
            [
                "t.gpu_name = ?",
                "t.backend = ?",
                "t.args_hash = c._hash",
                *(f"t.{column} = c.{column}" for column in self.args_columns),
                *([] if include_outliers else ["t.is_outlier = 0"]),
            ]
        )
        found: list[sqlite3.Row | None] = [None] * len(args_list)
        chunk = max(1, (_SQL_VARIABLES - 2) // len(columns))
        for start in range(0, len(args_list), chunk):
            part = args_list[start : start + chunk]
            values = ", ".join([f"({', '.join('?' * len(columns))})"] * len(part))
            bound: list[Any] = []
            for index, args in enumerate(part, start):
                key = self.db_key(args)
                bound += [index, self.args_hash(key), *key]
            # CROSS JOIN keeps `c` the outer loop, so `t` is probed by its index.
            sql = (
                f"WITH c({', '.join(columns)}) AS (VALUES {values}) "
                f"SELECT {', '.join(['c._i AS _i', *([select] if select else [])])} "
                f"FROM c CROSS JOIN {self.name} t WHERE {match}"
            )
            for row in conn.execute(sql, [*bound, gpu_name, backend]):
                found[row["_i"]] = row
        return found

    def _where(self, args: KernelArgs, backend: str, gpu_name: str) -> tuple[str, list[Any]]:
        """Unqualified columns: the caller's FROM names this table once (the
        logical select aliases it ``t``, whose columns these still resolve to)."""
        key = self.db_key(args)
        where = f"gpu_name = ? AND backend = ? AND {self.args_match_sql()}"
        return where, [gpu_name, backend, self.args_hash(key), *key]

    def _row_values(
        self,
        conn: sqlite3.Connection,
        row: ProfileRow,
        metric_columns: list[str],
        default_git_hash: str,
    ) -> list[Any]:
        key = self.db_key(row.args)
        metric_values = _metrics_to_db(row.metrics)
        provenance = (
            row.profiler_git_hash or default_git_hash,
            row.cuda_version or _cuda_version(),
            row.driver_version or _driver_version(),
            row.backend_version or _backend_version(row.backend),
        )
        return [
            row.gpu_name,
            row.backend,
            *key,
            self.args_hash(key),
            storage.run_key(conn, provenance),
            storage.epoch(row.profiler_run_at or _utc_now()),
            int(row.verified),
            *(metric_values[column] for column in metric_columns),
            storage.now_epoch(),
        ]

    def _metric_columns(self) -> list[str]:
        return METRIC_COLUMNS_BY_FAMILY[self.profiler_spec.metric_family]

    def _metric_create_defs(self) -> dict[str, str]:
        return METRIC_CREATE_DEFS_BY_FAMILY[self.profiler_spec.metric_family]

    def _validate_row_metric_family(self, rows: list[ProfileRow]) -> None:
        if self.profiler_spec.metric_family is MetricFamily.COMPUTE:
            expected_type: type[ComputeMetrics] | type[CommMetrics] = ComputeMetrics
        else:
            expected_type = CommMetrics
        for row in rows:
            if not isinstance(row.metrics, expected_type):
                raise ValueError(
                    f"{self.name} is a {self.profiler_spec.metric_family.value} metrics table; "
                    f"got {type(row.metrics).__name__}"
                )


def _metrics_to_db(metrics: Metrics) -> dict[str, Any]:
    if isinstance(metrics, ComputeMetrics):
        return {
            "time_ms": metrics.time_ms,
            "tflops": metrics.tflops,
            "memory_bandwidth_gbps": metrics.memory_bandwidth_gbps,
            "energy_j": metrics.energy_j,
        }
    if isinstance(metrics, CommMetrics):
        return {
            "time_ms": metrics.time_ms,
            "algbw_gbps": metrics.algbw_gbps,
            "busbw_gbps": metrics.busbw_gbps,
            "energy_j": metrics.energy_j,
        }
    raise TypeError(f"unsupported metrics type {type(metrics).__name__}")


def _metrics_from_row(row: sqlite3.Row, metric_family: MetricFamily) -> Metrics:
    if metric_family is MetricFamily.COMPUTE:
        return ComputeMetrics(
            time_ms=float(row["time_ms"]),
            tflops=float(row["tflops"]),
            memory_bandwidth_gbps=float(row["memory_bandwidth_gbps"]),
            energy_j=float(row["energy_j"] or 0.0),
        )
    return CommMetrics(
        time_ms=float(row["time_ms"]),
        algbw_gbps=float(row["algbw_gbps"]),
        busbw_gbps=float(row["busbw_gbps"]),
        energy_j=float(row["energy_j"] or 0.0),
    )


def _to_db_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return json.dumps(list(value), separators=(",", ":"))
    if isinstance(value, list):
        return json.dumps(value, separators=(",", ":"))
    return value


@cache
def _declared_types(args_schema: type[KernelArgs]) -> dict[str, str]:
    """Each args column's declared SQLite type, in column order."""
    return {name: _sqlite_type(annotation) for name, annotation in field_types(args_schema).items()}


def _sqlite_type(annotation: Any) -> str:
    origin = get_origin(annotation)
    if origin in (tuple, list):
        return "TEXT"
    if annotation is int:
        return "INTEGER"
    if annotation is float:
        return "REAL"
    return "TEXT"


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _current_git_hash() -> str:
    return _git_tree_stamp(Path(__file__).resolve().parents[2])


def _git_tree_stamp(repo_root: Path) -> str:
    """Stamp HEAD, marking measurements made from edited profiling code.

    A bare commit is reproducibility evidence only when the code that can alter
    a profile number is clean. Staged, unstaged, and untracked files under
    ``profiling/`` therefore add ``-dirty``. Unrelated simulator or docs edits
    do not weaken this profiling-specific stamp.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    lines = result.stdout.splitlines()
    if len(lines) != 2 or Path(lines[0]).resolve() != repo_root.resolve():
        return "unknown"
    head = lines[1].strip()
    if not head:
        return "unknown"
    try:
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                "profiling",
                ":(exclude,glob)profiling/**/*.db",
                ":(exclude,glob)profiling/**/*.db-*",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        # HEAD alone cannot prove that the profiling tree was clean.
        return f"{head}-dirty"
    return f"{head}-dirty" if status.stdout.strip() else head


@lru_cache(maxsize=1)
def _cuda_version() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    return getattr(torch.version, "cuda", None)


@lru_cache(maxsize=1)
def _driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    first_line = result.stdout.splitlines()[0].strip() if result.stdout.splitlines() else ""
    return first_line or None


@cache
def _backend_version(backend: str) -> str | None:
    try:
        return importlib_metadata.version(backend)
    except importlib_metadata.PackageNotFoundError:
        return None
