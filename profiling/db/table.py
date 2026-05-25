"""SQLite-backed L1b table abstraction."""

from __future__ import annotations

import json
import sqlite3
import subprocess
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from enum import Enum
from functools import cache, lru_cache
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, get_origin, get_type_hints

from profiling.db.args import KernelArgs
from profiling.db.kind import KernelKind
from profiling.db.migrate import SCHEMA_HASH, migrate
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
        self._ensure_schema()
        metric_columns = self._metric_columns()
        self._validate_row_metric_family(rows)
        columns = [
            "gpu_name",
            "backend",
            *self.args_columns,
            *STANDARD_COLUMNS[2:],
            *metric_columns,
        ]
        placeholders = ", ".join("?" for _ in columns)
        # Replacement policy: same (gpu_name, backend, args) overwrites the
        # measurement/provenance columns and marks the row as freshly
        # verified/non-outlier. Key columns are not updated.
        replace_columns = [*STANDARD_COLUMNS[2:], *metric_columns]
        updates = ", ".join(
            f"{column}=excluded.{column}" for column in replace_columns
        )
        sql = f"""
            INSERT INTO {self.name} ({", ".join(columns)})
            VALUES ({placeholders})
            ON CONFLICT(gpu_name, backend, {", ".join(self.args_columns)})
            DO UPDATE SET {updates}, is_outlier=0, retry_count=0, outlier_reason=NULL
        """
        with self._connect() as conn:
            conn.executemany(sql, [self._row_values(row, metric_columns) for row in rows])

    def query(
        self,
        args_list: list[KernelArgs],
        *,
        backend: str | None = None,
        gpu_name: str,
        include_outliers: bool = False,
    ) -> list[Metrics | MissingEntry]:
        self._ensure_schema()
        backend = backend or self.profiler_spec.backend
        with self._connect() as conn:
            return [
                self._query_one(conn, args, backend, gpu_name, include_outliers)
                for args in args_list
            ]

    def exists(
        self,
        args_list: list[KernelArgs],
        *,
        backend: str | None = None,
        gpu_name: str,
    ) -> list[bool]:
        self._ensure_schema()
        backend = backend or self.profiler_spec.backend
        with self._connect() as conn:
            return [self._exists_one(conn, args, backend, gpu_name) for args in args_list]

    def metadata(self) -> TableMetadata:
        self._ensure_schema()
        with self._connect() as conn:
            row = conn.execute(f"SELECT COUNT(*) AS n FROM {self.name}").fetchone()
            hashes = conn.execute(
                f"""
                SELECT DISTINCT profiler_git_hash
                FROM {self.name}
                WHERE profiler_git_hash IS NOT NULL AND profiler_git_hash != ''
                ORDER BY profiler_git_hash
                """
            ).fetchall()
        return TableMetadata(
            name=self.name,
            row_count=int(row["n"]),
            schema_hash=SCHEMA_HASH,
            profiler_git_hashes=tuple(str(hash_row["profiler_git_hash"]) for hash_row in hashes),
        )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        migrate(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        type_hints = get_type_hints(self.profiler_spec.args_schema)
        arg_defs = ",\n                ".join(
            f"{field.name} {_sqlite_type(type_hints[field.name])} NOT NULL"
            for field in fields(self.profiler_spec.args_schema)
        )
        metric_defs = ",\n                    ".join(
            f"{column} {column_def}"
            for column, column_def in self._metric_create_defs().items()
        )
        unique_args = ", ".join(self.args_columns)
        with self._connect() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gpu_name TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    {arg_defs},
                    profiler_git_hash TEXT NOT NULL,
                    profiler_run_at TEXT NOT NULL,
                    cuda_version TEXT,
                    driver_version TEXT,
                    backend_version TEXT,
                    verified INTEGER NOT NULL DEFAULT 0,
                    {metric_defs},
                    is_outlier INTEGER NOT NULL DEFAULT 0,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    outlier_reason TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(gpu_name, backend, {unique_args})
                )
                """
            )
            self._ensure_standard_columns(conn)
            self._ensure_metric_columns(conn)

    def _ensure_standard_columns(self, conn: sqlite3.Connection) -> None:
        existing = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({self.name})").fetchall()
        }
        alter_defs = {
            "profiler_git_hash": "TEXT DEFAULT 'unknown'",
            "profiler_run_at": "TEXT",
            "cuda_version": "TEXT",
            "driver_version": "TEXT",
            "backend_version": "TEXT",
            "verified": "INTEGER NOT NULL DEFAULT 0",
        }
        for column, column_def in alter_defs.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE {self.name} ADD COLUMN {column} {column_def}")

    def _ensure_metric_columns(self, conn: sqlite3.Connection) -> None:
        existing = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({self.name})").fetchall()
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

    def _query_one(
        self,
        conn: sqlite3.Connection,
        args: KernelArgs,
        backend: str,
        gpu_name: str,
        include_outliers: bool,
    ) -> Metrics | MissingEntry:
        where, values = self._where(args, backend, gpu_name)
        if not include_outliers:
            where += " AND is_outlier = 0"
        row = conn.execute(f"SELECT * FROM {self.name} WHERE {where} LIMIT 1", values).fetchone()
        if row is None:
            return MissingEntry(self.profiler_spec.kernel_kind, backend, gpu_name, args)
        return _metrics_from_row(row, self.profiler_spec.metric_family)

    def _exists_one(
        self,
        conn: sqlite3.Connection,
        args: KernelArgs,
        backend: str,
        gpu_name: str,
    ) -> bool:
        where, values = self._where(args, backend, gpu_name)
        row = conn.execute(
            f"SELECT 1 FROM {self.name} WHERE {where} AND is_outlier = 0 LIMIT 1",
            values,
        ).fetchone()
        return row is not None

    def _where(self, args: KernelArgs, backend: str, gpu_name: str) -> tuple[str, list[Any]]:
        arg_values = _args_to_db(args)
        where = ["gpu_name = ?", "backend = ?", *(f"{column} = ?" for column in self.args_columns)]
        values = [gpu_name, backend, *(arg_values[column] for column in self.args_columns)]
        return " AND ".join(where), values

    def _row_values(self, row: ProfileRow, metric_columns: list[str]) -> list[Any]:
        arg_values = _args_to_db(row.args)
        metric_values = _metrics_to_db(row.metrics)
        return [
            row.gpu_name,
            row.backend,
            *(arg_values[column] for column in self.args_columns),
            row.profiler_git_hash or _current_git_hash(),
            row.profiler_run_at or _utc_now(),
            row.cuda_version or _cuda_version(),
            row.driver_version or _driver_version(),
            row.backend_version or _backend_version(row.backend),
            int(row.verified),
            *(metric_values[column] for column in metric_columns),
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


def _args_to_db(args: KernelArgs) -> dict[str, Any]:
    return {key: _to_db_value(value) for key, value in asdict(args).items()}


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


@lru_cache(maxsize=1)
def _current_git_hash() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


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
