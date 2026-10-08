"""profile.db's compact v3 storage: args keys, provenance runs, the lossless
upgrade from v2, and the tables an older checkout wrote that migration drops."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import fields
from pathlib import Path
from typing import get_type_hints

import pytest

from profiling.db import storage
from profiling.db.args import DType
from profiling.db.merge import merge_profile_databases
from profiling.db.migrate import SCHEMA_VERSION, migrate
from profiling.db.registry import KernelProfilerSpec, iter_kernel_profiler_specs
from profiling.db.table import ProfileRow, Table, _sqlite_type, _to_db_value
from profiling.runners.metrics import ComputeMetrics

GPU = "NVIDIA H200"


def _spec(kind: str) -> KernelProfilerSpec:
    return next(iter_kernel_profiler_specs(kind))


def _finalize_args(num_tokens: int, fuse: bool):
    return _spec("moe_finalize_fuse_shared").args_schema(
        num_tokens=num_tokens, top_k=8, hidden_dim=7168, dtype=DType.BF16, fuse_shared_output=fuse
    )


def _moe_args(num_tokens: int):
    schema = _spec("bf16_fused_moe").args_schema
    values = {
        "num_tokens": num_tokens,
        "hidden_size": 3072,
        "intermediate_size": 1536,
        "num_experts": 4,
        "num_local_experts": 4,
        "top_k": 2,
        "dtype": DType.BF16,
        "routing_method": "minimax2",
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_numerator": 1,
        "routed_scaling_denominator": 1,
        "per_expert_batches": (num_tokens, 0, num_tokens, 0),
    }
    return schema(**{f.name: values[f.name] for f in fields(schema)})


# ── v2 fixture: the layout checkouts before schema v3 wrote ─────────────────


def _v2_kind_table(conn: sqlite3.Connection, spec: KernelProfilerSpec) -> None:
    hints = get_type_hints(spec.args_schema)
    args = [f.name for f in fields(spec.args_schema)]
    arg_defs = ", ".join(f"{a} {_sqlite_type(hints[a])} NOT NULL" for a in args)
    conn.execute(
        f"""
        CREATE TABLE {spec.table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gpu_name TEXT NOT NULL, backend TEXT NOT NULL, {arg_defs},
            profiler_git_hash TEXT NOT NULL, profiler_run_at TEXT NOT NULL,
            cuda_version TEXT, driver_version TEXT, backend_version TEXT,
            verified INTEGER NOT NULL DEFAULT 0,
            time_ms REAL NOT NULL, tflops REAL NOT NULL, memory_bandwidth_gbps REAL NOT NULL,
            energy_j REAL NOT NULL DEFAULT 0.0,
            is_outlier INTEGER NOT NULL DEFAULT 0, retry_count INTEGER NOT NULL DEFAULT 0,
            outlier_reason TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(gpu_name, backend, {", ".join(args)})
        )
        """
    )


def _v2_insert(conn: sqlite3.Connection, spec: KernelProfilerSpec, args, prov, n: int) -> None:
    values = {f.name: _to_db_value(getattr(args, f.name)) for f in fields(args)}
    columns = ["gpu_name", "backend", *values, "profiler_git_hash", "profiler_run_at"]
    columns += ["cuda_version", "driver_version", "backend_version", "verified"]
    columns += ["time_ms", "tflops", "memory_bandwidth_gbps", "created_at", "outlier_reason"]
    conn.execute(
        f"INSERT INTO {spec.table_name} ({', '.join(columns)}) VALUES "
        f"({', '.join('?' for _ in columns)})",
        (
            GPU,
            spec.backend,
            *values.values(),
            *prov,
            1,
            0.01 * n,
            float(n),
            2.0,
            f"2026-09-0{n} 01:02:03",
            "slow" if n == 2 else None,
        ),
    )


def _v2_db(path: Path) -> None:
    finalize, moe = _spec("moe_finalize_fuse_shared"), _spec("bf16_fused_moe")
    prov_a = ("abc", "2026-09-01T00:00:00+00:00", "12.9", "580.1", None)
    prov_b = ("def-dirty", "2026-09-02T00:00:07+00:00", None, "580.1", "0.5.1")
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("CREATE TABLE _db_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute("INSERT INTO _db_metadata VALUES ('schema_version', '2')")
        _v2_kind_table(conn, finalize)
        _v2_kind_table(conn, moe)
        _v2_insert(conn, finalize, _finalize_args(1, True), prov_a, 1)
        _v2_insert(conn, finalize, _finalize_args(1, False), prov_b, 2)
        _v2_insert(conn, moe, _moe_args(2), prov_a, 3)
        _retired_registry(conn)


def _retired_registry(conn: sqlite3.Connection) -> None:
    """The kernel-config registry tables an older checkout wrote, with a row each."""
    for table in (
        "_kernel_config",
        "_kernel_config_source",
        "_kernel_config_role",
        "_kernel_config_use",
        "_kernel_config_blob",
    ):
        conn.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, body TEXT NOT NULL)")
        conn.execute(f"INSERT INTO {table} (body) VALUES ('x')")


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def _rows(conn: sqlite3.Connection, table: str, sql: str) -> list[dict]:
    conn.row_factory = sqlite3.Row
    return [dict(row) for row in conn.execute(f"{sql} ORDER BY id")]


# ── tests ────────────────────────────────────────────────────────────────────


def test_upgrade_keeps_every_row_as_v2_stored_it(tmp_path: Path) -> None:
    db = tmp_path / "profile.db"
    _v2_db(db)
    with closing(sqlite3.connect(db)) as conn:
        before = {
            t: _rows(conn, t, f"SELECT * FROM {t}")
            for t in ("moe_finalize_fuse_shared", "bf16_fused_moe")
        }

    assert migrate(db) is True
    assert migrate(db) is False

    with closing(sqlite3.connect(db)) as conn:
        for table, rows in before.items():
            assert _rows(conn, table, storage.logical_select(conn, table)) == rows
        assert conn.execute(f"SELECT COUNT(*) FROM {storage.RUN_TABLE}").fetchone()[0] == 2
        version = conn.execute("SELECT value FROM _db_metadata WHERE key = 'schema_version'")
        assert version.fetchone()[0] == str(SCHEMA_VERSION)
        assert not any(name.startswith("_kernel_config") for name in _tables(conn))


def test_upgraded_rows_are_found_by_their_args(tmp_path: Path) -> None:
    db = tmp_path / "profile.db"
    _v2_db(db)
    migrate(db)

    finalize = Table(_spec("moe_finalize_fuse_shared"), db)
    moe = Table(_spec("bf16_fused_moe"), db)
    # A bool arg is stored as TEXT '1'/'0': its hash must be of the stored form.
    [on, off] = finalize.query([_finalize_args(1, True), _finalize_args(1, False)], gpu_name=GPU)
    assert (on.time_ms, off.time_ms) == (0.01, 0.02)
    [row] = moe.rows_for([_moe_args(2)], backend=moe.profiler_spec.backend, gpu_name=GPU)
    assert row["per_expert_batches"] == "[2,0,2,0]"
    assert row["profiler_git_hash"] == "abc"
    assert row["profiler_run_at"] == "2026-09-01T00:00:00+00:00"
    assert row["created_at"] == "2026-09-03 01:02:03"
    assert "args_hash" not in row and "run_key" not in row
    assert moe.exists([_moe_args(3)], gpu_name=GPU) == [False]


def test_readers_refuse_a_db_an_older_checkout_wrote(tmp_path: Path) -> None:
    db = tmp_path / "profile.db"
    _v2_db(db)

    with pytest.raises(RuntimeError, match="migrate-db"):
        Table(_spec("moe_finalize_fuse_shared"), db).query([_finalize_args(1, True)], gpu_name=GPU)


@pytest.mark.parametrize(
    ("value", "declared", "same_as"),
    [(True, "TEXT", "1"), (False, "TEXT", "0"), (2, "REAL", 2.0), (3.0, "INTEGER", 3)],
)
def test_args_hash_sees_what_sqlite_stores(value: object, declared: str, same_as: object) -> None:
    types = {"a": declared}
    assert storage.args_hash({"a": value}, types) == storage.args_hash({"a": same_as}, types)


def test_args_hash_refuses_a_value_sqlite_would_reformat() -> None:
    with pytest.raises(TypeError):
        storage.args_hash({"a": 0.5}, {"a": "TEXT"})


def _measure(db: Path, m: int) -> None:
    table = Table(_spec("single_gemm"), db)
    table.insert(
        [
            ProfileRow(
                args=table.profiler_spec.args_schema(m=m, n=6144, k=4096, dtype=DType.BF16),
                metrics=ComputeMetrics(time_ms=0.01 * m, tflops=1.0, memory_bandwidth_gbps=1.0),
                gpu_name=GPU,
                backend="torch",
                profiler_git_hash="abc",
                profiler_run_at="2026-09-26T00:00:00+00:00",
                cuda_version="12.9",
                driver_version="580.1",
            )
        ]
    )


def test_two_dbs_with_one_run_merge_without_conflict(tmp_path: Path) -> None:
    left, right = tmp_path / "left.db", tmp_path / "right.db"
    _measure(left, 1)
    _measure(right, 2)

    report = merge_profile_databases(left, right, tmp_path / "out.db")

    assert report.published and not report.conflicts
    with closing(sqlite3.connect(tmp_path / "out.db")) as conn:
        count = lambda t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]  # noqa: E731
        assert (count("single_gemm"), count(storage.RUN_TABLE)) == (2, 1)


def test_migration_drops_the_registry_an_older_v3_checkout_wrote(tmp_path: Path) -> None:
    left, right = tmp_path / "left.db", tmp_path / "right.db"
    _measure(left, 1)
    _measure(right, 2)
    with closing(sqlite3.connect(right)) as conn, conn:
        _retired_registry(conn)

    with pytest.raises(ValueError, match="migrate-db"):
        merge_profile_databases(left, right, tmp_path / "out.db")
    assert not (tmp_path / "out.db").exists()

    assert migrate(right) is True
    assert migrate(right) is False

    with closing(sqlite3.connect(right)) as conn:
        assert not any(name.startswith("_kernel_config") for name in _tables(conn))
    [row] = Table(_spec("single_gemm"), right).rows_for(
        [_spec("single_gemm").args_schema(m=2, n=6144, k=4096, dtype=DType.BF16)],
        backend="torch",
        gpu_name=GPU,
    )
    assert row["time_ms"] == 0.02
    assert merge_profile_databases(left, right, tmp_path / "out.db").published


def test_a_conflict_report_names_the_runs_by_their_provenance(tmp_path: Path) -> None:
    left, right = tmp_path / "left.db", tmp_path / "right.db"
    _measure(left, 1)
    table = Table(_spec("single_gemm"), right)
    table.insert(
        [
            ProfileRow(
                args=table.profiler_spec.args_schema(m=1, n=6144, k=4096, dtype=DType.BF16),
                metrics=ComputeMetrics(time_ms=0.5, tflops=1.0, memory_bandwidth_gbps=1.0),
                gpu_name=GPU,
                backend="torch",
                profiler_git_hash="other",
                profiler_run_at="2026-09-26T00:00:00+00:00",
                cuda_version="12.9",
                driver_version="580.1",
            )
        ]
    )

    report = merge_profile_databases(left, right, tmp_path / "out.db")

    [conflict] = report.conflicts
    assert "time_ms" in conflict.differing_columns and "run_key" in conflict.differing_columns
    assert (conflict.left["profiler_git_hash"], conflict.right["profiler_git_hash"]) == (
        "abc",
        "other",
    )
    assert json.dumps(report.to_dict())
