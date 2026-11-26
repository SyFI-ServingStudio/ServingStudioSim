"""The per-kind profile DB layout (``profiling.db.layout``): splitting a legacy
single file into a directory, merging directories kind file by kind file, and
``Table`` reading one file per kind."""

from __future__ import annotations

import hashlib
import sqlite3
from collections import OrderedDict
from contextlib import closing
from pathlib import Path

import pytest

from profiling.db import layout, storage
from profiling.db import table as table_mod
from profiling.db.args import DType
from profiling.db.merge import merge_profile_databases
from profiling.db.migrate import migrate_connection
from profiling.db.registry import iter_kernel_profiler_specs
from profiling.db.split import split_profile_db
from profiling.db.table import ProfileRow, Table
from profiling.runners.metrics import ComputeMetrics

GPU = "NVIDIA H200"
RUN_A = ("aaa", "12.9", "580.1", None)
RUN_B = ("bbb", "12.9", "580.1", "0.5.1")
RUN_C = ("ccc", None, None, None)
ORPHAN_KEY = 424242
ARGS = {"m": "INTEGER"}

# kind -> rows of (m, time_ms, run); a run of None is the orphan run_key.
Rows = dict[str, list[tuple[int, float, tuple | None]]]


def _create_kind(conn: sqlite3.Connection, kind: str) -> None:
    conn.execute(storage.kind_table_schema(kind, ["m INTEGER NOT NULL"], ["time_ms REAL NOT NULL"]))


def _insert(conn: sqlite3.Connection, kind: str, m: int, time_ms: float, run) -> None:
    key = ORPHAN_KEY if run is None else storage.run_key(conn, run)
    conn.execute(
        f"INSERT INTO {kind} (gpu_name, backend, m, args_hash, run_key, profiler_run_at, "
        "time_ms) VALUES (?, 'torch', ?, ?, ?, 1790000000, ?)",
        (GPU, m, storage.args_hash({"m": m}, ARGS), key, time_ms),
    )


def _fill(conn: sqlite3.Connection, rows: Rows) -> None:
    migrate_connection(conn)
    conn.execute(storage.RUN_SCHEMA)
    for kind, kind_rows in rows.items():
        _create_kind(conn, kind)
        for m, time_ms, run in kind_rows:
            _insert(conn, kind, m, time_ms, run)


def _legacy(path: Path, rows: Rows, *, unreferenced: tuple = ()) -> Path:
    """A current (v3) single-file DB holding every kind of ``rows``."""
    with closing(sqlite3.connect(path)) as conn, conn:
        _fill(conn, rows)
        for run in unreferenced:
            storage.run_key(conn, run)
    return path


def _directory(root: Path, rows: Rows) -> Path:
    """A profile DB directory: one file per kind of ``rows``."""
    root.mkdir()
    for kind, kind_rows in rows.items():
        with closing(sqlite3.connect(layout.kind_file(root, kind))) as conn, conn:
            _fill(conn, {kind: kind_rows})
    return root


def _select(path: Path, sql: str) -> list[tuple]:
    with closing(sqlite3.connect(path)) as conn:
        return conn.execute(sql).fetchall()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── split ────────────────────────────────────────────────────────────────────


def test_split_keeps_every_row_and_only_the_runs_each_kind_references(tmp_path: Path) -> None:
    legacy = _legacy(
        tmp_path / "profile.db",
        {
            "alpha": [(1, 0.1, RUN_A), (2, 0.2, RUN_B)],
            # RUN_B is shared with alpha; one row keys a run the file does not have.
            "beta": [(1, 0.3, RUN_B), (2, 0.4, None)],
            "gamma": [],
        },
        unreferenced=(RUN_C,),
    )
    with closing(sqlite3.connect(legacy)) as conn, conn:
        conn.execute("CREATE INDEX alpha_m ON alpha(m)")
    source_hash = _sha256(legacy)
    output = tmp_path / "profile_db"

    report = split_profile_db(legacy, output)

    assert [path.name for path in layout.kind_files(output)] == ["alpha.db", "beta.db"]
    assert report.empty_tables == ("gamma",)
    assert report.unreferenced_runs == 1
    tables = {table.table: table for table in report.tables}
    assert (tables["alpha"].rows, tables["alpha"].runs, tables["alpha"].orphan_rows) == (2, 2, 0)
    assert (tables["beta"].rows, tables["beta"].runs) == (2, 1)
    assert (tables["beta"].orphan_run_keys, tables["beta"].orphan_rows) == ((ORPHAN_KEY,), 1)
    assert report.summary == {
        "files": 2,
        "rows": 4,
        "empty_tables": 1,
        "orphan_rows": 1,
        "orphan_run_keys": 1,
        "unreferenced_runs": 1,
    }

    run_sql = f"SELECT * FROM {storage.RUN_TABLE} WHERE profiler_git_hash IN ({{}}) ORDER BY id"
    for kind, hashes in (("alpha", "'aaa', 'bbb'"), ("beta", "'bbb'")):
        kind_file = layout.kind_file(output, kind)
        # Every row, column for column, id included.
        assert _select(kind_file, f"SELECT * FROM {kind} ORDER BY id") == _select(
            legacy, f"SELECT * FROM {kind} ORDER BY id"
        )
        # Exactly the runs the kind's rows reference; the unreferenced run is not copied.
        assert _select(kind_file, f"SELECT * FROM {storage.RUN_TABLE} ORDER BY id") == _select(
            legacy, run_sql.format(hashes)
        )
        assert _select(kind_file, "SELECT * FROM _db_metadata ORDER BY key") == _select(
            legacy, "SELECT * FROM _db_metadata ORDER BY key"
        )
    assert _select(
        layout.kind_file(output, "alpha"),
        "SELECT name FROM sqlite_master WHERE type = 'index' AND name = 'alpha_m'",
    ) == [("alpha_m",)]
    assert _sha256(legacy) == source_hash


def test_split_refuses_a_directory_an_existing_output_and_an_older_schema(
    tmp_path: Path,
) -> None:
    legacy = _legacy(tmp_path / "profile.db", {"alpha": [(1, 0.1, RUN_A)]})
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        split_profile_db(legacy, existing)
    with pytest.raises(ValueError, match="not a single-file"):
        split_profile_db(existing, tmp_path / "out")

    with closing(sqlite3.connect(legacy)) as conn, conn:
        conn.execute("UPDATE _db_metadata SET value = '2' WHERE key = 'schema_version'")
    with pytest.raises(RuntimeError, match="migrate-db"):
        split_profile_db(legacy, tmp_path / "out")
    # A refused split leaves no output or staging directory behind.
    assert sorted(path.name for path in tmp_path.iterdir()) == ["existing", "profile.db"]


# ── merge ────────────────────────────────────────────────────────────────────

LEFT: Rows = {"alpha": [(1, 0.1, RUN_A)], "beta": [(1, 0.3, RUN_A)]}
RIGHT: Rows = {"beta": [(2, 0.4, RUN_B)], "gamma": [(3, 0.5, RUN_B)]}


def _assert_merged(output: Path, report) -> None:
    assert report.published and not report.conflicts
    assert [path.name for path in layout.kind_files(output)] == ["alpha.db", "beta.db", "gamma.db"]
    stats = {
        table.table: (table.left_rows, table.right_rows, table.inserted_rows)
        for table in report.tables
    }
    # One entry per kind: left only, both, right only.
    assert stats == {"alpha": (1, 0, 0), "beta": (1, 1, 1), "gamma": (0, 1, 1)}
    rows = "SELECT m, time_ms, r.profiler_git_hash FROM {} t JOIN _profile_run r USING (run_key)"
    assert _select(layout.kind_file(output, "alpha"), rows.format("alpha")) == [(1, 0.1, "aaa")]
    assert sorted(_select(layout.kind_file(output, "beta"), rows.format("beta"))) == [
        (1, 0.3, "aaa"),
        (2, 0.4, "bbb"),
    ]
    assert _select(layout.kind_file(output, "gamma"), rows.format("gamma")) == [(3, 0.5, "bbb")]
    payload = report.to_dict()
    assert payload["report_schema_version"] == 2
    assert sorted(payload["output"]["files"]) == ["alpha", "beta", "gamma"]
    assert payload["summary"]["table_count"] == 3
    assert payload["summary"]["inserted_rows"] == 2


def test_directories_merge_kinds_only_one_side_has_and_kinds_both_have(tmp_path: Path) -> None:
    left = _directory(tmp_path / "left", LEFT)
    right = _directory(tmp_path / "right", RIGHT)
    before = {path.name: _sha256(path) for root in (left, right) for path in root.iterdir()}

    report = merge_profile_databases(left, right, tmp_path / "merged")

    _assert_merged(tmp_path / "merged", report)
    assert report.left["legacy_file"] is False and sorted(report.right["files"]) == [
        "beta",
        "gamma",
    ]
    assert {path.name: _sha256(path) for root in (left, right) for path in root.iterdir()} == before


def test_a_legacy_single_file_merges_as_its_split(tmp_path: Path) -> None:
    left = _directory(tmp_path / "left", LEFT)
    right = _legacy(tmp_path / "right.db", RIGHT)
    right_hash = _sha256(right)

    report = merge_profile_databases(left, right, tmp_path / "merged")

    _assert_merged(tmp_path / "merged", report)
    assert report.right["legacy_file"] is True
    assert report.right["sha256"] == right_hash == _sha256(right)
    assert sorted(report.right["files"]) == ["beta", "gamma"]


# ── Table ────────────────────────────────────────────────────────────────────


def _gemm_row(m: int, time_ms: float) -> ProfileRow:
    spec = next(iter_kernel_profiler_specs("single_gemm"))
    return ProfileRow(
        args=spec.args_schema(m=m, n=128, k=128, dtype=DType.BF16),
        metrics=ComputeMetrics(time_ms=time_ms, tflops=1.0, memory_bandwidth_gbps=1.0),
        gpu_name=GPU,
        backend=spec.backend,
        profiler_git_hash="abc",
    )


def _finalize_row(num_tokens: int, time_ms: float) -> ProfileRow:
    spec = next(iter_kernel_profiler_specs("moe_finalize_fuse_shared"))
    return ProfileRow(
        args=spec.args_schema(
            num_tokens=num_tokens,
            top_k=8,
            hidden_dim=7168,
            dtype=DType.BF16,
            fuse_shared_output=True,
        ),
        metrics=ComputeMetrics(time_ms=time_ms, tflops=1.0, memory_bandwidth_gbps=1.0),
        gpu_name=GPU,
        backend=spec.backend,
        profiler_git_hash="abc",
    )


def test_table_refuses_a_legacy_single_file(tmp_path: Path) -> None:
    legacy = _legacy(tmp_path / "profile.db", {"single_gemm": []})
    spec = next(iter_kernel_profiler_specs("single_gemm"))

    with pytest.raises(layout.LegacyProfileDbError, match="split-db"):
        Table(spec, legacy)


def test_table_reuses_one_read_connection_per_kind_file_across_alternating_kinds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "profile_db"
    gemm = Table(next(iter_kernel_profiler_specs("single_gemm")), root)
    finalize = Table(next(iter_kernel_profiler_specs("moe_finalize_fuse_shared")), root)
    gemm.insert([_gemm_row(8, 0.01)])
    finalize.insert([_finalize_row(4, 0.02)])
    assert sorted(path.name for path in root.iterdir()) == [
        "moe_finalize_fuse_shared.db",
        "single_gemm.db",
    ]
    cache: OrderedDict = OrderedDict()
    monkeypatch.setattr(table_mod._read_only, "connections", cache, raising=False)

    gemm_conn = gemm._connect_read_only()
    finalize_conn = finalize._connect_read_only()
    assert gemm_conn is not None and finalize_conn is not None
    assert gemm_conn is not finalize_conn
    for _ in range(2):
        [g] = gemm.query([_gemm_row(8, 0).args], gpu_name=GPU)
        [f] = finalize.query([_finalize_row(4, 0).args], gpu_name=GPU)
        assert (g.time_ms, f.time_ms) == (0.01, 0.02)
        assert gemm._connect_read_only() is gemm_conn
        assert finalize._connect_read_only() is finalize_conn
    # The per-thread cache holds both kinds' files at once.
    assert set(cache) == {gemm.db_file.resolve(), finalize.db_file.resolve()}

    # A write to one kind's file reopens that file only.
    gemm.insert([_gemm_row(16, 0.03)])
    assert gemm._connect_read_only() is not gemm_conn
    assert finalize._connect_read_only() is finalize_conn
    assert [m.time_ms for m in gemm.query([_gemm_row(16, 0).args], gpu_name=GPU)] == [0.03]
