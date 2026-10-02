"""CPU tests for the rocpd kernel-dispatch parser (``rocprof_kernel_profiler``).

These build a synthetic rocpd SQLite database and prove the load-bearing, GPU-
independent logic: resolving the dispatch table + timestamp/name columns from an
unknown schema, extracting per-dispatch durations in ms, filtering by kernel
name, and folding multi-dispatch launches. No GPU and no rocprofiler-sdk needed.
"""

from __future__ import annotations

import sqlite3

import pytest

from profiling.profilers.rocprof_kernel_profiler import (
    kernel_dispatch_durations_from_rocpd,
    resolve_rocpd_schema,
    summarize_rocpd,
)
from profiling.runners.exceptions import ProfilerNotImplemented


def _write_joined_rocpd(path, dispatches):
    """Primary ROCm 7.x-shape db: dispatch table + joined kernel-symbol table.

    ``dispatches`` is a list of ``(start_ns, end_ns, kernel_name)``.
    """
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE rocpd_info_kernel_symbol (id INTEGER PRIMARY KEY, display_name TEXT)")
    conn.execute(
        "CREATE TABLE rocpd_kernel_dispatch "
        "(id INTEGER PRIMARY KEY, kernel_id INTEGER, start INTEGER, end INTEGER)"
    )
    names = {}
    for start, end, name in dispatches:
        if name not in names:
            names[name] = len(names) + 1
            conn.execute(
                "INSERT INTO rocpd_info_kernel_symbol (id, display_name) VALUES (?, ?)",
                (names[name], name),
            )
    for i, (start, end, name) in enumerate(dispatches, start=1):
        conn.execute(
            "INSERT INTO rocpd_kernel_dispatch (id, kernel_id, start, end) VALUES (?, ?, ?, ?)",
            (i, names[name], start, end),
        )
    conn.commit()
    conn.close()


def _write_flat_rocpd(path, dispatches):
    """Alternate shape: kernel name lives on the dispatch row (CSV-equivalent)."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE kernel_dispatch "
        "(dispatch_id INTEGER, start_timestamp INTEGER, end_timestamp INTEGER, kernel_name TEXT)"
    )
    for i, (start, end, name) in enumerate(dispatches, start=1):
        conn.execute(
            "INSERT INTO kernel_dispatch VALUES (?, ?, ?, ?)", (i, start, end, name)
        )
    conn.commit()
    conn.close()


def test_resolve_schema_joined(tmp_path):
    db = tmp_path / "joined.db"
    _write_joined_rocpd(db, [(0, 500_000, "rms_norm_kernel")])
    conn = sqlite3.connect(db)
    try:
        schema = resolve_rocpd_schema(conn)
    finally:
        conn.close()
    assert schema.dispatch_table == "rocpd_kernel_dispatch"
    assert schema.start_column == "start"
    assert schema.end_column == "end"
    assert schema.name_table == "rocpd_info_kernel_symbol"
    assert schema.name_column == "display_name"


def test_durations_ms_and_order(tmp_path):
    db = tmp_path / "joined.db"
    # 0.5 ms, 0.25 ms, 1.0 ms in ns; inserted out of start order to prove ORDER BY.
    _write_joined_rocpd(
        db,
        [
            (1_000, 251_000, "rms_norm_kernel"),  # start 1000 -> 0.25 ms
            (0, 500_000, "rms_norm_kernel"),  # start 0 -> 0.5 ms
            (2_000, 1_002_000, "rms_norm_kernel"),  # start 2000 -> 1.0 ms
        ],
    )
    durations = kernel_dispatch_durations_from_rocpd(str(db))
    assert durations == pytest.approx([0.5, 0.25, 1.0])


def test_name_filter(tmp_path):
    db = tmp_path / "joined.db"
    _write_joined_rocpd(
        db,
        [
            (0, 500_000, "rms_norm_kernel"),
            (500_000, 600_000, "elementwise_add"),
            (600_000, 1_100_000, "rms_norm_kernel"),
        ],
    )
    only_norm = kernel_dispatch_durations_from_rocpd(str(db), kernel_name_contains="rms_norm")
    assert only_norm == pytest.approx([0.5, 0.5])
    with pytest.raises(ValueError, match="no dispatch matched"):
        kernel_dispatch_durations_from_rocpd(str(db), kernel_name_contains="nonexistent")


def test_flat_schema_name_on_dispatch(tmp_path):
    db = tmp_path / "flat.db"
    _write_flat_rocpd(db, [(0, 500_000, "rms_norm_kernel"), (500_000, 750_000, "rms_norm_kernel")])
    durations = kernel_dispatch_durations_from_rocpd(str(db), kernel_name_contains="rms_norm")
    assert durations == pytest.approx([0.5, 0.25])


def test_summarize_one_dispatch_per_launch(tmp_path):
    db = tmp_path / "joined.db"
    _write_joined_rocpd(
        db, [(i * 1_000_000, i * 1_000_000 + 400_000, "rms_norm_kernel") for i in range(4)]
    )
    summary = summarize_rocpd(str(db), num_iter=4, kernel_name_contains="rms_norm")
    assert summary.num_iter == 4
    assert summary.per_iter_ms == pytest.approx([0.4, 0.4, 0.4, 0.4])
    assert summary.mean_ms == pytest.approx(0.4)


def test_summarize_folds_multi_dispatch_launches(tmp_path):
    db = tmp_path / "joined.db"
    # 4 launches x 2 dispatches each (0.3 + 0.1 ms per launch -> 0.4 ms).
    dispatches = []
    t = 0
    for _ in range(4):
        dispatches.append((t, t + 300_000, "kA"))
        t += 300_000
        dispatches.append((t, t + 100_000, "kB"))
        t += 100_000
    _write_joined_rocpd(db, dispatches)
    summary = summarize_rocpd(str(db), num_iter=4)
    assert summary.num_iter == 4
    assert summary.per_iter_ms == pytest.approx([0.4, 0.4, 0.4, 0.4])


def test_not_a_dispatch_db_raises(tmp_path):
    db = tmp_path / "bad.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE unrelated (x INTEGER)")
    conn.commit()
    conn.close()
    with pytest.raises(ValueError, match="no kernel-dispatch table"):
        kernel_dispatch_durations_from_rocpd(str(db))


def test_profile_kernel_without_sdk_raises():
    # On a CPU host rocprofiler-sdk is absent, so the in-process collector must
    # fail honestly rather than fabricate a time.
    with pytest.raises(ProfilerNotImplemented):
        from profiling.profilers.rocprof_kernel_profiler import profile_kernel

        profile_kernel(lambda: None, num_iter=4)
