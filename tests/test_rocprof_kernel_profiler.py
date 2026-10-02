"""CPU tests for the rocpd kernel-dispatch parser (``rocprof_kernel_profiler``).

These build a synthetic rocpd SQLite database and prove the load-bearing, GPU-
independent logic: resolving the dispatch table + timestamp/name columns from an
unknown schema, extracting per-dispatch durations in ms, filtering by kernel
name, and folding multi-dispatch launches. No GPU and no rocprofiler-sdk needed.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys

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


def _write_real_shape_rocpd(path, dispatches):
    """Reproduce the real rocprofv3 1.3.2 / ROCm 7.2 rocpd shape.

    Verified against a live MI210 capture: GUID-suffixed table names, a
    ``rocpd_kernel_dispatch_<guid>`` table with ``start``/``end`` BIGINT and a
    ``kernel_id`` FK into ``rocpd_info_kernel_symbol_<guid>`` (``display_name``),
    plus the ``region_name_id`` integer column that must NOT be mistaken for the
    kernel name. ``dispatches`` is ``(start_ns, end_ns, display_name)``.
    """
    guid = "0000b253_e83e_783e_9413_8df416218653"
    sym = f"rocpd_info_kernel_symbol_{guid}"
    disp = f"rocpd_kernel_dispatch_{guid}"
    conn = sqlite3.connect(path)
    conn.execute(f"CREATE TABLE {sym} (id INTEGER PRIMARY KEY, kernel_name TEXT, display_name TEXT)")
    conn.execute(
        f"CREATE TABLE {disp} (id INTEGER PRIMARY KEY, kernel_id INTEGER, "
        "queue_id INTEGER, stream_id INTEGER, start BIGINT, end BIGINT, "
        "region_name_id INTEGER, event_id INTEGER)"
    )
    conn.execute(f"CREATE TABLE rocpd_string_{guid} (id INTEGER PRIMARY KEY, string TEXT)")
    ids = {}
    for _s, _e, name in dispatches:
        if name not in ids:
            ids[name] = len(ids) + 1
            conn.execute(
                f"INSERT INTO {sym} (id, kernel_name, display_name) VALUES (?, ?, ?)",
                (ids[name], f"_mangled_{name}", name),
            )
    for i, (start, end, name) in enumerate(dispatches, start=1):
        conn.execute(
            f"INSERT INTO {disp} (id, kernel_id, queue_id, stream_id, start, end, "
            "region_name_id, event_id) VALUES (?, ?, 1, 0, ?, ?, ?, ?)",
            (i, ids[name], start, end, i, i),
        )
    conn.commit()
    conn.close()


def test_real_rocpd_shape_resolves_display_name_via_join(tmp_path):
    # The region_name_id decoy must NOT be picked as the name column; resolution
    # must join to the kernel-symbol table and read display_name.
    db = tmp_path / "real.db"
    _write_real_shape_rocpd(
        db,
        [
            (0, 42_000, "distribution_elementwise_grid_stride_kernel"),
            (100_000, 111_200, "vectorized_layer_norm_kernel<c10::BFloat16>"),
            (200_000, 211_200, "vectorized_layer_norm_kernel<c10::BFloat16>"),
        ],
    )
    conn = sqlite3.connect(db)
    try:
        schema = resolve_rocpd_schema(conn)
    finally:
        conn.close()
    assert "rocpd_kernel_dispatch_" in schema.dispatch_table
    assert (schema.start_column, schema.end_column) == ("start", "end")
    assert schema.name_table is not None and "kernel_symbol" in schema.name_table
    assert schema.name_column == "display_name"
    norm = kernel_dispatch_durations_from_rocpd(
        str(db), kernel_name_contains="vectorized_layer_norm_kernel"
    )
    assert norm == pytest.approx([0.0112, 0.0112])


def test_profile_kernel_without_sdk_raises():
    # On a CPU host rocprofiler-sdk is absent, so the in-process collector must
    # fail honestly rather than fabricate a time.
    with pytest.raises(ProfilerNotImplemented):
        from profiling.profilers.rocprof_kernel_profiler import profile_kernel

        profile_kernel(lambda: None, num_iter=4)


def test_find_rocprofv3_absent_raises(monkeypatch):
    # The whole-process capture must report a missing tracer clearly rather than
    # fail deep in a subprocess call.
    import profiling.profilers.rocprof_kernel_profiler as mod

    monkeypatch.setattr(mod.shutil, "which", lambda _name: None)
    monkeypatch.delenv("ROCM_HOME", raising=False)
    monkeypatch.delenv("ROCM_PATH", raising=False)
    monkeypatch.setattr(mod.Path, "exists", lambda self: False)
    with pytest.raises(ProfilerNotImplemented, match="rocprofv3 not found"):
        mod._find_rocprofv3()


def test_rocprof_run_builder_registry_has_rms_norm():
    # The launch driver must know how to rebuild the one wired kernel from its
    # (kind, backend) spec.
    import profiling.profilers.rocprof_run as rr

    assert ("rms_norm", "torch_rocm") in rr._BUILDERS


def test_rocprof_run_builder_registry_has_kda_recurrent_decode():
    # The KDA decode torch_rocm backend must be driveable under rocprofv3.
    import profiling.profilers.rocprof_run as rr

    assert ("kda_recurrent_decode", "torch_rocm") in rr._BUILDERS


def test_fold_per_launch_mean_sums_constant_dispatches():
    # 3 dispatches per launch, warmup=2, rep=3 -> 15 dispatches. The 6 warmup
    # dispatches are dropped; each timed launch's 3 dispatches are summed.
    from profiling.profilers.rocprof_kernel_profiler import _fold_per_launch_mean

    warmup_block = [9.0] * 6  # dropped
    timed = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]  # three launches of 6.0 ms
    assert _fold_per_launch_mean(warmup_block + timed, warmup=2, rep=3) == pytest.approx(6.0)


def test_fold_per_launch_mean_drops_one_time_init_prefix():
    # Real captures carry a small fixed prefix of device-init dispatches before
    # the first launch (seen: 4 on the MI300X image). With D=5, warmup=5, rep=20,
    # a 129-dispatch stream is prefix(4) + 125 uniform; the four prefix and the
    # 25 warmup dispatches drop, and each timed launch's 5 dispatches are summed.
    from profiling.profilers.rocprof_kernel_profiler import _fold_per_launch_mean

    prefix = [99.0] * 4
    launch = [1.0, 1.0, 1.0, 1.0, 1.0]  # each launch sums to 5.0 ms
    stream = prefix + launch * 25  # 4 + 125 = 129
    assert len(stream) == 129
    assert _fold_per_launch_mean(stream, warmup=5, rep=20) == pytest.approx(5.0)


def test_fold_per_launch_mean_rejects_too_few_dispatches():
    # Fewer dispatches than launches means no clean per-launch count; refuse to
    # guess rather than report a fabricated number.
    from profiling.profilers.rocprof_kernel_profiler import _fold_per_launch_mean
    from profiling.runners.exceptions import KernelLaunchFailed

    with pytest.raises(KernelLaunchFailed):
        _fold_per_launch_mean([1.0] * 3, warmup=2, rep=3)  # 3 < 5 launches


def test_measure_registered_rejects_fold_with_name_filter():
    # fold_per_launch counts every dispatch; a name filter would contradict it.
    from profiling.profilers.rocprof_kernel_profiler import measure_registered_via_rocprofv3

    with pytest.raises(ValueError):
        measure_registered_via_rocprofv3(
            kind="kda_recurrent_decode",
            backend="torch_rocm",
            spec={},
            kernel_name_contains="something",
            warmup=1,
            rep=1,
            fold_per_launch=True,
        )


def test_rocprof_run_import_does_not_eager_import_torch():
    # Importing the driver must not pull in torch (built lazily per spec).
    command = [
        sys.executable,
        "-c",
        "import sys; import profiling.profilers.rocprof_run; print('torch' in sys.modules)",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == "False"
