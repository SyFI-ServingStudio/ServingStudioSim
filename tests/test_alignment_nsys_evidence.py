from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from alignment.nsys.evidence import aggregate, kernel_category, main, merge_duration_ns


def test_maps_fork_child_from_exact_kernel_namespace(tmp_path: Path) -> None:
    database_path = tmp_path / "capture.sqlite"
    namespace_global_pid = 1234 << 24
    os_thread_id = 5678
    global_thread_id = namespace_global_pid + os_thread_id

    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE StringIds(id INTEGER, value TEXT);
            CREATE TABLE PROCESSES(globalPid INTEGER, pid INTEGER, name TEXT);
            CREATE TABLE NVTX_EVENTS(
                start INTEGER, end INTEGER, text TEXT, textId INTEGER, globalTid INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(
                start INTEGER, end INTEGER, globalTid INTEGER,
                correlationId INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
                start INTEGER, end INTEGER, globalPid INTEGER,
                correlationId INTEGER, deviceId INTEGER, demangledName INTEGER
            );
            INSERT INTO StringIds VALUES (1, 'flash_fwd_kernel');
            """
        )
        connection.execute(
            "INSERT INTO PROCESSES VALUES (?, ?, ?)",
            (999 << 24, 999, "launcher"),
        )
        connection.execute(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?)",
            (100, 300, "test.forward", None, global_thread_id),
        )
        connection.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?)",
            (120, 130, global_thread_id, 42),
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?)",
            [
                (150, 200, namespace_global_pid, 42, 3, 1),
                (230, 350, namespace_global_pid, 42, 3, 1),
            ],
        )

    report = aggregate(database_path, "test.")
    metrics = report["labels"][0]

    assert metrics["kernel_count"] == 2
    assert metrics["host_range_kernel_coverage_ns"] == 120
    assert metrics["host_range_kernel_coverage_fraction"] == 0.6
    assert metrics["kernel_busy_union_ns"] == 170
    assert metrics["kernel_span_time_ns"] == 200
    assert metrics["gpu_idle_within_kernel_span_ns"] == 30
    assert metrics["uncovered_host_time_ns"] == 80
    assert metrics["kernel_work_ns_by_category"] == {"attention": 170}
    assert metrics["top_kernels"] == [
        {"name": "flash_fwd_kernel", "kernel_work_ns": 170, "count": 2}
    ]
    assert report["thread_diagnostics"][0]["status"] == "mapped_kernel_namespace"
    assert report["thread_diagnostics"][0]["resolved_global_pid"] == namespace_global_pid
    assert report["artifact_kind"] == "nsys_range_evidence"


def test_shared_kernel_taxonomy_and_interval_union() -> None:
    assert kernel_category("ncclDevKernel_AllReduce") == "nccl_collective"
    assert kernel_category("flash_fwd_kernel") == "attention"
    assert merge_duration_ns([(0, 10), (5, 15), (20, 25)]) == 20


def test_range_label_may_be_interned_in_string_ids(tmp_path: Path) -> None:
    database_path = tmp_path / "interned.sqlite"
    namespace_global_pid = 22 << 24
    global_thread_id = namespace_global_pid + 7
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            f"""
            CREATE TABLE StringIds(id INTEGER, value TEXT);
            CREATE TABLE PROCESSES(globalPid INTEGER, pid INTEGER, name TEXT);
            CREATE TABLE NVTX_EVENTS(
                start INTEGER, end INTEGER, text TEXT, textId INTEGER, globalTid INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(
                start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
                start INTEGER, end INTEGER, globalPid INTEGER,
                correlationId INTEGER, deviceId INTEGER, demangledName INTEGER
            );
            INSERT INTO StringIds VALUES (1, 'test.interned');
            INSERT INTO StringIds VALUES (2, 'ncclDevKernel_AllReduce');
            INSERT INTO NVTX_EVENTS VALUES (10, 20, NULL, 1, {global_thread_id});
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
                (11, 12, {global_thread_id}, 5);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
                (13, 19, {namespace_global_pid}, 5, 0, 2);
            """
        )

    report = aggregate(database_path, "test.")

    assert report["labels"][0]["label"] == "test.interned"
    assert report["labels"][0]["kernel_count"] == 1

    kernel_events_path = tmp_path / "kernel-events.json"
    assert (
        main(
            [
                str(database_path),
                "--range-prefix",
                "test.",
                "--kernel-events-output",
                str(kernel_events_path),
            ]
        )
        == 0
    )
    events = json.loads(kernel_events_path.read_text())
    assert events["artifact_kind"] == "nsys_kernel_events"
    assert events["occurrences"][0]["kernels"][0]["category"] == "nccl_collective"
