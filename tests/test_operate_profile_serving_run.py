import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[1]
    / "skills"
    / "operate-profile-serving-run"
    / "scripts"
    / "aggregate_nsys.py"
)


@pytest.fixture
def nsys_fixture(tmp_path: Path) -> Path:
    database = tmp_path / "synthetic.sqlite"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE PROCESSES (
            globalPid INTEGER,
            pid INTEGER,
            name TEXT
        );
        CREATE TABLE NVTX_EVENTS (
            start INTEGER,
            end INTEGER,
            text TEXT,
            globalTid INTEGER
        );
        -- Deliberately mirrors the workspace export: no globalPid column.
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            start INTEGER,
            end INTEGER,
            globalTid INTEGER,
            correlationId INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start INTEGER,
            end INTEGER,
            globalPid INTEGER,
            correlationId INTEGER
        );
        """
    )

    first_global_pid = 100 << 24
    second_global_pid = 200 << 24
    first_worker_tid = first_global_pid + 110
    second_main_tid = second_global_pid + 200
    unmapped_tid = 999

    connection.executemany(
        "INSERT INTO PROCESSES VALUES (?, ?, ?)",
        [
            (first_global_pid, 100, "first"),
            (second_global_pid, 200, "second"),
        ],
    )
    connection.executemany(
        "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?)",
        [
            (100, 200, "vibeserve.decode", first_worker_tid),
            (300, 400, "vibeserve.decode", first_worker_tid),
            (100, 200, "vibeserve.second", second_main_tid),
            (500, 550, "vibeserve.unmapped", unmapped_tid),
            (0, 1000, "other.ignored", first_worker_tid),
        ],
    )
    connection.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?)",
        [
            (110, 112, first_worker_tid, 7),
            (305, 307, first_worker_tid, 8),
            (115, 117, second_main_tid, 7),
            (510, 512, unmapped_tid, 9),
        ],
    )
    connection.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?)",
        [
            (110, 150, first_global_pid, 7),
            (140, 220, first_global_pid, 7),
            (305, 450, first_global_pid, 8),
            (115, 125, second_global_pid, 7),
            (510, 520, second_global_pid, 9),
        ],
    )
    connection.commit()
    connection.close()
    return database


def _run_aggregator(database: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(database),
            "--range-prefix",
            "vibeserve.",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_process_scoped_correlation_union_clipping_and_occurrences(
    nsys_fixture: Path,
) -> None:
    database_before = nsys_fixture.read_bytes()
    report = _run_aggregator(nsys_fixture)
    labels = {item["label"]: item for item in report["labels"]}
    decode = labels["vibeserve.decode"]

    assert nsys_fixture.read_bytes() == database_before
    assert report["read_only"] is True
    assert report["schema"]["runtime_has_global_pid"] is False
    assert decode["occurrence_count"] == 2
    assert decode["runtime_call_count"] == 2
    assert decode["kernel_count"] == 3
    assert decode["host_wall_time_ns"] == 200
    assert decode["kernel_work_ns"] == 265
    assert decode["gpu_busy_time_ns"] == 185
    assert decode["uncovered_host_time_ns"] == 15
    assert decode["gpu_busy_fraction"] == pytest.approx(0.925)
    assert all(0.0 <= item["gpu_busy_fraction"] <= 1.0 for item in report["labels"])
    assert all(item["uncovered_host_time_ns"] >= 0 for item in report["labels"])

    # Process two reuses correlation ID 7, but its kernel must not cross-join.
    second = labels["vibeserve.second"]
    assert second["kernel_count"] == 1
    assert second["kernel_work_ns"] == 10
    assert second["gpu_busy_time_ns"] == 10


def test_non_main_and_unmapped_cuda_threads_are_explicit(
    nsys_fixture: Path,
) -> None:
    report = _run_aggregator(nsys_fixture)
    diagnostics = {item["status"]: item for item in report["thread_diagnostics"]}
    labels = {item["label"]: item for item in report["labels"]}

    assert diagnostics["mapped_non_main_thread"]["os_tid"] == 110
    assert diagnostics["mapped_non_main_thread"]["process_name"] == "first"
    assert diagnostics["mapped_non_main_thread"]["range_occurrence_count"] == 2
    assert diagnostics["mapped_main_thread"]["os_tid"] == 200
    assert diagnostics["mapped_main_thread"]["resolved_pid"] == 200
    assert diagnostics["unmapped"]["global_tid"] == 999
    assert diagnostics["unmapped"]["reason"]
    assert labels["vibeserve.unmapped"]["runtime_call_count"] == 1
    assert labels["vibeserve.unmapped"]["kernel_count"] == 0
    assert any("unmapped threads" in warning for warning in report["warnings"])


def test_missing_required_schema_is_rejected(tmp_path: Path) -> None:
    database = tmp_path / "invalid.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE PROCESSES (globalPid INTEGER, pid INTEGER, name TEXT)")
    connection.close()

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(database),
            "--range-prefix",
            "vibeserve.",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "required table is absent: NVTX_EVENTS" in result.stderr
