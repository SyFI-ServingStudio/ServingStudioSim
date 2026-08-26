from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from launcher.__main__ import main as launcher_main
from profiling.db.merge import merge_profile_databases
from profiling.db.migrate import migrate_connection


def _create_profile_db(
    path: Path,
    *,
    table_rows: dict[str, list[dict[str, object]]],
    schema_hash: str | None = None,
) -> None:
    with sqlite3.connect(path) as conn:
        migrate_connection(conn)
        if schema_hash is not None:
            conn.execute(
                "UPDATE _db_metadata SET value = ? WHERE key = 'schema_hash'",
                (schema_hash,),
            )
        for table_name, rows in table_rows.items():
            conn.execute(
                f"""
                CREATE TABLE {table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gpu_name TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    m INTEGER NOT NULL,
                    profiler_git_hash TEXT NOT NULL,
                    profiler_run_at TEXT NOT NULL,
                    cuda_version TEXT,
                    driver_version TEXT,
                    backend_version TEXT,
                    verified INTEGER NOT NULL DEFAULT 0,
                    time_ms REAL NOT NULL,
                    tflops REAL NOT NULL,
                    memory_bandwidth_gbps REAL NOT NULL,
                    energy_j REAL NOT NULL DEFAULT 0.0,
                    is_outlier INTEGER NOT NULL DEFAULT 0,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    outlier_reason TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(gpu_name, backend, m)
                )
                """
            )
            for row in rows:
                values = _profile_row(**row)
                columns = ", ".join(values)
                placeholders = ", ".join("?" for _ in values)
                conn.execute(
                    f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})",
                    tuple(values.values()),
                )


def _profile_row(
    *,
    m: int,
    time_ms: float,
    profiler_git_hash: str,
    created_at: str = "2026-01-01 00:00:00",
) -> dict[str, object]:
    return {
        "gpu_name": "NVIDIA B200",
        "backend": "torch",
        "m": m,
        "profiler_git_hash": profiler_git_hash,
        "profiler_run_at": "2026-01-01T00:00:00+00:00",
        "cuda_version": "13.0",
        "driver_version": "580.65.06",
        "backend_version": "2.8.0",
        "verified": 1,
        "time_ms": time_ms,
        "tflops": 10.0,
        "memory_bandwidth_gbps": 20.0,
        "energy_j": 0.0,
        "is_outlier": 0,
        "retry_count": 0,
        "outlier_reason": None,
        "created_at": created_at,
    }


def _rows(path: Path, table_name: str) -> list[sqlite3.Row]:
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(f"SELECT * FROM {table_name} ORDER BY m").fetchall()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_launcher_merge_db_preserves_one_sided_rows_and_tables(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = tmp_path / "left.db"
    right = tmp_path / "right.db"
    output = tmp_path / "merged.db"
    _create_profile_db(
        left,
        table_rows={"single_gemm": [{"m": 1, "time_ms": 1.0, "profiler_git_hash": "left-commit"}]},
    )
    _create_profile_db(
        right,
        table_rows={
            "single_gemm": [{"m": 2, "time_ms": 2.0, "profiler_git_hash": "right-commit"}],
            "branch_only_kernel": [{"m": 3, "time_ms": 3.0, "profiler_git_hash": "right-commit"}],
        },
    )
    input_hashes = (_sha256(left), _sha256(right))

    exit_code = launcher_main(
        [
            "kernel-profile",
            "merge-db",
            str(left),
            str(right),
            "--output",
            str(output),
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["summary"] == {
        "conflict_rows": 0,
        "duplicate_rows": 0,
        "inserted_rows": 2,
        "left_rows": 1,
        "right_rows": 2,
        "table_count": 2,
    }
    assert [(row["m"], row["profiler_git_hash"]) for row in _rows(output, "single_gemm")] == [
        (1, "left-commit"),
        (2, "right-commit"),
    ]
    assert _rows(output, "branch_only_kernel")[0]["profiler_git_hash"] == "right-commit"
    assert (_sha256(left), _sha256(right)) == input_hashes
    assert Path(f"{output}.merge-report.json").is_file()


def test_merge_db_deduplicates_identical_semantic_rows(tmp_path: Path) -> None:
    left = tmp_path / "left.db"
    right = tmp_path / "right.db"
    output = tmp_path / "merged.db"
    shared = {"m": 1, "time_ms": 1.0, "profiler_git_hash": "same-commit"}
    _create_profile_db(left, table_rows={"single_gemm": [shared]})
    _create_profile_db(
        right,
        table_rows={"single_gemm": [{**shared, "created_at": "2026-02-02 00:00:00"}]},
    )

    report = merge_profile_databases(left, right, output)

    assert report.published is True
    assert len(_rows(output, "single_gemm")) == 1
    assert report.tables[0].duplicate_rows == 1
    assert report.tables[0].conflict_rows == 0


def test_merge_db_reports_conflict_and_does_not_publish(tmp_path: Path) -> None:
    left = tmp_path / "left.db"
    right = tmp_path / "right.db"
    output = tmp_path / "merged.db"
    _create_profile_db(
        left,
        table_rows={"single_gemm": [{"m": 1, "time_ms": 1.0, "profiler_git_hash": "left-commit"}]},
    )
    _create_profile_db(
        right,
        table_rows={"single_gemm": [{"m": 1, "time_ms": 1.5, "profiler_git_hash": "right-commit"}]},
    )

    report = merge_profile_databases(left, right, output)

    assert report.published is False
    assert not output.exists()
    assert len(report.conflicts) == 1
    conflict = report.conflicts[0]
    assert conflict.key == {"gpu_name": "NVIDIA B200", "backend": "torch", "m": 1}
    assert conflict.differing_columns == ("profiler_git_hash", "time_ms")
    assert conflict.left["time_ms"] == 1.0
    assert conflict.right["time_ms"] == 1.5
    report_payload = json.loads(Path(f"{output}.merge-report.json").read_text())
    assert report_payload["output"]["published"] is False
    assert report_payload["conflicts"][0]["left"]["profiler_git_hash"] == "left-commit"
    assert report_payload["conflicts"][0]["right"]["profiler_git_hash"] == "right-commit"


def test_merge_db_rejects_schema_metadata_mismatch(tmp_path: Path) -> None:
    left = tmp_path / "left.db"
    right = tmp_path / "right.db"
    output = tmp_path / "merged.db"
    _create_profile_db(left, table_rows={})
    _create_profile_db(right, table_rows={}, schema_hash="incompatible-schema")

    with pytest.raises(ValueError, match="right profile DB schema metadata is incompatible"):
        merge_profile_databases(left, right, output)

    assert not output.exists()


def test_merge_db_does_not_overwrite_existing_output(tmp_path: Path) -> None:
    left = tmp_path / "left.db"
    right = tmp_path / "right.db"
    output = tmp_path / "merged.db"
    _create_profile_db(left, table_rows={})
    _create_profile_db(right, table_rows={})
    output.write_bytes(b"keep this output")

    with pytest.raises(FileExistsError, match="output already exists"):
        merge_profile_databases(left, right, output)

    assert output.read_bytes() == b"keep this output"
