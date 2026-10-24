"""TensorFold per-round comparison against a fake predict log_dir and nsys sqlite (CPU only)."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "tensorfold_compare", ROOT / "tools" / "tensorfold_compare" / "tensorfold_compare.py"
)
tc = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = tc  # dataclasses look their module up
_SPEC.loader.exec_module(tc)

MS = 1_000_000
PID = 7 << 24
TID = PID + 7
OFFSET = 10**12  # record clock minus nsys clock
DRIFT = 30_000  # ns the record clock gains per round: past the anchor tolerance in total

HOST = {"a": 4.0, "b": 0.5, "c": 3.0}  # ms; the synthetic host term the fit must recover


def _host(tree_streams, prefill_pieces):
    return HOST["a"] + HOST["b"] * tree_streams + HOST["c"] * prefill_pieces


def _write_capture(tmp_path, levels):
    """Sidecar + one sqlite per level, built from a known per-round truth."""
    cases, sqlite_paths, truth = [], {}, {}
    for level, rounds in levels.items():
        path = tmp_path / f"n{level}.sqlite"
        db = sqlite3.connect(path)
        db.executescript(
            """
            CREATE TABLE StringIds (id INTEGER, value TEXT);
            CREATE TABLE NVTX_EVENTS (start INT, end INT, text TEXT, textId INT, globalTid INT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INT, end INT, globalTid INT,
                correlationId INT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INT, end INT, globalPid INT,
                correlationId INT);
            INSERT INTO StringIds VALUES (1, 'tensorfold.engine.iteration');
            """
        )
        t = 5 * MS
        correlation = 0
        for index in range(rounds):
            tree, pieces = index % (level + 1), index % 2
            busy = (20 + index % 5) * MS
            wall = busy + int(_host(tree, pieces) * MS)
            # An idle iteration range with no record ahead of some rounds.
            if index % 4 == 3:
                db.execute("INSERT INTO NVTX_EVENTS VALUES (?, ?, NULL, 1, ?)", (t, t + MS, TID))
                t += 5 * MS
            end = t + wall
            cut = index == rounds - 1  # the capture stops mid-range
            db.execute(
                "INSERT INTO NVTX_EVENTS VALUES (?, ?, NULL, 1, ?)",
                (t, end - (40 * MS if cut else 0), TID),
            )
            # Two overlapping kernels and one after them: busy is their union.
            first = t + MS
            kernels = (
                (first, first + busy // 2),
                (first + busy // 4, first + busy // 2),
                (first + busy // 2, first + busy),
            )
            for k_start, k_end in kernels:
                correlation += 1
                db.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?)",
                    (t + correlation, t + correlation + 10, TID, correlation),
                )
                db.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?)",
                    (k_start, k_end, PID, correlation),
                )
            record_start = t + OFFSET + index * DRIFT + 300
            cases.append(
                {
                    "case_index": len(cases),
                    "level": level,
                    "round_index": index,
                    "kind": "multi_round",
                    "start_ns": record_start,
                    "end_ns": record_start + wall,
                    "wall_ns": wall,
                    "nvtx_range": "tensorfold.engine.iteration",
                    "streams": level,
                    "tree_streams": tree,
                    "verify_rows": 16 * level,
                    "prefill_pieces": pieces,
                    "prefill_rows": 512 * pieces,
                }
            )
            truth[len(cases) - 1] = (busy / MS, cut)
            t = end + 3 * MS
        db.commit()
        db.close()
        sqlite_paths[str(level)] = path
    sidecar = tmp_path / "sidecar.json"
    sidecar.write_text(json.dumps({"schema_version": 1, "cases": cases}))
    return sidecar, sqlite_paths, truth


def _write_log_dir(tmp_path, predicted):
    log_dir = tmp_path / "predict"
    (log_dir / "raw" / "cost_log").mkdir(parents=True)
    (log_dir / "prediction.cases.json").write_text(json.dumps([{}] * len(predicted)))
    pq.write_table(
        pa.table(
            {
                "iter_id": pa.array(range(len(predicted)), type=pa.uint64()),
                "section": ["iter"] * len(predicted),
                "total_time_ms": predicted,
            }
        ),
        log_dir / "raw" / "cost_log" / "worker_predict_0.parquet",
    )
    return log_dir


def test_compare_joins_rounds_fits_host_and_reports(tmp_path):
    sidecar, sqlite_paths, truth = _write_capture(tmp_path, {1: 12, 2: 14, 4: 14, 8: 10})
    # A perfect kernel model: predicted == measured busy.
    predicted = [truth[i][0] for i in range(len(truth))]
    log_dir = _write_log_dir(tmp_path, predicted)

    rounds_by_level, fit, summary = tc.compare(
        sidecar, log_dir, sqlite_paths, fit_levels={"2", "4"}
    )

    for rounds in rounds_by_level.values():
        for item in rounds:
            busy, cut = truth[item.case["case_index"]]
            if cut:
                assert item.status == "cut_by_capture_edge"
                continue
            assert item.status == "matched"
            assert item.busy_ms == pytest.approx(busy)
    coefficients = fit["coefficients"]
    assert coefficients["a_ms"] == pytest.approx(HOST["a"], abs=1e-6)
    assert coefficients["b_ms_per_tree_stream"] == pytest.approx(HOST["b"], abs=1e-6)
    assert coefficients["c_ms_per_prefill_piece"] == pytest.approx(HOST["c"], abs=1e-6)

    by_level = {row["level"]: row for row in summary}
    assert list(by_level) == ["1", "2", "4", "8"]
    assert by_level["1"]["role"] == "holdout" and by_level["2"]["role"] == "fit"
    for row in summary:
        assert row["cut_by_capture_edge"] == 1 and row["unmatched"] == 0
        assert row["kernel_err_median_pct"] == pytest.approx(0.0, abs=1e-9)
        assert row["wall_abs_err_p90_pct"] == pytest.approx(0.0, abs=1e-6)

    markdown = tc.write_outputs(tmp_path / "out", rounds_by_level, fit, summary)
    text = markdown.read_text()
    assert "| 8 | holdout | 9 |" in text
    assert (tmp_path / "out" / "rounds.csv").read_text().count("\n") == 1 + len(truth)
    assert "kernel_err_median_pct" in (tmp_path / "out" / "summary.csv").read_text()


def test_kernel_error_sign_and_missing_capture(tmp_path):
    sidecar, sqlite_paths, truth = _write_capture(tmp_path, {2: 8, 8: 6})
    predicted = [1.1 * truth[i][0] for i in range(len(truth))]
    log_dir = _write_log_dir(tmp_path, predicted)
    del sqlite_paths["8"]

    _, fit, summary = tc.compare(sidecar, log_dir, sqlite_paths, fit_levels={"2"})
    by_level = {row["level"]: row for row in summary}
    assert by_level["2"]["kernel_err_median_pct"] == pytest.approx(10.0)
    assert by_level["8"]["no_capture"] == 6 and by_level["8"]["matched"] == 0
    assert fit is not None


def test_prediction_must_cover_every_case(tmp_path):
    sidecar, sqlite_paths, truth = _write_capture(tmp_path, {2: 4})
    log_dir = _write_log_dir(tmp_path, [1.0] * (len(truth) - 1))
    with pytest.raises(ValueError, match="sidecar has"):
        tc.compare(sidecar, log_dir, sqlite_paths, fit_levels={"2"})
