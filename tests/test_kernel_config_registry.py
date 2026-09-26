"""profile.db's kernel-config registry: registration, reads and merge."""

from __future__ import annotations

import copy
import sqlite3
from pathlib import Path

import pytest

from profiling.db import kernel_config as kc
from profiling.db.args import DType
from profiling.db.batch import coerce_args
from profiling.db.merge import merge_profile_databases
from profiling.db.registry import iter_kernel_profiler_specs
from profiling.db.table import ProfileRow, Table
from profiling.runners.metrics import ComputeMetrics

GPU = "NVIDIA H200"
SOURCE = {"timing_predict": "presets/predict_x.json", "gpu": GPU, "arch": {"type": "x"}}


def _gemm_record(n: int, ms: list[int], role: str = "unified.qkv") -> dict[str, object]:
    """A single_gemm record as `simulator ... --kernel-configs-out` writes it."""
    return {
        "kind": "single_gemm",
        "profile_kind": "single_gemm",
        "gpu_name": GPU,
        "identity": {"n": n, "k": 4096, "dtype": "bf16"},
        "grid": {
            "cache_coords": ["m"],
            "axes": [[float(m) for m in ms]],
            "cells": [{"m": m, "n": n, "k": 4096, "dtype": "bf16"} for m in ms],
            "infeasible": [],
        },
        "uses": [{"pool": "main", "role": role}],
    }


def _document(*records: dict[str, object]) -> dict[str, object]:
    return {"schema_version": kc.RECORDS_SCHEMA_VERSION, "configs": list(records)}


def _gemm_table(db: Path) -> Table:
    return Table(next(iter_kernel_profiler_specs("single_gemm")), db)


def _measure(db: Path, n: int, ms: list[int]) -> None:
    table = _gemm_table(db)
    schema = table.profiler_spec.args_schema
    table.insert(
        [
            ProfileRow(
                args=schema(m=m, n=n, k=4096, dtype=DType.BF16),
                metrics=ComputeMetrics(time_ms=0.01 * m, tflops=1.0, memory_bandwidth_gbps=1.0),
                gpu_name=GPU,
                backend="torch",
                profiler_git_hash="abc",
                profiler_run_at="2026-09-26T00:00:00+00:00",
            )
            for m in ms
        ]
    )


def _read(db: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


def test_registration_stores_the_grid_and_every_use(tmp_path: Path) -> None:
    db = tmp_path / "profile.db"
    record = _gemm_record(6144, [1, 2, 4])
    record["uses"].append({"pool": "main", "role": "unified.o_proj"})

    report = kc.register_kernel_configs(db, _document(record), {"main": SOURCE})

    assert (report.configs_added, report.sources_added, report.uses_added) == (1, 1, 2)
    with _read(db) as conn:
        [config] = kc.registered_configs(conn, "single_gemm")
    assert config.kind == "single_gemm"
    assert config.config_hash == kc.content_hash(record["identity"])
    assert config.grid.cache_coords == ("m",)
    assert config.grid.axes == ((1.0, 2.0, 4.0),)
    assert config.grid.cells[2] == {"m": 4, "n": 6144, "k": 4096, "dtype": "bf16"}
    assert [(use.source, use.role) for use in config.uses] == [
        (SOURCE, "unified.qkv"),
        (SOURCE, "unified.o_proj"),
    ]


def test_registering_known_configs_leaves_the_file_untouched(tmp_path: Path) -> None:
    db = tmp_path / "profile.db"
    document = _document(_gemm_record(6144, [1, 2, 4]))
    kc.register_kernel_configs(db, document, {"main": SOURCE})
    before = (db.stat().st_mtime_ns, db.read_bytes())

    report = kc.register_kernel_configs(db, document, {"main": SOURCE})

    assert not report.written
    assert (db.stat().st_mtime_ns, db.read_bytes()) == before


def test_a_changed_grid_replaces_the_stored_one(tmp_path: Path) -> None:
    db = tmp_path / "profile.db"
    kc.register_kernel_configs(db, _document(_gemm_record(6144, [1, 2])), {"main": SOURCE})

    report = kc.register_kernel_configs(
        db, _document(_gemm_record(6144, [1, 2, 8])), {"main": SOURCE}
    )

    assert (report.configs_added, report.configs_regridded) == (0, 1)
    with _read(db) as conn:
        [config] = kc.registered_configs(conn, "single_gemm")
    assert config.grid.axes == ((1.0, 2.0, 8.0),)


def test_identity_hash_ignores_key_order(tmp_path: Path) -> None:
    record = _gemm_record(6144, [1])
    reordered = copy.deepcopy(record)
    reordered["identity"] = dict(reversed(list(record["identity"].items())))
    db = tmp_path / "profile.db"
    kc.register_kernel_configs(db, _document(record), {"main": SOURCE})

    report = kc.register_kernel_configs(db, _document(reordered), {"main": SOURCE})

    assert not report.written


def test_every_pool_needs_a_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"no source given for pool\(s\) \['main'\]"):
        kc.register_kernel_configs(
            tmp_path / "profile.db", _document(_gemm_record(6144, [1])), {"attn": SOURCE}
        )
    assert not (tmp_path / "profile.db").exists()


def test_a_wrong_schema_version_is_refused(tmp_path: Path) -> None:
    document = _document(_gemm_record(6144, [1]))
    document["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version 99"):
        kc.register_kernel_configs(tmp_path / "profile.db", document, {"main": SOURCE})


def test_a_cell_count_that_does_not_fill_the_grid_is_refused(tmp_path: Path) -> None:
    record = _gemm_record(6144, [1, 2])
    record["grid"]["cells"].pop()
    with pytest.raises(ValueError, match="1 cells for a grid of 2"):
        kc.register_kernel_configs(tmp_path / "profile.db", _document(record), {"main": SOURCE})


def test_grid_coords_are_row_major() -> None:
    grid = kc.ConfigGrid(
        cache_coords=("batch", "rows"),
        axes=((1.0, 2.0), (10.0, 20.0, 30.0)),
        cells=tuple({} for _ in range(6)),
        infeasible=frozenset(),
    )
    assert [grid.coords(i) for i in range(6)] == [
        (1.0, 10.0),
        (1.0, 20.0),
        (1.0, 30.0),
        (2.0, 10.0),
        (2.0, 20.0),
        (2.0, 30.0),
    ]


def test_coverage_counts_rows_some_cell_reads(tmp_path: Path) -> None:
    db = tmp_path / "profile.db"
    _measure(db, 6144, [1, 2, 4, 8])
    kc.register_kernel_configs(db, _document(_gemm_record(6144, [1, 2, 16])), {"main": SOURCE})

    with _read(db) as conn:
        coverage = kc.coverage(conn, db, "single_gemm")

    assert (coverage.rows, coverage.covered) == (4, 2)
    assert [row["m"] for row in coverage.uncovered_sample] == [4, 8]


def test_rows_for_joins_cells_to_measured_rows(tmp_path: Path) -> None:
    db = tmp_path / "profile.db"
    _measure(db, 6144, [1, 2])
    kc.register_kernel_configs(db, _document(_gemm_record(6144, [1, 2, 16])), {"main": SOURCE})
    table = _gemm_table(db)
    with _read(db) as conn:
        [config] = kc.registered_configs(conn, "single_gemm")
    schema = table.profiler_spec.args_schema

    rows = table.rows_for(
        [coerce_args(schema, cell) for cell in config.grid.cells], backend="torch", gpu_name=GPU
    )

    assert [None if row is None else row["m"] for row in rows] == [1, 2, None]


def test_merge_unions_registry_rows_by_content_hash(tmp_path: Path) -> None:
    left, right = tmp_path / "left.db", tmp_path / "right.db"
    _measure(left, 6144, [1])
    _measure(right, 6144, [1])
    kc.register_kernel_configs(left, _document(_gemm_record(6144, [1])), {"main": SOURCE})
    kc.register_kernel_configs(
        right,
        _document(_gemm_record(6144, [1], role="unified.o_proj"), _gemm_record(8192, [1])),
        {"main": SOURCE},
    )

    report = merge_profile_databases(left, right, tmp_path / "out.db")

    assert report.published
    with _read(tmp_path / "out.db") as conn:
        configs = kc.registered_configs(conn, "single_gemm")
    assert sorted(config.identity["n"] for config in configs) == [6144, 8192]
    [shared] = [config for config in configs if config.identity["n"] == 6144]
    assert sorted(use.role for use in shared.uses) == ["unified.o_proj", "unified.qkv"]


def test_merge_reports_a_config_whose_grid_differs(tmp_path: Path) -> None:
    left, right = tmp_path / "left.db", tmp_path / "right.db"
    _measure(left, 6144, [1])
    _measure(right, 6144, [1])
    kc.register_kernel_configs(left, _document(_gemm_record(6144, [1])), {"main": SOURCE})
    kc.register_kernel_configs(right, _document(_gemm_record(6144, [1, 2])), {"main": SOURCE})

    report = merge_profile_databases(left, right, tmp_path / "out.db")

    assert not report.published
    [conflict] = report.conflicts
    assert conflict.table == kc.CONFIG_TABLE
    assert set(conflict.differing_columns) == {"grid_axes", "cells"}


def test_cells_are_stored_by_column_and_read_back_whole() -> None:
    cells = [
        {"m": 1, "n": 6144, "per_group_batches": [1, 0]},
        {"m": 2, "n": 6144, "per_group_batches": [1, 1]},
    ]
    packed = kc.pack_cells(cells)
    assert packed == {
        "count": 2,
        "fixed": {"n": 6144},
        "swept": {"m": [1, 2], "per_group_batches": [[1, 0], [1, 1]]},
    }
    assert kc.unpack_cells(packed) == cells
    assert kc.unpack_cells(kc.pack_cells([])) == []


def test_measured_only_skips_configs_without_a_row(tmp_path: Path) -> None:
    db = tmp_path / "profile.db"
    _measure(db, 6144, [2])
    document = _document(_gemm_record(6144, [1, 2]), _gemm_record(8192, [1, 2], role="o"))

    report = kc.register_kernel_configs(db, document, {"main": SOURCE}, measured_only=True)

    assert (report.configs, report.configs_unmeasured, report.configs_added) == (2, 1, 1)
    with _read(db) as conn:
        [config] = kc.registered_configs(conn, "single_gemm")
    assert config.identity["n"] == 6144


def test_a_bool_arg_matches_the_text_the_table_stores(tmp_path: Path) -> None:
    """vllm_mla_rope's `is_neox_style` is a bool in a TEXT column (stored '0')."""
    db = tmp_path / "profile.db"
    table = Table(next(iter_kernel_profiler_specs("vllm_mla_rope")), db)
    schema = table.profiler_spec.args_schema
    cell = {
        "num_tokens": 1,
        "num_heads": 8,
        "qk_nope_head_dim": 192,
        "rope_dim": 64,
        "max_position": 1048576,
        "is_neox_style": False,
        "input_dtype": "bf16",
    }
    table.insert(
        [
            ProfileRow(
                args=coerce_args(schema, cell),
                metrics=ComputeMetrics(time_ms=0.01, tflops=1.0, memory_bandwidth_gbps=1.0),
                gpu_name=GPU,
                backend=table.profiler_spec.backend,
                profiler_git_hash="abc",
                profiler_run_at="2026-09-26T00:00:00+00:00",
            )
        ]
    )
    with _read(db) as conn:
        assert conn.execute("SELECT is_neox_style FROM vllm_mla_rope").fetchone() == ("0",)
        assert [len(ids) for ids in kc.cell_row_ids(conn, table, GPU, [cell])] == [1]
