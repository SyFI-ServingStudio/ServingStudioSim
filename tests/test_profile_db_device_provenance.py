"""A profile row's CUDA and driver provenance belongs to its own device family."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from profiling.db import table as table_module
from profiling.db.args import DType
from profiling.db.registry import find_kernel_profiler_spec
from profiling.db.table import ProfileRow, Table
from profiling.kernels.single_gemm import SingleGemmArgs
from profiling.runners.metrics import ComputeMetrics

_WORKER_VERSION = "torch-neuronx=worker-sdk; container_image_id=sha256:worker-image"


def _table(tmp_path: Path, device_family: str) -> Table:
    registered = find_kernel_profiler_spec("single_gemm", "torch")
    spec = replace(
        registered,
        backend=f"fixture_{device_family}",
        supports=replace(registered.supports, device_family=device_family),
    )
    return Table(spec, tmp_path / "profile.db")


def _row(table: Table, **metadata) -> ProfileRow:
    return ProfileRow(
        args=SingleGemmArgs(m=1, n=6144, k=4096, dtype=DType.BF16),
        metrics=ComputeMetrics(time_ms=0.5, tflops=1.0, memory_bandwidth_gbps=2.0),
        gpu_name="fixture device",
        backend=table.profiler_spec.backend,
        profiler_git_hash="worker-source",
        profiler_run_at="2026-10-07T00:00:00+00:00",
        **metadata,
    )


def _stored_provenance(table: Table, row: ProfileRow) -> tuple:
    with closing(sqlite3.connect(table.db_path)) as conn:
        stored = conn.execute(
            "SELECT r.cuda_version, r.driver_version, r.backend_version "
            "FROM single_gemm t JOIN _profile_run r USING (run_key) "
            "WHERE t.backend = ? AND t.gpu_name = ?",
            (row.backend, row.gpu_name),
        ).fetchone()
    [joined] = table.rows_for([row.args], backend=row.backend, gpu_name=row.gpu_name)
    assert stored == tuple(
        joined[name] for name in ("cuda_version", "driver_version", "backend_version")
    )
    return stored


@pytest.mark.parametrize("device_family", ["cuda", "neuron"])
def test_controller_cuda_defaults_apply_only_to_cuda_rows(tmp_path, monkeypatch, device_family):
    cuda = Mock(return_value="controller-cuda")
    driver = Mock(return_value="controller-driver")
    monkeypatch.setattr(table_module, "_cuda_version", cuda)
    monkeypatch.setattr(table_module, "_driver_version", driver)
    table = _table(tmp_path, device_family)
    row = _row(table, backend_version=_WORKER_VERSION)

    table.insert([row])

    expected = ("controller-cuda", "controller-driver") if device_family == "cuda" else (None, None)
    assert _stored_provenance(table, row) == (*expected, _WORKER_VERSION)
    assert cuda.call_count == driver.call_count == (1 if device_family == "cuda" else 0)


@pytest.mark.parametrize("device_family", ["cuda", "neuron"])
def test_explicit_worker_versions_are_kept_without_controller_probes(
    tmp_path, monkeypatch, device_family
):
    monkeypatch.setattr(table_module, "_cuda_version", Mock(side_effect=AssertionError))
    monkeypatch.setattr(table_module, "_driver_version", Mock(side_effect=AssertionError))
    table = _table(tmp_path, device_family)
    row = _row(
        table,
        cuda_version="worker-cuda",
        driver_version="worker-driver",
        backend_version=_WORKER_VERSION,
    )

    table.insert([row])

    assert _stored_provenance(table, row) == ("worker-cuda", "worker-driver", _WORKER_VERSION)


def test_a_replaced_neuron_row_does_not_keep_the_old_cuda_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(table_module, "_cuda_version", Mock(side_effect=AssertionError))
    monkeypatch.setattr(table_module, "_driver_version", Mock(side_effect=AssertionError))
    table = _table(tmp_path, "neuron")
    old = _row(table, cuda_version="old-cuda", driver_version="old-driver", backend_version="old")
    fresh = _row(table, backend_version=_WORKER_VERSION)

    table.insert([old])
    table.insert([fresh])

    assert _stored_provenance(table, fresh) == (None, None, _WORKER_VERSION)
