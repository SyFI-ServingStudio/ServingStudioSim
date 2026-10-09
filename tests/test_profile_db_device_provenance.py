"""Persist worker provenance without borrowing metadata from another device family."""

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

_NEURON_BACKEND = "neuron_nki_qkv"
_NEURON_VERSION = (
    "torch-neuronx=worker-sdk; nrt=worker-runtime; "
    "container_image_id=sha256:worker-image; host_driver_package=host-neuron-driver; "
    "checkpoint_sha256=worker-checkpoint; neff_sha256=worker-neff"
)


def _row(backend: str, **metadata) -> ProfileRow:
    return ProfileRow(
        args=SingleGemmArgs(m=1, n=6144, k=4096, dtype=DType.BF16),
        metrics=ComputeMetrics(time_ms=0.5, tflops=1.0, memory_bandwidth_gbps=2.0),
        # The registry backend, rather than a requested cache label, owns family.
        gpu_name="shared cache label",
        backend=backend,
        profiler_git_hash="worker-source",
        profiler_run_at="2026-10-07T00:00:00+00:00",
        **metadata,
    )


def _stored_provenance(table: Table, row: ProfileRow) -> tuple:
    # Read actual normalized storage and the public joined row independently.
    with closing(sqlite3.connect(table.db_path)) as conn:
        stored = conn.execute(
            "SELECT r.cuda_version, r.driver_version, r.backend_version "
            "FROM single_gemm t JOIN _profile_run r USING (run_key) "
            "WHERE t.backend = ? AND t.gpu_name = ?",
            (row.backend, row.gpu_name),
        ).fetchone()
        assert conn.execute("SELECT COUNT(*) FROM _profile_run").fetchone()[0] > 0
    [joined] = table.rows_for([row.args], backend=row.backend, gpu_name=row.gpu_name)
    assert joined is not None
    assert stored == tuple(
        joined[name] for name in ("cuda_version", "driver_version", "backend_version")
    )
    return stored


@pytest.mark.parametrize("table_backend", ["torch", _NEURON_BACKEND])
def test_mixed_backend_table_uses_each_row_family_for_controller_defaults(
    tmp_path: Path, monkeypatch, table_backend: str
) -> None:
    cuda = Mock(return_value="controller-cuda")
    driver = Mock(return_value="controller-nvidia-driver")
    monkeypatch.setattr(table_module, "_cuda_version", cuda)
    monkeypatch.setattr(table_module, "_driver_version", driver)
    backend_fallback = Mock(side_effect=AssertionError("worker backend provenance was lost"))
    monkeypatch.setattr(table_module, "_backend_version", backend_fallback)
    table = Table(find_kernel_profiler_spec("single_gemm", table_backend), tmp_path / "profile.db")
    neuron = _row(_NEURON_BACKEND, backend_version=_NEURON_VERSION)
    cuda_row = _row("torch", backend_version="worker-torch")

    table.insert([neuron, cuda_row])

    assert _stored_provenance(table, neuron) == (None, None, _NEURON_VERSION)
    assert _stored_provenance(table, cuda_row) == (
        "controller-cuda",
        "controller-nvidia-driver",
        "worker-torch",
    )
    cuda.assert_called_once_with()
    driver.assert_called_once_with()
    backend_fallback.assert_not_called()


@pytest.mark.parametrize("backend", ["torch", _NEURON_BACKEND])
def test_explicit_worker_versions_are_preserved_without_controller_probes(
    tmp_path: Path, monkeypatch, backend: str
) -> None:
    cuda = Mock(side_effect=AssertionError("controller CUDA probe must not run"))
    driver = Mock(side_effect=AssertionError("controller NVIDIA probe must not run"))
    monkeypatch.setattr(table_module, "_cuda_version", cuda)
    monkeypatch.setattr(table_module, "_driver_version", driver)
    table = Table(find_kernel_profiler_spec("single_gemm", "torch"), tmp_path / "profile.db")
    row = _row(
        backend,
        cuda_version="worker-cuda",
        driver_version="worker-driver",
        backend_version=_NEURON_VERSION if backend == _NEURON_BACKEND else "worker-torch",
    )

    table.insert([row])

    assert _stored_provenance(table, row) == (
        row.cuda_version,
        row.driver_version,
        row.backend_version,
    )
    cuda.assert_not_called()
    driver.assert_not_called()


@pytest.mark.parametrize("device_family", ["cuda", "neuron"])
def test_direct_unregistered_table_spec_keeps_its_declared_family(
    tmp_path: Path, monkeypatch, device_family: str
) -> None:
    cuda = Mock(return_value="controller-cuda")
    driver = Mock(return_value="controller-driver")
    monkeypatch.setattr(table_module, "_cuda_version", cuda)
    monkeypatch.setattr(table_module, "_driver_version", driver)
    registered = find_kernel_profiler_spec("single_gemm", "torch")
    spec = replace(
        registered,
        backend=f"unregistered_{device_family}",
        supports=replace(registered.supports, device_family=device_family),
    )
    table = Table(spec, tmp_path / "profile.db")
    row = _row(spec.backend, backend_version="declared-worker")

    table.insert([row])

    expected = ("controller-cuda", "controller-driver") if device_family == "cuda" else (None, None)
    assert _stored_provenance(table, row) == (*expected, "declared-worker")
    assert cuda.call_count == driver.call_count == (1 if device_family == "cuda" else 0)


def test_neuron_missing_versions_do_not_probe_controller_and_replacement_stays_null(
    tmp_path: Path, monkeypatch
) -> None:
    cuda = Mock(side_effect=AssertionError("Neuron must not probe controller CUDA"))
    driver = Mock(side_effect=AssertionError("Neuron must not probe controller NVIDIA"))
    monkeypatch.setattr(table_module, "_cuda_version", cuda)
    monkeypatch.setattr(table_module, "_driver_version", driver)
    table = Table(find_kernel_profiler_spec("single_gemm", "torch"), tmp_path / "profile.db")
    old = _row(
        _NEURON_BACKEND,
        cuda_version="old-cuda",
        driver_version="old-driver",
        backend_version="old-sdk",
    )
    fresh = _row(_NEURON_BACKEND, backend_version=_NEURON_VERSION)

    table.insert([old])
    table.insert([fresh])

    assert _stored_provenance(table, fresh) == (None, None, _NEURON_VERSION)
    cuda.assert_not_called()
    driver.assert_not_called()
