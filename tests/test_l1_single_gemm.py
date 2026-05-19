from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from profiling import perf_api
from profiling.db import (
    SCHEMA_HASH,
    SCHEMA_VERSION,
    DType,
    KernelKind,
    KernelProfilerSpec,
    MetricFamily,
    ProfileRow,
    RunnerRef,
    SingleGemmArgs,
    Table,
    find_kernel_profiler_spec,
    run_profile_batch,
)
from profiling.db.table import MissingEntry
from profiling.exec import (
    ENV_REGISTRY,
    ChunkResult,
    GpuChunk,
    GpuPool,
    LocalGpuPool,
    ProfileEnv,
    resolve_profile_env,
    set_default_pool,
)
from profiling.exec.local import LocalGpuChunk, find_idle_gpus
from profiling.runners.comm._launcher import (
    MultiGpuLauncher,
    NvshmemLauncher,
    TorchMpLauncher,
    VllmLauncher,
)
from profiling.runners.metrics import CommMetrics, ComputeMetrics


class SingleResultChunk(GpuChunk):
    def run(self, kernel_kind: KernelKind, specs: list[dict]) -> list[ChunkResult]:
        assert kernel_kind == KernelKind.GEMM_SINGLE
        assert specs == [{"m": 8, "n": 8, "k": 8, "dtype": "fp16", "backend": "torch"}]
        return [
            ChunkResult(
                metrics=ComputeMetrics(
                    time_ms=2.0,
                    tflops=0.1,
                    memory_bandwidth_gbps=0.2,
                    energy_j=0.0,
                ),
                gpu_name="FakeGPU",
            )
            for _ in specs
        ]


class SingleResultPool(GpuPool):
    def acquire_chunks(self, k: int, max_concurrent: int):
        assert k == 1
        assert max_concurrent == 1
        yield SingleResultChunk()


class ExplodingPool(GpuPool):
    def acquire_chunks(self, k: int, max_concurrent: int):
        del k, max_concurrent
        raise AssertionError("run_profile_batch should validate before acquiring GPUs")


class RecordingGpuChunk(GpuChunk):
    def __init__(self, gpu_name: str):
        self.gpu_name = gpu_name
        self.received_spec_batches: list[list[dict]] = []

    def run(self, kernel_kind: KernelKind, specs: list[dict]) -> list[ChunkResult]:
        assert kernel_kind == KernelKind.GEMM_SINGLE
        self.received_spec_batches.append(specs)
        return [
            ChunkResult(
                metrics=ComputeMetrics(
                    time_ms=float(spec["m"]),
                    tflops=float(spec["n"]),
                    memory_bandwidth_gbps=float(spec["k"]),
                    energy_j=0.0,
                ),
                gpu_name=self.gpu_name,
            )
            for spec in specs
        ]


class RecordingPool(GpuPool):
    def __init__(self, chunks: list[RecordingGpuChunk]):
        self.chunks = chunks
        self.acquire_calls: list[tuple[int, int]] = []

    def acquire_chunks(self, k: int, max_concurrent: int):
        self.acquire_calls.append((k, max_concurrent))
        yield from self.chunks


def test_dtype_from_value_accepts_runner_aliases():
    assert DType.from_value(DType.FP16) is DType.FP16
    assert DType.from_value("torch.float16") is DType.FP16
    assert DType.from_value("half") is DType.FP16
    assert DType.from_value("torch.bfloat16") is DType.BF16
    assert DType.from_value("torch.float32") is DType.FP32


def test_registry_does_not_import_runner_modules_eagerly():
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import profiling.db.registry; "
            "print('profiling.runners.gemm.torch' in sys.modules)"
        ),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)

    assert completed.stdout.strip() == "False"


def test_profile_env_errors_are_explicit(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown profiling env 'missing_env'"):
        resolve_profile_env("missing_env")

    missing_python = tmp_path / "env" / "bin" / "python"
    with pytest.raises(FileNotFoundError, match="profiling env 'broken_env'"):
        ProfileEnv("broken_env", missing_python).validate_python_executable()


def test_energy_perf_uses_total_energy_counter(monkeypatch: pytest.MonkeyPatch):
    from profiling.profilers import energy as energy_mod
    from profiling.profilers.energy import Energy

    energy_mod._TOTAL_ENERGY_SUPPORTED_BY_GPU.clear()

    class FakeCuda:
        sync_calls = 0

        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def current_device() -> int:
            return 0

        @staticmethod
        def synchronize() -> None:
            FakeCuda.sync_calls += 1

    class FakeTorch:
        cuda = FakeCuda

    class FakeNVMLError(Exception):
        def __init__(self, value: int):
            super().__init__(value)
            self.value = value

    class FakePynvml:
        NVML_ERROR_NOT_SUPPORTED = 999
        NVMLError = FakeNVMLError

        def __init__(self) -> None:
            self.total_energy_reads = [777, 1000, 1300]

        def nvmlInit(self) -> None:
            pass

        def nvmlDeviceGetHandleByIndex(self, device_idx: int) -> str:
            assert device_idx == 0
            return "handle"

        def nvmlDeviceGetUUID(self, handle: str) -> str:
            assert handle == "handle"
            return "GPU-counter"

        def nvmlDeviceGetTotalEnergyConsumption(self, handle: str) -> int:
            assert handle == "handle"
            return self.total_energy_reads.pop(0)

    fake_pynvml = FakePynvml()
    fn_calls = 0

    def fn() -> None:
        nonlocal fn_calls
        fn_calls += 1

    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setattr(Energy, "_estimate_iters", staticmethod(lambda *_, **__: 3))
    monkeypatch.setattr(energy_mod.time, "perf_counter", iter([0.0, 1.0]).__next__)

    assert Energy.perf(fn, warmup=0, min_duration_ms=1000) == pytest.approx(0.1)
    assert fn_calls == 3
    assert FakeCuda.sync_calls == 2


def test_energy_perf_uses_passed_timing_for_iteration_count(
    monkeypatch: pytest.MonkeyPatch,
):
    from profiling.profilers import energy as energy_mod
    from profiling.profilers.energy import Energy

    energy_mod._TOTAL_ENERGY_SUPPORTED_BY_GPU.clear()

    class FakeCuda:
        sync_calls = 0

        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def current_device() -> int:
            return 0

        @staticmethod
        def synchronize() -> None:
            FakeCuda.sync_calls += 1

    class FakeTorch:
        cuda = FakeCuda

    class FakeNVMLError(Exception):
        def __init__(self, value: int):
            super().__init__(value)
            self.value = value

    class FakePynvml:
        NVML_ERROR_NOT_SUPPORTED = 999
        NVMLError = FakeNVMLError

        def __init__(self) -> None:
            self.total_energy_reads = [777, 1000, 1400]

        def nvmlInit(self) -> None:
            pass

        def nvmlDeviceGetHandleByIndex(self, device_idx: int) -> str:
            assert device_idx == 0
            return "handle"

        def nvmlDeviceGetUUID(self, handle: str) -> str:
            assert handle == "handle"
            return "GPU-passed-timing"

        def nvmlDeviceGetTotalEnergyConsumption(self, handle: str) -> int:
            assert handle == "handle"
            return self.total_energy_reads.pop(0)

    def unexpected_estimate(*args: object, **kwargs: object) -> int:
        del args, kwargs
        raise AssertionError("passed timing must bypass Energy._estimate_iters")

    fake_pynvml = FakePynvml()
    fn_calls = 0

    def fn() -> None:
        nonlocal fn_calls
        fn_calls += 1

    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setattr(Energy, "_estimate_iters", staticmethod(unexpected_estimate))
    monkeypatch.setattr(energy_mod.time, "perf_counter", iter([0.0, 1.0]).__next__)

    assert Energy.perf(
        fn,
        warmup=0,
        min_duration_ms=1000,
        per_iter_time_ms=250.0,
    ) == pytest.approx(0.1)
    assert fn_calls == 4
    assert FakeCuda.sync_calls == 2


def test_energy_window_restarts_cleanly_when_planned_iters_are_short(
    monkeypatch: pytest.MonkeyPatch,
):
    from profiling.profilers import energy as energy_mod
    from profiling.profilers.energy import Energy

    energy_mod._TOTAL_ENERGY_SUPPORTED_BY_GPU.clear()

    perf_counter_values = iter([0.0, 0.4, 10.0, 11.1])
    monkeypatch.setattr(energy_mod.time, "perf_counter", lambda: next(perf_counter_values))

    class FakeCuda:
        sync_calls = 0

        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def current_device() -> int:
            return 0

        @staticmethod
        def synchronize() -> None:
            FakeCuda.sync_calls += 1

    class FakeTorch:
        cuda = FakeCuda

    class FakeNVMLError(Exception):
        def __init__(self, value: int):
            super().__init__(value)
            self.value = value

    class FakePynvml:
        NVML_ERROR_NOT_SUPPORTED = 999
        NVMLError = FakeNVMLError

        def __init__(self) -> None:
            self.total_energy_reads = [777, 1000, 1040, 2000, 3000]

        def nvmlInit(self) -> None:
            pass

        def nvmlDeviceGetHandleByIndex(self, device_idx: int) -> str:
            assert device_idx == 0
            return "handle"

        def nvmlDeviceGetUUID(self, handle: str) -> str:
            assert handle == "handle"
            return "GPU-clean-restart"

        def nvmlDeviceGetTotalEnergyConsumption(self, handle: str) -> int:
            assert handle == "handle"
            return self.total_energy_reads.pop(0)

    def unexpected_estimate(*args: object, **kwargs: object) -> int:
        del args, kwargs
        raise AssertionError("passed timing must bypass Energy._estimate_iters")

    fake_pynvml = FakePynvml()
    fn_calls = 0

    def fn() -> None:
        nonlocal fn_calls
        fn_calls += 1

    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setattr(Energy, "_estimate_iters", staticmethod(unexpected_estimate))

    assert Energy.perf(
        fn,
        warmup=0,
        min_duration_ms=1000,
        per_iter_time_ms=250.0,
    ) == pytest.approx(0.1)

    assert fn_calls == 14
    assert FakeCuda.sync_calls == 4
    assert fake_pynvml.total_energy_reads == []


def test_energy_perf_falls_back_to_power_polling(monkeypatch: pytest.MonkeyPatch):
    from profiling.profilers import energy as energy_mod
    from profiling.profilers.energy import Energy

    energy_mod._TOTAL_ENERGY_SUPPORTED_BY_GPU.clear()

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def current_device() -> int:
            return 0

        @staticmethod
        def synchronize() -> None:
            pass

    class FakeTorch:
        cuda = FakeCuda

    class FakeNVMLError(Exception):
        def __init__(self, value: int):
            super().__init__(value)
            self.value = value

    class FakePynvml:
        NVML_ERROR_NOT_SUPPORTED = 999
        NVMLError = FakeNVMLError

        def __init__(self) -> None:
            self.total_energy_calls = 0

        def nvmlInit(self) -> None:
            pass

        def nvmlDeviceGetHandleByIndex(self, device_idx: int) -> str:
            assert device_idx == 0
            return "handle"

        def nvmlDeviceGetUUID(self, handle: str) -> str:
            assert handle == "handle"
            return "GPU-polling"

        def nvmlDeviceGetTotalEnergyConsumption(self, handle: str) -> int:
            assert handle == "handle"
            self.total_energy_calls += 1
            raise FakeNVMLError(self.NVML_ERROR_NOT_SUPPORTED)

    class FakePoller:
        def __init__(self, pynvml: object, handle: object, *, poll_hz: int) -> None:
            assert pynvml is fake_pynvml
            assert handle == "handle"
            assert poll_hz == 100
            self.avg_watts = 50.0
            self.elapsed_s = 2.0
            self.error = None

        def __enter__(self) -> FakePoller:
            return self

        def __exit__(self, *args: object) -> None:
            pass

    fake_pynvml = FakePynvml()
    fn_calls = 0

    def fn() -> None:
        nonlocal fn_calls
        fn_calls += 1

    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setattr(Energy, "_estimate_iters", staticmethod(lambda *_, **__: 4))
    monkeypatch.setattr(energy_mod, "_NvmlPoller", FakePoller)

    assert Energy.perf(fn, warmup=0, min_duration_ms=1000) == pytest.approx(25.0)
    assert fn_calls == 4
    assert fake_pynvml.total_energy_calls == 1


def test_torch_single_gemm_passes_timer_result_to_energy(
    monkeypatch: pytest.MonkeyPatch,
):
    from profiling.runners.gemm import torch as torch_gemm_runner

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

    class FakeTensor:
        def __init__(self, rows: int, columns: int) -> None:
            self._rows = rows
            self._columns = columns

        def numel(self) -> int:
            return self._rows * self._columns

    class FakeTorch:
        cuda = FakeCuda
        float16 = "float16"
        bfloat16 = "bfloat16"
        float32 = "float32"

        @staticmethod
        def randn(rows: int, columns: int, *, dtype: object, device: str) -> FakeTensor:
            assert dtype == "float16"
            assert device == "cuda"
            return FakeTensor(rows, columns)

        @staticmethod
        def mm(a: FakeTensor, b: FakeTensor) -> FakeTensor:
            return FakeTensor(a._rows, b._columns)

    captured_energy_kwargs: dict[str, object] = {}

    def fake_do_bench(fn, *, warmup: int, rep: int) -> float:
        assert warmup == 10
        assert rep == 1000
        fn()
        return 2.5

    def fake_energy_perf(fn, **kwargs: object) -> float:
        fn()
        captured_energy_kwargs.update(kwargs)
        return 0.25

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setattr(torch_gemm_runner.Timer, "do_bench", staticmethod(fake_do_bench))
    monkeypatch.setattr(torch_gemm_runner.Energy, "perf", staticmethod(fake_energy_perf))

    metrics = torch_gemm_runner.profile_single_gemm(2, 3, 4, DType.FP16)

    assert metrics.time_ms == 2.5
    assert metrics.energy_j == 0.25
    assert captured_energy_kwargs["per_iter_time_ms"] == 2.5
    assert captured_energy_kwargs["min_duration_ms"] == 1000


def test_single_gemm_perf_api_query_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(perf_api, "DB_PATH", tmp_path / "profile.db")
    spec = {"m": 16, "n": 32, "k": 64, "dtype": "fp16"}

    missing = perf_api.get_single_gemm_times([spec], backend="torch", gpu_name="TestGPU")[0]
    assert isinstance(missing, MissingEntry)
    assert perf_api.count_missing_single_gemm([spec], backend="torch", gpu_name="TestGPU") == 1

    profiler_spec = find_kernel_profiler_spec(KernelKind.GEMM_SINGLE, "torch")
    table = Table(profiler_spec, perf_api.DB_PATH)
    args = SingleGemmArgs(m=16, n=32, k=64, dtype=DType.FP16)
    table.insert(
        [
            ProfileRow(
                args=args,
                metrics=ComputeMetrics(
                    time_ms=1.25,
                    tflops=0.006,
                    memory_bandwidth_gbps=0.001,
                    energy_j=0.0,
                ),
                gpu_name="TestGPU",
                backend="torch",
            )
        ]
    )

    result = perf_api.get_single_gemm_times([spec], backend="torch", gpu_name="TestGPU")[0]
    assert isinstance(result, ComputeMetrics)
    assert result.time_ms == 1.25
    assert perf_api.count_missing_single_gemm([spec], backend="torch", gpu_name="TestGPU") == 0

    db_metadata = perf_api.get_db_metadata()
    assert db_metadata.schema_version == SCHEMA_VERSION
    assert db_metadata.schema_hash == SCHEMA_HASH
    assert db_metadata.created_at is not None
    assert db_metadata.last_migrated_at is not None

    table_metadata = table.metadata()
    assert table_metadata.row_count == 1
    assert table_metadata.schema_hash == SCHEMA_HASH
    assert table_metadata.profiler_git_hashes

    versions = perf_api.get_profiler_versions(["gemm_single"])
    assert versions
    assert versions[0].op_family == "gemm_single"


def test_perf_api_force_refreshes_cached_rows_with_jit_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(perf_api, "DB_PATH", tmp_path / "profile.db")
    spec = {"m": 8, "n": 8, "k": 8, "dtype": "fp16"}
    profiler_spec = find_kernel_profiler_spec(KernelKind.GEMM_SINGLE, "torch")
    table = Table(profiler_spec, perf_api.DB_PATH)
    table.insert(
        [
            ProfileRow(
                args=SingleGemmArgs(m=8, n=8, k=8, dtype=DType.FP16),
                metrics=ComputeMetrics(
                    time_ms=99.0,
                    tflops=0.001,
                    memory_bandwidth_gbps=0.001,
                    energy_j=0.0,
                ),
                gpu_name="FakeGPU",
                backend="torch",
            )
        ]
    )

    set_default_pool(SingleResultPool())
    perf_api.disable_jit_profiling()
    try:
        result = perf_api.get_single_gemm_times(
            [spec],
            backend="torch",
            gpu_name="FakeGPU",
            force=True,
        )[0]
    finally:
        set_default_pool(None)
        perf_api.disable_jit_profiling()

    assert isinstance(result, ComputeMetrics)
    assert result.time_ms == 2.0
    cached = perf_api.get_single_gemm_times([spec], backend="torch", gpu_name="FakeGPU")[0]
    assert cached == result


def test_run_profile_batch_uses_gpu_pool_and_saves_table(tmp_path: Path):
    db_path = tmp_path / "profile.db"
    run_profile_batch(
        KernelKind.GEMM_SINGLE,
        [{"m": 8, "n": 8, "k": 8, "dtype": "fp16", "backend": "torch"}],
        pool=SingleResultPool(),
        db_path=db_path,
    )

    profiler_spec = find_kernel_profiler_spec(KernelKind.GEMM_SINGLE, "torch")
    table = Table(profiler_spec, db_path)
    saved = table.query(
        [SingleGemmArgs(m=8, n=8, k=8, dtype=DType.FP16)],
        backend="torch",
        gpu_name="FakeGPU",
    )[0]
    assert isinstance(saved, ComputeMetrics)
    assert saved.time_ms == 2.0


def test_table_schema_uses_metric_family_columns(tmp_path: Path):
    profiler_spec = find_kernel_profiler_spec(KernelKind.GEMM_SINGLE, "torch")
    table = Table(profiler_spec, tmp_path / "profile.db")
    table.metadata()

    with sqlite3.connect(tmp_path / "profile.db") as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(single_gemm)")}

    assert {"time_ms", "tflops", "memory_bandwidth_gbps", "energy_j"} <= columns
    assert "algbw_gbps" not in columns
    assert "busbw_gbps" not in columns
    assert "message_size_bytes" not in columns


def test_table_schema_drops_metric_columns_from_other_family(tmp_path: Path):
    db_path = tmp_path / "profile.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE single_gemm (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gpu_name TEXT NOT NULL,
                backend TEXT NOT NULL,
                m INTEGER NOT NULL,
                n INTEGER NOT NULL,
                k INTEGER NOT NULL,
                dtype TEXT NOT NULL,
                profiler_git_hash TEXT NOT NULL,
                profiler_run_at TEXT NOT NULL,
                cuda_version TEXT,
                driver_version TEXT,
                backend_version TEXT,
                verified INTEGER NOT NULL DEFAULT 0,
                time_ms REAL,
                tflops REAL,
                memory_bandwidth_gbps REAL,
                algbw_gbps REAL,
                busbw_gbps REAL,
                message_size_bytes INTEGER,
                energy_j REAL,
                is_outlier INTEGER NOT NULL DEFAULT 0,
                retry_count INTEGER NOT NULL DEFAULT 0,
                outlier_reason TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(gpu_name, backend, m, n, k, dtype)
            )
            """
        )

    profiler_spec = find_kernel_profiler_spec(KernelKind.GEMM_SINGLE, "torch")
    table = Table(profiler_spec, db_path)
    table.metadata()

    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(single_gemm)")}

    assert {"time_ms", "tflops", "memory_bandwidth_gbps", "energy_j"} <= columns
    assert "algbw_gbps" not in columns
    assert "busbw_gbps" not in columns
    assert "message_size_bytes" not in columns


def test_registry_rejects_table_metric_family_conflicts():
    from profiling.db.registry import _validate_registry

    profiler_spec = find_kernel_profiler_spec(KernelKind.GEMM_SINGLE, "torch")
    conflicting_spec = KernelProfilerSpec(
        kernel_kind=KernelKind.GEMM_SINGLE,
        backend="other",
        runner_ref=RunnerRef("profiling.runners.gemm.torch", "profile_single_gemm"),
        table_name=profiler_spec.table_name,
        args_schema=profiler_spec.args_schema,
        metric_family=MetricFamily.COMM,
        batch_outlier_policy=profiler_spec.batch_outlier_policy,
    )

    with pytest.raises(ValueError, match="conflicting table contract for single_gemm"):
        _validate_registry((profiler_spec, conflicting_spec))


def test_table_rejects_metrics_outside_registered_family(tmp_path: Path):
    profiler_spec = find_kernel_profiler_spec(KernelKind.GEMM_SINGLE, "torch")
    table = Table(profiler_spec, tmp_path / "profile.db")
    args = SingleGemmArgs(m=8, n=8, k=8, dtype=DType.FP16)

    with pytest.raises(ValueError, match="single_gemm is a compute metrics table"):
        table.insert(
            [
                ProfileRow(
                    args=args,
                    metrics=CommMetrics(
                        time_ms=1.0,
                        algbw_gbps=0.1,
                        busbw_gbps=0.2,
                        message_size_bytes=1024,
                        energy_j=0.0,
                    ),
                    gpu_name="FakeGPU",
                    backend="torch",
                ),
            ]
        )


def test_table_replacement_policy_overwrites_same_profile_point(tmp_path: Path):
    profiler_spec = find_kernel_profiler_spec(KernelKind.GEMM_SINGLE, "torch")
    table = Table(profiler_spec, tmp_path / "profile.db")
    args = SingleGemmArgs(m=8, n=8, k=8, dtype=DType.FP16)

    table.insert(
        [
            ProfileRow(
                args=args,
                metrics=ComputeMetrics(
                    time_ms=1.0,
                    tflops=0.1,
                    memory_bandwidth_gbps=0.2,
                    energy_j=0.0,
                ),
                gpu_name="FakeGPU",
                backend="torch",
            )
        ]
    )
    table.insert(
        [
            ProfileRow(
                args=args,
                metrics=ComputeMetrics(
                    time_ms=2.0,
                    tflops=0.3,
                    memory_bandwidth_gbps=0.4,
                    energy_j=0.0,
                ),
                gpu_name="FakeGPU",
                backend="torch",
            )
        ]
    )

    saved = table.query([args], backend="torch", gpu_name="FakeGPU")[0]
    assert isinstance(saved, ComputeMetrics)
    assert saved.time_ms == 2.0
    assert saved.tflops == 0.3


def test_run_profile_batch_balances_specs_across_chunks(tmp_path: Path):
    specs = [
        {"m": m, "n": 8, "k": 16, "dtype": "torch.float16", "backend": "torch"}
        for m in range(1, 6)
    ]
    recording_chunks = [RecordingGpuChunk("FakeGPU0"), RecordingGpuChunk("FakeGPU1")]
    pool = RecordingPool(recording_chunks)

    results = run_profile_batch(
        KernelKind.GEMM_SINGLE,
        specs,
        pool=pool,
        db_path=tmp_path / "profile.db",
    )

    assert pool.acquire_calls == [(1, len(specs))]
    assert [
        [spec["m"] for spec in spec_batch]
        for spec_batch in recording_chunks[0].received_spec_batches
    ] == [[1, 3, 5]]
    assert [
        [spec["m"] for spec in spec_batch]
        for spec_batch in recording_chunks[1].received_spec_batches
    ] == [[2, 4]]
    assert [metric.time_ms if metric is not None else None for metric in results] == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ]

    profiler_spec = find_kernel_profiler_spec(KernelKind.GEMM_SINGLE, "torch")
    table = Table(profiler_spec, tmp_path / "profile.db")
    assert table.metadata().row_count == len(specs)


def test_local_gpu_pool_yields_non_overlapping_chunks():
    reserved_chunks = list(
        LocalGpuPool(gpus=[0, 1, 2, 3, 4]).acquire_chunks(2, max_concurrent=3)
    )

    assert [chunk.gpus for chunk in reserved_chunks] == [[0, 1], [2, 3]]


def test_find_idle_gpus_respects_parent_cuda_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
):
    gpu_xml = (
        "<gpu>"
        "<uuid>{uuid}</uuid>"
        "<fb_memory_usage><used>0 MiB</used></fb_memory_usage>"
        "<utilization><gpu_util>0 %</gpu_util></utilization>"
        "</gpu>"
    )
    xml = "<nvidia_smi_log>" + "".join(
        gpu_xml.format(uuid=f"GPU-{gpu_index}")
        for gpu_index in range(4)
    ) + "</nvidia_smi_log>"

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    monkeypatch.setattr(
        "profiling.exec.local.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=xml,
            stderr="",
        ),
    )

    assert find_idle_gpus() == [3, 1]


def test_run_profile_batch_validates_specs_before_gpu_work():
    with pytest.raises(ValueError, match="missing required spec field 'k'"):
        run_profile_batch(
            KernelKind.GEMM_SINGLE,
            [{"m": 8, "n": 8, "dtype": "fp16", "backend": "torch"}],
            pool=ExplodingPool(),
        )


def test_run_profile_batch_rejects_unknown_backend_before_gpu_work():
    with pytest.raises(ValueError, match="unknown backend 'missing'"):
        run_profile_batch(
            KernelKind.GEMM_SINGLE,
            [{"m": 8, "n": 8, "k": 8, "dtype": "fp16", "backend": "missing"}],
            pool=ExplodingPool(),
        )


def test_perf_api_rejects_spec_backend_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(perf_api, "DB_PATH", tmp_path / "profile.db")
    with pytest.raises(ValueError, match="spec backend 'missing' does not match"):
        perf_api.get_single_gemm_times(
            [{"m": 8, "n": 8, "k": 8, "dtype": "fp16", "backend": "missing"}],
            backend="torch",
            gpu_name="TestGPU",
        )


def test_single_gemm_perf_api_example_profile_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the perf_api GEMM profiling example")

    gpu_name = torch.cuda.get_device_name(0)
    profile_specs = [
        {"m": m, "n": 8192, "k": 8192, "dtype": "fp16"}
        for m in [128, 256, 512, 1024, 2048, 4096, 8192]
    ]

    monkeypatch.setattr(perf_api, "DB_PATH", tmp_path / "profile.db")
    # This is the public-path example for future agents: enter through perf_api,
    # and let the facade's JIT miss path schedule the batch internally.
    set_default_pool(LocalGpuPool(gpus=[_first_visible_gpu_index()]))
    perf_api.disable_jit_profiling()
    try:
        assert (
            perf_api.count_missing_single_gemm(
                profile_specs,
                backend="torch",
                gpu_name=gpu_name,
            )
            == len(profile_specs)
        )

        perf_api.enable_jit_profiling()
        profile_results = perf_api.get_single_gemm_times(
            profile_specs,
            backend="torch",
            gpu_name=gpu_name,
        )

        assert len(profile_results) == len(profile_specs)
        assert all(isinstance(result, ComputeMetrics) for result in profile_results)
        compute_results = [
            result for result in profile_results if isinstance(result, ComputeMetrics)
        ]
        assert all(result.time_ms > 0 for result in compute_results)
        assert all(result.tflops > 0 for result in compute_results)
        assert compute_results[-1].time_ms > compute_results[0].time_ms
        assert compute_results[-1].tflops > 100
        assert (
            perf_api.count_missing_single_gemm(
                profile_specs,
                backend="torch",
                gpu_name=gpu_name,
            )
            == 0
        )

        perf_api.disable_jit_profiling()
        cached_results = perf_api.get_single_gemm_times(
            profile_specs,
            backend="torch",
            gpu_name=gpu_name,
        )
        assert cached_results == profile_results
    finally:
        perf_api.disable_jit_profiling()
        set_default_pool(None)


def test_local_chunk_rejects_unknown_backend_before_subprocess():
    with pytest.raises(ValueError, match="unknown backend 'missing'"):
        LocalGpuChunk([0]).run(
            KernelKind.GEMM_SINGLE,
            [{"m": 8, "n": 8, "k": 8, "dtype": "fp16", "backend": "missing"}],
        )


def test_local_chunk_rejects_mixed_backends_before_subprocess(monkeypatch: pytest.MonkeyPatch):
    def fake_resolve_spec_backend(_: KernelKind, spec: dict) -> str:
        return str(spec["backend"])

    monkeypatch.setattr(
        "profiling.exec.payload.resolve_spec_backend",
        fake_resolve_spec_backend,
    )
    with pytest.raises(ValueError, match="chunk specs must share one backend"):
        LocalGpuChunk([0]).run(
            KernelKind.GEMM_SINGLE,
            [
                {"m": 8, "n": 8, "k": 8, "dtype": "fp16", "backend": "torch"},
                {"m": 8, "n": 8, "k": 8, "dtype": "fp16", "backend": "other"},
            ],
        )


def test_local_chunk_uses_selected_external_python_and_project_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fake_python = tmp_path / "external_env" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_python.chmod(0o755)

    profiler_spec = find_kernel_profiler_spec(KernelKind.GEMM_SINGLE, "torch")
    external_profiler_spec = replace(profiler_spec, subprocess_env="external_env")

    def fake_find_kernel_profiler_spec(kernel_kind: KernelKind, backend: str):
        assert kernel_kind == KernelKind.GEMM_SINGLE
        assert backend == "torch"
        return external_profiler_spec

    def fake_resolve_profile_env(name: str | None) -> ProfileEnv:
        assert name == "external_env"
        return ProfileEnv("external_env", fake_python)

    captured: dict[str, object] = {}

    def fake_subprocess_run(cmd, *, env, capture_output, text, check):
        del capture_output, text, check
        captured["cmd"] = cmd
        captured["env"] = env
        output_path = Path(cmd[cmd.index("--worker-output") + 1])
        output_path.write_text(
            (
                '{"results":[{"ok":true,"metrics":{"kind":"compute",'
                '"time_ms":1.0,"tflops":0.1,"memory_bandwidth_gbps":0.2,'
                '"energy_j":0.0},"gpu_name":"FakeGPU"}]}'
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "profiling.exec.local.find_kernel_profiler_spec",
        fake_find_kernel_profiler_spec,
    )
    monkeypatch.setattr("profiling.exec.local.resolve_profile_env", fake_resolve_profile_env)
    monkeypatch.setattr("profiling.exec.local.subprocess.run", fake_subprocess_run)

    results = LocalGpuChunk([2]).run(
        KernelKind.GEMM_SINGLE,
        [{"m": 8, "n": 8, "k": 8, "dtype": "fp16", "backend": "torch"}],
    )

    assert captured["cmd"][:3] == [
        str(fake_python),
        "-m",
        "profiling.exec.local_worker",
    ]
    captured_env = captured["env"]
    assert isinstance(captured_env, dict)
    assert captured_env["CUDA_VISIBLE_DEVICES"] == "2"
    assert Path(captured_env["PYTHONPATH"].split(os.pathsep)[0]) == Path(__file__).parents[1]
    assert isinstance(results[0].metrics, ComputeMetrics)


def test_run_profile_batch_rejects_invalid_gpu_count():
    with pytest.raises(ValueError, match="gpu_count_fn must return >= 1"):
        run_profile_batch(
            KernelKind.GEMM_SINGLE,
            [{"m": 8, "n": 8, "k": 8, "dtype": "fp16", "backend": "torch"}],
            pool=ExplodingPool(),
            gpu_count_fn=lambda _: 0,
        )


def test_documented_profile_env_registry_complete():
    expected_envs = {
        "default_env",
        "flashinfer_pip_env",
        "flashinfer_local",
        "vllm_env",
    }
    assert set(ENV_REGISTRY) == expected_envs
    assert (
        ENV_REGISTRY["flashinfer_pip_env"].python_executable
        == ENV_REGISTRY["default_env"].python_executable
    )


def test_comm_launcher_interfaces_exist():
    assert issubclass(TorchMpLauncher, MultiGpuLauncher)
    assert issubclass(NvshmemLauncher, MultiGpuLauncher)
    assert issubclass(VllmLauncher, MultiGpuLauncher)

    with pytest.raises(ValueError):
        TorchMpLauncher(0)


def test_single_gemm_exec_smoke_cuda(tmp_path: Path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the torch GEMM smoke test")

    db_path = tmp_path / "profile.db"
    gpu_name = torch.cuda.get_device_name(0)
    run_profile_batch(
        KernelKind.GEMM_SINGLE,
        [{"m": 16, "n": 16, "k": 16, "dtype": "fp16", "backend": "torch"}],
        pool=LocalGpuPool(gpus=[0]),
        db_path=db_path,
    )
    profiler_spec = find_kernel_profiler_spec(KernelKind.GEMM_SINGLE, "torch")
    table = Table(profiler_spec, db_path)
    result = table.query(
        [SingleGemmArgs(m=16, n=16, k=16, dtype=DType.FP16)],
        backend="torch",
        gpu_name=gpu_name,
    )[0]
    assert isinstance(result, ComputeMetrics)
    assert result.time_ms > 0
    assert result.tflops >= 0


def _first_visible_gpu_index() -> int:
    raw_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not raw_visible_devices:
        return 0
    first_visible_device = raw_visible_devices.split(",", maxsplit=1)[0].strip()
    if not first_visible_device:
        return 0
    if not first_visible_device.isdecimal():
        pytest.skip(
            "perf_api GEMM profiling example needs a numeric CUDA_VISIBLE_DEVICES mask"
        )
    return int(first_visible_device)
