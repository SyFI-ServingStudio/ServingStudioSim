"""Focused L1 registry/schema tests for exact block-scale grouped GEMM."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

from profiling import perf_api
from profiling.db.args import DType, Fp8BlockscaleGroupedGemmArgs
from profiling.db.batch import args_to_spec, coerce_args
from profiling.db.registry import MetricFamily, find_kernel_profiler_spec, known_backends
from profiling.db.table import MissingEntry, ProfileRow, Table
from profiling.kernels.fp8_blockscale_grouped_gemm import KIND
from profiling.runners.metrics import ComputeMetrics


def _public_spec() -> dict:
    return {
        "n": 3072,
        "k": 4096,
        "dtype": "fp8_e4m3",
        "num_local_experts": 4,
        "num_input_tokens": 64,
        "experts_per_token": 8,
        "per_group_batches": [6, 3, 0, 5],
    }


def test_args_field_order_dtype_and_tuple_coercion():
    assert [field.name for field in fields(Fp8BlockscaleGroupedGemmArgs)] == [
        "n",
        "k",
        "dtype",
        "num_local_experts",
        "num_input_tokens",
        "experts_per_token",
        "per_group_batches",
    ]

    arguments = coerce_args(Fp8BlockscaleGroupedGemmArgs, _public_spec())
    assert arguments == Fp8BlockscaleGroupedGemmArgs(
        n=3072,
        k=4096,
        dtype=DType.FP8_E4M3,
        num_local_experts=4,
        num_input_tokens=64,
        experts_per_token=8,
        per_group_batches=(6, 3, 0, 5),
    )
    assert isinstance(arguments.per_group_batches, tuple)
    assert hash(arguments) == hash(coerce_args(Fp8BlockscaleGroupedGemmArgs, _public_spec()))


def test_registry_list_backend_and_capability_contract():
    profiler_spec = find_kernel_profiler_spec(KIND, "flashinfer_trtllm")

    assert KIND == "fp8_blockscale_grouped_gemm"
    assert known_backends(KIND) == ["flashinfer_trtllm"]
    assert profiler_spec.kernel_kind == KIND
    assert profiler_spec.table_name == KIND
    assert profiler_spec.backend == "flashinfer_trtllm"
    assert profiler_spec.args_schema is Fp8BlockscaleGroupedGemmArgs
    assert profiler_spec.metric_family is MetricFamily.COMPUTE
    assert profiler_spec.subprocess_env == "flashinfer_pip_env"
    assert profiler_spec.runner_ref.module_name == (
        "profiling.runners.gemm.flashinfer_trtllm_blockscale"
    )
    assert profiler_spec.runner_ref.function_name == (
        "profile_fp8_blockscale_grouped_gemm_flashinfer_trtllm"
    )
    assert profiler_spec.supports.allows(DType.FP8_E4M3, gpu="NVIDIA H100")
    assert profiler_spec.supports.allows(DType.FP8_E4M3, gpu="NVIDIA H200")
    assert not profiler_spec.supports.allows(DType.BF16, gpu="NVIDIA H200")
    assert not profiler_spec.supports.allows(DType.FP8_E4M3, gpu="NVIDIA B200")


def test_profiling_list_json_exposes_the_public_contract():
    completed = subprocess.run(
        [sys.executable, "-m", "profiling", "list", "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    matching = [profiler for profiler in payload["profilers"] if profiler["kernel_kind"] == KIND]
    assert matching == [
        {
            "args": (
                "n,k,dtype,num_local_experts,num_input_tokens,experts_per_token,per_group_batches"
            ),
            "backend": "flashinfer_trtllm",
            "count_fn": "count_missing_fp8_blockscale_grouped_gemm",
            "get_fn": "get_fp8_blockscale_grouped_gemm_times",
            "kernel_kind": KIND,
            "metric_family": "compute",
            "subprocess_env": "flashinfer_pip_env",
            "table": KIND,
        }
    ]


def test_kernel_registry_import_is_lazy_for_runner_torch_and_flashinfer():
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import profiling.kernels.fp8_blockscale_grouped_gemm; "
            "print('profiling.runners.gemm.flashinfer_trtllm_blockscale' "
            "in sys.modules, 'torch' in sys.modules, 'flashinfer' in sys.modules)"
        ),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == "False False False"


def test_generated_facade_symbols_exist():
    assert hasattr(perf_api, "get_fp8_blockscale_grouped_gemm_times")
    assert hasattr(perf_api, "count_missing_fp8_blockscale_grouped_gemm")


def test_temp_db_query_insert_roundtrip(tmp_path: Path, monkeypatch):
    database_path = tmp_path / "profile.db"
    monkeypatch.setattr(perf_api, "DB_PATH", database_path)
    public_spec = _public_spec()

    missing = perf_api.get_fp8_blockscale_grouped_gemm_times(
        [public_spec],
        backend="flashinfer_trtllm",
        gpu_name="TestGPU",
    )[0]
    assert isinstance(missing, MissingEntry)
    assert (
        perf_api.count_missing_fp8_blockscale_grouped_gemm(
            [public_spec],
            backend="flashinfer_trtllm",
            gpu_name="TestGPU",
        )
        == 1
    )

    profiler_spec = find_kernel_profiler_spec(KIND, "flashinfer_trtllm")
    arguments = coerce_args(Fp8BlockscaleGroupedGemmArgs, public_spec)
    Table(profiler_spec, database_path).insert(
        [
            ProfileRow(
                args=arguments,
                metrics=ComputeMetrics(
                    time_ms=0.092,
                    tflops=0.12,
                    memory_bandwidth_gbps=0.34,
                    energy_j=0.01,
                ),
                gpu_name="TestGPU",
                backend="flashinfer_trtllm",
            )
        ]
    )

    result = perf_api.get_fp8_blockscale_grouped_gemm_times(
        [public_spec],
        backend="flashinfer_trtllm",
        gpu_name="TestGPU",
    )[0]
    assert isinstance(result, ComputeMetrics)
    assert result.time_ms == 0.092
    assert (
        perf_api.count_missing_fp8_blockscale_grouped_gemm(
            [public_spec],
            backend="flashinfer_trtllm",
            gpu_name="TestGPU",
        )
        == 0
    )


def test_registered_list_runner_receives_only_schema_kwargs(monkeypatch):
    from profiling.runners.gemm import flashinfer_trtllm_blockscale as runner

    captured_kwargs = []
    expected_metrics = ComputeMetrics(
        time_ms=1.0,
        tflops=2.0,
        memory_bandwidth_gbps=3.0,
        energy_j=4.0,
    )

    def mock_runner(**kwargs):
        captured_kwargs.append(kwargs)
        return expected_metrics

    monkeypatch.setattr(
        runner,
        "profile_fp8_blockscale_grouped_gemm_flashinfer_trtllm",
        mock_runner,
    )
    profiler_spec = find_kernel_profiler_spec(KIND, "flashinfer_trtllm")
    typed_arguments = coerce_args(Fp8BlockscaleGroupedGemmArgs, _public_spec())
    results = profiler_spec.load_list_runner()([args_to_spec(typed_arguments)])

    assert len(results) == 1
    assert results[0].metrics == expected_metrics
    assert results[0].error is None
    assert captured_kwargs == [
        {
            "n": 3072,
            "k": 4096,
            "dtype": "fp8_e4m3",
            "num_local_experts": 4,
            "num_input_tokens": 64,
            "experts_per_token": 8,
            "per_group_batches": (6, 3, 0, 5),
        }
    ]
