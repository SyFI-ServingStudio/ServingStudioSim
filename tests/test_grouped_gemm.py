from __future__ import annotations

import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest

from profiling import perf_api
from profiling.db import (
    DType,
    MetricFamily,
    ProfileRow,
    Table,
    find_kernel_profiler_spec,
    known_backends,
)
from profiling.db.batch import coerce_args
from profiling.db.table import MissingEntry
from profiling.kernels.grouped_gemm import KIND, GroupedGemmArgs
from profiling.runners.metrics import ComputeMetrics


def test_kind_wire_string_is_snake_case():
    assert KIND == "grouped_gemm"


def test_args_field_set_and_declaration_order():
    # Wire schema shared with Rust GroupedGemmKernel enumerate (minus `backend`).
    assert [f.name for f in fields(GroupedGemmArgs)] == [
        "n",
        "k",
        "dtype",
        "num_local_experts",
        "per_group_batches",
    ]


def test_coerce_args_turns_json_list_into_hashable_tuple():
    # The Rust facade ships per_group_batches as a JSON list; coercion must
    # rebuild the declared tuple[int, ...] so the frozen args stay hashable and
    # the DType is normalized.
    spec = {
        "n": 4096,
        "k": 8192,
        "dtype": "bf16",
        "num_local_experts": 3,
        "per_group_batches": [3, 5, 0],
    }
    args = coerce_args(GroupedGemmArgs, spec)
    assert args.per_group_batches == (3, 5, 0)
    assert isinstance(args.per_group_batches, tuple)
    assert args.dtype is DType.BF16
    # Hashable identity (a frozen dataclass with a list field would raise here).
    assert hash(args) == hash(coerce_args(GroupedGemmArgs, spec))


def test_register_spec_shape():
    spec = find_kernel_profiler_spec("grouped_gemm", "torch")
    assert spec.kernel_kind == KIND
    assert spec.table_name == KIND
    assert spec.backend == "torch"
    assert spec.metric_family is MetricFamily.COMPUTE
    assert spec.args_schema is GroupedGemmArgs
    assert spec.runner_ref.module_name == "profiling.runners.gemm.torch"
    assert spec.runner_ref.function_name == "profile_grouped_gemm"


def test_deepgemm_backend_registered_sharing_table_and_args():
    spec = find_kernel_profiler_spec("grouped_gemm", "deepgemm")
    assert spec.kernel_kind == KIND
    assert spec.table_name == KIND  # shares the torch backend's table + schema
    assert spec.backend == "deepgemm"
    assert spec.metric_family is MetricFamily.COMPUTE
    assert spec.args_schema is GroupedGemmArgs
    assert spec.subprocess_env is None  # default env (deep_gemm is a project dep)
    assert spec.runner_ref.module_name == "profiling.runners.gemm.deepgemm"
    assert spec.runner_ref.function_name == "profile_grouped_gemm"


def test_known_backends_has_torch_and_deepgemm():
    assert set(known_backends("grouped_gemm")) == {"torch", "deepgemm"}


def test_facade_functions_exist():
    assert hasattr(perf_api, "get_grouped_gemm_times")
    assert hasattr(perf_api, "count_missing_grouped_gemm")


def test_importing_kernel_module_does_not_import_runner():
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import profiling.kernels.grouped_gemm; "
            "print(any(m.startswith('profiling.runners.gemm.') for m in sys.modules))"
        ),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == "False"


def test_perf_api_query_path_round_trips_tuple_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(perf_api, "DB_PATH", tmp_path / "profile.db")
    # Spec ships per_group_batches as a JSON list (as the Rust facade does).
    spec = {
        "n": 32,
        "k": 64,
        "dtype": "fp16",
        "num_local_experts": 3,
        "per_group_batches": [4, 2, 0],
    }

    missing = perf_api.get_grouped_gemm_times([spec], backend="torch", gpu_name="TestGPU")[0]
    assert isinstance(missing, MissingEntry)
    assert perf_api.count_missing_grouped_gemm([spec], backend="torch", gpu_name="TestGPU") == 1

    profiler_spec = find_kernel_profiler_spec("grouped_gemm", "torch")
    table = Table(profiler_spec, perf_api.DB_PATH)
    args = GroupedGemmArgs(
        n=32, k=64, dtype=DType.FP16, num_local_experts=3, per_group_batches=(4, 2, 0)
    )
    table.insert(
        [
            ProfileRow(
                args=args,
                metrics=ComputeMetrics(
                    time_ms=1.5,
                    tflops=0.01,
                    memory_bandwidth_gbps=0.02,
                    energy_j=0.0,
                ),
                gpu_name="TestGPU",
                backend="torch",
            )
        ]
    )

    # The list-shaped spec must hit the row keyed by the tuple JSON column.
    result = perf_api.get_grouped_gemm_times([spec], backend="torch", gpu_name="TestGPU")[0]
    assert isinstance(result, ComputeMetrics)
    assert result.time_ms == 1.5
    assert perf_api.count_missing_grouped_gemm([spec], backend="torch", gpu_name="TestGPU") == 0
