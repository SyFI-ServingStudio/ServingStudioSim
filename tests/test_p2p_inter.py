"""Unit tests for the ``p2p_inter`` L1 kernel kind.

Imports the per-kernel module DIRECTLY so they pass independent of the barrel;
no shared ``profile.db``, no GPU / multi-process work.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from profiling import perf_api
from profiling.db.args import DType
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import KernelProfilerSpec, MetricFamily, RunnerRef
from profiling.kernels import p2p_inter as p2p_inter_kernel
from profiling.kernels.p2p_inter import KIND, P2pInterArgs


def test_args_field_contract_matches_runner_kwargs():
    field_names = [field.name for field in P2pInterArgs.__dataclass_fields__.values()]
    assert field_names == ["message_size_bytes", "dtype", "fabric"]


def test_args_are_frozen_and_dtype_coerces():
    args = P2pInterArgs(message_size_bytes=1 << 20, dtype=DType.BF16, fabric="infiniband")
    assert args.message_size_bytes == 1 << 20
    assert args.dtype is DType.BF16
    assert args.fabric == "infiniband"
    with pytest.raises(Exception):
        args.message_size_bytes = 4  # frozen


def test_kind_wire_string_and_table_stem():
    assert KIND == "p2p_inter"
    assert p2p_inter_kernel.KIND == "p2p_inter"


# Both backends resolve to the analytical lookup-table runner (no real comm).
@pytest.mark.parametrize("backend", ["nccl", "nvshmem"])
def test_register_call_built_a_comm_spec(backend: str):
    module_name = "profiling.runners.comm.p2p_inter"
    spec = KernelProfilerSpec(
        kernel_kind=KIND,
        backend=backend,
        runner_ref=RunnerRef(module_name=module_name, function_name="profile_p2p"),
        table_name=KIND,
        args_schema=P2pInterArgs,
        metric_family=MetricFamily.COMM,
        batch_outlier_policy=BatchOutlierPolicy(),
        gpu_count_fn=lambda s: 1,
    )
    assert spec.kernel_kind == "p2p_inter"
    assert spec.backend == backend
    assert spec.table_name == spec.kernel_kind  # facade-stem invariant
    assert spec.metric_family is MetricFamily.COMM
    assert spec.runner_ref.module_name == module_name
    assert spec.runner_ref.function_name == "profile_p2p"
    assert spec.gpu_count_fn({}) == 1


def test_profiled_curve_is_interpolated_not_measured():
    from profiling.runners.comm.p2p_inter import _compute_inter_device_p2p_time_ms

    # On-grid points return the table value exactly (us -> ms).
    assert _compute_inter_device_p2p_time_ms(1024, "NVIDIA H100") == pytest.approx(0.03019)
    assert _compute_inter_device_p2p_time_ms(1 << 30, "B200") == pytest.approx(21.759)
    # Between grid points: linear interpolation, monotonic in size.
    mid = _compute_inter_device_p2p_time_ms(100_000, "H200")
    assert 0.03092 < mid < 0.03210


def test_fallback_bandwidth_model_for_non_profiled_gpu():
    from profiling.runners.comm.p2p_inter import _compute_inter_device_p2p_time_ms

    # A100 = 22 GB/s; sub-32KB transfers are padded to the 32KB floor.
    assert _compute_inter_device_p2p_time_ms(1024, "NVIDIA A100") == pytest.approx(
        (32 * 1024) / (22.0 * 1e9) * 1000.0
    )
    # Unknown GPU falls back to the default 44 GB/s.
    assert _compute_inter_device_p2p_time_ms(1 << 20, "MysteryGPU") == pytest.approx(
        (1 << 20) / (44.0 * 1e9) * 1000.0
    )


def test_facade_functions_exist():
    assert hasattr(perf_api, "get_p2p_inter_times")
    assert hasattr(perf_api, "count_missing_p2p_inter")


def test_importing_kernel_module_does_not_eager_import_runner():
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import profiling.kernels.p2p_inter; "
            "print('profiling.runners.comm.p2p_inter' in sys.modules)"
        ),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == "False"
