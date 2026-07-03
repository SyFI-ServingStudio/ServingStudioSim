"""Unit tests for the ``p2p_intra`` L1 kernel kind.

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
from profiling.kernels import p2p_intra as p2p_intra_kernel
from profiling.kernels.p2p_intra import KIND, P2pIntraArgs


def test_args_field_contract_matches_runner_kwargs():
    # Field set / order must equal the per-spec kwargs the list runner reads out of
    # each spec dict (message_size_bytes, dtype, fabric) and the Rust enumerate fields
    # (minus `backend`). No num_gpus: p2p is always 2 endpoints. The runner is now
    # list-native (`profile_p2p_batch(list[dict])`), but each element still carries
    # exactly these schema fields.
    field_names = [field.name for field in P2pIntraArgs.__dataclass_fields__.values()]
    assert field_names == ["message_size_bytes", "dtype", "fabric"]


def test_args_are_frozen_and_dtype_coerces():
    args = P2pIntraArgs(message_size_bytes=1 << 20, dtype=DType.BF16, fabric="nvlink")
    assert args.message_size_bytes == 1 << 20
    assert args.dtype is DType.BF16
    assert args.fabric == "nvlink"
    with pytest.raises(Exception):
        args.message_size_bytes = 4  # frozen


def test_kind_wire_string_and_table_stem():
    assert KIND == "p2p_intra"
    assert p2p_intra_kernel.KIND == "p2p_intra"


@pytest.mark.parametrize(
    "backend, module_name",
    [
        ("nccl", "profiling.runners.comm.p2p"),
        ("nvshmem", "profiling.runners.comm.p2p_nvshmem"),
    ],
)
def test_register_call_built_a_comm_spec(backend: str, module_name: str):
    spec = KernelProfilerSpec(
        kernel_kind=KIND,
        backend=backend,
        runner_ref=RunnerRef(module_name=module_name, function_name="profile_p2p_batch"),
        table_name=KIND,
        args_schema=P2pIntraArgs,
        metric_family=MetricFamily.COMM,
        batch_outlier_policy=BatchOutlierPolicy(),
        gpu_count_fn=lambda s: 2,
        list_native=True,
    )
    assert spec.kernel_kind == "p2p_intra"
    assert spec.backend == backend
    assert spec.table_name == spec.kernel_kind  # facade-stem invariant
    assert spec.metric_family is MetricFamily.COMM
    assert spec.runner_ref.module_name == module_name
    # Comm runners are list-native: the worker calls the batch entry once per chunk.
    assert spec.runner_ref.function_name == "profile_p2p_batch"
    assert spec.list_native is True
    assert spec.gpu_count_fn({}) == 2


def test_facade_functions_exist():
    assert hasattr(perf_api, "get_p2p_intra_times")
    assert hasattr(perf_api, "count_missing_p2p_intra")


def test_importing_kernel_module_does_not_eager_import_runner():
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import profiling.kernels.p2p_intra; "
            "print(any(m in sys.modules for m in "
            "('profiling.runners.comm.p2p', 'profiling.runners.comm.p2p_nvshmem')))"
        ),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == "False"
