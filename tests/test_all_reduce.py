"""Unit tests for the ``all_reduce`` L1 kernel kind.

Parallel-safety note: these import the per-kernel module DIRECTLY
(``import profiling.kernels.all_reduce``) so they pass before the shared barrel
``profiling/kernels/__init__.py`` is wired. They do not depend on the registry
being fully loaded, do not touch the shared ``profile.db``, and do not run any
GPU / multi-process work.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from profiling.db.args import DType
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import KernelProfilerSpec, MetricFamily, RunnerRef
from profiling.kernels import all_reduce as all_reduce_kernel
from profiling.kernels.all_reduce import KIND, AllReduceArgs


def test_args_field_contract_matches_runner_kwargs():
    # Field set / order is the load-bearing contract (A5): it must equal the
    # runner kwargs (num_gpus, message_size_bytes, dtype, fabric) and the Rust
    # enumerate fields (minus `backend`).
    field_names = [field.name for field in AllReduceArgs.__dataclass_fields__.values()]
    assert field_names == ["num_gpus", "message_size_bytes", "dtype", "fabric"]


def test_args_are_frozen_and_dtype_coerces():
    args = AllReduceArgs(
        num_gpus=8, message_size_bytes=1 << 20, dtype=DType.BF16, fabric="nvlink"
    )
    assert args.num_gpus == 8
    assert args.message_size_bytes == 1 << 20
    assert args.dtype is DType.BF16
    assert args.fabric == "nvlink"
    with pytest.raises(Exception):
        args.num_gpus = 4  # frozen
    assert DType.from_value("float16") is DType.FP16


def test_kind_wire_string_and_table_stem():
    # KIND must equal the Rust KernelSpec::KIND and the registry table_name so
    # the cross-language facade name resolves.
    assert KIND == "all_reduce"
    assert all_reduce_kernel.KIND == "all_reduce"


@pytest.mark.parametrize(
    "backend, module_name",
    [
        ("nccl", "profiling.runners.comm.nccl"),
        ("nvshmem", "profiling.runners.comm.nvshmem"),
    ],
)
def test_register_call_built_a_comm_spec(backend: str, module_name: str):
    # Rebuild the exact spec the module registers per backend and assert shape.
    spec = KernelProfilerSpec(
        kernel_kind=KIND,
        backend=backend,
        runner_ref=RunnerRef(module_name=module_name, function_name="profile_all_reduce"),
        table_name=KIND,
        args_schema=AllReduceArgs,
        metric_family=MetricFamily.COMM,
        batch_outlier_policy=BatchOutlierPolicy(),
        gpu_count_fn=lambda s: int(s["num_gpus"]),
    )
    assert spec.kernel_kind == "all_reduce"
    assert spec.backend == backend
    assert spec.table_name == spec.kernel_kind  # facade-stem invariant
    assert spec.metric_family is MetricFamily.COMM
    assert spec.runner_ref.module_name == module_name
    assert spec.runner_ref.function_name == "profile_all_reduce"
    assert spec.gpu_count_fn({"num_gpus": 4}) == 4


def test_importing_kernel_module_does_not_eager_import_runner():
    # Lazy-import invariant (L1 design §3.2.1): importing the per-kernel module
    # must NOT pull in either comm runner (which would import torch / a
    # multi-process launcher in the main process).
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import profiling.kernels.all_reduce; "
            "print(any(m in sys.modules for m in "
            "('profiling.runners.comm.nccl', 'profiling.runners.comm.nvshmem')))"
        ),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == "False"
