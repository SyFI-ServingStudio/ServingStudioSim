"""Unit tests for the ``flashinfer_attn_decode`` L1 kernel kind.

Parallel-safety note (mirrors ``test_elementwise.py``): these import the
per-kernel module DIRECTLY so they pass before the shared barrel
``profiling/kernels/__init__.py`` is wired. They do not depend on the registry
being fully loaded, do not touch the shared ``profile.db``, and run no GPU work.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from profiling.db.args import DType
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import KernelProfilerSpec, MetricFamily, RunnerRef
from profiling.kernels import flashinfer_attn_decode as decode_kernel
from profiling.kernels.flashinfer_attn_decode import KIND, FlashinferAttnDecodeArgs

_BACKENDS = ("fa2", "fa3", "trt", "cudnn")
_RUNNER_MODULE = "profiling.runners.attention.flashinfer_decode"


def test_args_field_contract_matches_runner_kwargs():
    # Field set / order is the load-bearing contract (A5): Config dims first,
    # then the 2D decode sweep coords (batch_size, total_tokens). batch is a real
    # axis; total_tokens is total kv (runner derives avg_len = total // batch).
    field_names = [f.name for f in FlashinferAttnDecodeArgs.__dataclass_fields__.values()]
    assert field_names == [
        "num_qo_heads",
        "num_kv_heads",
        "head_dim",
        "q_dtype",
        "kv_dtype",
        "o_dtype",
        "batch_size",
        "total_tokens",
    ]


def test_args_are_frozen():
    args = FlashinferAttnDecodeArgs(
        num_qo_heads=32,
        num_kv_heads=8,
        head_dim=128,
        q_dtype=DType.BF16,
        kv_dtype=DType.BF16,
        o_dtype=DType.BF16,
        batch_size=64,
        total_tokens=262144,
    )
    assert args.batch_size == 64
    assert args.total_tokens == 262144
    with pytest.raises(Exception):
        args.batch_size = 8  # frozen


def test_kind_wire_string_and_table_stem():
    assert KIND == "flashinfer_attn_decode"
    assert decode_kernel.KIND == "flashinfer_attn_decode"


def test_register_call_built_one_compute_spec_per_backend():
    for backend in _BACKENDS:
        spec = KernelProfilerSpec(
            kernel_kind=KIND,
            backend=backend,
            runner_ref=RunnerRef(
                module_name=_RUNNER_MODULE,
                function_name=f"profile_flashinfer_attn_decode_{backend}",
            ),
            table_name=KIND,
            args_schema=FlashinferAttnDecodeArgs,
            metric_family=MetricFamily.COMPUTE,
            batch_outlier_policy=BatchOutlierPolicy(),
            subprocess_env="flashinfer_pip_env",
        )
        assert spec.kernel_kind == "flashinfer_attn_decode"
        assert spec.backend == backend
        assert spec.table_name == spec.kernel_kind  # facade-stem invariant
        assert spec.metric_family is MetricFamily.COMPUTE
        assert spec.runner_ref.module_name == _RUNNER_MODULE
        assert spec.runner_ref.function_name == f"profile_flashinfer_attn_decode_{backend}"


def test_importing_kernel_module_does_not_eager_import_runner():
    # Lazy-import invariant (L1 design §3.2.1): importing the per-kernel module
    # must NOT pull in the runner module (which imports flashinfer/torch).
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import profiling.kernels.flashinfer_attn_decode; "
            "print('profiling.runners.attention.flashinfer_decode' in sys.modules)"
        ),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == "False"
