"""Unit tests for the ``rms_norm`` L1 kernel kind.

Parallel-safety note: these import the per-kernel module DIRECTLY
(``import profiling.kernels.rms_norm``) so they pass before the shared barrel
``profiling/kernels/__init__.py`` is wired by the orchestrator. They do not
depend on the registry being fully loaded, do not touch the shared
``profile.db``, and do not run any GPU work.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from profiling.db.args import DType
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import KernelProfilerSpec, MetricFamily, RunnerRef
from profiling.kernels import rms_norm as rms_norm_kernel
from profiling.kernels.rms_norm import KIND, RmsNormArgs


def test_args_field_contract_matches_runner_kwargs():
    # Field set / order is the load-bearing contract (A5): it must equal the
    # torch runner kwargs (m, hidden, dtype) and the Rust enumerate fields
    # (minus `backend`).
    field_names = [field.name for field in RmsNormArgs.__dataclass_fields__.values()]
    assert field_names == ["m", "hidden", "dtype"]


def test_args_are_frozen_and_dtype_coerces():
    # KernelArgs subclasses are frozen identities; DType.from_value coercion is
    # exercised by the runner, so confirm the canonical values round-trip.
    args = RmsNormArgs(m=512, hidden=4096, dtype=DType.BF16)
    assert args.m == 512
    assert args.hidden == 4096
    assert args.dtype is DType.BF16
    with pytest.raises(Exception):
        args.m = 8  # frozen
    assert DType.from_value("torch.bfloat16") is DType.BF16


def test_kind_wire_string_and_table_stem():
    # KIND must equal the Rust KernelSpec::KIND and the registry table_name so
    # the cross-language facade name resolves (registry validator enforces
    # table_name == kernel_kind).
    assert KIND == "rms_norm"


def test_register_call_built_a_flashinfer_compute_spec():
    # The module's import-time register(...) side effect must produce a spec we
    # can introspect without loading the full registry barrel. Rebuild the
    # exact spec the module registers and assert its shape.
    spec = KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer",
        runner_ref=RunnerRef(
            module_name="profiling.runners.norm.flashinfer",
            function_name="profile_rms_norm",
        ),
        table_name=KIND,
        args_schema=RmsNormArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
    )
    assert spec.kernel_kind == "rms_norm"
    assert spec.backend == "flashinfer"
    assert spec.table_name == spec.kernel_kind  # facade-stem invariant
    assert spec.metric_family is MetricFamily.COMPUTE
    assert spec.runner_ref.module_name == "profiling.runners.norm.flashinfer"
    assert spec.runner_ref.function_name == "profile_rms_norm"
    # The module object exposes the same KIND constant.
    assert rms_norm_kernel.KIND == "rms_norm"


def test_importing_kernel_module_does_not_eager_import_runner():
    # Lazy-import invariant (L1 design §3.2.1): importing the per-kernel module
    # must NOT pull in the flashinfer runner module (which would import
    # torch/cuda/flashinfer in the main process).
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import profiling.kernels.rms_norm; "
            "print('profiling.runners.norm.flashinfer' in sys.modules)"
        ),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == "False"
