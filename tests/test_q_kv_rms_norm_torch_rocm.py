"""Focused CPU tests for the MI300X q_kv_rms_norm torch_rocm backend. No GPU/DB."""

from __future__ import annotations

import inspect
import subprocess
import sys

import pytest

from profiling.db.registry import find_kernel_profiler_spec
from profiling.runners.attention import q_kv_rms_norm_torch_rocm as rocm


def test_torch_rocm_backend_registered_for_mi300x() -> None:
    spec = find_kernel_profiler_spec("q_kv_rms_norm", "torch_rocm")
    assert spec.supports.gpus == frozenset({"MI300X"})
    assert spec.subprocess_env == "vllm_rocm_env"
    assert spec.runner_ref.function_name == "profile_q_kv_rms_norm_torch_rocm"

    nvidia = find_kernel_profiler_spec("q_kv_rms_norm", "vllm_triton")
    assert "MI300X" not in nvidia.supports.gpus


def test_signature_matches_kind_args() -> None:
    params = list(inspect.signature(rocm.profile_q_kv_rms_norm_torch_rocm).parameters)
    assert params == ["num_tokens", "q_dim", "kv_dim", "rms_eps", "dtype"]


@pytest.mark.parametrize(
    "override",
    [{"q_dim": 2048}, {"kv_dim": 256}, {"dtype": "fp16"}, {"num_tokens": 0}],
)
def test_rejects_shapes_outside_the_measured_path(override: dict) -> None:
    base = dict(num_tokens=512, q_dim=1536, kv_dim=512, rms_eps=1e-5, dtype="bf16")
    with pytest.raises(ValueError):
        rocm._validate_args(**{**base, **override})


def test_dispatches_per_launch_is_one() -> None:
    # Measured on the MI300X image: the fused call is one kernel dispatch/call.
    assert rocm._DISPATCHES_PER_LAUNCH == 1


def test_rocprof_run_builder_registered() -> None:
    import profiling.profilers.rocprof_run as rr

    assert ("q_kv_rms_norm", "torch_rocm") in rr._BUILDERS


def test_import_light() -> None:
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import profiling.runners.attention.q_kv_rms_norm_torch_rocm as r; "
            "print('torch' in sys.modules, 'vllm' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False False"
