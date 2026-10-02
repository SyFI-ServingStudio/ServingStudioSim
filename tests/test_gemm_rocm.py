"""Focused CPU tests for the MI300X GEMM ROCm backends. No GPU/DB.

Covers single_gemm rocm_scaled_mm (fp8) + torch_rocm (bf16) and batched_gemm
torch_rocm (bf16): registration, MI300X gating (NVIDIA rows untouched), runner
signatures, shape validation, and the rocprof launch-builder wiring.
"""

from __future__ import annotations

import inspect
import subprocess
import sys

import pytest

from profiling.db.args import DType
from profiling.db.registry import find_kernel_profiler_spec
from profiling.runners.gemm import rocm_scaled_mm, torch_rocm


def test_single_gemm_rocm_scaled_mm_registered_for_mi300x() -> None:
    spec = find_kernel_profiler_spec("single_gemm", "rocm_scaled_mm")
    assert spec.supports.gpus == frozenset({"MI300X"})
    assert spec.supports.compute == frozenset({DType.FP8_E4M3})
    assert spec.subprocess_env == "vllm_rocm_env"
    assert spec.runner_ref.function_name == "profile_single_gemm_scaled_mm"
    assert spec.table_name == "single_gemm"

    # The NVIDIA fp8 dense GEMM (deepgemm) is not weakened onto MI300X.
    deepgemm = find_kernel_profiler_spec("single_gemm", "deepgemm")
    assert "MI300X" not in (deepgemm.supports.gpus or frozenset())


def test_single_gemm_torch_rocm_registered_for_mi300x() -> None:
    spec = find_kernel_profiler_spec("single_gemm", "torch_rocm")
    assert spec.supports.gpus == frozenset({"MI300X"})
    assert DType.BF16 in spec.supports.compute
    assert spec.subprocess_env == "vllm_rocm_env"
    assert spec.runner_ref.function_name == "profile_single_gemm_torch_rocm"

    # The B200-gated bf16 row (torch_linear_vllm) keeps its NVIDIA-only gate.
    nvidia = find_kernel_profiler_spec("single_gemm", "torch_linear_vllm")
    assert nvidia.supports.gpus == frozenset({"NVIDIA B200"})


def test_batched_gemm_torch_rocm_registered_for_mi300x() -> None:
    spec = find_kernel_profiler_spec("batched_gemm", "torch_rocm")
    assert spec.supports.gpus == frozenset({"MI300X"})
    assert spec.supports.compute == frozenset({DType.BF16})
    assert spec.subprocess_env == "vllm_rocm_env"
    assert spec.runner_ref.function_name == "profile_batched_gemm_torch_rocm"

    nvidia = find_kernel_profiler_spec("batched_gemm", "torch_mla_q_absorb_no_rope")
    assert "MI300X" not in (nvidia.supports.gpus or frozenset())


def test_runner_signatures_match_kind_args() -> None:
    assert list(inspect.signature(rocm_scaled_mm.profile_single_gemm_scaled_mm).parameters) == [
        "m",
        "n",
        "k",
        "dtype",
    ]
    assert list(inspect.signature(torch_rocm.profile_single_gemm_torch_rocm).parameters) == [
        "m",
        "n",
        "k",
        "dtype",
    ]
    assert list(inspect.signature(torch_rocm.profile_batched_gemm_torch_rocm).parameters) == [
        "num_batches",
        "m",
        "n",
        "k",
        "dtype",
    ]


@pytest.mark.parametrize("override", [{"m": 0}, {"n": -1}, {"k": 0}, {"dtype": "bf16"}])
def test_scaled_mm_rejects_bad_shapes_and_non_fp8(override: dict) -> None:
    base = dict(m=512, n=4096, k=2048, dtype="fp8_e4m3")
    with pytest.raises(ValueError):
        rocm_scaled_mm._validate_args(**{**base, **override})


@pytest.mark.parametrize("override", [{"m": 0}, {"k": -1}, {"dtype": "fp8_e4m3"}])
def test_single_torch_rocm_rejects_bad_shapes_and_fp8(override: dict) -> None:
    base = dict(m=512, n=4096, k=2048, dtype="bf16")
    with pytest.raises(ValueError):
        torch_rocm._validate_single(**{**base, **override})


@pytest.mark.parametrize("override", [{"num_batches": 0}, {"m": 0}, {"dtype": "fp8_e4m3"}])
def test_batched_torch_rocm_rejects_bad_shapes_and_fp8(override: dict) -> None:
    base = dict(num_batches=128, m=1, n=512, k=512, dtype="bf16")
    with pytest.raises(ValueError):
        torch_rocm._validate_batched(**{**base, **override})


def test_rocprof_run_builders_registered() -> None:
    import profiling.profilers.rocprof_run as rr

    assert ("single_gemm", "rocm_scaled_mm") in rr._BUILDERS
    assert ("single_gemm", "torch_rocm") in rr._BUILDERS
    assert ("batched_gemm", "torch_rocm") in rr._BUILDERS


@pytest.mark.parametrize(
    "module",
    [
        "profiling.runners.gemm.rocm_scaled_mm",
        "profiling.runners.gemm.torch_rocm",
    ],
)
def test_import_light(module: str) -> None:
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; import {module} as r; "
            "print('torch' in sys.modules, 'vllm' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False False"
