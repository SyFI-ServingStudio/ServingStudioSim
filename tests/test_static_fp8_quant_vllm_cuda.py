"""CPU tests for the CUDA static FP8 quant backend: registration and refused variants."""

from __future__ import annotations

import pytest

from profiling.db.registry import find_kernel_profiler_spec, find_table
from profiling.runners.elementwise.static_fp8_quant_vllm_cuda import (
    profile_static_fp8_quant_vllm_cuda,
)
from profiling.runners.exceptions import ProfilerNotImplemented


def test_backend_is_registered_beside_inductor():
    spec = find_kernel_profiler_spec("static_fp8_quant", "vllm_cuda")
    assert spec.runner_ref.function_name == "profile_static_fp8_quant_vllm_cuda"
    assert find_table("static_fp8_quant", "vllm_cuda") == find_table(
        "static_fp8_quant", "vllm_inductor"
    )


def test_gated_and_non_bf16_rows_are_refused_not_timed():
    with pytest.raises(ProfilerNotImplemented):
        profile_static_fp8_quant_vllm_cuda(8, 32768, True, "bf16")
    with pytest.raises(ProfilerNotImplemented):
        profile_static_fp8_quant_vllm_cuda(8, 32768, False, "fp16")
    with pytest.raises(ValueError):
        profile_static_fp8_quant_vllm_cuda(0, 32768, False, "bf16")
