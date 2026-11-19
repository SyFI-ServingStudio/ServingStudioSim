"""CPU tests for the CUDA static FP8 quant backend: the variants it refuses."""

from __future__ import annotations

import pytest

from profiling.runners.elementwise.static_fp8_quant_vllm_cuda import (
    profile_static_fp8_quant_vllm_cuda,
)
from profiling.runners.exceptions import ProfilerNotImplemented


def test_gated_and_non_bf16_rows_are_refused_not_timed():
    # The gate fusion exists only under Inductor: timing the plain CUDA quant
    # for a gated row would store a time that omits the gate.
    with pytest.raises(ProfilerNotImplemented):
        profile_static_fp8_quant_vllm_cuda(8, 32768, True, "bf16")
    with pytest.raises(ProfilerNotImplemented):
        profile_static_fp8_quant_vllm_cuda(8, 32768, False, "fp16")
    with pytest.raises(ValueError):
        profile_static_fp8_quant_vllm_cuda(0, 32768, False, "bf16")
