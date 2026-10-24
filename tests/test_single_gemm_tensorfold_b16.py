"""CPU checks for the ``single_gemm`` ``tensorfold_b16`` backend.

GPU timing and the check against TensorFold's b16_kernel run through the public
``python -m profiling run`` smoke on a GB10.
"""

from __future__ import annotations

import pytest

from profiling.db.args import DType
from profiling.db.registry import find_kernel_profiler_spec
from profiling.runners.gemm.tensorfold_b16 import (
    profile_single_gemm_tensorfold_b16,
    profile_single_gemm_tensorfold_b16_prompt,
)


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        # The row's n is the pair's total width, split into two equal weights.
        ({"m": 16, "n": 95, "k": 5120, "dtype": "bf16"}, "even"),
        # b16_kernel reads bf16/fp16 rows; an fp8 key would name a different GEMM.
        ({"m": 16, "n": 96, "k": 5120, "dtype": "fp8_e4m3"}, "bf16"),
        # Below 8-element K chunks matmul_pair falls back to two launches.
        ({"m": 16, "n": 96, "k": 5124, "dtype": "bf16"}, "multiple of 8"),
        ({"m": 0, "n": 96, "k": 5120, "dtype": "bf16"}, "m >= 1"),
    ],
)
def test_runner_rejects_specs_before_gpu_work(spec, message):
    with pytest.raises(ValueError, match=message):
        profile_single_gemm_tensorfold_b16(**spec)


def test_backend_is_bf16_in_tensorfold_env():
    spec = find_kernel_profiler_spec("single_gemm", "tensorfold_b16")
    assert spec.subprocess_env == "tensorfold_env"
    assert spec.supports.compute == frozenset({DType.BF16})


def test_prompt_runner_requires_whole_64_input_stages():
    # Below 64-input K stages prompt_pair silently runs two prompt() launches,
    # which is not the pair launch this backend names.
    with pytest.raises(ValueError, match="multiple of 64"):
        profile_single_gemm_tensorfold_b16_prompt(m=1024, n=96, k=5128, dtype="bf16")
    with pytest.raises(ValueError, match="even"):
        profile_single_gemm_tensorfold_b16_prompt(m=1024, n=97, k=5120, dtype="bf16")
