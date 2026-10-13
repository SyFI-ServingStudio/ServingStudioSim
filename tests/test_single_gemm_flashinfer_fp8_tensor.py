"""CPU checks for the ``single_gemm`` ``flashinfer_fp8_tensor`` backend.

GPU timing and the numerical check against the production call are covered by
the public ``python -m profiling run`` smoke on a B200; these tests guard the
pieces that can regress without a GPU.
"""

from __future__ import annotations

import pytest

from profiling.runners.exceptions import KernelLaunchFailed
from profiling.runners.gemm.flashinfer_fp8_tensor import (
    check_against_reference,
    profile_single_gemm_flashinfer_fp8_tensor,
)


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"m": 8, "n": 16384, "k": 5120, "dtype": "bf16"}, "dtype=fp8_e4m3"),
        ({"m": 8, "n": 16384, "k": 5120, "dtype": "nvfp4_e2m1"}, "dtype=fp8_e4m3"),
        ({"m": 0, "n": 16384, "k": 5120, "dtype": "fp8_e4m3"}, "m >= 1"),
        ({"m": 8, "n": 16392, "k": 5120, "dtype": "fp8_e4m3"}, "multiples of 16"),
        ({"m": 8, "n": 16384, "k": 5128, "dtype": "fp8_e4m3"}, "multiples of 16"),
    ],
)
def test_runner_rejects_specs_before_gpu_work(spec, message):
    # A BF16 or NVFP4 row must never be timed as per-tensor FP8, and shapes
    # cuBLASLt FP8 cannot address must fail before any device allocation.
    with pytest.raises(ValueError, match=message):
        profile_single_gemm_flashinfer_fp8_tensor(**spec)


def test_reference_applies_both_per_tensor_scales():
    # The oracle must fail an output that drops a scale: otherwise a production
    # call that ignored scale_a or scale_b would pass the correctness gate.
    torch = pytest.importorskip("torch")
    gen = torch.Generator().manual_seed(0)
    a = torch.randn(32, 64, generator=gen).to(torch.float8_e4m3fn)
    w = torch.randn(48, 64, generator=gen).to(torch.float8_e4m3fn)
    unscaled = (a.float() @ w.float().t()).to(torch.bfloat16)
    from profiling.runners.gemm.flashinfer_fp8_tensor import _A_SCALE, _W_SCALE

    check_against_reference(torch, a, w, unscaled * (_A_SCALE * _W_SCALE))
    with pytest.raises(KernelLaunchFailed):
        check_against_reference(torch, a, w, unscaled)
