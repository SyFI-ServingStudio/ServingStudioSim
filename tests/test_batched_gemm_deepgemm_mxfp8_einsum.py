"""CPU checks for the ``batched_gemm`` ``deepgemm_mxfp8_einsum_grouped_o_proj`` backend.

GPU timing and the check of the production DeepGEMM output against the
dequantized reference run in the public ``python -m profiling run`` smoke on a
B200; these tests guard the pieces that can regress without a GPU.
"""

from __future__ import annotations

import pytest

from profiling.db.args import DType
from profiling.db.registry import find_kernel_profiler_spec
from profiling.runners.gemm.deepgemm_mxfp8_einsum import (
    BACKEND,
    dequantize_mxfp8,
    profile_batched_gemm_deepgemm_mxfp8_einsum_grouped_o_proj,
    quantize_mxfp8,
)

_OK = {"num_batches": 2, "m": 48, "n": 1024, "k": 4096, "dtype": "mxfp8_e4m3"}


def test_support_is_mxfp8_on_b200_only():
    # An fp8_e4m3 (per-tensor/128-block) or bf16 batched_gemm row must never
    # be served by this MXFP8 backend, and only B200 was smoked.
    support = find_kernel_profiler_spec("batched_gemm", BACKEND).supports
    assert support.allows(DType.MXFP8_E4M3, gpu="NVIDIA B200")
    assert not support.allows(DType.FP8_E4M3, gpu="NVIDIA B200")
    assert not support.allows(DType.BF16, gpu="NVIDIA B200")
    assert not support.allows(DType.MXFP8_E4M3, gpu="NVIDIA H200")


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"dtype": "fp8_e4m3"}, "dtype=mxfp8_e4m3"),
        ({"num_batches": 0}, "num_batches"),
        # The mega-attention buffer only has 8 group slots.
        ({"num_batches": 9}, "num_batches"),
        ({"m": 0}, "m >= 1"),
        ({"n": 2048}, r"\(n, k\) == \(1024, 4096\)"),
        ({"k": 2048}, r"\(n, k\) == \(1024, 4096\)"),
    ],
)
def test_runner_rejects_specs_outside_the_frozen_wo_a_layout(override, message):
    # Rejection must happen before any CUDA or fork import.
    with pytest.raises(ValueError, match=message):
        profile_batched_gemm_deepgemm_mxfp8_einsum_grouped_o_proj(**{**_OK, **override})


def test_mxfp8_quantizer_is_power_of_two_per_32_block_and_round_trips():
    # The synthetic operands and the oracle share this quantizer; a wrong
    # block size or ue8m0 bias would silently weaken the correctness check.
    torch = pytest.importorskip("torch")
    x = torch.randn(3, 128) * torch.tensor([1e-3, 1.0, 1e3]).view(3, 1)
    data, scale = quantize_mxfp8(torch, x)
    assert data.dtype == torch.float8_e4m3fn and scale.dtype == torch.uint8
    assert scale.shape == (3, 4)
    back = dequantize_mxfp8(torch, data, scale)
    assert ((back - x).norm() / x.norm()).item() < 0.05
    assert back.abs().amax().item() <= x.abs().amax().item() * 1.07
