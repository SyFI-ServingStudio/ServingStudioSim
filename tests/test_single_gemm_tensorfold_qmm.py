"""CPU checks for the ``single_gemm`` ``tensorfold_qmm`` backend.

GPU timing and the numerical check against TensorFold's group_kernel run through
the public ``python -m profiling run`` smoke on a GB10; these guard the pieces
that can regress without a GPU or the TensorFold env.
"""

from __future__ import annotations

import pytest

from profiling.db.args import DType
from profiling.runners.gemm.tensorfold_qmm import (
    dequantize_mlx,
    logical_bytes,
    profile_single_gemm_tensorfold_qmm,
)


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        # bf16 or nvfp4 rows would silently key a different operation under this backend.
        ({"m": 8, "n": 5120, "k": 5120, "dtype": "bf16"}, "w4a16_affine_g64"),
        ({"m": 8, "n": 5120, "k": 5120, "dtype": "nvfp4_e2m1"}, "w4a16_affine_g64"),
        ({"m": 0, "n": 5120, "k": 5120, "dtype": "w4a16_affine_g64"}, "m >= 1"),
        # quantize4 and qmm_group take whole 64-input groups.
        ({"m": 8, "n": 5120, "k": 5136, "dtype": "w4a16_affine_g64"}, "multiple of 64"),
    ],
)
def test_runner_rejects_specs_before_gpu_work(spec, message):
    with pytest.raises(ValueError, match=message):
        profile_single_gemm_tensorfold_qmm(**spec)


def test_dequantize_reads_mlx_nibbles_low_first_with_group_scale_and_bias():
    # dflash2.quantize4 stores input 8i + j in nibble j of word i, and w = q * scale + bias
    # per 64 inputs; a swapped nibble order or a dropped bias would make the oracle
    # describe a different weight than the one the kernel multiplies.
    torch = pytest.importorskip("torch")
    codes = torch.arange(64) % 16
    words = torch.zeros(1, 8, dtype=torch.int32)
    for j in range(8):
        words |= (codes.view(8, 8)[:, j].to(torch.int32) << (4 * j)).view(1, 8)
    scales = torch.tensor([[0.5]], dtype=torch.bfloat16)
    biases = torch.tensor([[-1.0]], dtype=torch.bfloat16)
    expected = codes.float() * 0.5 - 1.0
    assert torch.equal(dequantize_mlx(torch, words, scales, biases), expected.view(1, 64))


def test_dequantize_handles_the_sign_bit_of_the_top_nibble():
    # A code of 8..15 in nibble 7 sets the int32 sign bit; an arithmetic shift must not leak it.
    torch = pytest.importorskip("torch")
    words = torch.full((1, 8), -1, dtype=torch.int32)                      # every nibble 15
    out = dequantize_mlx(torch, words, torch.ones(1, 1, dtype=torch.bfloat16), torch.zeros(1, 1, dtype=torch.bfloat16))
    assert torch.equal(out, torch.full((1, 64), 15.0))


def test_logical_bytes_count_rows_group_sums_codes_affine_planes_and_output():
    m, n, k = 2, 128, 128
    assert logical_bytes(m, n, k) == 2 * m * k + 4 * m * 2 + n * k // 2 + 4 * n * 2 + 2 * m * n
    assert DType.from_value("w4a16_affine_g64") is DType.W4A16_AFFINE_G64
