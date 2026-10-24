"""CPU checks for the ``single_gemm`` ``tensorfold_lane`` / ``tensorfold_ck`` backends.

GPU timing and the numerical check against TensorFold's GEMM run through the
public ``python -m profiling run`` smoke on a GB10; these guard the pieces that
can regress without a GPU or the TensorFold env.
"""

from __future__ import annotations

import pytest

from profiling.db.args import DType
from profiling.runners.gemm.tensorfold import (
    _decode_e2m1,
    _fp8_logical_order,
    logical_bytes,
    profile_single_gemm_tensorfold_ck,
    profile_single_gemm_tensorfold_lane,
)


@pytest.mark.parametrize(
    ("runner", "spec", "message"),
    [
        (profile_single_gemm_tensorfold_lane, {"m": 8, "n": 5120, "k": 5120, "dtype": "bf16"}, "fp8_e4m3 or nvfp4"),
        (profile_single_gemm_tensorfold_ck, {"m": 8, "n": 5120, "k": 5120, "dtype": "mxfp8_e4m3"}, "fp8_e4m3 or nvfp4"),
        (profile_single_gemm_tensorfold_lane, {"m": 0, "n": 5120, "k": 5120, "dtype": "fp8_e4m3"}, "m >= 1"),
        (profile_single_gemm_tensorfold_lane, {"m": 8, "n": 5120, "k": 5136, "dtype": "nvfp4_e2m1"}, "multiple of 64"),
        # The prompt GEMM's 128-column blocks would read past a 64-padded NVFP4 weight.
        (profile_single_gemm_tensorfold_ck, {"m": 8, "n": 320, "k": 5120, "dtype": "nvfp4_e2m1"}, "multiple of 128"),
    ],
)
def test_runner_rejects_specs_before_gpu_work(runner, spec, message):
    with pytest.raises(ValueError, match=message):
        runner(**spec)


def test_lane_accepts_a_64_column_drafter_head_view_shape():
    # The drafter's vocabulary-subset head runs the lane GEMM on n = 320 (five
    # 64-column tiles); only the prompt GEMM needs 128-column multiples. With no
    # TensorFold env the runner gets past validation and stops at the import.
    from profiling.runners.exceptions import ProfilerNotImplemented

    try:
        import tensorfold  # noqa: F401
    except ImportError:
        with pytest.raises(ProfilerNotImplemented):
            profile_single_gemm_tensorfold_lane(m=16, n=320, k=5120, dtype="nvfp4_e2m1")
    else:
        pytest.skip("TensorFold is importable here; the GPU smoke covers this path")


def test_e2m1_decode_reads_low_nibble_first_with_sign_bit():
    # A swapped nibble order would pair every activation with the wrong weight
    # input and the oracle would no longer describe the checkpoint layout.
    torch = pytest.importorskip("torch")
    packed = torch.tensor([[0x2F, 0x81]], dtype=torch.uint8)
    assert _decode_e2m1(torch, packed).tolist() == [[-6.0, 1.0, 0.5, -0.0]]


def test_fp8_logical_order_inverts_quant8_fragment_order():
    # quant8 stores input 2q + j%2 + 8(j//2) at byte 4q + j of each 16-byte
    # group (act.cu); the oracle must undo exactly that permutation.
    torch = pytest.importorskip("torch")
    logical = torch.arange(32, dtype=torch.uint8).view(1, 32)
    stored = torch.empty_like(logical)
    for g in range(2):
        for p in range(16):
            q, j = divmod(p, 4)
            stored[0, 16 * g + p] = logical[0, 16 * g + 2 * q + j % 2 + 8 * (j // 2)]
    assert torch.equal(_fp8_logical_order(torch, stored), logical)


def test_logical_bytes_follow_the_quantized_operand_widths():
    assert logical_bytes(1, 64, 128, DType.FP8_E4M3) == 128 + 64 * 128 + 2 * 64
    assert logical_bytes(1, 64, 128, DType.NVFP4_E2M1) == (128 + 64 * 128) // 2 + (128 + 64 * 128) // 16 + 2 * 64
