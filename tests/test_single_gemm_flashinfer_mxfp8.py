"""CPU checks for the ``single_gemm`` ``flashinfer_mxfp8`` backend.

GPU timing and the numerical check against the production kernel are covered by
the public ``python -m profiling run`` smoke on a B200; these tests guard the
pieces that can regress without a GPU.
"""

from __future__ import annotations

import pytest

from profiling.db.args import DType
from profiling.runners.gemm.flashinfer_mxfp8 import (
    _dequantize,
    profile_single_gemm_flashinfer_mxfp8,
)


def test_mxfp8_dtype_wire_value_and_alias():
    # A drifted wire literal would orphan DB rows and break the Rust mirror.
    assert DType.from_value("mxfp8_e4m3") is DType.MXFP8_E4M3
    assert DType.from_value("MXFP8") is DType.MXFP8_E4M3
    assert DType.MXFP8_E4M3.value == "mxfp8_e4m3"


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"m": 48, "n": 1792, "k": 5120, "dtype": "fp8_e4m3"}, "dtype=mxfp8_e4m3"),
        ({"m": 48, "n": 1792, "k": 5120, "dtype": "bf16"}, "dtype=mxfp8_e4m3"),
        ({"m": 0, "n": 1792, "k": 5120, "dtype": "mxfp8_e4m3"}, "m >= 1"),
        ({"m": 48, "n": 1792, "k": 5136, "dtype": "mxfp8_e4m3"}, "k % 32"),
        ({"m": 48, "n": 1792, "k": 96, "dtype": "mxfp8_e4m3"}, "k >= 128"),
        ({"m": 48, "n": 64, "k": 5120, "dtype": "mxfp8_e4m3"}, "n >= 128"),
    ],
)
def test_runner_rejects_specs_the_production_linear_cannot_run(spec, message):
    # A per-tensor FP8 or BF16 row must never be timed as MXFP8, and shapes the
    # fork's apply_weights asserts on must fail before any GPU work.
    with pytest.raises(ValueError, match=message):
        profile_single_gemm_flashinfer_mxfp8(**spec)


def test_reference_dequantization_applies_ue8m0_bias_per_32_block():
    # The correctness oracle is only as good as its dequantizer: ue8m0 is a
    # biased power-of-two exponent (127 == 1.0) applied to one 32-wide block.
    torch = pytest.importorskip("torch")
    data = torch.ones(2, 64, dtype=torch.float32).to(torch.float8_e4m3fn)
    scales = torch.tensor([[127, 128], [126, 130]], dtype=torch.uint8)
    out = _dequantize(torch, data, scales)
    assert torch.equal(out[0, :32], torch.full((32,), 1.0))
    assert torch.equal(out[0, 32:], torch.full((32,), 2.0))
    assert torch.equal(out[1, :32], torch.full((32,), 0.5))
    assert torch.equal(out[1, 32:], torch.full((32,), 8.0))
