"""CPU tests for the gemm_swiglu_quant reference and the tensorfold_ck runner's shape gate."""

from __future__ import annotations

import pytest
import torch

from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.gemm import gemm_swiglu_quant_reference as ref
from profiling.runners.gemm.tensorfold_gemm_swiglu_quant import _validate


def test_e2m1_rounds_ties_to_the_even_code_and_saturates() -> None:
    # A wrong tie rule flips codes on exactly these boundaries, which the
    # kernel (nvfp4q::e2m1) rounds to codes 0, 2, 2, 4, 4, 6, 6.
    v = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 100.0, -0.1, -0.3, -6.0])
    assert ref.e2m1_encode(torch, v).tolist() == [0, 2, 2, 4, 4, 6, 6, 7, 0, 9, 15]


def test_quantized_rows_use_the_checkpoint_layout() -> None:
    # 2 rows x 64 columns: one 64-column tile, four 16-element blocks a row.
    # Catches a transposed scale plane or swapped nibbles: the consumer GEMM
    # reads scales [n/64, m, 4] and element 2j from byte j's low nibble.
    h = torch.zeros((2, 64))
    h[0, 0], h[0, 1] = 6.0, 3.0  # row 0, block 0: amax 6
    h[1, 48] = -1.0  # row 1, block 3: amax 1
    codes, scales = ref.quantize_rows(torch, h, qg=1.0)
    assert codes.shape == (2, 32) and scales.shape == (1, 2, 4)
    assert scales[0, 0].tolist() == [0x38, 0, 0, 0]  # e4m3(1.0)
    assert ref.e4m3_value(torch, scales[0, 1, 3]).item() == pytest.approx(1.0 / 6.0, rel=0.07)
    assert codes[0, 0].item() == 0x57  # low nibble 6.0 (code 7), high 3.0 (code 5)
    back = ref.dequantize_rows(torch, codes, scales, act=1.0)
    torch.testing.assert_close(back[0, :2], torch.tensor([6.0, 3.0]))
    assert back[1, 48].item() == pytest.approx(-1.0, rel=0.05)


def test_reference_quantizes_silu_gate_times_up() -> None:
    x = torch.tensor([[1.0] * 128])
    wg = torch.zeros((128, 128))
    wu = torch.zeros((128, 128))
    wg[0, :] = 2.0 / 128  # gate column 0 = 2
    wu[0, :] = 3.0 / 128  # up column 0 = 3
    _, _, h = ref.gemm_swiglu_quant_reference(torch, x, wg, wu, qg=1.0)
    assert h[0, 0].item() == pytest.approx(2.0 / (1.0 + torch.exp(torch.tensor(-2.0)).item()) * 3.0)
    assert h[0, 1:].abs().max().item() == 0.0


@pytest.mark.parametrize(
    ("n", "k", "dtype", "error"),
    [
        (17408, 5120, "bf16", ProfilerNotImplemented),
        (17408 + 64, 5120, "nvfp4_e2m1", ValueError),  # mlp_prompt runs unfused
        (17408, 5120 + 64, "nvfp4_e2m1", ValueError),
    ],
)
def test_runner_refuses_shapes_tensorfold_does_not_fuse(n, k, dtype, error) -> None:
    with pytest.raises(error):
        _validate(128, n, k, dtype)


def test_runner_accepts_the_production_shape() -> None:
    assert _validate(4096, 17408, 5120, "nvfp4_e2m1") == (4096, 17408, 5120)
