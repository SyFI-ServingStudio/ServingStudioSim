import pytest
import torch

from profiling.runners.elementwise.hadamard_fp8_quant_vllm_triton import (
    _checked_rows,
    _validate_args,
    reference,
)
from profiling.runners.exceptions import ProfilerNotImplemented


def _butterfly(x: torch.Tensor) -> torch.Tensor:
    """In-place-order FWHT, independent of the oracle's dense Sylvester matrix."""
    x = x.double().clone()
    n = x.shape[1]
    h = 1
    while h < n:
        x = x.view(x.shape[0], n // (2 * h), 2, h)
        a, b = x[:, :, 0, :], x[:, :, 1, :]
        x = torch.stack((a + b, a - b), dim=2).reshape(x.shape[0], n)
        h *= 2
    return x / n**0.5


def test_oracle_rotation_matches_a_butterfly_transform():
    # Catches a wrong Hadamard ordering or normalization in the oracle, which
    # would otherwise make the GPU check pass or fail for the wrong reason.
    q = torch.randn((64, 128), generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)
    q_ref, scale = reference(torch, q)
    rotated = _butterfly(q).to(torch.bfloat16).double()
    torch.testing.assert_close(q_ref * scale, rotated.clamp(-448 * scale, 448 * scale))
    assert torch.equal(scale, torch.exp2(torch.log2(scale).round()))
    assert (q_ref.abs().amax(dim=1) <= 448).all()
    assert (q_ref.abs().amax(dim=1) > 224).all()


@pytest.mark.parametrize("num_rows", [1, 33, 32_768])
def test_any_row_count_is_accepted(num_rows):
    _validate_args(num_rows, 128, "bf16")


def test_unbuilt_head_width_is_reported_as_unsupported():
    with pytest.raises(ProfilerNotImplemented):
        _validate_args(32, 64, "bf16")
    with pytest.raises(ValueError):
        _validate_args(0, 128, "bf16")


@pytest.mark.parametrize("rows", [1, 32, 33, 4095, 4096, 4100, 8192])
def test_checked_rows_cover_head_and_ragged_tail_once(rows):
    # Caught a negative arange below 4096 rows that failed every small spec.
    idx = _checked_rows(torch, rows, "cpu").tolist()
    assert idx == sorted(set(idx))
    assert idx[0] == 0 and idx[-1] == rows - 1
    assert all(r in idx for r in range(max(0, rows - 64), rows))
