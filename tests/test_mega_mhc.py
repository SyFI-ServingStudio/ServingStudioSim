"""mega_mhc runner pieces that can break without a GPU."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from profiling.runners.mhc import mega_mhc_deepgemm as mhc  # noqa: E402


def test_decode_fp8_applies_one_ue8m0_exponent_per_32_channels() -> None:
    """A wrong byte order or group width would scale the wrong channels."""
    y = torch.ones(2, 128).to(torch.float8_e4m3fn)
    exps = torch.tensor([[127, 128, 126, 130], [127, 127, 127, 127]], dtype=torch.uint8)
    words = exps.view(torch.int32)  # little-endian: byte k scales channels 32k..32k+31
    out = mhc.decode_fp8(torch, y, words)
    assert out[0, ::32].tolist() == [1.0, 2.0, 0.5, 8.0]
    assert out[1].unique().tolist() == [1.0]


def test_shared_sf_rows_tile_and_stay_distinct() -> None:
    """block_m 240: tokens 0..239 fill one 256-row tile, 240 starts the next."""
    rows = mhc.shared_sf_rows(torch, 600, 240, "cpu")
    assert rows.unique().numel() == 600
    assert int(rows[:240].max()) < 256 and int(rows[240]) == 256
    assert rows[:4].tolist() == [0, 4, 8, 12] and int(rows[32]) == 1


@pytest.mark.parametrize(
    "args, match",
    [
        ((16, 5120, 2, "bf16", "gemm"), "hc_mult"),
        ((16, 5000, 4, "bf16", "gemm"), "hidden_size"),
        ((16, 5120, 4, "fp8_e4m3", "gemm"), "bf16"),
        ((16, 5120, 4, "bf16", "col"), "sf_layout"),
        ((0, 5120, 4, "bf16", "gemm"), "num_tokens"),
    ],
)
def test_validate_rejects_shapes_deepgemm_refuses(args, match) -> None:
    with pytest.raises(ValueError, match=match):
        mhc.validate(*args)
