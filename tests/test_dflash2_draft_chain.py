"""Input rejection of the DFlash2 draft-chain runners (no GPU)."""

from __future__ import annotations

import pytest

from profiling.runners.dflash2 import draft_chain
from profiling.runners.exceptions import ProfilerNotImplemented

_QK = dict(
    num_tokens=64,
    num_qo_heads=32,
    num_kv_heads=8,
    head_dim=128,
    page_size=16,
    input_dtype="bf16",
    kv_dtype="fp8_e4m3",
)


@pytest.mark.parametrize(
    "overrides",
    [{"kv_dtype": "bf16"}, {"input_dtype": "fp16"}, {"head_dim": 64}],
)
def test_qk_write_refuses_what_the_fused_kernel_is_not_built_for(overrides: dict) -> None:
    # Defect: timing the fp8-write kernel under a bf16-cache or other-head-dim
    # label; its norm reduction is laid out for 128-wide heads.
    with pytest.raises(ProfilerNotImplemented):
        draft_chain.profile_qk_norm_rope_fp8_kv_write_triton(**(_QK | overrides))


def test_qk_write_rejects_a_non_positive_row_count() -> None:
    with pytest.raises(ValueError):
        draft_chain.profile_qk_norm_rope_fp8_kv_write_triton(**(_QK | {"num_tokens": 0}))
