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


_LAYERED = dict(
    num_tokens=64,
    num_layers=5,
    num_kv_heads=8,
    head_dim=128,
    page_size=16,
    input_dtype="bf16",
    kv_dtype="fp8_e4m3",
)


def test_context_kv_store_is_measured_with_bf16_input_and_an_fp8_cache_only() -> None:
    for overrides in ({"kv_dtype": "bf16"}, {"input_dtype": "fp16"}):
        with pytest.raises(ProfilerNotImplemented):
            draft_chain.profile_layered_kv_norm_rope_cache_write_vllm_triton(
                **(_LAYERED | overrides)
            )


_CONV = dict(num_tokens=64, hidden=4096, group_size=16, taps=2, block_len=8, dtype="bf16")


def test_conv_residual_norm_needs_whole_groups_and_whole_blocks() -> None:
    # Defect: a row count that cuts a block would let taps read across requests.
    with pytest.raises(ValueError):
        draft_chain.profile_grouped_conv_residual_rms_norm_vllm_triton(
            **(_CONV | {"num_tokens": 60})
        )
    for overrides in ({"group_size": 24}, {"dtype": "fp16"}):
        with pytest.raises(ProfilerNotImplemented):
            draft_chain.profile_grouped_conv_residual_rms_norm_vllm_triton(**(_CONV | overrides))
