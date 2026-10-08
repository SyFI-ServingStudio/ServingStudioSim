import pytest

from profiling.runners.attention.dsa_indexer_q_rope_quant_weight_fold_cutedsl import (
    _validate_args,
)
from profiling.runners.exceptions import ProfilerNotImplemented


def _shape(num_tokens: int, *, max_model_len: int = 65536):
    return _validate_args(
        num_tokens,
        64,
        128,
        64,
        max_model_len,
        max(8192, num_tokens),
        128**-0.5,
        64**-0.5,
        448.0,
        1.0e-4,
        "int64",
        "bf16",
        "fp32",
        "bf16",
        "fp8_e4m3",
        "fp32",
        "gptj_interleaved_trailing",
        "per_token_head_fp8_pow2_ceil_folded_weight",
    )


def test_coarsen_boundary_matches_public_dispatch():
    assert _shape(511).coarsen == 1
    assert _shape(512).coarsen == 4
    assert _shape(513).coarsen == 4


def test_runtime_context_is_explicit_but_not_frozen_to_checkpoint_maximum():
    assert _shape(1, max_model_len=65536).max_model_len == 65536
    assert _shape(1, max_model_len=1048576).max_model_len == 1048576
    assert _shape(1, max_model_len=2_097_152).max_model_len == 2_097_152
    with pytest.raises(ValueError, match="max_model_len"):
        _shape(1, max_model_len=0)


@pytest.mark.parametrize("num_heads", [4, 32, 64, 128])
def test_any_head_count_the_kernel_tiles_is_accepted(num_heads):
    assert _shape_with_heads(num_heads).num_heads == num_heads


@pytest.mark.parametrize("num_heads", [0, 6, 30])
def test_head_count_must_tile_by_the_compiled_coarsening(num_heads):
    with pytest.raises(ProfilerNotImplemented, match="multiple of 4"):
        _shape_with_heads(num_heads)


def test_batch_capacity_and_model_identity_fail_closed():
    with pytest.raises(ValueError, match="cover num_tokens"):
        _validate_args(
            8193,
            64,
            128,
            64,
            65536,
            8192,
            128**-0.5,
            64**-0.5,
            448.0,
            1.0e-4,
            "int64",
            "bf16",
            "fp32",
            "bf16",
            "fp8_e4m3",
            "fp32",
            "gptj_interleaved_trailing",
            "per_token_head_fp8_pow2_ceil_folded_weight",
        )
    with pytest.raises(ProfilerNotImplemented, match="head_dim"):
        _validate_args(
            1,
            64,
            256,
            64,
            65536,
            8192,
            128**-0.5,
            64**-0.5,
            448.0,
            1.0e-4,
            "int64",
            "bf16",
            "fp32",
            "bf16",
            "fp8_e4m3",
            "fp32",
            "gptj_interleaved_trailing",
            "per_token_head_fp8_pow2_ceil_folded_weight",
        )


def test_batch_capacity_is_not_capped():
    assert _shape(40_000).num_tokens == 40_000


def _shape_with_heads(num_heads: int):
    return _validate_args(
        1,
        num_heads,
        128,
        64,
        65536,
        8192,
        128**-0.5,
        max(num_heads, 1) ** -0.5,
        448.0,
        1.0e-4,
        "int64",
        "bf16",
        "fp32",
        "bf16",
        "fp8_e4m3",
        "fp32",
        "gptj_interleaved_trailing",
        "per_token_head_fp8_pow2_ceil_folded_weight",
    )
