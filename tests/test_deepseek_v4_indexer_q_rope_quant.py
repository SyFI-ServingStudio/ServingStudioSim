import pytest

from profiling.runners.attention.deepseek_v4_indexer_q_rope_quant_cutedsl import (
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


def test_coarsen_boundary_matches_public_dispatch():
    assert _shape(511).coarsen == 1
    assert _shape(512).coarsen == 4
    assert _shape(513).coarsen == 4


def test_runtime_context_is_explicit_but_not_frozen_to_checkpoint_maximum():
    assert _shape(1, max_model_len=65536).max_model_len == 65536
    assert _shape(1, max_model_len=1048576).max_model_len == 1048576


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
    with pytest.raises(ProfilerNotImplemented):
        _shape_with_heads(32)


def _shape_with_heads(num_heads: int):
    return _validate_args(
        1,
        num_heads,
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
