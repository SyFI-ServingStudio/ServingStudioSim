"""CPU-side contract tests for ``dsa_indexer_q_rope_quant``."""

from __future__ import annotations

from dataclasses import fields

import pytest

from profiling.db.args import DType
from profiling.db.batch import coerce_args
from profiling.db.registry import find_kernel_profiler_spec
from profiling.kernels.dsa_indexer_q_rope_quant import KIND, DsaIndexerQRopeQuantArgs
from profiling.runners.attention.dsa_indexer_q_rope_quant import (
    profile_dsa_indexer_q_rope_quant_sglang_cuda,
)
from profiling.runners.exceptions import ProfilerNotImplemented


def _spec(**overrides):
    spec = {
        "num_tokens": 2048,
        "num_heads": 32,
        "head_dim": 128,
        "rope_dim": 64,
        "rope_layout": "rope_first",
        "hadamard": False,
        "input_dtype": DType.BF16,
        "q_output_dtype": DType.FP8_E4M3,
        "weight_output_dtype": DType.FP32,
    }
    spec.update(overrides)
    return spec


def test_schema_order_and_runner_route_are_the_public_contract():
    assert [field.name for field in fields(DsaIndexerQRopeQuantArgs)] == [
        "num_tokens",
        "num_heads",
        "head_dim",
        "rope_dim",
        "rope_layout",
        "hadamard",
        "input_dtype",
        "q_output_dtype",
        "weight_output_dtype",
    ]
    args = coerce_args(DsaIndexerQRopeQuantArgs, _spec(input_dtype="bfloat16"))
    assert args.input_dtype is DType.BF16

    registered = find_kernel_profiler_spec(KIND, "sglang_cuda")
    assert registered.table_name == KIND
    assert registered.subprocess_env == "sglang_env"
    assert registered.runner_ref.function_name == ("profile_dsa_indexer_q_rope_quant_sglang_cuda")


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"rope_layout": "rope_last"}, "rope_first"),
        ({"hadamard": True}, "rope_first"),
        ({"head_dim": 192}, "head_dim, rope_dim"),
        ({"rope_dim": 128}, "head_dim, rope_dim"),
        ({"q_output_dtype": DType.BF16}, "q_output_dtype=fp8_e4m3"),
    ],
)
def test_refuses_a_different_cuda_instantiation_before_framework_import(override, message):
    with pytest.raises(ProfilerNotImplemented, match=message):
        profile_dsa_indexer_q_rope_quant_sglang_cuda(**_spec(**override))


def test_rejects_degenerate_shapes_before_allocation():
    with pytest.raises(ValueError, match="num_tokens"):
        profile_dsa_indexer_q_rope_quant_sglang_cuda(**_spec(num_tokens=0))
