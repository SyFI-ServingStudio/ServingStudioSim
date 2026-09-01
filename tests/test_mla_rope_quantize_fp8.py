"""CPU-side contract tests for ``mla_rope_quantize_fp8``."""

from __future__ import annotations

from dataclasses import fields

import pytest

from profiling.db.args import DType
from profiling.db.batch import coerce_args
from profiling.db.registry import find_kernel_profiler_spec
from profiling.kernels.mla_rope_quantize_fp8 import KIND, MlaRopeQuantizeFp8Args
from profiling.runners.attention.mla_rope_quantize_fp8 import (
    profile_mla_rope_quantize_fp8_flashinfer,
)
from profiling.runners.exceptions import ProfilerNotImplemented


def _spec(**overrides):
    spec = {
        "num_tokens": 2048,
        "num_heads": 16,
        "kv_lora_rank": 512,
        "rope_dim": 64,
        "max_position": 1_048_576,
        "is_neox_style": False,
        "input_dtype": DType.BF16,
        "quant_dtype": DType.FP8_E4M3,
    }
    spec.update(overrides)
    return spec


def test_schema_order_and_runner_route_are_the_public_contract():
    assert [field.name for field in fields(MlaRopeQuantizeFp8Args)] == [
        "num_tokens",
        "num_heads",
        "kv_lora_rank",
        "rope_dim",
        "max_position",
        "is_neox_style",
        "input_dtype",
        "quant_dtype",
    ]
    args = coerce_args(MlaRopeQuantizeFp8Args, _spec(input_dtype="bfloat16"))
    assert args.input_dtype is DType.BF16

    registered = find_kernel_profiler_spec(KIND, "flashinfer")
    assert registered.table_name == KIND
    assert registered.subprocess_env == "sglang_env"
    assert registered.runner_ref.function_name == "profile_mla_rope_quantize_fp8_flashinfer"


def test_refuses_non_fp8_output_before_framework_import():
    with pytest.raises(ProfilerNotImplemented, match="quant_dtype=fp8_e4m3"):
        profile_mla_rope_quantize_fp8_flashinfer(**_spec(quant_dtype=DType.BF16))


@pytest.mark.parametrize("override", [{"num_tokens": 0}, {"num_heads": -1}, {"rope_dim": 63}])
def test_rejects_degenerate_shapes_before_allocation(override):
    with pytest.raises(ValueError):
        profile_mla_rope_quantize_fp8_flashinfer(**_spec(**override))
