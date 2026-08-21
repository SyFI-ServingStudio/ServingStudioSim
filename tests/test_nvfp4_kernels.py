from __future__ import annotations

import importlib

import pytest

from profiling.db.args import DType
from profiling.db.registry import MetricFamily, find_kernel_profiler_spec
from profiling.kernels.nvfp4_moe import KIND as MOE_KIND
from profiling.kernels.nvfp4_moe import Nvfp4MoeArgs
from profiling.kernels.nvfp4_quant import KIND as QUANT_KIND
from profiling.kernels.nvfp4_quant import Nvfp4QuantArgs


def test_nvfp4_specs_are_b200_only_and_lazy() -> None:
    quant = find_kernel_profiler_spec(QUANT_KIND, "vllm_cuda")
    moe = find_kernel_profiler_spec(MOE_KIND, "flashinfer_trtllm")

    assert quant.args_schema is Nvfp4QuantArgs
    assert moe.args_schema is Nvfp4MoeArgs
    assert quant.metric_family is MetricFamily.COMPUTE
    assert moe.metric_family is MetricFamily.COMPUTE
    assert quant.supports.compute == frozenset({DType.BF16})
    assert moe.supports.compute == frozenset({DType.BF16})
    assert quant.supports.gpus == frozenset({"NVIDIA B200"})
    assert moe.supports.gpus == frozenset({"NVIDIA B200"})
    assert quant.subprocess_env == "vllm_env"
    assert moe.subprocess_env == "vllm_env"


def test_nvfp4_quant_validation_matches_checkpoint_contract() -> None:
    runner = importlib.import_module("profiling.runners.elementwise.nvfp4_quant")

    assert runner._validate_args(17, 6144, 16, "bf16", "linear_e4m3") == (
        17,
        6144,
    )
    with pytest.raises(ValueError, match="group_size=16"):
        runner._validate_args(17, 6144, 32, "bf16", "trtllm_swizzled_e4m3")
    with pytest.raises(ValueError, match="bf16"):
        runner._validate_args(17, 6144, 16, "fp16", "trtllm_swizzled_e4m3")


def test_nvfp4_moe_validation_accepts_ep4_and_ep8_shards() -> None:
    runner = importlib.import_module("profiling.runners.moe.nvfp4_moe")
    common = {
        "num_tokens": 17,
        "hidden_size": 6144,
        "intermediate_size": 2048,
        "num_experts": 256,
        "top_k": 8,
        "input_dtype": "bf16",
        "weight_format": "nvfp4_e2m1",
        "group_size": 16,
        "routing_method": "minimax2",
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_numerator": 5,
        "routed_scaling_denominator": 2,
    }

    assert runner._validate_args(
        **common, num_local_experts=64, local_expert_offset=192
    )["local_expert_offset"] == 192
    assert runner._validate_args(
        **common, num_local_experts=32, local_expert_offset=224
    )["num_local_experts"] == 32
    with pytest.raises(ValueError, match="outside the global expert range"):
        runner._validate_args(
            **common, num_local_experts=64, local_expert_offset=224
        )
