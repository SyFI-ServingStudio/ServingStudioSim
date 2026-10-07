"""Contracts for the block-FP8 DP/EP dispatch gather (hidden states plus router logits)."""

from __future__ import annotations

import pytest

from profiling.db.args import DType
from profiling.db.registry import backend_supports
from profiling.runners.comm import moe_ep_collectives_vllm_pynccl as runner
from profiling.runners.exceptions import ProfilerNotImplemented


def test_block_fp8_gather_runs_on_b200_and_bf16_stays_supported():
    for dtype in (DType.BF16, DType.FP8_E4M3):
        assert backend_supports("moe_ep_all_gather", "vllm_pynccl", dtype, gpu="NVIDIA B200")
    assert backend_supports("moe_ep_all_gather", "vllm_pynccl", DType.BF16, gpu="NVIDIA H200")


def test_gathered_tensors_follow_dispatch_router_logits_order():
    # bf16: hidden states, then router logits.
    assert runner.all_gather_columns(4096, 288, "bf16") == (
        (4096, "bfloat16", 2),
        (288, "float32", 4),
    )
    # Block FP8 (4096 hidden, 288 experts): fp8 activations, fp32 logits, then
    # the 32 fp32 group scales vLLM passes as the extra tensor -> 5376 B/token.
    columns = runner.all_gather_columns(4096, 288, "fp8_e4m3")
    assert columns == ((4096, "float8_e4m3fn", 1), (288, "float32", 4), (32, "float32", 4))
    assert sum(width * size for width, _, size in columns) == 5376


def test_validation_accepts_block_fp8_and_rejects_other_dtypes():
    validated = runner._validate_all_gather(8, (8192,) * 8, 4096, 288, "fp8_e4m3", "fp32", "nvlink")
    assert validated == (8, (8192,) * 8, 4096, 288)
    with pytest.raises(ValueError, match="multiple of 128"):
        runner._validate_all_gather(4, (1, 1, 1, 1), 4100, 288, "fp8_e4m3", "fp32", "nvlink")
    with pytest.raises(ProfilerNotImplemented, match="bf16 or fp8_e4m3"):
        runner._validate_all_gather(4, (1, 1, 1, 1), 4096, 288, "nvfp4_e2m1", "fp32", "nvlink")
