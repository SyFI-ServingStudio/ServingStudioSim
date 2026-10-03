"""Contracts for the NVFP4 DP/EP dispatch gather and the B200 collective limits."""

from __future__ import annotations

import pytest

from profiling.db.args import DType
from profiling.db.registry import MetricFamily, backend_supports, find_kernel_profiler_spec
from profiling.kernels.moe_ep_quantized_all_gather import MoeEpQuantizedAllGatherArgs
from profiling.runners.comm import moe_ep_collectives_vllm_pynccl as runner
from profiling.runners.exceptions import ProfilerNotImplemented


def test_registry_runs_the_quantized_gather_on_b200_only():
    spec = find_kernel_profiler_spec("moe_ep_quantized_all_gather", "vllm_pynccl")
    assert spec.args_schema is MoeEpQuantizedAllGatherArgs
    assert spec.metric_family is MetricFamily.COMM
    assert spec.runner_ref.function_name == "profile_moe_ep_quantized_all_gather_batch"
    assert spec.gpu_count_fn({"num_gpus": 8}) == 8
    assert backend_supports(
        "moe_ep_quantized_all_gather", "vllm_pynccl", DType.NVFP4_E2M1, gpu="NVIDIA B200"
    )
    assert not backend_supports(
        "moe_ep_quantized_all_gather", "vllm_pynccl", DType.NVFP4_E2M1, gpu="NVIDIA H200"
    )
    assert backend_supports("moe_ep_reduce_scatter", "vllm_pynccl", DType.BF16, gpu="NVIDIA B200")


def test_token_bytes_match_vllm_dispatch_payload():
    # GLM-5.2: 6144 hidden, top-8 -> 3072 packed + 384 scales + 32 weights + 32 ids.
    assert runner.quantized_all_gather_token_bytes(6144, 8) == 3520


def test_validation_accepts_dp8_full_batches_and_rejects_unquantized_input():
    validated = runner._validate_quantized_all_gather(
        8, (8192,) * 8, 6144, 8, "nvfp4_e2m1", "nvlink"
    )
    assert validated == (8, (8192,) * 8, 6144, 8)
    with pytest.raises(ProfilerNotImplemented, match="nvfp4_e2m1"):
        runner._validate_quantized_all_gather(4, (1, 1, 1, 1), 6144, 8, "bf16", "nvlink")
    with pytest.raises(ValueError, match="multiple of 16"):
        runner._validate_quantized_all_gather(4, (1, 1, 1, 1), 6148, 8, "nvfp4_e2m1", "nvlink")
    with pytest.raises(ProfilerNotImplemented, match="total tokens"):
        runner._validate_topology(8, (8193,) * 8, 6144, "nvlink")
