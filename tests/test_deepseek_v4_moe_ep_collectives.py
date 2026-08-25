"""Behavioral contracts for DeepSeek V4's ragged naive-DP/EP collectives."""

from __future__ import annotations

from dataclasses import fields

import pytest

from profiling.db.registry import MetricFamily, find_kernel_profiler_spec
from profiling.kernels.moe_ep_all_gather import MoeEpAllGatherArgs
from profiling.kernels.moe_ep_reduce_scatter import MoeEpReduceScatterArgs
from profiling.runners.comm import moe_ep_collectives_vllm_pynccl as runner


def test_registry_preserves_the_observed_per_rank_topology():
    expected = [
        (
            "moe_ep_all_gather",
            MoeEpAllGatherArgs,
            "profile_moe_ep_all_gather_batch",
        ),
        (
            "moe_ep_reduce_scatter",
            MoeEpReduceScatterArgs,
            "profile_moe_ep_reduce_scatter_batch",
        ),
    ]
    for kind, arguments, function_name in expected:
        profiler_spec = find_kernel_profiler_spec(kind, "vllm_pynccl")
        assert profiler_spec.args_schema is arguments
        assert profiler_spec.metric_family is MetricFamily.COMM
        assert profiler_spec.runner_ref.function_name == function_name
        assert profiler_spec.list_native
        assert profiler_spec.gpu_count_fn({"num_gpus": 4}) == 4

    assert [field.name for field in fields(MoeEpAllGatherArgs)][:2] == [
        "num_gpus",
        "per_rank_tokens",
    ]
    assert [field.name for field in fields(MoeEpReduceScatterArgs)][:2] == [
        "num_gpus",
        "per_rank_tokens",
    ]


def test_ragged_and_zero_rank_work_are_valid_but_shape_collisions_are_not():
    validated = runner._validate_topology(4, (128, 96, 0, 32), 4096, "nvlink")
    assert validated == (4, (128, 96, 0, 32), 4096)

    with pytest.raises(ValueError, match="num_gpus=4 entries"):
        runner._validate_topology(4, (128, 96), 4096, "nvlink")
    with pytest.raises(ValueError, match="non-negative"):
        runner._validate_topology(4, (128, -1, 96, 32), 4096, "nvlink")


def test_metrics_use_the_slowest_rank_and_physical_wire_bytes():
    metrics = runner._metric_payload([0.08, 0.12, 0.10, 0.09], 1_200_000, 3_600_000)
    assert metrics["time_ms"] == 0.12
    assert metrics["algbw_gbps"] == pytest.approx(10.0)
    assert metrics["busbw_gbps"] == pytest.approx(30.0)
