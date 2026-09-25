"""Kimi-K3 chunked-prefix MLA state merge."""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND = "mla_merge_state"


@dataclass(frozen=True)
class MlaMergeStateArgs(KernelArgs):
    num_tokens: int
    num_heads: int
    value_dim: int
    dtype: DType


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="sglang_triton",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.kimi_k3_prefill",
            function_name="profile_mla_merge_state_sglang_triton",
        ),
        table_name=KIND,
        args_schema=MlaMergeStateArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="sglang_k3_env",
    )
)
