"""Kimi-K3 fused KDA decode profiling contract."""

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

# This is a separate kind because the production callable includes the KDA
# convolution, recurrent update, and gated normalization in one input contract.
KIND = "kda_fused_decode"


@dataclass(frozen=True)
class KdaFusedDecodeArgs(KernelArgs):
    batch_size: int
    num_heads: int
    head_k_dim: int
    head_v_dim: int
    dtype: DType
    state_dtype: DType
    lower_bound: float


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="sglang_fused",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.kda_fused_decode",
            function_name="profile_kda_fused_decode_sglang_fused",
        ),
        table_name=KIND,
        args_schema=KdaFusedDecodeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="sglang_k3_env",
    )
)
