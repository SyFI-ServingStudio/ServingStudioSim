"""Kimi-K3 recurrent KDA decode profiling contract."""

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

KIND = "kda_recurrent_decode"


@dataclass(frozen=True)
class KdaRecurrentDecodeArgs(KernelArgs):
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
        backend="torch",
        supports=BackendSupport(compute=frozenset({DType.BF16})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.kda_recurrent_decode",
            function_name="profile_kda_recurrent_decode_torch",
        ),
        table_name=KIND,
        args_schema=KdaRecurrentDecodeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="sglang_triton",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.kda_recurrent_decode",
            function_name="profile_kda_recurrent_decode_sglang_triton",
        ),
        table_name=KIND,
        args_schema=KdaRecurrentDecodeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="sglang_k3_env",
    )
)
