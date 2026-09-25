"""Kimi-K3 eager prefill three-way BF16 add."""

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

KIND = "k3_add3_prefill"


@dataclass(frozen=True)
class K3Add3PrefillArgs(KernelArgs):
    num_tokens: int
    hidden_size: int
    dtype: DType


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="sglang_k3",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.elementwise.kimi_k3",
            function_name="profile_k3_add3_prefill",
        ),
        table_name=KIND,
        args_schema=K3Add3PrefillArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="sglang_k3_env",
    )
)
