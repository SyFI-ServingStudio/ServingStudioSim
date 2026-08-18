"""Qwen GDN gated RMS normalization kernel kind.

The initial ``torch`` backend measures the complete multi-launch semantic
reference. It is a correctness/performance baseline, not the production fused
normalization launch, and must not be selected for production simulation after
the vLLM backend is registered.
"""

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

KIND: str = "gdn_gated_rms_norm"


@dataclass(frozen=True)
class GdnGatedRmsNormArgs(KernelArgs):
    m: int
    hidden: int
    dtype: DType


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(compute=frozenset({DType.BF16})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.gdn_gated_rms_norm_torch",
            function_name="profile_gdn_gated_rms_norm",
        ),
        table_name=KIND,
        args_schema=GdnGatedRmsNormArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        runner_ref=RunnerRef(
            module_name=("profiling.runners.attention.gdn_gated_rms_norm_vllm_triton"),
            function_name="profile_gdn_gated_rms_norm_vllm_triton",
        ),
        table_name=KIND,
        args_schema=GdnGatedRmsNormArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
    )
)
