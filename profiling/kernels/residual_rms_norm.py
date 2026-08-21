"""Residual-add RMSNorm kernel kind.

This module owns the ``residual_rms_norm`` wire string, its Args schema, the
multi-launch Torch semantic backend, and the production-aligned fused vLLM CUDA
backend.

Both runners are referenced lazily so importing the registry does not import
Torch, vLLM, or CUDA runtime code.
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

KIND: str = "residual_rms_norm"


@dataclass(frozen=True)
class ResidualRmsNormArgs(KernelArgs):
    m: int
    hidden: int
    dtype: DType


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(compute=frozenset({DType.BF16, DType.FP16})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.norm.residual_rms_norm_torch",
            function_name="profile_residual_rms_norm",
        ),
        table_name=KIND,
        args_schema=ResidualRmsNormArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cuda",
        supports=BackendSupport(
            compute=frozenset({DType.BF16, DType.FP16}),
            gpus=frozenset({"NVIDIA H200", "NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.norm.residual_rms_norm_vllm_cuda",
            function_name="profile_residual_rms_norm_vllm_cuda",
        ),
        table_name=KIND,
        args_schema=ResidualRmsNormArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
    )
)
