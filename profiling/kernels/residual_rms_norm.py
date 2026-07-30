"""Residual-add RMSNorm kernel kind.

This module owns the ``residual_rms_norm`` wire string, its Args schema, and
the minimal Torch semantic backend registration. The backend profiles the
multi-launch Torch reference, not vLLM's fused implementation.

The runner is referenced lazily so importing the registry does not import
Torch or CUDA runtime code.
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
