"""Batched-GEMM kernel kind with production-layout backend identities.

The first backend reproduces GLM-5.2 MLA Q absorption. Its name deliberately
freezes the model-specific 256-wide Q storage and 448-wide packed KV weight
layout instead of presenting those constants as generic batched GEMM behavior.
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

KIND: str = "batched_gemm"


@dataclass(frozen=True)
class BatchedGemmArgs(KernelArgs):
    num_batches: int
    m: int
    n: int
    k: int
    dtype: DType


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch_mla_q_absorb_glm52",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.batched_gemm",
            function_name="profile_mla_q_absorb_glm52",
        ),
        table_name=KIND,
        args_schema=BatchedGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
    )
)
