"""SGLang deferred-MoE finalize with an optional fused shared-expert add."""

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

KIND = "moe_finalize_fuse_shared"


@dataclass(frozen=True)
class MoeFinalizeFuseSharedArgs(KernelArgs):
    num_tokens: int
    top_k: int
    hidden_dim: int
    dtype: DType
    fuse_shared_output: bool


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="sglang_cuda",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.moe.moe_finalize_fuse_shared",
            function_name="profile_moe_finalize_fuse_shared_sglang",
        ),
        table_name=KIND,
        args_schema=MoeFinalizeFuseSharedArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="sglang_env",
    )
)
