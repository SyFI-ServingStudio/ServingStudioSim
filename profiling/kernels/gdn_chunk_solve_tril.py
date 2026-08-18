"""Qwen GDN chunk-local triangular-solve kernel kind.

The initial ``torch`` backend measures a multi-launch semantic implementation.
It is a correctness/performance baseline, not the production fused Triton
launch, and must not be selected for production simulation after the vLLM
backend is registered.
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

KIND: str = "gdn_chunk_solve_tril"


@dataclass(frozen=True)
class GdnChunkSolveTrilArgs(KernelArgs):
    num_tokens: int
    num_chunks: int
    max_chunk_tokens: int
    num_heads: int
    dtype: DType


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(compute=frozenset({DType.BF16})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.gdn_chunk_solve_tril_torch",
            function_name="profile_gdn_chunk_solve_tril",
        ),
        table_name=KIND,
        args_schema=GdnChunkSolveTrilArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="default_env",
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
            module_name="profiling.runners.attention.gdn_chunk_solve_tril_vllm_triton",
            function_name="profile_gdn_chunk_solve_tril_vllm_triton",
        ),
        table_name=KIND,
        args_schema=GdnChunkSolveTrilArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
    )
)
