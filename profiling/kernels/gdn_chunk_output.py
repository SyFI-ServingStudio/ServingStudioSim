"""Qwen GDN fused chunk-output kernel kind.

The Torch backend is a multi-launch semantic baseline. Production simulation
must select the fused vLLM Triton backend.
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

KIND: str = "gdn_chunk_output"


@dataclass(frozen=True)
class GdnChunkOutputArgs(KernelArgs):
    num_tokens: int
    num_chunks: int
    num_key_heads: int
    num_heads: int
    key_head_dim: int
    value_head_dim: int
    dtype: DType


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(compute=frozenset({DType.BF16})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.gdn_chunk_output_torch",
            function_name="profile_gdn_chunk_output",
        ),
        table_name=KIND,
        args_schema=GdnChunkOutputArgs,
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
            module_name="profiling.runners.attention.gdn_chunk_output_vllm_triton",
            function_name="profile_gdn_chunk_output_vllm_triton",
        ),
        table_name=KIND,
        args_schema=GdnChunkOutputArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
    )
)
