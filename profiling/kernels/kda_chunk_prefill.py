"""Kimi-K3 SGLang chunked KDA prefill kernel group."""

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

KIND = "kda_chunk_prefill"


@dataclass(frozen=True)
class KdaChunkPrefillArgs(KernelArgs):
    num_tokens: int
    max_sequence_length: int
    num_sequences: int
    prefix_len: int
    num_heads: int
    head_dim: int
    dtype: DType
    state_dtype: DType
    lower_bound: float


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="sglang_triton",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.kimi_k3_prefill",
            function_name="profile_kda_chunk_prefill_sglang_triton",
        ),
        table_name=KIND,
        args_schema=KdaChunkPrefillArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="sglang_k3_env",
    )
)
