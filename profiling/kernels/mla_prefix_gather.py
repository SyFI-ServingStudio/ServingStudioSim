"""Kimi-K3 chunked-prefix MLA index and latent-KV gather."""

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

KIND = "mla_prefix_gather"


@dataclass(frozen=True)
class MlaPrefixGatherArgs(KernelArgs):
    batch_size: int
    num_tokens: int
    kv_lora_rank: int
    rope_dim: int
    dtype: DType
    cache_dtype: DType


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="sglang_triton",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.FP8_E4M3}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.kimi_k3_prefill",
            function_name="profile_mla_prefix_gather_sglang_triton",
        ),
        table_name=KIND,
        args_schema=MlaPrefixGatherArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="sglang_k3_env",
    )
)
