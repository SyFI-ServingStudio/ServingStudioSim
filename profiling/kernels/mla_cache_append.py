"""Plain MLA paged-cache append kernel kind.

The first backend measures the two-write Torch semantic composite for
GLM-5.2's 512-wide latent plus 64-wide RoPE cache entry. The production-aligned
backend measures vLLM's fused one-launch CUDA implementation.
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

KIND: str = "mla_cache_append"


@dataclass(frozen=True)
class MlaCacheAppendArgs(KernelArgs):
    num_tokens: int
    kv_lora_rank: int
    rope_dim: int
    block_size: int
    input_dtype: DType
    kv_dtype: DType
    cache_format: str


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.BF16, DType.FP8_E4M3}),
            gpus=frozenset({"NVIDIA H200", "NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.mla_cache_append",
            function_name="profile_mla_cache_append_torch",
        ),
        table_name=KIND,
        args_schema=MlaCacheAppendArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cuda",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA H200", "NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.mla_cache_append",
            function_name="profile_mla_cache_append_vllm_cuda",
        ),
        table_name=KIND,
        args_schema=MlaCacheAppendArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
    )
)
