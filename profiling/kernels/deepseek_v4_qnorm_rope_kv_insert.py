"""DeepSeek V4 fused Q normalization/RoPE and packed KV insert."""

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

KIND = "deepseek_v4_qnorm_rope_kv_insert"


@dataclass(frozen=True)
class DeepseekV4QnormRopeKvInsertArgs(KernelArgs):
    num_tokens: int
    num_insert_tokens: int
    num_heads: int
    padded_heads: int
    head_dim: int
    rope_dim: int
    block_size: int
    rms_eps: float
    input_dtype: DType
    cache_dtype: str
    cache_layout: str
    scale_format: str


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cuda",
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.deepseek_v4_qnorm_rope_kv_insert_vllm_cuda",
            function_name="profile_deepseek_v4_qnorm_rope_kv_insert_vllm_cuda",
        ),
        table_name=KIND,
        args_schema=DeepseekV4QnormRopeKvInsertArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.FP8_E4M3}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        subprocess_env="vllm_env",
    )
)

__all__ = ["DeepseekV4QnormRopeKvInsertArgs", "KIND"]
