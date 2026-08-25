"""DeepSeek V4 indexer decode MQA-logits operation."""

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

KIND = "deepseek_v4_indexer_mqa_logits_decode"


@dataclass(frozen=True)
class DeepseekV4IndexerMqaLogitsDecodeArgs(KernelArgs):
    batch_size: int
    context_len: int
    next_n: int
    max_model_len: int
    num_heads: int
    head_dim: int
    block_size: int
    q_dtype: DType
    cache_dtype: DType
    scale_dtype: DType
    weight_dtype: DType
    output_dtype: DType
    context_mode: str
    page_mapping: str
    cache_format: str
    clean_logits: bool


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_deepgemm_fp8",
        runner_ref=RunnerRef(
            module_name=(
                "profiling.runners.attention."
                "deepseek_v4_indexer_mqa_logits_decode_deepgemm"
            ),
            function_name="profile_deepseek_v4_indexer_mqa_logits_decode_deepgemm",
        ),
        table_name=KIND,
        args_schema=DeepseekV4IndexerMqaLogitsDecodeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=BackendSupport(
            compute=frozenset({DType.FP8_E4M3}),
            kv=frozenset({DType.FP8_E4M3}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        subprocess_env="vllm_env",
    )
)

__all__ = ["DeepseekV4IndexerMqaLogitsDecodeArgs", "KIND"]
