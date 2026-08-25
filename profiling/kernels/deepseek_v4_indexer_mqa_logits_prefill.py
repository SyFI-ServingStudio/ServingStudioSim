"""DeepSeek V4 C4 indexer prefill MQA-logits operation."""

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

KIND = "deepseek_v4_indexer_mqa_logits_prefill"


@dataclass(frozen=True)
class DeepseekV4IndexerMqaLogitsPrefillArgs(KernelArgs):
    query_context_pairs: tuple[tuple[int, int], ...]
    max_model_len: int
    max_num_batched_tokens: int
    max_logits_bytes: int
    compress_ratio: int
    num_heads: int
    head_dim: int
    q_dtype: DType
    k_dtype: DType
    k_scale_dtype: DType
    weight_dtype: DType
    output_dtype: DType
    clean_logits: bool


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_deepgemm_fp8",
        runner_ref=RunnerRef(
            module_name=(
                "profiling.runners.attention.deepseek_v4_indexer_mqa_logits_prefill_deepgemm"
            ),
            function_name="profile_deepseek_v4_indexer_mqa_logits_prefill_deepgemm",
        ),
        table_name=KIND,
        args_schema=DeepseekV4IndexerMqaLogitsPrefillArgs,
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

__all__ = ["DeepseekV4IndexerMqaLogitsPrefillArgs", "KIND"]
