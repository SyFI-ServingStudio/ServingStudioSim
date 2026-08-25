"""DeepSeek V4 C4 indexer prefill top-k operation."""

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

KIND = "deepseek_v4_indexer_topk_prefill"


@dataclass(frozen=True)
class DeepseekV4IndexerTopkPrefillArgs(KernelArgs):
    query_context_pairs: tuple[tuple[int, int], ...]
    max_model_len: int
    max_num_batched_tokens: int
    max_logits_bytes: int
    compress_ratio: int
    top_k: int
    logits_dtype: DType
    index_dtype: str


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cuda",
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.deepseek_v4_indexer_topk_prefill_cuda",
            function_name="profile_deepseek_v4_indexer_topk_prefill_cuda",
        ),
        table_name=KIND,
        args_schema=DeepseekV4IndexerTopkPrefillArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=BackendSupport(
            compute=frozenset({DType.FP32}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        subprocess_env="vllm_env",
    )
)

__all__ = ["DeepseekV4IndexerTopkPrefillArgs", "KIND"]
