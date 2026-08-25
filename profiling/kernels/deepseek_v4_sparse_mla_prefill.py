"""DeepSeek V4 BF16 sparse-MLA prefill over one request batch."""

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

KIND = "deepseek_v4_sparse_mla_prefill"


@dataclass(frozen=True)
class DeepseekV4SparseMlaPrefillArgs(KernelArgs):
    query_context_pairs: tuple[tuple[int, int], ...]
    max_model_len: int
    max_num_batched_tokens: int
    prefill_chunk_size: int
    compress_ratio: int
    window_size: int
    selected_k: int
    selected_index_pattern: str
    num_heads: int
    num_kv_heads: int
    head_dim: int
    value_dim: int
    softmax_scale: float
    q_dtype: DType
    cache_dtype: DType
    index_dtype: str
    output_dtype: DType
    cache_layout: str


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_flashmla_bf16",
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.deepseek_v4_sparse_mla_prefill_flashmla",
            function_name="profile_deepseek_v4_sparse_mla_prefill_flashmla",
        ),
        table_name=KIND,
        args_schema=DeepseekV4SparseMlaPrefillArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        subprocess_env="vllm_env",
    )
)

__all__ = ["DeepseekV4SparseMlaPrefillArgs", "KIND"]
