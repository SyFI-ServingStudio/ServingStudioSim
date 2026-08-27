"""GLM-5.2 varlen sparse-MLA prefill over one request batch."""

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

KIND = "dsa_sparse_mla_prefill"


@dataclass(frozen=True)
class DsaSparseMlaPrefillArgs(KernelArgs):
    query_context_pairs: tuple[tuple[int, int], ...]
    num_heads: int
    num_kv_heads: int
    selected_k: int
    latent_dim: int
    rope_dim: int
    value_dim: int
    softmax_scale: float
    q_dtype: DType
    cache_dtype: DType
    index_dtype: str
    output_dtype: DType
    index_distribution: str
    cache_layout: str


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_trtllm_fp8",
        supports=BackendSupport(
            compute=frozenset({DType.FP8_E4M3}),
            kv=frozenset({DType.FP8_E4M3}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_sparse_mla_prefill",
            function_name="profile_dsa_sparse_mla_prefill_flashinfer_trtllm_fp8",
        ),
        table_name=KIND,
        args_schema=DsaSparseMlaPrefillArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
    )
)

__all__ = ["DsaSparseMlaPrefillArgs", "KIND"]
