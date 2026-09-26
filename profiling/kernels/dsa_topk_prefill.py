"""DSA prefill top-k index-selection kernel kind."""

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

KIND: str = "dsa_topk_prefill"


@dataclass(frozen=True)
class DsaTopkPrefillArgs(KernelArgs):
    num_queries: int
    num_keys: int
    num_sequences: int
    top_k: int
    logits_row_stride: int
    logits_dtype: DType
    index_dtype: str
    span_mode: str


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(
            compute=frozenset({DType.FP32}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_topk_prefill",
            function_name="profile_dsa_topk_prefill_torch",
        ),
        table_name=KIND,
        args_schema=DsaTopkPrefillArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cuda",
        supports=BackendSupport(
            compute=frozenset({DType.FP32}),
            gpus=frozenset({"NVIDIA H200", "NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_topk_prefill",
            function_name="profile_dsa_topk_prefill_vllm_cuda",
        ),
        table_name=KIND,
        args_schema=DsaTopkPrefillArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="sglang_cuda",
        supports=BackendSupport(
            compute=frozenset({DType.FP32}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_topk_prefill",
            function_name="profile_dsa_topk_prefill_sglang_cuda",
        ),
        table_name=KIND,
        args_schema=DsaTopkPrefillArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="sglang_env",
    )
)

# The vLLM fork's top_k_per_row_prefill (GLM-5.3-Flash production). The kpool
# indexer selects index_topk / index_kpool = 512 pools per row
# (`vllm::topKPerRowPrefill<512>` in capture 20260925_4).
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_fork_cuda",
        supports=BackendSupport(
            compute=frozenset({DType.FP32}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_topk_prefill",
            function_name="profile_dsa_topk_prefill_vllm_fork_cuda",
        ),
        table_name=KIND,
        args_schema=DsaTopkPrefillArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_fork_env",
    )
)
