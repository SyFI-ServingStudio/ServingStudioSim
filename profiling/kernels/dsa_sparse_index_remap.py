"""GLM-5.2 request-local to global sparse-index remap kernel kind."""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import KernelArgs
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND: str = "dsa_sparse_index_remap"


@dataclass(frozen=True)
class DsaSparseIndexRemapArgs(KernelArgs):
    num_queries: int
    num_requests: int
    selected_k: int
    block_size: int
    max_blocks_per_request: int
    request_row_counts: str
    local_span_lengths: str
    valid_counts: str
    index_distribution: str
    page_table_mapping: str
    workspace_partition: str
    return_valid_counts: bool
    index_dtype: str


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(
            compute=None,
            gpus=frozenset({"NVIDIA H200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_sparse_index_remap",
            function_name="profile_dsa_sparse_index_remap_torch",
        ),
        table_name=KIND,
        args_schema=DsaSparseIndexRemapArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
    )
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        supports=BackendSupport(
            compute=None,
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_sparse_index_remap",
            function_name="profile_dsa_sparse_index_remap_vllm_triton",
        ),
        table_name=KIND,
        args_schema=DsaSparseIndexRemapArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
    )
)


# Alignment-fork callable (GLM-5.3-Flash). selected_k is the index-table width:
# 2048 (GLM-5.2 layout) or 2176 (kpool round_up(2048 + 4 - 1, 128)).
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_fork_triton",
        supports=BackendSupport(
            compute=None,
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_sparse_index_remap",
            function_name="profile_dsa_sparse_index_remap_vllm_fork_triton",
        ),
        table_name=KIND,
        args_schema=DsaSparseIndexRemapArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_fork_env",
    )
)
