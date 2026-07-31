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
