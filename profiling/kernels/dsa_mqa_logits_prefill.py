"""DSA prefill MQA-logits kernel kind."""

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

KIND: str = "dsa_mqa_logits_prefill"


@dataclass(frozen=True)
class DsaMqaLogitsPrefillArgs(KernelArgs):
    num_queries: int
    num_keys: int
    num_sequences: int
    num_heads: int
    head_dim: int
    q_dtype: DType
    k_dtype: DType
    k_scale_dtype: DType
    weight_dtype: DType
    output_dtype: DType
    span_mode: str
    clean_logits: bool


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(
            compute=frozenset({DType.FP8_E4M3}),
            kv=frozenset({DType.FP8_E4M3}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_mqa_logits_prefill",
            function_name="profile_dsa_mqa_logits_prefill_torch",
        ),
        table_name=KIND,
        args_schema=DsaMqaLogitsPrefillArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
    )
)
