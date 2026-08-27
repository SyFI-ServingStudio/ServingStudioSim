"""DSA paged-decode MQA-logits kernel kind."""

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

KIND: str = "dsa_paged_mqa_logits_decode"


@dataclass(frozen=True)
class DsaPagedMqaLogitsDecodeArgs(KernelArgs):
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
        backend="torch",
        supports=BackendSupport(
            compute=frozenset({DType.FP8_E4M3}),
            kv=frozenset({DType.FP8_E4M3}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_paged_mqa_logits_decode",
            function_name="profile_dsa_paged_mqa_logits_decode_torch",
        ),
        table_name=KIND,
        args_schema=DsaPagedMqaLogitsDecodeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_deepgemm_fp8",
        supports=BackendSupport(
            compute=frozenset({DType.FP8_E4M3}),
            kv=frozenset({DType.FP8_E4M3}),
            gpus=frozenset({"NVIDIA H200", "NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_paged_mqa_logits_decode",
            function_name=("profile_dsa_paged_mqa_logits_decode_vllm_deepgemm_fp8"),
        ),
        table_name=KIND,
        args_schema=DsaPagedMqaLogitsDecodeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
    )
)
