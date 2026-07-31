"""DSA persistent decode top-k index-selection kernel kind."""

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

KIND: str = "dsa_persistent_topk_decode"


@dataclass(frozen=True)
class DsaPersistentTopkDecodeArgs(KernelArgs):
    batch_size: int
    context_len: int
    next_n: int
    max_model_len: int
    top_k: int
    logits_row_stride: int
    logits_dtype: DType
    index_dtype: str
    context_mode: str


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(
            compute=frozenset({DType.FP32}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_persistent_topk_decode",
            function_name="profile_dsa_persistent_topk_decode_torch",
        ),
        table_name=KIND,
        args_schema=DsaPersistentTopkDecodeArgs,
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
            gpus=frozenset({"NVIDIA H200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_persistent_topk_decode",
            function_name="profile_dsa_persistent_topk_decode_vllm_cuda",
        ),
        table_name=KIND,
        args_schema=DsaPersistentTopkDecodeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
    )
)
