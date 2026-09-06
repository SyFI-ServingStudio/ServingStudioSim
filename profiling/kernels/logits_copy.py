"""Typed logits conversion or clone into contiguous storage."""

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

KIND = "logits_copy"


@dataclass(frozen=True)
class LogitsCopyArgs(KernelArgs):
    num_rows: int
    vocab_size: int
    row_stride: int
    input_dtype: DType
    output_dtype: DType


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        runner_ref=RunnerRef(
            module_name="profiling.runners.logits.torch",
            function_name="profile_logits_copy",
        ),
        table_name=KIND,
        args_schema=LogitsCopyArgs,
        supports=BackendSupport(
            compute=frozenset({DType.BF16, DType.FP32}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
    )
)
