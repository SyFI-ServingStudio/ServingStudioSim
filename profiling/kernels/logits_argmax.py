"""Torch first-max greedy vocabulary sampling."""

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

KIND = "logits_argmax"


@dataclass(frozen=True)
class LogitsArgmaxArgs(KernelArgs):
    num_rows: int
    vocab_size: int
    row_stride: int
    dtype: DType


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        runner_ref=RunnerRef(
            module_name="profiling.runners.logits.torch",
            function_name="profile_logits_argmax",
        ),
        table_name=KIND,
        args_schema=LogitsArgmaxArgs,
        supports=BackendSupport(
            compute=frozenset({DType.BF16, DType.FP32}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
    )
)
