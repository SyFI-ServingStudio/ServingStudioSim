"""NCCL vocabulary all-gather including rank-major to vocabulary layout copy."""

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

KIND = "vocab_parallel_all_gather"


@dataclass(frozen=True)
class VocabParallelAllGatherArgs(KernelArgs):
    num_gpus: int
    num_rows: int
    vocab_size_per_rank: int
    dtype: DType
    fabric: str = "nvlink"


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch_nccl",
        runner_ref=RunnerRef(
            module_name="profiling.runners.comm.vocab_parallel_all_gather",
            function_name="profile_vocab_parallel_all_gather_batch",
        ),
        table_name=KIND,
        args_schema=VocabParallelAllGatherArgs,
        supports=BackendSupport(
            compute=frozenset({DType.BF16, DType.FP32}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        metric_family=MetricFamily.COMM,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        gpu_count_fn=lambda spec: int(spec["num_gpus"]),
        list_native=True,
    )
)
