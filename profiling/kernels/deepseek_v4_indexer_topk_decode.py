"""DeepSeek V4 indexer decode persistent top-k operation."""

from profiling.db.args import DType
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)
from profiling.kernels.dsa_persistent_topk_decode import DsaPersistentTopkDecodeArgs

KIND = "deepseek_v4_indexer_topk_decode"


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cuda",
        runner_ref=RunnerRef(
            module_name=(
                "profiling.runners.attention.deepseek_v4_indexer_topk_decode_cuda"
            ),
            function_name="profile_deepseek_v4_indexer_topk_decode_cuda",
        ),
        table_name=KIND,
        args_schema=DsaPersistentTopkDecodeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=BackendSupport(
            compute=frozenset({DType.FP32}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        subprocess_env="vllm_env",
    )
)

__all__ = ["KIND"]
