"""DeepSeek learned/hash sqrt-softplus routing as one physical CUDA kind."""

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

KIND = "moe_topk_softplus_sqrt"


@dataclass(frozen=True)
class MoeTopkSoftplusSqrtArgs(KernelArgs):
    selection_mode: str
    num_tokens: int
    num_experts: int
    top_k: int
    hash_vocab_size: int
    logits_dtype: DType


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cuda",
        runner_ref=RunnerRef(
            module_name="profiling.runners.moe.moe_topk_softplus_sqrt_vllm_cuda",
            function_name="profile_moe_topk_softplus_sqrt_vllm_cuda",
        ),
        table_name=KIND,
        args_schema=MoeTopkSoftplusSqrtArgs,
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
