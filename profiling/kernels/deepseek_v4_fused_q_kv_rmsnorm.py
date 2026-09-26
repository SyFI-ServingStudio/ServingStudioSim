"""DeepSeek V4 fused QR/KV RMSNorm."""

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

KIND = "deepseek_v4_fused_q_kv_rmsnorm"


@dataclass(frozen=True)
class DeepseekV4FusedQKvRmsnormArgs(KernelArgs):
    num_tokens: int
    q_dim: int
    kv_dim: int
    rms_eps: float
    dtype: DType


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.deepseek_v4_fused_q_kv_rmsnorm_vllm_triton",
            function_name="profile_deepseek_v4_fused_q_kv_rmsnorm_vllm_triton",
        ),
        table_name=KIND,
        args_schema=DeepseekV4FusedQKvRmsnormArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=BackendSupport(compute=frozenset({DType.BF16}), gpus=frozenset({"NVIDIA H200"})),
        subprocess_env="vllm_env",
    )
)

# Alignment-fork shared op (GLM-5.3-Flash MLA front end, rms_eps 1e-5).
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_fork_triton",
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.deepseek_v4_fused_q_kv_rmsnorm_vllm_triton",
            function_name="profile_deepseek_v4_fused_q_kv_rmsnorm_vllm_fork_triton",
        ),
        table_name=KIND,
        args_schema=DeepseekV4FusedQKvRmsnormArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=BackendSupport(compute=frozenset({DType.BF16}), gpus=frozenset({"NVIDIA B200"})),
        subprocess_env="vllm_fork_env",
    )
)

__all__ = ["DeepseekV4FusedQKvRmsnormArgs", "KIND"]
