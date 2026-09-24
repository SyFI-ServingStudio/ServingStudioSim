"""Batched-GEMM kernel kind with GLM production-layout backend identities.

The backend names deliberately freeze the model-specific Q-absorption and V-up
storage layouts instead of presenting their constants as generic batched GEMM
behavior.
"""

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

KIND: str = "batched_gemm"


@dataclass(frozen=True)
class BatchedGemmArgs(KernelArgs):
    num_batches: int
    m: int
    n: int
    k: int
    dtype: DType


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch_mla_q_absorb_glm52",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA H200", "NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.batched_gemm",
            function_name="profile_mla_q_absorb_glm52",
        ),
        table_name=KIND,
        args_schema=BatchedGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch_mla_v_up_glm52",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA H200", "NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.batched_gemm",
            function_name="profile_mla_v_up_glm52",
        ),
        table_name=KIND,
        args_schema=BatchedGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
    )
)


# GLM-5.3-Flash on the alignment fork: qk_nope 256 (no RoPE part), v_head 256,
# kv_lora 512, unpadded attention output. Same torch.bmm call as GLM-5.2.
for _backend, _function in (
    ("torch_mla_q_absorb_glm53", "profile_mla_q_absorb_glm53"),
    ("torch_mla_v_up_glm53", "profile_mla_v_up_glm53"),
):
    register(
        KernelProfilerSpec(
            kernel_kind=KIND,
            backend=_backend,
            supports=BackendSupport(
                compute=frozenset({DType.BF16}),
                gpus=frozenset({"NVIDIA B200"}),
            ),
            runner_ref=RunnerRef(
                module_name="profiling.runners.gemm.batched_gemm",
                function_name=_function,
            ),
            table_name=KIND,
            args_schema=BatchedGemmArgs,
            metric_family=MetricFamily.COMPUTE,
            batch_outlier_policy=BatchOutlierPolicy(),
            subprocess_env="vllm_fork_env",
        )
    )
