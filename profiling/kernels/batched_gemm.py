"""Batched-GEMM kernel kind with model production-layout backend identities.

The backend names deliberately freeze model-specific storage layouts (GLM's
Q-absorption and V-up, DeepSeek-V4.1's ``wo_a`` einsum input) instead of
presenting their constants as generic batched GEMM behavior. ``dtype`` is the
format of both operands; the output is bf16.
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

# DeepSeek-V4.1 wo_a (vLLM fork): one DeepGEMM fp8_einsum "bhr,hdr->bhd" over
# the local wo_a groups. Both operands are MXFP8 (e4m3 + ue8m0 per 32 K), the
# activation is the mega-attention output in its padded 8-slot layout, bf16 out.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="deepgemm_mxfp8_einsum_dsv41_wo_a",
        supports=BackendSupport(
            compute=frozenset({DType.MXFP8_E4M3}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.deepgemm_mxfp8_einsum",
            function_name="profile_batched_gemm_deepgemm_mxfp8_einsum_dsv41_wo_a",
        ),
        table_name=KIND,
        args_schema=BatchedGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_fork_env",
    )
)
