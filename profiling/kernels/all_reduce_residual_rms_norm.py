"""FlashInfer TRT-LLM fused all-reduce + residual + RMSNorm kernel kind.

This is deliberately separate from ``all_reduce``: the production vLLM path
launches one ``kARResidualRMSNorm`` device kernel whose grid depends on the 2-D
``[num_tokens, hidden_dim]`` shape. Recording it as a pure byte-keyed collective
would lose that shape and double-count the separately modeled norm.
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

KIND: str = "all_reduce_residual_rms_norm"


@dataclass(frozen=True)
class AllReduceResidualRmsNormArgs(KernelArgs):
    num_gpus: int
    num_tokens: int
    hidden_dim: int
    dtype: DType
    fabric: str
    strategy: str
    launch_with_pdl: bool
    trigger_completion_at_end: bool
    fp32_acc: bool


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_trtllm",
        # The alignment target is the validated H200 BF16 vLLM path. Broaden
        # this declaration only after a real smoke on each added dtype/GPU.
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.comm.flashinfer_trtllm",
            function_name="profile_all_reduce_residual_rms_norm_batch",
        ),
        table_name=KIND,
        args_schema=AllReduceResidualRmsNormArgs,
        metric_family=MetricFamily.COMM,
        batch_outlier_policy=BatchOutlierPolicy(),
        # FlashInfer 0.6.11 in the project environment implements the same
        # TRT-LLM fused kernel contract and boots its IPC workspace reliably.
        # The vLLM Torch 2.11 environment currently fails during the symmetric
        # workspace bootstrap before the kernel can launch.
        subprocess_env="flashinfer_pip_env",
        gpu_count_fn=lambda spec: int(spec["num_gpus"]),
        list_native=True,
    )
)
