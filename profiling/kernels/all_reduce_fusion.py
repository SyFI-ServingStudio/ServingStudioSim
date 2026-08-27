"""FlashInfer TRT-LLM shape-aware standalone all-reduce kernel kind.

Unlike the generic byte-keyed ``all_reduce`` kind, vLLM's FlashInfer path
requires a contiguous ``[num_tokens, hidden_dim]`` tensor and changes its PDL
completion policy at 16 tokens. Those two dimensions therefore belong in this
kernel's cache identity.
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

KIND: str = "all_reduce_fusion"


@dataclass(frozen=True)
class AllReduceFusionArgs(KernelArgs):
    num_gpus: int
    num_tokens: int
    hidden_dim: int
    dtype: DType
    fabric: str


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_trtllm",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.comm.flashinfer_trtllm",
            function_name="profile_all_reduce_fusion_batch",
        ),
        table_name=KIND,
        args_schema=AllReduceFusionArgs,
        metric_family=MetricFamily.COMM,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="flashinfer_pip_env",
        gpu_count_fn=lambda spec: int(spec["num_gpus"]),
        list_native=True,
    )
)
