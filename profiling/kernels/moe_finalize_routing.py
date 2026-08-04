"""FlashInfer/TensorRT-LLM MoE finalize-routing kernel kind.

The kernel unpermutes EP-local expert outputs, applies router scales, and
reduces the local top-k contributions into one BF16 row per original token.
The following EP all-reduce is a separate communication kernel.
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

KIND: str = "moe_finalize_routing"


@dataclass(frozen=True)
class MoeFinalizeRoutingArgs(KernelArgs):
    token_count: int
    hidden_size: int
    top_k: int
    num_experts_per_rank: int
    local_routed_token_count: int
    dtype: DType


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_trtllm",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA H100", "NVIDIA H200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.moe.flashinfer_trtllm_finalize",
            function_name="profile_moe_finalize_routing_flashinfer_trtllm",
        ),
        table_name=KIND,
        args_schema=MoeFinalizeRoutingArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        # Compile against the same FlashInfer/Torch stack as the measured vLLM
        # run; this kernel's vendored source changed across FlashInfer releases.
        subprocess_env="vllm_env",
    )
)
