"""FlashInfer TRT-LLM shape-aware standalone all-reduce kernel kind.

Unlike the generic byte-keyed ``all_reduce`` kind, vLLM's FlashInfer path
requires a contiguous ``[num_tokens, hidden_dim]`` tensor and changes its PDL
completion policy at 16 tokens. Those two dimensions therefore belong in this
kernel's cache identity.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import BackendDoc, KernelDoc, arg
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
    num_gpus: int = arg(unit="GPUs", doc="GPUs in the tensor-parallel group.")
    num_tokens: int = arg(unit="tokens", doc="Token rows in each GPU's input tensor.")
    hidden_dim: int = arg(unit="elements", doc="Features in each input row.")
    dtype: DType = arg(doc="Element type of the input tensor.")
    fabric: str = arg(
        doc=(
            "Interconnect label used to identify the measurement; the run uses the "
            "reserved GPUs' links."
        )
    )


DOC = KernelDoc(
    title="All-reduce, FlashInfer TRT-LLM kernel",
    summary=(
        "Sum a [num_tokens, hidden_dim] tensor across the GPUs of a "
        "tensor-parallel group with FlashInfer's TRT-LLM all-reduce."
    ),
    description=(
        "vLLM can end a tensor-parallel block with FlashInfer's TRT-LLM "
        "all-reduce instead of NCCL. Its cost depends on the [num_tokens, "
        "hidden_dim] shape rather than on the bytes alone, and vLLM changes its"
        " launch at 16 tokens, so it is measured by shape, unlike the "
        "byte-keyed all_reduce. Each GPU contributes a random normal tensor."
    ),
    category="Communication",
    formula=(
        "output = Σᵣ inputᵣ, on each GPU",
        "algbw = num_tokens·hidden_dim·bytes per element / time",
        "busbw = algbw · 2(num_gpus − 1) / num_gpus",
    ),
    default_metric="busbw_gbps",
    method=(
        "Wall-clock time of CUDA graph replays, since vLLM runs this collective"
        " inside a graph. Each graph holds one all-reduce. After 50 direct "
        "calls and 50 replays, 100 replays are timed from a barrier to a device"
        " synchronize, and the slowest rank's mean is kept."
    ),
    caveats=(
        "PDL is on, and completion is signaled at kernel end only above 16 "
        "tokens, as vLLM launches it.",
        "Host time around graph replays includes the replay launch cost, not only kernel time.",
        "The fabric argument labels the row; the run uses whatever links "
        "connect the reserved GPUs.",
    ),
    # No separate PyTorch reference implementation exists for this kind.
    reference=None,
)


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
        doc=BackendDoc(
            summary=(
                "FlashInfer's TRT-LLM allreduce_fusion call with the kAllReduce "
                "pattern and PDL enabled."
            ),
            url="https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/comm/allreduce.py",
        ),
    )
)
