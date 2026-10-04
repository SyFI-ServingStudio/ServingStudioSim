"""FlashInfer TRT-LLM fused all-reduce + residual + RMSNorm kernel kind.

This is deliberately separate from ``all_reduce``: the production vLLM path
launches one ``kARResidualRMSNorm`` device kernel whose grid depends on the 2-D
``[num_tokens, hidden_dim]`` shape. Recording it as a pure byte-keyed collective
would lose that shape and double-count the separately modeled norm.
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

KIND: str = "all_reduce_residual_rms_norm"


@dataclass(frozen=True)
class AllReduceResidualRmsNormArgs(KernelArgs):
    num_gpus: int = arg(unit="GPUs", doc="GPUs in the tensor-parallel group.")
    num_tokens: int = arg(unit="tokens", doc="Token rows in each GPU's input tensor.")
    hidden_dim: int = arg(unit="elements", doc="Features in each input row.")
    dtype: DType = arg(doc="Element type of the input and residual tensors.")
    fabric: str = arg(
        doc=(
            "Interconnect label used to identify the measurement; the run uses the "
            "reserved GPUs' links."
        )
    )
    strategy: str = arg(doc="All-reduce strategy: auto, oneshot or twoshot.")
    launch_with_pdl: bool = arg(doc="Whether programmatic dependent launch is enabled.")
    trigger_completion_at_end: bool = arg(
        doc="Whether PDL completion is signaled at the end of the device kernel."
    )
    fp32_acc: bool = arg(doc="Whether the all-reduce uses fp32 accumulation.")


DOC = KernelDoc(
    title="All-reduce + residual add + RMSNorm",
    summary=(
        "Sum a tensor across GPUs, add the residual and RMS-normalize the result in a fused call."
    ),
    description=(
        "vLLM fuses the all-reduce at the end of a tensor-parallel block with "
        "the residual add and RMSNorm that follow into one FlashInfer TRT-LLM "
        "kernel, up to a per-GPU fusion-size limit, above which it runs them "
        "unfused. The input and residual are random normal [num_tokens, "
        "hidden_dim] tensors; the norm weight is all ones and ε is 1e-6."
    ),
    category="Communication",
    formula=(
        "s = Σᵣ inputᵣ + residual",
        "y = s / √(mean(s²) + 1e-6) · weight",
        "algbw = num_tokens·hidden_dim·bytes per element / time",
        "busbw = algbw · 2(num_gpus − 1) / num_gpus",
    ),
    default_metric="time_ms",
    method=(
        "Wall-clock time of CUDA graph replays, since vLLM runs this call "
        "inside a graph. Each graph holds ten fused calls. After 50 direct "
        "calls and 5 replays, 100 calls are timed from a barrier to a device "
        "synchronize, and the slowest rank's mean per call is kept."
    ),
    caveats=(
        "algbw and busbw divide the all-reduce input size by the time of the "
        "whole fused call, so they do not isolate link traffic from the add and"
        " the norm.",
        "Host time around graph replays includes the replay launch cost, not only kernel time.",
        "The fabric argument labels the row; the run uses whatever links "
        "connect the reserved GPUs.",
    ),
    # No separate PyTorch reference implements the entire fused call.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_trtllm",
        # The H200 and B200 BF16 paths use the same FlashInfer TRT-LLM fusion
        # contract; each GPU keeps independent measured rows in profile.db.
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
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
        doc=BackendDoc(
            summary=(
                "FlashInfer's TRT-LLM allreduce_fusion call with the kARResidualRMSNorm pattern."
            ),
            url="https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/comm/allreduce.py",
        ),
    )
)
