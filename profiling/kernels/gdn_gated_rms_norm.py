"""Qwen GDN gated RMS normalization kernel kind.

The initial ``torch`` backend measures the complete multi-launch semantic
reference. It is a correctness/performance baseline, not the production fused
normalization launch, and must not be selected for production simulation after
the vLLM backend is registered.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import CUPTI_METHOD, BackendDoc, KernelDoc, arg
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND: str = "gdn_gated_rms_norm"


@dataclass(frozen=True)
class GdnGatedRmsNormArgs(KernelArgs):
    m: int = arg(unit="rows", doc="Value-head rows across the token batch.")
    hidden: int = arg(unit="elements", doc="Features normalized within each row.")
    dtype: DType = arg(doc="Element type of the activation, gate, weight and output.")


DOC = KernelDoc(
    title="Gated DeltaNet output RMSNorm",
    summary="RMS-normalize each recurrent output row and multiply it by a SiLU gate.",
    description=(
        "The last step of Qwen3.6's Gated DeltaNet block before the output "
        "projection: each value head's output row is RMS-normalized, scaled by "
        "a learned weight and multiplied by a SiLU gate. m counts rows, one per"
        " token per value head, and hidden is the row width. Statistics and "
        "gating are computed in FP32 and the output is rounded to BF16."
    ),
    category="Normalization",
    formula=(
        "y = (x / √(mean(x²) + 1e−6) · weight) · (z · sigmoid(z))",
        "FLOPs = 7·m·hidden + 2·m",
        "bytes = (3·m·hidden + hidden)·bytes(dtype)",
        "TFLOPS = FLOPs / time",
        "GB/s = bytes / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "torch counts every launch of the reference; vllm_triton counts only "
        "layer_norm_fwd_kernel. Allocation and the vLLM output check run before"
        " timing."
    ),
    caveats=(
        "torch uses several launches; vllm_triton fuses normalization and gating in one.",
        "GB/s counts logical input and output bytes, not the torch backend's FP32 intermediates.",
    ),
    reference="profiling.runners.attention.gdn_gated_rms_norm_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(compute=frozenset({DType.BF16})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.gdn_gated_rms_norm_torch",
            function_name="profile_gdn_gated_rms_norm",
        ),
        table_name=KIND,
        args_schema=GdnGatedRmsNormArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "The PyTorch reference, timed across its separate RMSNorm and SiLU-gate launches."
            )
        ),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        runner_ref=RunnerRef(
            module_name=("profiling.runners.attention.gdn_gated_rms_norm_vllm_triton"),
            function_name="profile_gdn_gated_rms_norm_vllm_triton",
        ),
        table_name=KIND,
        args_schema=GdnGatedRmsNormArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary="vLLM's rmsnorm_fn Triton call fuses rowwise RMSNorm and the SiLU output gate.",
            url="https://github.com/vllm-project/vllm/blob/main/vllm/third_party/flash_linear_attention/ops/layernorm_guard.py",
        ),
        subprocess_env="vllm_env",
    )
)
