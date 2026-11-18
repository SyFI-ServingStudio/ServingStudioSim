"""Grouped convolution finish, residual add and RMSNorm in one launch.

A DFlash2 draft layer closes each section with a grouped convolution over the
query block (each channel mixes its own value with the previous ``taps - 1``
positions of the same block, with per-group coefficients), then adds the
residual and RMS-normalizes. One launch does all three, writing the updated
residual and the normalized output. It replaces the convolution kernel and the
fused add + RMSNorm.
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

KIND = "grouped_conv_residual_rms_norm"


@dataclass(frozen=True)
class GroupedConvResidualRmsNormArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Rows, whole blocks of block_len.")
    hidden: int = arg(unit="elements", doc="Channels per row.")
    group_size: int = arg(unit="elements", doc="Channels sharing one coefficient.")
    taps: int = arg(unit="positions", doc="Convolution taps within a block.")
    block_len: int = arg(unit="tokens", doc="Rows per block; taps do not cross blocks.")
    dtype: DType = arg(doc="Element type of the activations and residual.")


DOC = KernelDoc(
    title="Grouped conv finish + residual RMSNorm",
    summary=(
        "Finish a block-local grouped convolution, add the residual and "
        "RMS-normalize in one launch."
    ),
    description=(
        "The vLLM fork's Beam conv_residual_norm: per row, a grouped convolution "
        "over the previous taps of its block, the residual add (written back) and "
        "RMSNorm with the next section's weight."
    ),
    category="Normalization",
    subcategory="RMSNorm",
    formula=("GB/s = num_tokens · hidden · 8 bytes / time",),
    default_metric="time_ms",
    method=(
        f"{CUPTI_METHOD} Before timing, output and residual are compared with "
        "the incumbent dflash2_grouped_conv + vLLM fused_add_rms_norm (relative "
        "error below 1%)."
    ),
    caveats=(),
    reference=None,
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        supports=BackendSupport(compute=frozenset({DType.BF16})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.dflash2.draft_chain",
            function_name="profile_grouped_conv_residual_rms_norm_vllm_triton",
        ),
        table_name=KIND,
        args_schema=GroupedConvResidualRmsNormArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary="vLLM fork vllm.models.beam.nvidia.dflash2_conv_ops.conv_residual_norm."
        ),
    )
)

__all__ = ["GroupedConvResidualRmsNormArgs", "KIND"]
