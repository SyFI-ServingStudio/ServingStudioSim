"""Grouped Gemma RMSNorm: one RMS per contiguous group of a row."""

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

KIND = "grouped_rms_norm"


@dataclass(frozen=True)
class GroupedRmsNormArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Token rows.")
    hidden_size: int = arg(unit="elements", doc="Features in each group, normalized together.")
    num_groups: int = arg(unit="groups", doc="Contiguous groups per row.")
    hidden_dtype: DType = arg(doc="Element type of the input, weight and output.")


DOC = KernelDoc(
    title="Grouped RMSNorm",
    summary="RMS-normalize each contiguous group of a row separately.",
    description=(
        "A row of num_groups · hidden_size features is split into num_groups "
        "contiguous groups, and each group is normalized by its own RMS and scaled "
        "by a Gemma (1 + w) weight that covers the whole row. Gated hyper-connections "
        "run it on the stream-outer residual (one group per stream) when an HC "
        "module has no pending combine to fuse, such as the first layer. One Triton "
        "program per (token, group). Inputs are seeded random bf16 rows."
    ),
    category="Normalization",
    subcategory="RMSNorm",
    formula=(
        "y_g = x_g / √(mean(x_g²) + ε) · (1 + w_g), per token and group",
        "GB/s = 2 · (2·num_tokens·num_groups·hidden_size + num_groups·hidden_size) / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} Three warm-up calls and a comparison with an independent "
        "PyTorch reference precede capture; every launch of the call is counted."
    ),
    caveats=(),
    reference="profiling.runners.mhc.hc_reference",
)


# A portable Triton kernel (bf16 storage, fp32 math, no tensor cores); Triton
# lowers bf16 only on SM80+. Programmatic dependent launch is enabled on SM90+
# by the op itself and is part of the measured launch.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            min_compute_capability=(8, 0),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.mhc.hc_vllm_triton",
            function_name="profile_grouped_rms_norm_vllm_triton",
        ),
        table_name=KIND,
        args_schema=GroupedRmsNormArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary="The fork's grouped_gemma_rmsnorm Triton op.",
            url="https://github.com/serendipity-zk/vllm/blob/892da0822f7e866cec8e519727f014c45a5ad792/vllm/models/qwen4_exp/nvidia/ops/hc.py",
        ),
    )
)

__all__ = ["KIND", "GroupedRmsNormArgs"]
