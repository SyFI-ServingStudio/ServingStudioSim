"""Delayed hyper-connection combine fused with the next HC module's per-stream Gemma RMSNorm."""

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

KIND = "hc_combine_norm"


@dataclass(frozen=True)
class HcCombineNormArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Token rows.")
    hidden_size: int = arg(unit="elements", doc="Features in each residual stream.")
    hc_mult: int = arg(unit="streams", doc="Parallel residual streams per token.")
    hidden_dtype: DType = arg(doc="Element type of the streams, block output and outputs.")


DOC = KernelDoc(
    title="HC combine + per-stream norm",
    summary=(
        "Add a block's output into every residual stream with a learned gate, then "
        "RMS-normalize each stream, in one launch."
    ),
    description=(
        "Gated hyper-connections keep hc_mult residual streams of hidden_size "
        "features per token (stream-outer). The combine of a finished block is "
        "delayed to the next HC module, where one Triton program per (token, "
        "stream) adds coeff·block to the stream, stores the bf16 result, and "
        "normalizes it with a Gemma (1 + w) per-stream weight. coeff = "
        "2·sigmoid(logit / hc_mult) from the previous module's injection logits. "
        "Inputs are seeded random bf16 rows; the logits are a contiguous [tokens, "
        "hc_mult] tensor."
    ),
    category="Normalization",
    subcategory="Hyper-connections",
    formula=(
        "s_j ← s_j + 2·σ(logit_j / hc_mult) · block; y_j = s_j / √(mean(s_j²) + ε) · (1 + w_j)",
        "GB/s = 2 · (num_tokens · (3·hc_mult·hidden_size + hidden_size + hc_mult) "
        "+ hc_mult·hidden_size) / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} Three warm-up calls and a comparison with an independent "
        "PyTorch reference precede capture; every launch of the call is counted."
    ),
    caveats=(
        "Without injection logits (unit coefficient, the MTP layer's first combine) "
        "the op compiles a variant that has no separate row: on B200 it is about "
        "0.3-0.4 µs faster below 256 rows and within 3% above 1k rows.",
        "The production logits are a strided slice of the down-projection output; "
        "that ran within 1% of the contiguous logits used here.",
    ),
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
            function_name="profile_hc_combine_norm_vllm_triton",
        ),
        table_name=KIND,
        args_schema=HcCombineNormArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary="The fork's hc_combine_norm Triton op (learned injection logits).",
            url="https://github.com/serendipity-zk/vllm/blob/892da0822f7e866cec8e519727f014c45a5ad792/vllm/models/qwen4_exp/nvidia/ops/hc.py",
        ),
    )
)

__all__ = ["KIND", "HcCombineNormArgs"]
