"""Materialize a delayed hyper-connection combine without a following norm."""

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

KIND = "hc_combine"


@dataclass(frozen=True)
class HcCombineArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Token rows.")
    hidden_size: int = arg(unit="elements", doc="Features in each residual stream.")
    hc_mult: int = arg(unit="streams", doc="Parallel residual streams per token.")
    hidden_dtype: DType = arg(doc="Element type of the streams, block output and outputs.")


DOC = KernelDoc(
    title="HC combine",
    summary="Add a block's output into every residual stream with a learned gate.",
    description=(
        "The standalone form of the delayed hyper-connection combine, used where "
        "something other than an HC module consumes the multi-stream residual "
        "next (for example an addition into the streams). One Triton program per "
        "(token, 512-feature tile) reads the block tile once and all hc_mult "
        "stream tiles, and writes the combined streams. coeff = 2·sigmoid(logit "
        "/ hc_mult). Inputs are seeded random bf16 rows."
    ),
    category="Normalization",
    subcategory="Hyper-connections",
    formula=(
        "s_j ← s_j + 2·σ(logit_j / hc_mult) · block",
        "GB/s = 2 · num_tokens · (2·hc_mult·hidden_size + hidden_size + hc_mult) / time",
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
            function_name="profile_hc_combine_vllm_triton",
        ),
        table_name=KIND,
        args_schema=HcCombineArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary="The fork's hc_combine Triton op (learned injection logits).",
            url="https://github.com/serendipity-zk/vllm/blob/892da0822f7e866cec8e519727f014c45a5ad792/vllm/models/qwen4_exp/nvidia/ops/hc.py",
        ),
    )
)

__all__ = ["KIND", "HcCombineArgs"]
