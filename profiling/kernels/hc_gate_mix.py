"""Hyper-connection block input: gated mean over the normalized residual streams."""

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

KIND = "hc_gate_mix"


@dataclass(frozen=True)
class HcGateMixArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Token rows.")
    hidden_size: int = arg(unit="elements", doc="Features in each residual stream.")
    hc_mult: int = arg(unit="streams", doc="Parallel residual streams per token.")
    hidden_dtype: DType = arg(doc="Element type of the streams, block output and outputs.")


DOC = KernelDoc(
    title="HC gate mix",
    summary=(
        "Collapse the normalized residual streams into one block input with a "
        "per-feature sigmoid gate."
    ),
    description=(
        "The last step of a gated hyper-connection module: the low-rank up "
        "projection produces a gate for every stream feature, and the block input "
        "is the mean over the hc_mult streams of sigmoid(gate)·x, where x is the "
        "per-stream normalized residual. One Triton program per (token, "
        "512-feature tile) loops over the streams. Inputs are seeded random bf16 rows."
    ),
    category="Normalization",
    subcategory="Hyper-connections",
    formula=(
        "out = (1 / hc_mult) · Σ_j σ(gate_j) · x_j",
        "GB/s = 2 · num_tokens · (2·hc_mult·hidden_size + hidden_size) / time",
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
            function_name="profile_hc_gate_mix_vllm_triton",
        ),
        table_name=KIND,
        args_schema=HcGateMixArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary="The fork's hc_gate_mix Triton op.",
            url="https://github.com/serendipity-zk/vllm/blob/892da0822f7e866cec8e519727f014c45a5ad792/vllm/models/qwen4_exp/nvidia/ops/hc.py",
        ),
    )
)

__all__ = ["KIND", "HcGateMixArgs"]
