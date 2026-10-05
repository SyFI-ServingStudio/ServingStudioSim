"""Fused RMSNorm of the query-side and KV-side projections."""

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

KIND = "q_kv_rms_norm"


@dataclass(frozen=True)
class QKvRmsNormArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Tokens normalized together.")
    q_dim: int = arg(unit="elements", doc="Width of each query-side input row.")
    kv_dim: int = arg(unit="elements", doc="Width of each KV-side input row.")
    rms_eps: float = arg(
        unit="unitless",
        doc="Constant added to each mean square before RMS normalization.",
    )
    dtype: DType = arg(doc="Element type of the inputs and weights.")


DOC = KernelDoc(
    title="Query and KV RMSNorm",
    summary="RMS-normalize query-side and KV-side projections with separate weights.",
    description=(
        "After the attention input projections, the query-side rows (q_dim "
        "wide) and the KV-side rows (kv_dim wide) are RMS-normalized with their"
        " own weights in one call, before the queries and cache entries are "
        "built. Inputs and weights are seeded random BF16 values."
    ),
    category="Normalization",
    subcategory="RMSNorm",
    formula=(
        "yq = q / √(mean(q²) + rms_eps) · q_weight",
        "ykv = kv / √(mean(kv²) + rms_eps) · kv_weight",
        "GB/s = 2 · (2 · num_tokens · (q_dim + kv_dim) + q_dim + kv_dim) / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(f"{CUPTI_METHOD} Three warm-up calls run first; every launch of the call is counted."),
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.q_kv_rms_norm_vllm_triton",
            function_name="profile_q_kv_rms_norm_vllm_triton",
        ),
        table_name=KIND,
        args_schema=QKvRmsNormArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=BackendSupport(compute=frozenset({DType.BF16})),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary="vLLM's fused_q_kv_rmsnorm Triton call normalizes both projections together.",
            url="https://github.com/vllm-project/vllm/blob/main/vllm/models/common/ops/fused_qk_rmsnorm.py",
        ),
    )
)

__all__ = ["QKvRmsNormArgs", "KIND"]
