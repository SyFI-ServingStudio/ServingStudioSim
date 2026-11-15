"""DeepInfer's RMSNorm with a fused MXFP8 cast (``tilelang_ops/norm.py`` ``norm``).

Wire string: ``"rms_norm_mxfp8"``.

One ``_norm_kernel`` launch. With ``fp8_attn`` (asserted on in
``mla_attn.py``) DeepInfer's ``RMSNorm`` passes ``get_fp8_cast_params()``:
E4M3, 32 elements per scale, rounded UE8M0 scales packed into int32 in the
TMA-aligned column-major layout DeepGEMM reads. DeepSeek-V4.1 prefill runs it
as ``q_a_norm`` ([T, q_lora_rank], FP8 only) and as the attention layer norm
after a shifted mHC pre ([T, hidden], FP8 plus the bf16 rows,
``return_bf16=True``).

Why a new kind: ``rms_norm`` writes bf16 only; this boundary also quantizes.
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

KIND = "rms_norm_mxfp8"


@dataclass(frozen=True)
class RmsNormMxfp8Args(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Rows normalized.")
    hidden: int = arg(unit="elements", doc="Features normalized per row.")
    return_bf16: bool = arg(doc="Whether the bf16 normalized rows are written too.")
    dtype: DType = arg(doc="Input and weight element type (bf16).")


DOC = KernelDoc(
    title="RMSNorm with MXFP8 cast",
    summary="RMSNorm whose output is cast to E4M3 with packed UE8M0 scales in one launch.",
    description=(
        "Each row is RMS-normalized, scaled by the weight and cast to E4M3 with "
        "one rounded UE8M0 scale per 32 elements; the scales are packed four to "
        "an int32 in the TMA-aligned column-major layout. Optionally the bf16 "
        "normalized row is written too."
    ),
    category="Normalization",
    subcategory="RMSNorm",
    formula=("bytes = 2·T·hidden read + T·hidden (+ 2·T·hidden for bf16) + T·hidden/32 written",),
    default_metric="time_ms",
    method=CUPTI_METHOD,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="tilelang",
        supports=BackendSupport(compute=frozenset({DType.BF16}), min_compute_capability=(9, 0)),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.deepinfer_kv_tilelang",
            function_name="profile_rms_norm_mxfp8_tilelang",
        ),
        table_name=KIND,
        args_schema=RmsNormMxfp8Args,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="deepinfer_env",
        doc=BackendDoc(
            summary=(
                "DeepInfer's TileLang norm with get_fp8_cast_params(), checked by "
                "DeepInfer's test_norm at the same hidden size and output form."
            ),
        ),
    )
)
