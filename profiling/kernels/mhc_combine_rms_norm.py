"""DeepSeek-V4 mHC stream collapse + RMSNorm of the sublayer input, as SGLang runs it.

Source (SGLang v0.5.23): ``srt/models/deepseek_v4_mhc.py`` ``_combine`` (line 467) ->
``kernels/ops/layernorm/hc_combine_norm.py`` ``hc_combine_norm`` (line 55): one Triton launch,
``_hc_combine_norm`` (1/2/4 CTAs per row for <= 8 / <= 48 / more rows) or
``_hc_combine_norm_prefill`` (one CTA per row from 4096 rows). It computes
``y = RMSNorm(sum_k pre[k] · R[k])`` with the sublayer's norm weight, rounding the combine to bf16
first. SGLang takes it on Blackwell (SM10x and SM12x) for bf16, hc_mult 4, hidden 5120 -- the
DeepSeek-V4.1-Flash geometry the kernel hardcodes. The MXFP8-quantized variant
(``hc_combine_norm_mxfp8``) is only taken when the next GEMM consumes MXFP8 input, which the block-32
FP8 linears on SM12x do not.

Not taken when the previous post already produced the normed input (``HcNormed`` / ``HcQuantized``
short-circuit, e.g. the fused all-reduce + post + combine-norm kernels or the >= 4096-row
``mhc_post_combine_norm_prefill`` seam); the worklet decides whether the op runs.
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

KIND = "mhc_combine_rms_norm"


@dataclass(frozen=True)
class MhcCombineRmsNormArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Token rows.")
    hidden_size: int = arg(unit="elements", doc="Features in each residual stream.")
    hc_mult: int = arg(unit="streams", doc="Parallel residual streams per token.")
    hidden_dtype: DType = arg(doc="Element type of the streams, the norm weight and the output.")


DOC = KernelDoc(
    title="mHC combine + RMSNorm",
    summary=(
        "Collapse a token's hc_mult residual streams with the pre coefficients and RMS-normalize "
        "the result into the next sublayer's input, in one launch."
    ),
    description=(
        "The second step of a DeepSeek-V4 mHC pre-block: y = RMSNorm(Σ_k pre[k]·R[k]) · w, with "
        "the combine rounded to bf16 before the norm as the unfused op does. One Triton program per "
        "row (split over the hidden columns for small batches). Inputs are seeded random bf16 "
        "streams, fp32 pre coefficients and a bf16 norm weight."
    ),
    category="Normalization",
    subcategory="Hyper-connections",
    formula=(
        "y = bf16(Σ_k pre[k]·R[k]) · rsqrt(mean(·²) + ε) · w",
        "GB/s = num_tokens · (hc_mult + 1) · hidden_size · 2 / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} Compared with a PyTorch fp32 reference of the same math before timing."
    ),
    caveats=("The SGLang kernel only exists for hc_mult 4 and hidden_size 5120.",),
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="sglang_triton",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            min_compute_capability=(8, 0),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.mhc.mhc_sglang_triton",
            function_name="profile_mhc_combine_rms_norm_sglang_triton",
        ),
        table_name=KIND,
        args_schema=MhcCombineRmsNormArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="sglang_env",
        doc=BackendDoc(
            summary="SGLang hc_combine_norm (kernels/ops/layernorm/hc_combine_norm.py:55).",
            url="https://github.com/sgl-project/sglang",
        ),
    )
)

__all__ = ["KIND", "MhcCombineRmsNormArgs"]
