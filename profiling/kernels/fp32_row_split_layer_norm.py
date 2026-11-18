"""Split an fp32 GEMM output row into bf16 spans and a LayerNorm span."""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import KernelArgs
from profiling.db.doc import CUPTI_METHOD, BackendDoc, KernelDoc, arg
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND = "fp32_row_split_layer_norm"


@dataclass(frozen=True)
class Fp32RowSplitLayerNormArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Rows of the fp32 input, one per token.")
    bf16_lead_cols: int = arg(
        unit="elements", doc="Leading columns of each row, rounded to bf16 into their own buffer."
    )
    norm_cols: int = arg(
        unit="elements",
        doc="Next columns, LayerNormed (fp32 math on their bf16-rounded values) into bf16.",
    )
    bf16_trail_cols: int = arg(
        unit="elements",
        doc="Columns after the LayerNorm span, rounded to bf16 into their own buffer.",
    )
    fp32_tail_cols: int = arg(
        unit="elements", doc="Last columns of each row, left in place as the fp32 output."
    )


DOC = KernelDoc(
    title="FP32 row split with LayerNorm",
    summary="Round spans of an fp32 GEMM output row to bf16 and LayerNorm one span, in one launch.",
    description=(
        "When several projections of the same activation are computed as one "
        "GEMM over concatenated weights with an fp32 output (so a column group "
        "that needs fp32 keeps it), one epilogue launch turns each row back into "
        "what the separate projections produced: the leading bf16_lead_cols "
        "columns and the bf16_trail_cols columns after the LayerNorm span are "
        "rounded to bf16 into their own buffers, the norm_cols columns between "
        "them are rounded to bf16 and LayerNormed with an fp32 weight and bias "
        "into bf16, and the fp32_tail_cols last columns are not touched (callers "
        "read them in place). Rounding before the norm matches a bf16 GEMM "
        "output feeding an fp32 LayerNorm. Inputs are seeded random fp32 rows."
    ),
    category="Normalization",
    subcategory="LayerNorm",
    formula=(
        "lead = bf16(y[:, :L]); trail = bf16(y[:, L+N:L+N+T])",
        "normed = bf16(LayerNorm(float(bf16(y[:, L:L+N])), weight, bias, eps))",
        "GB/s = [6·num_tokens·(L + N + T) + 8·N] / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} One launch per call; three warm-up calls run first. "
        "Before timing, the output is checked against a Torch reference: the "
        "two bf16 spans bit-identical, the LayerNorm span within 1 bf16 ulp "
        "(or 1e-6 where it cancels to near zero), "
        "and the fp32 row unmodified."
    ),
    caveats=(
        "The triton backend has its row geometry compiled in (2048, 128, 128, "
        "32 columns) and refuses any other.",
    ),
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="triton",
        runner_ref=RunnerRef(
            module_name="profiling.runners.norm.fp32_row_split_layer_norm_triton",
            function_name="profile_fp32_row_split_layer_norm_triton",
        ),
        table_name=KIND,
        args_schema=Fp32RowSplitLayerNormArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        # The kind fixes its dtypes (fp32 in, bf16 out, fp32 LayerNorm
        # parameters), so there is no dtype axis. No capability rule: a
        # portable Triton kernel (plain loads, sums, stores).
        supports=BackendSupport(compute=None),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "A repo-vendored Triton kernel, one program per row: the whole "
                "row in registers, the LayerNorm reduction with tl.sum."
            ),
            url=None,
        ),
    )
)

__all__ = ["Fp32RowSplitLayerNormArgs", "KIND"]
