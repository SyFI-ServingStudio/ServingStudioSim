"""DeepInfer's fused sampler: temperature, top-p/top-k filter, log-softmax and sampling in one call.

Wire string: ``"fused_sample"`` -- matches Rust ``KernelSpec::KIND`` in
``simulator/src/timing/kernels/fused_sample.rs``.

``tilelang_ops/fused_sample.py`` ``fused_sample(logits, SamplingKernelInputs,
...)``: two TileLang launches (``_fused_sample_kernel``, a per-row log-sum-exp
pass, then ``_fused_sample_kernel_1``: radix-select thresholds for top-p and
top-k, the log-softmax write, and the seeded sample), one CTA per row. With a
draft distribution (``last_nextn_token_logprobs``) it also runs rejection
sampling (``do_reject_sampling``, a separate JIT specialization). DeepInfer
calls it from ``sampler_utils.process_sampling`` on the verify rows (rejection
on) and once per DFlash Markov step (rejection off), over the force_topk-wide
top-k values when ``DEEPINFER_SAMPLER_FORCE_TOPK`` is set, else over the
vocabulary.

Why a new kind: no existing kind samples; its cost is not a byte count (one
CTA's sequential radix passes per row, ~10 us at 384 x 1024 where the bytes
move in under 1 us).

Key: ``num_rows``; ``num_columns`` (the row width); ``reject_sampling``; and
whether the rows' top-p and top-k filters are active. The filters select the
radix passes: measured 7.1 us with neither, 12.5 us with top_p 0.95, 11.2 us
with top_k 50 (384 x 1024, sampling only; logs/20261120_2_ab), while the
temperature value and the input distribution moved it by under 6%.
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

KIND = "fused_sample"


@dataclass(frozen=True)
class FusedSampleArgs(KernelArgs):
    num_rows: int = arg(unit="rows", doc="Logit rows sampled, one per sampled position.")
    num_columns: int = arg(
        unit="elements", doc="Logits in each row (force_topk width or vocabulary)."
    )
    reject_sampling: bool = arg(doc="Whether the call verifies draft tokens by rejection sampling.")
    top_p_filter: bool = arg(doc="Whether the rows' top-p filter is active (top_p < 1).")
    top_k_filter: bool = arg(doc="Whether the rows' top-k filter is active (top_k > 0).")
    dtype: DType = arg(doc="Element type of the logits and logprobs (fp32).")


DOC = KernelDoc(
    title="Fused sampler (DeepInfer)",
    summary="Filter, normalize and sample each logit row, with optional rejection sampling.",
    description=(
        "One CTA per row scales the row by its temperature, finds the top-p and top-k "
        "thresholds by radix selection, writes the filtered log-softmax and draws a "
        "seeded sample; with rejection sampling it also accepts or replaces the row's "
        "draft token against the draft distribution. A first launch computes each "
        "row's log-sum-exp. Logits are seeded standard-normal; temperature 1, top_p "
        "0.95 and top_k 50 when the filter is active."
    ),
    category="Other",
    formula=("bytes = num_rows · num_columns · (4 + 4 + 4 · reject_sampling)",),
    default_metric="time_ms",
    method=CUPTI_METHOD + " Both launches of the call are counted.",
    caveats=(
        "Every row uses the same filter settings; a batch mixing filtered and "
        "unfiltered rows is not measured.",
        "Synthetic logits at temperature 1: DeepInfer's decode capture (temperature "
        "0.3, top_p 0.95) ran the verify call (384 rows) 1.34x and the Markov call "
        "(64 rows) 1.06x as long as these rows; the cause is not verified.",
    ),
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="tilelang",
        # No capability rule: TileLang JIT-compiles for the current device with
        # TMA lowering and warp specialization disabled (cuda/fused_sample.py
        # pass_configs). Measured on SM100 only.
        supports=BackendSupport(compute=frozenset({DType.FP32})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.sampling.fused_sample_tilelang",
            function_name="profile_fused_sample_tilelang",
        ),
        table_name=KIND,
        args_schema=FusedSampleArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="deepinfer_env",
        doc=BackendDoc(summary="DeepInfer tilelang_ops fused_sample with SamplingKernelInputs."),
    )
)

__all__ = ["KIND", "FusedSampleArgs"]
