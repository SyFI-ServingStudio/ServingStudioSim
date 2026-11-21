"""DeepSelect's top-k with values over fp32 sampler logits, as DeepInfer's sampler calls it.

Wire string: ``"deep_select_logits_topk"`` -- matches Rust ``KernelSpec::KIND`` in
``simulator/src/timing/kernels/deep_select_logits_topk.rs``.

``deep_select.interface.topk(logits, top_k, sorted, indices_type=int32)`` with
values returned: one ``topk_select_fp32::topk_kernel`` launch, one CTA per row.
Two DeepInfer call sites:

* ``sampler_utils.process_sampling`` with ``DEEPINFER_SAMPLER_FORCE_TOPK``:
  ``topk(logits, force_topk, False, ...)`` over the full-vocabulary fp32 logits
  of every verify row and of every DFlash Markov step (``sorted=False``);
* ``Sampler._prepare_top_info`` when requests ask for top logprobs:
  ``topk(logprobs, max_top_logprobs, True, ...)`` over the force_topk-wide
  logprobs (``sorted=True``).

Why not a backend of ``deep_select_topk``: that kind is DeepSelect's indices-only
call (``return_value=False``, ``sorted_index=True``, bf16 rows bounded by
``end``), a different kernel instantiation (``TopkSelectConfig`` flags) and a
different output contract. Why not a backend of ``logits_topk``: that kind is
fixed to sorted output (vLLM's only call form), and the force_topk call is
unsorted. ``sorted`` is part of each call site's contract and selects the
kernel instantiation (``TopkSelectConfig<float, int, sorted, 0, 1, ...>`` in the
capture), so it is a key field: at 384 x 129280, top 1024, sorted takes 94.8 us
and unsorted 76.0 us (logs/20261120_2_ab).

Key: ``num_rows``, ``num_columns`` (row width: the vocabulary, or force_topk),
``top_k``, ``sorted``, ``dtype``.
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

KIND = "deep_select_logits_topk"


@dataclass(frozen=True)
class DeepSelectLogitsTopkArgs(KernelArgs):
    num_rows: int = arg(unit="rows", doc="Logit rows selected from, one per sampled position.")
    num_columns: int = arg(
        unit="elements", doc="Scores in each row (vocabulary or force_topk width)."
    )
    top_k: int = arg(unit="elements", doc="Values and indices kept per row.")
    sorted: bool = arg(doc="Whether the kept values come back in descending order.")
    dtype: DType = arg(doc="Element type of the scores and returned values (fp32).")


DOC = KernelDoc(
    title="Sampler top-k (DeepSelect, fp32 values)",
    summary="Keep the top_k largest fp32 scores of each row with their int32 indices.",
    description=(
        "One CTA per row selects the top_k largest of the row's num_columns fp32 scores "
        "and writes their values and int32 indices, in descending order when sorted, "
        "else in selection order. DeepInfer's sampler restricts sampling to the "
        "force_topk largest logits this way (sorted=False) and picks the requested "
        "top logprobs from them (sorted=True). Scores are seeded standard-normal."
    ),
    category="Other",
    formula=("bytes = num_rows · (4 · num_columns + 8 · top_k)",),
    default_metric="time_ms",
    method=CUPTI_METHOD + " Four rows' selected values are checked against torch.topk first.",
    caveats=(
        "Rows are contiguous; DeepInfer's Markov steps pass every fifth row of the "
        "draft logits (a strided view), measured within 2% of contiguous rows.",
    ),
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="deep_select",
        # DeepSelect's CUDA kernels are built for the SM10x family only (vLLM's
        # deepselect.cmake lists "10.0f"; DeepInfer builds the same tree).
        supports=BackendSupport(
            compute=frozenset({DType.FP32}),
            sm_targets=frozenset({"sm_100f"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.sampling.deep_select_logits_topk",
            function_name="profile_deep_select_logits_topk",
        ),
        table_name=KIND,
        args_schema=DeepSelectLogitsTopkArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="deepinfer_env",
        doc=BackendDoc(
            summary="DeepSelect topk on fp32 rows returning values and int32 indices.",
            url="https://github.com/deepseek-ai/DeepSelect",
        ),
    )
)

__all__ = ["KIND", "DeepSelectLogitsTopkArgs"]
