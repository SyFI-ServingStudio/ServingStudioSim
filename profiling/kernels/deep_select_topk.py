"""DeepSelect's bf16 row top-k as DeepInfer's prefill indexer calls it.

Wire string: ``"deep_select_topk"`` -- matches Rust ``KernelSpec::KIND`` in
``simulator/src/timing/kernels/deep_select_topk.rs``.

``deep_select.interface.topk(scores, top_k, sorted=False, end=..., indices_type
=int32, sorted_index=True, output_idx=..., idx_oob_fill_value=-1,
return_value=False)``: one ``topk_select_bf16_normal::topk_kernel`` launch, one
CTA per row (``v4_modules/index_attn.py``: ``_topk_prefill`` on every logits
chunk, ``_topk_candidate_block`` on the source layer's block scores).

Why not a backend of ``dsa_topk_prefill``: that kind's rows are fp32 causal
tails of one request with DeepGEMM's fp32 padding; DeepInfer selects from bf16
rows of three different shapes (dense compressed tails, block scores and
candidate logits), each bounded by its own ``end``. The cost of one CTA
follows its own row's length, so the key is the row count and the valid
length of a row; a call whose rows differ in length is priced at their mean.

Key: ``num_rows``, ``row_len`` (valid scores per row), ``top_k``, dtype.
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

KIND = "deep_select_topk"


@dataclass(frozen=True)
class DeepSelectTopkArgs(KernelArgs):
    num_rows: int = arg(unit="rows", doc="Score rows selected from.")
    row_len: int = arg(unit="elements", doc="Valid scores in each row (the row's end).")
    top_k: int = arg(unit="elements", doc="Indices kept per row.")
    dtype: DType = arg(doc="Element type of the scores (bf16).")


DOC = KernelDoc(
    title="Indexer top-k (DeepSelect, bf16)",
    summary="Keep the indices of the top_k largest scores of each row's valid prefix.",
    description=(
        "One CTA per row selects the top_k largest of the row's first row_len bf16 "
        "scores and writes their int32 indices in ascending order (-1 past a short "
        "row). Scores are seeded standard-normal in rows padded to 1024 bytes."
    ),
    category="Attention",
    subcategory="DSA",
    formula=("bytes = num_rows · (2 · row_len + 4 · top_k + 8)",),
    default_metric="time_ms",
    method=CUPTI_METHOD + " Four rows' selections are checked against torch.topk first.",
    caveats=(
        "Every row is row_len long; DeepInfer's rows differ slightly within a call "
        "(causal windows), which the integration prices at the mean length.",
        "Measured with the per-row output offsets of the dense call site; the "
        "block-score and candidate call sites pass none.",
    ),
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="deep_select",
        # DeepSelect's CUDA kernels are built for the SM10x family only (vLLM's
        # deepselect.cmake lists "10.0f"; DeepInfer builds the same tree).
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            sm_targets=frozenset({"sm_100f"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.deep_select_topk_cuda",
            function_name="profile_deep_select_topk_cuda",
        ),
        table_name=KIND,
        args_schema=DeepSelectTopkArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="deepinfer_env",
        doc=BackendDoc(
            summary="DeepSelect topk on bf16 rows with per-row end, sorted indices, no values.",
            url="https://github.com/deepseek-ai/DeepSelect",
        ),
    )
)
