"""DeepInfer's candidate compact lengths (``tilelang_ops/candidate_blocks.py``
``get_candidate_compact_lens``).

Wire string: ``"candidate_compact_lens"``.

One ``_get_candidate_compact_lens_kernel`` launch counts, per row, the valid
key tokens inside its candidate blocks. DeepSeek-V4.1 prefill runs it once per
step on the first candidate consumer (``index_attn.py``
``_get_candidate_metadata``).

Why a new kind: no existing kind does this reduction.
"""

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

KIND = "candidate_compact_lens"


@dataclass(frozen=True)
class CandidateCompactLensArgs(KernelArgs):
    num_rows: int = arg(unit="tokens", doc="Query rows.")
    topk_blocks: int = arg(unit="blocks", doc="Candidate blocks per row.")
    block_size: int = arg(unit="tokens", doc="Key tokens per candidate block (8 or 16).")


DOC = KernelDoc(
    title="Candidate compact lengths",
    summary="Counts each row's valid key tokens inside its candidate blocks.",
    description=(
        "Reads [num_rows, topk_blocks] int32 block ids and the row's key "
        "length; writes one int32 count per row."
    ),
    category="Attention",
    subcategory="Compressed sparse MLA",
    formula=("bytes = 4·num_rows·(topk_blocks + 2)",),
    default_metric="time_ms",
    method=CUPTI_METHOD,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="tilelang",
        supports=BackendSupport(compute=None, min_compute_capability=(9, 0)),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.deepinfer_kv_tilelang",
            function_name="profile_candidate_compact_lens_tilelang",
        ),
        table_name=KIND,
        args_schema=CandidateCompactLensArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="deepinfer_env",
        doc=BackendDoc(
            summary=(
                "DeepInfer's TileLang get_candidate_compact_lens, checked equal to its reference."
            ),
        ),
    )
)
