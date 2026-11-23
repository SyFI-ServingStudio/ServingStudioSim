"""DeepInfer's candidate index restore (``tilelang_ops/candidate_blocks.py``
``restore_candidate_indices_inplace``).

Wire string: ``"candidate_index_restore"``.

One ``_restore_candidate_indices_inplace_kernel`` launch maps each row's
top-k compact indices back to key indices through its candidate blocks and
adds the row's key offset, in place. DeepSeek-V4.1 prefill runs it after each
top-k chunk of a candidate consumer layer (``index_attn.py``
``_get_prefill_metadata``).

Why a new kind: no existing kind does this remap.
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

KIND = "candidate_index_restore"


@dataclass(frozen=True)
class CandidateIndexRestoreArgs(KernelArgs):
    num_rows: int = arg(unit="tokens", doc="Query rows.")
    topk_tokens: int = arg(unit="tokens", doc="Top-k indices per row.")
    topk_blocks: int = arg(unit="blocks", doc="Candidate blocks per row.")
    block_size: int = arg(unit="tokens", doc="Key tokens per candidate block (8 or 16).")


DOC = KernelDoc(
    title="Candidate index restore",
    summary="Maps compact top-k indices back to key indices through the candidate blocks.",
    description=(
        "For each of a row's topk_tokens compact indices, gathers its candidate "
        "block id and rewrites the index as block·block_size + offset + row key "
        "offset, or -1."
    ),
    category="Attention",
    subcategory="Compressed sparse MLA",
    formula=("bytes = 8·num_rows·topk_tokens + 4·num_rows·(topk_blocks + 1)",),
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
            function_name="profile_candidate_index_restore_tilelang",
        ),
        table_name=KIND,
        args_schema=CandidateIndexRestoreArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="deepinfer_env",
        doc=BackendDoc(
            summary=(
                "DeepInfer's TileLang restore_candidate_indices_inplace with a row "
                "offset, checked equal to its reference; only the restore launch is "
                "timed."
            ),
        ),
    )
)
