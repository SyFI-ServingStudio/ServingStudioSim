"""Pooled DSA indexer back: indexer logits to the final sparse-attention page table.

The decode top-k over pooled indexer logits and the page-table build that
follows it (``dsa_pooled_topk_page_table``) as one operation: each row's
top-``top_k`` pools are selected and written out directly as the packed
physical page table with valid counts; the pool ids never leave the kernel.

Not ``dsa_persistent_topk_decode`` (its output is the selected positions) and
not ``dsa_pooled_topk_page_table`` (its input is the selected pools).
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

KIND: str = "dsa_pooled_logits_page_table"


@dataclass(frozen=True)
class DsaPooledLogitsPageTableArgs(KernelArgs):
    batch_size: int = arg(unit="requests", doc="Requests decoded together.")
    context_len: int = arg(unit="pools", doc="Complete pools in each request's longest row.")
    next_n: int = arg(unit="tokens", doc="Query rows per request in a decode step.")
    max_model_len: int = arg(unit="tokens", doc="Tokens one request's block table can address.")
    logits_row_stride: int = arg(unit="elements", doc="Physical spacing between logit rows.")
    top_k: int = arg(unit="pools", doc="Pools selected per query row.")
    kpool: int = arg(unit="tokens", doc="Tokens per pool.")
    page_table_width: int = arg(unit="slots", doc="Width of each page-table row.")
    block_size: int = arg(unit="tokens", doc="Tokens in each KV cache block.")
    logits_dtype: DType = arg(doc="Element type of the indexer logits.")
    index_dtype: str = arg(doc="Element type of the page-table slots.")
    context_mode: str = arg(
        doc="Score distribution: uniform_measured draws from a measured live-indexer distribution "
        "(as dsa_persistent_topk_decode), uniform_random standard normals."
    )


DOC = KernelDoc(
    title="Pooled top-k with page table",
    summary="Select each row's top pools from indexer logits and write its packed page table.",
    description=(
        "Each of the batch_size · next_n rows has seq tokens, seq / kpool complete pools and one "
        "FP32 logit per pool. The kernel selects the top_k largest logits of the row (all of them "
        "when the row is shorter) and writes the row's page table as dsa_pooled_topk_page_table "
        "does: the tokens of the selected pools and the trailing partial pool, mapped through the "
        "block table, packed to the front, -1 after, with a valid count. Rows of a request have "
        "4·context_len − (next_n − 1 − j) − (request mod 4) tokens."
    ),
    category="Attention",
    subcategory="DSA",
    formula=(
        "rows = batch_size · next_n; L_j = seq_j / kpool pools",
        "selected = top_k pools of logits[j, :L_j]; page table as dsa_pooled_topk_page_table",
        "logical bytes = 4·Σ_j L_j + rows · (4·top_k + 8 + 4·(top_k + kpool − 1) + 4·page_table_width + 4)",
        "GB/s = logical bytes / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} One launch. Before timing, the selected pool values are checked against "
        "torch.topk per row and the page table and valid counts against the reference built from "
        "the kernel's own selection."
    ),
    caveats=(
        "Rows whose top_k-th score is tied may select other pools than another top-k kernel; the "
        "selected values are the same.",
        "Time follows the score distribution (as dsa_persistent_topk_decode).",
        "The vendored kernel is compiled for top_k 512, kpool 4, page_table_width 2176 and "
        "64-token blocks; the runner refuses other shapes.",
    ),
    reference="profiling.runners.attention.dsa_pooled_page_table_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="cuda_threshold_filter_radix",
        supports=BackendSupport(
            compute=frozenset({DType.FP32}),
            sm_targets=frozenset({"sm_100a"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_pooled_page_table",
            function_name="profile_dsa_pooled_logits_page_table_cuda_threshold_filter_radix",
        ),
        table_name=KIND,
        args_schema=DsaPooledLogitsPageTableArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "Vendored CUDA kernel (not a fork path): one CTA of 512 threads per row streams the "
                "row's logits once, takes a threshold from an 8-ary count search over the first "
                "batch, keeps the scores above it in per-lane buffers, selects exactly by an 11-bit "
                "radix over the survivors (exact fallback over the whole row on overflow) and builds "
                "the page table in its epilogue."
            ),
        ),
    )
)

__all__ = ["KIND", "DsaPooledLogitsPageTableArgs"]
