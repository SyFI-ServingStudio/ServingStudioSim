"""Pooled DSA indexer back: selected pools to the final sparse-attention page table.

A pooled (kpool) DSA indexer selects ``top_k`` pools of ``kpool`` tokens per
query row. Sparse MLA reads a page table of physical cache slots: every token
of the selected pools plus the row's trailing partial pool, packed to the
front of a ``page_table_width`` row with one valid count per row. The vLLM
fork builds it in six launches after the top-k (buffer fill, pool-id cast,
seq-len + 1, pool expand, buffer copy, index remap); this kind is that whole
boundary as one operation.

Not ``dsa_sparse_index_remap``: that kind's input is token indices already
expanded into the page-table width; this one takes pool ids and sequence
lengths and builds the token row (pool expansion and tail) itself.
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

KIND: str = "dsa_pooled_topk_page_table"


@dataclass(frozen=True)
class DsaPooledTopkPageTableArgs(KernelArgs):
    batch_size: int = arg(unit="requests", doc="Requests decoded together.")
    context_len: int = arg(unit="pools", doc="Complete pools in each request's longest row.")
    next_n: int = arg(unit="tokens", doc="Query rows per request in a decode step.")
    max_model_len: int = arg(unit="tokens", doc="Tokens one request's block table can address.")
    top_k: int = arg(unit="pools", doc="Pools selected per query row.")
    kpool: int = arg(unit="tokens", doc="Tokens per pool.")
    page_table_width: int = arg(unit="slots", doc="Width of each page-table row.")
    block_size: int = arg(unit="tokens", doc="Tokens in each KV cache block.")
    index_dtype: str = arg(doc="Element type of pool ids and page-table slots.")


DOC = KernelDoc(
    title="Pooled top-k to page table",
    summary="Turn each query row's selected pools into its packed sparse-attention page table.",
    description=(
        "Each of the batch_size · next_n rows holds top_k selected pool ids (-1 past a short row) and "
        "its token count. Pool p covers tokens kpool·p … kpool·p + kpool − 1; the row's trailing "
        "partial pool adds seq mod kpool tokens. Every token whose block is inside the request's "
        "block table maps to block_table[request, token / block_size] · block_size + token mod "
        "block_size. The valid slots are packed to the front of the row in column order, the rest "
        "is -1, and each row's valid count is written. Rows of a request have token counts "
        "4·context_len − (next_n − 1 − j) − (request mod 4); the pool ids are the top-k of "
        "scores drawn from the measured live-indexer distribution."
    ),
    category="Attention",
    subcategory="DSA",
    formula=(
        "rows = batch_size · next_n; history slots = top_k · kpool; tail slots < kpool",
        "slot = block_table[req, tok / block_size] · block_size + tok mod block_size",
        "logical bytes = rows · (4·top_k + 8 + 4·(top_k + kpool − 1) + 4·page_table_width + 4)",
        "GB/s = logical bytes / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=f"{CUPTI_METHOD} One launch. The page table is checked bit-exact before timing.",
    caveats=(
        "The vendored kernel is compiled for top_k 512, kpool 4, page_table_width 2176 and "
        "64-token blocks; the runner refuses other shapes.",
        "GB/s counts pool ids, one block-table entry per pool and tail token, and the output row, "
        "not physical transactions.",
    ),
    reference="profiling.runners.attention.dsa_pooled_page_table_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="cuda_row_scan_compact",
        # back.cu is built for compute_100a only.
        supports=BackendSupport(compute=None, sm_targets=frozenset({"sm_100a"})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_pooled_page_table",
            function_name="profile_dsa_pooled_topk_page_table_cuda_row_scan_compact",
        ),
        table_name=KIND,
        args_schema=DsaPooledTopkPageTableArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        # The image carries nvcc and the Torch the extension builds against.
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "Vendored CUDA kernel (not a fork path): one CTA of 512 threads per row loads the "
                "row's pool ids, gathers one block-table entry per pool and per tail token, places "
                "the valid slots by one block-wide scan and fills the rest with -1. Bit-exact to the "
                "fork's six-launch chain."
            ),
        ),
    )
)

__all__ = ["KIND", "DsaPooledTopkPageTableArgs"]
