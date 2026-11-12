"""Expand a sparse indexer's selected compressed groups into the logical tokens to attend.

Every selected group becomes ``compress_ratio`` consecutive token ids, the
row's open causal tail follows, ``-1`` pads the rest, and a trailing column
holds the valid count the attention kernel loops to.
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

KIND = "sparse_index_expand"


@dataclass(frozen=True)
class SparseIndexExpandArgs(KernelArgs):
    num_rows: int = arg(unit="tokens", doc="Query rows whose selections are expanded.")
    token_topk: int = arg(unit="tokens", doc="Tokens selected from complete groups per row.")
    compress_ratio: int = arg(unit="tokens", doc="Tokens per compressed group.")


DOC = KernelDoc(
    title="Sparse index expansion",
    summary="Turn selected compressed groups into packed logical token ids plus a count.",
    description=(
        "One program per row and block of 256 output columns writes a "
        "[num_rows, token_topk + ratio] int32 buffer. Every measured row has its "
        "full selection; the time does not depend on the context (under 3% from "
        "512 to 131072 tokens)."
    ),
    category="Attention",
    subcategory="Sparse GQA",
    formula=("GB/s = num_rows · (4·token_topk/ratio + 4·(token_topk + ratio) + 12) / time",),
    default_metric="time_ms",
    method=f"{CUPTI_METHOD} The one launch of the public call is counted.",
    caveats=("GB/s counts the selection read and the packed buffer written.",),
    reference=None,
)


# Integer-only portable Triton (fork models/qwen4_exp/nvidia/ops/qsa_indexer.py:222-276).
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        supports=BackendSupport(compute=None, min_compute_capability=(7, 5)),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.sparse_index_expand_vllm_triton",
            function_name="profile_sparse_index_expand_vllm_triton",
        ),
        table_name=KIND,
        args_schema=SparseIndexExpandArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary="The alignment vLLM fork's expand_qsa_block_indices (Triton).",
            url="https://github.com/serendipity-zk/vllm/blob/892da0822f7e866cec8e519727f014c45a5ad792/vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py",
        ),
    )
)

__all__ = ["KIND", "SparseIndexExpandArgs"]
