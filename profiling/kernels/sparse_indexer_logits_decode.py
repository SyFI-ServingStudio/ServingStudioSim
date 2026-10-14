"""Decode scores of a weight-free sparse-attention indexer over compressed paged keys.

Every request scores its ``next_n`` decode/verify rows against all of its
compressed index keys: per query head the ReLU of the query-key dot product,
summed over heads with no per-head weight and no key scale. The scores feed a
top-k (``dsa_persistent_topk_decode``) that picks the groups to attend.
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

KIND = "sparse_indexer_logits_decode"


@dataclass(frozen=True)
class SparseIndexerLogitsDecodeArgs(KernelArgs):
    batch_size: int = arg(unit="requests", doc="Requests decoded together.")
    num_keys: int = arg(unit="keys", doc="Compressed index keys visible to every row.")
    logits_width: int = arg(
        unit="keys",
        doc=(
            "Columns of the padded score matrix, the page-table capacity in keys; "
            "the launch grid spans all of them."
        ),
    )
    next_n: int = arg(unit="tokens", doc="Query rows per request (1 + speculative tokens).")
    num_heads: int = arg(unit="heads", doc="Indexer query heads.")
    head_dim: int = arg(unit="elements", doc="Width of each query head and key.")
    dtype: DType = arg(doc="Element type of the queries and the cached keys (they match).")


DOC = KernelDoc(
    title="Sparse indexer decode logits",
    summary="Score decode rows against every visible compressed key with a ReLU head sum.",
    description=(
        "The Triton kernel launches one program per request and block of 64 "
        "columns across the full padded width, and programs past the request's "
        "visible keys exit at once; so the width, which a deployment sets from "
        "its maximum length, changes the time even for short contexts. Pages of "
        "keys are scattered at random; scores are FP32."
    ),
    category="Attention",
    subcategory="Sparse GQA",
    formula=(
        "logit[r, c] = Σₕ ReLU(q[r, h] · key[c]) for c < num_keys",
        "TFLOPS = 2 · batch_size · next_n · num_heads · head_dim · num_keys / time",
        "GB/s = (batch_size · num_keys · head_dim · bytes(dtype) "
        "+ 4 · batch_size · next_n · num_keys) / time",
    ),
    default_metric="time_ms",
    method=(
        f"{CUPTI_METHOD} The public decode selection call runs whole; only the "
        "_qsa_mqa_paged_uniform_kernel launch is counted (its top-k is a separate "
        "kind). A PyTorch reference checks sampled rows first."
    ),
    caveats=(
        "Every row of the batch sees the same num_keys.",
        "GB/s counts keys read once per request and the valid scores written.",
    ),
    reference="profiling.runners.attention.sparse_indexer_logits_reference",
)


# Portable Triton (fork models/qwen4_exp/nvidia/ops/qsa_indexer.py:19-114); fp8
# tl.dot lowers on SM89+.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        supports=BackendSupport(
            compute=frozenset({DType.BF16, DType.FP8_E4M3}),
            min_compute_capability=(8, 9),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.sparse_indexer_logits_vllm_triton",
            function_name="profile_sparse_indexer_logits_decode_vllm_triton",
        ),
        table_name=KIND,
        args_schema=SparseIndexerLogitsDecodeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary="The alignment vLLM fork's _qsa_mqa_paged_uniform_kernel (Triton).",
            url="https://github.com/serendipity-zk/vllm/blob/892da0822f7e866cec8e519727f014c45a5ad792/vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py",
        ),
    )
)

__all__ = ["KIND", "SparseIndexerLogitsDecodeArgs"]
