"""DeepGEMM's FP4 candidate-restricted decode indexer logits over paged keys.

Wire string: ``"sparse_mqa_logits_decode"`` -- matches Rust
``KernelSpec::KIND`` in ``simulator/src/timing/kernels/sparse_mqa_logits_decode.rs``.

``deep_gemm.fp8_fp4_paged_sparse_mqa_logits((q, q_sf), kv_cache, weights,
metadata, candidate_blocks, candidate_block_size)`` as DeepInfer's candidate
consumer layers call it in decode (``v4_modules/index_attn.py``
``_compute_decode_p_sum``, sparse branch): every query row scores only the keys
of its ``candidate_blocks`` blocks of ``candidate_block_size`` keys, read from
the paged FP4 index-K cache through the row's block table; bf16 logits of
width ``candidate_blocks · candidate_block_size``. One
``sm100_paged_sparse_mqa_logits`` launch per consumer layer; the schedule is
the separate ``sparse_mqa_logits_decode_metadata`` kind.

Why not a backend of ``sparse_mqa_logits_prefill``: that kind's call is the
non-paged ``fp8_fp4_sparse_mqa_logits`` over one request's contiguous keys
(``ks``/``ke`` spans, ``num_queries`` tail rows of one context). This call reads
paged keys of many requests, ``next_n`` rows each, through block tables and
request indices: a different callable, operand layout and args meaning.

Key: requests x verify rows, the requests' index-key context, page size and the
candidate geometry.
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

KIND = "sparse_mqa_logits_decode"


@dataclass(frozen=True)
class SparseMqaLogitsDecodeArgs(KernelArgs):
    batch_size: int = arg(unit="requests", doc="Requests decoded together.")
    context_len: int = arg(unit="tokens", doc="Index keys cached per request.")
    next_n: int = arg(unit="tokens", doc="Query rows per request (verify width).")
    block_size: int = arg(unit="tokens", doc="Index keys per cache page.")
    candidate_blocks: int = arg(unit="blocks", doc="Candidate key blocks per row.")
    candidate_block_size: int = arg(unit="tokens", doc="Index keys per candidate block.")
    num_heads: int = arg(unit="heads", doc="Indexer query heads.")
    head_dim: int = arg(unit="elements", doc="Width of each query head and key.")
    dtype: DType = arg(doc="Element type of index Q and K (FP4 E2M1 with UE8M0 scales per 32).")


DOC = KernelDoc(
    title="Decode indexer logits over candidate blocks (FP4, paged)",
    summary="Score each decode row against the paged keys of its candidate blocks only.",
    description=(
        "Each row keeps min(candidate_blocks, its valid blocks) ascending key blocks, "
        "its last valid block among them; the rows of a request share most blocks. "
        "The kernel pairs rows, gathers the union of their blocks from the pages and "
        "writes ReLU-weighted head sums as bf16 for every candidate key."
    ),
    category="Attention",
    subcategory="DSA",
    formula=(
        "T = batch_size · next_n · min(context_len, min(candidate_blocks, ⌈context_len / "
        "candidate_block_size⌉) · candidate_block_size)",
        "FLOPs = 2 · T · num_heads · head_dim",
    ),
    default_metric="time_ms",
    method=CUPTI_METHOD + " Rows of the first and last request are checked against fp32 math"
    " on the dequantized operands first.",
    caveats=(
        "Real candidate sets come from the source layer's scores; the measured rows "
        "share a per-request block ranking perturbed per row.",
        "Every request holds context_len keys in unique, scattered pages.",
    ),
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="deepgemm_sm100",
        # attention.hpp fp8_fp4_paged_sparse_mqa_logits: DG_HOST_ASSERT(arch_major == 10).
        supports=BackendSupport(
            compute=frozenset({DType.MXFP4_E2M1}),
            kv=frozenset({DType.MXFP4_E2M1}),
            sm_targets=frozenset({"sm_100f"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.sparse_mqa_logits_decode_deepgemm",
            function_name="profile_sparse_mqa_logits_decode_deepgemm",
        ),
        table_name=KIND,
        args_schema=SparseMqaLogitsDecodeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="deepinfer_env",
        doc=BackendDoc(
            summary=(
                "DeepGEMM fp8_fp4_paged_sparse_mqa_logits, "
                "as DeepInfer's decode candidate consumers call it."
            ),
            url="https://github.com/deepseek-ai/DeepGEMM",
        ),
    )
)
