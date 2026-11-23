"""DeepGEMM's FP4 candidate-restricted prefill indexer logits.

Wire string: ``"sparse_mqa_logits_prefill"`` -- matches Rust
``KernelSpec::KIND`` in ``simulator/src/timing/kernels/sparse_mqa_logits_prefill.rs``.

``deep_gemm.fp8_fp4_sparse_mqa_logits(..., use_unaligned_ks=True)`` as
DeepInfer's candidate consumer layers call it (``v4_modules/index_attn.py``
``_compute_prefill_p_sum``, sparse branch): each row scores only the keys of
the ``candidate_blocks`` blocks of ``candidate_block_size`` keys its group's
source layer picked, writing bf16 logits of width
``candidate_blocks · candidate_block_size``. One ``sm100_sparse_mqa_logits``
launch per consumer layer and query chunk; the schedule it reads is the
separate ``sparse_mqa_logits_metadata`` kind.

Why a new kind: no existing kind scores a per-row list of key blocks; the
dense kinds' cost follows the key window, this one follows the candidate
count.

Key: one request's causal-tail chunk (as ``mqa_logits_prefill_chunk``) and
the candidate geometry.
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

KIND = "sparse_mqa_logits_prefill"


@dataclass(frozen=True)
class SparseMqaLogitsPrefillArgs(KernelArgs):
    num_queries: int = arg(unit="tokens", doc="Query rows scored in the call.")
    context_len: int = arg(
        unit="tokens", doc="Context of the request, the rows included; the rows are its last tokens."
    )
    compress_ratio: int = arg(unit="tokens", doc="Context tokens per compressed index key.")
    candidate_blocks: int = arg(unit="blocks", doc="Candidate key blocks per row.")
    candidate_block_size: int = arg(unit="tokens", doc="Compressed keys per candidate block.")
    num_heads: int = arg(unit="heads", doc="Indexer query heads.")
    head_dim: int = arg(unit="elements", doc="Width of each query head and key.")
    dtype: DType = arg(doc="Element type of index Q and K (FP4 E2M1 with UE8M0 scales per 32).")


DOC = KernelDoc(
    title="Prefill indexer logits over candidate blocks (FP4)",
    summary="Score each prefill row against the keys of its candidate blocks only.",
    description=(
        "Each row keeps min(candidate_blocks, its valid blocks) ascending key blocks, "
        "its last valid block among them; rows of a request share most blocks. The "
        "kernel writes ReLU-weighted head sums as bf16 for every candidate key."
    ),
    category="Attention",
    subcategory="DSA",
    formula=(
        "T = Σ_rows min(row keys, min(candidate_blocks, row blocks) · candidate_block_size)",
        "FLOPs = 2 · T · num_heads · head_dim",
    ),
    default_metric="time_ms",
    method=CUPTI_METHOD + " Four rows are checked against fp32 math on the dequantized operands first.",
    caveats=(
        "Real candidate sets come from the source layer's scores; the measured rows "
        "share a per-request block ranking perturbed per row.",
    ),
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="deepgemm_sm100",
        # attention.hpp fp8_fp4_sparse_mqa_logits: DG_HOST_ASSERT(arch_major == 10).
        supports=BackendSupport(
            compute=frozenset({DType.MXFP4_E2M1}),
            sm_targets=frozenset({"sm_100f"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.sparse_mqa_logits_prefill_deepgemm",
            function_name="profile_sparse_mqa_logits_prefill_deepgemm",
        ),
        table_name=KIND,
        args_schema=SparseMqaLogitsPrefillArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="deepinfer_env",
        doc=BackendDoc(
            summary="DeepGEMM fp8_fp4_sparse_mqa_logits with unaligned ks, as DeepInfer calls it.",
            url="https://github.com/deepseek-ai/DeepGEMM",
        ),
    )
)
