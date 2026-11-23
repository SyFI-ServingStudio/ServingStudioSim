"""DeepGEMM's schedule for the candidate-restricted indexer logits.

Wire string: ``"sparse_mqa_logits_metadata"`` -- matches Rust
``KernelSpec::KIND`` in ``simulator/src/timing/kernels/sparse_mqa_logits_metadata.rs``.

``deep_gemm.get_sparse_mqa_logits_metadata(ks, ke, num_kv_tokens,
absolute_candidate_blocks, q_dtype, candidate_block_size,
use_unaligned_ks=True)``: one ``sm100_sparse_mqa_logits_metadata`` launch.
DeepInfer (``index_attn.py`` ``_get_sparse_prefill_metadata``) caches the
result per query chunk across the consumer layers of a candidate group, so it
runs once per chunk and group, not once per layer; that is why it is a kind
of its own rather than part of ``sparse_mqa_logits_prefill``.

Key: as ``sparse_mqa_logits_prefill`` without the head geometry.
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

KIND = "sparse_mqa_logits_metadata"


@dataclass(frozen=True)
class SparseMqaLogitsMetadataArgs(KernelArgs):
    num_queries: int = arg(unit="tokens", doc="Query rows of the chunk.")
    context_len: int = arg(
        unit="tokens", doc="Context of the request, the rows included; the rows are its last tokens."
    )
    compress_ratio: int = arg(unit="tokens", doc="Context tokens per compressed index key.")
    candidate_blocks: int = arg(unit="blocks", doc="Candidate key blocks per row.")
    candidate_block_size: int = arg(unit="tokens", doc="Compressed keys per candidate block.")
    dtype: DType = arg(doc="Element type of index Q and K; it picks the schedule's KV split.")


DOC = KernelDoc(
    title="Sparse indexer logits schedule",
    summary="Plan the work split of the candidate-restricted indexer logits for one chunk.",
    description=(
        "Reads every row's key window and candidate blocks and writes the work "
        "schedule fp8_fp4_sparse_mqa_logits consumes."
    ),
    category="Attention",
    subcategory="DSA",
    formula=("bytes = 4 · num_queries · (candidate_blocks + 2)",),
    default_metric="time_ms",
    method=CUPTI_METHOD + " The schedule is checked by one scoring launch against fp32 math first.",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="deepgemm_sm100",
        # attention.hpp get_sparse_mqa_logits_metadata: DG_HOST_ASSERT(arch_major == 10).
        supports=BackendSupport(
            compute=frozenset({DType.MXFP4_E2M1}),
            sm_targets=frozenset({"sm_100f"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.sparse_mqa_logits_prefill_deepgemm",
            function_name="profile_sparse_mqa_logits_metadata_deepgemm",
        ),
        table_name=KIND,
        args_schema=SparseMqaLogitsMetadataArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="deepinfer_env",
        doc=BackendDoc(
            summary="DeepGEMM get_sparse_mqa_logits_metadata with unaligned ks, as DeepInfer calls it.",
            url="https://github.com/deepseek-ai/DeepGEMM",
        ),
    )
)
