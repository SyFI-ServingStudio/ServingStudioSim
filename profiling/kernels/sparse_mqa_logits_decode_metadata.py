"""DeepGEMM's schedule for the paged candidate-restricted decode indexer logits.

Wire string: ``"sparse_mqa_logits_decode_metadata"`` -- matches Rust
``KernelSpec::KIND`` in ``simulator/src/timing/kernels/sparse_mqa_logits_decode_metadata.rs``.

``deep_gemm.get_paged_sparse_mqa_logits_metadata(klen, block_tables,
query_indices, page_size, candidate_blocks, q_dtype, candidate_block_size)``:
one ``sm100_sparse_mqa_logits_metadata`` launch (paged mode). DeepInfer
(``index_attn.py`` ``_get_decode_metadata``) caches it per candidate group for
the step, so it runs once per group, not once per consumer layer.

Why not a backend of ``sparse_mqa_logits_metadata``: that kind is the
non-paged schedule over one request's ``ks``/``ke`` spans; this one resolves
candidate blocks to physical pages through every row's block table and groups
rows by request index.

Key: as ``sparse_mqa_logits_decode`` without the head geometry.
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

KIND = "sparse_mqa_logits_decode_metadata"


@dataclass(frozen=True)
class SparseMqaLogitsDecodeMetadataArgs(KernelArgs):
    batch_size: int = arg(unit="requests", doc="Requests decoded together.")
    context_len: int = arg(unit="tokens", doc="Index keys cached per request.")
    next_n: int = arg(unit="tokens", doc="Query rows per request (verify width).")
    block_size: int = arg(unit="tokens", doc="Index keys per cache page.")
    candidate_blocks: int = arg(unit="blocks", doc="Candidate key blocks per row.")
    candidate_block_size: int = arg(unit="tokens", doc="Index keys per candidate block.")
    dtype: DType = arg(doc="Element type of index Q and K; it picks the schedule's KV split.")


DOC = KernelDoc(
    title="Paged sparse decode indexer logits schedule",
    summary="Plan the work split of the candidate-restricted decode indexer logits.",
    description=(
        "Reads every row's candidate blocks, block table and length and writes the "
        "page-resolved schedule fp8_fp4_paged_sparse_mqa_logits consumes."
    ),
    category="Attention",
    subcategory="DSA",
    formula=(
        "bytes = 4 · batch_size · next_n · (candidate_blocks + ⌈context_len / block_size⌉ + 2)",
    ),
    default_metric="time_ms",
    method=CUPTI_METHOD + " The schedule is checked by one scoring launch against fp32 math first.",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="deepgemm_sm100",
        # attention.hpp get_paged_sparse_mqa_logits_metadata: DG_HOST_ASSERT(arch_major == 10).
        supports=BackendSupport(
            compute=frozenset({DType.MXFP4_E2M1}),
            sm_targets=frozenset({"sm_100f"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.sparse_mqa_logits_decode_deepgemm",
            function_name="profile_sparse_mqa_logits_decode_metadata_deepgemm",
        ),
        table_name=KIND,
        args_schema=SparseMqaLogitsDecodeMetadataArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="deepinfer_env",
        doc=BackendDoc(
            summary=(
                "DeepGEMM get_paged_sparse_mqa_logits_metadata, "
                "as DeepInfer's decode candidate consumers call it."
            ),
            url="https://github.com/deepseek-ai/DeepGEMM",
        ),
    )
)
