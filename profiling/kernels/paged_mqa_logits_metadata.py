"""DeepGEMM's schedule for the dense paged decode indexer logits (SM100 varlen).

Wire string: ``"paged_mqa_logits_metadata"`` -- matches Rust
``KernelSpec::KIND`` in ``simulator/src/timing/kernels/paged_mqa_logits_metadata.rs``.

``deep_gemm.get_paged_mqa_logits_metadata(klen.unsqueeze(1), block_size,
num_sms, indices=query_indices)``: one single-CTA
``sm100_paged_mqa_logits_metadata`` launch that groups the rows into requests
and splits their KV work across the SMs. DeepInfer (``index_attn.py``
``_get_decode_metadata``) caches it per compress ratio for the step, so it runs
once per ratio, not once per index layer: that is why it is a kind of its own
rather than part of ``dsa_paged_mqa_logits_decode``.

Key: the rows (requests x verify rows) and the page size.
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

KIND = "paged_mqa_logits_metadata"


@dataclass(frozen=True)
class PagedMqaLogitsMetadataArgs(KernelArgs):
    batch_size: int = arg(unit="requests", doc="Requests decoded together.")
    next_n: int = arg(unit="tokens", doc="Query rows per request (verify width).")
    block_size: int = arg(unit="tokens", doc="Index keys per cache page.")


DOC = KernelDoc(
    title="Decode indexer logits schedule",
    summary="Plan the per-SM work split of the dense paged decode indexer logits.",
    description=(
        "One CTA scans the rows' request indices into requests, prefix-sums their "
        "KV split cost and writes each SM's start point."
    ),
    category="Attention",
    subcategory="DSA",
    formula=("bytes = 8 · batch_size · next_n + 8 · (SMs + 1)",),
    default_metric="time_ms",
    method=CUPTI_METHOD + " The schedule is checked by one scoring launch against fp32 math first.",
    caveats=("Measured at a 4096-key context; the single CTA's work is per row, not per key.",),
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="deepgemm_sm100",
        # attention.hpp get_paged_mqa_logits_metadata: SM100 takes the varlen
        # (indices) form only; the check launch is the FP4 scorer.
        supports=BackendSupport(
            compute=frozenset({DType.MXFP4_E2M1}),
            sm_targets=frozenset({"sm_100f"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_paged_mqa_logits_decode_deepinfer",
            function_name="profile_paged_mqa_logits_metadata_deepgemm",
        ),
        table_name=KIND,
        args_schema=PagedMqaLogitsMetadataArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="deepinfer_env",
        doc=BackendDoc(
            summary=(
                "DeepGEMM get_paged_mqa_logits_metadata with indices, "
                "as DeepInfer's decode indexer calls it."
            ),
            url="https://github.com/deepseek-ai/DeepGEMM",
        ),
    )
)
