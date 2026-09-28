"""DSA indexer prefill top-k over compressed keys, for one request batch."""

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

KIND = "dsa_compressed_topk_prefill"


@dataclass(frozen=True)
class DsaCompressedTopkPrefillArgs(KernelArgs):
    query_context_pairs: tuple[tuple[int, int], ...] = arg(
        doc="Per-request (query tokens, context tokens) in the prefill batch."
    )
    max_model_len: int = arg(unit="tokens", doc="Maximum context length allowed by the model.")
    max_num_batched_tokens: int = arg(
        unit="tokens", doc="Maximum query tokens allowed in the batch."
    )
    max_logits_bytes: int = arg(unit="bytes", doc="Limit on logical logits in each chunk.")
    compress_ratio: int = arg(
        unit="tokens", doc="Source tokens represented by one compressed cache position."
    )
    top_k: int = arg(unit="tokens", doc="Compressed cache positions selected per query.")
    logits_dtype: DType = arg(doc="Element type of the indexer logits.")
    index_dtype: str = arg(doc="Element type of the selected indices.")


DOC = KernelDoc(
    title="Compressed prefill indexer top-k",
    summary="Select compressed cache positions from indexer logits for each prefill query.",
    description=(
        "In prefill over a compressed KV cache, the DSA indexer keeps the top_k"
        " highest-scoring compressed keys for each query; sparse MLA then reads"
        " only those. Each query_context_pairs entry gives one request's query "
        "count and full context; a query considers only the compressed keys "
        "visible at its position. The batch is cut into logits chunks under "
        "max_logits_bytes, as vLLM does."
    ),
    category="Attention",
    subcategory="DSA",
    formula=(
        "indices[q] = top_k(logits[q, row_start[q]:row_end[q]], top_k)",
        "V = Σq (row_end[q] − row_start[q]); Q = total query rows",
        "GB/s = (4·V + 8·Q + 4·Q·top_k) / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "Every launch across the logits chunks is counted; a chunk with more "
        "than 12,288 query rows adds a radix top-k launch."
    ),
    caveats=(
        "Logits increase with key position, so the selection is deterministic "
        "rather than driven by model scores.",
        "Only compress_ratio 4, top_k 512 and a 512 MiB logits limit on H200 are measured.",
    ),
    # The output check is local to the measured runner; there is no separate reference module.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cuda",
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_compressed_topk_prefill_cuda",
            function_name="profile_dsa_compressed_topk_prefill_cuda",
        ),
        table_name=KIND,
        args_schema=DsaCompressedTopkPrefillArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=BackendSupport(
            compute=frozenset({DType.FP32}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM's top_k_per_row_prefill CUDA operation selects local top-512 "
                "indices from FP32 logits."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/sampler.cu",
        ),
    )
)

__all__ = ["DsaCompressedTopkPrefillArgs", "KIND"]
