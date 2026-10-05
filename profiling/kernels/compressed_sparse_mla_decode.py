"""Compressed sparse MLA decode over window and selected compressed keys (FP8, graph replay)."""

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

KIND = "compressed_sparse_mla_decode"


@dataclass(frozen=True)
class CompressedSparseMlaDecodeArgs(KernelArgs):
    """Workload and model identity for one sparse decode graph replay.

    Counts remain per flattened query row because FlashMLA's planner consumes
    every row. Production also flattens speculative decode tokens this way and
    keeps ``s_q=1``. ``extra_index_capacity`` distinguishes runtime context
    configurations, notably C128 width 512 at 65K versus 8192 at 1M.
    """

    swa_valid_counts: tuple[int, ...] = arg(
        unit="tokens", doc="Valid sliding-window keys for each flattened decode query."
    )
    extra_valid_counts: tuple[int, ...] = arg(
        unit="tokens", doc="Valid selected compressed keys for each decode query."
    )
    num_heads: int = arg(unit="heads", doc="Query heads on this GPU.")
    num_kv_heads: int = arg(unit="heads", doc="KV heads in the cache.")
    head_dim: int = arg(unit="elements", doc="Elements in each query and key head.")
    value_dim: int = arg(unit="elements", doc="Elements in each output head.")
    swa_window: int = arg(unit="tokens", doc="Maximum sliding-window keys per query.")
    extra_index_capacity: int = arg(unit="tokens", doc="Allocated selected-key indices per query.")
    compress_ratio: int = arg(
        unit="tokens", doc="Source tokens represented by one compressed cache position."
    )
    q_dtype: DType = arg(doc="Query element type.")
    cache_dtype: DType = arg(doc="Cache element type.")
    output_dtype: DType = arg(doc="Output element type.")
    planner_mode: str = arg(doc="Whether a sparse launch runs before graph capture.")


DOC = KernelDoc(
    title="Compressed sparse MLA decode",
    summary="Attend from each decode query to sliding-window and optional compressed keys.",
    description=(
        "Compressed sparse MLA in decode: each query row attends to its "
        "sliding-window keys and, in compressed layers, to the compressed keys "
        "the indexer selected, in one FlashMLA call. The two sources have "
        "separate FP8 page sets and padded index arrays; swa_valid_counts and "
        "extra_valid_counts give the valid entries per row. As in serving, the "
        "call is captured in a CUDA graph and replayed."
    ),
    category="Attention",
    subcategory="Compressed sparse MLA",
    formula=(
        "O = softmax(q · K_selectedᵀ / √head_dim) · V_selected, per query head",
        "S = Σ(swa_valid_counts) + Σ(extra_valid_counts); R = len(swa_valid_counts)",
        "TFLOPS = 2·num_heads·(head_dim + value_dim)·S / time",
        "GB/s = [R·num_heads·head_dim·2 + S·(584 + 4) + "
        "R·4·(2 if compress_ratio > 1 else 1) + R·num_heads·(value_dim·2 + 4) "
        "+ num_heads·4] / time",
    ),
    default_metric="tflops",
    method=(
        f"{CUPTI_METHOD} "
        "The launches of each graph replay are counted, after three warm-up "
        "replays. Graph capture and metadata construction are excluded; in "
        "reused planner mode one sparse launch also runs before capture."
    ),
    caveats=(
        "Queries are zero and cache rows hold patterned values with contiguous "
        "valid indices; FlashMLA's work does not depend on the values.",
        "GB/s counts the valid selected rows and indices, not the padding of the index arrays.",
    ),
    # The expected-output calculation is local to the measured runner.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_flashmla_fp8_cudagraph",
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.compressed_sparse_mla_decode_flashmla",
            function_name="profile_compressed_sparse_mla_decode_flashmla",
        ),
        table_name=KIND,
        args_schema=CompressedSparseMlaDecodeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        # FlashMLA's Arch::is_sm90a() / is_sm100f(): exactly SM90, or any SM10x.
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.FP8_E4M3}),
            sm_targets=frozenset({"sm_90a", "sm_100f"}),
        ),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "FlashMLA flash_mla_with_kvcache, as vLLM calls it, over FP8 "
                "sliding-window and selected-key caches, replayed from a CUDA graph."
            ),
            url="https://github.com/deepseek-ai/FlashMLA",
        ),
    )
)

__all__ = ["CompressedSparseMlaDecodeArgs", "KIND"]
