"""Paged FP8 sparse MLA over a sliding-window cache and a compressed cache, one call per layer.

Wire string: ``"paged_sparse_mla"``.

Why a new kind rather than ``compressed_sparse_mla_decode`` / ``_prefill``: on
SM12x SGLang (v0.5.23, ``--attention-backend dsv4``) runs every DeepSeek-V4-family
attention layer -- decode, speculative verify *and* chunked prefill -- through one
paged call, ``flash_mla_with_kvcache_sm120``
(``kernels/ops/attention/flash_mla_sm120.py:298``, called from
``srt/layers/attention/deepseek_v4_backend.py:4335``), reading the FP8 SWA and
compressed caches in place. Which FlashInfer kernel runs is set by the number of
query rows, not by the phase: up to ``SM120_DECODE_MAX_TOKENS`` (64) rows take the
split-K decode kernel plus its merge (``_flash_mla_flashinfer``, line 583), more
take the paged prefill kernel (``_flash_mla_sm120_prefill``, line 215;
``sparse_mla_prefill_mg_dual_kernel`` in the trace). So a 16-request DSpark verify
(16 x 7 = 112 rows) already runs the prefill kernel. ``compressed_sparse_mla_prefill``
models vLLM's BF16 gathered-workspace prefill, a different computation, and the
decode kind keys one valid-count per row, which a 2048-row prefill chunk cannot use
as a cache key. The valid counts here are derived from positions instead (every
query of one call shares the request geometry), which is exact for the uniform
batches the arch emits and saturates for contexts past ``extra_index_capacity x
compress_ratio`` tokens.
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

KIND = "paged_sparse_mla"


@dataclass(frozen=True)
class PagedSparseMlaArgs(KernelArgs):
    """One attention layer's call for a batch of equal-geometry requests.

    Row ``i`` of a request (0-based) sits at position ``context_before + i``. It
    attends to ``min(swa_window, pos + 1)`` sliding-window keys and, when
    ``compress_ratio > 0``, to ``min(extra_index_capacity, (pos + 1) // compress_ratio)``
    selected compressed keys.
    """

    num_requests: int = arg(unit="requests", doc="Requests in the call.")
    rows_per_request: int = arg(
        unit="tokens",
        doc="Query rows per request: 1 for decode, draft_tokens + 1 for speculative verify, the chunk for prefill.",
    )
    context_before: int = arg(unit="tokens", doc="Cached tokens before each request's first query row.")
    num_heads: int = arg(unit="heads", doc="Query heads on this GPU before any kernel padding.")
    head_dim: int = arg(unit="elements", doc="Elements in each query and key head (nope + rope).")
    value_dim: int = arg(unit="elements", doc="Elements in each output head.")
    swa_window: int = arg(unit="tokens", doc="Sliding-window keys per query.")
    swa_index_capacity: int = arg(unit="tokens", doc="Allocated sliding-window indices per query (multiple of 64).")
    compress_ratio: int = arg(unit="tokens", doc="Source tokens per compressed key; 0 = window-only layer.")
    extra_index_capacity: int = arg(unit="tokens", doc="Allocated compressed-key indices per query (multiple of 64).")
    page_size: int = arg(unit="tokens", doc="Tokens per physical page of the sliding-window cache.")
    kv_layout: str = arg(doc="Cache row format, e.g. 'v4' (448 fp8 nope + 64 bf16 rope + 8 scale bytes).")
    q_dtype: DType = arg(doc="Query element type.")
    cache_dtype: DType = arg(doc="Cache element type.")
    output_dtype: DType = arg(doc="Output element type.")


DOC = KernelDoc(
    title="Paged sparse MLA (sliding window + compressed keys)",
    summary="Attend each query row to its sliding-window keys and its selected compressed keys, from paged FP8 caches.",
    description=(
        "One attention layer of a DeepSeek-V4-family model as SGLang runs it on "
        "SM12x: a single paged call over two FP8 caches, the sliding-window cache "
        "and, in compressed layers, the compressed cache, each with its own padded "
        "index array and valid lengths. The call includes the page re-split of the "
        "touched sliding-window pages (256-token SGLang pages to the 64-token pages "
        "FlashInfer reads), the attention kernel, and the split-K merge on the "
        "decode path."
    ),
    category="Attention",
    subcategory="Compressed sparse MLA",
    formula=(
        "O = softmax(q · K_selectedᵀ · head_dim^-0.5, sink) · V_selected, per query head",
        "S = Σ_rows (swa_valid + extra_valid); R = num_requests · rows_per_request",
        "TFLOPS = 2·num_heads·(head_dim + value_dim)·S / time",
        "GB/s = [2·R·num_heads·head_dim + S·bytes_per_cache_row + 2·R·num_heads·value_dim] / time",
    ),
    default_metric="tflops",
    method=(
        f"{CUPTI_METHOD} Every launch of the call (page mark and split, attention, "
        "merge) is counted after three warm-up calls; cache construction is excluded."
    ),
    caveats=(
        "Selected compressed keys are spread evenly over the visible compressed "
        "positions rather than chosen by an indexer.",
        "Every request in a call has the same geometry (rows and context).",
    ),
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="sglang_flashinfer_sm120",
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.paged_sparse_mla_sglang_sm120",
            function_name="profile_paged_sparse_mla_sglang_sm120",
        ),
        table_name=KIND,
        args_schema=PagedSparseMlaArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        # flash_mla_sm120 imports FlashInfer's _sparse_mla_sm120 (CUTLASS SM120).
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.FP8_E4M3}),
            sm_targets=frozenset({"sm_120f"}),
        ),
        subprocess_env="sglang_env",
        doc=BackendDoc(
            summary=(
                "SGLang flash_mla_with_kvcache_sm120 (kernels/ops/attention/flash_mla_sm120.py:298) "
                "with SGLANG_SM120_FLASHMLA_BACKEND=flashinfer: FlashInfer sparse_mla_sm120 "
                "decode + merge for <= 64 rows, the paged prefill kernel above, over V4-layout "
                "FP8 caches written with SGLang's fused_store_cache."
            ),
            url="https://github.com/sgl-project/sglang",
        ),
    )
)

__all__ = ["KIND", "PagedSparseMlaArgs"]
