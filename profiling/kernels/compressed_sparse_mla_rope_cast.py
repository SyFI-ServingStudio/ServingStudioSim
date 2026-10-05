"""Compressed sparse MLA with fused query RoPE and output inverse RoPE / FP8 cast.

One layer's decode or prefill segment of FlashMLA's "mega" attention.

Source: alignment fork ``servingstudio-alignment-v41``. The kernel is
``vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py``
(``DeepseekV4MegaAttnAttention``), which ``nvidia/model.py`` selects by
default on SM100. Per layer, ``forward_mqa`` makes one call per segment: the
prefill tokens and the decode tokens write disjoint ranges of one output
buffer. ``mode`` picks the segment:

- ``decode`` is ``_forward_decode_mega``, one
  ``fused_norm_rope_attn_rope_cast_decode`` launch over the paged caches;
- ``prefill`` is ``_forward_prefill_mega``. For each chunk of up to
  ``prefill_chunk_size`` requests, it launches the compressed-cache gather
  (``_dequantize_and_gather_k_nvfp4`` when ratio > 0), the MXFP8 SWA gather
  (CuTe DSL), ``combine_topk_swa_indices`` and one
  ``fused_norm_rope_attn_rope_cast_fwd``. All of these launches are in the slot.

The kernel math has Q RoPE (GPT-J pairs, last 64 of 512 dims) and 1-KV-head
sparse MLA with an attention sink. It attends over the 128-token sliding
window (MXFP8, 528 B/token) plus, for ``compress_ratio`` 1 or 2, up to
``index_topk`` indexer-selected compressed rows (NVFP4 ``nvfp4_ds_mla``,
288 B/token). The output then gets the inverse RoPE and an FP8 E4M3 cast with
UE8M0 per-32 scales, written in ``wo_a``'s permuted layout. The fork passes
``enable_q_norm=False``, so the kernel does no Q RMSNorm: V4.1 norms ``qr``
before ``wq_b``. Outside the slot: the fused Q-pad/KV insert, the decode
global top-k remap (``compute_global_topk_indices_and_lens``), and
``wo_a``/``wo_b``.

``num_heads`` is the kernel's padded Q head count (64 or 128). TP4 on the
64-head checkpoint leaves 16 live heads, padded to 64. The kernel runs, and
writes output for, all padded heads, so the live count does not change the
execution. ``query_context_pairs`` holds ``(query_len, context_len)`` per
request, with context including the query. In decode mode every query token
is one flattened row. ``max_model_len``, ``max_num_batched_tokens`` and
``prefill_chunk_size`` drive the prefill chunk planner, and decode ignores them.
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

KIND = "compressed_sparse_mla_rope_cast"


@dataclass(frozen=True)
class CompressedSparseMlaRopeCastArgs(KernelArgs):
    mode: str = arg(doc="Segment one call times: decode or prefill.")
    query_context_pairs: tuple[tuple[int, int], ...] = arg(
        doc=(
            "(query tokens, context tokens) per request, in tokens; the context includes the query."
        )
    )
    compress_ratio: int = arg(
        unit="tokens",
        doc="Source tokens pooled into one compressed key; 0 means window keys only.",
    )
    window_size: int = arg(unit="tokens", doc="Sliding-window keys each query can attend to.")
    index_topk: int = arg(
        unit="tokens", doc="Most compressed keys the indexer selects for one query token."
    )
    num_heads: int = arg(
        unit="heads", doc="Query heads the kernel runs, after padding the live heads."
    )
    head_dim: int = arg(unit="elements", doc="Elements in each query head and latent key.")
    rope_dim: int = arg(unit="elements", doc="Trailing head elements rotated by RoPE.")
    max_model_len: int = arg(
        unit="tokens", doc="Longest context the prefill chunk planner provisions for."
    )
    max_num_batched_tokens: int = arg(
        unit="tokens", doc="Most tokens in one scheduler step, sizing the prefill gather buffers."
    )
    prefill_chunk_size: int = arg(
        unit="requests", doc="Most prefill requests gathered and attended in one chunk."
    )
    q_dtype: DType = arg(doc="Element type of the query input.")
    swa_cache_format: str = arg(doc="Record format of the sliding-window KV cache.")
    compressed_cache_format: str = arg(
        doc="Record format of the compressed KV cache, or none without one."
    )
    output_dtype: DType = arg(doc="Element type of the attention output after the cast.")


DOC = KernelDoc(
    title="Compressed sparse MLA with fused RoPE and FP8 cast",
    summary=(
        "Rotate the queries, attend to sliding-window and selected compressed keys, "
        "then un-rotate the output and cast it to FP8, in one attention segment."
    ),
    description=(
        "A fused form of compressed sparse MLA. Each query head gets RoPE on its "
        "last rope_dim elements, attends over one shared latent KV head, with an "
        "attention sink, to its sliding-window keys and, in compressed layers, "
        "to up to index_topk compressed keys the indexer selected. The output gets "
        "the inverse RoPE and an FP8 E4M3 cast with one UE8M0 scale per 32 "
        "elements, written in the layout the output projection reads. Per layer "
        "the decode and prefill tokens are two calls on disjoint output rows. "
        "Decode is one launch over the paged caches. Prefill runs, per chunk of "
        "up to prefill_chunk_size requests, the compressed and window cache "
        "gathers, the index merge and one attention launch, all counted here."
    ),
    category="Attention",
    subcategory="Compressed sparse MLA",
    formula=(
        "O = RoPE⁻¹(softmax(RoPE(q) · K_selᵀ / √head_dim, sink) · V_sel), cast to FP8",
        "T = Σ query tokens; S = window keys + compressed keys selected over all T rows",
        "TFLOPS = 4 · num_heads · head_dim · S / time",
        "decode GB/s = [T · num_heads · (2 · head_dim + head_dim + head_dim/32) "
        "+ 4 · T · (window_size + index_topk if compress_ratio > 0) "
        "+ record bytes of the S selected keys] / time",
        "prefill GB/s: the same query and output bytes, plus (record + 1024) bytes "
        "per gathered row and 1028 bytes per selected key",
    ),
    default_metric="tflops",
    method=(
        f"{CUPTI_METHOD} "
        "Every launch of the segment is counted, including prefill's gathers and "
        "index merge. Before timing, the flashmla_mega output is checked against "
        "the torch composite. Each query token selects exactly "
        "min((position + 1) // compress_ratio, index_topk) causal compressed keys, "
        "spread evenly, so no indexer runs."
    ),
    caveats=(
        "Only head_dim 512, rope_dim 64, window_size 128, index_topk 512, "
        "prefill_chunk_size 4, num_heads 64 or 128, compress_ratio 0, 1 or 2, bf16 "
        "queries, an MXFP8 window cache (528 B per token) and an NVFP4 (288 B) or "
        "MXFP8 compressed cache are measured, on B200.",
        "num_heads is the padded head count: the kernel computes every padded "
        "head, so fewer live heads take the same time.",
        "Decode takes at most 2048 query rows and 256 requests. The global top-k "
        "remap that precedes decode is outside the call; the runner passes "
        "precomputed cache slots.",
        "The kernel applies no query RMSNorm.",
    ),
    reference="profiling.runners.attention.compressed_sparse_mla_rope_cast_reference",
)


# No capability rule: the Torch reference runs on any CUDA device.
_TORCH_SUPPORT = BackendSupport(
    compute=frozenset({DType.BF16}),
    kv=frozenset({DType.FP8_E4M3}),
)
# FlashMLA's mega-attention kernel has only an SM100 build; vLLM gates the
# layer on is_device_capability_family(100) (flash_mla_mega_attn.py).
_MEGA_SUPPORT = BackendSupport(
    compute=frozenset({DType.BF16}),
    kv=frozenset({DType.FP8_E4M3}),
    sm_targets=frozenset({"sm_100f"}),
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.compressed_sparse_mla_rope_cast",
            function_name="profile_compressed_sparse_mla_rope_cast_torch",
        ),
        table_name=KIND,
        args_schema=CompressedSparseMlaRopeCastArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=_TORCH_SUPPORT,
        doc=BackendDoc(
            summary=(
                "Unfused PyTorch composite of the same math (cache decode, RoPE, "
                "sparse attention with sink, inverse RoPE, FP8 cast); a semantic "
                "reference, not a serving path."
            ),
        ),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashmla_mega",
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.compressed_sparse_mla_rope_cast",
            function_name="profile_compressed_sparse_mla_rope_cast_flashmla_mega",
        ),
        table_name=KIND,
        args_schema=CompressedSparseMlaRopeCastArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=_MEGA_SUPPORT,
        subprocess_env="vllm_upstream_fork_env",
        doc=BackendDoc(
            summary=(
                "FlashMLA's fused_norm_rope_attn_rope_cast decode and prefill kernels, "
                "through the vLLM mega-attention layer's segment methods, prefill "
                "gathers included."
            ),
            url="https://github.com/deepseek-ai/FlashMLA",
        ),
    )
)

__all__ = ["CompressedSparseMlaRopeCastArgs", "KIND"]
