"""Prefill sparse MLA core attention with fused Q norm, RoPE and FP8 output cast.

Wire string: ``"sparse_mla_norm_rope_cast_prefill"`` -- matches Rust
``KernelSpec::KIND`` in
``simulator/src/timing/kernels/sparse_mla_norm_rope_cast_prefill.rs``.

One ``sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn::fwd_kernel``
launch per layer and GPU: FlashMLA's ``fused_norm_rope_attn_rope_cast.prefill``
as DeepInfer's ``prefill_attn_varlen_sparse`` calls it
(``layers/attention_backend.py``, kwargs from ``V4AttentionBase.
_fused_norm_rope_cast_kwargs`` in ``models/v4_modules/attn.py``). Per query row
it applies the Q RMSNorm (``enable_q_norm``) and GPT-J RoPE on the last 64 dims,
attends with one latent KV head and an attention sink to the rows its index row
names in a gathered bf16 KV buffer, then applies the inverse RoPE and casts the
output to FP8 E4M3 with packed UE8M0 per-32 scales in ``wo_a``'s permuted layout.
The index row is the request's sliding window (``window_size`` columns) followed,
on compressed layers, by the indexer's ``index_topk`` compressed rows.

Why a new kind and not a backend of ``compressed_sparse_mla_rope_cast``: that
kind's prefill row is vLLM's whole chunk segment (paged MXFP8/NVFP4 cache
gathers, the index combine and this attention, no Q norm) keyed by cache
formats. DeepInfer launches its gathers as separate kernels
(``_load_paged_kv_cache_kernel``, ``_kv_index_cat_kernel``, ``_store_kv_cache``,
a DtoD copy), so its boundary is the attention launch alone, on a bf16 buffer,
with the Q norm fused.

Key: the step's ``(query, context)`` requests, the layer's window, compression
ratio and top-k, and the model's head geometry. The context sets each request's
compressed rows (``context // ratio``), over which the top-k spreads: at 16384
query rows (ratio 2) contexts of 16384 / 131072 / 524288 took 1419 / 1503 / 1799
us on B200.
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

KIND = "sparse_mla_norm_rope_cast_prefill"


@dataclass(frozen=True)
class SparseMlaNormRopeCastPrefillArgs(KernelArgs):
    query_context_pairs: tuple[tuple[int, int], ...] = arg(
        doc="(query rows, context tokens) per request; the context includes the query."
    )
    window_size: int = arg(unit="tokens", doc="Sliding-window keys in each query row's index.")
    compress_ratio: int = arg(
        unit="tokens",
        doc="Source tokens pooled into one compressed key; 0 means window keys only.",
    )
    index_topk: int = arg(
        unit="tokens", doc="Compressed keys the indexer selects per query row; 0 without them."
    )
    num_heads: int = arg(unit="heads", doc="Query heads per row (64 or 128).")
    head_dim: int = arg(unit="elements", doc="Elements in each query head and latent key.")
    rope_dim: int = arg(unit="elements", doc="Trailing head elements rotated by RoPE.")
    n_wv_group: int = arg(unit="groups", doc="Output projection groups (num_heads / 8).")
    enable_q_norm: bool = arg(doc="Whether the kernel applies the per-head Q RMSNorm.")
    q_dtype: DType = arg(doc="Element type of the query input.")
    kv_dtype: DType = arg(doc="Element type of the gathered KV buffer.")
    output_dtype: DType = arg(doc="Element type of the attention output after the cast.")


DOC = KernelDoc(
    title="Prefill sparse MLA with fused Q norm, RoPE and FP8 cast",
    summary=(
        "Normalize and rotate the queries, attend to each row's window and selected "
        "compressed keys in a gathered bf16 buffer, then un-rotate and cast the output to FP8."
    ),
    description=(
        "FlashMLA's fused prefill kernel as DeepInfer calls it: per query row, an RMSNorm "
        "over each head, RoPE on the last rope_dim elements, sparse attention with one "
        "latent KV head and an attention sink over the window_size + index_topk keys its "
        "index row names in a gathered bf16 KV buffer, the inverse RoPE and an FP8 E4M3 "
        "cast with one UE8M0 scale per 32 elements in the output projection's layout. "
        "The buffer gathers and the index concatenation are separate launches."
    ),
    category="Attention",
    subcategory="Compressed sparse MLA",
    formula=(
        "O = RoPE⁻¹(softmax(RoPE(norm(q)) · K_idxᵀ / √head_dim, sink) · V_idx), cast to FP8",
        "TFLOPS = 4 · num_heads · head_dim · (valid index entries) / time",
        "GB/s = [T · num_heads · (2 · head_dim + head_dim + head_dim/32) "
        "+ 4 · T · (window_size + index_topk) + 2 · head_dim · valid entries] / time",
    ),
    default_metric="tflops",
    method=(
        f"{CUPTI_METHOD} One fwd_kernel launch per call. Before timing, sampled rows "
        "of the FP8 output are checked against an fp32 Torch composite. Each query row "
        "selects min((position + 1) // compress_ratio, index_topk) causal compressed "
        "rows, one per equal stratum, so no indexer runs."
    ),
    caveats=(
        "The runner takes head_dim 512, rope_dim 64, num_heads 64 or 128 and contexts "
        "up to 1048576.",
        "The kernel's time follows the index width, not the count of valid entries: "
        "YoCo's clipped window is built as the full window.",
        "Real indexer selections are not stratified; their reuse across neighbouring "
        "rows changes the time (shared per-request selections ran 7% faster at 16384 rows).",
        "Each request's rows reuse only their own top-k rows, so many short requests "
        "run slower than one request of the same rows: 4 x (140, 131072) at ratio 1 took "
        "63.0 us against 56.6 us for the rows in one request.",
        "Only the sparse top-k layout is profiled; the dense topk_length layout "
        "(compressed layers without an indexer) is not.",
    ),
    reference="profiling.runners.attention.sparse_mla_norm_rope_cast_prefill",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashmla_deepinfer",
        # The fused kernel is SM100/SM103 only (flash_mla
        # fused_norm_rope_attn_rope_cast.prefill docstring).
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.BF16}),
            sm_targets=frozenset({"sm_100f"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.sparse_mla_norm_rope_cast_prefill",
            function_name="profile_sparse_mla_norm_rope_cast_prefill_flashmla",
        ),
        table_name=KIND,
        args_schema=SparseMlaNormRopeCastPrefillArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="deepinfer_env",
        doc=BackendDoc(
            summary=(
                "FlashMLA fused_norm_rope_attn_rope_cast.prefill from DeepInfer's image, "
                "with DeepInfer's kwargs and gathered-buffer index layout."
            ),
            url="https://github.com/deepseek-ai/FlashMLA",
        ),
    )
)

__all__ = ["KIND", "SparseMlaNormRopeCastPrefillArgs"]
