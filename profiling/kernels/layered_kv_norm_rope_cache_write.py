"""Every layer's context K/V: grouped K RMSNorm, NeoX RoPE and fp8 paged-cache write.

A drafter that precomputes all its layers' K/V from the target's hidden states
holds a ``[num_tokens, num_layers, 2, num_kv_heads, head_dim]`` projection. One
launch normalizes each layer's keys with that layer's weight, applies NeoX
RoPE, and writes K and V as fp8 into each layer's paged FlashAttention-layout
cache (``[pages, kv_heads, page, 2 * head_dim]``). It replaces the layer-major
transpose, the grouped RMSNorm, the RoPE call and one ``reshape_and_cache_flash``
per layer.
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

KIND = "layered_kv_norm_rope_cache_write"


@dataclass(frozen=True)
class LayeredKvNormRopeCacheWriteArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Context rows whose K/V are written.")
    num_layers: int = arg(unit="layers", doc="Layers whose caches receive the rows.")
    num_kv_heads: int = arg(unit="heads", doc="Key and value heads per layer.")
    head_dim: int = arg(unit="elements", doc="Elements per head.")
    page_size: int = arg(unit="tokens", doc="Tokens per KV-cache page.")
    input_dtype: DType = arg(doc="Element type of the K/V projection.")
    kv_dtype: DType = arg(doc="Element type of the KV cache.")


DOC = KernelDoc(
    title="Multi-layer K norm, RoPE and KV cache write",
    summary=(
        "Normalize, rotate and scatter every layer's context K/V into its "
        "paged cache in one launch."
    ),
    description=(
        "The DFlash2 drafter turns the target's hidden states into all of its "
        "layers' K/V with one GEMM; this one Triton launch (the vLLM fork's Beam "
        "store_context_kv) finishes them: per-layer K RMSNorm, NeoX RoPE, and the "
        "fp8 write of K and V into each layer's cache."
    ),
    category="Attention",
    subcategory="MHA / GQA",
    formula=(
        "GB/s = num_tokens · num_layers · 2 · Hkv · head_dim · (in bytes + cache bytes) / time",
    ),
    default_metric="time_ms",
    method=(
        f"{CUPTI_METHOD} Before timing, every layer's cache bytes are compared "
        "with the incumbent sequence (transpose, vLLM rms_norm, rotary_embedding, "
        "reshape_and_cache_flash per layer): at least 99.9% must match."
    ),
    caveats=("Scales are 1.0 (the draft is unquantized).",),
    reference=None,
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        supports=BackendSupport(compute=frozenset({DType.BF16}), kv=frozenset({DType.FP8_E4M3})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.dflash2.draft_chain",
            function_name="profile_layered_kv_norm_rope_cache_write_vllm_triton",
        ),
        table_name=KIND,
        args_schema=LayeredKvNormRopeCacheWriteArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary="vLLM fork vllm.models.beam.nvidia.dflash2_context_ops.store_context_kv."
        ),
    )
)

__all__ = ["KIND", "LayeredKvNormRopeCacheWriteArgs"]
