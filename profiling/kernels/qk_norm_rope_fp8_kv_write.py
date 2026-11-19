"""Fused per-head q/k RMSNorm, NeoX RoPE, fp8 query quant and fp8 K/V paged-cache write.

One launch takes a GQA layer's packed ``qkv`` projection output for ``num_tokens``
rows: every query and key head is RMS-normalized with its own weight and rotated
by NeoX RoPE at the row's position; queries are written as fp8 (static scale)
for an fp8 attention call, and the rotated keys with the raw values are written
as fp8 into a paged FlashAttention-layout cache (``[pages, kv_heads, page,
2 * head_dim]``, K then V) at each row's slot. It replaces q_norm, k_norm, RoPE,
the query QuantFP8 and ``reshape_and_cache_flash``.
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

KIND = "qk_norm_rope_fp8_kv_write"


@dataclass(frozen=True)
class QkNormRopeFp8KvWriteArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Rows of the packed qkv projection output.")
    num_qo_heads: int = arg(unit="heads", doc="Query heads per row.")
    num_kv_heads: int = arg(unit="heads", doc="Key and value heads per row.")
    head_dim: int = arg(unit="elements", doc="Elements per head.")
    page_size: int = arg(unit="tokens", doc="Tokens per KV-cache page.")
    input_dtype: DType = arg(doc="Element type of the qkv input.")
    kv_dtype: DType = arg(doc="Element type of the query output and the KV cache.")


DOC = KernelDoc(
    title="Q/K norm, RoPE, fp8 query and KV cache write",
    summary=(
        "Normalize and rotate queries and keys, quantize them to fp8 and write "
        "K/V into the paged cache."
    ),
    description=(
        "Between a GQA layer's qkv projection and an fp8 attention call, one "
        "Triton launch normalizes each query and key head, applies NeoX RoPE, "
        "writes the queries as fp8 and scatters the rotated keys and the values "
        "as fp8 into the paged cache. The DFlash2 draft uses it in place of five "
        "launches."
    ),
    category="Attention",
    subcategory="MHA / GQA",
    formula=(
        "q' = fp8(RoPE(RMSNorm(q))), k' = fp8(RoPE(RMSNorm(k))), v' = fp8(v)",
        "GB/s = num_tokens · head_dim · ((Hq + 2 Hkv) · 2 + Hq + 2 Hkv) / time",
    ),
    default_metric="time_ms",
    method=(
        f"{CUPTI_METHOD} Before timing, the query and cache fp8 codes are "
        "compared with the incumbent sequence (native norm/RoPE/quant plus vLLM "
        "reshape_and_cache_flash): at least 90% of codes must match."
    ),
    caveats=(
        "Not bit-exact with the incumbent: about 95% of fp8 codes match, the "
        "rest differ by bf16 rounding of small values.",
        "Scales are 1.0 (the draft is unquantized).",
    ),
    reference=None,
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="triton",
        supports=BackendSupport(compute=frozenset({DType.BF16}), kv=frozenset({DType.FP8_E4M3})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.dflash2.draft_chain",
            function_name="profile_qk_norm_rope_fp8_kv_write_triton",
        ),
        table_name=KIND,
        args_schema=QkNormRopeFp8KvWriteArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "In-repo qk_norm_rope_fp8 Triton kernel (runners/dflash2/draft_chain_vendor.py)."
            )
        ),
    )
)

__all__ = ["KIND", "QkNormRopeFp8KvWriteArgs"]
