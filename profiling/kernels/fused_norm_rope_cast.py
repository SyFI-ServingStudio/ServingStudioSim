"""DeepInfer's fused RMSNorm + RoPE + cast + optional paged store
(``tilelang_ops/fused_transform.py`` ``fused_norm_rope_hadamard_cast_store_kvcache``).

Wire string: ``"fused_norm_rope_cast"``.

One ``_fused_norm_rope_hadamard_cast_store_kvcache_kernel`` launch per call.
DeepSeek-V4.1 prefill calls it four ways, all keyed here by geometry:

- SWA KV (``attn.py`` ``_prepare_swa_kv``): [T, 512], norm, RoPE 64,
  MXFP8 round trip back to bf16 (``output=cast_back``).
- Main compressed KV store (``compress.py`` ``forward_store``): [T/r, 512],
  norm, RoPE, NVFP4 into the main cache, plus the post-norm bf16 copy the
  KProj indexer reads (``output=cache_and_norm``).
- Index Q (``index_attn.py`` ``_quantize_q``): [T, 32, 128], RoPE, MXFP4
  values and scales (``output=quantized``).
- Index K store (``index_attn.py`` ``store_index_kv``): [T/r, 128], norm,
  RoPE, MXFP4 into the index cache (``output=cache``).

Why a new kind: no existing kind has this fused boundary (norm, rope and the
DeepInfer cache encodings in one launch).
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

KIND = "fused_norm_rope_cast"


@dataclass(frozen=True)
class FusedNormRopeCastArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Rows (tokens) transformed.")
    num_heads: int = arg(unit="heads", doc="Heads per token.")
    head_dim: int = arg(unit="elements", doc="Elements per head.")
    rope_dim: int = arg(unit="elements", doc="Trailing elements of each head that get RoPE.")
    norm: bool = arg(doc="Whether each head is RMS-normalized first.")
    cast_format: str = arg(
        doc=(
            "Cast: mxfp8 (E4M3, 32 per UE8M0 scale), nvfp4 (E2M1, 16 per E4M3 "
            "scale) or mxfp4 (E2M1, 32 per UE8M0 scale)."
        )
    )
    output: str = arg(
        doc=(
            "Destination: cast_back (bf16 round trip), quantized (values and "
            "scales), cache (paged store) or cache_and_norm (paged store plus "
            "the post-norm bf16 rows)."
        )
    )
    block_size: int = arg(
        unit="tokens", doc="Tokens per cache page for the cache outputs; 0 otherwise."
    )
    dtype: DType = arg(doc="Input element type (bf16).")


DOC = KernelDoc(
    title="Fused norm, RoPE and cast",
    summary=(
        "RMSNorm, RoPE and an FP8/FP4 cast of attention rows in one launch, "
        "optionally stored to a paged cache."
    ),
    description=(
        "Per token and head: optional RMSNorm over head_dim, interleaved RoPE "
        "on the last rope_dim elements, then a per-group MXFP8/NVFP4/MXFP4 "
        "cast. The result is cast back to bf16, returned as values plus scales, "
        "or written through a slot map into DeepInfer's token-interleaved main "
        "cache or block-sectioned index cache."
    ),
    category="Attention",
    subcategory="Compressed sparse MLA",
    formula=("bytes = 2·T·H·D read + the output the destination writes",),
    default_metric="time_ms",
    method=CUPTI_METHOD,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="tilelang",
        supports=BackendSupport(compute=frozenset({DType.BF16}), min_compute_capability=(9, 0)),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.deepinfer_kv_tilelang",
            function_name="profile_fused_norm_rope_cast_tilelang",
        ),
        table_name=KIND,
        args_schema=FusedNormRopeCastArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="deepinfer_env",
        doc=BackendDoc(
            summary=(
                "DeepInfer's TileLang fused_norm_rope_hadamard_cast_store_kvcache, "
                "checked bit-exact by DeepInfer's own correctness test on the same "
                "case."
            ),
        ),
    )
)
