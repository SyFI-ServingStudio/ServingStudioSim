"""Relative-position attention (tml FA4) kernel kind.

All Python-side per-kernel knowledge for ``rel_bias_attention`` lives here: the
wire string ``KIND``, the ``RelBiasAttentionArgs`` schema, and the
``register(...)`` call that wires the kind into ``profiling.db.registry``.

Source: the alignment vLLM fork (``alignment/profiler/vllm``). Per layer,
``InklingAttention._attention`` (``vllm/models/inkling/nvidia/attention.py``)
computes ``num_splits = inkling_fa4_num_splits(...)`` and makes ONE call to
``InklingFA4RelAttentionKernel.kernel``
(``vllm/models/inkling/nvidia/ops/fa4_rel_attention.py``) over the whole
varlen batch, decode and prefill requests together. On SM10x that call is
``vllm.third_party.tml_fa4.flash_attn_varlen_func(rel_bias=...)``, which issues:

- ``CuSeqlensToBlocks`` + ``CuBlocksToBatch`` when the bucketed max query
  length exceeds one M block (in practice any request with more than 64 query
  tokens at GQA 4);
- ``ShearingBias`` (every call): copies the (T, H, E) relative logits into a
  padded, KV-tile-aligned bias tensor whose padding encodes the causal and
  window masks;
- ``FlashPrepareScheduler`` when ``num_splits > 1``;
- ``FlashAttentionForwardSm100`` with the bias tiles added to the scores;
- ``FlashAttentionForwardCombine`` when ``num_splits > 1``.

The whole call is the L1 unit; CUPTI sums every launch. The math per query
token at absolute position ``i`` and key ``j <= i`` is
``s = (q_i . k_j) / head_dim + rel_logits[i, h, i - j]`` when
``i - j < rel_extent`` (else ``+ 0``), with local layers also masking
``i - j >= rel_extent`` (window = rel_extent tokens including self).

``query_context_pairs`` is the exact batch: ``(query tokens, context tokens)``
per request, the context including the query. Decode requests are ``(1, kv)``;
a mixed batch is one call, so it is ONE spec (never a decode spec plus a
prefill spec). ``max_model_len`` is identity because it drives the global
layers' split count.
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

KIND = "rel_bias_attention"


@dataclass(frozen=True)
class RelBiasAttentionArgs(KernelArgs):
    num_qo_heads: int = arg(unit="heads", doc="Query and output heads on this GPU.")
    num_kv_heads: int = arg(unit="heads", doc="KV heads on this GPU.")
    head_dim: int = arg(unit="elements", doc="Dimension of each query, key and value head.")
    is_local: bool = arg(
        doc="Sliding-window layer (window = rel_extent tokens) rather than full causal."
    )
    rel_extent: int = arg(
        unit="tokens", doc="Distances that carry a relative-position logit; the window if local."
    )
    page_size: int = arg(unit="tokens", doc="Tokens per KV-cache page.")
    kv_layout: str = arg(
        doc="KV-cache page layout; lbnhc packs K and V of one (token, head) side by side."
    )
    q_dtype: DType = arg(doc="Query, relative-logit and output element type.")
    kv_dtype: DType = arg(doc="Key and value cache element type.")
    max_model_len: int = arg(
        unit="tokens", doc="Model context limit; sizes the page table and the global split count."
    )
    query_context_pairs: tuple[tuple[int, int], ...] = arg(
        doc=(
            "(query tokens, context tokens) per request in the one varlen call; "
            "the context includes the query."
        )
    )


DOC = KernelDoc(
    title="Relative-position attention",
    summary=(
        "Paged causal or sliding-window GQA attention with a per-query relative-position "
        "logit added to every score, over one mixed decode and prefill batch."
    ),
    description=(
        "No RoPE is applied: each query row carries rel_extent learned "
        "logits, one per distance, added to its scores against the last rel_extent "
        "keys. Local layers also restrict keys to that window; global layers are "
        "full causal with no bias beyond it. One call covers the whole varlen batch "
        "over a paged LBNHC KV cache (K and V of one token and head packed side by "
        "side). On Blackwell the call first shears the logits into a KV-tile-aligned "
        "bias tensor, then runs FlashAttention 4 with bias tiles; global layers split "
        "the KV range by the fork's inkling_fa4_num_splits rule and combine the "
        "partials."
    ),
    category="Attention",
    subcategory="MHA / GQA",
    formula=(
        "O = softmax(Q · Kᵀ / head_dim + B_rel) · V, B_rel[i, j] = rel_logits[i, h, i − j] "
        "for 0 ≤ i − j < rel_extent, else 0 (global) or masked (local)",
        "n_i = keys query i sees: i + 1 (global) or min(i + 1, rel_extent) (local)",
        "TFLOPS = 4 · num_qo_heads · head_dim · Σ n_i / time",
        "GB/s = (Q + O + rel_logits + each request's visible K and V once) bytes / time",
    ),
    default_metric="tflops",
    method=(
        f"{CUPTI_METHOD} "
        "Every launch of the call is counted: the cu-blocks metadata pair, "
        "ShearingBias, the split scheduler, FlashAttention forward and the "
        "combine, whichever the batch issues. num_splits follows the fork's "
        "inkling_fa4_num_splits. Before timing, the output is checked against a "
        "PyTorch reference on sampled query rows."
    ),
    caveats=(
        "Queries, keys, values and logits are random; q is drawn with standard "
        "deviation 4 so both the QK and the bias terms move the softmax.",
        "The cache insert and the logits projection belong to the QKV-prep kind, not to this call.",
        "KV pages are shuffled across a pool sized to the batch; the page table is "
        "max_model_len / page_size wide, as vLLM's.",
    ),
    reference="profiling.runners.attention.rel_bias_attention_reference",
)


# tml_fa4's sheared-bias path runs only on SM10x/SM11x (fa4_rel_attention.py
# `_use_sheared_bias`); FlashAttentionForwardSm100 is built for sm_100f.
_SUPPORT = BackendSupport(
    compute=frozenset({DType.BF16}),
    kv=frozenset({DType.BF16}),
    sm_targets=frozenset({"sm_100f"}),
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="tml_fa4",
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.rel_bias_attention",
            function_name="profile_rel_bias_attention_tml_fa4",
        ),
        table_name=KIND,
        args_schema=RelBiasAttentionArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=_SUPPORT,
        subprocess_env="vllm_upstream_fork_env",
        doc=BackendDoc(
            summary=(
                "The fork's InklingFA4RelAttentionKernel.kernel: tml_fa4 "
                "flash_attn_varlen_func with the sheared relative bias, split by "
                "inkling_fa4_num_splits."
            ),
        ),
    )
)

__all__ = ["RelBiasAttentionArgs", "KIND"]
