"""Paged batch prefill attention kernel kind.

All Python-side per-kernel knowledge for ``paged_attn_batch_prefill`` lives
here: the wire string ``KIND``, the ``PagedAttnBatchPrefillArgs`` schema, and
the ``register(...)`` call that wires the kind into ``profiling.db.registry``.

Source: vLLM ``FlashInferImpl.forward`` (``vllm/v1/attention/backends/
flashinfer.py``) on a TRTLLM-attention deployment (quantized query). The
prefill requests of a step (more than one query token) make ONE
``flashinfer.prefill.trtllm_batch_context_with_kv_cache`` call per layer:
causal attention of each request's new tokens over its cached prefix and
themselves, read through the page table of a paged cache in HND layout whose
last dimension packs a head's K then V channels. A step's decode requests go
to ``paged_attn_batch_decode``, a separate call.

The batch is keyed by its ``(cached prefix, new tokens)`` pairs. The kernel
runs one CTA per 128-token query tile and query head, each reading every key up
to its tile's last token; which batches the simulator profiles and how it maps
a real step onto them is its own choice.
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

KIND = "paged_attn_batch_prefill"


@dataclass(frozen=True)
class PagedAttnBatchPrefillArgs(KernelArgs):
    num_qo_heads: int = arg(unit="heads", doc="Query and output heads on this GPU.")
    num_kv_heads: int = arg(unit="heads", doc="KV heads on this GPU.")
    head_dim: int = arg(unit="elements", doc="Dimension of each query, key and value head.")
    q_dtype: DType = arg(doc="Query element type.")
    kv_dtype: DType = arg(doc="Key and value cache element type.")
    o_dtype: DType = arg(doc="Output element type.")
    requests: tuple[tuple[int, int], ...] = arg(
        doc="(cached prefix tokens, new tokens) per prefill request in the call."
    )


DOC = KernelDoc(
    title="Paged batch prefill attention",
    summary=(
        "Causal GQA attention of several requests' new tokens over their cached "
        "prefixes and themselves, through a paged KV cache, in one call."
    ),
    description=(
        "One call covers every prefill request of a step, whether a fresh prompt, "
        "a chunk of a long prompt or a prompt after a prefix-cache hit. Each "
        "request's new tokens attend causally to its cached prefix and to each "
        "other."
    ),
    category="Attention",
    subcategory="MHA / GQA",
    formula=(
        "O[t, h] = softmax(Q[t, h] · K[:p_t, h // G]ᵀ · scale) · V[:p_t, h // G]",
        "p_t = keys token t sees: its request's prefix plus its position + 1",
        "TFLOPS = 4 · num_qo_heads · head_dim · Σ p_t / time",
        "GB/s = (Q + O + every request's K and V once) bytes / time",
    ),
    default_metric="tflops",
    method=(
        f"{CUPTI_METHOD} "
        "Every launch of the call counts. Before timing, the output is checked "
        "against a PyTorch reference on sampled query rows."
    ),
    caveats=(
        "Queries, keys and values are random; the quantization scales are 1.",
        "Pages hold 32 tokens: 16, 32 and 64 measured within 1%, so the page size "
        "is not part of the key.",
        "The kernel tiles each request's new tokens by 128 and a partial tile costs "
        "a full one, so many short requests cost more than one long request with "
        "the same query-key pairs.",
    ),
    reference="profiling.runners.attention.fa4_varlen_attention_reference",
)


# trtllm-gen context cubins ship for sm_100f; vLLM routes FP8 queries over an
# FP8 cache to them (flashinfer.py use_trtllm_attention).
_SUPPORT = BackendSupport(
    compute=frozenset({DType.FP8_E4M3}),
    kv=frozenset({DType.FP8_E4M3}),
    sm_targets=frozenset({"sm_100f"}),
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_trtllm_gen",
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.paged_attn_batch_trtllm_gen",
            function_name="profile_paged_attn_batch_prefill",
        ),
        table_name=KIND,
        args_schema=PagedAttnBatchPrefillArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=_SUPPORT,
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "FlashInfer's trtllm_batch_context_with_kv_cache as vLLM's "
                "FlashInferImpl.forward calls it for the prefill requests."
            ),
        ),
    )
)

__all__ = ["KIND", "PagedAttnBatchPrefillArgs"]
