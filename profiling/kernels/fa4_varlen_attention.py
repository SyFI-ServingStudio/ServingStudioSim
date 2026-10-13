"""FlashAttention-4 varlen attention kernel kind.

All Python-side per-kernel knowledge for ``fa4_varlen_attention`` lives here:
the wire string ``KIND``, the ``Fa4VarlenAttentionArgs`` schema, and the
``register(...)`` call that wires the kind into ``profiling.db.registry``.

Source: vLLM ``FlashAttentionDiffKVImpl.forward``
(``vllm/v1/attention/backends/flash_attn_diffkv.py``). Per layer and scheduler
step it makes ONE ``flash_attn_varlen_func(fa_version=4)`` call over the whole
varlen batch, decode and prefill requests together, reading K and V as the two
slices of one paged cache whose last dimension packs ``head_dim_qk`` key
channels then ``head_dim_v`` value channels. vLLM passes ``num_splits=0`` on
FA4 (its split cap is set only for FA3's ahead-of-time scheduler), so FA4
chooses the KV split itself. The call issues:

- ``FlashPrepareScheduler`` when the KV range is split (per-request split
  counts on device);
- the SM100 forward kernel;
- ``_flash_attn_fwd_combine`` when the KV range is split;
- tile-cumsum / blocks-to-batch helpers for a non-split batch of more than 256
  requests.

The whole call is the L1 unit; CUPTI takes the busy union of its launches.

``query_context_pairs`` is the exact batch: ``(query tokens, context tokens)``
per request, the context including the query. A mixed batch is one call, so it
is ONE spec. With ``cuda_graph`` a decode-only step (every query 1 token)
replays a FULL CUDA graph captured at ``padded_requests``: its host-side split
choice was frozen at capture over the padded batch and ``max_seqlen_k =
max_model_len``, and the padding requests run as empty queries over empty
contexts.
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

KIND = "fa4_varlen_attention"


@dataclass(frozen=True)
class Fa4VarlenAttentionArgs(KernelArgs):
    num_qo_heads: int = arg(unit="heads", doc="Query and output heads on this GPU.")
    num_kv_heads: int = arg(unit="heads", doc="KV heads on this GPU.")
    head_dim_qk: int = arg(unit="elements", doc="Dimension of each query and key head.")
    head_dim_v: int = arg(unit="elements", doc="Dimension of each value and output head.")
    window: int = arg(
        unit="tokens",
        doc="Sliding window including the query token itself; 0 means full causal.",
    )
    sinks: bool = arg(doc="Per-head attention-sink logits join every softmax denominator.")
    page_size: int = arg(unit="tokens", doc="Tokens per KV-cache page.")
    kv_layout: str = arg(
        doc="KV-cache page layout; lbnhc packs the K then V channels of one (token, head)."
    )
    q_dtype: DType = arg(doc="Query, sink and output element type.")
    kv_dtype: DType = arg(doc="Key and value cache element type.")
    max_model_len: int = arg(
        unit="tokens",
        doc="Model context limit; sizes the page table and a graph step's split choice.",
    )
    cuda_graph: bool = arg(
        doc="Decode-only steps replay a FULL CUDA graph; false runs every step eagerly."
    )
    query_context_pairs: tuple[tuple[int, int], ...] = arg(
        doc=(
            "(query tokens, context tokens) per request in the one varlen call; "
            "the context includes the query."
        )
    )
    padded_requests: int = arg(
        unit="requests",
        doc=(
            "Capture size a FULL-graph decode step is padded to (0 = unpadded); "
            "ignored for a step that runs eagerly."
        ),
    )


DOC = KernelDoc(
    title="FlashAttention-4 varlen attention",
    summary=(
        "Paged causal or sliding-window GQA attention with optional attention sinks "
        "and a value head dim that may differ from the query/key head dim, over one "
        "mixed decode and prefill batch."
    ),
    description=(
        "One flash_attn_varlen_func(fa_version=4) call covers the whole varlen batch "
        "of a scheduler step over a paged cache whose pages pack each token's key "
        "and value channels side by side. FlashAttention 4 chooses how many ways "
        "to split each request's KV range from the batch's longest query and "
        "context: a split call adds a scheduler launch that lowers the split per "
        "request and a combine launch that merges the partial outputs. A "
        "decode-only step small enough for a FULL CUDA graph keeps the split choice "
        "made at capture over the padded batch and the model's longest context."
    ),
    category="Attention",
    subcategory="MHA / GQA",
    formula=(
        "O = softmax([Q · Kᵀ · scale, sink]) · [V; 0]",
        "n_i = keys query i sees: i + 1 (full causal) or min(i + 1, window)",
        "TFLOPS = 2 · num_qo_heads · (head_dim_qk + head_dim_v) · Σ n_i / time",
        "GB/s = (Q + O + each request's visible K and V once) bytes / time",
    ),
    default_metric="tflops",
    method=(
        f"{CUPTI_METHOD} "
        "Every launch of the call is counted as the busy union of their "
        "intervals: the split scheduler, the forward kernel and the combine, "
        "whichever the batch issues. A FULL-graph decode step runs with the "
        "capture-time padded batch and max_seqlen_k = max_model_len, as vLLM "
        "replays it. Before timing, the output is checked against a PyTorch "
        "reference on sampled query rows."
    ),
    caveats=(
        "Queries, keys, values and sinks are random.",
        "The KV-cache insert belongs to its own kind, not to this call.",
        "KV pages are shuffled across a pool sized to the batch; the page table is "
        "max_model_len / page_size wide, as vLLM's, and pages wholly before a "
        "sliding window point at the null page, as vLLM's sliding-window manager "
        "leaves them.",
    ),
    reference="profiling.runners.attention.fa4_varlen_attention_reference",
)


# FA4's SM100 forward is built for sm_100f; vLLM picks FA4 for different QK/V
# head dims with sinks (flash_attn_diffkv.py get_flash_attn_version).
_SUPPORT = BackendSupport(
    compute=frozenset({DType.BF16}),
    kv=frozenset({DType.BF16}),
    sm_targets=frozenset({"sm_100f"}),
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_fa4",
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.fa4_varlen_attention",
            function_name="profile_fa4_varlen_attention_vllm_fa4",
        ),
        table_name=KIND,
        args_schema=Fa4VarlenAttentionArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=_SUPPORT,
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM's vllm_flash_attn flash_attn_varlen_func(fa_version=4, "
                "num_splits=0) as FlashAttentionDiffKVImpl.forward calls it."
            ),
        ),
    )
)

__all__ = ["Fa4VarlenAttentionArgs", "KIND"]
