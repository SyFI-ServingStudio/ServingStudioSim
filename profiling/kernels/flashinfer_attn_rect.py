"""FlashInfer rect (non-causal) attention kernel kind.

All Python-side per-kernel knowledge for ``flashinfer_attn_rect`` lives here: the
wire string ``KIND``, the ``FlashinferAttnRectArgs`` schema, and the
``register(...)`` calls (one per backend) that wire this kernel into
``profiling.db.registry``.

Wire string ``"flashinfer_attn_rect"`` matches Rust ``KernelSpec::KIND`` in
``simulator/src/timing/kernels/flashinfer_attn_rect.rs`` and the facade stem
``get_flashinfer_attn_rect_times`` / ``count_missing_flashinfer_attn_rect``.

Rect is the **non-causal** sibling of ``flashinfer_attn_prefill`` (``causal=False``
on the runner side), but is parametrized by ``(q_len, kv_len)`` *directly* rather
than prefill's ``(prefix_len, append_len)``. For non-causal attention there is no
causal prefix/append split — only the two lengths matter — and the direct form
also lets rect express ``q_len > kv_len`` (wide rectangles), which the prefill
encoding (``kv_len = prefix_len + append_len`` >= ``q_len``) cannot. The runner
(shared with prefill in
``profiling.runners.attention.flashinfer_attn_prefill_and_rect``) passes these
straight to the ragged wrapper.

Backends ``{fa2, fa3, cudnn, vllm_fa4_split32, vllm_fa4_nosplit}`` are ServingStudioSim
backend strings, not kinds: each
registers its own spec -> the same ``args_schema`` and table, routing to its own
``profile_flashinfer_attn_rect_<backend>`` entry. Unsupported combos raise
``ProfilerNotImplemented`` at profile time: ``cudnn+fp8``.

Shape split: static Config = ``(num_qo_heads, num_kv_heads, head_dim, q_dtype,
kv_dtype, o_dtype)``; runtime 2D sweep Input = ``(q_len, kv_len)``.
Metric family COMPUTE, cache ``Cache2DLinear`` (two monotonic axes).

Importing this module appends ``KernelProfilerSpec`` rows to the registry. The
runner module is referenced lazily via ``RunnerRef`` so the main process never
eager-imports flashinfer/torch/cuda.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import CUPTI_METHOD, BackendDoc, KernelDoc, arg, optional_arg
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND: str = "flashinfer_attn_rect"

_RUNNER_MODULE = "profiling.runners.attention.flashinfer_attn_prefill_and_rect"
_BACKENDS = ("fa2", "fa3", "cudnn")

# Per-backend capability on compute (q_dtype) and kv-cache (kv_dtype) axes; fa2
# runs bf16-q / fp8-kv. See flashinfer_attn_prefill for the rationale.
_SUPPORTS = {
    "fa2": BackendSupport(
        compute=frozenset({DType.BF16}), kv=frozenset({DType.BF16, DType.FP8_E4M3})
    ),
    "fa3": BackendSupport(
        compute=frozenset({DType.BF16, DType.FP8_E4M3}),
        kv=frozenset({DType.BF16, DType.FP8_E4M3}),
    ),
    "cudnn": BackendSupport(compute=frozenset({DType.BF16}), kv=frozenset({DType.BF16})),
}


@dataclass(frozen=True)
class FlashinferAttnRectArgs(KernelArgs):
    # Static attention dims (Rust <Name>KernelConfig, minus `backends`).
    num_qo_heads: int = arg(unit="heads", doc="Query and output heads on this GPU.")
    num_kv_heads: int = arg(unit="heads", doc="KV heads on this GPU.")
    head_dim: int = arg(unit="elements", doc="Dimension of each query, key and value head.")
    q_dtype: DType = arg(doc="Query element type.")
    kv_dtype: DType = arg(doc="Key and value element type.")
    o_dtype: DType = arg(doc="Output element type.")
    # Runtime 2D sweep coords (Rust <Name>KernelInput). Direct, not prefix/append.
    q_len: int = arg(unit="tokens", doc="Query tokens in each request.")
    kv_len: int = arg(unit="tokens", doc="Key and value tokens each request attends.")
    # Rows written before the batch axis existed measured one request.
    num_requests: int | None = optional_arg(
        absent=1,
        unit="requests",
        doc="Requests in the one call, each with its own q_len queries and kv_len keys and values.",
    )


DOC = KernelDoc(
    title="Non-causal rectangular attention",
    summary="Attend every query token over every key and value token in one request.",
    description=(
        "Attention without a causal mask: every query token sees every key and "
        "value token of its request. The DFlash2 draft layer uses it so each "
        "drafted position sees the whole context. One call covers num_requests "
        "requests, each a block of q_len queries over its own kv_len keys and "
        "values, so the keys and values are read once per request."
    ),
    category="Attention",
    subcategory="MHA / GQA",
    formula=(
        "O = softmax(Q · Kᵀ / √head_dim) · V, per query head",
        "TFLOPS = 4 · num_requests · q_len · kv_len · num_qo_heads · head_dim / time",
        "GB/s = num_requests · (q_len · num_qo_heads · head_dim · (Q + O bytes) "
        "+ 2 · kv_len · num_kv_heads · head_dim · KV bytes) / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "Planning and input construction are not timed; every launch of the "
        "attention call is counted. fa2 first times the default plan and fixed "
        "KV split sizes of 1,024 to 16,384 tokens below kv_len, then measures "
        "the fastest."
    ),
    caveats=(
        "Every request in a call has the same q_len and kv_len; a batch of "
        "different lengths is priced at its mean.",
        "fa2, fa3 and cudnn read contiguous (ragged) K and V; the vllm_fa4 "
        "backends read a paged fp8 cache (page 16; pages of 16 and 64 measured "
        "the same). cudnn measures one request only.",
        "The fp8 path needs fp8 queries and KV with a non-fp8 output, so bf16 "
        "queries over fp8 KV are not measured.",
        "cudnn measures no fp8.",
    ),
    # No separate PyTorch reference implementation exists for this attention kind.
    reference=None,
)

_BACKEND_DOCS = {
    "fa2": BackendDoc(
        summary="FlashInfer FA2 non-causal ragged attention with a measured KV split search."
    ),
    "fa3": BackendDoc(summary="FlashInfer's non-causal ragged wrapper with FA3 kernels."),
    "cudnn": BackendDoc(
        summary="PyTorch scaled_dot_product_attention pinned to cuDNN for non-causal attention."
    ),
}


_FA4_RUNNER_MODULE = "profiling.runners.attention.fa4_block_rect"
# FA4's SM100 forward is built for sm_100f. The draft attends fp8 queries over an
# fp8 KV cache (kv_cache_scheme fp8; QuantFP8 on the query) with a bf16 output.
_FA4_SUPPORT = BackendSupport(
    compute=frozenset({DType.FP8_E4M3}),
    kv=frozenset({DType.FP8_E4M3}),
    sm_targets=frozenset({"sm_100f"}),
)
_FA4_BACKEND_DOCS = {
    "vllm_fa4_split32": BackendDoc(
        summary=(
            "vLLM fork flash_attn_varlen_func(fa_version=4, causal=False) over an fp8 "
            "paged cache with num_splits=32, the max_num_splits a FULL CUDA graph "
            "passes: split forward plus combine."
        ),
    ),
    "vllm_fa4_nosplit": BackendDoc(
        summary=(
            "The same FA4 call with num_splits=1: the forward kernel alone, output "
            "bit-identical to num_splits=32."
        ),
    ),
}

for _backend, _doc in _FA4_BACKEND_DOCS.items():
    register(
        KernelProfilerSpec(
            kernel_kind=KIND,
            backend=_backend,
            supports=_FA4_SUPPORT,
            runner_ref=RunnerRef(
                module_name=_FA4_RUNNER_MODULE,
                function_name=f"profile_flashinfer_attn_rect_{_backend}",
            ),
            table_name=KIND,
            args_schema=FlashinferAttnRectArgs,
            metric_family=MetricFamily.COMPUTE,
            batch_outlier_policy=BatchOutlierPolicy(),
            subprocess_env="vllm_env",
            doc=_doc,
        )
    )

for _backend in _BACKENDS:
    register(
        KernelProfilerSpec(
            kernel_kind=KIND,
            backend=_backend,
            supports=_SUPPORTS[_backend],
            runner_ref=RunnerRef(
                module_name=_RUNNER_MODULE,
                function_name=f"profile_flashinfer_attn_rect_{_backend}",
            ),
            table_name=KIND,
            args_schema=FlashinferAttnRectArgs,
            metric_family=MetricFamily.COMPUTE,
            batch_outlier_policy=BatchOutlierPolicy(),
            subprocess_env="flashinfer_pip_env",
            doc=_BACKEND_DOCS[_backend],
        )
    )
