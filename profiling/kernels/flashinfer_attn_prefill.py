"""FlashInfer prefill (causal) attention kernel kind.

All Python-side per-kernel knowledge for ``flashinfer_attn_prefill`` lives here:
the wire string ``KIND``, the ``FlashinferAttnPrefillArgs`` schema, and the
``register(...)`` calls (one per backend) that wire this kernel into
``profiling.db.registry``.

Wire string ``"flashinfer_attn_prefill"`` matches Rust ``KernelSpec::KIND`` in
``simulator/src/timing/kernels/flashinfer_attn_prefill.rs`` and the facade stem
``get_flashinfer_attn_prefill_times`` / ``count_missing_flashinfer_attn_prefill``.

Prefill is causal ragged attention; it *merges* the ref's pure-prefill + chunked
ops (pure prefill = ``prefix_len == 0``). The novel bit is that **cache dims are
not the kernel params**: the cache identity is ``(prefix_len, append_len)`` and
the runner derives ``q_len = append_len``, ``kv_len = prefix_len + append_len``
(that mapping lives in ``profiling.runners.attention.flashinfer_attn_prefill_and_rect``).

Backends ``{fa2, fa3, trt, cudnn}`` are ServingStudioSim backend strings, not kinds: each
registers its own spec -> the same ``args_schema`` and table, routing to its own
``profile_flashinfer_attn_prefill_<backend>`` entry (the worker strips ``backend``
before calling, so the runner is selected by backend, not told it). Unsupported
combos raise ``ProfilerNotImplemented`` at profile time: ``cudnn+fp8``,
``cudnn`` causal with ``prefix_len>0``, and ``trt`` entirely (trtllm-gen has no
ragged kernel and is SM10x-only — kept registered for a future paged SM10x path).

Shape split: static Config = ``(num_qo_heads, num_kv_heads, head_dim, q_dtype,
kv_dtype, o_dtype)``; runtime 2D sweep Input = ``(prefix_len, append_len)``.
Metric family COMPUTE, cache ``Cache2DLinear`` (two monotonic axes).

Importing this module appends ``KernelProfilerSpec`` rows to the registry. The
runner module is referenced lazily via ``RunnerRef`` so the main process never
eager-imports flashinfer/torch/cuda.
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

KIND: str = "flashinfer_attn_prefill"

_RUNNER_MODULE = "profiling.runners.attention.flashinfer_attn_prefill_and_rect"
_BACKENDS = ("fa2", "fa3", "trt", "cudnn")

# Per-backend capability on TWO axes — compute (q_dtype) and kv-cache (kv_dtype):
#   fa2   : bf16 compute, but fp8 KV is fine (bf16-q / fp8-kv is a prod config).
#   fa3   : fp8 compute + fp8 KV.
#   cudnn : bf16 only on both axes (no fp8 at all).
#   trt   : fp8, trtllm-gen kernels for the SM10x family only (sm_100f; also
#           runtime-gated on the ragged path, which is not modeled here).
_SUPPORTS = {
    "fa2": BackendSupport(
        compute=frozenset({DType.BF16}), kv=frozenset({DType.BF16, DType.FP8_E4M3})
    ),
    "fa3": BackendSupport(
        compute=frozenset({DType.BF16, DType.FP8_E4M3}),
        kv=frozenset({DType.BF16, DType.FP8_E4M3}),
    ),
    "cudnn": BackendSupport(
        compute=frozenset({DType.BF16}), kv=frozenset({DType.BF16})
    ),
    "trt": BackendSupport(
        compute=frozenset({DType.FP8_E4M3}),
        kv=frozenset({DType.FP8_E4M3}),
        sm_targets=frozenset({"sm_100f"}),
    ),
}


@dataclass(frozen=True)
class FlashinferAttnPrefillArgs(KernelArgs):
    # Static attention dims (Rust <Name>KernelConfig, minus `backends`).
    num_qo_heads: int = arg(unit="heads", doc="Query and output heads on this GPU.")
    num_kv_heads: int = arg(unit="heads", doc="KV heads on this GPU.")
    head_dim: int = arg(unit="elements", doc="Dimension of each query, key and value head.")
    q_dtype: DType = arg(doc="Query element type.")
    kv_dtype: DType = arg(doc="Key and value element type.")
    o_dtype: DType = arg(doc="Output element type.")
    # Runtime 2D sweep coords (Rust <Name>KernelInput).
    prefix_len: int = arg(unit="tokens", doc="Cached tokens preceding the new query block.")
    append_len: int = arg(unit="tokens", doc="New query and KV tokens in this append.")


DOC = KernelDoc(
    title="Causal prefill attention",
    summary="Attend a block of new query tokens over its preceding context and itself.",
    description=(
        "The attention step of prefill and chunked prefill for MHA and GQA "
        "models. A request brings append_len new tokens; each attends causally "
        "to the prefix_len tokens already cached and to the new tokens before "
        "it. The measurement is one request with contiguous K and V, so q_len ="
        " append_len and kv_len = prefix_len + append_len."
    ),
    category="Attention",
    subcategory="MHA / GQA",
    formula=(
        "O = softmax(Q · Kᵀ / √head_dim, causal mask) · V, per query head",
        "q_len = append_len, kv_len = prefix_len + append_len",
        "TFLOPS = 2 · append_len · (2·prefix_len + append_len) · num_qo_heads ·"
        " head_dim / time",
        "GB/s = bytes of Q, K, V and O / time",
    ),
    default_metric="tflops",
    method=(
        f"{CUPTI_METHOD} "
        "Planning and input construction are not timed; every launch of the "
        "attention call is counted. fa2 first times the default plan and fixed "
        "KV split sizes of 1,024 to 16,384 tokens below kv_len, then measures "
        "the fastest."
    ),
    caveats=(
        "One request with contiguous K and V is measured, not a batch of "
        "requests of different lengths or a paged cache.",
        "The fp8 path needs fp8 queries and KV with a non-fp8 output, so bf16 "
        "queries over an fp8 cache are not measured.",
        "cudnn measures only prefix_len = 0 and no fp8; trt has no ragged "
        "kernel and records no rows.",
    ),
    # No separate PyTorch reference implementation exists for this attention kind.
    reference=None,
)

_BACKEND_DOCS = {
    "fa2": BackendDoc(summary="FlashInfer FA2 ragged prefill with a measured KV split search."),
    "fa3": BackendDoc(summary="FlashInfer's ragged prefill wrapper with FA3 kernels."),
    "trt": BackendDoc(
        summary="FlashInfer trtllm-gen is registered but has no ragged prefill kernel to measure."
    ),
    "cudnn": BackendDoc(
        summary="PyTorch scaled_dot_product_attention pinned to cuDNN for fresh causal prefill."
    ),
}


for _backend in _BACKENDS:
    register(
        KernelProfilerSpec(
            kernel_kind=KIND,
            backend=_backend,
            supports=_SUPPORTS[_backend],
            runner_ref=RunnerRef(
                module_name=_RUNNER_MODULE,
                function_name=f"profile_flashinfer_attn_prefill_{_backend}",
            ),
            table_name=KIND,
            args_schema=FlashinferAttnPrefillArgs,
            metric_family=MetricFamily.COMPUTE,
            batch_outlier_policy=BatchOutlierPolicy(),
            subprocess_env="flashinfer_pip_env",
            doc=_BACKEND_DOCS[_backend],
        )
    )
