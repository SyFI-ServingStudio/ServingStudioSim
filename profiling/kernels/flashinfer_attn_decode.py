"""FlashInfer decode attention kernel kind.

All Python-side per-kernel knowledge for ``flashinfer_attn_decode`` lives here:
the wire string ``KIND``, the ``FlashinferAttnDecodeArgs`` schema, and the
``register(...)`` calls (one per backend) that wire this kernel into
``profiling.db.registry``.

Wire string ``"flashinfer_attn_decode"`` matches Rust ``KernelSpec::KIND`` in
``simulator/src/timing/kernels/flashinfer_attn_decode.rs`` and the facade stem
``get_flashinfer_attn_decode_times`` / ``count_missing_flashinfer_attn_decode``.

Decode is batched single-token-query attention against a paged KV cache (a
different wrapper than prefill/rect — ``BatchDecodeWithPagedKVCacheWrapper``), so
it has its own runner module ``profiling.runners.attention.flashinfer_decode``.

Cache exception (vs prefill/rect): batch is a real sweep axis. Cache identity =
``(batch_size, total_tokens)`` — the TOTAL kv tokens across the batch, so that a
plain rectangular ``total_tokens`` cap keeps every grid corner feasible (the
product, not each axis, bounds memory/profiling cost). The runner derives the
per-request mean length ``avg_len = max(1, total_tokens // batch_size)`` (which
approximates a heterogeneous batch by its mean kv length; error tracked
separately), then sets ``q_len = 1`` per request, ``kv_len = avg_len``, non-causal.

Backends ``{fa2, fa2_cudagraph, fa3, trt, cudnn}``: each registers its own spec
-> the same ``args_schema`` and table, routing to
``profile_flashinfer_attn_decode_<backend>``. ``fa2_cudagraph`` keeps FA2 math
but enables FlashInfer's CUDA-graph plan, matching vLLM pure decode's split main
+ merge launch sequence. ``trt`` is B200/sm100-only and raises
``ProfilerNotImplemented`` here (kept registered for future B200); ``cudnn``
raises for fp8.

Shape split: static Config = ``(num_qo_heads, num_kv_heads, head_dim, q_dtype,
kv_dtype, o_dtype)``; runtime 2D sweep Input = ``(batch_size, total_tokens)``.
Metric family COMPUTE, cache ``Cache2DLinear``.

Importing this module appends ``KernelProfilerSpec`` rows to the registry. The
runner module is referenced lazily via ``RunnerRef``.
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

KIND: str = "flashinfer_attn_decode"

_RUNNER_MODULE = "profiling.runners.attention.flashinfer_decode"
_BACKENDS = ("fa2", "fa2_cudagraph", "fa3", "trt", "cudnn")

# Per-backend capability on compute (q_dtype) and kv-cache (kv_dtype) axes; fa2
# runs bf16-q / fp8-kv. See flashinfer_attn_prefill for the rationale.
_SUPPORTS = {
    "fa2": BackendSupport(
        compute=frozenset({DType.BF16}), kv=frozenset({DType.BF16, DType.FP8_E4M3})
    ),
    # Same compiled FA2 kernel family and dtype support as ``fa2``; only the
    # FlashInfer plan/launch composition differs.
    "fa2_cudagraph": BackendSupport(
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
        gpus=frozenset({"NVIDIA B200"}),  # trtllm-gen kernels are Blackwell-only
    ),
}


@dataclass(frozen=True)
class FlashinferAttnDecodeArgs(KernelArgs):
    # Static attention dims (Rust <Name>KernelConfig, minus `backends`).
    num_qo_heads: int = arg(unit="heads", doc="Query and output heads on this GPU.")
    num_kv_heads: int = arg(
        unit="heads", doc="KV heads on this GPU. Fewer than num_qo_heads means GQA."
    )
    head_dim: int = arg(unit="elements", doc="Dimension of each query, key and value head.")
    q_dtype: DType = arg(doc="Query element type.")
    kv_dtype: DType = arg(doc="KV cache element type.")
    o_dtype: DType = arg(doc="Output element type.")
    # Runtime 2D sweep coords (Rust <Name>KernelInput). total_tokens = total kv
    # across the batch; the runner derives avg_len = max(1, total // batch).
    batch_size: int = arg(unit="requests", doc="Requests decoded together.")
    total_tokens: int = arg(unit="tokens", doc="KV tokens summed over the batch.")


DOC = KernelDoc(
    title="Paged decode attention",
    summary=(
        "One query token per request attending over its paged KV cache, for a whole decode batch."
    ),
    description=(
        "The attention step of decode for MHA and GQA models. Each request brings "
        "one new query token and reads its full KV cache from paged memory. The "
        "measurement uses a uniform batch: every request gets the mean KV length, "
        "total_tokens divided by batch_size, rounded down."
    ),
    category="Attention",
    subcategory="MHA / GQA",
    formula=(
        "O = softmax(q · Kᵀ / √head_dim) · V, per request and query head",
        "kv_len = max(1, ⌊total_tokens / batch_size⌋), q_len = 1",
        "TFLOPS = 4 · batch_size · kv_len · num_qo_heads · head_dim / time",
        "GB/s = bytes of q, K, V and O / time",
    ),
    # Decode reads the whole KV cache per token, so bandwidth is the telling metric.
    default_metric="memory_bandwidth_gbps",
    caveats=(
        "A real batch mixes short and long requests. The uniform-length batch "
        "measured here can differ from it, most for skewed batches.",
        "fa2_cudagraph runs the same FA2 kernels with FlashInfer's CUDA-graph plan, "
        "which splits the KV and adds a merge kernel, as vLLM does in pure decode. "
        "The kernels are launched directly; no captured graph is replayed.",
    ),
    method=(
        f"{CUPTI_METHOD} The paged backends store K and V in 16-token pages (NHD layout), and GB/s "
        "counts every allocated page. FlashInfer runs with use_tensor_cores=True. "
        "fa3 and fa2_cudagraph run once before timing so one-time setup kernels are "
        "left out. With fp8 KV the cache is quantized per head and the kernel gets "
        "one mean scale each for K and V."
    ),
    # No reference implementation exists for this kind.
    reference=None,
)

_DECODE_DOCS_URL = "https://docs.flashinfer.ai/api/decode.html"
_BACKEND_DOCS = {
    "fa2": BackendDoc(
        summary=(
            "FlashInfer BatchDecodeWithPagedKVCacheWrapper with FA2 kernels and the ordinary plan."
        ),
        url=_DECODE_DOCS_URL,
    ),
    "fa2_cudagraph": BackendDoc(
        summary=(
            "FA2 with FlashInfer's CUDA-graph plan: a split-KV main kernel plus a "
            "merge kernel, as in vLLM pure decode."
        ),
        url=_DECODE_DOCS_URL,
    ),
    "fa3": BackendDoc(
        summary="FlashInfer BatchDecodeWithPagedKVCacheWrapper with FA3 kernels.",
        url=_DECODE_DOCS_URL,
    ),
    "trt": BackendDoc(
        summary=(
            "FlashInfer's trtllm-gen kernels, for Blackwell. Registered but not "
            "measured yet: the runner raises on every GPU."
        ),
        url=_DECODE_DOCS_URL,
    ),
    "cudnn": BackendDoc(
        summary=(
            "PyTorch scaled_dot_product_attention pinned to the cuDNN backend, over "
            "contiguous (unpaged) K and V. No fp8."
        ),
        url="https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html",
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
                function_name=f"profile_flashinfer_attn_decode_{_backend}",
            ),
            table_name=KIND,
            args_schema=FlashinferAttnDecodeArgs,
            metric_family=MetricFamily.COMPUTE,
            batch_outlier_policy=BatchOutlierPolicy(),
            subprocess_env="flashinfer_pip_env",
            doc=_BACKEND_DOCS[_backend],
        )
    )
