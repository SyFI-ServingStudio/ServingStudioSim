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
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    q_dtype: DType
    kv_dtype: DType
    o_dtype: DType
    # Runtime 2D sweep coords (Rust <Name>KernelInput). total_tokens = total kv
    # across the batch; the runner derives avg_len = max(1, total // batch).
    batch_size: int
    total_tokens: int


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
        )
    )
