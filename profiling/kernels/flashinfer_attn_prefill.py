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

Backends ``{fa2, fa3, trt, cudnn}`` are VibeSim backend strings, not kinds: each
registers its own spec -> the same ``args_schema`` and table, routing to its own
``profile_flashinfer_attn_prefill_<backend>`` entry (the worker strips ``backend``
before calling, so the runner is selected by backend, not told it). Unsupported
combos raise ``ProfilerNotImplemented`` at profile time: ``cudnn+fp8``,
``cudnn`` causal with ``prefix_len>0``, and ``trt`` entirely (trtllm-gen has no
ragged kernel and is B200-only — kept registered for a future paged/B200 path).

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
#   trt   : fp8, Blackwell-only (gpus-gated to B200; also runtime-gated on the
#           ragged path, which is not modeled here).
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
        gpus=frozenset({"NVIDIA B200"}),  # trtllm-gen kernels are Blackwell-only
    ),
}


@dataclass(frozen=True)
class FlashinferAttnPrefillArgs(KernelArgs):
    # Static attention dims (Rust <Name>KernelConfig, minus `backends`).
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    q_dtype: DType
    kv_dtype: DType
    o_dtype: DType
    # Runtime 2D sweep coords (Rust <Name>KernelInput).
    prefix_len: int
    append_len: int


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
        )
    )
