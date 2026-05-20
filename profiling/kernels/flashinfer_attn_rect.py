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

Backends ``{fa2, fa3, trt, cudnn}`` are MLSim backend strings, not kinds: each
registers its own spec -> the same ``args_schema`` and table, routing to its own
``profile_flashinfer_attn_rect_<backend>`` entry. Unsupported combos raise
``ProfilerNotImplemented`` at profile time: ``cudnn+fp8``, and ``trt`` entirely
(trtllm-gen has no ragged kernel and is B200-only — kept registered for a future
paged/B200 path).

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
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND: str = "flashinfer_attn_rect"

_RUNNER_MODULE = "profiling.runners.attention.flashinfer_attn_prefill_and_rect"
_BACKENDS = ("fa2", "fa3", "trt", "cudnn")


@dataclass(frozen=True)
class FlashinferAttnRectArgs(KernelArgs):
    # Static attention dims (Rust <Name>KernelConfig, minus `backends`).
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    q_dtype: DType
    kv_dtype: DType
    o_dtype: DType
    # Runtime 2D sweep coords (Rust <Name>KernelInput). Direct, not prefix/append.
    q_len: int
    kv_len: int


for _backend in _BACKENDS:
    register(
        KernelProfilerSpec(
            kernel_kind=KIND,
            backend=_backend,
            runner_ref=RunnerRef(
                module_name=_RUNNER_MODULE,
                function_name=f"profile_flashinfer_attn_rect_{_backend}",
            ),
            table_name=KIND,
            args_schema=FlashinferAttnRectArgs,
            metric_family=MetricFamily.COMPUTE,
            batch_outlier_policy=BatchOutlierPolicy(),
            subprocess_env="flashinfer_pip_env",
        )
    )
