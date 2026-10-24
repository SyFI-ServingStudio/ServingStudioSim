"""TensorFold's DFlash2 block attention as ``sliding_block_attention``.

This runner only allocates tensors, times one public call, and returns
metrics. DB writes, subprocess selection and registry routing live in L1b.

It times ``tensorfold.families.qwen3_5.cuda.draft_attention.block_attention(q,
k, v, keys, values, length, window, scale, causal)`` (TensorFold perf-study
6469730a), called the way ``dflash2.py`` ``_layer_fast`` calls it once per
drafter layer:

- ``q`` ``(num_qo_heads, batch_size * block_rows, head_dim)`` and ``k`` / ``v``
  ``(num_kv_heads, batch_size * block_rows, head_dim)`` bf16, each stream's
  block in turn (``_prep``'s outputs);
- ``keys`` / ``values``: one contiguous ``(num_kv_heads, context_len,
  head_dim)`` bf16 context per stream (``add_taps_streams``' outputs);
- ``scale = head_dim ** -0.5``.

The call copies its pointer/length table to the device (pinned host tensor,
non-blocking, a memcpy) and allocates the output, then makes one
``_block_attention`` launch, which CUPTI times by name.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.runners.attention._gdn_common import exact_int
from profiling.runners.attention.sliding_block_attention_reference import (
    sliding_block_attention_reference,
)
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics

_MODULE = "tensorfold.families.qwen3_5.cuda.draft_attention"
_KERNEL_NAME = "_block_attention"
_LABEL = "sliding_block_attention:tensorfold_triton"
# Guard contexts for the oracle (a short, a window-full and a mid stream); the
# window is clipped to 100 so the window mask cuts real keys.
_GUARD_CONTEXTS = (5, 100, 70)
_GUARD_WINDOW = 100
# bf16 output of an fp32-accumulated softmax: a few bf16 ulps of the unit-scale output.
MAX_ABS_TOL = 3e-2


@dataclass(frozen=True)
class _Args:
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    block_rows: int
    window: int
    causal: bool
    batch_size: int
    context_len: int


def validate_args(
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: DType | str,
    block_rows: int,
    window: int,
    causal: bool,
    batch_size: int,
    context_len: int,
) -> _Args:
    """``block_attention``'s contract: bf16, a power-of-two head_dim, a non-empty context within the window."""

    if not isinstance(causal, bool):
        raise ValueError(f"causal must be a bool, got {causal!r}")
    args = _Args(
        num_qo_heads=exact_int("num_qo_heads", num_qo_heads),
        num_kv_heads=exact_int("num_kv_heads", num_kv_heads),
        head_dim=exact_int("head_dim", head_dim),
        block_rows=exact_int("block_rows", block_rows),
        window=exact_int("window", window),
        causal=causal,
        batch_size=exact_int("batch_size", batch_size),
        context_len=exact_int("context_len", context_len),
    )
    if DType.from_value(dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_LABEL} measures bf16 only, got {dtype}")
    if args.head_dim < 16 or args.head_dim & (args.head_dim - 1):
        raise ProfilerNotImplemented(f"{_LABEL} needs a power-of-two head_dim, got {head_dim}")
    if min(args.num_kv_heads, args.block_rows, args.window, args.batch_size) < 1:
        raise ValueError("num_kv_heads, block_rows, window and batch_size must be >= 1")
    if args.num_qo_heads < 1 or args.num_qo_heads % args.num_kv_heads:
        raise ValueError("num_qo_heads must be a positive multiple of num_kv_heads")
    if not 1 <= args.context_len <= args.window:
        # launch_blocks drafts only streams with a context, never past the window.
        raise ValueError(
            f"context_len must lie in [1, window={args.window}], got {args.context_len}"
        )
    return args


def _bf16(torch: Any, gen: Any, *shape: int) -> Any:
    return torch.randn(*shape, generator=gen, device="cuda").to(torch.bfloat16)


def build_operands(
    torch: Any, gen: Any, args: _Args, contexts: tuple[int, ...]
) -> tuple[Any, Any, Any, list[Any], list[Any]]:
    rows = len(contexts) * args.block_rows
    q = _bf16(torch, gen, args.num_qo_heads, rows, args.head_dim)
    k = _bf16(torch, gen, args.num_kv_heads, rows, args.head_dim)
    v = _bf16(torch, gen, args.num_kv_heads, rows, args.head_dim)
    keys = [_bf16(torch, gen, args.num_kv_heads, n, args.head_dim) for n in contexts]
    values = [_bf16(torch, gen, args.num_kv_heads, n, args.head_dim) for n in contexts]
    return q, k, v, keys, values


def check_correctness(torch: Any, block_attention: Any, args: _Args) -> float:
    """One ragged guard call against the fp32 reference; return the max abs error."""

    gen = torch.Generator(device="cuda").manual_seed(1)
    q, k, v, keys, values = build_operands(torch, gen, args, _GUARD_CONTEXTS)
    scale = args.head_dim**-0.5
    out = block_attention(
        q, k, v, keys, values, args.block_rows, _GUARD_WINDOW, scale, args.causal
    )
    torch.cuda.synchronize()
    worst = 0.0
    length = args.block_rows
    for s in range(len(_GUARD_CONTEXTS)):
        rows = slice(s * length, (s + 1) * length)
        want = sliding_block_attention_reference(
            torch,
            q[:, rows],
            k[:, rows],
            v[:, rows],
            keys[s],
            values[s],
            window=_GUARD_WINDOW,
            scale=scale,
            causal=args.causal,
        )
        worst = max(worst, float((out[rows].float() - want).abs().max()))
    if not worst < MAX_ABS_TOL:
        raise AssertionError(f"{_LABEL} max abs err {worst:.2e} vs reference (tol {MAX_ABS_TOL})")
    return worst


def work(args: _Args) -> tuple[int, int]:
    """FLOPs and logical bytes of one call (the kind's DOC formula)."""

    rows = args.batch_size * args.block_rows
    flops = 4 * rows * args.num_qo_heads * args.head_dim * (args.context_len + args.block_rows)
    element = 2
    q_o = 2 * rows * args.num_qo_heads * args.head_dim * element
    block_kv = 2 * rows * args.num_kv_heads * args.head_dim * element
    context_kv = 2 * args.batch_size * args.context_len * args.num_kv_heads * args.head_dim * element
    return flops, q_o + block_kv + context_kv


def profile_sliding_block_attention_tensorfold_triton(
    *,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: DType | str,
    block_rows: int,
    window: int,
    causal: bool,
    batch_size: int,
    context_len: int,
) -> ComputeMetrics:
    args = validate_args(
        num_qo_heads,
        num_kv_heads,
        head_dim,
        dtype,
        block_rows,
        window,
        causal,
        batch_size,
        context_len,
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_LABEL}") from exc
    try:
        block_attention = importlib.import_module(_MODULE).block_attention
    except (ImportError, AttributeError) as exc:
        raise ProfilerNotImplemented(
            f"{_LABEL} needs {_MODULE}.block_attention (tensorfold_env)"
        ) from exc

    from profiling.profilers.energy import Energy
    from profiling.profilers.timer import Timer

    try:
        check_correctness(torch, block_attention, args)
        gen = torch.Generator(device="cuda").manual_seed(0)
        q, k, v, keys, values = build_operands(
            torch, gen, args, (args.context_len,) * args.batch_size
        )
        scale = args.head_dim**-0.5
        torch.cuda.synchronize()

        def kernel() -> Any:
            return block_attention(
                q, k, v, keys, values, args.block_rows, args.window, scale, args.causal
            )

        time_ms = Timer.cupti(kernel, kernel_name=_KERNEL_NAME)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
        flops, bytes_accessed = work(args)
        seconds = time_ms / 1000.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=float(flops / seconds / 1e12) if time_ms > 0 else 0.0,
            memory_bandwidth_gbps=float(bytes_accessed / seconds / 1e9) if time_ms > 0 else 0.0,
            energy_j=float(energy_j),
        )
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_LABEL} ran out of CUDA memory") from exc
    except (ProfilerNotImplemented, ValueError, TypeError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_LABEL} failed: {exc}") from exc
