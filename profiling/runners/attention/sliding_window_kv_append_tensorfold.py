"""TensorFold's DFlash2 context append as ``sliding_window_kv_append``.

This runner only allocates tensors, times one public call, and returns
metrics. DB writes, subprocess selection and registry routing live in L1b.

It times ``tensorfold.families.qwen3_5.cuda.draft_attention.append(new, olds,
sizes, window)`` (TensorFold perf-study 6469730a), called the way
``dflash2.py`` ``add_taps_streams`` calls it once per drafter layer for keys and
once for values:

- ``new`` is ``(num_kv_heads, batch_size * num_tokens, head_dim)`` bf16, each
  stream's new rows in turn (``_prep``'s k or v output);
- ``olds`` holds each stream's contiguous ``(num_kv_heads, context_len,
  head_dim)`` context, or None when the stream has none yet;
- the call allocates every stream's output, copies its pointer/size table to
  the device (pinned host tensor, non-blocking) and makes one ``_append``
  launch.

CUPTI times the ``_append`` launch by name (the table copy is a memcpy, not a
kernel). Every stream gets the same ``context_len`` and ``num_tokens``.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.runners.attention._gdn_common import exact_int
from profiling.runners.attention.sliding_window_kv_append_reference import (
    sliding_window_kv_append_reference,
)
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics

_MODULE = "tensorfold.families.qwen3_5.cuda.draft_attention"
_KERNEL_NAME = "_append"
_LABEL = "sliding_window_kv_append:tensorfold_triton"
# Guard batch for the bit-exact check: a stream with no context, one whose
# window overflows, and one that stays under the window (window 40).
_GUARD_WINDOW = 40
_GUARD = ((0, 7), (35, 9), (12, 5))


@dataclass(frozen=True)
class _Args:
    num_kv_heads: int
    head_dim: int
    window: int
    batch_size: int
    context_len: int
    num_tokens: int

    @property
    def keep(self) -> int:
        return min(self.window, self.context_len + self.num_tokens)


def validate_args(
    num_kv_heads: int,
    head_dim: int,
    dtype: DType | str,
    window: int,
    batch_size: int,
    context_len: int,
    num_tokens: int,
) -> _Args:
    """``append``'s contract: bf16 rows, a power-of-two head_dim, a context within the window."""

    args = _Args(
        num_kv_heads=exact_int("num_kv_heads", num_kv_heads),
        head_dim=exact_int("head_dim", head_dim),
        window=exact_int("window", window),
        batch_size=exact_int("batch_size", batch_size),
        context_len=exact_int("context_len", context_len),
        num_tokens=exact_int("num_tokens", num_tokens),
    )
    if DType.from_value(dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_LABEL} measures bf16 rows only, got {dtype}")
    if args.head_dim < 16 or args.head_dim & (args.head_dim - 1):
        # _append's tl.arange(0, D) needs a power of two.
        raise ProfilerNotImplemented(f"{_LABEL} needs a power-of-two head_dim, got {head_dim}")
    if min(args.num_kv_heads, args.window, args.batch_size, args.num_tokens) < 1:
        raise ValueError("num_kv_heads, window, batch_size and num_tokens must be >= 1")
    if not 0 <= args.context_len <= args.window:
        # A context is always the result of an earlier append: never past the window.
        raise ValueError(
            f"context_len must lie in [0, window={args.window}], got {args.context_len}"
        )
    return args


def check_correctness(torch: Any, append: Any, *, num_kv_heads: int, head_dim: int) -> None:
    """One ragged guard call compared bit for bit with the concatenate-and-slice reference."""

    gen = torch.Generator(device="cuda").manual_seed(1)
    new = torch.randn(
        num_kv_heads, sum(a for _, a in _GUARD), head_dim, generator=gen, device="cuda"
    ).to(torch.bfloat16)
    olds = [
        None
        if n == 0
        else torch.randn(num_kv_heads, n, head_dim, generator=gen, device="cuda").to(
            torch.bfloat16
        )
        for n, _ in _GUARD
    ]
    outs = append(new, olds, [a for _, a in _GUARD], _GUARD_WINDOW)
    torch.cuda.synchronize()
    first = 0
    for (_, add), old, got in zip(_GUARD, olds, outs):
        want = sliding_window_kv_append_reference(
            torch, old, new[:, first : first + add], _GUARD_WINDOW
        )
        first += add
        if got.shape != want.shape or not torch.equal(got, want):
            raise AssertionError(
                f"{_LABEL}: stream output {tuple(got.shape)} differs from the reference "
                f"{tuple(want.shape)}"
            )


def build_operands(torch: Any, args: _Args) -> tuple[Any, list[Any]]:
    gen = torch.Generator(device="cuda").manual_seed(0)
    new = torch.randn(
        args.num_kv_heads,
        args.batch_size * args.num_tokens,
        args.head_dim,
        generator=gen,
        device="cuda",
    ).to(torch.bfloat16)
    olds = [
        None
        if args.context_len == 0
        else torch.randn(
            args.num_kv_heads, args.context_len, args.head_dim, generator=gen, device="cuda"
        ).to(torch.bfloat16)
        for _ in range(args.batch_size)
    ]
    return new, olds


def logical_bytes(args: _Args) -> int:
    """Each kept row read once and written once, every stream."""

    return 2 * args.batch_size * args.keep * args.num_kv_heads * args.head_dim * 2


def profile_sliding_window_kv_append_tensorfold_triton(
    *,
    num_kv_heads: int,
    head_dim: int,
    dtype: DType | str,
    window: int,
    batch_size: int,
    context_len: int,
    num_tokens: int,
) -> ComputeMetrics:
    args = validate_args(
        num_kv_heads, head_dim, dtype, window, batch_size, context_len, num_tokens
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_LABEL}") from exc
    try:
        append = importlib.import_module(_MODULE).append
    except (ImportError, AttributeError) as exc:
        raise ProfilerNotImplemented(f"{_LABEL} needs {_MODULE}.append (tensorfold_env)") from exc

    from profiling.profilers.energy import Energy
    from profiling.profilers.timer import Timer

    try:
        check_correctness(torch, append, num_kv_heads=args.num_kv_heads, head_dim=args.head_dim)
        new, olds = build_operands(torch, args)
        sizes = [args.num_tokens] * args.batch_size
        torch.cuda.synchronize()

        def kernel() -> Any:
            return append(new, olds, sizes, args.window)

        time_ms = Timer.cupti(kernel, kernel_name=_KERNEL_NAME)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
        seconds = time_ms / 1000.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=0.0,
            memory_bandwidth_gbps=float(logical_bytes(args) / seconds / 1e9) if time_ms > 0 else 0.0,
            energy_j=float(energy_j),
        )
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_LABEL} ran out of CUDA memory") from exc
    except (ProfilerNotImplemented, ValueError, TypeError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_LABEL} failed: {exc}") from exc
