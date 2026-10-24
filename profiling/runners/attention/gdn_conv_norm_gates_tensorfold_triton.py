"""TensorFold's Triton ``_gdn_pre`` as the ``gdn_conv_norm_gates`` backend.

This runner only allocates tensors, times one public call, and returns metrics.
DB writes, subprocess selection and registry routing live in L1b.

It times ``tensorfold.families.qwen3_5.cuda.glue.gdn_pre(qkv, conv_state,
conv_w, windows, a, b, A_log, dt_bias, kh=, vh=, dk=, stream_ids=, nkeep=)``
(TensorFold perf-study 6469730a), the call every GDN layer makes after its
input projections:

- ``forward.py`` ``multi_tree_forward`` (every verify round): ``windows``
  ``(W, kernel_size)`` holds each row's draft-tree ancestors, ``stream_ids``
  each row's stream, ``conv_state`` the streams' states concatenated;
- ``prefill.py`` ``prefill_rows`` (several prompts in one prefill step): the
  same MULTI call with chain windows;
- ``prefill.py`` ``prefill_chunk`` (one prompt's chunk): no ``stream_ids``,
  the kernel's ``MULTI=False`` specialization.

The kernel is one launch on grid ``(W, 2·kh + vh)``, a 2-warp program per row
and head, whatever the mode. The runner times the MULTI call with one stream
and chain windows (row ``t`` reads rows ``t − keep .. t``, the first ``keep``
from the state). Measured on GB10 (CUDA-event diagnostics, 3 repeats):

- 1, 16 or 64 streams over the same rows are within repeat noise at 256-4096
  rows (no trend), so the kind has no stream count;
- the single-stream ``MULTI=False`` specialization runs within about ±15% of
  the MULTI call at 1024-4096 rows, with the sign changing between sizes, and
  within 3% at up to 256 rows; the batch capture shows the same 1024-row launch
  at 196 us in ``prefill_chunk`` and 188 us in ``prefill_rows``, so the kind has
  no mode field either.

Allocation, window construction, Triton's JIT compile and the Torch oracle run
before timing; CUPTI counts only ``_gdn_pre``.
"""

from __future__ import annotations

import importlib
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention._gdn_common import exact_int
from profiling.runners.attention.gdn_conv_norm_gates_reference import (
    gdn_conv_norm_gates_reference,
)
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics

_MODULE = "tensorfold.families.qwen3_5.cuda.glue"
_CALLABLE = "gdn_pre"
_KERNEL_NAME = "_gdn_pre"
_LABEL = "gdn_conv_norm_gates:tensorfold_triton"
# The oracle checks the first and last rows (the first ``keep`` read the state).
_CHECK_EDGE_ROWS = 128
# q, k, v are bf16 (2^-8 relative rounding, plus rows whose fp32 sum order
# rounds the other way); g and beta are fp32 transcendental results.
BF16_REL_FROBENIUS_TOL = 1e-2
FP32_MAX_REL_TOL = 1e-4


def validate_args(
    num_tokens: int,
    num_qk_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    kernel_size: int,
    dtype: DType | str,
) -> None:
    """The kernel's contract: bf16, one power-of-two head width for q, k and v, a filter of 2+ taps."""
    if DType.from_value(dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_LABEL} is bf16 only; got dtype={dtype}")
    for name, value in (
        ("num_tokens", num_tokens),
        ("num_qk_heads", num_qk_heads),
        ("num_value_heads", num_value_heads),
        ("key_head_dim", key_head_dim),
        ("value_head_dim", value_head_dim),
    ):
        if value < 1:
            raise ValueError(f"{_LABEL} needs {name} >= 1, got {value}")
    if kernel_size < 2:
        raise ValueError(f"{_LABEL} needs kernel_size >= 2 (a state of kernel_size - 1 rows), got {kernel_size}")
    if key_head_dim != value_head_dim or key_head_dim & (key_head_dim - 1):
        # _gdn_pre takes one DK for every head (tl.arange(0, DK)): a power of two.
        raise ProfilerNotImplemented(
            f"{_LABEL} needs key_head_dim == value_head_dim, a power of two; "
            f"got {key_head_dim}, {value_head_dim}"
        )


def build_operands(
    torch: Any,
    num_tokens: int,
    num_qk_heads: int,
    num_value_heads: int,
    head_dim: int,
    kernel_size: int,
) -> dict[str, Any]:
    """One stream's chain: row t's window is [state; qkv] rows t .. t + keep."""
    keep = kernel_size - 1
    width = (2 * num_qk_heads + num_value_heads) * head_dim
    gen = torch.Generator(device="cuda").manual_seed(0)

    def randn(*shape: int, scale: float = 1.0, dtype: Any = torch.bfloat16) -> Any:
        return torch.randn(*shape, device="cuda", generator=gen).mul_(scale).to(dtype)

    windows = (
        torch.arange(num_tokens, device="cuda", dtype=torch.int32)[:, None]
        + torch.arange(kernel_size, device="cuda", dtype=torch.int32)[None, :]
    ).contiguous()
    return {
        "qkv": randn(num_tokens, width),
        "conv_state": randn(keep, width),
        # A trained depthwise filter's taps are O(0.1-1); the scale only keeps SiLU
        # off its saturated tails.
        "conv_w": randn(width, kernel_size, scale=0.5),
        "windows": windows,
        "stream_ids": torch.zeros(num_tokens, device="cuda", dtype=torch.int32),
        "a": randn(num_tokens, num_value_heads),
        "b": randn(num_tokens, num_value_heads),
        "A_log": randn(num_value_heads, scale=0.5, dtype=torch.float32),
        "dt_bias": randn(num_value_heads, scale=0.5, dtype=torch.float32),
    }


def check_correctness(
    torch: Any, ops: dict[str, Any], out: tuple[Any, ...], num_qk_heads: int, num_value_heads: int, head_dim: int
) -> dict[str, float]:
    """Compare the call's outputs on edge rows with the Torch reference; return the errors."""
    rows_total = ops["windows"].shape[0]
    idx = torch.unique(
        torch.cat(
            [
                torch.arange(min(rows_total, _CHECK_EDGE_ROWS), device="cuda"),
                torch.arange(max(0, rows_total - _CHECK_EDGE_ROWS), rows_total, device="cuda"),
            ]
        )
    )
    want = gdn_conv_norm_gates_reference(
        ops["qkv"],
        ops["conv_state"],
        ops["conv_w"],
        ops["windows"][idx],
        ops["stream_ids"][idx],
        ops["a"][idx],
        ops["b"][idx],
        ops["A_log"],
        ops["dt_bias"],
        num_qk_heads=num_qk_heads,
        num_value_heads=num_value_heads,
        key_head_dim=head_dim,
        value_head_dim=head_dim,
    )
    errors: dict[str, float] = {}
    for name, got, ref in zip(("q", "k", "v", "g", "beta"), out, want):
        got = got[idx]
        if got.shape != ref.shape or got.dtype != ref.dtype:
            raise AssertionError(f"{_LABEL} {name}: got {tuple(got.shape)} {got.dtype}, want {tuple(ref.shape)} {ref.dtype}")
        if ref.dtype == torch.bfloat16:
            err = float((got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-30))
            tol = BF16_REL_FROBENIUS_TOL
        else:
            err = float(((got - ref).abs() / ref.abs().clamp_min(1e-6)).max())
            tol = FP32_MAX_REL_TOL
        if not err <= tol:
            raise AssertionError(f"{_LABEL} {name} error {err:.2e} > {tol:.0e}")
        errors[name] = err
    return errors


def semantic_flops(
    num_tokens: int, num_qk_heads: int, num_value_heads: int, key_head_dim: int, value_head_dim: int, kernel_size: int
) -> int:
    width = 2 * num_qk_heads * key_head_dim + num_value_heads * value_head_dim
    return num_tokens * (2 * kernel_size * width + 4 * width)


def logical_bytes(
    num_tokens: int, num_qk_heads: int, num_value_heads: int, key_head_dim: int, value_head_dim: int, kernel_size: int
) -> int:
    """bf16 window reads (kernel_size rows a token), q/k/v and raw-gate traffic, fp32 g/beta, the filter."""
    width = 2 * num_qk_heads * key_head_dim + num_value_heads * value_head_dim
    bf16 = num_tokens * (kernel_size * width + width + 2 * num_value_heads) + width * kernel_size
    return 2 * bf16 + 4 * 2 * num_tokens * num_value_heads


def profile_gdn_conv_norm_gates_tensorfold_triton(
    num_tokens: int,
    num_qk_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    kernel_size: int,
    dtype: DType | str,
) -> ComputeMetrics:
    num_tokens = exact_int("num_tokens", num_tokens)
    num_qk_heads = exact_int("num_qk_heads", num_qk_heads)
    num_value_heads = exact_int("num_value_heads", num_value_heads)
    key_head_dim = exact_int("key_head_dim", key_head_dim)
    value_head_dim = exact_int("value_head_dim", value_head_dim)
    kernel_size = exact_int("kernel_size", kernel_size)
    validate_args(num_tokens, num_qk_heads, num_value_heads, key_head_dim, value_head_dim, kernel_size, dtype)
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_LABEL}") from exc
    try:
        gdn_pre = getattr(importlib.import_module(_MODULE), _CALLABLE)
    except (ImportError, AttributeError) as exc:
        raise ProfilerNotImplemented(f"{_LABEL} needs {_MODULE}.{_CALLABLE} (tensorfold_env)") from exc

    try:
        ops = build_operands(torch, num_tokens, num_qk_heads, num_value_heads, key_head_dim, kernel_size)

        def kernel() -> Any:
            return gdn_pre(
                ops["qkv"],
                ops["conv_state"],
                ops["conv_w"],
                ops["windows"],
                ops["a"],
                ops["b"],
                ops["A_log"],
                ops["dt_bias"],
                kh=num_qk_heads,
                vh=num_value_heads,
                dk=key_head_dim,
                stream_ids=ops["stream_ids"],
                nkeep=kernel_size - 1,
            )

        out = kernel()                     # compiles the Triton kernel
        torch.cuda.synchronize()
        check_correctness(torch, ops, out, num_qk_heads, num_value_heads, key_head_dim)
        del out

        time_ms = Timer.cupti(kernel, kernel_name=_KERNEL_NAME)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_LABEL} ran out of CUDA memory") from exc
    except (ProfilerNotImplemented, ValueError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_LABEL} failed: {exc}") from exc

    dims = (num_tokens, num_qk_heads, num_value_heads, key_head_dim, value_head_dim, kernel_size)
    seconds = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=float(semantic_flops(*dims) / seconds / 1e12) if time_ms > 0 else 0.0,
        memory_bandwidth_gbps=float(logical_bytes(*dims) / seconds / 1e9) if time_ms > 0 else 0.0,
        energy_j=float(energy_j),
    )
