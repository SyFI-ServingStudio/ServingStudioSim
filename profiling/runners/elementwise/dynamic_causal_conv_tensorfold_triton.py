"""TensorFold's Triton ``_dconv_kernel`` as the ``dynamic_causal_conv`` backend.

This runner only allocates tensors, times one public call, and returns metrics.
DB writes, subprocess selection and registry routing live in L1b.

It times ``tensorfold.families.qwen3_5.cuda.dflash2._dconv(x, dyn, base, branch,
group_size, residual, seg)`` (perf-study 6469730a), called four times a drafter
layer by ``_layer_fast``: branch 0 on the normed row before attention and before
the MLP, branch 1 with the residual on the attention and MLP outputs.

- ``x`` ``(R, hidden)`` bf16; ``dyn`` the kernel projection's output viewed
  ``(R, 2, 2, hidden / group_size)``; ``base`` ``(2, 2, hidden)`` bf16;
- ``seg`` = 16, the drafting block on GB10 (every stream's block back to back);
- ``residual`` ``(R, hidden)`` bf16 or None.

One ``_dconv_kernel`` launch on grid ``(R, cdiv(hidden, 1024))``, four warps.
"""

from __future__ import annotations

import importlib
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_MODULE = "tensorfold.families.qwen3_5.cuda.dflash2"
_KERNEL_NAME = "_dconv_kernel"
_LABEL = "dynamic_causal_conv:tensorfold_triton"
_SEGMENT = 16
_GUARD_ROWS = 40


def validate_args(
    num_tokens: int, hidden: int, group_size: int, residual: bool, dtype: DType | str
) -> None:
    for name, value in (("num_tokens", num_tokens), ("hidden", hidden), ("group_size", group_size)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if not isinstance(residual, bool):
        raise ValueError(f"residual must be a bool, got {residual!r}")
    if DType.from_value(dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_LABEL} is bf16 only, got {dtype}")
    if hidden % group_size:
        raise ValueError(f"{_LABEL} needs group_size dividing hidden, got {group_size}, {hidden}")


def build_operands(torch: Any, rows: int, hidden: int, group_size: int, residual: bool):
    x = torch.randn(rows, hidden, device="cuda").to(torch.bfloat16)
    dyn = (0.1 * torch.randn(rows, 2, 2, hidden // group_size, device="cuda")).to(torch.bfloat16)
    base = torch.randn(2, 2, hidden, device="cuda").to(torch.bfloat16)
    res = torch.randn(rows, hidden, device="cuda").to(torch.bfloat16) if residual else None
    return x, dyn, base, res


def profile_dynamic_causal_conv_tensorfold_triton(
    num_tokens: int, hidden: int, group_size: int, residual: bool, dtype: DType | str
) -> ComputeMetrics:
    validate_args(num_tokens, hidden, group_size, residual, dtype)
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_LABEL}") from exc
    try:
        dconv = importlib.import_module(_MODULE)._dconv
    except (ImportError, AttributeError) as exc:
        raise ProfilerNotImplemented(f"{_LABEL} needs {_MODULE}._dconv (tensorfold_env)") from exc
    from profiling.runners.elementwise.dynamic_causal_conv_reference import (
        dynamic_causal_conv_reference,
    )

    branch = 1 if residual else 0
    try:
        x, dyn, base, res = build_operands(torch, _GUARD_ROWS, hidden, group_size, residual)
        got = dconv(x, dyn, base, branch, group_size, res, seg=_SEGMENT)
        want = dynamic_causal_conv_reference(torch, x, dyn, base, branch, group_size, _SEGMENT, res)
        err = float((got.float() - want.float()).abs().max() / want.float().abs().max())
        if got.shape != want.shape or not err < 1e-2:
            raise AssertionError(f"{_LABEL} max err {err:.2e} relative to the largest output")
        x, dyn, base, res = build_operands(torch, num_tokens, hidden, group_size, residual)
        torch.cuda.synchronize()

        def kernel() -> Any:
            return dconv(x, dyn, base, branch, group_size, res, seg=_SEGMENT)

        time_ms = Timer.cupti(kernel, warmup=5, kernel_name=_KERNEL_NAME)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_LABEL} ran out of CUDA memory") from exc
    except (ProfilerNotImplemented, ValueError, AssertionError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_LABEL} failed: {exc}") from exc

    nbytes = num_tokens * hidden * 2 * (3 if residual else 2)
    seconds = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(nbytes / seconds / 1e9) if time_ms > 0 else 0.0,
        energy_j=float(energy_j),
    )


__all__ = ["profile_dynamic_causal_conv_tensorfold_triton"]
