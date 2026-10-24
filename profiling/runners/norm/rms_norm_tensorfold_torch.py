"""TensorFold's Torch RMSNorm as an ``rms_norm`` backend.

This runner only allocates tensors, times one public call, and returns metrics.
DB writes, subprocess selection and registry routing live in L1b.

It times ``torch.nn.functional.rms_norm(x, (hidden,), w, eps)`` in TensorFold's
own Torch (2.14.1+cu130), the call the DFlash2 drafter makes for every layer's
input and post-attention norm and for its final norm
(``families/qwen3_5/cuda/dflash2.py`` ``_layer_fast`` / ``launch_blocks``):
``x`` ``(m, hidden)`` bf16 contiguous, ``w`` ``(hidden,)`` bf16. Torch runs it as
one ``vectorized_layer_norm_kernel`` launch; CUPTI counts every launch of the
call (``kernel_name=None``), so a dtype cast Torch might add would be counted.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics

_LABEL = "rms_norm:tensorfold_torch"
# The drafter's rms_norm_eps (Qwen3.8-27B-DFlash2 config).
_EPS = 1e-6
_GUARD_ROWS = 37
# Torch computes in fp32 and rounds once to bf16; the oracle rounds the same way
# but may order the sum differently: a bf16 ulp.
MAX_REL_TOL = 1e-2


def validate_args(m: int, hidden: int, dtype: DType | str) -> None:
    if DType.from_value(dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_LABEL} measures the drafter's bf16 rows only, got {dtype}")
    if isinstance(m, bool) or isinstance(hidden, bool) or m <= 0 or hidden <= 0:
        raise ValueError(f"{_LABEL} needs m and hidden > 0, got m={m}, hidden={hidden}")


def reference(torch: Any, x: Any, w: Any, eps: float) -> Any:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps) * w.float()).to(
        torch.bfloat16
    )


def check_correctness(torch: Any, hidden: int) -> float:
    x = torch.randn(_GUARD_ROWS, hidden, device="cuda").to(torch.bfloat16)
    w = (1.0 + 0.1 * torch.randn(hidden, device="cuda")).to(torch.bfloat16)
    got = torch.nn.functional.rms_norm(x, (hidden,), w, _EPS)
    want = reference(torch, x, w, _EPS)
    err = float(((got.float() - want.float()).abs() / want.float().abs().clamp_min(1e-2)).max())
    if not err < MAX_REL_TOL:
        raise AssertionError(f"{_LABEL} max rel err {err:.2e} (tol {MAX_REL_TOL:.0e})")
    return err


def profile_rms_norm_tensorfold_torch(m: int, hidden: int, dtype: DType | str) -> ComputeMetrics:
    validate_args(m, hidden, dtype)
    m, hidden = int(m), int(hidden)
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_LABEL}") from exc

    from profiling.profilers.energy import Energy
    from profiling.profilers.timer import Timer

    try:
        check_correctness(torch, hidden)
        x = torch.randn(m, hidden, device="cuda").to(torch.bfloat16)
        w = (1.0 + 0.1 * torch.randn(hidden, device="cuda")).to(torch.bfloat16)
        torch.cuda.synchronize()

        def kernel() -> Any:
            return torch.nn.functional.rms_norm(x, (hidden,), w, _EPS)

        time_ms = Timer.cupti(kernel)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
        # vllm_cuda's counts: x read, y written, weight read; 4 ops an element + 2 a row.
        bytes_accessed = (2 * m * hidden + hidden) * DType.BF16.size_bytes()
        flops = 4 * m * hidden + 2 * m
        seconds = time_ms / 1000.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=float(flops / seconds / 1e12) if time_ms > 0 else 0.0,
            memory_bandwidth_gbps=float(bytes_accessed / seconds / 1e9) if time_ms > 0 else 0.0,
            energy_j=float(energy_j),
        )
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_LABEL} ran out of CUDA memory") from exc
    except (ProfilerNotImplemented, ValueError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_LABEL} failed: {exc}") from exc
