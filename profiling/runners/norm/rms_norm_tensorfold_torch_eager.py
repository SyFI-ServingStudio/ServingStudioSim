"""TensorFold's eager fp32 RMSNorm (DFlash2 ``_norm``) as an ``rms_norm`` backend.

This runner only allocates tensors, times one public call, and returns metrics.
DB writes, subprocess selection and registry routing live in L1b.

It times ``tensorfold.families.qwen3_5.cuda.dflash2._norm(x, weight, eps)``
(perf-study 6469730a), the norm the DFlash2 drafter applies to its ``fc``
projection of the target taps every time it takes new context rows
(``add_taps`` in prefill, ``add_taps_streams`` in the commit):
``xf = x.float(); bf16(xf * rsqrt(mean(xf * xf) + eps) * weight.float())`` on
``(m, hidden)`` bf16 rows. Eager Torch runs it as nine launches (two casts,
the square, the mean reduction, the eps add, rsqrt, two broadcast multiplies and
the bf16 cast); CUPTI counts every launch (``kernel_name=None``).
"""

from __future__ import annotations

import importlib
from typing import Any

from profiling.db.args import DType
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics
from profiling.runners.norm.rms_norm_tensorfold_torch import MAX_REL_TOL, reference, validate_args

_LABEL = "rms_norm:tensorfold_torch_eager"
_MODULE = "tensorfold.families.qwen3_5.cuda.dflash2"
# The drafter's rms_norm_eps (Qwen3.8-27B-DFlash2 config).
_EPS = 1e-6
_GUARD_ROWS = 37


def profile_rms_norm_tensorfold_torch_eager(
    m: int, hidden: int, dtype: DType | str
) -> ComputeMetrics:
    validate_args(m, hidden, dtype)
    m, hidden = int(m), int(hidden)
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_LABEL}") from exc
    try:
        norm = importlib.import_module(_MODULE)._norm
    except (ImportError, AttributeError) as exc:
        raise ProfilerNotImplemented(f"{_LABEL} needs {_MODULE}._norm (tensorfold_env)") from exc

    from profiling.profilers.energy import Energy
    from profiling.profilers.timer import Timer

    try:
        w = (1.0 + 0.1 * torch.randn(hidden, device="cuda")).to(torch.bfloat16)
        guard = torch.randn(_GUARD_ROWS, hidden, device="cuda").to(torch.bfloat16)
        got, want = norm(guard, w, _EPS), reference(torch, guard, w, _EPS)
        err = float(((got.float() - want.float()).abs() / want.float().abs().clamp_min(1e-2)).max())
        if not err < MAX_REL_TOL:
            raise AssertionError(f"{_LABEL} max rel err {err:.2e} (tol {MAX_REL_TOL:.0e})")
        x = torch.randn(m, hidden, device="cuda").to(torch.bfloat16)
        torch.cuda.synchronize()

        def kernel() -> Any:
            return norm(x, w, _EPS)

        time_ms = Timer.cupti(kernel, warmup=5)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
        # Same logical counts as the other rms_norm rows (x read, y written, weight read).
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
    except (ProfilerNotImplemented, ValueError, AssertionError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_LABEL} failed: {exc}") from exc


__all__ = ["profile_rms_norm_tensorfold_torch_eager"]
