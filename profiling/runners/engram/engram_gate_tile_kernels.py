"""DeepInfer's Engram gate -- the ``tile_kernels`` backend of ``engram_gate``.

The timed call is the one ``Engram.forward`` issues
(``deepinfer/model_executor/models/v4_modules/engram.py``):
``tile_kernels.engram.engram_gate_fwd(hidden_states, kv, weight_fused, eps,
clamp_value, save_for_backward=False, image_token_mask=None, inplace=True)``,
one ``engram_gate_fwd_kernel_cuda_kernel`` launch. ``weight_fused`` comes from
``tile_kernels.engram.fused_weight`` once, outside timing, as the module's
constructor does. In place: every launch rewrites the same streams, which only
changes their values, not the work.

Check: one launch against an fp32 Torch transcription of the gate on a copy of
the streams (see :func:`reference`).
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

BACKEND = "tile_kernels"
# Numerical constants of the module; the kernel's work does not depend on them.
EPS = 1e-6
CLAMP_VALUE = 1e-6
CHECK_TOKENS = 256
TOL = 3e-2


def _validate(
    num_tokens: int, hc_mult: int, hidden_size: int, dtype: DType | str
) -> tuple[int, int, int]:
    if DType.from_value(dtype) is not DType.BF16:
        raise ValueError(f"{BACKEND} engram_gate requires dtype=bf16, got {dtype!r}")
    values = int(num_tokens), int(hc_mult), int(hidden_size)
    if min(values) < 1:
        raise ValueError(f"{BACKEND} engram_gate needs positive sizes, got {values}")
    return values


def reference(
    torch: Any, x: Any, kv: Any, weight_fused: Any, eps: float, clamp_value: float
) -> Any:
    """fp32 Engram gate: x + gate * v per stream (TileKernels' engram_gate_fwd math)."""
    hidden_size = x.shape[-1]
    xf, kf, vf = x.float(), kv[:, :-1].float(), kv[:, -1:].float()
    rstd_x = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    rstd_k = torch.rsqrt(kf.pow(2).mean(-1, keepdim=True) + eps)
    dot = (xf * weight_fused.unsqueeze(0) * kf).sum(-1, keepdim=True)
    score = dot * rstd_x * rstd_k * hidden_size**-0.5
    gate = torch.sigmoid(torch.copysign(torch.sqrt(score.abs().clamp_min(clamp_value)), score))
    return xf + gate * vf


def profile_engram_gate_tile_kernels(
    num_tokens: int, hc_mult: int, hidden_size: int, dtype: DType | str
) -> ComputeMetrics:
    num_tokens, hc_mult, hidden_size = _validate(num_tokens, hc_mult, hidden_size, dtype)
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"torch is required for {BACKEND}") from exc
    try:
        from tile_kernels.engram import engram_gate_fwd, fused_weight
    except ImportError as exc:
        raise ProfilerNotImplemented(
            f"{BACKEND} engram_gate requires DeepInfer's TileKernels (deepinfer_env)"
        ) from exc
    try:
        torch.manual_seed(0)
        device = torch.device("cuda")
        x = torch.randn(num_tokens, hc_mult, hidden_size, device=device, dtype=torch.bfloat16)
        kv = torch.randn(num_tokens, hc_mult + 1, hidden_size, device=device, dtype=torch.bfloat16)
        weight_fused = fused_weight(
            1 + torch.randn(hc_mult, hidden_size, device=device, dtype=torch.bfloat16) / 8,
            1 + torch.randn(hc_mult, hidden_size, device=device, dtype=torch.bfloat16) / 8,
        )

        def run_once() -> Any:
            return engram_gate_fwd(
                x,
                kv,
                weight_fused,
                EPS,
                CLAMP_VALUE,
                save_for_backward=False,
                image_token_mask=None,
                inplace=True,
            )

        rows = slice(0, min(num_tokens, CHECK_TOKENS))
        want = reference(torch, x[rows], kv[rows], weight_fused, EPS, CLAMP_VALUE)
        run_once()  # JIT compile, outside timing; updates x in place.
        torch.cuda.synchronize()
        got = x[rows].float()
        err = ((got - want).norm() / want.norm().clamp_min(1e-30)).item()
        if not err <= TOL:
            raise KernelLaunchFailed(
                f"{BACKEND} engram_gate output mismatch: relative error {err:.3e} > {TOL:.0e}"
            )

        time_ms = Timer.cupti(run_once, warmup=3)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    except (ProfilerNotImplemented, KernelLaunchFailed):
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError(f"{BACKEND} engram_gate ran out of memory") from exc
        raise KernelLaunchFailed(f"{BACKEND} engram_gate failed: {exc}") from exc

    # Streams read and written, keys plus value read.
    bytes_accessed = 2 * num_tokens * hidden_size * (2 * hc_mult + hc_mult + 1)
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=bytes_accessed / elapsed_s / 1e9,
        energy_j=float(energy_j),
    )


__all__ = ["profile_engram_gate_tile_kernels", "reference"]
