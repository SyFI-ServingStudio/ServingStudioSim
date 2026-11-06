"""Static per-tensor FP8 dense linear on Hopper: the ``vllm_cutlass_fp8_tensor``
backend of ``single_gemm``.

On SM90 vLLM serves ModelOpt ``FP8`` linears (one FP32 weight scale, one static
FP32 activation scale) with ``CutlassFP8ScaledMMLinearKernel``
(``model_executor/kernels/linear/scaled_mm/cutlass.py``; the SM10x-only FlashInfer
kernel ahead of it in ``_POSSIBLE_FP8_KERNELS`` is not supported there). Its
``apply_scaled_mm`` calls ``ops.cutlass_scaled_mm(A, B, scale_a, scale_b, bf16)``
on the row-major E4M3 activation and the column-major ``[k, n]`` weight view; the
Hopper alignment captures (2026-11-03) show it as ``cutlass_3x_gemm_sm90_fp8``.
This runner times that call. ``n`` and ``k`` must be multiples of 16, as the
loader pads them.

The activation quantization runs before this call, fused into the preceding norm
by Inductor (``static_fp8_quant`` / ``multi_add_rms_norm``); it is not part of
this row. Runs in ``vllm_env``.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

ALIGNMENT = 16
REL_FROBENIUS_TOL = 1e-2
_A_SCALE = 0.73
_W_SCALE = 1.13


def _validate(m: int, n: int, k: int, dtype: DType | str) -> tuple[int, int, int]:
    if DType.from_value(dtype) is not DType.FP8_E4M3:
        raise ValueError(f"vllm_cutlass_fp8_tensor requires dtype=fp8_e4m3, got {dtype!r}")
    m, n, k = int(m), int(n), int(k)
    if m < 1 or n < 1 or k < 1 or n % ALIGNMENT or k % ALIGNMENT:
        raise ValueError(
            f"vllm_cutlass_fp8_tensor requires m >= 1 and n, k multiples of {ALIGNMENT}, got m={m} n={n} k={k}"
        )
    return m, n, k


def _load_runtime() -> tuple[Any, Any]:
    try:
        import torch
        from vllm import _custom_ops as ops
    except ImportError as exc:
        raise ProfilerNotImplemented("vllm_cutlass_fp8_tensor requires the vLLM environment (vllm_env)") from exc
    return torch, ops


def profile_single_gemm_vllm_cutlass_fp8_tensor(m: int, n: int, k: int, dtype: DType | str) -> ComputeMetrics:
    m, n, k = _validate(m, n, k, dtype)
    torch, ops = _load_runtime()
    try:
        g = torch.Generator(device="cuda").manual_seed(0)
        a = torch.randn(m, k, device="cuda", generator=g).to(torch.float8_e4m3fn)
        weight = torch.randn(n, k, device="cuda", generator=g).to(torch.float8_e4m3fn)
        weight_t = weight.t()  # [k, n] column-major, as the loader stores it
        scale_a = torch.tensor([[_A_SCALE]], device="cuda", dtype=torch.float32)
        scale_w = torch.tensor([[_W_SCALE]], device="cuda", dtype=torch.float32)

        def run_once() -> Any:
            return ops.cutlass_scaled_mm(a, weight_t, scale_a=scale_a, scale_b=scale_w, out_dtype=torch.bfloat16)

        out = run_once()
        torch.cuda.synchronize()
        ref = (a.float() * _A_SCALE) @ (weight.float() * _W_SCALE).t()
        err = ((out.float() - ref).norm() / ref.norm().clamp_min(1e-30)).item()
        if not err <= REL_FROBENIUS_TOL:
            raise KernelLaunchFailed(
                f"vllm_cutlass_fp8_tensor output mismatch: relative Frobenius error {err:.3e} > {REL_FROBENIUS_TOL:.0e}"
            )
        del out, ref
        time_ms = Timer.cupti(run_once, warmup=3)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    except (ProfilerNotImplemented, KernelLaunchFailed):
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError("vllm_cutlass_fp8_tensor ran out of memory") from exc
        raise KernelLaunchFailed(f"vllm_cutlass_fp8_tensor failed: {exc}") from exc

    bytes_accessed = m * k + n * k + 2 * m * n
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=2 * m * n * k / elapsed_s / 1e12,
        memory_bandwidth_gbps=bytes_accessed / elapsed_s / 1e9,
        energy_j=float(energy_j),
    )


__all__ = ["profile_single_gemm_vllm_cutlass_fp8_tensor"]
