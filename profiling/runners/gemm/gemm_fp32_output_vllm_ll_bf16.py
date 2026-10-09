"""Profile the alignment vLLM fork's ``_linear_with_fp32_out`` GEMM dispatch.

The timed callable is ``vllm.models.inkling.nvidia.moe._linear_with_fp32_out``
itself, so the runner follows the production dispatch rather than restating it:

- ``m <= 64`` (``_INKLING_LL_BF16_MAX_TOKENS``), bf16, contiguous, ``k % 8 == 0``,
  SM90+ and cute-DSL importable: ``ll_bf16.ll_bf16_gemm``. Inside it, ``m <= 4``
  or ``k < 2048`` runs the ``LLBf16Dotprod`` kernel (compiled per ``m``; ``m == 1``
  uses the PDL weight-prefetch instance), otherwise ``LLBf16SplitK`` with the
  arch-tuned or default (split_k, stages);
- otherwise ``torch.mm(x, w.T, out_dtype=torch.float32)`` (cuBLAS, one kernel
  or a split-K pair).

The cute-DSL kernels compile on the first call, which runs before timing.
CUPTI counts every launch of one call, overlapping launches once.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "gemm_fp32_output:vllm_ll_bf16_auto"
# FP32 accumulation of bf16 products in a different order than the reference;
# relative to the output's Frobenius norm.
_REL_FROBENIUS_TOL = 1e-4


def _validate_args(m: int, n: int, k: int, input_dtype: DType | str) -> tuple[int, int, int]:
    for name, value in (("m", m), ("n", n), ("k", k)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if DType.from_value(input_dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_BACKEND} requires input_dtype=bf16")
    return m, n, k


def _check_output(torch: Any, hidden: Any, weight: Any, actual: Any) -> None:
    if actual.dtype is not torch.float32:
        raise AssertionError(f"expected FP32 output, got {actual.dtype}")
    if tuple(actual.shape) != (hidden.shape[0], weight.shape[0]):
        raise AssertionError(f"unexpected output shape {tuple(actual.shape)}")
    # bf16 products are exact in fp32, so an fp32 GEMM over the upcast inputs
    # is an independent reference that differs only by summation order.
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        expected = hidden.float() @ weight.float().T
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    error = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)
    if not bool(torch.isfinite(actual).all()) or float(error) > _REL_FROBENIUS_TOL:
        raise AssertionError(f"relative Frobenius error {float(error):.3e} exceeds tolerance")


def profile_gemm_fp32_output_vllm_ll_bf16(
    m: int,
    n: int,
    k: int,
    input_dtype: DType | str,
) -> ComputeMetrics:
    m, n, k = _validate_args(m, n, k, input_dtype)
    try:
        import torch
        from vllm.models.inkling.nvidia.moe import _linear_with_fp32_out
    except ImportError as exc:
        raise ProfilerNotImplemented(
            f"{_BACKEND} requires the alignment vLLM fork (vllm_upstream_fork_env)"
        ) from exc

    try:
        device = torch.device("cuda", torch.cuda.current_device())
        generator = torch.Generator(device=device).manual_seed(31)
        hidden = torch.randn((m, k), dtype=torch.bfloat16, device=device, generator=generator)
        # The production router weight is a replicated [n, k] bf16 Parameter.
        weight = torch.randn((n, k), dtype=torch.bfloat16, device=device, generator=generator)

        def launch():
            return _linear_with_fp32_out(hidden, weight)

        # First call compiles the cute-DSL kernel for this dispatch key.
        first = launch()
        torch.cuda.synchronize()
        _check_output(torch, hidden, weight, first)
        time_ms = Timer.cupti(launch, warmup=5, interval_union=True)
        energy_j = Energy.perf(launch, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {exc}") from exc

    elapsed_s = time_ms / 1000.0
    flops = 2 * m * n * k
    logical_bytes = 2 * m * k + 2 * n * k + 4 * m * n
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=float(flops / elapsed_s / 1e12 if elapsed_s else 0.0),
        memory_bandwidth_gbps=float(logical_bytes / elapsed_s / 1e9 if elapsed_s else 0.0),
        energy_j=float(energy_j),
    )


__all__ = ["profile_gemm_fp32_output_vllm_ll_bf16"]
