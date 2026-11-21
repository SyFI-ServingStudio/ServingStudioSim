"""DeepInfer's Markov-head accumulate -- ``deepgemm_cublaslt`` backend of ``gemm_fp32_accumulate``.

The timed call is ``deep_gemm.bf16_gemm_nt(a, b, d, d)`` with bf16 ``a [m, k]``
and ``b [n, k]`` and an fp32 ``d [m, n]`` that is both the accumulator input
and the output, as ``ReplicatedLinear.apply_weight_bf16(x, out_acc=logits)``
issues it for DFlash's ``markov_w2``. DeepGEMM forwards it to cuBLASLt with
beta = 1 (``smxx_cublaslt.hpp``). Every timed launch adds into the same
buffer; the values grow but the work does not depend on them.

Check: one call adds ``a.float() @ b.float().T`` to a copy of the start value.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

BACKEND = "deepgemm_cublaslt"


def validate(m: int, n: int, k: int, input_dtype: DType | str) -> tuple[int, int, int]:
    if DType.from_value(input_dtype) is not DType.BF16:
        raise ValueError(
            f"{BACKEND} gemm_fp32_accumulate takes input_dtype=bf16, got {input_dtype!r}"
        )
    values = int(m), int(n), int(k)
    if min(values) < 1:
        raise ValueError(f"gemm_fp32_accumulate needs positive sizes, got {values}")
    return values


def profile_gemm_fp32_accumulate_deepgemm(
    m: int, n: int, k: int, input_dtype: DType | str
) -> ComputeMetrics:
    m, n, k = validate(m, n, k, input_dtype)
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"torch is required for {BACKEND}") from exc
    try:
        import deep_gemm
    except ImportError as exc:
        raise ProfilerNotImplemented(
            f"{BACKEND} needs DeepInfer's DeepGEMM (deepinfer_env)"
        ) from exc
    try:
        g = torch.Generator(device="cuda").manual_seed(31)
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g)
        d = torch.randn(m, n, device="cuda", dtype=torch.float32, generator=g)

        def run_once() -> Any:
            deep_gemm.bf16_gemm_nt(a, b, d, d)

        rows = torch.tensor(sorted({0, m // 2, m - 1}), device="cuda")
        start = d[rows].clone()
        run_once()
        torch.cuda.synchronize()
        want = start + a[rows].float() @ b.float().T
        torch.testing.assert_close(d[rows], want, atol=0.05, rtol=0.01)

        time_ms = Timer.cupti(run_once, warmup=5, kernel_name=None)
        energy_j = Energy.perf(run_once, warmup=5, per_iter_time_ms=time_ms)
    except (ProfilerNotImplemented, KernelLaunchFailed):
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError(f"{BACKEND} gemm_fp32_accumulate ran out of memory") from exc
        raise KernelLaunchFailed(f"{BACKEND} gemm_fp32_accumulate failed: {exc}") from exc

    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=2.0 * m * n * k / elapsed_s / 1e12,
        memory_bandwidth_gbps=(2 * (m * k + n * k) + 8 * m * n) / elapsed_s / 1e9,
        energy_j=float(energy_j),
    )


__all__ = ["profile_gemm_fp32_accumulate_deepgemm", "validate"]
