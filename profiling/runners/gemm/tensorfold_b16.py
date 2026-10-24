"""TensorFold's bf16 GDN gate pair: the ``tensorfold_b16`` (verify) and
``tensorfold_b16_prompt`` (prefill) backends of ``single_gemm``.

A Qwen3.5-family checkpoint keeps each GDN layer's ``in_proj_b`` and
``in_proj_a`` (``num_value_heads`` outputs each, bf16) unquantized. In a verify
forward TensorFold multiplies the layer's normed rows by both in one launch:
``families/qwen3_5/cuda/forward.py`` ``_mm_group`` ->
``families/qwen3_5/cuda/b16.py`` ``matmul_pair(x, w_b, w_a)`` ->
``b16_linear_pair`` -> ``b16_kernel`` (``b16.cu``) on grid
``(ceil(n_half / WARPS), ceil(m / R), 2)``: grid z picks the weight, a warp
computes one output column for R rows with CUDA-core FMAs (R, WARPS = 1, 4
below 64 rows; 4, 8 below 512; 16, 16 from 512).

The row's ``n`` is the pair's total output width (``2 · num_value_heads``),
split into two equal halves. Pricing the pair as one GEMM of that width is
exact in launch shape: one ``b16_linear`` of width ``n`` launches the same
number of blocks of the same kernel, and measured on GB10 (K = 5120, two
48-wide halves vs one 96-wide weight, three repeats) the pair times like it at
every row count from 1 to 1024 (median ratio 1.00, every repeat within about
±10% except one 64-row run at 0.79 that did not repeat).

The timed closure is exactly ``matmul_pair(x, w0, w1)`` on bf16 rows (the call
casts nothing then); CUPTI sums its launches (``kernel_name=None``), which is
the one ``b16_kernel``. The Torch oracle multiplies in fp32 outside timing.

In prefill TensorFold takes the same pair through ``families/qwen3_5/cuda/
prefill.py`` ``_mm_group`` -> ``b16.prompt_pair(x, w_b, w_a)`` ->
``b16_prompt_pair`` -> ``b16_prompt_kernel`` (``b16.cu``): bf16 mma.sync tiles
of BM x 64 on grid ``(ceil(n_half / 64), ceil(m / BM), 2)``, BM picked from
the pair's total width (32 until the grid fills two blocks a SM, then 64 and
128). The ``tensorfold_b16_prompt`` row again keys the pair's total width
``n``. Here the two halves are not interchangeable with one ``n``-wide launch:
grid z (the weight) is the slowest index, so all of one half's tiles run before
the other's and each half re-reads the rows (the likely cause of the gap
below; inferred from the launch order, not measured separately). Up to about 3.7k rows the pair
times like one ``n``-wide ``b16_prompt`` (ratio 0.92-1.05 on GB10); at 4096
rows, once the rows no longer stay in L2, it is 23-30% slower, and at 8192
about 1.9x. The row measures the pair launch itself, so that cost is in it.

Runs in ``tensorfold_env``, against which ``b16.cu`` JIT-compiles on first use.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

BACKEND = "tensorfold_b16"
# b16_linear_pair reads 16-byte (8-element) chunks along K.
K_ALIGN = 8
# b16_prompt_pair loads 64-input stages along K.
K_ALIGN_PROMPT = 64
# fp32 accumulation in a different order than Torch, then one bf16 rounding.
REL_FROBENIUS_TOL = 1e-2
_WEIGHT_STD = 0.02


def _validate(
    m: int, n: int, k: int, dtype: DType | str, k_align: int = K_ALIGN
) -> tuple[int, int, int]:
    dtype = DType.from_value(dtype)
    if dtype is not DType.BF16:
        raise ValueError(f"{BACKEND} requires dtype bf16, got {dtype.value}")
    m, n, k = int(m), int(n), int(k)
    if m < 1:
        raise ValueError(f"{BACKEND} requires m >= 1, got m={m}")
    if n < 2 or n % 2:
        raise ValueError(f"{BACKEND} times a pair of equal halves: n must be even and >= 2, got n={n}")
    if k < k_align or k % k_align:
        raise ValueError(f"{BACKEND} requires k to be a positive multiple of {k_align}, got k={k}")
    return m, n, k


def _load_runtime() -> tuple[Any, Any]:
    try:
        import torch
        from tensorfold.families.qwen3_5.cuda import b16
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{BACKEND} requires the TensorFold environment (tensorfold_env)") from exc
    return torch, b16


def check_against_reference(torch: Any, x: Any, w0: Any, w1: Any, outs: Any) -> float:
    """Relative Frobenius error of both halves against an fp32 Torch matmul."""
    got = torch.cat([outs[0], outs[1]], dim=1).float()
    ref = x.float() @ torch.cat([w0, w1]).float().t()
    err = ((got - ref).norm() / ref.norm().clamp_min(1e-30)).item()
    if not err <= REL_FROBENIUS_TOL:
        raise KernelLaunchFailed(
            f"{BACKEND} output mismatch: relative Frobenius error {err:.3e} > {REL_FROBENIUS_TOL:.0e}"
        )
    return err


def _profile_pair(m: int, n: int, k: int, pair_fn: str) -> ComputeMetrics:
    torch, b16 = _load_runtime()
    pair = getattr(b16, pair_fn)
    try:
        gen = torch.Generator(device="cuda").manual_seed(0)
        half = n // 2
        w0 = torch.randn(half, k, device="cuda", generator=gen).mul_(_WEIGHT_STD).to(torch.bfloat16)
        w1 = torch.randn(half, k, device="cuda", generator=gen).mul_(_WEIGHT_STD).to(torch.bfloat16)
        x = torch.randn(m, k, device="cuda", generator=gen).to(torch.bfloat16)

        def run_once() -> Any:
            return pair(x, w0, w1)

        outs = run_once()                  # the first call JIT-compiles b16.cu
        torch.cuda.synchronize()
        check_against_reference(torch, x, w0, w1, outs)
        del outs

        # Every launch of the call counts (kernel_name=None): one pair launch.
        time_ms = Timer.cupti(run_once, warmup=3)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    except (ProfilerNotImplemented, KernelLaunchFailed):
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError(f"{BACKEND} ran out of memory") from exc
        raise KernelLaunchFailed(f"{BACKEND} failed: {exc}") from exc

    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=2 * m * n * k / elapsed_s / 1e12,
        memory_bandwidth_gbps=2 * (m * k + n * k + m * n) / elapsed_s / 1e9,
        energy_j=float(energy_j),
    )


def profile_single_gemm_tensorfold_b16(m: int, n: int, k: int, dtype: DType | str) -> ComputeMetrics:
    m, n, k = _validate(m, n, k, dtype)
    return _profile_pair(m, n, k, "matmul_pair")


def profile_single_gemm_tensorfold_b16_prompt(
    m: int, n: int, k: int, dtype: DType | str
) -> ComputeMetrics:
    # Below 64-input K stages prompt_pair falls back to two prompt() launches.
    m, n, k = _validate(m, n, k, dtype, K_ALIGN_PROMPT)
    return _profile_pair(m, n, k, "prompt_pair")


__all__ = [
    "check_against_reference",
    "profile_single_gemm_tensorfold_b16",
    "profile_single_gemm_tensorfold_b16_prompt",
]
