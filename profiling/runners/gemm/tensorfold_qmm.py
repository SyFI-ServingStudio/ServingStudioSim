"""TensorFold's affine 4-bit lane matmul: the ``tensorfold_qmm`` backend of ``single_gemm``.

TensorFold's DFlash2 drafter re-quantizes every large bf16 projection at load
(``families/qwen3_5/cuda/dflash2.py``: ``packed_draft`` admits it, ``quantize4``
rounds it to MLX-style affine 4-bit codes with one bf16 scale and bias per 64
inputs, ``qmm_fast.tile`` packs it into the lane kernel's tile layout). Each
projection then runs ``qmm_fast.matmul(x, q, xs)``: bf16 rows times the packed
weight, bf16 output. On SM12x that is ``tensorfold.cuda.kernels.qmm.matmul`` ->
``qmm_group`` with one part, the ``group_kernel`` launch of ``qmm_group.cu``
(bf16 mma.sync m16n8k16 on codes widened to bf16; the per-group scale and the
``bias * sum(x)`` correction are applied in fp32). Its K split comes from the
weight shape alone (``qmm.split_k``) and is reduced in a thread-block cluster
inside the same launch.

The timed closure is exactly ``qmm_fast.matmul(x, q, xs)`` with the rows' fp32
64-input group sums ``xs`` already computed: production either passes them in
(the down projection takes them from the SwiGLU kernel, gate|up from one
``group_sums`` for both) or lets ``matmul`` launch the small Triton
``_group_sums`` first, which is not part of this GEMM. CUPTI sums every launch in
the closure (``kernel_name=None``); with ``xs`` given the closure is the single
``group_kernel``.

The oracle dequantizes the MLX codes (before tiling, so it also checks the tile
pack) as ``q * scale + bias`` and multiplies the bf16 rows in fp32, outside
timing.

Runs in ``tensorfold_env``: the TensorFold checkout's own Torch stack, against
which its CUDA extension JIT-compiles on first use.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

BACKEND = "tensorfold_qmm"
# quantize4 / qmm_group take groups of 64 inputs along K.
GROUP = 64
# bf16 output rounding (2^-8 relative) plus accumulation-order differences.
REL_FROBENIUS_TOL = 1e-2
# The oracle checks at most this many rows and output columns.
_CHECK_ROWS = 256
_CHECK_COLS = 4096
# Typical spread of a trained projection's weights; quantize4 is scale-free anyway.
_WEIGHT_STD = 0.02


def _validate(m: int, n: int, k: int, dtype: DType | str) -> tuple[int, int, int]:
    dtype = DType.from_value(dtype)
    if dtype is not DType.W4A16_AFFINE_G64:
        raise ValueError(f"{BACKEND} requires dtype w4a16_affine_g64, got {dtype.value}")
    m, n, k = int(m), int(n), int(k)
    if m < 1 or n < 1:
        raise ValueError(f"{BACKEND} requires m >= 1 and n >= 1, got m={m}, n={n}")
    if k < GROUP or k % GROUP:
        raise ValueError(f"{BACKEND} requires k to be a positive multiple of {GROUP}, got k={k}")
    return m, n, k


def _load_runtime() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from tensorfold.cuda.kernels import qmm
        from tensorfold.families.qwen3_5.cuda import qmm_fast
        from tensorfold.families.qwen3_5.cuda.dflash2 import quantize4
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{BACKEND} requires the TensorFold environment (tensorfold_env)") from exc
    return torch, qmm, qmm_fast, quantize4


def dequantize_mlx(torch: Any, words: Any, scales: Any, biases: Any) -> Any:
    """MLX affine 4-bit (n, k/8) int32 words -> fp32 (n, k): input 8i + j is nibble j of word i, ``q * s + b``."""
    shifts = torch.arange(8, device=words.device, dtype=torch.int32) * 4
    q = ((words[..., None] >> shifts) & 0xF).flatten(-2).float()
    return q * scales.float().repeat_interleave(GROUP, dim=-1) + biases.float().repeat_interleave(GROUP, dim=-1)


def check_against_reference(torch: Any, x: Any, mlx: Any, out: Any) -> float:
    """Relative Frobenius error of the kernel output against Torch on the dequantized weight."""
    m, n = out.shape
    rows = torch.unique(torch.linspace(0, m - 1, min(m, _CHECK_ROWS), device=out.device).long())
    cols = torch.unique(torch.linspace(0, n - 1, min(n, _CHECK_COLS), device=out.device).long())
    w = dequantize_mlx(torch, mlx.weight[cols], mlx.scales[cols], mlx.biases[cols])
    ref = x[rows].float() @ w.t()
    got = out[rows][:, cols].float()
    err = ((got - ref).norm() / ref.norm().clamp_min(1e-30)).item()
    if not err <= REL_FROBENIUS_TOL:
        raise KernelLaunchFailed(
            f"{BACKEND} output mismatch: relative Frobenius error {err:.3e} > {REL_FROBENIUS_TOL:.0e}"
        )
    return err


def profile_single_gemm_tensorfold_qmm(m: int, n: int, k: int, dtype: DType | str) -> ComputeMetrics:
    m, n, k = _validate(m, n, k, dtype)
    torch, qmm, qmm_fast, quantize4 = _load_runtime()
    if not qmm.grouped(torch.cuda.current_device()):
        raise ProfilerNotImplemented(f"{BACKEND} times the SM12x grouped lane kernel (qmm_group)")
    try:
        gen = torch.Generator(device="cuda").manual_seed(0)
        w = torch.randn(n, k, device="cuda", generator=gen, dtype=torch.float32).mul_(_WEIGHT_STD)
        mlx = quantize4(w.to(torch.bfloat16))           # DFlash2's load-time quantization
        del w
        q = qmm_fast.tile(mlx)                          # and its packing into the lane kernel's tiles
        x = torch.randn(m, k, device="cuda", generator=gen, dtype=torch.float32).to(torch.bfloat16)
        xs = qmm.group_sums(x, GROUP)                   # computed before the GEMM in production

        def run_once() -> Any:
            return qmm_fast.matmul(x, q, xs)

        # The first call JIT-compiles the extension (minutes, once per cache).
        out = run_once()
        torch.cuda.synchronize()
        check_against_reference(torch, x, mlx, out)
        del mlx, out
        torch.cuda.empty_cache()

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
        memory_bandwidth_gbps=logical_bytes(m, n, k) / elapsed_s / 1e9,
        energy_j=float(energy_j),
    )


def logical_bytes(m: int, n: int, k: int) -> int:
    """bf16 rows and their fp32 group sums, 4-bit codes with bf16 scale and bias per group, bf16 output."""
    groups = k // GROUP
    return 2 * m * k + 4 * m * groups + n * k // 2 + 2 * 2 * n * groups + 2 * m * n


__all__ = [
    "check_against_reference",
    "dequantize_mlx",
    "logical_bytes",
    "profile_single_gemm_tensorfold_qmm",
]
