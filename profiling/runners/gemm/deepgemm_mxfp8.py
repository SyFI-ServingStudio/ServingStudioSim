"""DeepInfer's FP8 dense linear on SM100 — the ``deepgemm_mxfp8`` backend of
``single_gemm``.

The timed slot is DeepInfer's ``ReplicatedLinear.apply_weight_fp8``
(``deepinfer/model_executor/layers/linear.py``) on an activation its producer
already quantized, which is how every FP8 linear of DeepSeek-V4.1 runs there:
the producer (``tile_kernels.quant.per_token_cast``, a fused SwiGLU/HC cast, or
the attention ``wv`` einsum's FP8 epilogue) hands the linear an
``(e4m3, sf)`` pair, and the linear issues one
``deep_gemm.fp8_fp4_gemm_nt(x, weight, out, recipe=(1, 1, 32))``. DeepGEMM runs
it as one ``deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl`` launch with bf16 output.

Operands, as DeepInfer lays them out:

- A ``[m, k]`` e4m3 with one UE8M0 scale per token per 32 k-elements, written
  by the production producer ``tile_kernels.quant.per_token_cast`` with
  DeepInfer's ``get_fp8_cast_params()``: packed UE8M0 in int32, TMA-aligned
  MN-major. The runner makes it outside timing from a bf16 normal matrix.
- B ``[n, k]`` e4m3 with the checkpoint's UE8M0 scale per 32×32 block, which
  the production ``ReplicatedLinear`` constructor expands and packs once with
  ``deep_gemm.transform_sf_into_required_layout(recipe=(1, 32, 32))``.
- Output ``[m, n]`` bf16, row stride padded to 32 bytes by the layer's own
  ``_allocate_output``.

No accumulation (``out_acc``) and no bias: DeepInfer's prefill linears launch
with ``kWithAccumulation = false`` (nsys capture template arguments).

The kernel tile configuration comes from DeepGEMM's host heuristic for the
exact ``(m, n, k)``; there is no autotuner. CUPTI time sums every launch of the
call (``kernel_name=None``): a scale layout DeepGEMM had to re-transform would
add launches and show up in the time instead of hiding.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

BACKEND = "deepgemm_mxfp8"
MXFP8_BLOCK = 32
# The numerical check reads at most this many output rows (rows are independent).
CHECK_ROWS = 2048
# bf16 output rounding (2^-8 relative) plus accumulation-order differences.
REL_FROBENIUS_TOL = 1e-2


def _validate(m: int, n: int, k: int, dtype: DType | str) -> tuple[int, int, int]:
    if DType.from_value(dtype) is not DType.MXFP8_E4M3:
        raise ValueError(f"{BACKEND} single-GEMM requires dtype=mxfp8_e4m3, got {dtype!r}")
    m, n, k = int(m), int(n), int(k)
    if m < 1 or n < 1:
        raise ValueError(f"{BACKEND} requires m >= 1 and n >= 1, got m={m}, n={n}")
    if k < MXFP8_BLOCK or k % MXFP8_BLOCK:
        raise ValueError(
            f"{BACKEND} requires k to be a positive multiple of {MXFP8_BLOCK}, got k={k}"
        )
    return m, n, k


def _load_runtime() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"torch is required for {BACKEND}") from exc
    try:
        import tile_kernels
        from deep_gemm.utils import per_block_cast_to_fp8, per_token_cast_to_fp8
        from deepinfer.model_executor.layers.linear import (
            ReplicatedLinear,
            get_fp8_cast_params,
        )
    except ImportError as exc:
        raise ProfilerNotImplemented(
            f"{BACKEND} requires DeepInfer and its vendored DeepGEMM (deepinfer_env)"
        ) from exc

    def quantize_activation(x: Any) -> tuple[Any, Any]:
        return tile_kernels.quant.per_token_cast(x, **get_fp8_cast_params())

    return torch, ReplicatedLinear, quantize_activation, (per_token_cast_to_fp8, per_block_cast_to_fp8)


def _dequantize(torch: Any, data: Any, scale: Any, block_rows: int) -> Any:
    """fp32 value of ``data`` from unpacked fp32 power-of-two scales.

    ``scale`` holds one value per ``block_rows`` rows and 32 columns.
    """
    rows, cols = data.shape
    factor = scale.float().repeat_interleave(block_rows, dim=0)[:rows]
    factor = factor.repeat_interleave(MXFP8_BLOCK, dim=1)[:, :cols]
    return data.float() * factor


def check_against_dequantized_reference(
    torch: Any, casts: Any, x: Any, w_q: Any, w_scale: Any, out: Any
) -> float:
    """Relative Frobenius error of ``out`` against fp32 ``deq(x) @ deq(W)^T``.

    The reference re-quantizes the activation with DeepGEMM's own row-major
    reference cast (same UE8M0 rounding) and reads the weight's unpacked block
    scales, so a mistake in the packed layouts the timed call reads shows up
    as a large error instead of being shared with the oracle. Only a strided
    sample of rows is compared; output rows are independent.
    """
    per_token_cast_to_fp8, _ = casts
    step = max(1, x.shape[0] // CHECK_ROWS)
    rows = slice(0, x.shape[0], step)
    x_q, x_scale = per_token_cast_to_fp8(x[rows].contiguous(), use_ue8m0=True, gran_k=MXFP8_BLOCK)
    ref = _dequantize(torch, x_q, x_scale, 1) @ _dequantize(torch, w_q, w_scale, MXFP8_BLOCK).t()
    err = ((out[rows].float() - ref).norm() / ref.norm().clamp_min(1e-30)).item()
    if not err <= REL_FROBENIUS_TOL:
        raise KernelLaunchFailed(
            f"{BACKEND} output mismatch: relative Frobenius error {err:.3e} "
            f"> {REL_FROBENIUS_TOL:.0e}"
        )
    return err


def profile_single_gemm_deepgemm_mxfp8(
    m: int,
    n: int,
    k: int,
    dtype: DType | str,
) -> ComputeMetrics:
    m, n, k = _validate(m, n, k, dtype)
    torch, ReplicatedLinear, quantize_activation, casts = _load_runtime()
    _, per_block_cast_to_fp8 = casts
    try:
        torch.manual_seed(0)
        device = torch.device("cuda")

        # Weight: e4m3 with a fp32 UE8M0 scale per 32x32 block, the checkpoint
        # form; the production constructor packs it into DeepGEMM's layout.
        w_q, w_scale = per_block_cast_to_fp8(
            torch.randn(n, k, device=device, dtype=torch.bfloat16),
            use_ue8m0=True,
            gran_k=MXFP8_BLOCK,
        )
        weight = torch.nn.Parameter(w_q, requires_grad=False)
        weight.sf = w_scale
        layer = ReplicatedLinear(k, n, weight=weight)

        x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        activation = quantize_activation(x)

        def run_once() -> Any:
            return layer.apply_weight_fp8(activation)

        out = run_once()  # DeepGEMM JIT compile, outside timing.
        torch.cuda.synchronize()
        check_against_dequantized_reference(torch, casts, x, w_q, w_scale, out)
        del out, x

        time_ms = Timer.cupti(run_once, warmup=3)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    except (ProfilerNotImplemented, KernelLaunchFailed):
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError(f"{BACKEND} ran out of memory") from exc
        raise KernelLaunchFailed(f"{BACKEND} failed: {exc}") from exc

    # Both e4m3 operands with their UE8M0 scales (one byte per 32 elements of A
    # per row, of B per row after expansion), bf16 output.
    scale_bytes = (m + n) * (k // MXFP8_BLOCK)
    bytes_accessed = m * k + n * k + scale_bytes + 2 * m * n
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=2 * m * n * k / elapsed_s / 1e12,
        memory_bandwidth_gbps=bytes_accessed / elapsed_s / 1e9,
        energy_j=float(energy_j),
    )


__all__ = ["check_against_dequantized_reference", "profile_single_gemm_deepgemm_mxfp8"]
