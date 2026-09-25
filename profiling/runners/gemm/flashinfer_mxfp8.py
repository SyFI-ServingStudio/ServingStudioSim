"""vLLM-fork MXFP8 dense linear on SM100 — the ``flashinfer_mxfp8`` backend of
``single_gemm``.

The timed slot is the fork's ``FlashInferCutedslMxfp8LinearKernel.apply_weights``
(``vllm/model_executor/kernels/linear/mxfp8/flashinfer.py``), the first MXFP8
linear kernel vLLM selects on CUDA. One call issues a fixed sequence:

1. ``mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)`` — FlashInfer
   ``mxfp8_quantize(backend="cute-dsl")``: bf16 -> e4m3 plus one ue8m0 scale per
   32 elements, scales in the F8_128x4 swizzled layout;
2. ``mm_mxfp8(..., backend="cute-dsl")`` — FlashInfer's
   ``Sm100BlockScaledPersistentDenseGemm`` (or its split-K sibling), bf16 out.

The weight goes through the production ``process_weights_after_loading``: e4m3
``[N, K]`` data stored column-major as ``[K, N]`` and a row-major ``[N, K/32]``
ue8m0 scale swizzled to F8_128x4. Synthetic weights are quantized from a bf16
normal matrix with the fork's own MXFP8 quantizer.

vLLM enables FlashInfer autotune at kernel warmup by default, so the runner
tunes the exact call once (persisted via ``autotune_cached``) before timing.
CUPTI time is the sum of every kernel in the sequence (``kernel_name=None``).
The GEMM launches with PDL, but on B200 the interval union of the two kernels
measured within 0.3% of the sum at every smoked shape, so the overlap is
negligible.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.autotune_cache import autotune_cached
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

MXFP8_BLOCK = 32
# ``apply_weights`` asserts both K and N are at least this.
MIN_DIM = 128
# bf16 output rounding (2^-8 relative) plus accumulation-order differences.
REL_FROBENIUS_TOL = 1e-2


def _validate(m: int, n: int, k: int, dtype: DType | str) -> tuple[int, int, int]:
    if DType.from_value(dtype) is not DType.MXFP8_E4M3:
        raise ValueError(f"flashinfer_mxfp8 single-GEMM requires dtype=mxfp8_e4m3, got {dtype!r}")
    m, n, k = int(m), int(n), int(k)
    if m < 1:
        raise ValueError(f"flashinfer_mxfp8 requires m >= 1, got m={m}")
    if k < MIN_DIM or k % MXFP8_BLOCK:
        raise ValueError(
            f"flashinfer_mxfp8 requires k >= {MIN_DIM} and k % {MXFP8_BLOCK} == 0, got k={k}"
        )
    if n < MIN_DIM:
        raise ValueError(f"flashinfer_mxfp8 requires n >= {MIN_DIM}, got n={n}")
    return m, n, k


def _load_runtime() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented("torch is required for flashinfer_mxfp8") from exc
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("flashinfer_mxfp8 requires CUDA")
    if tuple(torch.cuda.get_device_capability()) != (10, 0):
        raise ProfilerNotImplemented("flashinfer_mxfp8 is verified on SM100 (B200) only")
    try:
        from flashinfer.autotuner import autotune
        from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
            FlashInferCutedslMxfp8LinearKernel,
        )
        from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
            Mxfp8LinearLayerConfig,
        )
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
            mxfp8_e4m3_quantize,
        )
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "flashinfer_mxfp8 requires the vLLM fork environment (vllm_fork_env)"
        ) from exc
    supported, reason = FlashInferCutedslMxfp8LinearKernel.is_supported()
    if not supported:
        raise ProfilerNotImplemented(f"FlashInfer CuTe-DSL MXFP8 linear unavailable: {reason}")
    kernel = FlashInferCutedslMxfp8LinearKernel(Mxfp8LinearLayerConfig())
    return torch, autotune, kernel, mxfp8_e4m3_quantize


def _dequantize(torch: Any, data: Any, scale_2d: Any) -> Any:
    """fp32 value of an MXFP8 tensor from its data and row-major ue8m0 scales."""

    rows, cols = data.shape
    blocks = data.float().view(rows, cols // MXFP8_BLOCK, MXFP8_BLOCK)
    factor = torch.exp2(scale_2d.float() - 127.0).unsqueeze(-1)
    return (blocks * factor).view(rows, cols)


def check_against_dequantized_reference(
    torch: Any, quantize: Any, x: Any, weight_q: Any, weight_scale_2d: Any, out: Any
) -> float:
    """Relative Frobenius error of ``out`` against fp32 ``deq(x) @ deq(W)^T``.

    The activation is re-quantized with the linear (unswizzled) scale layout;
    MXFP8 quantization is deterministic, so a swizzle mistake in the timed path
    shows up as a large error rather than being shared by the reference.
    """

    x_q, x_scale = quantize(x, is_sf_swizzled_layout=False)
    ref = (
        _dequantize(torch, x_q, x_scale.view(x.shape[0], -1))
        @ _dequantize(torch, weight_q, weight_scale_2d).t()
    )
    err = ((out.float() - ref).norm() / ref.norm().clamp_min(1e-30)).item()
    if not err <= REL_FROBENIUS_TOL:
        raise KernelLaunchFailed(
            f"flashinfer_mxfp8 output mismatch: relative Frobenius error {err:.3e} "
            f"> {REL_FROBENIUS_TOL:.0e}"
        )
    return err


def profile_single_gemm_flashinfer_mxfp8(
    m: int,
    n: int,
    k: int,
    dtype: DType | str,
) -> ComputeMetrics:
    m, n, k = _validate(m, n, k, dtype)
    torch, autotune, kernel, quantize = _load_runtime()
    try:
        torch.manual_seed(0)
        weight_q, weight_scale_2d = quantize(
            torch.randn(n, k, device="cuda", dtype=torch.bfloat16),
            is_sf_swizzled_layout=False,
        )
        weight_scale_2d = weight_scale_2d.view(n, -1)
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(weight_q.clone(), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(weight_scale_2d.clone(), requires_grad=False)
        kernel.process_weights_after_loading(layer)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)

        def run_once() -> Any:
            return kernel.apply_weights(layer, x)

        with autotune_cached(autotune, "single_gemm.flashinfer_mxfp8"):
            run_once()
        out = run_once()
        torch.cuda.synchronize()
        check_against_dequantized_reference(torch, quantize, x, weight_q, weight_scale_2d, out)
        del out

        time_ms = Timer.cupti(run_once, warmup=3)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    except (ProfilerNotImplemented, KernelLaunchFailed):
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError("flashinfer_mxfp8 ran out of memory") from exc
        raise KernelLaunchFailed(f"flashinfer_mxfp8 failed: {exc}") from exc

    # Quant reads bf16 x and writes e4m3 + scales; the GEMM reads both e4m3
    # operands with their scales and writes bf16.
    scale_bytes = (m + n) * (k // MXFP8_BLOCK)
    bytes_accessed = 2 * m * k + 2 * m * k + n * k + scale_bytes + 2 * m * n
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=2 * m * n * k / elapsed_s / 1e12,
        memory_bandwidth_gbps=bytes_accessed / elapsed_s / 1e9,
        energy_j=float(energy_j),
    )


__all__ = ["check_against_dequantized_reference", "profile_single_gemm_flashinfer_mxfp8"]
