"""Static per-tensor FP8 dense linear through vLLM's CUTLASS kernel: the
``vllm_cutlass_fp8`` backend of ``single_gemm``.

vLLM serves ModelOpt ``FP8`` linears (one FP32 scale for the weight, one static
FP32 scale for the activation) with the first ``_POSSIBLE_FP8_KERNELS`` entry
whose ``is_supported`` passes (``model_executor/kernels/linear/__init__.py``).
Below SM10x, where ``FlashInferFP8ScaledMMLinearKernel`` declines, that is
``CutlassFP8ScaledMMLinearKernel`` (``kernels/linear/scaled_mm/cutlass.py``);
its ``is_supported`` is ``cutlass_fp8_supported()``, i.e. the C++
``cutlass_scaled_mm_supports_fp8``: compute capability 8.9+ (CUDA 12.4+ on
SM89, 12.0+ from SM90).

The ModelOpt loader (``KFp8StaticTensor.process``) stores the ``[n, k]`` E4M3
weight as its ``[k, n]`` column-major transpose with the max of the per-shard
scales as one 0-d FP32 scale. ``process_weights_after_loading`` zero-pads that
weight to multiples of 16 in ``k`` and ``n`` when needed (no-op otherwise), and
``apply_scaled_mm`` pads the activation's ``k``, calls
``vllm._custom_ops.cutlass_scaled_mm`` and slices the output back to ``n``. The
runner runs both methods of the production class on a kernel instance built
without ``__init__`` (which only constructs the ``QuantFP8`` activation quantizer
this row excludes), and times ``apply_scaled_mm``.

The activation quantization (``QuantFP8``, static per tensor) is fused by
Inductor into the preceding norm or elementwise kernel; it is not part of this
row. ``cutlass_scaled_mm`` is one CUTLASS launch on SM89; the CUPTI sum also
counts the pad/slice copies when ``k`` or ``n`` is not a multiple of 16.
Runs in ``vllm_env``: the pinned vLLM build owns the CUTLASS extension.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

BACKEND = "vllm_cutlass_fp8"
# CutlassFP8ScaledMMLinearKernel pads k and n of the weight to this multiple.
ALIGNMENT = 16
# bf16 output rounding (2^-8 relative) plus accumulation-order differences.
REL_FROBENIUS_TOL = 1e-2
# Non-power-of-two per-tensor scales, so a dropped or swapped scale shows up.
_A_SCALE = 0.73
_W_SCALE = 1.13


def _validate(m: int, n: int, k: int, dtype: DType | str) -> tuple[int, int, int]:
    if DType.from_value(dtype) is not DType.FP8_E4M3:
        raise ValueError(f"{BACKEND} requires dtype=fp8_e4m3, got {dtype!r}")
    m, n, k = int(m), int(n), int(k)
    if m < 1 or n < 1 or k < 1:
        raise ValueError(f"{BACKEND} requires m, n, k >= 1, got m={m}, n={n}, k={k}")
    return m, n, k


def check_against_reference(torch: Any, a: Any, weight: Any, out: Any) -> float:
    """Relative Frobenius error of ``out`` against fp32 ``(sa·A) @ (sw·W)^T``."""

    ref = (a.float() * _A_SCALE) @ (weight.float() * _W_SCALE).t()
    if tuple(out.shape) != tuple(ref.shape):
        raise KernelLaunchFailed(
            f"{BACKEND} output shape {tuple(out.shape)} != reference {tuple(ref.shape)}"
        )
    err = ((out.float() - ref).norm() / ref.norm().clamp_min(1e-30)).item()
    if not err <= REL_FROBENIUS_TOL:
        raise KernelLaunchFailed(
            f"{BACKEND} output mismatch: relative Frobenius error {err:.3e} "
            f"> {REL_FROBENIUS_TOL:.0e}"
        )
    return err


def profile_single_gemm_vllm_cutlass_fp8(
    m: int,
    n: int,
    k: int,
    dtype: DType | str,
) -> ComputeMetrics:
    m, n, k = _validate(m, n, k, dtype)
    try:
        import torch
        from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
            CutlassFP8ScaledMMLinearKernel,
        )
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{BACKEND} requires the vLLM environment (vllm_env)") from exc

    try:
        generator = torch.Generator(device="cuda").manual_seed(0)
        a = torch.randn(m, k, device="cuda", generator=generator).to(torch.float8_e4m3fn)
        weight = torch.randn(n, k, device="cuda", generator=generator).to(torch.float8_e4m3fn)

        # The loader's layout: [k, n] column-major weight, 0-d FP32 scales.
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(weight.t(), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(
            torch.tensor(_W_SCALE, device="cuda", dtype=torch.float32), requires_grad=False
        )
        scale_a = torch.tensor(_A_SCALE, device="cuda", dtype=torch.float32)

        # The production kernel object without __init__: that only builds the
        # QuantFP8 activation quantizer, which this row excludes.
        kernel = CutlassFP8ScaledMMLinearKernel.__new__(CutlassFP8ScaledMMLinearKernel)
        kernel.layer_param_names = ["weight", "weight_scale", "input_scale", "input_scale_ub"]
        kernel.logical_output_size = None
        kernel.process_weights_after_loading(layer)
        w = layer.weight
        w_s = layer.weight_scale

        def run_once() -> Any:
            return kernel.apply_scaled_mm(
                A=a,
                B=w,
                out_dtype=torch.bfloat16,
                As=scale_a,
                Bs=w_s,
                bias=None,
                output_shape=[m, n],
            )

        out = run_once()
        torch.cuda.synchronize()
        check_against_reference(torch, a, weight, out)
        del out

        # One CUTLASS launch, plus pad/slice copies on unaligned shapes: sum
        # every launch of the logical call.
        time_ms = Timer.cupti(run_once, warmup=3)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    except (ProfilerNotImplemented, KernelLaunchFailed):
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError(f"{BACKEND} ran out of memory") from exc
        raise KernelLaunchFailed(f"{BACKEND} failed: {exc}") from exc

    # One read of each E4M3 operand and the bf16 output write.
    bytes_accessed = m * k + n * k + 2 * m * n
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=2 * m * n * k / elapsed_s / 1e12,
        memory_bandwidth_gbps=bytes_accessed / elapsed_s / 1e9,
        energy_j=float(energy_j),
    )


__all__ = ["check_against_reference", "profile_single_gemm_vllm_cutlass_fp8"]
