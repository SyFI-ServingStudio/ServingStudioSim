"""NVFP4 weight-only (W4A16) dense linear through Marlin: the ``vllm_marlin_nvfp4``
backend of ``single_gemm``.

GPUs without FP4 tensor cores (Hopper, Ada, Ampere) serve ModelOpt NVFP4 linears
with ``MarlinNvFp4LinearKernel`` (``model_executor/kernels/linear/nvfp4/marlin.py``):
at load time ``prepare_fp4_layer_for_marlin`` repacks the packed E2M1 weight and
its per-16 E4M3 scales into Marlin's tile layout, and every forward calls
``apply_fp4_marlin_linear`` on the BF16 activation (no activation quantization:
the input stays BF16). Marlin splits a large ``m`` over several launches (1 at
decode sizes, 2-3 for mixed steps in the captures); the CUPTI timer sums every
launch of the call, so a row is the whole linear. Measured on the ServingStudio
vLLM fork on H100 and H200 (2026-11-03 captures), this is 50-77% of Qwen3.8-27B's
GPU time on Hopper.

The runner builds the layer the same way (same attribute names and shapes as the
ModelOpt loader leaves them), runs vLLM's own repack, and times
``apply_fp4_marlin_linear``. The timed call includes the input padding vLLM does
when ``k`` is not a Marlin tile multiple; the repack is load-time and is not part
of the row. Runs in ``vllm_env``.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

GROUP_SIZE = 16
# bf16 activations and output; Marlin dequantizes FP4 to bf16 in-register.
REL_FROBENIUS_TOL = 2e-2
# E2M1 magnitudes indexed by the low 3 bits of a nibble; bit 3 is the sign.
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_GLOBAL_SCALE = 0.37
_CHECK_ROWS = 256


def _validate(m: int, n: int, k: int, dtype: DType | str) -> tuple[int, int, int]:
    if DType.from_value(dtype) is not DType.NVFP4_E2M1:
        raise ValueError(f"vllm_marlin_nvfp4 requires dtype=nvfp4_e2m1, got {dtype!r}")
    m, n, k = int(m), int(n), int(k)
    if m < 1 or n < 1 or k < 1:
        raise ValueError(f"vllm_marlin_nvfp4 requires positive m, n, k, got m={m} n={n} k={k}")
    if k % GROUP_SIZE:
        raise ValueError(f"vllm_marlin_nvfp4 requires k to be a multiple of {GROUP_SIZE}, got k={k}")
    return m, n, k


def _load_runtime() -> tuple[Any, Any, Any]:
    try:
        import torch
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            apply_fp4_marlin_linear,
            prepare_fp4_layer_for_marlin,
        )
    except ImportError as exc:
        raise ProfilerNotImplemented("vllm_marlin_nvfp4 requires the vLLM environment (vllm_env)") from exc
    return torch, prepare_fp4_layer_for_marlin, apply_fp4_marlin_linear


def _dequantize(torch: Any, packed: Any, scales: Any, global_scale: float, n: int, k: int) -> Any:
    """fp32 [n, k] weight from the ModelOpt layout: two E2M1 nibbles per byte (low nibble
    = even column), one E4M3 scale per 16 columns, one FP32 global scale."""
    lut = torch.tensor(_E2M1, device=packed.device, dtype=torch.float32)
    lo, hi = (packed & 0x0F).long(), (packed >> 4).long()
    def nib(v: Any) -> Any:
        mag = lut[v & 0x7]
        return torch.where((v & 0x8) != 0, -mag, mag)
    w = torch.stack((nib(lo), nib(hi)), dim=-1).reshape(n, k)
    return w * scales.float().repeat_interleave(GROUP_SIZE, dim=1) * global_scale


def profile_single_gemm_vllm_marlin_nvfp4(m: int, n: int, k: int, dtype: DType | str) -> ComputeMetrics:
    m, n, k = _validate(m, n, k, dtype)
    torch, prepare, apply = _load_runtime()
    try:
        g = torch.Generator(device="cuda").manual_seed(0)
        packed = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8, generator=g)
        scales = (torch.rand(n, k // GROUP_SIZE, device="cuda", generator=g) * 3 + 0.25).to(torch.float8_e4m3fn)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
        reference_w = _dequantize(torch, packed, scales, _GLOBAL_SCALE, n, k)

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(packed, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scales, requires_grad=False)
        layer.weight_global_scale = torch.nn.Parameter(
            torch.tensor(_GLOBAL_SCALE, device="cuda", dtype=torch.float32), requires_grad=False
        )
        layer.output_size_per_partition = n
        layer.input_size_per_partition = k
        layer.params_dtype = torch.bfloat16
        prepare(layer)

        def run_once() -> Any:
            return apply(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                weight_global_scale=layer.weight_global_scale,
                workspace=None,
                size_n=n,
                size_k=k,
            )

        out = run_once()
        torch.cuda.synchronize()
        # Check a spread of at most _CHECK_ROWS output rows: a full fp32 reference of the
        # lm_head at m=65536 (65536 x 248320) would not fit beside the output.
        rows = torch.linspace(0, m - 1, min(m, _CHECK_ROWS), device="cuda").round().long().unique()
        ref = x[rows].float() @ reference_w.t()
        err = ((out[rows].float() - ref).norm() / ref.norm().clamp_min(1e-30)).item()
        if not err <= REL_FROBENIUS_TOL:
            raise KernelLaunchFailed(
                f"vllm_marlin_nvfp4 output mismatch: relative Frobenius error {err:.3e} > {REL_FROBENIUS_TOL:.0e}"
            )
        del out, ref, reference_w
        # Marlin may launch a reduction alongside the GEMM; sum every launch.
        time_ms = Timer.cupti(run_once, warmup=3)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    except (ProfilerNotImplemented, KernelLaunchFailed):
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError("vllm_marlin_nvfp4 ran out of memory") from exc
        raise KernelLaunchFailed(f"vllm_marlin_nvfp4 failed: {exc}") from exc

    # bf16 activation in, packed FP4 weight + E4M3 scales, bf16 output.
    bytes_accessed = 2 * m * k + n * k * (1 / 2 + 1 / GROUP_SIZE) + 2 * m * n
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=2 * m * n * k / elapsed_s / 1e12,
        memory_bandwidth_gbps=bytes_accessed / elapsed_s / 1e9,
        energy_j=float(energy_j),
    )


__all__ = ["profile_single_gemm_vllm_marlin_nvfp4"]
