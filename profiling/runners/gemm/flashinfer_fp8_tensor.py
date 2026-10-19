"""Static per-tensor FP8 dense linear on SM10x: the ``flashinfer_fp8_tensor``
backend of ``single_gemm``.

vLLM serves ModelOpt ``FP8`` linears (one FP32 scale for the weight, one static
FP32 scale for the activation) with ``FlashInferFP8ScaledMMLinearKernel``, the
first entry of ``_POSSIBLE_FP8_KERNELS`` on CUDA
(``model_executor/kernels/linear/__init__.py``; ``is_supported`` needs compute
capability 100+, ``can_implement`` per-tensor scales on both operands). Its
``apply_scaled_mm`` calls ``vllm.utils.flashinfer.flashinfer_scaled_fp8_mm``,
which issues ``flashinfer.bmm_fp8(a, b, scale_a, scale_b, out_dtype, "auto")``
on the row-major E4M3 activation and the column-major ``weight.t()`` view the
ModelOpt loader stores. This runner times that call.

``backend="auto"`` lets FlashInfer's AutoTuner choose among the CUTLASS SM10x,
cuBLASLt and cuDNN runners and their tactics. vLLM tunes them at kernel warmup
(``enable_flashinfer_autotune`` is on from ``-O1``) once per hybrid token bucket
and, at serving time, maps a runtime ``m`` up to its bucket
(``map_to_hybrid_bucket_uncapped``). The runner reproduces both halves: it
tunes the hybrid buckets up to ``m``'s own bucket (persisted through
``autotune_cached``), then times the call outside the tuning context, where the
lookup takes the serving path.

The activation quantization (``QuantFP8``, static per tensor) runs before this
call, or is fused into the preceding norm; it is not part of this row.
Runs in ``vllm_env``: the pinned vLLM image owns FlashInfer.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.autotune_cache import autotune_cached
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

# cuBLASLt FP8 needs 16-byte aligned leading dimensions on both operands.
ALIGNMENT = 16
# bf16 output rounding (2^-8 relative) plus accumulation-order differences.
REL_FROBENIUS_TOL = 1e-2
# Non-power-of-two per-tensor scales, so a dropped or swapped scale shows up.
_A_SCALE = 0.73
_W_SCALE = 1.13


def _validate(m: int, n: int, k: int, dtype: DType | str) -> tuple[int, int, int]:
    if DType.from_value(dtype) is not DType.FP8_E4M3:
        raise ValueError(f"flashinfer_fp8_tensor requires dtype=fp8_e4m3, got {dtype!r}")
    m, n, k = int(m), int(n), int(k)
    if m < 1:
        raise ValueError(f"flashinfer_fp8_tensor requires m >= 1, got m={m}")
    if n < 1 or k < 1 or n % ALIGNMENT or k % ALIGNMENT:
        raise ValueError(
            f"flashinfer_fp8_tensor requires n and k to be positive multiples of {ALIGNMENT}, "
            f"got n={n}, k={k}"
        )
    return m, n, k


def _load_runtime() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from flashinfer.autotuner import autotune
        from flashinfer.fused_moe.utils import (
            get_hybrid_num_tokens_buckets,
            map_to_hybrid_bucket_uncapped,
        )
        from vllm.utils.flashinfer import flashinfer_scaled_fp8_mm
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "flashinfer_fp8_tensor requires the vLLM environment (vllm_env)"
        ) from exc

    def serving_buckets(m: int) -> tuple[int, ...]:
        """The warmup's hybrid buckets, up to the one a runtime ``m`` maps to."""
        return tuple(get_hybrid_num_tokens_buckets(map_to_hybrid_bucket_uncapped(m)))

    return torch, autotune, serving_buckets, flashinfer_scaled_fp8_mm


def check_against_reference(torch: Any, a: Any, weight: Any, out: Any) -> float:
    """Relative Frobenius error of ``out`` against fp32 ``(sa·A) @ (sw·W)^T``."""

    ref = (a.float() * _A_SCALE) @ (weight.float() * _W_SCALE).t()
    err = ((out.float() - ref).norm() / ref.norm().clamp_min(1e-30)).item()
    if not err <= REL_FROBENIUS_TOL:
        raise KernelLaunchFailed(
            f"flashinfer_fp8_tensor output mismatch: relative Frobenius error {err:.3e} "
            f"> {REL_FROBENIUS_TOL:.0e}"
        )
    return err


def profile_single_gemm_flashinfer_fp8_tensor(
    m: int,
    n: int,
    k: int,
    dtype: DType | str,
) -> ComputeMetrics:
    m, n, k = _validate(m, n, k, dtype)
    torch, autotune, serving_buckets, scaled_fp8_mm = _load_runtime()
    try:
        generator = torch.Generator(device="cuda").manual_seed(0)
        a = torch.randn(m, k, device="cuda", generator=generator).to(torch.float8_e4m3fn)
        weight = torch.randn(n, k, device="cuda", generator=generator).to(torch.float8_e4m3fn)
        # The ModelOpt loader stores the [n, k] weight transposed: a [k, n]
        # column-major view, which is what bmm_fp8 expects for its B operand.
        weight_t = weight.t()
        # 0-d scales, as the loader leaves them (``input_scale.max()``,
        # ``weight_scale.max()``); their shape is part of FlashInfer's tuning key.
        scale_a = torch.tensor(_A_SCALE, device="cuda", dtype=torch.float32)
        scale_w = torch.tensor(_W_SCALE, device="cuda", dtype=torch.float32)

        def run_once() -> Any:
            return scaled_fp8_mm(a, weight_t, scale_a, scale_w, torch.bfloat16)

        with (
            autotune_cached(autotune, "single_gemm.flashinfer_fp8_tensor"),
            autotune(tuning_buckets=serving_buckets(m)),
        ):
            run_once()
        out = run_once()
        torch.cuda.synchronize()
        check_against_reference(torch, a, weight, out)
        del out

        # bmm_fp8 is one launch for the CUTLASS and cuBLASLt runners; sum every
        # launch so a runner that splits the GEMM is priced whole.
        time_ms = Timer.cupti(run_once, warmup=3)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    except (ProfilerNotImplemented, KernelLaunchFailed):
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError("flashinfer_fp8_tensor ran out of memory") from exc
        raise KernelLaunchFailed(f"flashinfer_fp8_tensor failed: {exc}") from exc

    # One read of each E4M3 operand and the bf16 output write.
    bytes_accessed = m * k + n * k + 2 * m * n
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=2 * m * n * k / elapsed_s / 1e12,
        memory_bandwidth_gbps=bytes_accessed / elapsed_s / 1e9,
        energy_j=float(energy_j),
    )


__all__ = ["check_against_reference", "profile_single_gemm_flashinfer_fp8_tensor"]
