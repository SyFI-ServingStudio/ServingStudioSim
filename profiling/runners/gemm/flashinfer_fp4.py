"""NVFP4 dense-linear GEMM runner: the ``flashinfer_cutedsl`` backend of ``single_gemm``.

L1a-only: allocate tensors, time one logical GEMM, return metrics. It times
what vLLM's ``FlashInferCuteDslNvFp4LinearKernel.apply_weights`` (the default
NVFP4 W4A4 linear kernel on SM10x, ``model_executor/kernels/linear/nvfp4/
flashinfer.py`` at the alignment fork's commit) runs after the activation
quantization: ``pad_nvfp4_activation_for_cutlass`` ->
``flashinfer_scaled_fp4_mm(..., backend="cute-dsl")`` -> ``slice_nvfp4_output``.

``scaled_fp4_quant`` produces the timed GEMM's inputs and is its own kernel kind
(``nvfp4_quant``), so it runs once outside the timed closure. The padding and
slicing calls are no-ops, and launch nothing, when ``k`` and ``n`` are multiples
of 32; otherwise they launch the same copy kernels vLLM pays, and the CUPTI sum
counts them.

vLLM skips FlashInfer autotuning for the ``fp4_gemm`` op when this kernel is
selected (``warmup/kernel_warmup.py::_flashinfer_autotune_skip_ops``), so
``mm_fp4`` takes its analytical heuristic tactic; this process never enters
autotune mode either.

Runs in ``vllm_env``: the pinned vLLM image owns FlashInfer and its CuTe-DSL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

GROUP_SIZE = 16
# NVFP4 global-scale convention: a tensor's amax maps to FP4 max (6) times the
# E4M3 scale max (448).
_FP4_MAX = 6.0
_E4M3_MAX = 448.0


@dataclass
class Nvfp4LinearOperands:
    """Inputs of the timed call.

    ``x_fp4`` / ``x_blockscale`` are vLLM's ``scaled_fp4_quant`` output with the
    swizzled 128x4 scale layout; ``weight`` / ``weight_scale`` are the checkpoint
    tensors after ``FlashInferCuteDslNvFp4LinearKernel.process_weights_after_loading``
    (swizzled scales, N/K padded to 32).
    """

    x_fp4: Any
    x_blockscale: Any
    weight: Any
    weight_scale: Any
    weights_padding_cols: int
    alpha: Any


def prepare_operands(m: int, n: int, k: int, *, seed: int = 0) -> Nvfp4LinearOperands:
    """Build NVFP4 activation and weight operands the way vLLM produces them."""
    import torch
    from vllm._custom_ops import scaled_fp4_quant
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        pad_nvfp4_weight_for_cutlass,
        swizzle_blockscale,
    )

    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda", generator=generator)
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda", generator=generator)

    def global_scale(t: Any) -> Any:
        return (_E4M3_MAX * _FP4_MAX / t.float().abs().amax()).to(torch.float32)

    x_gs = global_scale(x)
    w_gs = global_scale(w)

    # Checkpoint-style weight: packed FP4 plus row-major E4M3 group scales.
    w_fp4, w_scale = scaled_fp4_quant(w, w_gs, is_sf_swizzled_layout=False)
    # FlashInferCuteDslNvFp4LinearKernel.process_weights_after_loading.
    weight_scale = swizzle_blockscale(w_scale)
    weight, weights_padding_cols = pad_nvfp4_weight_for_cutlass(w_fp4)

    # apply_weights' activation quantization (the separate nvfp4_quant kind).
    x_fp4, x_blockscale = scaled_fp4_quant(
        x, x_gs, is_sf_swizzled_layout=True, backend="flashinfer-cutedsl"
    )

    return Nvfp4LinearOperands(
        x_fp4=x_fp4,
        x_blockscale=x_blockscale,
        weight=weight,
        weight_scale=weight_scale,
        weights_padding_cols=int(weights_padding_cols),
        alpha=(1.0 / (x_gs * w_gs)).reshape(()).to(torch.float32),
    )


def make_apply(ops: Nvfp4LinearOperands, n: int) -> Any:
    """The post-quantization tail of vLLM's ``apply_weights``, as one closure."""
    import torch
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        pad_nvfp4_activation_for_cutlass,
        slice_nvfp4_output,
    )
    from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm

    def apply() -> Any:
        x_fp4 = pad_nvfp4_activation_for_cutlass(ops.x_fp4, ops.weights_padding_cols)
        out = flashinfer_scaled_fp4_mm(
            x_fp4,
            ops.weight,
            ops.x_blockscale,
            ops.weight_scale,
            ops.alpha,
            torch.bfloat16,
            backend="cute-dsl",
        )
        return slice_nvfp4_output(out, n)

    return apply


def _validate(m: int, n: int, k: int, dtype: DType | str) -> tuple[int, int, int]:
    dtype = DType.from_value(dtype)
    if dtype is not DType.NVFP4_E2M1:
        raise ValueError(
            f"flashinfer_cutedsl single-GEMM is NVFP4-only but got dtype={dtype.value}"
        )
    m, n, k = int(m), int(n), int(k)
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError(f"m, n, k must be positive, got m={m}, n={n}, k={k}")
    # scaled_fp4_quant groups the reduction dimension in 16-element blocks.
    if k % GROUP_SIZE != 0:
        raise ValueError(f"NVFP4 requires k divisible by {GROUP_SIZE}, got k={k}")
    return m, n, k


def profile_single_gemm(m: int, n: int, k: int, dtype: DType | str) -> ComputeMetrics:
    m, n, k = _validate(m, n, k, dtype)
    try:
        import torch
        import vllm.utils.flashinfer  # noqa: F401
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "the vLLM environment (vllm_env) is required for the flashinfer_cutedsl GEMM"
        ) from exc

    try:
        ops = prepare_operands(m, n, k)
        apply = make_apply(ops, n)
        # The first call JIT-compiles the CuTe-DSL kernel for this m bucket's
        # heuristic tactic.
        apply()
        torch.cuda.synchronize()

        # mm_fp4's cute-dsl path is one launch; padding/slicing copies, when the
        # shape needs them, are part of the same logical linear and are summed.
        time_ms = Timer.cupti(apply)
        energy_j = Energy.perf(apply, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError("flashinfer_cutedsl NVFP4 GEMM ran out of CUDA memory") from exc
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc

    flops = 2 * m * n * k
    tflops = (flops / (time_ms / 1000.0)) / 1e12 if time_ms > 0 else 0.0
    bytes_accessed = nvfp4_gemm_logical_bytes(m, n, k)
    bandwidth_gbps = (bytes_accessed / (time_ms / 1000.0)) / 1e9 if time_ms > 0 else 0.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=float(tflops),
        memory_bandwidth_gbps=float(bandwidth_gbps),
        energy_j=float(energy_j),
    )


def nvfp4_gemm_logical_bytes(m: int, n: int, k: int) -> int:
    """Packed FP4 A and B (1/2 byte each), one E4M3 scale per 16 k-elements of
    each, and the bf16 output. Padding and global scales are excluded."""
    return (m * k + n * k) // 2 + (m * k + n * k) // GROUP_SIZE + 2 * m * n
