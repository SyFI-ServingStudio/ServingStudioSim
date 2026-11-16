"""DeepSeek-V4.x block-32 FP8 dense linear as SGLang serves it on SM120: the
``sglang_triton_block32_fp8`` backend of ``single_gemm``.

DeepSeek-V4.1-Flash stores its dense linears as FP8 E4M3 with one UE8M0
(power-of-two) scale per 32x32 weight block (``quantization_config``:
``weight_block_size = [32, 32]``, ``scale_fmt = "ue8m0"``). SGLang v0.5.23
``dispatch_w8a8_block_fp8_linear`` (``srt/layers/quantization/fp8_utils.py``)
sends every non-128 block layout to
``partial(triton_w8a8_block_fp8_linear, act_scale_ue8m0=True)`` unless DeepGEMM
serves block-32 (SM90 by default, SM100 on request) -- so on SM12x (RTX PRO
6000) each linear is:

1. ``sglang_per_token_group_quant_fp8(x, 32, scale_ue8m0=True)``: the bf16
   activation to FP8 with one power-of-two scale per 32 elements of a row;
2. ``w8a8_block_fp8_matmul_triton``: the Triton ``_w8a8_block_fp8_matmul``
   GEMM, bf16 output.

The BF16 weight-expansion shortcut in the same function needs
``_block_fp8_bf16_weight``, which ``Fp8LinearMethod`` builds only on SM90
(``SGLANG_OPT_HOPPER_BLOCK_FP8_BF16``), so it is not part of this path. The
row times the whole callable (both launches): that is the sequence the 2026-11-05
SGLang trace on 4x RTX PRO 6000 shows for every q/kv-a, wq_b, wo_b, shared-expert
and indexer projection. Runs in ``sglang_env``.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

BLOCK = 32
# FP8 activation (per-32 power-of-two scale) and FP8 weight against an fp32
# reference on the dequantized weight: e4m3 rounding of the activation dominates.
REL_FROBENIUS_TOL = 6e-2
_CHECK_ROWS = 256


def _validate(m: int, n: int, k: int, dtype: DType | str) -> tuple[int, int, int]:
    if DType.from_value(dtype) is not DType.FP8_E4M3:
        raise ValueError(f"sglang_triton_block32_fp8 requires dtype=fp8_e4m3, got {dtype!r}")
    m, n, k = int(m), int(n), int(k)
    if m < 1 or n < 1 or k < 1:
        raise ValueError(f"sglang_triton_block32_fp8 requires positive m, n, k, got m={m} n={n} k={k}")
    if k % BLOCK:
        raise ValueError(f"sglang_triton_block32_fp8 requires k to be a multiple of {BLOCK}, got k={k}")
    return m, n, k


def _load_runtime() -> tuple[Any, Any]:
    try:
        import torch
        from sglang.srt.layers.quantization.fp8_utils import triton_w8a8_block_fp8_linear
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "sglang_triton_block32_fp8 requires the SGLang environment (sglang_env)"
        ) from exc
    return torch, triton_w8a8_block_fp8_linear


def profile_single_gemm_sglang_triton_block32_fp8(
    m: int, n: int, k: int, dtype: DType | str
) -> ComputeMetrics:
    m, n, k = _validate(m, n, k, dtype)
    torch, linear = _load_runtime()
    try:
        g = torch.Generator(device="cuda").manual_seed(0)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
        weight = (torch.randn(n, k, device="cuda", generator=g) * 0.5).to(torch.float8_e4m3fn)
        nb, kb = -(-n // BLOCK), k // BLOCK
        # UE8M0 checkpoints store fp32 powers of two (scale_fmt = "ue8m0").
        exponents = torch.randint(-9, -4, (nb, kb), device="cuda", generator=g).float()
        weight_scale = torch.pow(2.0, exponents)

        def run_once() -> Any:
            return linear(x, weight, [BLOCK, BLOCK], weight_scale, act_scale_ue8m0=True)

        out = run_once()
        torch.cuda.synchronize()
        rows = torch.linspace(0, m - 1, min(m, _CHECK_ROWS), device="cuda").round().long().unique()
        w_deq = weight.float() * weight_scale.repeat_interleave(BLOCK, 0)[:n].repeat_interleave(BLOCK, 1)
        ref = x[rows].float() @ w_deq.t()
        err = ((out[rows].float() - ref).norm() / ref.norm().clamp_min(1e-30)).item()
        if not err <= REL_FROBENIUS_TOL:
            raise KernelLaunchFailed(
                f"sglang_triton_block32_fp8 output mismatch: relative Frobenius error {err:.3e} > "
                f"{REL_FROBENIUS_TOL:.0e}"
            )
        del out, ref, w_deq
        # Two launches (activation quant + GEMM): sum every launch.
        time_ms = Timer.cupti(run_once, warmup=3)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    except (ProfilerNotImplemented, KernelLaunchFailed):
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError("sglang_triton_block32_fp8 ran out of memory") from exc
        raise KernelLaunchFailed(f"sglang_triton_block32_fp8 failed: {exc}") from exc

    # bf16 activation read, fp8 activation write + read, fp8 weight + scales, bf16 output.
    bytes_accessed = 2 * m * k + 2 * m * k + n * k + 4 * (n * k) / (BLOCK * BLOCK) + 2 * m * n
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=2 * m * n * k / elapsed_s / 1e12,
        memory_bandwidth_gbps=bytes_accessed / elapsed_s / 1e9,
        energy_j=float(energy_j),
    )


__all__ = ["profile_single_gemm_sglang_triton_block32_fp8"]
