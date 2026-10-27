"""NVFP4 weight-only (W4A16) dense linear through vLLM's Marlin kernel: the
``vllm_marlin_nvfp4`` backend of ``single_gemm``.

On GPUs without FP4 tensor cores, vLLM's NVFP4 kernel selection
(``_POSSIBLE_NVFP4_KERNELS`` in ``model_executor/kernels/linear/__init__.py``)
falls through to ``MarlinNvFp4LinearKernel`` (``kernels/linear/nvfp4/marlin.py``),
whose ``is_supported`` is ``is_fp4_marlin_supported()`` (compute capability
7.5+). The weight stays NVFP4 (E2M1 values, one E4M3 scale per 16 k-elements,
the ModelOpt per-tensor global scale ``weight_scale_2 = amax / 2688``); the
activation is NOT quantized: Marlin dequantizes the weight in registers and
multiplies the bf16 activation directly, bf16 out.

The runner reproduces the production weight path once, outside timing:
checkpoint-layout operands (``[n, k/2]`` packed uint8, ``[n, k/16]`` E4M3
scales, 0-d FP32 global scale, as ``KNvfp4Static.process`` leaves them), then
``prepare_fp4_layer_for_marlin`` (tile padding, ``gptq_marlin_repack``, scale
permutation to Marlin's S0E5M3 format and the global-scale exponent bias), as
``MarlinNvFp4LinearKernel.process_weights_after_loading`` does at load. It
times ``apply_fp4_marlin_linear``, the body of ``apply_weights``, with every
launch summed: ``marlin_gemm`` splits a large ``m`` into one launch per slice of
at most ``16 · 64`` rows when ``n > 4096`` (``max_par`` in ``marlin.cu``), and
the activation pad / output slice copies run when ``k`` or ``n`` needs tile
padding. ``apply_weights`` passes ``workspace=None``, which resolves to the
worker's persistent zero-initialized lock buffer (Marlin resets it after each
call); the runner allocates that buffer once and passes it, so the timed call
does not pay a per-call ``torch.zeros`` vLLM never issues.

Runs in ``vllm_env``: the pinned vLLM build owns the Marlin extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

BACKEND = "vllm_marlin_nvfp4"
GROUP_SIZE = 16
_FP4_MAX = 6.0
_E4M3_MAX = 448.0
# E2M1 magnitudes by 3-bit code, and the rounding midpoints between them.
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
# bf16 activation/output rounding plus accumulation-order differences.
REL_FROBENIUS_TOL = 1e-2
# Output rows checked against the fp32 oracle (spread over every Marlin m slice).
_CHECK_ROWS = 512
# Weight rows quantized / dequantized per chunk, to bound fp32 temporaries.
_ROW_CHUNK = 8192


def _validate(m: int, n: int, k: int, dtype: DType | str) -> tuple[int, int, int]:
    if DType.from_value(dtype) is not DType.NVFP4_E2M1:
        raise ValueError(f"{BACKEND} requires dtype=nvfp4_e2m1 (the weight format), got {dtype!r}")
    m, n, k = int(m), int(n), int(k)
    if m < 1 or n < 1 or k < 1:
        raise ValueError(f"{BACKEND} requires m, n, k >= 1, got m={m}, n={n}, k={k}")
    # The checkpoint stores one E4M3 scale per 16 k-elements.
    if k % GROUP_SIZE:
        raise ValueError(f"{BACKEND} requires k divisible by {GROUP_SIZE}, got k={k}")
    return m, n, k


def quantize_nvfp4_weight(torch: Any, w: Any) -> tuple[Any, Any, Any]:
    """ModelOpt-style NVFP4 weight: ``(packed [n, k/2] uint8, scales [n, k/16]
    E4M3, global scale 0-d FP32)``, independent of any vLLM quantizer.

    Element ``2j`` sits in the low nibble of byte ``j``; a code is
    ``sign << 3 | magnitude index`` into ``_E2M1_VALUES``. The global scale is
    the dequantization multiplier ``amax / (6 · 448)`` (``weight_scale_2``).
    """
    n, k = w.shape
    global_scale = (w.float().abs().amax() / (_FP4_MAX * _E4M3_MAX)).to(torch.float32)
    midpoints = torch.tensor(_E2M1_MIDPOINTS, device=w.device, dtype=torch.float32)
    packed = torch.empty(n, k // 2, dtype=torch.uint8, device=w.device)
    scales = torch.empty(n, k // GROUP_SIZE, dtype=torch.float8_e4m3fn, device=w.device)
    for r0 in range(0, n, _ROW_CHUNK):
        blocks = w[r0 : r0 + _ROW_CHUNK].float().view(-1, k // GROUP_SIZE, GROUP_SIZE)
        block_scale = (blocks.abs().amax(-1) / _FP4_MAX / global_scale).to(torch.float8_e4m3fn)
        denom = (block_scale.float() * global_scale).unsqueeze(-1)
        scaled = torch.where(denom > 0, blocks / denom, torch.zeros_like(blocks))
        idx = torch.bucketize(scaled.abs().clamp(max=_FP4_MAX), midpoints).to(torch.uint8)
        codes = (idx | ((scaled < 0).to(torch.uint8) << 3)).view(blocks.shape[0], k)
        packed[r0 : r0 + _ROW_CHUNK] = codes[:, 0::2] | (codes[:, 1::2] << 4)
        scales[r0 : r0 + _ROW_CHUNK] = block_scale
    return packed, scales, global_scale


def dequantize_nvfp4_rows(torch: Any, packed: Any, scales: Any, global_scale: Any) -> Any:
    """fp32 ``[rows, k]`` weight decoded from ``quantize_nvfp4_weight``'s layout."""
    rows, half_k = packed.shape
    k = half_k * 2
    values = torch.tensor(_E2M1_VALUES, device=packed.device, dtype=torch.float32)
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).view(rows, k).long()
    mag = values[codes & 0x7]
    signed = torch.where((codes & 0x8) != 0, -mag, mag)
    block = scales.float().repeat_interleave(GROUP_SIZE, dim=1)
    return signed * block * global_scale


def check_against_reference(
    torch: Any, x: Any, packed: Any, scales: Any, global_scale: Any, out: Any
) -> float:
    """Relative Frobenius error of ``out`` on a spread of rows against the fp32
    ``x @ dequant(W)^T`` oracle."""
    m = x.shape[0]
    n = packed.shape[0]
    if tuple(out.shape) != (m, n):
        raise KernelLaunchFailed(f"{BACKEND} output shape {tuple(out.shape)} != {(m, n)}")
    rows = torch.unique(
        torch.cat(
            [
                torch.linspace(0, m - 1, min(m, _CHECK_ROWS), device=x.device).round().long(),
                torch.tensor([m - 1], device=x.device),
            ]
        )
    )
    xs = x[rows].float()
    num = torch.zeros((), device=x.device, dtype=torch.float64)
    den = torch.zeros((), device=x.device, dtype=torch.float64)
    for r0 in range(0, n, _ROW_CHUNK):
        w = dequantize_nvfp4_rows(
            torch, packed[r0 : r0 + _ROW_CHUNK], scales[r0 : r0 + _ROW_CHUNK], global_scale
        )
        ref = xs @ w.t()
        got = out[rows, r0 : r0 + _ROW_CHUNK].float()
        num += (got - ref).double().pow(2).sum()
        den += ref.double().pow(2).sum()
    err = (num.sqrt() / den.sqrt().clamp_min(1e-30)).item()
    if not err <= REL_FROBENIUS_TOL:
        raise KernelLaunchFailed(
            f"{BACKEND} output mismatch: relative Frobenius error {err:.3e} "
            f"> {REL_FROBENIUS_TOL:.0e}"
        )
    return err


@dataclass
class _MarlinWeight:
    """One ``(n, k)`` weight: the checkpoint-layout tensors the oracle decodes
    and the layer after Marlin's load-time repack."""

    n: int
    k: int
    packed: Any
    scales: Any
    global_scale: Any
    layer: Any


# The weight does not depend on m: a sweep over m reuses the last (n, k) weight
# instead of re-quantizing and re-packing it (seconds for an LM head) per point.
_WEIGHT_CACHE: list[_MarlinWeight] = []


def _prepared_weight(torch: Any, n: int, k: int, prepare: Any) -> _MarlinWeight:
    if _WEIGHT_CACHE and (_WEIGHT_CACHE[0].n, _WEIGHT_CACHE[0].k) == (n, k):
        return _WEIGHT_CACHE[0]
    _WEIGHT_CACHE.clear()
    torch.cuda.empty_cache()
    generator = torch.Generator(device="cuda").manual_seed(0)
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda", generator=generator)
    packed, scales, global_scale = quantize_nvfp4_weight(torch, w)
    del w

    # The layer as KNvfp4Static.process leaves it, then Marlin's load step
    # (MarlinNvFp4LinearKernel.process_weights_after_loading).
    layer = torch.nn.Module()
    layer.output_size_per_partition = n
    layer.input_size_per_partition = k
    layer.params_dtype = torch.bfloat16
    layer.weight = torch.nn.Parameter(packed.clone(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scales.clone(), requires_grad=False)
    layer.weight_global_scale = torch.nn.Parameter(global_scale.clone(), requires_grad=False)
    prepare(layer)
    torch.cuda.synchronize()
    _WEIGHT_CACHE.append(_MarlinWeight(n, k, packed, scales, global_scale, layer))
    return _WEIGHT_CACHE[0]


def profile_single_gemm_vllm_marlin_nvfp4(
    m: int,
    n: int,
    k: int,
    dtype: DType | str,
) -> ComputeMetrics:
    m, n, k = _validate(m, n, k, dtype)
    try:
        import torch
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            MARLIN_MAX_BLOCKS_PER_SM,
            marlin_make_workspace_new,
        )
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            apply_fp4_marlin_linear,
            prepare_fp4_layer_for_marlin,
        )
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{BACKEND} requires the vLLM environment (vllm_env)") from exc

    try:
        weight = _prepared_weight(torch, n, k, prepare_fp4_layer_for_marlin)
        generator = torch.Generator(device="cuda").manual_seed(1)
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda", generator=generator)
        layer = weight.layer
        workspace = marlin_make_workspace_new(x.device, MARLIN_MAX_BLOCKS_PER_SM)

        def run_once() -> Any:
            return apply_fp4_marlin_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                weight_global_scale=layer.weight_global_scale,
                workspace=workspace,
                size_n=n,
                size_k=k,
                bias=None,
            )

        out = run_once()
        torch.cuda.synchronize()
        check_against_reference(torch, x, weight.packed, weight.scales, weight.global_scale, out)
        del out

        # Large m issues one marlin launch per <=1024-row slice (n > 4096):
        # the CUPTI sum prices the whole apply call.
        time_ms = Timer.cupti(run_once, warmup=3)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    except (ProfilerNotImplemented, KernelLaunchFailed):
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError(f"{BACKEND} ran out of memory") from exc
        raise KernelLaunchFailed(f"{BACKEND} failed: {exc}") from exc

    bytes_accessed = marlin_nvfp4_logical_bytes(m, n, k)
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=2 * m * n * k / elapsed_s / 1e12,
        memory_bandwidth_gbps=bytes_accessed / elapsed_s / 1e9,
        energy_j=float(energy_j),
    )


def marlin_nvfp4_logical_bytes(m: int, n: int, k: int) -> int:
    """bf16 activation read, packed FP4 weight (1/2 byte) with one 1-byte scale
    per 16 k-elements, and the bf16 output write. Tile padding is excluded."""
    return 2 * m * k + n * k // 2 + n * k // GROUP_SIZE + 2 * m * n


__all__ = [
    "check_against_reference",
    "dequantize_nvfp4_rows",
    "marlin_nvfp4_logical_bytes",
    "profile_single_gemm_vllm_marlin_nvfp4",
    "quantize_nvfp4_weight",
]
