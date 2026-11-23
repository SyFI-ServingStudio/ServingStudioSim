"""DeepInfer's grouped attention output projection (``wv``, DeepSeek-V4.1 ``wo_a``)
as one DeepGEMM MXFP8 einsum with an FP8 output — the
``deepgemm_mxfp8_einsum_grouped_o_proj_fp8_out`` backend of ``batched_gemm``.

Source: DeepInfer ``V4AttentionBase._wv_and_o_proj``
(``deepinfer/model_executor/models/v4_modules/attn.py``). With the fused
attention producer and 32-element casting it calls
``deep_gemm.fp8_einsum("bhr,hdr->bhd", (a, sfa), (w, sfw), (out, sfo), recipe=(1, 1, 32))``
with an FP8 output tuple (``use_fp8_einsum_output``), so the einsum's epilogue
casts the result to e4m3 with a UE8M0 scale per 32 output elements for the
next linear (``out_proj``). DeepGEMM runs it as one
``sm100_fp8_fp4_gemm_1d1d_impl`` launch with ``GemmType::Batched``, an e4m3
``cd`` type and the ``CastToFP8WithClamp`` epilogue (nsys capture template).
That is a different kernel instance from vLLM's bf16-output einsum, the
``deepgemm_mxfp8_einsum_grouped_o_proj`` backend.

Operands, per rank (``h = num_batches`` local ``wv`` groups):

- A ``[m, h, k]`` e4m3, contiguous, with packed UE8M0 scales per 32 along k in
  DeepInfer's TMA-aligned MN-major layout. The fused attention kernel writes it;
  the runner builds the same layout with DeepInfer's unfused fallback
  (``tile_kernels.quant.per_token_cast`` over ``[m, h * k]``, viewed per group),
  outside timing.
- B ``[h, n, k]`` e4m3 with the checkpoint's per-32x32-block UE8M0 scale,
  expanded per row and packed through ``transform_sf_into_required_layout``
  with ``num_groups = h``, as ``_init_wv_o_proj`` does. The load-time
  ``permute_wv_proj`` only relabels k on both operands and is omitted.
- Output ``[m, h, n]`` e4m3 plus a ``[m, h * n / 128]`` int32 packed scale,
  MN-major with the TMA-aligned token stride, allocated as DeepInfer does.

CUPTI time sums every launch of the call (``kernel_name=None``).
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

BACKEND = "deepgemm_mxfp8_einsum_grouped_o_proj_fp8_out"
MXFP8_BLOCK = 32
# Four UE8M0 bytes pack into one int32 scale word.
PACKED_SF_ELEMENTS = 4 * MXFP8_BLOCK
# The check reads at most this many tokens (rows are independent).
CHECK_ROWS = 1024
# e4m3 output rounding (2^-4 relative) dominates; accumulation order adds little.
REL_FROBENIUS_TOL = 5e-2


def _validate(
    num_batches: int, m: int, n: int, k: int, dtype: DType | str
) -> tuple[int, int, int, int]:
    if DType.from_value(dtype) is not DType.MXFP8_E4M3:
        raise ValueError(f"{BACKEND} requires dtype=mxfp8_e4m3, got {dtype!r}")
    num_batches, m, n, k = int(num_batches), int(m), int(n), int(k)
    if num_batches < 1 or m < 1:
        raise ValueError(f"{BACKEND} requires num_batches >= 1 and m >= 1")
    for name, value in (("n", n), ("k", k)):
        if value < PACKED_SF_ELEMENTS or value % PACKED_SF_ELEMENTS:
            raise ValueError(
                f"{BACKEND} requires {name} to be a positive multiple of "
                f"{PACKED_SF_ELEMENTS} (packed UE8M0 scale words), got {name}={value}"
            )
    return num_batches, m, n, k


def _load_runtime() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"torch is required for {BACKEND}") from exc
    try:
        import deep_gemm
        import tile_kernels
        from deep_gemm.utils import per_block_cast_to_fp8, per_token_cast_to_fp8
        from deepinfer.model_executor.layers.linear import get_fp8_cast_params
    except ImportError as exc:
        raise ProfilerNotImplemented(
            f"{BACKEND} requires DeepInfer and its vendored DeepGEMM (deepinfer_env)"
        ) from exc

    def quantize_activation(x: Any) -> tuple[Any, Any]:
        return tile_kernels.quant.per_token_cast(x, **get_fp8_cast_params())

    return torch, deep_gemm, quantize_activation, (per_token_cast_to_fp8, per_block_cast_to_fp8)


def _dequantize(torch: Any, data: Any, scale: Any, block_rows: int) -> Any:
    """fp32 value of a 2-D ``data`` from unpacked fp32 power-of-two scales,
    one per ``block_rows`` rows and 32 columns."""
    rows, cols = data.shape
    factor = scale.float().repeat_interleave(block_rows, dim=0)[:rows]
    factor = factor.repeat_interleave(MXFP8_BLOCK, dim=1)[:, :cols]
    return data.float() * factor


def _unpack_ue8m0(torch: Any, packed: Any) -> Any:
    """fp32 power-of-two scales from int32 words of four little-endian UE8M0 bytes."""
    exponents = packed.contiguous().view(torch.uint8).float()
    return torch.exp2(exponents - 127.0)


def check_against_dequantized_reference(
    torch: Any,
    casts: Any,
    x: Any,
    w_q: Any,
    w_scale: Any,
    out: Any,
    out_sf: Any,
) -> float:
    """Relative Frobenius error of the dequantized FP8 output against fp32
    ``einsum(deq(A), deq(W))`` on a strided sample of tokens.

    The output is dequantized from its own packed scale, so a wrong output
    scale layout or epilogue shows up as a large error.
    """
    per_token_cast_to_fp8, _ = casts
    m, h, k = x.shape
    n = w_q.shape[0] // h
    step = max(1, m // CHECK_ROWS)
    rows = slice(0, m, step)
    xs = x[rows].reshape(-1, k).contiguous()
    x_q, x_scale = per_token_cast_to_fp8(xs, use_ue8m0=True, gran_k=MXFP8_BLOCK)
    a = _dequantize(torch, x_q, x_scale, 1).view(-1, h, k)
    w = _dequantize(torch, w_q, w_scale, MXFP8_BLOCK).view(h, n, k)
    ref = torch.einsum("bhr,hdr->bhd", a, w)
    sf = _unpack_ue8m0(torch, out_sf[rows])  # [rows, h * n / 32]
    got = (out[rows].float().reshape(-1, h * n) * sf.repeat_interleave(MXFP8_BLOCK, dim=1)).view(
        -1, h, n
    )
    err = ((got - ref).norm() / ref.norm().clamp_min(1e-30)).item()
    if not err <= REL_FROBENIUS_TOL:
        raise KernelLaunchFailed(
            f"{BACKEND} output mismatch: relative Frobenius error {err:.3e} "
            f"> {REL_FROBENIUS_TOL:.0e}"
        )
    return err


def profile_batched_gemm_deepgemm_mxfp8_einsum_grouped_o_proj_fp8_out(
    num_batches: int,
    m: int,
    n: int,
    k: int,
    dtype: DType | str,
) -> ComputeMetrics:
    h, m, n, k = _validate(num_batches, m, n, k, dtype)
    torch, deep_gemm, quantize_activation, casts = _load_runtime()
    _, per_block_cast_to_fp8 = casts
    try:
        torch.manual_seed(0)
        device = torch.device("cuda")
        sf_type = torch.int32

        # Activation [m, h, k]: DeepInfer's per-token cast over [m, h * k], viewed per group.
        x = torch.randn(m, h, k, device=device, dtype=torch.bfloat16)
        a_q, a_sf = quantize_activation(x.view(m, h * k))
        a_arg = (a_q.view(m, h, k), a_sf.view(m, h, k // PACKED_SF_ELEMENTS))

        # Weight [h, n, k]: checkpoint block scales, expanded per row and packed per group.
        w_q, w_scale = per_block_cast_to_fp8(
            torch.randn(h * n, k, device=device, dtype=torch.bfloat16),
            use_ue8m0=True,
            gran_k=MXFP8_BLOCK,
        )
        w_sf = w_scale.view(h, n // MXFP8_BLOCK, k // MXFP8_BLOCK).repeat_interleave(
            MXFP8_BLOCK, dim=1
        )
        w_sf = deep_gemm.transform_sf_into_required_layout(
            w_sf, n, k, num_groups=h, recipe=(1, 1, MXFP8_BLOCK), is_sfa=False
        )
        b_arg = (w_q.view(h, n, k), w_sf)

        # FP8 output and its packed MN-major scale, allocated as _wv_and_o_proj does.
        out = torch.empty((m, h, n), dtype=torch.float8_e4m3fn, device=device)
        out_sf = torch.empty_strided(
            (m, h * n // PACKED_SF_ELEMENTS),
            (1, deep_gemm.get_tma_aligned_size(m, sf_type.itemsize)),
            dtype=sf_type,
            device=device,
        )

        def run_once() -> None:
            deep_gemm.fp8_einsum(
                "bhr,hdr->bhd", a_arg, b_arg, (out, out_sf), recipe=(1, 1, MXFP8_BLOCK)
            )

        run_once()  # DeepGEMM JIT compile, outside timing.
        torch.cuda.synchronize()
        check_against_dequantized_reference(torch, casts, x, w_q, w_scale, out, out_sf)
        del x

        time_ms = Timer.cupti(run_once, warmup=3)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    except (ProfilerNotImplemented, KernelLaunchFailed):
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError(f"{BACKEND} ran out of memory") from exc
        raise KernelLaunchFailed(f"{BACKEND} failed: {exc}") from exc

    # e4m3 operands with their per-32 scales; e4m3 output with its per-32 scale.
    bytes_accessed = h * (m * k + n * k + m * n) * (1 + 1 / MXFP8_BLOCK)
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=2 * h * m * n * k / elapsed_s / 1e12,
        memory_bandwidth_gbps=bytes_accessed / elapsed_s / 1e9,
        energy_j=float(energy_j),
    )


__all__ = [
    "check_against_dequantized_reference",
    "profile_batched_gemm_deepgemm_mxfp8_einsum_grouped_o_proj_fp8_out",
]
