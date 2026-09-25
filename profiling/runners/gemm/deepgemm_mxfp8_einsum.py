"""DeepSeek-V4.1 ``wo_a`` grouped MXFP8 einsum — the
``deepgemm_mxfp8_einsum_dsv41_wo_a`` backend of ``batched_gemm``.

Source: alignment fork ``servingstudio-alignment-v41``.
``DeepseekV4MegaAttnAttention._o_proj``
(``vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py``) calls
``vllm.utils.deep_gemm.fp8_einsum("bhr,hdr->bhd", ...)`` once per layer with
``recipe=(1, 1, 32)``. DeepGEMM runs it as one
``deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl`` launch (the SM100 FP8/FP4 kernel
family; both operands here are FP8).

Operands, per rank (``h = num_batches`` local ``wo_a`` groups):

- A ``[T, h, K]``: the mega-attention output. It is FP8 E4M3 with one UE8M0
  scale per 32 elements along K. The attention kernel already did the inverse
  RoPE, the cast and the permuted layout, so no input-quant launch exists
  before the einsum. The buffer comes from the fork's own
  ``alloc_mega_attn_output``. It holds all ``64 / 8 = 8`` padded head-group
  slots, and the einsum reads the first ``h``. So the A row stride is
  ``8 * K``, and the scale is packed-UE8M0 int32, MN-major (token stride 1).
  This backend freezes that layout, the way the GLM backends of this kind
  freeze theirs.
- B ``[h, N, K]``: ``wo_a`` as MXFP8 (E4M3 data plus a UE8M0 scale per row and
  32 K elements). The checkpoint stores ``[32, 32]`` blocks, which the fork's
  ModelOpt MXFP8 path expands to per-row scales. It goes through the
  production ``DeepGemmMxfp8BmmLinearKernel.process_weights_after_loading``
  (reshape to 3-D, ``transform_sf_into_required_layout``).
- Output ``[T, h, N]`` bf16.

``K = 8 heads * 512 head_dim = 4096`` and ``N = o_lora_rank = 1024``. The
load-time ``wo_a`` column permutation is omitted: it only relabels K on both
operands.

CUPTI time sums every launch of the call (``kernel_name=None``). On B200 that
is the single GEMM launch.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

BACKEND = "deepgemm_mxfp8_einsum_dsv41_wo_a"
MXFP8_BLOCK = 32
RECIPE = (1, 1, MXFP8_BLOCK)
# 8 heads per wo_a group times the 512-wide V head.
WO_A_K = 4096
# o_lora_rank.
WO_A_N = 1024
# Mega attention pads Q to 64 heads, i.e. 8 group slots of 8 heads each, and
# writes all of them. TP1/2/4/8 on the 64-head model use 8/4/2/1 of them.
MEGA_ATTN_GROUP_SLOTS = 8
# bf16 output rounding (2^-8 relative) plus accumulation-order differences.
REL_FROBENIUS_TOL = 1e-2
_E4M3_MAX = 448.0


def _validate(
    num_batches: int, m: int, n: int, k: int, dtype: DType | str
) -> tuple[int, int, int, int]:
    if DType.from_value(dtype) is not DType.MXFP8_E4M3:
        raise ValueError(f"{BACKEND} requires dtype=mxfp8_e4m3, got {dtype!r}")
    num_batches, m, n, k = int(num_batches), int(m), int(n), int(k)
    if not 1 <= num_batches <= MEGA_ATTN_GROUP_SLOTS:
        raise ValueError(
            f"{BACKEND} requires 1 <= num_batches <= {MEGA_ATTN_GROUP_SLOTS} "
            f"(local wo_a groups), got {num_batches}"
        )
    if m < 1:
        raise ValueError(f"{BACKEND} requires m >= 1, got m={m}")
    if (n, k) != (WO_A_N, WO_A_K):
        raise ValueError(f"{BACKEND} requires (n, k) == ({WO_A_N}, {WO_A_K}), got ({n}, {k})")
    return num_batches, m, n, k


def _load_runtime() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"torch is required for {BACKEND}") from exc
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(f"{BACKEND} requires CUDA")
    if tuple(torch.cuda.get_device_capability()) != (10, 0):
        raise ProfilerNotImplemented(f"{BACKEND} is verified on SM100 (B200) only")
    try:
        from vllm.model_executor.kernels.linear.mxfp8.deep_gemm import (
            DeepGemmMxfp8BmmLinearKernel,
        )
        from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
            Mxfp8LinearLayerConfig,
        )
        from vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn import (
            alloc_mega_attn_output,
        )
        from vllm.utils.deep_gemm import fp8_einsum
    except ImportError as exc:
        raise ProfilerNotImplemented(
            f"{BACKEND} requires the vLLM fork environment (vllm_fork_env)"
        ) from exc
    supported, reason = DeepGemmMxfp8BmmLinearKernel.is_supported()
    if not supported:
        raise ProfilerNotImplemented(f"DeepGEMM MXFP8 BMM unavailable: {reason}")

    def make_kernel(num_batches: int) -> Any:
        return DeepGemmMxfp8BmmLinearKernel(Mxfp8LinearLayerConfig(bmm_batch_size=num_batches))

    return torch, make_kernel, alloc_mega_attn_output, fp8_einsum


def quantize_mxfp8(torch: Any, x: Any) -> tuple[Any, Any]:
    """E4M3 data and uint8 UE8M0 exponents, one per 32 elements of the last dim.

    The scale is the smallest power of two that maps the block amax into the
    E4M3 range, which is the rounding the fork's MXFP8 writers use.
    """
    *lead, cols = x.shape
    blocks = x.float().reshape(*lead, cols // MXFP8_BLOCK, MXFP8_BLOCK)
    amax = blocks.abs().amax(dim=-1).clamp_min(1e-30)
    exponent = torch.ceil(torch.log2(amax / _E4M3_MAX)).clamp(-127, 127)
    data = (blocks / torch.exp2(exponent).unsqueeze(-1)).to(torch.float8_e4m3fn)
    return data.reshape(*lead, cols), (exponent + 127).to(torch.uint8)


def dequantize_mxfp8(torch: Any, data: Any, scale: Any) -> Any:
    """fp32 value of an MXFP8 tensor from its data and unpacked UE8M0 scales."""
    *lead, cols = data.shape
    blocks = data.float().reshape(*lead, cols // MXFP8_BLOCK, MXFP8_BLOCK)
    factor = torch.exp2(scale.float() - 127.0).unsqueeze(-1)
    return (blocks * factor).reshape(*lead, cols)


def check_against_dequantized_reference(
    torch: Any, a_q: Any, a_s: Any, w_q: Any, w_s: Any, out: Any
) -> float:
    """Relative Frobenius error of ``out`` vs fp32 ``deq(A) x deq(W)^T`` per group.

    The reference reads the unpacked row-major scales, so a mistake in the
    packed MN-major activation scale or the transformed weight scale shows up
    as a large error instead of being shared with the oracle.
    """
    ref = torch.einsum(
        "bhr,hdr->bhd", dequantize_mxfp8(torch, a_q, a_s), dequantize_mxfp8(torch, w_q, w_s)
    )
    err = ((out.float() - ref).norm() / ref.norm().clamp_min(1e-30)).item()
    if not err <= REL_FROBENIUS_TOL:
        raise KernelLaunchFailed(
            f"{BACKEND} output mismatch: relative Frobenius error {err:.3e} "
            f"> {REL_FROBENIUS_TOL:.0e}"
        )
    return err


def profile_batched_gemm_deepgemm_mxfp8_einsum_dsv41_wo_a(
    num_batches: int,
    m: int,
    n: int,
    k: int,
    dtype: DType | str,
) -> ComputeMetrics:
    num_batches, m, n, k = _validate(num_batches, m, n, k, dtype)
    torch, make_kernel, alloc_mega_attn_output, fp8_einsum = _load_runtime()
    try:
        torch.manual_seed(0)
        device = torch.device("cuda")

        # Activation: the production mega-attention output buffer, all slots filled.
        attn_out = alloc_mega_attn_output(m, MEGA_ATTN_GROUP_SLOTS, device)
        a_full, a_scale_full = quantize_mxfp8(
            torch, torch.randn(m, MEGA_ATTN_GROUP_SLOTS, k, device=device, dtype=torch.bfloat16)
        )
        attn_out.data.copy_(a_full)
        # Four consecutive K-block exponents pack little-endian into one int32.
        attn_out.scale.copy_(a_scale_full.contiguous().view(torch.int32))
        a_arg = (attn_out.data[:, :num_batches], attn_out.scale[:, :num_batches])

        # Weight: [h * N, K] MXFP8 through the production BMM weight transform.
        w_q, w_s = quantize_mxfp8(
            torch, torch.randn(num_batches * n, k, device=device, dtype=torch.bfloat16)
        )
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(w_q.clone(), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(w_s.clone(), requires_grad=False)
        make_kernel(num_batches).process_weights_after_loading(layer)
        b_arg = (layer.weight, layer.weight_scale)

        out = torch.empty((m, num_batches, n), dtype=torch.bfloat16, device=device)

        def run_once() -> None:
            fp8_einsum("bhr,hdr->bhd", a_arg, b_arg, out, recipe=RECIPE)

        run_once()  # DeepGEMM JIT compile, outside timing.
        torch.cuda.synchronize()
        check_against_dequantized_reference(
            torch,
            a_full[:, :num_batches],
            a_scale_full[:, :num_batches],
            w_q.view(num_batches, n, k),
            w_s.view(num_batches, n, k // MXFP8_BLOCK),
            out,
        )
        del a_full, a_scale_full, w_q, w_s

        time_ms = Timer.cupti(run_once, warmup=3)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    except (ProfilerNotImplemented, KernelLaunchFailed):
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError(f"{BACKEND} ran out of memory") from exc
        raise KernelLaunchFailed(f"{BACKEND} failed: {exc}") from exc

    # Logical traffic: both e4m3 operands, their ue8m0 scales, bf16 output.
    scale_bytes = num_batches * (m + n) * (k // MXFP8_BLOCK)
    bytes_accessed = num_batches * (m * k + n * k + 2 * m * n) + scale_bytes
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=2 * num_batches * m * n * k / elapsed_s / 1e12,
        memory_bandwidth_gbps=bytes_accessed / elapsed_s / 1e9,
        energy_j=float(energy_j),
    )


__all__ = [
    "check_against_dequantized_reference",
    "dequantize_mxfp8",
    "profile_batched_gemm_deepgemm_mxfp8_einsum_dsv41_wo_a",
    "quantize_mxfp8",
]
