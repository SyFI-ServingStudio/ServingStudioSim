"""Persistent CUDA runner for ``kda_recoverssm_conv_verify`` (backend ``cuda_persistent_wy_mma``).

The kernel is the MoE session's fused conv + verify, vendored read-only in
``kda_conv_verify_fused_vendor/kfused.cu`` (see its header), launched as that
session measured it (rung 2, minb 4, ABL 6064). One persistent launch, one CTA
per (request, head): it loads the head's 32 KB checkpoint into tensor-core
fragments, copies the 3 conv history rows and the window's raw ``q|k|v`` and
gate rows into shared memory, runs the width-4 convolution + SiLU, writes the
rolled conv window back in place, verifies the window in WY form (two bf16
hi/lo tensor-core GEMMs and a forward substitution) without writing the
checkpoint, and writes the outputs and the per-request fp32 correction and
key/gate records the commit reads, in the fork's layouts.

It reads and writes only the fork's default SD conv page (``[state_len, dim]``,
history row ``r`` at ``conv + r * dim``; there is no dim stride), takes fp32
conv taps, and is compiled for an 8-token window (``SQ``), so it serves
``spec_query_len`` 8 only. The operands are the CuTe runner's
(``build_conv_verify_operands``, SD); one call is checked against the same Torch
oracles and tolerances as the CuTe backends before timing. Timing is the CUPTI
sum of the launch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention.kda_recoverssm_vllm_cute import (
    ConvVerifyOperands,
    ConvVerifyShape,
    build_conv_verify_operands,
    check_conv_verify,
    conv_verify_bytes,
    conv_verify_flops,
    validate_conv_verify_args,
)
from profiling.runners.attention.kda_recoverssm_vllm_triton import LOWER_BOUND, _import_torch, _metrics
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

BACKEND = "kda_recoverssm_conv_verify:cuda_persistent_wy_mma"
_SOURCE = Path(__file__).resolve().parent / "kda_conv_verify_fused_vendor" / "kfused.cu"
_MODULE = "kda_recoverssm_conv_verify_wy_mma"
_WINDOW = 8  # SQ in the kernel
_RUNG, _MINB, _ABL = 2, 4, 6064  # the measured variant
_ext: Any = None


def _extension() -> Any:
    """Build the vendored kernel once per process (cached under HOME)."""
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load

        _ext = load(
            name=_MODULE,
            sources=[str(_SOURCE)],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "-std=c++20"],
            verbose=False,
        )
    return _ext


def fused_call(torch: Any, ext: Any, operands: ConvVerifyOperands, shape: ConvVerifyShape) -> Any:
    """One launch over ``operands`` (the CuTe runner's SD operands)."""
    rows, channels = shape.conv_state_len, shape.conv_dim
    page = operands.conv_page
    if page.stride(2) != 1 or page.stride(1) != channels:
        raise ValueError("the fused conv + verify needs [state_len, dim] (SD) conv pages")
    if operands.conv_weight.dtype != torch.float32:
        raise ValueError("the fused conv + verify takes fp32 conv taps")
    pool = page.as_strided((page.shape[0] * page.stride(0),), (1,), 0)
    page_bytes = page.stride(0) * 2
    conv_bytes = rows * channels * 2
    width = operands.projected.shape[1]
    records = operands.records
    stamps = torch.zeros(1, dtype=torch.int64, device=pool.device)
    args = (
        _RUNG, _MINB, pool, page_bytes, conv_bytes, operands.state_indices, operands.accepted,
        operands.mixed_qkv, width, operands.raw_gate, operands.raw_gate.stride(0), operands.raw_beta, width,
        operands.a_log, operands.dt_bias, operands.conv_weight, operands.out, records.correction,
        records.key_gate, LOWER_BOUND, shape.batch_size, shape.num_heads, rows, _ABL, 0, stamps,
    )
    return lambda: ext.fused(*args)


def profile_kda_recoverssm_conv_verify_cuda_persistent_wy_mma(
    batch_size: int,
    num_heads: int,
    head_dim: int,
    spec_query_len: int,
    dtype: DType | str,
    conv_weight_dtype: DType | str,
) -> ComputeMetrics:
    """Profile one fused conv + WY-form tensor-core verify launch on SD conv pages."""
    shape = validate_conv_verify_args(
        batch_size, num_heads, head_dim, spec_query_len, dtype, conv_weight_dtype, dim_first=False
    )
    if shape.spec_query_len != _WINDOW:
        raise ValueError(f"the fused conv + verify is compiled for a {_WINDOW}-token window")
    if shape.conv_weight_dtype is not DType.FP32:
        raise ValueError("the fused conv + verify takes fp32 conv taps")
    torch = _import_torch(BACKEND)
    try:
        from vllm.models.kimi_k3.nvidia.ops.recoverssm import KDARecoverSSMRecords

        ext = _extension()
        device = torch.device("cuda", torch.cuda.current_device())
        operands = build_conv_verify_operands(torch, KDARecoverSSMRecords, shape, device=device)
        kernel = fused_call(torch, ext, operands, shape)
        check_conv_verify(torch, None, operands, shape, invoke=kernel)
        time_ms = Timer.cupti(kernel, warmup=5, kernel_name=None)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{BACKEND} ran out of CUDA memory") from exc
    except (ProfilerNotImplemented, ValueError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{BACKEND} failed: {exc}") from exc
    return _metrics(time_ms, energy_j, conv_verify_flops(shape), conv_verify_bytes(shape))


__all__ = ["BACKEND", "fused_call", "profile_kda_recoverssm_conv_verify_cuda_persistent_wy_mma"]
