"""Kimi-K3 tvm_ffi fused conv + KDA + gated-norm runner."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics


@dataclass(frozen=True)
class _Args:
    batch_size: int
    num_heads: int
    head_k_dim: int
    head_v_dim: int
    dtype: DType
    state_dtype: DType
    lower_bound: float


def _validate_args(
    batch_size: int,
    num_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    dtype: DType | str,
    state_dtype: DType | str,
    lower_bound: float,
) -> _Args:
    args = _Args(
        int(batch_size),
        int(num_heads),
        int(head_k_dim),
        int(head_v_dim),
        DType.from_value(dtype),
        DType.from_value(state_dtype),
        float(lower_bound),
    )
    if (
        args.batch_size <= 0
        or args.num_heads != 12
        or args.head_k_dim != 128
        or args.head_v_dim != 128
    ):
        raise ValueError(
            "SGLang K3 fused KDA requires positive batch, 12 heads, and 128-wide heads"
        )
    if args.dtype is not DType.BF16 or args.state_dtype is not DType.FP32:
        raise ValueError("SGLang K3 fused KDA requires dtype=bf16 and state_dtype=fp32")
    if not math.isfinite(args.lower_bound) or args.lower_bound >= 0.0:
        raise ValueError("KDA lower_bound must be a finite negative value")
    return args


def _build_operands(torch: Any, args: _Args) -> dict[str, Any]:
    device = torch.device("cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    h, k, v = args.num_heads, args.head_k_dim, args.head_v_dim
    seg = h * k
    slots = args.batch_size + 1
    bf16 = torch.bfloat16
    fp32 = torch.float32
    return {
        "mixed_qkv": torch.randn(
            (args.batch_size, 3 * seg), dtype=bf16, device=device, generator=generator
        ).contiguous(),
        "a": torch.randn(
            (args.batch_size, seg), dtype=bf16, device=device, generator=generator
        ).contiguous(),
        "b": torch.randn(
            (args.batch_size, h), dtype=bf16, device=device, generator=generator
        ).contiguous(),
        "conv_states": torch.randn(
            (slots, 3, 3 * seg), dtype=bf16, device=device, generator=generator
        ).contiguous(),
        "w_q_t": torch.randn((4, seg), dtype=fp32, device=device, generator=generator).contiguous(),
        "w_k_t": torch.randn((4, seg), dtype=fp32, device=device, generator=generator).contiguous(),
        "w_v_t": torch.randn((4, seg), dtype=fp32, device=device, generator=generator).contiguous(),
        "conv_bias": torch.zeros(3 * seg, dtype=fp32, device=device),
        "A_log": torch.zeros(h, dtype=fp32, device=device),
        "dt_bias": torch.zeros(seg, dtype=fp32, device=device),
        "onorm_g": torch.randn(
            (args.batch_size, seg), dtype=bf16, device=device, generator=generator
        ).contiguous(),
        "onorm_weight": torch.ones(seg, dtype=fp32, device=device),
        "ssm_states": torch.randn(
            (slots, h, v, k), dtype=fp32, device=device, generator=generator
        ).contiguous(),
        "cache_indices": torch.arange(1, args.batch_size + 1, dtype=torch.int32, device=device),
    }


def _invoke(fn: Any, operands: dict[str, Any], args: _Args) -> Any:
    return fn(
        operands["mixed_qkv"],
        operands["a"],
        operands["b"],
        operands["conv_states"],
        operands["w_q_t"],
        operands["w_k_t"],
        operands["w_v_t"],
        operands["conv_bias"],
        operands["A_log"],
        operands["dt_bias"],
        operands["onorm_g"],
        operands["onorm_weight"],
        operands["ssm_states"],
        operands["cache_indices"],
        scale=args.head_k_dim**-0.5,
        onorm_eps=1e-6,
        lower_bound=args.lower_bound,
    )


def profile_kda_fused_decode_sglang_fused(
    batch_size: int,
    num_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    dtype: DType | str,
    state_dtype: DType | str,
    lower_bound: float,
) -> ComputeMetrics:
    args = _validate_args(
        batch_size, num_heads, head_k_dim, head_v_dim, dtype, state_dtype, lower_bound
    )
    try:
        import torch

        fused_module = importlib.import_module("sglang.kernels.ops.attention.kda_fused_decode")
        fn = getattr(fused_module, "kda_fused_decode")
    except ImportError as exc:
        raise ProfilerNotImplemented("SGLang K3 fused KDA callable is required") from exc
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for the SGLang fused KDA profiler")
    operands = _build_operands(torch, args)

    def kernel() -> Any:
        return _invoke(fn, operands, args)

    try:
        time_ms = Timer.cupti(kernel)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
    elapsed_s = time_ms / 1000.0
    flops = 2 * args.batch_size * args.num_heads * args.head_k_dim * args.head_v_dim * 8
    bytes_accessed = (
        args.batch_size * 3 * args.num_heads * args.head_k_dim * args.dtype.size_bytes()
        + args.batch_size
        * args.num_heads
        * args.head_v_dim
        * args.head_k_dim
        * args.state_dtype.size_bytes()
    )
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=flops / elapsed_s / 1e12 if elapsed_s else 0.0,
        memory_bandwidth_gbps=bytes_accessed / elapsed_s / 1e9 if elapsed_s else 0.0,
        energy_j=float(energy_j),
    )
