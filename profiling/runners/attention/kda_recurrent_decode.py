"""Kimi-K3 KDA recurrent decode runners.

The Torch path is a small semantic reference. The SGLang path calls the exact
``TritonKDAKernel.decode`` callable and measures its complete launch boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_LOWER_BOUND = -5.0


@dataclass(frozen=True)
class _Args:
    batch_size: int
    num_heads: int
    head_k_dim: int
    head_v_dim: int
    dtype: DType
    state_dtype: DType
    lower_bound: float


@dataclass(frozen=True)
class _Operands:
    q: Any
    k: Any
    v: Any
    a: Any
    b: Any
    A_log: Any
    dt_bias: Any
    state: Any
    cache_indices: Any
    cu_seqlens: Any


def _validate_args(
    batch_size: int,
    num_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    dtype: DType | str,
    state_dtype: DType | str,
    lower_bound: float,
    *,
    production: bool = False,
) -> _Args:
    args = _Args(
        batch_size=int(batch_size),
        num_heads=int(num_heads),
        head_k_dim=int(head_k_dim),
        head_v_dim=int(head_v_dim),
        dtype=DType.from_value(dtype),
        state_dtype=DType.from_value(state_dtype),
        lower_bound=float(lower_bound),
    )
    if min(args.batch_size, args.num_heads, args.head_k_dim, args.head_v_dim) <= 0:
        raise ValueError("KDA dimensions and batch_size must be positive")
    if args.dtype is not DType.BF16:
        raise ValueError("KDA decode supports BF16 activations")
    if args.state_dtype not in (DType.BF16, DType.FP32):
        raise ValueError("KDA state_dtype must be fp32 or bf16")
    if not math.isfinite(args.lower_bound) or args.lower_bound >= 0.0:
        raise ValueError("KDA lower_bound must be a finite negative value")
    if production and (args.num_heads, args.head_k_dim, args.head_v_dim) != (12, 128, 128):
        raise ValueError("SGLang K3 KDA decode requires num_heads=12 and head dims 128")
    if production and args.state_dtype is not DType.FP32:
        raise ValueError("SGLang K3 KDA decode requires state_dtype=fp32")
    return args


def _build_operands(torch: Any, args: _Args, *, device: Any) -> _Operands:
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    activation = args.dtype.torch()
    state_dtype = args.state_dtype.torch()
    shape = (1, args.batch_size, args.num_heads)
    q = torch.randn((*shape, args.head_k_dim), dtype=activation, device=device, generator=generator)
    k = torch.randn((*shape, args.head_k_dim), dtype=activation, device=device, generator=generator)
    v = torch.randn((*shape, args.head_v_dim), dtype=activation, device=device, generator=generator)
    a = torch.randn(
        (args.batch_size, args.num_heads * args.head_k_dim),
        dtype=activation,
        device=device,
        generator=generator,
    )
    b = torch.randn(
        (1, args.batch_size, args.num_heads),
        dtype=activation,
        device=device,
        generator=generator,
    )
    # SGLang keeps the learned gate parameters in FP32 independently of the
    # recurrent-state pool dtype.
    A_log = torch.zeros(args.num_heads, dtype=torch.float32, device=device)
    dt_bias = torch.zeros(args.num_heads * args.head_k_dim, dtype=torch.float32, device=device)
    state = (
        torch.randn(
            (args.batch_size + 1, args.num_heads, args.head_v_dim, args.head_k_dim),
            dtype=state_dtype,
            device=device,
            generator=generator,
        )
        * 0.01
    )
    return _Operands(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        a.contiguous(),
        b.contiguous(),
        A_log.contiguous(),
        dt_bias.contiguous(),
        state.contiguous(),
        torch.arange(1, args.batch_size + 1, dtype=torch.int32, device=device),
        torch.arange(args.batch_size + 1, dtype=torch.int32, device=device),
    )


def kda_recurrent_decode_reference(
    q: Any,
    k: Any,
    v: Any,
    a: Any,
    b: Any,
    A_log: Any,
    dt_bias: Any,
    state: Any,
    lower_bound: float = _LOWER_BOUND,
) -> Any:
    """Run one KDA update with the Triton decode layout on any Torch device."""
    import torch

    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("q must have shape [1, batch, heads, head_k_dim]")
    if k.shape != q.shape or v.shape[:3] != q.shape[:3]:
        raise ValueError("q, k, and v must share the [1, batch, heads] prefix")
    batch_size, num_heads = q.shape[1:3]
    if a.shape == (batch_size, num_heads * q.shape[-1]):
        a = a.reshape(batch_size, num_heads, q.shape[-1])
    elif a.shape != (batch_size, num_heads, q.shape[-1]):
        raise ValueError(
            "a must have shape [batch, heads, head_k_dim] or [batch, heads*head_k_dim]"
        )
    if b.shape == (1, batch_size, num_heads):
        b = b[0]
    elif b.shape != (batch_size, num_heads):
        raise ValueError("b must have shape [batch, heads] or [1, batch, heads]")
    if state.shape != (batch_size + 1, num_heads, v.shape[-1], q.shape[-1]):
        raise ValueError("state must have shape [batch+1, heads, head_v_dim, head_k_dim]")
    qf = q.float()
    kf = k.float()
    vf = v.float()
    qf = qf / torch.sqrt(qf.square().sum(dim=-1, keepdim=True) + 1e-6)
    kf = kf / torch.sqrt(kf.square().sum(dim=-1, keepdim=True) + 1e-6)
    qf = qf * (q.shape[-1] ** -0.5)
    gate = float(lower_bound) * torch.sigmoid(
        torch.exp(A_log.float()).view(1, 1, num_heads, 1)
        * (a.float() + dt_bias.float().view(1, num_heads, -1))
    )
    # The reference uses state[h, v, k], matching SGLang's pool layout.
    selected = state[1 : batch_size + 1].float()
    selected.mul_(torch.exp(gate[0].unsqueeze(-2)))
    delta = vf[0] - torch.einsum("bhvk,bhk->bhv", selected, kf[0])
    delta.mul_(torch.sigmoid(b.float()).unsqueeze(-1))
    selected.add_(delta.unsqueeze(-1) * kf[0].unsqueeze(-2))
    output = torch.einsum("bhvk,bhk->bhv", selected, qf[0]).unsqueeze(0)
    state[1 : batch_size + 1].copy_(selected.to(state.dtype))
    return output.to(v.dtype)


def _semantic_flops(args: _Args) -> int:
    state = args.batch_size * args.num_heads * args.head_k_dim * args.head_v_dim
    return int(7 * state + 8 * args.batch_size * args.num_heads * args.head_k_dim)


def _logical_bytes(args: _Args) -> float:
    activation = (
        2 * args.batch_size * args.num_heads * args.head_k_dim
        + args.batch_size * args.num_heads * args.head_v_dim
        + args.batch_size * args.num_heads * args.head_k_dim
        + args.batch_size * args.num_heads
    ) * args.dtype.size_bytes()
    state = args.batch_size * args.num_heads * args.head_v_dim * args.head_k_dim
    return activation + state * args.state_dtype.size_bytes()


def _metrics(args: _Args, time_ms: float, energy_j: float) -> ComputeMetrics:
    elapsed_s = time_ms / 1000.0
    flops = _semantic_flops(args)
    bytes_accessed = _logical_bytes(args)
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=flops / elapsed_s / 1e12 if elapsed_s else 0.0,
        memory_bandwidth_gbps=bytes_accessed / elapsed_s / 1e9 if elapsed_s else 0.0,
        energy_j=float(energy_j),
    )


def profile_kda_recurrent_decode_torch(
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
    except ImportError as exc:
        raise ProfilerNotImplemented("PyTorch is required for KDA reference profiling") from exc
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for the torch KDA profiler")
    operands = _build_operands(torch, args, device=torch.device("cuda"))

    def kernel() -> Any:
        return kda_recurrent_decode_reference(
            operands.q,
            operands.k,
            operands.v,
            operands.a,
            operands.b,
            operands.A_log,
            operands.dt_bias,
            operands.state,
            args.lower_bound,
        )

    try:
        time_ms = Timer.cupti(kernel)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
    return _metrics(args, time_ms, energy_j)


def profile_kda_recurrent_decode_sglang_triton(
    batch_size: int,
    num_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    dtype: DType | str,
    state_dtype: DType | str,
    lower_bound: float,
) -> ComputeMetrics:
    args = _validate_args(
        batch_size,
        num_heads,
        head_k_dim,
        head_v_dim,
        dtype,
        state_dtype,
        lower_bound,
        production=True,
    )
    try:
        import torch
        from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
            fused_sigmoid_gating_delta_rule_update,
        )
    except ImportError as exc:
        raise ProfilerNotImplemented("SGLang K3 and its KDA Triton callable are required") from exc
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for the SGLang KDA profiler")
    operands = _build_operands(torch, args, device=torch.device("cuda"))

    def kernel() -> Any:
        return fused_sigmoid_gating_delta_rule_update(
            A_log=operands.A_log,
            dt_bias=operands.dt_bias,
            q=operands.q,
            k=operands.k,
            v=operands.v,
            a=operands.a,
            b=operands.b,
            initial_state_source=operands.state,
            initial_state_indices=operands.cache_indices,
            cu_seqlens=operands.cu_seqlens,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=True,
            lower_bound=args.lower_bound,
        )

    try:
        time_ms = Timer.cupti(kernel)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
    return _metrics(args, time_ms, energy_j)


__all__ = [
    "kda_recurrent_decode_reference",
    "profile_kda_recurrent_decode_torch",
    "profile_kda_recurrent_decode_sglang_triton",
]
