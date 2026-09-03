"""Profile SGLang's public deferred-MoE finalize callable."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "moe_finalize_fuse_shared:sglang_cuda"


@dataclass(frozen=True)
class _Launch:
    fn: Any
    gemm2_out: Any
    permuted_idx: Any
    expert_weights: Any
    shared_output: Any
    top_k: int
    enable_pdl: bool

    def run(self) -> None:
        self.fn(
            self.gemm2_out,
            self.permuted_idx,
            self.expert_weights,
            self.shared_output,
            self.top_k,
            self.enable_pdl,
        )


def _validate_args(
    num_tokens: int,
    top_k: int,
    hidden_dim: int,
    dtype: DType | str,
    fuse_shared_output: bool,
) -> DType:
    for name, value in (("num_tokens", num_tokens), ("top_k", top_k), ("hidden_dim", hidden_dim)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(fuse_shared_output) is not bool:
        raise ValueError("fuse_shared_output must be a bool")
    resolved = DType.from_value(dtype)
    if resolved is not DType.BF16:
        raise ProfilerNotImplemented(f"{_BACKEND} requires dtype=bf16")
    return resolved


def _prepare(
    torch: Any,
    fn: Any,
    *,
    num_tokens: int,
    top_k: int,
    hidden_dim: int,
    fused: bool,
) -> _Launch:
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(0)
    gemm2_out = torch.randn(
        (num_tokens * top_k, hidden_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    permuted_idx = torch.randperm(num_tokens * top_k, device=device, generator=generator).to(
        torch.int32
    )
    expert_weights = torch.rand(
        (num_tokens, top_k), dtype=torch.float32, device=device, generator=generator
    )
    shared_output = (
        torch.randn(
            (num_tokens, hidden_dim),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        if fused
        else None
    )
    return _Launch(
        fn=fn,
        gemm2_out=gemm2_out,
        permuted_idx=permuted_idx,
        expert_weights=expert_weights,
        shared_output=shared_output,
        top_k=top_k,
        # This backend is registered only for B200, where SGLang's
        # `is_arch_support_pdl()` returns true for the production call.
        enable_pdl=True,
    )


def profile_moe_finalize_fuse_shared_sglang(
    num_tokens: int,
    top_k: int,
    hidden_dim: int,
    dtype: DType | str,
    fuse_shared_output: bool,
) -> ComputeMetrics:
    _validate_args(num_tokens, top_k, hidden_dim, dtype, fuse_shared_output)
    try:
        import torch
        from sglang.kernels.ops.moe.moe_finalize_fuse_shared import moe_finalize_fuse_shared
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires the SGLang environment") from exc

    try:
        if not torch.cuda.is_available():
            raise ProfilerNotImplemented(f"{_BACKEND} requires CUDA")
        launch = _prepare(
            torch,
            moe_finalize_fuse_shared,
            num_tokens=num_tokens,
            top_k=top_k,
            hidden_dim=hidden_dim,
            fused=fuse_shared_output,
        )
        # The first call builds the JIT kernel; compilation is not part of the
        # steady-state serving launch measured by this L1 row.
        launch.run()
        torch.cuda.synchronize()
        time_ms = Timer.cupti(launch.run, warmup=5)
        energy_j = Energy.perf(launch.run, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed") from exc

    elapsed_s = time_ms / 1000.0
    flops = num_tokens * hidden_dim * (2 * top_k + (1 if fuse_shared_output else 0))
    logical_bytes = (
        2 * num_tokens * hidden_dim * (top_k + 1 + (1 if fuse_shared_output else 0))
        + num_tokens * top_k * 8
    )
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=float(flops / elapsed_s / 1e12 if elapsed_s else 0.0),
        memory_bandwidth_gbps=float(logical_bytes / elapsed_s / 1e9 if elapsed_s else 0.0),
        energy_j=float(energy_j),
    )


__all__ = ["profile_moe_finalize_fuse_shared_sglang"]
