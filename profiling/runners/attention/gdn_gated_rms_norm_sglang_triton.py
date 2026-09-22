"""SGLang Triton gated RMSNorm runner for Kimi-K3."""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics


def _validate_args(m: int, hidden: int, dtype: DType | str) -> tuple[int, int, DType]:
    m = int(m)
    hidden = int(hidden)
    dtype = DType.from_value(dtype)
    if m <= 0 or hidden != 128:
        raise ValueError("SGLang K3 gated RMSNorm requires positive m and hidden=128")
    if dtype is not DType.BF16:
        raise ValueError("SGLang K3 gated RMSNorm requires dtype=bf16")
    return m, hidden, dtype


def profile_gdn_gated_rms_norm_sglang_triton(
    m: int,
    hidden: int,
    dtype: DType | str,
) -> ComputeMetrics:
    m, hidden, dtype = _validate_args(m, hidden, dtype)
    try:
        import torch
        from sglang.kernels.ops.attention.fla.fused_norm_gate import FusedRMSNormGated
    except ImportError as exc:
        raise ProfilerNotImplemented("SGLang FusedRMSNormGated is required") from exc
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for SGLang gated RMSNorm profiling")
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    x = torch.randn((m, hidden), dtype=torch.bfloat16, device=device, generator=generator)
    gate = torch.randn((m, hidden), dtype=torch.bfloat16, device=device, generator=generator)
    module = FusedRMSNormGated(
        hidden_size=hidden,
        eps=1e-5,
        activation="sigmoid",
        device=device,
        dtype=torch.bfloat16,
    )
    module.weight.data.fill_(1.0)

    def kernel() -> Any:
        return module(x, gate)

    try:
        time_ms = Timer.cupti(kernel)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
    seconds = time_ms / 1000.0
    flops = 7 * m * hidden + 2 * m
    bytes_accessed = (3 * m * hidden + hidden) * dtype.size_bytes()
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=flops / seconds / 1e12 if seconds else 0.0,
        memory_bandwidth_gbps=bytes_accessed / seconds / 1e9 if seconds else 0.0,
        energy_j=float(energy_j),
    )
