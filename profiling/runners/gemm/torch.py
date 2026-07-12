"""Torch GEMM runners.

This file is L1a-only: it allocates tensors, times one kernel, and returns
metrics. DB writes, JIT policy, subprocess selection, and registry routing all
live in L1b.
"""

from __future__ import annotations

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics


def profile_single_gemm(
    m: int,
    n: int,
    k: int,
    dtype: DType | str,
) -> ComputeMetrics:
    dtype = DType.from_value(dtype)

    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented("torch is required for the torch GEMM runner") from exc

    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for the torch GEMM runner")

    try:
        torch_dtype = dtype.torch()
        a = torch.randn(m, k, dtype=torch_dtype, device="cuda")
        b = torch.randn(k, n, dtype=torch_dtype, device="cuda")

        def kernel():
            return torch.mm(a, b)

        # cupti samples adaptively until convergence (kernel-only, cold L2),
        # no warmup; kernel_name=None sums the GEMM kernel(s) in the window.
        time_ms = Timer.cupti(kernel)
        energy_j = Energy.perf(
            kernel,
            warmup=5,
            per_iter_time_ms=time_ms,
        )

        flops = 2 * m * n * k
        tflops = (flops / (time_ms / 1000.0)) / 1e12 if time_ms > 0 else 0.0
        bytes_accessed = int((a.numel() + b.numel() + m * n) * dtype.size_bytes())
        bandwidth_gbps = (bytes_accessed / (time_ms / 1000.0)) / 1e9 if time_ms > 0 else 0.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=float(tflops),
            memory_bandwidth_gbps=float(bandwidth_gbps),
            energy_j=float(energy_j),
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc


def profile_single_gemm_linear(
    m: int,
    n: int,
    k: int,
    dtype: DType | str,
) -> ComputeMetrics:
    """Profile the model-linear layout used by vLLM and Transformers.

    Unlike ``profile_single_gemm``, which owns a contiguous ``(k, n)`` RHS and
    calls ``torch.mm``, this backend owns the canonical ``(n, k)`` weight and
    calls ``F.linear``. The distinction is part of backend identity because it
    selects different NvJet transpose/tile variants on Hopper.
    """
    dtype = DType.from_value(dtype)

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise ProfilerNotImplemented("torch is required for the torch_linear backend") from exc

    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for the torch_linear backend")

    try:
        torch_dtype = dtype.torch()
        activations = torch.randn(m, k, dtype=torch_dtype, device="cuda")
        weight = torch.randn(n, k, dtype=torch_dtype, device="cuda")

        def kernel():
            return functional.linear(activations, weight)

        # Time the canonical model-weight layout through the shared CUPTI timer.
        time_ms = Timer.cupti(kernel)
        energy_j = Energy.perf(
            kernel,
            warmup=5,
            per_iter_time_ms=time_ms,
        )

        flops = 2 * m * n * k
        tflops = (flops / (time_ms / 1000.0)) / 1e12 if time_ms > 0.0 else 0.0
        elements = activations.numel() + weight.numel() + m * n
        bytes_accessed = int(elements * dtype.size_bytes())
        bandwidth_gbps = (bytes_accessed / (time_ms / 1000.0)) / 1e9 if time_ms > 0.0 else 0.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=float(tflops),
            memory_bandwidth_gbps=float(bandwidth_gbps),
            energy_j=float(energy_j),
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc


def profile_grouped_gemm(
    n: int,
    k: int,
    dtype: DType | str,
    num_local_experts: int,
    per_group_batches: tuple[int, ...] | list[int],
) -> ComputeMetrics:
    dtype = DType.from_value(dtype)
    batches = tuple(int(b) for b in per_group_batches)
    if len(batches) != int(num_local_experts):
        raise ValueError(
            f"per_group_batches has {len(batches)} entries, expected num_local_experts="
            f"{num_local_experts}"
        )

    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented("torch is required for the torch grouped-GEMM runner") from exc

    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for the torch grouped-GEMM runner")

    total_m = sum(batches)
    if total_m == 0:
        raise ValueError("per_group_batches sums to 0; no tokens to profile")

    try:
        torch_dtype = dtype.torch()
        # Single fused grouped GEMM via torch._grouped_mm — the real MoE
        # expert-compute kernel, NOT a per-expert mm loop. `a` stacks every
        # expert's tokens row-wise ((Σ m_g, k)); `w` is the per-expert weight
        # stack ((E, k, n)); `offs` are the cumulative group-end rows so the one
        # kernel dispatches all experts (variable m_g, empty groups allowed). This
        # is the torchtitan/torchao MoE primitive — distribution (m_g spread) is
        # the cost driver, and one kernel models it without Python dispatch gaps.
        a = torch.randn(total_m, k, dtype=torch_dtype, device="cuda")
        w = torch.randn(num_local_experts, k, n, dtype=torch_dtype, device="cuda")
        ends, acc = [], 0
        for b in batches:
            acc += b
            ends.append(acc)
        offs = torch.tensor(ends, dtype=torch.int32, device="cuda")

        def kernel():
            return torch._grouped_mm(a, w, offs=offs)

        # cupti times the single grouped kernel (kernel-only, cold L2, no warmup).
        time_ms = Timer.cupti(kernel)
        energy_j = Energy.perf(
            kernel,
            warmup=5,
            per_iter_time_ms=time_ms,
        )

        flops = 2 * total_m * n * k
        tflops = (flops / (time_ms / 1000.0)) / 1e12 if time_ms > 0 else 0.0
        # Read A (Σ m_g·k) + all weights (E·k·n), write out (Σ m_g·n).
        elems = total_m * k + num_local_experts * k * n + total_m * n
        bytes_accessed = int(elems * dtype.size_bytes())
        bandwidth_gbps = (bytes_accessed / (time_ms / 1000.0)) / 1e9 if time_ms > 0 else 0.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=float(tflops),
            memory_bandwidth_gbps=float(bandwidth_gbps),
            energy_j=float(energy_j),
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
