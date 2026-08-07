"""Triton element-wise runner.

This file is L1a-only: it allocates tensors, times one kernel, and returns
metrics. DB writes, JIT policy, subprocess selection, and registry routing all
live in L1b.

Mirrors ``ref/profile/elementwise/elementwise_triton.py``: a generic byte-level
fan-in elementwise/reduce (``input_size_bytes`` -> ``output_size_bytes``,
dtype-agnostic ``uint8``), timed with CUPTI kernel-only (the reference flushes
L2 each launch, so we use ``Timer.cupti`` whose default also flushes). Covers
MoE activation (2N->N), reduce (xN->N), copy (N->N), and zero-fill (0->N).

Unlike the torch/flashinfer runners, ``triton`` is imported at MODULE scope
(not lazily inside the function): a ``@triton.jit`` kernel resolves ``tl.*`` from
its module globals, so the kernels must be defined at module level. The import
is guarded so loading this module without triton degrades to
``ProfilerNotImplemented`` rather than an ImportError. This module is only ever
imported in the profiling worker subprocess (via ``RunnerRef``), never in the
main process.
"""

from __future__ import annotations

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

# CUPTI matches these device kernel names (substring). Keep aligned with the
# @triton.jit function names below and with ref TARGET_KERNEL_NAMES.
_FAN_KERNEL_NAME = "elementwise_fan_kernel"
_ZERO_KERNEL_NAME = "elementwise_zero_kernel"
_INT32_MAX = 2_147_483_647

try:
    import triton
    import triton.language as tl

    _TRITON_CONFIGS = [
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8),
    ]

    @triton.autotune(configs=_TRITON_CONFIGS, key=["OUTPUT_SIZE", "FAN_IN"])
    @triton.jit
    def _elementwise_fan_kernel(
        input_ptr,
        output_ptr,
        OUTPUT_SIZE: tl.constexpr,
        FAN_IN: tl.constexpr,
        USE_INT64: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if USE_INT64:
            # SM address arithmetic must not wrap at signed int32 for wide
            # buffers. Only the pointer-index width changes on this path.
            pid = pid.to(tl.int64)
            block_size = tl.full((), BLOCK_SIZE, tl.int64)
            output_size = tl.full((), OUTPUT_SIZE, tl.int64)
            offsets = pid * block_size + tl.arange(0, BLOCK_SIZE).to(tl.int64)
            mask = offsets < output_size
        else:
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < OUTPUT_SIZE
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
        for fan_idx in tl.range(0, FAN_IN):
            if USE_INT64:
                fan_offset = fan_idx.to(tl.int64) * output_size
            else:
                fan_offset = fan_idx * OUTPUT_SIZE
            values = tl.load(
                input_ptr + fan_offset + offsets,
                mask=mask,
                other=0,
            ).to(tl.int32)
            acc += values
        tl.store(output_ptr + offsets, acc.to(tl.uint8), mask=mask)

    @triton.autotune(configs=_TRITON_CONFIGS, key=["OUTPUT_SIZE"])
    @triton.jit
    def _elementwise_zero_kernel(
        output_ptr,
        OUTPUT_SIZE: tl.constexpr,
        USE_INT64: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if USE_INT64:
            # Match the fan kernel's wide-pointer contract for zero-fill.
            pid = pid.to(tl.int64)
            block_size = tl.full((), BLOCK_SIZE, tl.int64)
            output_size = tl.full((), OUTPUT_SIZE, tl.int64)
            offsets = pid * block_size + tl.arange(0, BLOCK_SIZE).to(tl.int64)
            mask = offsets < output_size
        else:
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < OUTPUT_SIZE
        tl.store(output_ptr + offsets, tl.zeros((BLOCK_SIZE,), dtype=tl.uint8), mask=mask)

    _TRITON_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover - exercised only without triton
    _TRITON_IMPORT_ERROR = exc


def _rounded_fan_in(input_size_bytes: int, output_size_bytes: int) -> int:
    return max(1, int(input_size_bytes / output_size_bytes + 0.5))


def _requires_int64_offsets(input_size_bytes: int, output_size_bytes: int) -> bool:
    """Whether the launched buffers cross Triton's signed-int32 index range."""
    effective_input = (
        _rounded_fan_in(input_size_bytes, output_size_bytes) * output_size_bytes
        if input_size_bytes > 0
        else 0
    )
    return max(effective_input, output_size_bytes) > _INT32_MAX


def profile_elementwise(
    input_size_bytes: int,
    output_size_bytes: int,
) -> ComputeMetrics:
    input_size_bytes = int(input_size_bytes)
    output_size_bytes = int(output_size_bytes)
    if input_size_bytes < 0:
        raise ValueError("input_size_bytes must be >= 0")
    if output_size_bytes <= 0:
        raise ValueError("output_size_bytes must be > 0")

    if _TRITON_IMPORT_ERROR is not None:
        raise ProfilerNotImplemented(
            "triton is required for the triton elementwise runner"
        ) from _TRITON_IMPORT_ERROR

    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented("torch is required for the triton elementwise runner") from exc

    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for the triton elementwise runner")

    try:
        output_tensor = torch.empty(output_size_bytes, dtype=torch.uint8, device="cuda")
        if input_size_bytes == 0:
            fan_in = 0
            effective_input = 1
            kernel_name = _ZERO_KERNEL_NAME
        else:
            fan_in = _rounded_fan_in(input_size_bytes, output_size_bytes)
            effective_input = fan_in * output_size_bytes
            kernel_name = _FAN_KERNEL_NAME
        input_tensor = torch.empty(effective_input, dtype=torch.uint8, device="cuda")
        use_int64 = _requires_int64_offsets(input_size_bytes, output_size_bytes)

        def grid(meta):
            return (triton.cdiv(output_size_bytes, meta["BLOCK_SIZE"]),)

        def run_once():
            if fan_in == 0:
                _elementwise_zero_kernel[grid](
                    output_tensor,
                    OUTPUT_SIZE=output_size_bytes,
                    USE_INT64=use_int64,
                )
            else:
                _elementwise_fan_kernel[grid](
                    input_tensor,
                    output_tensor,
                    OUTPUT_SIZE=output_size_bytes,
                    FAN_IN=fan_in,
                    USE_INT64=use_int64,
                )

        # The @autotune'd Triton kernel benchmarks every config on its FIRST
        # launch for a new (OUTPUT_SIZE, FAN_IN) key. Those trial launches would
        # otherwise fall inside the CUPTI capture window and be summed into the
        # measured kernel time (~17x inflation: a shape is profiled once in a
        # fresh subprocess, so every DB row was the polluted first call). Warm up
        # so autotune resolves OUTSIDE the window — mirrors ref's warmup=100
        # before its capture. Autotune caches after one launch; a few warmups are
        # cheap insurance. (Other runners skip warmup safely because flashinfer /
        # deepgemm kernels aren't Triton-autotuned.)
        time_ms = Timer.cupti(run_once, warmup=5, kernel_name=kernel_name)
        energy_j = Energy.perf(
            run_once,
            warmup=5,
            per_iter_time_ms=time_ms,
        )

        # Bandwidth-bound byte mover (read input + write output). The reference
        # reports no FLOPs for elementwise, so tflops stays 0.
        bytes_accessed = input_size_bytes + output_size_bytes
        bandwidth_gbps = (bytes_accessed / (time_ms / 1000.0)) / 1e9 if time_ms > 0 else 0.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=0.0,
            memory_bandwidth_gbps=float(bandwidth_gbps),
            energy_j=float(energy_j),
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
