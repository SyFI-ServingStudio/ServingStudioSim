"""Profiling runners for GLM-5.2's plain MLA cache append.

The Torch timed callable contains only the two indexed writes that implement
the semantic composite. The production-aligned vLLM callable is its single
fused ``concat_and_cache_mla_kernel`` launch.
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

_KV_LORA_RANK = 512
_ROPE_DIM = 64
_BLOCK_SIZE = 64
_CACHE_FORMAT = "plain"
_REQUIRED_GPU = "NVIDIA H200"
_VLLM_KERNEL_NAME = "concat_and_cache_mla_kernel"
_VLLM_CACHE_DTYPE = "auto"


@dataclass(frozen=True)
class _MlaCacheAppendOperands:
    kv_c: Any
    k_pe_backing: Any
    k_pe: Any
    cache: Any
    slot_mapping: Any
    block_indices: Any
    block_offsets: Any


def _validate_args(
    num_tokens: int,
    kv_lora_rank: int,
    rope_dim: int,
    block_size: int,
    input_dtype: DType | str,
    kv_dtype: DType | str,
    cache_format: str,
) -> tuple[int, int, int, int, DType, DType, str]:
    num_tokens = int(num_tokens)
    kv_lora_rank = int(kv_lora_rank)
    rope_dim = int(rope_dim)
    block_size = int(block_size)
    input_dtype = DType.from_value(input_dtype)
    kv_dtype = DType.from_value(kv_dtype)
    cache_format = str(cache_format)

    if (
        num_tokens <= 0
        or kv_lora_rank <= 0
        or rope_dim <= 0
        or block_size <= 0
    ):
        raise ValueError(
            "num_tokens, kv_lora_rank, rope_dim, and block_size must be > 0, "
            f"got {num_tokens}, {kv_lora_rank}, {rope_dim}, and {block_size}"
        )
    if (kv_lora_rank, rope_dim, block_size) != (
        _KV_LORA_RANK,
        _ROPE_DIM,
        _BLOCK_SIZE,
    ):
        raise ValueError(
            "torch mla_cache_append requires "
            "(kv_lora_rank, rope_dim, block_size) == (512, 64, 64), "
            f"got ({kv_lora_rank}, {rope_dim}, {block_size})"
        )
    if input_dtype is not DType.BF16 or kv_dtype is not DType.BF16:
        raise ValueError(
            "torch mla_cache_append requires input_dtype=kv_dtype=bf16, "
            f"got {input_dtype.value} and {kv_dtype.value}"
        )
    if cache_format != _CACHE_FORMAT:
        raise ValueError(
            "torch mla_cache_append requires cache_format='plain', "
            f"got {cache_format!r}"
        )
    return (
        num_tokens,
        kv_lora_rank,
        rope_dim,
        block_size,
        input_dtype,
        kv_dtype,
        cache_format,
    )


def _validate_cuda_device(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(
            "CUDA is required for the torch mla_cache_append backend"
        )
    gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if gpu_name != _REQUIRED_GPU:
        raise ProfilerNotImplemented(
            "torch mla_cache_append is verified only on "
            f"{_REQUIRED_GPU}, got {gpu_name}"
        )


def _validate_vllm_cuda_device(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(
            "CUDA is required for the mla_cache_append vllm_cuda backend"
        )
    gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if gpu_name != _REQUIRED_GPU:
        raise ProfilerNotImplemented(
            "mla_cache_append vllm_cuda is verified only on "
            f"{_REQUIRED_GPU}, got {gpu_name}"
        )


def _build_operands(
    torch: Any,
    *,
    num_tokens: int,
    kv_lora_rank: int,
    rope_dim: int,
    block_size: int,
    torch_dtype: Any,
    device: str,
) -> _MlaCacheAppendOperands:
    """Allocate exact plain-cache operands and deterministic scattered slots."""
    kv_c = torch.randn(
        (num_tokens, kv_lora_rank),
        dtype=torch_dtype,
        device=device,
    )
    k_pe_backing = torch.randn(
        (num_tokens, 1, rope_dim),
        dtype=torch_dtype,
        device=device,
    )
    k_pe = k_pe_backing.squeeze(1)

    num_blocks = max(256, math.ceil(num_tokens / block_size) + 1)
    num_slots = num_blocks * block_size
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    slot_mapping = torch.randperm(
        num_slots,
        dtype=torch.int64,
        device=device,
        generator=generator,
    )[:num_tokens]
    block_indices = torch.div(
        slot_mapping,
        block_size,
        rounding_mode="floor",
    )
    block_offsets = slot_mapping % block_size
    cache = torch.full(
        (num_blocks, block_size, kv_lora_rank + rope_dim),
        -1,
        dtype=torch_dtype,
        device=device,
    )
    return _MlaCacheAppendOperands(
        kv_c=kv_c,
        k_pe_backing=k_pe_backing,
        k_pe=k_pe,
        cache=cache,
        slot_mapping=slot_mapping,
        block_indices=block_indices,
        block_offsets=block_offsets,
    )


def _write_cache(operands: _MlaCacheAppendOperands) -> None:
    """Execute only the two indexed cache writes in the semantic composite."""
    kv_lora_rank = operands.kv_c.shape[1]
    operands.cache[
        operands.block_indices,
        operands.block_offsets,
        :kv_lora_rank,
    ] = operands.kv_c
    operands.cache[
        operands.block_indices,
        operands.block_offsets,
        kv_lora_rank:,
    ] = operands.k_pe


def _logical_bytes(
    *,
    num_tokens: int,
    kv_lora_rank: int,
    rope_dim: int,
    input_dtype: DType,
    kv_dtype: DType,
) -> float:
    """Return logical read/write traffic, excluding allocation/initialization."""
    row_width = kv_lora_rank + rope_dim
    return num_tokens * (
        row_width * input_dtype.size_bytes()
        + row_width * kv_dtype.size_bytes()
        + 8
    )


def profile_mla_cache_append_torch(
    num_tokens: int,
    kv_lora_rank: int,
    rope_dim: int,
    block_size: int,
    input_dtype: DType | str,
    kv_dtype: DType | str,
    cache_format: str,
) -> ComputeMetrics:
    """Profile the complete two-launch Torch MLA cache-append composite."""
    (
        num_tokens,
        kv_lora_rank,
        rope_dim,
        block_size,
        input_dtype,
        kv_dtype,
        _cache_format,
    ) = _validate_args(
        num_tokens,
        kv_lora_rank,
        rope_dim,
        block_size,
        input_dtype,
        kv_dtype,
        cache_format,
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "torch is required for the torch mla_cache_append backend"
        ) from exc

    _validate_cuda_device(torch)

    try:
        operands = _build_operands(
            torch,
            num_tokens=num_tokens,
            kv_lora_rank=kv_lora_rank,
            rope_dim=rope_dim,
            block_size=block_size,
            torch_dtype=input_dtype.torch(),
            device="cuda",
        )

        def kernel() -> None:
            operands.cache[
                operands.block_indices,
                operands.block_offsets,
                :kv_lora_rank,
            ] = operands.kv_c
            operands.cache[
                operands.block_indices,
                operands.block_offsets,
                kv_lora_rank:,
            ] = operands.k_pe

        # This intentionally measures the total two-launch Torch composite.
        time_ms = Timer.cuda_event(kernel, warmup=5)
        energy_j = Energy.perf(
            kernel,
            warmup=5,
            per_iter_time_ms=time_ms,
        )
        logical_bytes = _logical_bytes(
            num_tokens=num_tokens,
            kv_lora_rank=kv_lora_rank,
            rope_dim=rope_dim,
            input_dtype=input_dtype,
            kv_dtype=kv_dtype,
        )
        bandwidth_gbps = (
            logical_bytes / (time_ms / 1000.0) / 1e9 if time_ms > 0 else 0.0
        )
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=0.0,
            memory_bandwidth_gbps=float(bandwidth_gbps),
            energy_j=float(energy_j),
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc


def profile_mla_cache_append_vllm_cuda(
    num_tokens: int,
    kv_lora_rank: int,
    rope_dim: int,
    block_size: int,
    input_dtype: DType | str,
    kv_dtype: DType | str,
    cache_format: str,
) -> ComputeMetrics:
    """Profile vLLM's one-launch fused plain MLA cache append."""
    (
        num_tokens,
        kv_lora_rank,
        rope_dim,
        block_size,
        input_dtype,
        kv_dtype,
        _cache_format,
    ) = _validate_args(
        num_tokens,
        kv_lora_rank,
        rope_dim,
        block_size,
        input_dtype,
        kv_dtype,
        cache_format,
    )
    try:
        import torch
        from vllm import _custom_ops as ops
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "the instrumented vLLM environment is required for "
            "the mla_cache_append vllm_cuda backend"
        ) from exc

    _validate_vllm_cuda_device(torch)

    try:
        operands = _build_operands(
            torch,
            num_tokens=num_tokens,
            kv_lora_rank=kv_lora_rank,
            rope_dim=rope_dim,
            block_size=block_size,
            torch_dtype=input_dtype.torch(),
            device="cuda",
        )
        scale = torch.ones((), dtype=torch.float32, device="cuda")

        def kernel() -> None:
            ops.concat_and_cache_mla(
                operands.kv_c,
                operands.k_pe,
                operands.cache,
                operands.slot_mapping,
                _VLLM_CACHE_DTYPE,
                scale,
            )

        # Allocation and layout setup stay outside the callable. The CUPTI
        # filter selects only vLLM's one fused device launch.
        time_ms = Timer.cupti(kernel, kernel_name=_VLLM_KERNEL_NAME)
        energy_j = Energy.perf(
            kernel,
            warmup=5,
            per_iter_time_ms=time_ms,
        )

        # Logical traffic describes semantic reads/writes, not physical memory
        # transactions performed by the fused CUDA implementation.
        logical_bytes = _logical_bytes(
            num_tokens=num_tokens,
            kv_lora_rank=kv_lora_rank,
            rope_dim=rope_dim,
            input_dtype=input_dtype,
            kv_dtype=kv_dtype,
        )
        bandwidth_gbps = (
            logical_bytes / (time_ms / 1000.0) / 1e9 if time_ms > 0 else 0.0
        )
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=0.0,
            memory_bandwidth_gbps=float(bandwidth_gbps),
            energy_j=float(energy_j),
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
