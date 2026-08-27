"""Torch profiling runner for the DSA index-key cache append."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_INDEX_DIM = 128
_BLOCK_SIZE = 64
_QUANT_BLOCK_SIZE = 128
_INPUT_DTYPE = DType.BF16
_CACHE_DTYPE = DType.FP8_E4M3
_SCALE_FORMAT = "ue8m0"
_CACHE_FORMAT = "page_planar_fp8_fp32_scale"
_REQUIRED_GPU = "NVIDIA H200"
_VLLM_SUPPORTED_GPUS = ("NVIDIA H200", "NVIDIA B200")
_VLLM_KERNEL_NAME = "indexer_k_quant_and_cache_kernel"
_FP8_E4M3_MAX = 448.0
_AMAX_FLOOR = 1e-4


@dataclass(frozen=True)
class _DsaIndexCacheAppendOperands:
    k: Any
    cache: Any
    slot_mapping: Any
    block_indices: Any
    block_offsets: Any
    key_plane: Any
    scale_plane: Any


def _validate_args(
    num_tokens: int,
    index_dim: int,
    block_size: int,
    quant_block_size: int,
    input_dtype: DType | str,
    cache_dtype: DType | str,
    scale_format: str,
    cache_format: str,
) -> tuple[int, int, int, int, DType, DType, str, str]:
    num_tokens = int(num_tokens)
    index_dim = int(index_dim)
    block_size = int(block_size)
    quant_block_size = int(quant_block_size)
    input_dtype = DType.from_value(input_dtype)
    cache_dtype = DType.from_value(cache_dtype)
    scale_format = str(scale_format)
    cache_format = str(cache_format)

    if num_tokens <= 0 or index_dim <= 0 or block_size <= 0 or quant_block_size <= 0:
        raise ValueError(
            "num_tokens, index_dim, block_size, and quant_block_size must be > 0, "
            f"got {num_tokens}, {index_dim}, {block_size}, and {quant_block_size}"
        )
    if (index_dim, block_size, quant_block_size) != (
        _INDEX_DIM,
        _BLOCK_SIZE,
        _QUANT_BLOCK_SIZE,
    ):
        raise ValueError(
            "torch dsa_index_cache_append requires "
            "(index_dim, block_size, quant_block_size) == (128, 64, 128), "
            f"got ({index_dim}, {block_size}, {quant_block_size})"
        )
    if input_dtype is not _INPUT_DTYPE or cache_dtype is not _CACHE_DTYPE:
        raise ValueError(
            "torch dsa_index_cache_append requires "
            "input_dtype=bf16 and cache_dtype=fp8_e4m3, "
            f"got {input_dtype.value} and {cache_dtype.value}"
        )
    if scale_format != _SCALE_FORMAT:
        raise ValueError(
            f"torch dsa_index_cache_append requires scale_format='ue8m0', got {scale_format!r}"
        )
    if cache_format != _CACHE_FORMAT:
        raise ValueError(
            "torch dsa_index_cache_append requires "
            "cache_format='page_planar_fp8_fp32_scale', "
            f"got {cache_format!r}"
        )
    return (
        num_tokens,
        index_dim,
        block_size,
        quant_block_size,
        input_dtype,
        cache_dtype,
        scale_format,
        cache_format,
    )


def _validate_cuda_device(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(
            "CUDA is required for the torch dsa_index_cache_append backend"
        )
    gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if gpu_name != _REQUIRED_GPU:
        raise ProfilerNotImplemented(
            f"torch dsa_index_cache_append is verified only on {_REQUIRED_GPU}, got {gpu_name}"
        )


def _validate_vllm_cuda_device(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(
            "CUDA is required for the dsa_index_cache_append vllm_cuda backend"
        )
    gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if gpu_name not in _VLLM_SUPPORTED_GPUS:
        raise ProfilerNotImplemented(
            "dsa_index_cache_append vllm_cuda is verified only on "
            f"{' or '.join(_VLLM_SUPPORTED_GPUS)}, got {gpu_name}"
        )


def _build_operands(
    torch: Any,
    *,
    num_tokens: int,
    index_dim: int,
    block_size: int,
    quant_block_size: int,
    torch_dtype: Any,
    device: str,
) -> _DsaIndexCacheAppendOperands:
    """Allocate exact page-planar operands and deterministic scattered slots."""
    num_groups = index_dim // quant_block_size
    cache_width = index_dim + num_groups * 4
    num_blocks = max(256, math.ceil(num_tokens / block_size) + 1)
    num_slots = num_blocks * block_size

    k = torch.randn(
        (num_tokens, index_dim),
        dtype=torch_dtype,
        device=device,
    )
    cache = torch.full(
        (num_blocks, block_size, cache_width),
        0xA5,
        dtype=torch.uint8,
        device=device,
    )
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

    pages = cache.view(num_blocks, -1)
    key_plane = pages[:, : block_size * index_dim].view(
        num_blocks,
        block_size,
        index_dim,
    )
    scale_plane = pages[:, block_size * index_dim :].view(
        num_blocks,
        block_size,
        num_groups,
        4,
    )
    return _DsaIndexCacheAppendOperands(
        k=k,
        cache=cache,
        slot_mapping=slot_mapping,
        block_indices=block_indices,
        block_offsets=block_offsets,
        key_plane=key_plane,
        scale_plane=scale_plane,
    )


def _write_cache(
    torch: Any,
    operands: _DsaIndexCacheAppendOperands,
    *,
    quant_block_size: int,
) -> None:
    """Execute the complete vectorized Torch quantize-and-write composite."""
    num_tokens, index_dim = operands.k.shape
    num_groups = index_dim // quant_block_size
    grouped_k_fp32 = operands.k.view(
        num_tokens,
        num_groups,
        quant_block_size,
    ).float()
    amax = grouped_k_fp32.abs().amax(dim=-1)
    base_scale = torch.clamp_min(amax, _AMAX_FLOOR) / _FP8_E4M3_MAX
    scales = torch.exp2(torch.ceil(torch.log2(base_scale)))
    quantized = (grouped_k_fp32 / scales.unsqueeze(-1)).to(torch.float8_e4m3fn).view(torch.uint8)

    operands.key_plane[
        operands.block_indices,
        operands.block_offsets,
    ] = quantized.reshape(num_tokens, index_dim)
    operands.scale_plane[
        operands.block_indices,
        operands.block_offsets,
    ] = scales.contiguous().view(torch.uint8).reshape(num_tokens, num_groups, 4)


def _logical_bytes(
    *,
    num_tokens: int,
    index_dim: int,
    quant_block_size: int,
    input_dtype: DType,
    cache_dtype: DType,
) -> float:
    """Return semantic traffic, excluding Torch intermediates/cache-line effects."""
    num_groups = index_dim // quant_block_size
    return num_tokens * (
        index_dim * input_dtype.size_bytes()
        + 8
        + index_dim * cache_dtype.size_bytes()
        + num_groups * 4
    )


def profile_dsa_index_cache_append_torch(
    num_tokens: int,
    index_dim: int,
    block_size: int,
    quant_block_size: int,
    input_dtype: DType | str,
    cache_dtype: DType | str,
    scale_format: str,
    cache_format: str,
) -> ComputeMetrics:
    """Profile the complete multi-launch Torch semantic composite."""
    (
        num_tokens,
        index_dim,
        block_size,
        quant_block_size,
        input_dtype,
        cache_dtype,
        _scale_format,
        _cache_format,
    ) = _validate_args(
        num_tokens,
        index_dim,
        block_size,
        quant_block_size,
        input_dtype,
        cache_dtype,
        scale_format,
        cache_format,
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "torch is required for the torch dsa_index_cache_append backend"
        ) from exc

    _validate_cuda_device(torch)

    try:
        operands = _build_operands(
            torch,
            num_tokens=num_tokens,
            index_dim=index_dim,
            block_size=block_size,
            quant_block_size=quant_block_size,
            torch_dtype=input_dtype.torch(),
            device="cuda",
        )

        def kernel() -> None:
            _write_cache(
                torch,
                operands,
                quant_block_size=quant_block_size,
            )

        # This intentionally measures the entire multi-launch Torch semantic
        # composite, not vLLM's production fused implementation.
        time_ms = Timer.cuda_event(kernel, warmup=5)
        energy_j = Energy.perf(
            kernel,
            warmup=5,
            per_iter_time_ms=time_ms,
        )
        logical_bytes = _logical_bytes(
            num_tokens=num_tokens,
            index_dim=index_dim,
            quant_block_size=quant_block_size,
            input_dtype=input_dtype,
            cache_dtype=cache_dtype,
        )
        bandwidth_gbps = logical_bytes / (time_ms / 1000.0) / 1e9 if time_ms > 0 else 0.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=0.0,
            memory_bandwidth_gbps=float(bandwidth_gbps),
            energy_j=float(energy_j),
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc


def profile_dsa_index_cache_append_vllm_cuda(
    num_tokens: int,
    index_dim: int,
    block_size: int,
    quant_block_size: int,
    input_dtype: DType | str,
    cache_dtype: DType | str,
    scale_format: str,
    cache_format: str,
) -> ComputeMetrics:
    """Profile vLLM's one-launch fused DSA index-key cache append."""
    (
        num_tokens,
        index_dim,
        block_size,
        quant_block_size,
        input_dtype,
        cache_dtype,
        scale_format,
        _cache_format,
    ) = _validate_args(
        num_tokens,
        index_dim,
        block_size,
        quant_block_size,
        input_dtype,
        cache_dtype,
        scale_format,
        cache_format,
    )
    try:
        import torch
        from vllm import _custom_ops as ops
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "the instrumented vLLM environment is required for "
            "the dsa_index_cache_append vllm_cuda backend"
        ) from exc

    _validate_vllm_cuda_device(torch)

    try:
        operands = _build_operands(
            torch,
            num_tokens=num_tokens,
            index_dim=index_dim,
            block_size=block_size,
            quant_block_size=quant_block_size,
            torch_dtype=input_dtype.torch(),
            device="cuda",
        )

        def kernel() -> None:
            ops.indexer_k_quant_and_cache(
                operands.k,
                operands.cache,
                operands.slot_mapping,
                quant_block_size,
                scale_format,
            )

        # Allocation and page-layout construction stay outside timing. The
        # filter selects only vLLM's one fused CUDA launch.
        time_ms = Timer.cupti(kernel, kernel_name=_VLLM_KERNEL_NAME)
        energy_j = Energy.perf(
            kernel,
            warmup=5,
            per_iter_time_ms=time_ms,
        )

        # This is semantic traffic, not the fused kernel's physical memory
        # transactions or cache-line traffic.
        logical_bytes = _logical_bytes(
            num_tokens=num_tokens,
            index_dim=index_dim,
            quant_block_size=quant_block_size,
            input_dtype=input_dtype,
            cache_dtype=cache_dtype,
        )
        bandwidth_gbps = logical_bytes / (time_ms / 1000.0) / 1e9 if time_ms > 0 else 0.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=0.0,
            memory_bandwidth_gbps=float(bandwidth_gbps),
            energy_j=float(energy_j),
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
