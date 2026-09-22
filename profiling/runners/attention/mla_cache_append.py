"""Profiling runners for paged MLA cache append.

The Torch timed callable contains only the two indexed writes that implement
the semantic composite. The production-aligned vLLM callable is its single
fused ``concat_and_cache_mla_kernel`` launch. SGLang has a legacy plain
pre-quantized path and a Kimi-K3 fused bf16-to-fp8 paged path.
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
_PAGE_PLANAR_FP8_CACHE_FORMAT = "page_planar_fp8"
_REQUIRED_GPU = "NVIDIA H200"
_VLLM_SUPPORTED_GPUS = ("NVIDIA H200", "NVIDIA B200")
_VLLM_KERNEL_NAME = "concat_and_cache_mla_kernel"
_SGLANG_SUPPORTED_GPUS = ("NVIDIA B200",)
_SGLANG_Q_HEADS_PER_RANK = 12
_SGLANG_FP8_KERNEL_NAME = "set_mla_kv_concat_q_fp8_kernel"


@dataclass(frozen=True)
class _MlaCacheAppendOperands:
    kv_c: Any
    k_pe_backing: Any
    k_pe: Any
    cache: Any
    slot_mapping: Any
    block_indices: Any
    block_offsets: Any


@dataclass(frozen=True)
class _SglangPagePlanarFp8Operands:
    kv_buffer: Any
    kv_buffer_2d: Any
    locations: Any
    cache_k_nope: Any
    cache_k_rope: Any
    q_nope: Any
    q_rope: Any


def _validate_args(
    num_tokens: int,
    kv_lora_rank: int,
    rope_dim: int,
    block_size: int,
    input_dtype: DType | str,
    kv_dtype: DType | str,
    cache_format: str,
    *,
    allow_fp8_cache: bool = False,
    allow_fp8_input: bool = False,
    allowed_cache_formats: tuple[str, ...] = (_CACHE_FORMAT,),
    backend_name: str = "torch",
) -> tuple[int, int, int, int, DType, DType, str]:
    num_tokens = int(num_tokens)
    kv_lora_rank = int(kv_lora_rank)
    rope_dim = int(rope_dim)
    block_size = int(block_size)
    input_dtype = DType.from_value(input_dtype)
    kv_dtype = DType.from_value(kv_dtype)
    cache_format = str(cache_format)

    if num_tokens <= 0 or kv_lora_rank <= 0 or rope_dim <= 0 or block_size <= 0:
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
            f"{backend_name} mla_cache_append requires "
            "(kv_lora_rank, rope_dim, block_size) == (512, 64, 64), "
            f"got ({kv_lora_rank}, {rope_dim}, {block_size})"
        )
    supported_kv_dtypes = {DType.BF16, DType.FP8_E4M3} if allow_fp8_cache else {DType.BF16}
    supported_input_dtypes = {DType.BF16, DType.FP8_E4M3} if allow_fp8_input else {DType.BF16}
    if input_dtype not in supported_input_dtypes or kv_dtype not in supported_kv_dtypes:
        raise ValueError(
            f"mla_cache_append requires {'BF16 or FP8 E4M3' if allow_fp8_input else 'BF16'} "
            "input and "
            f"{'BF16 or FP8 E4M3' if allow_fp8_cache else 'BF16'} cache, "
            f"got {input_dtype.value} and {kv_dtype.value}"
        )
    if cache_format not in allowed_cache_formats:
        if len(allowed_cache_formats) == 1:
            format_requirement = f"cache_format={allowed_cache_formats[0]!r}"
        else:
            format_requirement = f"cache_format in {tuple(allowed_cache_formats)!r}"
        raise ValueError(
            f"{backend_name} mla_cache_append requires {format_requirement}, got {cache_format!r}"
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
        raise ProfilerNotImplemented("CUDA is required for the torch mla_cache_append backend")
    gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if gpu_name != _REQUIRED_GPU:
        raise ProfilerNotImplemented(
            f"torch mla_cache_append is verified only on {_REQUIRED_GPU}, got {gpu_name}"
        )


def _validate_vllm_cuda_device(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for the mla_cache_append vllm_cuda backend")
    gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if gpu_name not in _VLLM_SUPPORTED_GPUS:
        raise ProfilerNotImplemented(
            "mla_cache_append vllm_cuda is verified only on "
            f"{' or '.join(_VLLM_SUPPORTED_GPUS)}, got {gpu_name}"
        )


def _build_operands(
    torch: Any,
    *,
    num_tokens: int,
    kv_lora_rank: int,
    rope_dim: int,
    block_size: int,
    torch_dtype: Any,
    cache_torch_dtype: Any,
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
        dtype=cache_torch_dtype,
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
        row_width * input_dtype.size_bytes() + row_width * kv_dtype.size_bytes() + 8
    )


def _page_planar_fp8_logical_bytes(
    *,
    num_tokens: int,
    kv_lora_rank: int,
    rope_dim: int,
    input_dtype: DType,
    kv_dtype: DType,
) -> float:
    """Return Kimi-K3's logical latent-KV read/write traffic.

    The fused SGLang callable also builds the query output, but this metric is
    for the cache-append operation and therefore counts only its KV source and
    destination rows. The location tensor is indexing metadata, not payload.
    """
    row_width = kv_lora_rank + rope_dim
    return num_tokens * row_width * (input_dtype.size_bytes() + kv_dtype.size_bytes())


def _build_sglang_page_planar_fp8_operands(
    torch: Any,
    *,
    num_tokens: int,
    kv_lora_rank: int,
    rope_dim: int,
    block_size: int,
    torch_dtype: Any,
    device: Any,
) -> _SglangPagePlanarFp8Operands:
    """Allocate SGLang's flattened paged pool and scattered decode inputs."""
    num_pages = max(256, math.ceil(num_tokens / block_size))
    num_slots = num_pages * block_size
    row_width = kv_lora_rank + rope_dim
    generator = torch.Generator(device=device)
    generator.manual_seed(42)

    # MLATokenToKVPool uses ``size + page_size`` rows for its NHD backing.
    kv_buffer = torch.zeros(
        (num_slots + block_size, 1, row_width),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    cache_k_nope = torch.randn(
        (num_tokens, kv_lora_rank),
        dtype=torch_dtype,
        device=device,
        generator=generator,
    )
    cache_k_rope = torch.randn(
        (num_tokens, rope_dim),
        dtype=torch_dtype,
        device=device,
        generator=generator,
    )
    q_nope = torch.randn(
        (num_tokens, _SGLANG_Q_HEADS_PER_RANK, kv_lora_rank),
        dtype=torch_dtype,
        device=device,
        generator=generator,
    )
    q_rope = torch.randn(
        (num_tokens, _SGLANG_Q_HEADS_PER_RANK, rope_dim),
        dtype=torch_dtype,
        device=device,
        generator=generator,
    )
    locations = torch.randperm(
        num_slots,
        dtype=torch.int64,
        device=device,
        generator=generator,
    )[:num_tokens]
    return _SglangPagePlanarFp8Operands(
        kv_buffer=kv_buffer,
        kv_buffer_2d=kv_buffer.view(kv_buffer.shape[0], -1),
        locations=locations,
        cache_k_nope=cache_k_nope,
        cache_k_rope=cache_k_rope,
        q_nope=q_nope,
        q_rope=q_rope,
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
            cache_torch_dtype=kv_dtype.torch(),
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
        bandwidth_gbps = logical_bytes / (time_ms / 1000.0) / 1e9 if time_ms > 0 else 0.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=0.0,
            memory_bandwidth_gbps=float(bandwidth_gbps),
            energy_j=float(energy_j),
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc


def _validate_sglang_cuda_device(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for mla_cache_append:sglang_cuda")
    gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if gpu_name not in _SGLANG_SUPPORTED_GPUS:
        raise ProfilerNotImplemented(
            "mla_cache_append:sglang_cuda is verified only on "
            f"{' or '.join(_SGLANG_SUPPORTED_GPUS)}, got {gpu_name}"
        )


def profile_mla_cache_append_sglang_cuda(
    num_tokens: int,
    kv_lora_rank: int,
    rope_dim: int,
    block_size: int,
    input_dtype: DType | str,
    kv_dtype: DType | str,
    cache_format: str,
) -> ComputeMetrics:
    """Profile SGLang's plain or fused fp8 MLA cache append path."""
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
        allow_fp8_cache=True,
        allow_fp8_input=True,
        allowed_cache_formats=(_CACHE_FORMAT, _PAGE_PLANAR_FP8_CACHE_FORMAT),
        backend_name="sglang_cuda",
    )
    if cache_format == _PAGE_PLANAR_FP8_CACHE_FORMAT:
        if input_dtype is not DType.BF16 or kv_dtype is not DType.FP8_E4M3:
            raise ValueError(
                "mla_cache_append:sglang_cuda page_planar_fp8 requires "
                "BF16 input and FP8 E4M3 cache"
            )
    elif input_dtype is not DType.FP8_E4M3 or kv_dtype is not DType.FP8_E4M3:
        raise ProfilerNotImplemented(
            "mla_cache_append:sglang_cuda plain requires pre-quantized FP8 input and FP8 cache"
        )
    try:
        import torch

        if cache_format == _PAGE_PLANAR_FP8_CACHE_FORMAT:
            from sglang.kernels.ops.attention.set_mla_kv_concat_q import (
                set_mla_kv_concat_q_fp8,
            )
        else:
            from sglang.kernels.ops.kvcache.mla_buffer import set_mla_kv_buffer_triton
    except (ImportError, OSError) as exc:
        raise ProfilerNotImplemented(
            "mla_cache_append:sglang_cuda requires the SGLang environment"
        ) from exc

    _validate_sglang_cuda_device(torch)
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        if cache_format == _PAGE_PLANAR_FP8_CACHE_FORMAT:
            operands = _build_sglang_page_planar_fp8_operands(
                torch,
                num_tokens=num_tokens,
                kv_lora_rank=kv_lora_rank,
                rope_dim=rope_dim,
                block_size=block_size,
                torch_dtype=input_dtype.torch(),
                device=device,
            )

            def kernel() -> None:
                set_mla_kv_concat_q_fp8(
                    kv_buffer=operands.kv_buffer_2d,
                    loc=operands.locations,
                    cache_k_nope=operands.cache_k_nope,
                    cache_k_rope=operands.cache_k_rope,
                    q_nope=operands.q_nope,
                    q_rope=operands.q_rope,
                    dcp_world_size=1,
                    dcp_rank=0,
                )

            kernel_name = _SGLANG_FP8_KERNEL_NAME
        else:
            generator = torch.Generator(device=device).manual_seed(42)
            num_slots = max(256, math.ceil(num_tokens / block_size) + 1) * block_size
            kv_buffer = torch.zeros(
                (num_slots, 1, kv_lora_rank + rope_dim),
                dtype=torch.uint8,
                device=device,
            )
            cache_k_nope = torch.randint(
                0,
                255,
                (num_tokens, 1, kv_lora_rank),
                dtype=torch.uint8,
                device=device,
                generator=generator,
            )
            cache_k_rope = torch.randint(
                0,
                255,
                (num_tokens, 1, rope_dim),
                dtype=torch.uint8,
                device=device,
                generator=generator,
            )
            locations = torch.randperm(
                num_slots,
                device=device,
                generator=generator,
            )[:num_tokens].to(torch.int64)

            def kernel() -> None:
                set_mla_kv_buffer_triton(kv_buffer, locations, cache_k_nope, cache_k_rope)

            kernel_name = None

        kernel()
        torch.cuda.synchronize()
        if kernel_name is None:
            time_ms = Timer.cupti(kernel, warmup=5)
        else:
            time_ms = Timer.cupti(
                kernel,
                warmup=5,
                kernel_name=kernel_name,
            )
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc

    if cache_format == _PAGE_PLANAR_FP8_CACHE_FORMAT:
        logical_bytes = _page_planar_fp8_logical_bytes(
            num_tokens=num_tokens,
            kv_lora_rank=kv_lora_rank,
            rope_dim=rope_dim,
            input_dtype=input_dtype,
            kv_dtype=kv_dtype,
        )
    else:
        logical_bytes = _logical_bytes(
            num_tokens=num_tokens,
            kv_lora_rank=kv_lora_rank,
            rope_dim=rope_dim,
            input_dtype=input_dtype,
            kv_dtype=kv_dtype,
        )
    bandwidth_gbps = logical_bytes / (time_ms / 1000.0) / 1e9 if time_ms > 0 else 0.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(bandwidth_gbps),
        energy_j=float(energy_j),
    )


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
        allow_fp8_cache=True,
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
            cache_torch_dtype=(
                torch.float8_e4m3fn if kv_dtype is DType.FP8_E4M3 else kv_dtype.torch()
            ),
            device="cuda",
        )
        scale = torch.ones((), dtype=torch.float32, device="cuda")

        def kernel() -> None:
            ops.concat_and_cache_mla(
                operands.kv_c,
                operands.k_pe,
                operands.cache,
                operands.slot_mapping,
                "auto" if kv_dtype is DType.BF16 else "fp8_e4m3",
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
        bandwidth_gbps = logical_bytes / (time_ms / 1000.0) / 1e9 if time_ms > 0 else 0.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=0.0,
            memory_bandwidth_gbps=float(bandwidth_gbps),
            energy_j=float(energy_j),
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
