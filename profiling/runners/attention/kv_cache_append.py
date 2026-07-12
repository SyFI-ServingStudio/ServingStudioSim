"""Runners for appending newly produced K/V rows to a paged cache.

The Torch path is the executable semantic reference.  The vLLM path measures
the exact CUDA kernel observed in alignment traces:
``vllm::reshape_and_cache_flash_kernel``.  Tensor allocation and slot-map
construction remain outside both timing windows.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_VLLM_KERNEL_NAME = "reshape_and_cache_flash_kernel"
_SUPPORTED_LAYOUTS = {"NHD", "HND"}
_SUPPORTED_SCALE_GRANULARITIES = {"tensor", "head"}


def _validate_shape(
    *,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    cache_layout: str,
    scale_granularity: str,
    num_tokens: int,
) -> tuple[str, str]:
    if num_kv_heads <= 0 or head_dim <= 0 or block_size <= 0:
        raise ValueError("num_kv_heads, head_dim, and block_size must be > 0")
    if num_tokens <= 0:
        raise ValueError("num_tokens must be > 0")
    layout = str(cache_layout).upper()
    if layout not in _SUPPORTED_LAYOUTS:
        raise ValueError(f"cache_layout must be one of {sorted(_SUPPORTED_LAYOUTS)}")
    granularity = str(scale_granularity).lower()
    if granularity not in _SUPPORTED_SCALE_GRANULARITIES:
        raise ValueError(
            f"scale_granularity must be one of {sorted(_SUPPORTED_SCALE_GRANULARITIES)}"
        )
    return layout, granularity


def _make_inputs(
    torch: Any,
    *,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    input_dtype: DType,
    kv_dtype: DType,
    cache_layout: str,
    scale_granularity: str,
    num_tokens: int,
    use_vllm_cache_factory: bool,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    input_torch_dtype = input_dtype.torch()
    key = torch.randn(
        num_tokens,
        num_kv_heads,
        head_dim,
        dtype=input_torch_dtype,
        device="cuda",
    )
    value = torch.randn_like(key)

    # Keep more pages than the strict minimum for small decode batches, then
    # scatter unique slots through them.  This mirrors serving better than a
    # compact 0..T write while keeping allocation bounded for long prefills.
    num_blocks = max(256, math.ceil(num_tokens / block_size))
    num_slots = num_blocks * block_size
    slot_mapping = torch.randperm(num_slots, device="cuda", dtype=torch.int64)[:num_tokens]

    if use_vllm_cache_factory:
        from vllm.utils.torch_utils import create_kv_caches_with_random_flash

        vllm_cache_dtype = "fp8" if kv_dtype is DType.FP8_E4M3 else "auto"
        key_caches, value_caches = create_kv_caches_with_random_flash(
            num_blocks,
            block_size,
            1,
            num_kv_heads,
            head_dim,
            vllm_cache_dtype,
            input_torch_dtype,
            seed=42,
            device="cuda",
            cache_layout=cache_layout,
        )
        key_cache, value_cache = key_caches[0], value_caches[0]
    else:
        if kv_dtype is not input_dtype:
            raise ValueError(
                "torch reference requires input_dtype == kv_dtype; "
                "use vllm_cuda for dtype conversion"
            )
        logical_shape = (num_blocks, block_size, num_kv_heads, head_dim)
        if cache_layout == "NHD":
            key_cache = torch.empty(logical_shape, dtype=input_torch_dtype, device="cuda")
            value_cache = torch.empty_like(key_cache)
        else:
            # vLLM keeps the public logical shape [B, page, H, D] and expresses
            # HND only through strides. Mirror create_kv_caches_with_random_flash
            # instead of swapping the public dimensions.
            physical_shape = (num_blocks, num_kv_heads, block_size, head_dim)
            key_cache = torch.empty(physical_shape, dtype=input_torch_dtype, device="cuda").permute(
                0, 2, 1, 3
            )
            value_cache = torch.empty(
                physical_shape, dtype=input_torch_dtype, device="cuda"
            ).permute(0, 2, 1, 3)

    scale_shape = (1,) if scale_granularity == "tensor" else (num_kv_heads,)
    k_scale = torch.ones(scale_shape, dtype=torch.float32, device="cuda")
    v_scale = torch.ones_like(k_scale)
    return key, value, key_cache, value_cache, slot_mapping, k_scale, v_scale


def _metrics(
    kernel: Callable[[], object],
    *,
    kernel_name: str | None,
    num_kv_heads: int,
    head_dim: int,
    num_tokens: int,
    input_dtype: DType,
    kv_dtype: DType,
) -> ComputeMetrics:
    if kernel_name is None:
        # The Torch reference is two indexed writes; CUDA events intentionally
        # measure their combined semantic operation rather than one launch.
        time_ms = Timer.cuda_event(kernel, warmup=5)
    else:
        time_ms = Timer.cupti(kernel, warmup=5, kernel_name=kernel_name)
    energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    elements = 2 * num_tokens * num_kv_heads * head_dim  # K + V
    bytes_accessed = (
        elements * input_dtype.size_bytes()
        + elements * kv_dtype.size_bytes()
        + num_tokens * 8  # int64 slot_mapping
    )
    bandwidth_gbps = (bytes_accessed / (time_ms / 1000.0)) / 1e9 if time_ms > 0 else 0.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(bandwidth_gbps),
        energy_j=float(energy_j),
    )


def profile_kv_cache_append_torch(
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    input_dtype: DType | str,
    kv_dtype: DType | str,
    cache_layout: str,
    scale_granularity: str,
    num_tokens: int,
) -> ComputeMetrics:
    input_dtype = DType.from_value(input_dtype)
    kv_dtype = DType.from_value(kv_dtype)
    cache_layout, scale_granularity = _validate_shape(
        num_kv_heads=int(num_kv_heads),
        head_dim=int(head_dim),
        block_size=int(block_size),
        cache_layout=cache_layout,
        scale_granularity=scale_granularity,
        num_tokens=int(num_tokens),
    )
    if scale_granularity != "tensor":
        raise ValueError("torch reference currently supports tensor scale granularity only")
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented("torch is required for the Torch reference") from exc
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for KV-cache append profiling")

    try:
        key, value, key_cache, value_cache, slot_mapping, _, _ = _make_inputs(
            torch,
            num_kv_heads=int(num_kv_heads),
            head_dim=int(head_dim),
            block_size=int(block_size),
            input_dtype=input_dtype,
            kv_dtype=kv_dtype,
            cache_layout=cache_layout,
            scale_granularity=scale_granularity,
            num_tokens=int(num_tokens),
            use_vllm_cache_factory=False,
        )
        block_indices = torch.div(slot_mapping, int(block_size), rounding_mode="floor")
        block_offsets = slot_mapping % int(block_size)

        def kernel():
            key_cache[block_indices, block_offsets] = key
            value_cache[block_indices, block_offsets] = value

        return _metrics(
            kernel,
            kernel_name=None,
            num_kv_heads=int(num_kv_heads),
            head_dim=int(head_dim),
            num_tokens=int(num_tokens),
            input_dtype=input_dtype,
            kv_dtype=kv_dtype,
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc


def profile_kv_cache_append_vllm_cuda(
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    input_dtype: DType | str,
    kv_dtype: DType | str,
    cache_layout: str,
    scale_granularity: str,
    num_tokens: int,
) -> ComputeMetrics:
    input_dtype = DType.from_value(input_dtype)
    kv_dtype = DType.from_value(kv_dtype)
    cache_layout, scale_granularity = _validate_shape(
        num_kv_heads=int(num_kv_heads),
        head_dim=int(head_dim),
        block_size=int(block_size),
        cache_layout=cache_layout,
        scale_granularity=scale_granularity,
        num_tokens=int(num_tokens),
    )
    if kv_dtype not in {input_dtype, DType.FP8_E4M3}:
        raise ValueError("vllm_cuda supports same-dtype cache or fp8_e4m3 cache")
    if kv_dtype is DType.FP8_E4M3 and head_dim % 16 != 0:
        raise ValueError("fp8 KV cache requires head_dim divisible by 16")
    try:
        import torch
        from vllm import _custom_ops as ops
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "the instrumented vLLM environment is required for vllm_cuda"
        ) from exc
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for KV-cache append profiling")

    try:
        key, value, key_cache, value_cache, slot_mapping, k_scale, v_scale = _make_inputs(
            torch,
            num_kv_heads=int(num_kv_heads),
            head_dim=int(head_dim),
            block_size=int(block_size),
            input_dtype=input_dtype,
            kv_dtype=kv_dtype,
            cache_layout=cache_layout,
            scale_granularity=scale_granularity,
            num_tokens=int(num_tokens),
            use_vllm_cache_factory=True,
        )
        vllm_cache_dtype = "fp8" if kv_dtype is DType.FP8_E4M3 else "auto"

        def kernel():
            return ops.reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                vllm_cache_dtype,
                k_scale,
                v_scale,
            )

        return _metrics(
            kernel,
            kernel_name=_VLLM_KERNEL_NAME,
            num_kv_heads=int(num_kv_heads),
            head_dim=int(head_dim),
            num_tokens=int(num_tokens),
            input_dtype=input_dtype,
            kv_dtype=kv_dtype,
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
