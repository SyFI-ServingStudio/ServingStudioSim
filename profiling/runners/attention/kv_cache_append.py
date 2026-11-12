"""Runners for appending newly produced K/V rows to a paged cache.

The Torch path is the executable semantic reference.  The ``vllm_cuda`` path
measures the exact CUDA kernel observed in alignment traces:
``vllm::reshape_and_cache_flash_kernel``, which writes separate K and V caches
of one head size.  The ``vllm_triton_diffkv`` path measures vLLM's
``triton_reshape_and_cache_flash_diffkv``, which packs K and V of different
head sizes into one cache, ``[num_blocks, block_size, num_kv_heads,
head_dim + head_dim_v]``.  Tensor allocation and slot-map construction remain
outside every timing window.
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
_VLLM_DIFFKV_KERNEL_NAME = "reshape_and_cache_kernel_flash_diffkv"
_SUPPORTED_LAYOUTS = {"NHD", "HND"}
_SUPPORTED_SCALE_GRANULARITIES = {"tensor", "head"}


def _validate_shape(
    *,
    num_kv_heads: int,
    head_dim: int,
    head_dim_v: int,
    block_size: int,
    cache_layout: str,
    scale_granularity: str,
    num_tokens: int,
) -> tuple[str, str]:
    if num_kv_heads <= 0 or head_dim <= 0 or head_dim_v <= 0 or block_size <= 0:
        raise ValueError("num_kv_heads, head_dim, head_dim_v, and block_size must be > 0")
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


def _num_blocks(num_tokens: int, block_size: int) -> int:
    # Keep more pages than the strict minimum for small decode batches, then
    # scatter unique slots through them.  This mirrors serving better than a
    # compact 0..T write while keeping allocation bounded for long prefills.
    return max(256, math.ceil(num_tokens / block_size))


def _slot_mapping(torch: Any, *, num_tokens: int, block_size: int) -> Any:
    num_slots = _num_blocks(num_tokens, block_size) * block_size
    return torch.randperm(num_slots, device="cuda", dtype=torch.int64)[:num_tokens]


def _paged_cache(
    torch: Any,
    *,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    width: int,
    dtype: Any,
    cache_layout: str,
) -> Any:
    """A ``[num_blocks, block_size, num_kv_heads, width]`` cache view.

    vLLM keeps this public logical shape and expresses HND only through strides
    (``create_kv_caches_with_random_flash``, the diffkv ``kv_cache.transpose``),
    so HND allocates ``[B, H, page, width]`` and permutes the view.
    """
    if cache_layout == "NHD":
        return torch.empty(
            (num_blocks, block_size, num_kv_heads, width), dtype=dtype, device="cuda"
        )
    physical = torch.empty(
        (num_blocks, num_kv_heads, block_size, width), dtype=dtype, device="cuda"
    )
    return physical.permute(0, 2, 1, 3)


def _new_rows(
    torch: Any,
    *,
    num_tokens: int,
    num_kv_heads: int,
    head_dim: int,
    head_dim_v: int,
    input_dtype: DType,
) -> tuple[Any, Any]:
    dtype = input_dtype.torch()
    key = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    value = torch.randn(num_tokens, num_kv_heads, head_dim_v, dtype=dtype, device="cuda")
    return key, value


def _scales(torch: Any, *, num_kv_heads: int, scale_granularity: str) -> tuple[Any, Any]:
    scale_shape = (1,) if scale_granularity == "tensor" else (num_kv_heads,)
    k_scale = torch.ones(scale_shape, dtype=torch.float32, device="cuda")
    return k_scale, torch.ones_like(k_scale)


def _metrics(
    kernel: Callable[[], object],
    *,
    kernel_name: str | None,
    num_kv_heads: int,
    head_dim: int,
    head_dim_v: int,
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
    elements = num_tokens * num_kv_heads * (head_dim + head_dim_v)  # K + V
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
    head_dim_v: int | None = None,
) -> ComputeMetrics:
    input_dtype = DType.from_value(input_dtype)
    kv_dtype = DType.from_value(kv_dtype)
    num_kv_heads, head_dim = int(num_kv_heads), int(head_dim)
    # Absent head_dim_v is symmetric K/V (the arg's `same_as`).
    head_dim_v = head_dim if head_dim_v is None else int(head_dim_v)
    block_size, num_tokens = int(block_size), int(num_tokens)
    cache_layout, scale_granularity = _validate_shape(
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        block_size=block_size,
        cache_layout=cache_layout,
        scale_granularity=scale_granularity,
        num_tokens=num_tokens,
    )
    if scale_granularity != "tensor":
        raise ValueError("torch reference currently supports tensor scale granularity only")
    if kv_dtype is not input_dtype:
        raise ValueError(
            "torch reference requires input_dtype == kv_dtype; use vllm_cuda for dtype conversion"
        )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented("torch is required for the Torch reference") from exc

    try:
        key, value = _new_rows(
            torch,
            num_tokens=num_tokens,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            input_dtype=input_dtype,
        )
        slot_mapping = _slot_mapping(torch, num_tokens=num_tokens, block_size=block_size)
        caches = [
            _paged_cache(
                torch,
                num_blocks=_num_blocks(num_tokens, block_size),
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                width=width,
                dtype=input_dtype.torch(),
                cache_layout=cache_layout,
            )
            for width in (head_dim, head_dim_v)
        ]
        key_cache, value_cache = caches
        block_indices = torch.div(slot_mapping, block_size, rounding_mode="floor")
        block_offsets = slot_mapping % block_size

        def kernel():
            key_cache[block_indices, block_offsets] = key
            value_cache[block_indices, block_offsets] = value

        return _metrics(
            kernel,
            kernel_name=None,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            num_tokens=num_tokens,
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
    head_dim_v: int | None = None,
) -> ComputeMetrics:
    input_dtype = DType.from_value(input_dtype)
    kv_dtype = DType.from_value(kv_dtype)
    num_kv_heads, head_dim = int(num_kv_heads), int(head_dim)
    # Absent head_dim_v is symmetric K/V (the arg's `same_as`).
    head_dim_v = head_dim if head_dim_v is None else int(head_dim_v)
    block_size, num_tokens = int(block_size), int(num_tokens)
    cache_layout, scale_granularity = _validate_shape(
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        block_size=block_size,
        cache_layout=cache_layout,
        scale_granularity=scale_granularity,
        num_tokens=num_tokens,
    )
    if head_dim_v != head_dim:
        raise ValueError(
            "vllm_cuda writes K and V caches of one head size; use vllm_triton_diffkv "
            "for head_dim_v != head_dim"
        )
    if kv_dtype not in {input_dtype, DType.FP8_E4M3}:
        raise ValueError("vllm_cuda supports same-dtype cache or fp8_e4m3 cache")
    if kv_dtype is DType.FP8_E4M3 and head_dim % 16 != 0:
        raise ValueError("fp8 KV cache requires head_dim divisible by 16")
    try:
        import torch
        from vllm import _custom_ops as ops
        from vllm.utils.torch_utils import create_kv_caches_with_random_flash
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "the instrumented vLLM environment is required for vllm_cuda"
        ) from exc

    try:
        key, value = _new_rows(
            torch,
            num_tokens=num_tokens,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            input_dtype=input_dtype,
        )
        slot_mapping = _slot_mapping(torch, num_tokens=num_tokens, block_size=block_size)
        vllm_cache_dtype = "fp8" if kv_dtype is DType.FP8_E4M3 else "auto"
        key_caches, value_caches = create_kv_caches_with_random_flash(
            _num_blocks(num_tokens, block_size),
            block_size,
            1,
            num_kv_heads,
            head_dim,
            vllm_cache_dtype,
            input_dtype.torch(),
            seed=42,
            device="cuda",
            cache_layout=cache_layout,
        )
        key_cache, value_cache = key_caches[0], value_caches[0]
        k_scale, v_scale = _scales(
            torch, num_kv_heads=num_kv_heads, scale_granularity=scale_granularity
        )

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
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            num_tokens=num_tokens,
            input_dtype=input_dtype,
            kv_dtype=kv_dtype,
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc


def check_packed_append(
    torch: Any,
    *,
    key: Any,
    value: Any,
    kv_cache: Any,
    slot_mapping: Any,
    block_size: int,
) -> None:
    """Torch oracle for the packed write: each token's slot holds ``K | V``.

    Raises when any appended row differs from its source, so a stride or
    offset mismatch fails before it is timed.
    """
    head_dim = key.shape[-1]
    block_indices = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_offsets = slot_mapping % block_size
    written = kv_cache[block_indices, block_offsets]
    if not torch.equal(written[..., :head_dim], key) or not torch.equal(
        written[..., head_dim:], value
    ):
        raise KernelLaunchFailed("packed K|V append wrote rows that differ from its inputs")


def profile_kv_cache_append_vllm_triton_diffkv(
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    input_dtype: DType | str,
    kv_dtype: DType | str,
    cache_layout: str,
    scale_granularity: str,
    num_tokens: int,
    head_dim_v: int | None = None,
) -> ComputeMetrics:
    input_dtype = DType.from_value(input_dtype)
    kv_dtype = DType.from_value(kv_dtype)
    num_kv_heads, head_dim = int(num_kv_heads), int(head_dim)
    # Absent head_dim_v is symmetric K/V (the arg's `same_as`).
    head_dim_v = head_dim if head_dim_v is None else int(head_dim_v)
    block_size, num_tokens = int(block_size), int(num_tokens)
    cache_layout, scale_granularity = _validate_shape(
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        block_size=block_size,
        cache_layout=cache_layout,
        scale_granularity=scale_granularity,
        num_tokens=num_tokens,
    )
    if kv_dtype is not input_dtype:
        raise ValueError("vllm_triton_diffkv is registered for a cache of the input dtype only")
    if scale_granularity != "tensor":
        # The kernel loads one scalar k_scale / v_scale.
        raise ValueError("vllm_triton_diffkv supports tensor scale granularity only")
    try:
        import torch
        from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
            triton_reshape_and_cache_flash_diffkv,
        )
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "the instrumented vLLM environment is required for vllm_triton_diffkv"
        ) from exc

    try:
        key, value = _new_rows(
            torch,
            num_tokens=num_tokens,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            input_dtype=input_dtype,
        )
        slot_mapping = _slot_mapping(torch, num_tokens=num_tokens, block_size=block_size)
        kv_cache = _paged_cache(
            torch,
            num_blocks=_num_blocks(num_tokens, block_size),
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            width=head_dim + head_dim_v,
            dtype=input_dtype.torch(),
            cache_layout=cache_layout,
        )
        k_scale, v_scale = _scales(torch, num_kv_heads=num_kv_heads, scale_granularity="tensor")

        def kernel():
            # FlashAttentionDiffKVImpl.do_kv_cache_update passes its
            # [B, H, page, K+V] view transposed to this [B, page, H, K+V] one.
            return triton_reshape_and_cache_flash_diffkv(
                key, value, kv_cache, slot_mapping, "auto", k_scale, v_scale
            )

        kernel()
        torch.cuda.synchronize()
        check_packed_append(
            torch,
            key=key,
            value=value,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
            block_size=block_size,
        )
        return _metrics(
            kernel,
            kernel_name=_VLLM_DIFFKV_KERNEL_NAME,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            num_tokens=num_tokens,
            input_dtype=input_dtype,
            kv_dtype=kv_dtype,
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
