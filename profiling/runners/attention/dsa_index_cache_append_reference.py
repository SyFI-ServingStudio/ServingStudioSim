"""Torch semantics for the DSA index-key quantization and cache append."""

from __future__ import annotations

import torch

_SUPPORTED_INPUT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_SUPPORTED_SCALE_FORMAT = "ue8m0"
_SUPPORTED_CACHE_FORMAT = "page_planar_fp8_fp32_scale"
_FP8_E4M3_MAX = 448.0
_AMAX_FLOOR = 1e-4


def dsa_index_cache_append_reference(
    k: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    quant_block_size: int = 128,
    scale_format: str = _SUPPORTED_SCALE_FORMAT,
    cache_format: str = _SUPPORTED_CACHE_FORMAT,
) -> torch.Tensor:
    """Quantize index keys into a page-planar FP8-key/FP32-scale cache.

    Reduction and scale calculations use FP32. The cache stores all FP8 key
    bytes for a page first, followed by raw FP32 scale bytes. The function
    mutates and returns only ``cache``; padded K rows beyond ``slot_mapping``
    are ignored.

    Source boundary: vLLM v0.23.0 commit
    0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665,
    ``indexer_k_quant_and_cache`` / ``indexer_k_quant_and_cache_kernel``.
    Later profiling backends construct production operands and time either a
    Torch composite or the fused vLLM CUDA launch. FP16/FP32 support here is
    for semantic validation only and does not imply profiler backend support.
    """
    _validate(
        k,
        cache,
        slot_mapping,
        quant_block_size=quant_block_size,
        scale_format=scale_format,
        cache_format=cache_format,
    )

    num_actual = slot_mapping.shape[0]
    index_dim = k.shape[1]
    block_size = cache.shape[1]
    num_groups = index_dim // quant_block_size
    page_bytes = block_size * cache.shape[2]
    scale_plane_offset = block_size * index_dim
    cache_bytes = cache.view(-1)

    for row in range(num_actual):
        slot = int(slot_mapping[row].item())
        if slot < 0:
            continue

        block, offset = divmod(slot, block_size)
        page_offset = block * page_bytes
        for group in range(num_groups):
            group_start = group * quant_block_size
            group_end = group_start + quant_block_size
            values_fp32 = k[row, group_start:group_end].float()
            amax = values_fp32.abs().amax()
            base_scale = torch.clamp_min(amax, _AMAX_FLOOR) / _FP8_E4M3_MAX
            scale = torch.exp2(torch.ceil(torch.log2(base_scale)))

            quantized_bytes = (values_fp32 / scale).to(torch.float8_e4m3fn).view(torch.uint8)
            key_offset = page_offset + offset * index_dim + group_start
            cache_bytes[key_offset : key_offset + quant_block_size].copy_(quantized_bytes)

            scale_offset = page_offset + scale_plane_offset + (offset * num_groups + group) * 4
            scale_bytes = scale.reshape(1).view(torch.uint8)
            cache_bytes[scale_offset : scale_offset + 4].copy_(scale_bytes)

    return cache


def _validate(
    k: object,
    cache: object,
    slot_mapping: object,
    *,
    quant_block_size: object,
    scale_format: object,
    cache_format: object,
) -> None:
    for name, tensor in (
        ("k", k),
        ("cache", cache),
        ("slot_mapping", slot_mapping),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    assert isinstance(k, torch.Tensor)
    assert isinstance(cache, torch.Tensor)
    assert isinstance(slot_mapping, torch.Tensor)

    if not isinstance(quant_block_size, int) or isinstance(quant_block_size, bool):
        raise TypeError("quant_block_size must be an integer")
    if quant_block_size <= 0:
        raise ValueError("quant_block_size must be positive")
    if scale_format != _SUPPORTED_SCALE_FORMAT:
        raise ValueError(f"scale_format must be {_SUPPORTED_SCALE_FORMAT!r}, got {scale_format!r}")
    if cache_format != _SUPPORTED_CACHE_FORMAT:
        raise ValueError(f"cache_format must be {_SUPPORTED_CACHE_FORMAT!r}, got {cache_format!r}")

    _validate_rank_and_shape(k, cache, slot_mapping)
    _validate_dtype_and_device(k, cache, slot_mapping)
    _validate_layout(k, cache, slot_mapping)

    index_dim = k.shape[1]
    if index_dim % quant_block_size != 0:
        raise ValueError(
            f"index_dim {index_dim} must be divisible by quant_block_size {quant_block_size}"
        )
    num_groups = index_dim // quant_block_size
    expected_width = index_dim + 4 * num_groups
    if cache.shape[2] != expected_width:
        raise ValueError(
            "cache final width must equal index_dim + 4*num_groups "
            f"({expected_width}), got {cache.shape[2]}"
        )
    if slot_mapping.shape[0] > k.shape[0]:
        raise ValueError(
            "slot_mapping length must not exceed the K backing-row count, "
            f"got {slot_mapping.shape[0]} and {k.shape[0]}"
        )

    if torch._C._overlaps(cache, k):
        raise ValueError("cache must not alias k")
    if torch._C._overlaps(cache, slot_mapping):
        raise ValueError("cache must not alias slot_mapping")

    participating_k = k[: slot_mapping.shape[0]]
    if not bool(torch.isfinite(participating_k).all().item()):
        raise ValueError("participating k rows must contain only finite values")

    nonnegative_slots = slot_mapping[slot_mapping >= 0]
    if nonnegative_slots.numel() == 0:
        return

    capacity = cache.shape[0] * cache.shape[1]
    if bool(torch.any(nonnegative_slots >= capacity).item()):
        raise ValueError(f"nonnegative slot values must be less than {capacity}")
    if torch.unique(nonnegative_slots).numel() != nonnegative_slots.numel():
        raise ValueError("duplicate nonnegative slots are not allowed")


def _validate_rank_and_shape(
    k: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    for name, tensor, expected_rank in (
        ("k", k, 2),
        ("cache", cache, 3),
        ("slot_mapping", slot_mapping, 1),
    ):
        if tensor.ndim != expected_rank:
            raise ValueError(f"{name} must be rank {expected_rank}, got rank {tensor.ndim}")
        if any(dimension <= 0 for dimension in tensor.shape):
            raise ValueError(f"{name} dimensions must be positive, got {tuple(tensor.shape)}")


def _validate_dtype_and_device(
    k: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    if k.dtype not in _SUPPORTED_INPUT_DTYPES:
        raise TypeError(
            f"k dtype must be torch.bfloat16, torch.float16, or torch.float32, got {k.dtype}"
        )
    if cache.dtype is not torch.uint8:
        raise TypeError(f"cache dtype must be torch.uint8, got {cache.dtype}")
    if slot_mapping.dtype is not torch.int64:
        raise TypeError(f"slot_mapping dtype must be torch.int64, got {slot_mapping.dtype}")

    for name, tensor in (("cache", cache), ("slot_mapping", slot_mapping)):
        if tensor.device != k.device:
            raise ValueError(
                f"{name} must be on the same device as k, got {tensor.device} and {k.device}"
            )
    if k.device.type == "meta":
        raise ValueError("meta tensors are not supported")


def _validate_layout(
    k: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    for name, tensor in (
        ("k", k),
        ("cache", cache),
        ("slot_mapping", slot_mapping),
    ):
        if tensor.layout is not torch.strided:
            raise ValueError(f"{name} must have torch.strided layout")
        if int(torch._debug_has_internal_overlap(tensor)) == 1:
            raise ValueError(f"{name} must not have internal overlap")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
