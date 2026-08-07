"""Torch semantic reference for the plain MLA paged-cache append."""

from __future__ import annotations

import torch

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def mla_cache_append_reference(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> torch.Tensor:
    """Append latent and RoPE rows to one paged cache and return that cache.

    Validation and slot addressing are completed before the cache is mutated.
    Negative slots are padding sentinels and do not write anything.

    Source boundary: vLLM v0.23.0 commit
    0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665,
    ``concat_and_cache_mla`` / ``concat_and_cache_mla_kernel``. This reference
    covers the plain same-dtype format; mixed ``fp8_ds_mla`` is deferred.
    Profiling backends construct the exact GLM layouts and time either this
    Torch composite or vLLM's one fused CUDA launch.
    """
    _validate_tensors(kv_c, k_pe, cache, slot_mapping)

    valid_rows = torch.nonzero(slot_mapping >= 0, as_tuple=False).flatten()
    if valid_rows.numel() == 0:
        return cache

    slots = slot_mapping[valid_rows]
    block_size = cache.shape[1]
    block_indices = torch.div(slots, block_size, rounding_mode="floor")
    block_offsets = slots % block_size
    kv_lora_rank = kv_c.shape[1]

    cache[block_indices, block_offsets, :kv_lora_rank] = kv_c[valid_rows]
    cache[block_indices, block_offsets, kv_lora_rank:] = k_pe[valid_rows]
    return cache


def _validate_tensors(
    kv_c: object,
    k_pe: object,
    cache: object,
    slot_mapping: object,
) -> None:
    tensors = {
        "kv_c": kv_c,
        "k_pe": k_pe,
        "cache": cache,
        "slot_mapping": slot_mapping,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    assert isinstance(kv_c, torch.Tensor)
    assert isinstance(k_pe, torch.Tensor)
    assert isinstance(cache, torch.Tensor)
    assert isinstance(slot_mapping, torch.Tensor)

    _validate_rank_and_shape(kv_c, k_pe, cache, slot_mapping)
    _validate_dtype_and_device(kv_c, k_pe, cache, slot_mapping)
    _validate_layout(kv_c, k_pe, cache, slot_mapping)

    if k_pe.shape[0] != kv_c.shape[0]:
        raise ValueError(
            "kv_c and k_pe must have the same backing-row count, "
            f"got {kv_c.shape[0]} and {k_pe.shape[0]}"
        )
    if slot_mapping.shape[0] > kv_c.shape[0]:
        raise ValueError(
            "slot_mapping length must not exceed the input backing-row count, "
            f"got {slot_mapping.shape[0]} and {kv_c.shape[0]}"
        )

    expected_width = kv_c.shape[1] + k_pe.shape[1]
    if cache.shape[2] != expected_width:
        raise ValueError(
            f"cache final width must equal R + P ({expected_width}), "
            f"got {cache.shape[2]}"
        )

    if torch._C._overlaps(cache, kv_c) or torch._C._overlaps(cache, k_pe):
        raise ValueError("cache must not alias kv_c or k_pe")

    positive_slots = slot_mapping[slot_mapping >= 0]
    if positive_slots.numel() == 0:
        return

    num_cache_slots = cache.shape[0] * cache.shape[1]
    if bool(torch.any(positive_slots >= num_cache_slots).item()):
        raise ValueError(
            f"nonnegative slot values must be less than {num_cache_slots}"
        )
    if torch.unique(positive_slots).numel() != positive_slots.numel():
        raise ValueError("duplicate nonnegative slots are not allowed")


def _validate_rank_and_shape(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    expected_ranks = {"kv_c": 2, "k_pe": 2, "cache": 3, "slot_mapping": 1}
    for name, tensor in (
        ("kv_c", kv_c),
        ("k_pe", k_pe),
        ("cache", cache),
        ("slot_mapping", slot_mapping),
    ):
        expected_rank = expected_ranks[name]
        if tensor.ndim != expected_rank:
            raise ValueError(
                f"{name} must be rank {expected_rank}, got rank {tensor.ndim}"
            )
        if any(dimension <= 0 for dimension in tensor.shape):
            raise ValueError(
                f"{name} dimensions must be positive, got {tuple(tensor.shape)}"
            )


def _validate_dtype_and_device(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    if kv_c.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            "kv_c dtype must be torch.bfloat16, torch.float16, or torch.float32, "
            f"got {kv_c.dtype}"
        )
    if k_pe.dtype != kv_c.dtype:
        raise TypeError(
            f"k_pe dtype must match kv_c dtype {kv_c.dtype}, got {k_pe.dtype}"
        )
    if cache.dtype != kv_c.dtype:
        raise TypeError(
            f"cache dtype must match kv_c dtype {kv_c.dtype}, got {cache.dtype}"
        )
    if slot_mapping.dtype is not torch.int64:
        raise TypeError(
            f"slot_mapping dtype must be torch.int64, got {slot_mapping.dtype}"
        )

    for name, tensor in (
        ("k_pe", k_pe),
        ("cache", cache),
        ("slot_mapping", slot_mapping),
    ):
        if tensor.device != kv_c.device:
            raise ValueError(
                f"{name} must be on the same device as kv_c, "
                f"got {tensor.device} and {kv_c.device}"
            )


def _validate_layout(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    for name, tensor in (("kv_c", kv_c), ("k_pe", k_pe), ("cache", cache)):
        if tensor.layout is not torch.strided:
            raise ValueError(f"{name} must have torch.strided layout")
        if tensor.stride(-1) != 1:
            raise ValueError(f"{name} innermost stride must be 1")
        # Status 1 is definite overlap. Status 2 means Torch cannot prove the
        # layout either way and must remain accepted for valid row-strided views.
        if int(torch._debug_has_internal_overlap(tensor)) == 1:
            raise ValueError(f"{name} must not have internal overlap")

    if slot_mapping.layout is not torch.strided:
        raise ValueError("slot_mapping must have torch.strided layout")
    if not slot_mapping.is_contiguous():
        raise ValueError("slot_mapping must be contiguous")
