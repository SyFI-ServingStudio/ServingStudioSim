"""CPU contract tests for the DSA index-key cache-append reference."""

from __future__ import annotations

import math

import pytest
import torch

from profiling.runners.attention.dsa_index_cache_append_reference import (
    dsa_index_cache_append_reference,
)

_SCALE_FORMAT = "ue8m0"
_CACHE_FORMAT = "page_planar_fp8_fp32_scale"


def _values(rows: int, dim: int, dtype: torch.dtype) -> torch.Tensor:
    linear = torch.arange(rows * dim, dtype=torch.float32).reshape(rows, dim)
    values = torch.sin(linear * 0.173) * 9.0 + torch.cos(linear * 0.037) * 0.7
    return values.to(dtype)


def _oracle(
    k: torch.Tensor,
    cache: torch.Tensor,
    mapping: torch.Tensor,
    quant_block_size: int,
) -> torch.Tensor:
    expected = cache.clone()
    flat = expected.reshape(-1)
    index_dim = k.shape[1]
    block_size = cache.shape[1]
    groups = index_dim // quant_block_size
    page_bytes = block_size * cache.shape[2]

    for row, slot in enumerate(mapping.tolist()):
        if slot < 0:
            continue
        block, offset = divmod(slot, block_size)
        for group in range(groups):
            start = group * quant_block_size
            values = k[row, start : start + quant_block_size].to(torch.float32)
            amax = max(float(values.abs().max()), 1e-4)
            scale_value = math.pow(2.0, math.ceil(math.log2(amax / 448.0)))
            scale = torch.tensor([scale_value], dtype=torch.float32)
            raw_key = (values / scale_value).to(torch.float8_e4m3fn).view(torch.uint8)

            key_offset = block * page_bytes + offset * index_dim + start
            flat[key_offset : key_offset + quant_block_size].copy_(raw_key)
            scale_offset = (
                block * page_bytes + block_size * index_dim + (offset * groups + group) * 4
            )
            flat[scale_offset : scale_offset + 4].copy_(scale.view(torch.uint8))
    return expected


def _valid_inputs(
    dtype: torch.dtype = torch.float32,
    *,
    rows: int = 4,
    dim: int = 8,
    block_size: int = 4,
    quant_block_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    groups = dim // quant_block_size
    k = _values(rows, dim, dtype)
    cache = torch.full(
        (2, block_size, dim + groups * 4),
        0xA5,
        dtype=torch.uint8,
    )
    mapping = torch.tensor([5, -1, 0, 7][:rows], dtype=torch.int64)
    return k, cache, mapping


def _assert_unchanged(
    tensors: tuple[object, ...],
    snapshots: tuple[torch.Tensor | None, ...],
) -> None:
    for tensor, snapshot in zip(tensors, snapshots, strict=True):
        if snapshot is None or not isinstance(tensor, torch.Tensor):
            continue
        torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0, equal_nan=True)


def _assert_rejected(
    k: object,
    cache: object,
    mapping: object,
    *,
    error: type[Exception],
    match: str,
    quant_block_size: object = 4,
    scale_format: object = _SCALE_FORMAT,
    cache_format: object = _CACHE_FORMAT,
) -> None:
    tensors = (k, cache, mapping)
    snapshots = tuple(
        tensor.clone()
        if isinstance(tensor, torch.Tensor) and tensor.device.type != "meta"
        else None
        for tensor in tensors
    )
    with pytest.raises(error, match=match):
        dsa_index_cache_append_reference(
            k,  # type: ignore[arg-type]
            cache,  # type: ignore[arg-type]
            mapping,  # type: ignore[arg-type]
            quant_block_size=quant_block_size,  # type: ignore[arg-type]
            scale_format=scale_format,  # type: ignore[arg-type]
            cache_format=cache_format,  # type: ignore[arg-type]
        )
    _assert_unchanged(tensors, snapshots)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_contiguous_correctness_raw_bytes_identity_and_immutability(dtype):
    k, cache, mapping = _valid_inputs(dtype)
    expected = _oracle(k, cache, mapping, 4)
    k_before = k.clone()
    mapping_before = mapping.clone()
    storage_ptr = cache.untyped_storage().data_ptr()

    actual = dsa_index_cache_append_reference(k, cache, mapping, quant_block_size=4)

    assert actual is cache
    assert actual.untyped_storage().data_ptr() == storage_ptr
    assert torch.equal(actual, expected)
    assert torch.equal(k, k_before)
    assert torch.equal(mapping, mapping_before)


def test_exact_glm_page_planar_layout_and_offsets():
    dim = 128
    block_size = 64
    k = _values(5, dim, torch.bfloat16)
    cache = torch.full((3, block_size, 132), 0xCD, dtype=torch.uint8)
    mapping = torch.tensor([0, 63, 64, 130, 17], dtype=torch.int64)
    expected = _oracle(k, cache, mapping, 128)

    actual = dsa_index_cache_append_reference(k, cache, mapping)

    assert actual.shape == (3, 64, 132)
    assert actual.stride() == (8448, 132, 1)
    assert torch.equal(actual, expected)
    flat = actual.view(-1)
    page_bytes = 8448
    assert torch.equal(flat[0:128], expected.view(-1)[0:128])
    assert torch.equal(flat[8192:8196], expected.view(-1)[8192:8196])
    block, offset = divmod(130, 64)
    assert block * page_bytes + offset * dim == 2 * 8448 + 2 * 128
    assert block * page_bytes + block_size * dim + offset * 4 == 25088 + 8
    assert torch.all(flat[128 : 17 * 128] == 0xCD)


def test_multiple_quant_groups_have_independent_page_planar_scales():
    dim = 256
    quant_block = 128
    block_size = 4
    k = torch.zeros((2, dim), dtype=torch.float32)
    k[0, :128] = torch.linspace(-1.0, 1.0, 128)
    k[0, 128:] = torch.linspace(-31.0, 31.0, 128)
    k[1] = _values(1, dim, torch.float32)[0]
    cache = torch.full((2, block_size, 264), 0x7B, dtype=torch.uint8)
    mapping = torch.tensor([3, 4], dtype=torch.int64)
    expected = _oracle(k, cache, mapping, quant_block)

    actual = dsa_index_cache_append_reference(k, cache, mapping, quant_block_size=quant_block)

    assert torch.equal(actual, expected)
    flat = actual.view(-1)
    page_bytes = block_size * 264
    first_scale = block_size * dim + (3 * 2) * 4
    second_scale = first_scale + 4
    assert not torch.equal(
        flat[first_scale : first_scale + 4],
        flat[second_scale : second_scale + 4],
    )
    block_one_scale = page_bytes + block_size * dim
    assert torch.equal(
        flat[block_one_scale : block_one_scale + 8],
        expected.view(-1)[block_one_scale : block_one_scale + 8],
    )


def test_negative_slots_and_repeated_sentinels_are_noops():
    k = _values(6, 8, torch.float32)
    cache = torch.full((2, 4, 16), 0x91, dtype=torch.uint8)
    mapping = torch.tensor([3, -1, -7, -1, 6, -7], dtype=torch.int64)
    expected = _oracle(k, cache, mapping, 4)

    actual = dsa_index_cache_append_reference(k, cache, mapping, quant_block_size=4)

    assert torch.equal(actual, expected)
    assert torch.count_nonzero(actual != 0x91) > 0


def test_padded_backing_uses_only_mapping_length_prefix():
    k = _values(7, 8, torch.float32)
    k[3:] = float("nan")
    cache = torch.full((2, 4, 16), 0x51, dtype=torch.uint8)
    mapping = torch.tensor([5, -1, 2], dtype=torch.int64)
    expected = _oracle(k[:3], cache, mapping, 4)
    k_before = k.clone()

    actual = dsa_index_cache_append_reference(k, cache, mapping, quant_block_size=4)

    assert torch.equal(actual, expected)
    torch.testing.assert_close(k, k_before, rtol=0, atol=0, equal_nan=True)


def test_all_zero_rows_have_zero_keys_and_exact_two_to_minus_22_scales():
    k = torch.zeros((2, 128), dtype=torch.bfloat16)
    cache = torch.full((1, 64, 132), 0xFF, dtype=torch.uint8)
    mapping = torch.tensor([0, 5], dtype=torch.int64)

    actual = dsa_index_cache_append_reference(k, cache, mapping)
    flat = actual.view(-1)
    expected_scale = torch.tensor([2.0**-22], dtype=torch.float32).view(torch.uint8)

    assert torch.count_nonzero(flat[0:128]) == 0
    assert torch.count_nonzero(flat[5 * 128 : 6 * 128]) == 0
    assert torch.equal(flat[8192:8196], expected_scale)
    assert torch.equal(flat[8192 + 5 * 4 : 8192 + 6 * 4], expected_scale)


def test_nontrivial_sign_rounding_and_scale_boundaries():
    values = torch.tensor(
        [
            -448.0,
            -3.5,
            -1e-4,
            0.0,
            1e-4,
            3.5,
            447.0,
            448.0,
        ],
        dtype=torch.float32,
    ).reshape(1, 8)
    cache = torch.full((1, 2, 12), 0x6D, dtype=torch.uint8)
    mapping = torch.tensor([1], dtype=torch.int64)
    expected = _oracle(values, cache, mapping, 8)

    actual = dsa_index_cache_append_reference(values, cache, mapping, quant_block_size=8)

    assert torch.equal(actual, expected)
    key_offset = 8
    assert actual.view(-1)[key_offset : key_offset + 8].tolist() == [
        254,
        198,
        128,
        0,
        0,
        70,
        126,
        126,
    ]


@pytest.mark.parametrize("name", ["k", "cache", "slot_mapping"])
def test_rejects_non_tensors_atomically(name):
    args = list(_valid_inputs())
    index = ("k", "cache", "slot_mapping").index(name)
    args[index] = object()
    _assert_rejected(*args, error=TypeError, match=rf"{name} must be a torch.Tensor")


@pytest.mark.parametrize(
    ("index", "replacement", "message"),
    [
        (0, torch.ones(8), "k must be rank 2"),
        (1, torch.ones((4, 12), dtype=torch.uint8), "cache must be rank 3"),
        (
            2,
            torch.ones((1, 4), dtype=torch.int64),
            "slot_mapping must be rank 1",
        ),
    ],
)
def test_rejects_wrong_ranks_atomically(index, replacement, message):
    args = list(_valid_inputs())
    args[index] = replacement
    _assert_rejected(*args, error=ValueError, match=message)


@pytest.mark.parametrize(
    ("index", "replacement", "message"),
    [
        (0, torch.empty((0, 8)), "k dimensions must be positive"),
        (0, torch.empty((4, 0)), "k dimensions must be positive"),
        (1, torch.empty((0, 4, 16), dtype=torch.uint8), "cache dimensions"),
        (1, torch.empty((2, 0, 16), dtype=torch.uint8), "cache dimensions"),
        (1, torch.empty((2, 4, 0), dtype=torch.uint8), "cache dimensions"),
        (
            2,
            torch.empty((0,), dtype=torch.int64),
            "slot_mapping dimensions must be positive",
        ),
    ],
)
def test_rejects_empty_dimensions_atomically(index, replacement, message):
    args = list(_valid_inputs())
    args[index] = replacement
    _assert_rejected(*args, error=ValueError, match=message)


@pytest.mark.parametrize("dtype", [torch.float64, torch.int32, torch.float8_e4m3fn])
def test_rejects_unsupported_k_dtype_atomically(dtype):
    k, cache, mapping = _valid_inputs()
    k = torch.empty(k.shape, dtype=dtype)
    _assert_rejected(k, cache, mapping, error=TypeError, match="k dtype must be")


def test_rejects_non_uint8_cache_atomically():
    k, cache, mapping = _valid_inputs()
    _assert_rejected(
        k,
        cache.float(),
        mapping,
        error=TypeError,
        match="cache dtype must be torch.uint8",
    )


def test_rejects_non_int64_mapping_atomically():
    k, cache, mapping = _valid_inputs()
    _assert_rejected(
        k,
        cache,
        mapping.int(),
        error=TypeError,
        match="slot_mapping dtype must be torch.int64",
    )


@pytest.mark.parametrize("tensor_name", ["k", "cache", "slot_mapping"])
def test_rejects_noncontiguous_tensors_atomically(tensor_name):
    k, cache, mapping = _valid_inputs()
    if tensor_name == "k":
        k = torch.ones((4, 16))[:, ::2]
    elif tensor_name == "cache":
        cache = torch.empty((2, 16, 4), dtype=torch.uint8).transpose(1, 2)
    else:
        mapping = torch.tensor([5, 99, -1, 99, 0, 99, 7, 99])[::2]
    _assert_rejected(
        k,
        cache,
        mapping,
        error=ValueError,
        match=rf"{tensor_name} must be contiguous",
    )


@pytest.mark.parametrize("tensor_name", ["k", "cache", "slot_mapping"])
def test_rejects_definite_internal_overlap_atomically(tensor_name):
    k, cache, mapping = _valid_inputs(rows=2)
    if tensor_name == "k":
        k = torch.ones(8).as_strided((2, 8), (0, 1))
    elif tensor_name == "cache":
        cache = torch.empty(1, dtype=torch.uint8).as_strided((2, 4, 16), (0, 0, 0))
    else:
        mapping = torch.zeros(1, dtype=torch.int64).as_strided((2,), (0,))
    assert (
        int(
            torch._debug_has_internal_overlap(
                (k, cache, mapping)[("k", "cache", "slot_mapping").index(tensor_name)]
            )
        )
        == 1
    )
    _assert_rejected(
        k,
        cache,
        mapping,
        error=ValueError,
        match=rf"{tensor_name} must not have internal overlap",
    )


def test_rejects_device_mismatch_with_meta_atomically():
    k, cache, mapping = _valid_inputs()
    meta_cache = torch.empty(cache.shape, dtype=torch.uint8, device="meta")
    _assert_rejected(
        k,
        meta_cache,
        mapping,
        error=ValueError,
        match="cache must be on the same device as k",
    )


def test_rejects_empty_or_too_long_mapping_atomically():
    k, cache, mapping = _valid_inputs()
    _assert_rejected(
        k,
        cache,
        torch.empty(0, dtype=torch.int64),
        error=ValueError,
        match="slot_mapping dimensions must be positive",
    )
    _assert_rejected(
        k,
        cache,
        torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64),
        error=ValueError,
        match="must not exceed the K backing-row count",
    )


@pytest.mark.parametrize(
    ("quant_block_size", "error", "message"),
    [
        (True, TypeError, "must be an integer"),
        (4.0, TypeError, "must be an integer"),
        (0, ValueError, "must be positive"),
        (-4, ValueError, "must be positive"),
        (3, ValueError, "must be divisible"),
    ],
)
def test_rejects_invalid_quant_block_atomically(quant_block_size, error, message):
    _assert_rejected(
        *_valid_inputs(),
        error=error,
        match=message,
        quant_block_size=quant_block_size,
    )


def test_rejects_wrong_cache_width_atomically():
    k, _, mapping = _valid_inputs()
    cache = torch.empty((2, 4, 15), dtype=torch.uint8)
    _assert_rejected(
        k,
        cache,
        mapping,
        error=ValueError,
        match="cache final width must equal",
    )


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("scale_format", "float32", "scale_format must be 'ue8m0'"),
        ("cache_format", "interleaved", "cache_format must be"),
    ],
)
def test_rejects_unsupported_formats_atomically(name, value, message):
    kwargs = {name: value}
    _assert_rejected(
        *_valid_inputs(),
        error=ValueError,
        match=message,
        **kwargs,
    )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_participating_k_atomically(bad_value):
    k, cache, mapping = _valid_inputs()
    k[0, 0] = bad_value
    _assert_rejected(
        k,
        cache,
        mapping,
        error=ValueError,
        match="participating k rows must contain only finite values",
    )


def test_rejects_duplicate_nonnegative_slots_atomically():
    k, cache, _ = _valid_inputs()
    mapping = torch.tensor([1, -1, 1, -1], dtype=torch.int64)
    _assert_rejected(
        k,
        cache,
        mapping,
        error=ValueError,
        match="duplicate nonnegative slots",
    )


@pytest.mark.parametrize("slot", [8, 99])
def test_rejects_out_of_range_positive_slots_atomically(slot):
    k, cache, mapping = _valid_inputs()
    mapping[0] = slot
    _assert_rejected(
        k,
        cache,
        mapping,
        error=ValueError,
        match="nonnegative slot values must be less than 8",
    )


def test_rejects_cache_k_storage_alias_atomically():
    backing = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    k = backing
    cache = backing.view(torch.uint8).reshape(-1)[:8].reshape(1, 1, 8)
    mapping = torch.tensor([0], dtype=torch.int64)
    _assert_rejected(
        k,
        cache,
        mapping,
        error=ValueError,
        match="cache must not alias k",
    )


def test_rejects_cache_mapping_storage_alias_atomically():
    mapping_backing = torch.tensor([0, 9], dtype=torch.int64)
    mapping = mapping_backing[:1]
    cache = mapping_backing.view(torch.uint8)[:8].reshape(1, 1, 8)
    k = torch.ones((1, 4), dtype=torch.float32)
    _assert_rejected(
        k,
        cache,
        mapping,
        error=ValueError,
        match="cache must not alias slot_mapping",
    )
