"""CPU contract tests for the plain MLA cache-append reference."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from profiling.runners.attention.mla_cache_append_reference import (
    mla_cache_append_reference,
)


def _values(
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    *,
    start: int = 0,
) -> torch.Tensor:
    values = torch.arange(start, start + torch.tensor(shape).prod().item())
    return (values.to(torch.float32) * 0.125 - 3.0).reshape(shape).to(dtype)


def _manual_oracle(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> torch.Tensor:
    expected = cache.clone()
    block_size = cache.shape[1]
    rank = kv_c.shape[1]
    for row, slot in enumerate(slot_mapping.tolist()):
        if slot < 0:
            continue
        block, offset = divmod(slot, block_size)
        expected[block, offset, :rank].copy_(kv_c[row])
        expected[block, offset, rank:].copy_(k_pe[row])
    return expected


def _valid_inputs(
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    kv_c = _values((4, 3), dtype, start=1)
    k_pe = _values((4, 2), dtype, start=101)
    cache = torch.full((2, 4, 5), -17.0, dtype=dtype)
    slot_mapping = torch.tensor([5, -1, 0, 7], dtype=torch.int64)
    return kv_c, k_pe, cache, slot_mapping


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_contiguous_correctness_identity_and_immutability(dtype):
    kv_c, k_pe, cache, slot_mapping = _valid_inputs(dtype)
    expected = _manual_oracle(kv_c, k_pe, cache, slot_mapping)
    kv_before = kv_c.clone()
    k_pe_before = k_pe.clone()
    mapping_before = slot_mapping.clone()
    cache_storage = cache.untyped_storage().data_ptr()

    actual = mla_cache_append_reference(kv_c, k_pe, cache, slot_mapping)

    assert actual is cache
    assert actual.untyped_storage().data_ptr() == cache_storage
    assert actual.dtype == dtype
    assert torch.equal(actual, expected)
    assert torch.equal(kv_c, kv_before)
    assert torch.equal(k_pe, k_pe_before)
    assert torch.equal(slot_mapping, mapping_before)
    assert torch.all(cache[0, 1:] == expected[0, 1:])
    assert torch.all(cache[1, :1] == -17)


def test_exact_glm_plain_layout_with_scattered_slots():
    num_backing = 5
    kv_c = _values((num_backing, 512), torch.bfloat16, start=1)
    k_pe = _values((num_backing, 64), torch.bfloat16, start=5001)
    cache = torch.full((3, 64, 576), -9.0, dtype=torch.bfloat16)
    slot_mapping = torch.tensor([130, 1, 191, 64, 17], dtype=torch.int64)
    expected = _manual_oracle(kv_c, k_pe, cache, slot_mapping)

    actual = mla_cache_append_reference(kv_c, k_pe, cache, slot_mapping)

    assert actual.shape == (3, 64, 576)
    assert actual.stride() == (64 * 576, 576, 1)
    assert torch.equal(actual, expected)
    for row, slot in enumerate(slot_mapping.tolist()):
        block, offset = divmod(slot, 64)
        assert torch.equal(actual[block, offset, :512], kv_c[row])
        assert torch.equal(actual[block, offset, 512:], k_pe[row])


def test_negative_slots_are_noops_and_repeated_sentinels_are_valid():
    kv_c = _values((6, 3), start=1)
    k_pe = _values((6, 2), start=101)
    cache = torch.full((2, 4, 5), -23.0)
    mapping = torch.tensor([3, -1, -7, -1, 6, -7], dtype=torch.int64)
    expected = _manual_oracle(kv_c, k_pe, cache, mapping)

    actual = mla_cache_append_reference(kv_c, k_pe, cache, mapping)

    assert torch.equal(actual, expected)
    assert torch.equal(actual[0, 3], torch.cat((kv_c[0], k_pe[0])))
    assert torch.equal(actual[1, 2], torch.cat((kv_c[4], k_pe[4])))
    untouched = torch.ones((2, 4), dtype=torch.bool)
    untouched[0, 3] = False
    untouched[1, 2] = False
    assert torch.all(actual[untouched] == -23)


def test_padded_activation_backing_uses_only_mapping_length_prefix():
    kv_c = _values((7, 3), start=1)
    k_pe = _values((7, 2), start=101)
    cache = torch.full((2, 4, 5), -31.0)
    mapping = torch.tensor([5, -1, 2], dtype=torch.int64)
    expected = _manual_oracle(kv_c, k_pe, cache, mapping)

    actual = mla_cache_append_reference(kv_c, k_pe, cache, mapping)

    assert torch.equal(actual, expected)
    assert torch.equal(actual[1, 1], torch.cat((kv_c[0], k_pe[0])))
    assert torch.equal(actual[0, 2], torch.cat((kv_c[2], k_pe[2])))
    for padded_row in range(mapping.numel(), kv_c.shape[0]):
        candidate = torch.cat((kv_c[padded_row], k_pe[padded_row]))
        assert not torch.any(torch.all(actual == candidate, dim=-1))


def test_valid_row_strided_views_are_accepted_without_contiguity():
    kv_base = _values((4, 7), start=1)
    k_pe_base = _values((4, 5), start=101)
    cache_base = torch.full((2, 4, 10), -41.0)
    kv_c = kv_base[:, 1:4]
    k_pe = k_pe_base[:, 2:4]
    cache = cache_base[..., 2:7]
    mapping = torch.tensor([7, 0, -1, 5], dtype=torch.int64)
    expected = _manual_oracle(kv_c, k_pe, cache, mapping)
    storage_ptr = cache.untyped_storage().data_ptr()

    assert kv_c.stride() == (7, 1)
    assert k_pe.stride() == (5, 1)
    assert cache.stride() == (40, 10, 1)
    assert kv_c.storage_offset() == 1
    assert k_pe.storage_offset() == 2
    assert cache.storage_offset() == 2
    assert not kv_c.is_contiguous()
    assert not k_pe.is_contiguous()
    assert not cache.is_contiguous()

    actual = mla_cache_append_reference(kv_c, k_pe, cache, mapping)

    assert actual is cache
    assert actual.untyped_storage().data_ptr() == storage_ptr
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("name", ["kv_c", "k_pe", "cache", "slot_mapping"])
def test_rejects_non_tensor_inputs(name):
    args = list(_valid_inputs())
    index = ("kv_c", "k_pe", "cache", "slot_mapping").index(name)
    args[index] = object()

    with pytest.raises(TypeError, match=rf"{name} must be a torch.Tensor"):
        mla_cache_append_reference(*args)


@pytest.mark.parametrize(
    ("index", "replacement", "message"),
    [
        (0, torch.ones(12), "kv_c must be rank 2"),
        (1, torch.ones((4, 1, 2)), "k_pe must be rank 2"),
        (2, torch.ones((8, 5)), "cache must be rank 3"),
        (3, torch.ones((1, 4), dtype=torch.int64), "slot_mapping must be rank 1"),
    ],
)
def test_rejects_wrong_ranks(index, replacement, message):
    args = list(_valid_inputs())
    args[index] = replacement

    with pytest.raises(ValueError, match=message):
        mla_cache_append_reference(*args)


@pytest.mark.parametrize(
    ("index", "replacement", "message"),
    [
        (0, torch.empty((0, 3)), "kv_c dimensions must be positive"),
        (0, torch.empty((4, 0)), "kv_c dimensions must be positive"),
        (1, torch.empty((4, 0)), "k_pe dimensions must be positive"),
        (2, torch.empty((0, 4, 5)), "cache dimensions must be positive"),
        (2, torch.empty((2, 0, 5)), "cache dimensions must be positive"),
        (
            3,
            torch.empty((0,), dtype=torch.int64),
            "slot_mapping dimensions must be positive",
        ),
    ],
)
def test_rejects_empty_dimensions(index, replacement, message):
    args = list(_valid_inputs())
    args[index] = replacement

    with pytest.raises(ValueError, match=message):
        mla_cache_append_reference(*args)


@pytest.mark.parametrize(
    "dtype",
    [torch.float64, torch.int32, torch.float8_e4m3fn],
)
def test_rejects_unsupported_data_dtypes(dtype):
    kv_c = torch.empty((4, 3), dtype=dtype)
    k_pe = torch.empty((4, 2), dtype=dtype)
    cache = torch.empty((2, 4, 5), dtype=dtype)
    mapping = torch.tensor([0], dtype=torch.int64)

    with pytest.raises(TypeError, match="kv_c dtype must be"):
        mla_cache_append_reference(kv_c, k_pe, cache, mapping)


def test_rejects_k_pe_and_cache_dtype_mismatches():
    kv_c, k_pe, cache, mapping = _valid_inputs()
    with pytest.raises(TypeError, match="k_pe dtype must match"):
        mla_cache_append_reference(kv_c, k_pe.to(torch.float16), cache, mapping)
    with pytest.raises(TypeError, match="cache dtype must match"):
        mla_cache_append_reference(kv_c, k_pe, cache.to(torch.float16), mapping)


@pytest.mark.parametrize("name", ["k_pe", "cache", "slot_mapping"])
def test_rejects_cpu_meta_device_mismatch(name):
    kv_c, k_pe, cache, mapping = _valid_inputs()
    tensors = {"k_pe": k_pe, "cache": cache, "slot_mapping": mapping}
    tensors[name] = tensors[name].to("meta")

    with pytest.raises(ValueError, match=rf"{name} must be on the same device"):
        mla_cache_append_reference(
            kv_c,
            tensors["k_pe"],
            tensors["cache"],
            tensors["slot_mapping"],
        )


def test_rejects_mismatched_backing_rows_and_mapping_too_long():
    kv_c, k_pe, cache, mapping = _valid_inputs()
    with pytest.raises(ValueError, match="same backing-row count"):
        mla_cache_append_reference(kv_c, k_pe[:3], cache, mapping)
    with pytest.raises(ValueError, match="must not exceed"):
        mla_cache_append_reference(
            kv_c,
            k_pe,
            cache,
            torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64),
        )


def test_rejects_mapping_dtype_and_noncontiguous_mapping():
    kv_c, k_pe, cache, mapping = _valid_inputs()
    with pytest.raises(TypeError, match="slot_mapping dtype must be torch.int64"):
        mla_cache_append_reference(kv_c, k_pe, cache, mapping.to(torch.int32))

    noncontiguous = torch.tensor(
        [0, 99, 1, 99, 2, 99, 3, 99], dtype=torch.int64
    )[::2]
    assert not noncontiguous.is_contiguous()
    with pytest.raises(ValueError, match="slot_mapping must be contiguous"):
        mla_cache_append_reference(kv_c, k_pe, cache, noncontiguous)


def test_rejects_wrong_cache_width():
    kv_c, k_pe, _, mapping = _valid_inputs()
    cache = torch.empty((2, 4, 6))

    with pytest.raises(ValueError, match=r"cache final width must equal R \+ P"):
        mla_cache_append_reference(kv_c, k_pe, cache, mapping)


@pytest.mark.parametrize("name", ["kv_c", "k_pe", "cache"])
def test_rejects_nonunit_innermost_stride(name):
    kv_c, k_pe, cache, mapping = _valid_inputs()
    tensors = {"kv_c": kv_c, "k_pe": k_pe, "cache": cache}
    if name == "kv_c":
        tensors[name] = torch.empty((4, 6))[:, ::2]
    elif name == "k_pe":
        tensors[name] = torch.empty((4, 4))[:, ::2]
    else:
        tensors[name] = torch.empty((2, 4, 10))[..., ::2]
    assert tensors[name].stride(-1) == 2

    with pytest.raises(ValueError, match=rf"{name} innermost stride must be 1"):
        mla_cache_append_reference(
            tensors["kv_c"],
            tensors["k_pe"],
            tensors["cache"],
            mapping,
        )


@pytest.mark.parametrize("name", ["kv_c", "k_pe", "cache"])
def test_rejects_definite_internal_overlap(name):
    kv_c, k_pe, cache, mapping = _valid_inputs()
    tensors = {"kv_c": kv_c, "k_pe": k_pe, "cache": cache}
    if name == "kv_c":
        tensors[name] = torch.empty((1, 3)).expand(4, 3)
    elif name == "k_pe":
        tensors[name] = torch.empty((1, 2)).expand(4, 2)
    else:
        tensors[name] = torch.empty((1, 1, 5)).expand(2, 4, 5)
    assert int(torch._debug_has_internal_overlap(tensors[name])) == 1

    with pytest.raises(ValueError, match=rf"{name} must not have internal overlap"):
        mla_cache_append_reference(
            tensors["kv_c"],
            tensors["k_pe"],
            tensors["cache"],
            mapping,
        )


def test_rejects_nonstrided_data_layout():
    kv_c, k_pe, cache, mapping = _valid_inputs()
    sparse_kv_c = kv_c.to_sparse()

    with pytest.raises(ValueError, match="kv_c must have torch.strided layout"):
        mla_cache_append_reference(sparse_kv_c, k_pe, cache, mapping)


def test_rejects_positive_out_of_range_and_duplicate_slots():
    kv_c, k_pe, cache, _ = _valid_inputs()
    with pytest.raises(ValueError, match="must be less than 8"):
        mla_cache_append_reference(
            kv_c,
            k_pe,
            cache,
            torch.tensor([0, 8], dtype=torch.int64),
        )
    with pytest.raises(ValueError, match="duplicate nonnegative slots"):
        mla_cache_append_reference(
            kv_c,
            k_pe,
            cache,
            torch.tensor([3, -1, 3], dtype=torch.int64),
        )


@pytest.mark.parametrize("aliased_input", ["kv_c", "k_pe"])
def test_rejects_cache_aliasing_inputs(aliased_input):
    shared = torch.arange(40, dtype=torch.float32)
    cache = shared[:40].view(2, 4, 5)
    kv_c = shared[:12].view(4, 3) if aliased_input == "kv_c" else torch.ones((4, 3))
    k_pe = shared[:8].view(4, 2) if aliased_input == "k_pe" else torch.ones((4, 2))
    mapping = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

    with pytest.raises(ValueError, match="cache must not alias"):
        mla_cache_append_reference(kv_c, k_pe, cache, mapping)


def _assert_failure_is_atomic(
    mutate: Callable[
        [
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        None,
    ],
    exception: type[Exception],
    message: str,
) -> None:
    kv_c, k_pe, cache, mapping = _valid_inputs()
    mutate(kv_c, k_pe, cache, mapping)
    snapshots = tuple(tensor.clone() for tensor in (kv_c, k_pe, cache, mapping))

    with pytest.raises(exception, match=message):
        mla_cache_append_reference(kv_c, k_pe, cache, mapping)

    for tensor, snapshot in zip((kv_c, k_pe, cache, mapping), snapshots, strict=True):
        assert torch.equal(tensor, snapshot)


@pytest.mark.parametrize(
    ("mutate", "exception", "message"),
    [
        (
            lambda _kv, _kp, _cache, mapping: mapping.fill_(99),
            ValueError,
            "must be less than",
        ),
        (
            lambda _kv, _kp, _cache, mapping: mapping.copy_(
                torch.tensor([3, -1, 3, -1])
            ),
            ValueError,
            "duplicate nonnegative slots",
        ),
        (
            lambda _kv, _kp, cache, _mapping: cache.resize_(2, 4, 6),
            ValueError,
            "cache final width",
        ),
    ],
)
def test_validation_failures_are_atomic(mutate, exception, message):
    _assert_failure_is_atomic(mutate, exception, message)
