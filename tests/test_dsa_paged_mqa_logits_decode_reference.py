"""CPU contract tests for the DSA paged decode MQA-logits reference."""

from __future__ import annotations

import math

import pytest
import torch

from profiling.runners.attention.dsa_paged_mqa_logits_decode_reference import (
    dsa_paged_mqa_logits_decode_reference,
)

_DTYPES = [
    torch.float8_e4m3fn,
    torch.bfloat16,
    torch.float16,
    torch.float32,
]


def _values(shape: tuple[int, ...], dtype: torch.dtype, phase: float = 0.0) -> torch.Tensor:
    count = math.prod(shape)
    linear = torch.arange(count, dtype=torch.float32)
    return (torch.sin(linear * 0.157 + phase) * 1.5).reshape(shape).to(dtype)


def _make_cache(
    num_pages: int,
    block_size: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cache = torch.zeros((num_pages, block_size, 1, head_dim + 4), dtype=torch.uint8)
    decoded_keys = torch.empty((num_pages, block_size, head_dim), dtype=torch.float32)
    decoded_scales = torch.empty((num_pages, block_size), dtype=torch.float32)
    for page_id in range(num_pages):
        page = cache[page_id].reshape(-1)
        for offset in range(block_size):
            raw_values = _values((head_dim,), torch.float32, page_id * 0.37 + offset * 0.11)
            raw_values *= 1 + ((page_id + offset) % 3)
            fp8 = raw_values.to(torch.float8_e4m3fn)
            scale = torch.tensor(0.125 * (1 + ((page_id * 3 + offset) % 7)), dtype=torch.float32)
            key_start = offset * head_dim
            scale_start = block_size * head_dim + offset * 4
            page[key_start : key_start + head_dim].copy_(fp8.view(torch.uint8))
            page[scale_start : scale_start + 4].copy_(scale.reshape(1).view(torch.uint8))
            decoded_keys[page_id, offset] = fp8.float()
            decoded_scales[page_id, offset] = scale
    return cache, decoded_keys, decoded_scales


def _oracle(
    q: torch.Tensor,
    decoded_keys: torch.Tensor,
    decoded_scales: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    block_size: int,
    max_model_len: int,
    clean_logits: bool,
) -> torch.Tensor:
    b, next_n, heads, _ = q.shape
    invalid = float("-inf") if clean_logits else float("nan")
    expected = torch.full((b * next_n, max_model_len), invalid)
    for batch in range(b):
        for prediction in range(next_n):
            row = batch * next_n + prediction
            for position in range(int(context_lens[batch, prediction])):
                logical_page, offset = divmod(position, block_size)
                page = int(block_table[batch, logical_page])
                reduced = 0.0
                for head in range(heads):
                    dot = float(
                        (q[batch, prediction, head].float() * decoded_keys[page, offset]).sum(
                            dtype=torch.float32
                        )
                    )
                    reduced += float(weights[row, head]) * max(dot, 0.0)
                expected[row, position] = reduced * float(decoded_scales[page, offset])
    return expected


def _inputs(
    dtype: torch.dtype = torch.float32,
    *,
    batch: int = 2,
    next_n: int = 2,
    heads: int = 4,
    dim: int = 8,
    block_size: int = 4,
    max_model_len: int = 8,
) -> tuple[torch.Tensor, ...]:
    q = _values((batch, next_n, heads, dim), dtype, 0.2)
    cache, _, _ = _make_cache(4, block_size, dim)
    weights = _values((batch * next_n, heads), torch.float32, 0.9)
    contexts = torch.tensor([[0, 5], [3, 8]], dtype=torch.int32)
    block_table = torch.tensor([[2, 0], [2, 3]], dtype=torch.int32)
    return q, cache, weights, contexts, block_table


def _assert_unchanged(
    tensors: tuple[object, ...],
    snapshots: tuple[torch.Tensor | None, ...],
) -> None:
    for tensor, snapshot in zip(tensors, snapshots, strict=True):
        if isinstance(tensor, torch.Tensor) and snapshot is not None:
            torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0, equal_nan=True)


def _assert_rejected(
    q: object,
    cache: object,
    weights: object,
    contexts: object,
    block_table: object,
    *,
    error: type[Exception],
    match: str,
    block_size: object = 4,
    max_model_len: object = 8,
    clean_logits: object = False,
) -> None:
    tensors = (q, cache, weights, contexts, block_table)
    snapshots = tuple(
        tensor.clone()
        if isinstance(tensor, torch.Tensor) and tensor.device.type != "meta"
        else None
        for tensor in tensors
    )
    with pytest.raises(error, match=match):
        dsa_paged_mqa_logits_decode_reference(
            q,  # type: ignore[arg-type]
            cache,  # type: ignore[arg-type]
            weights,  # type: ignore[arg-type]
            contexts,  # type: ignore[arg-type]
            block_table,  # type: ignore[arg-type]
            block_size=block_size,  # type: ignore[arg-type]
            max_model_len=max_model_len,  # type: ignore[arg-type]
            clean_logits=clean_logits,  # type: ignore[arg-type]
        )
    _assert_unchanged(tensors, snapshots)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("clean_logits", [False, True])
def test_correctness_dtypes_mixed_contexts_and_immutability(dtype, clean_logits):
    q, cache, weights, contexts, block_table = _inputs(dtype)
    _, keys, scales = _make_cache(4, 4, 8)
    inputs = (q, cache, weights, contexts, block_table)
    snapshots = tuple(tensor.clone() for tensor in inputs)
    expected = _oracle(
        q,
        keys,
        scales,
        weights,
        contexts,
        block_table,
        block_size=4,
        max_model_len=8,
        clean_logits=clean_logits,
    )

    actual = dsa_paged_mqa_logits_decode_reference(
        *inputs,
        block_size=4,
        max_model_len=8,
        clean_logits=clean_logits,
    )

    assert actual.shape == (4, 8)
    assert actual.dtype is torch.float32
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=2e-4, equal_nan=True)
    _assert_unchanged(inputs, snapshots)


@pytest.mark.parametrize("next_n", [1, 2])
def test_exact_glm_page_planar_layout_offsets_and_next_n(next_n):
    batch, heads, dim, block_size = 1, 64, 128, 64
    max_len = 70
    q = _values((batch, next_n, heads, dim), torch.float8_e4m3fn, 0.4)
    cache, keys, scales = _make_cache(3, block_size, dim)
    weights = torch.linspace(-1.0, 1.0, next_n * heads).reshape(next_n, heads)
    contexts = torch.tensor([[65] if next_n == 1 else [65, 3]], dtype=torch.int32)
    block_table = torch.tensor([[2, 0]], dtype=torch.int32)
    expected = _oracle(
        q,
        keys,
        scales,
        weights,
        contexts,
        block_table,
        block_size=block_size,
        max_model_len=max_len,
        clean_logits=False,
    )

    actual = dsa_paged_mqa_logits_decode_reference(
        q,
        cache,
        weights,
        contexts,
        block_table,
        block_size=block_size,
        max_model_len=max_len,
    )

    assert cache.shape == (3, 64, 1, 132)
    assert cache.stride() == (8448, 132, 132, 1)
    page = cache[2].reshape(-1)
    assert torch.equal(page[:128], keys[2, 0].to(torch.float8_e4m3fn).view(torch.uint8))
    assert torch.equal(page[8192:8196], scales[2, 0].reshape(1).view(torch.uint8))
    assert 8192 == block_size * dim
    assert 8448 == block_size * (dim + 4)
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=2e-4, equal_nan=True)


def test_scattered_pages_prefix_sharing_and_zero_length_padding():
    q, cache, weights, contexts, block_table = _inputs()
    _, keys, scales = _make_cache(4, 4, 8)
    assert block_table[0, 0] == block_table[1, 0]
    expected = _oracle(
        q,
        keys,
        scales,
        weights,
        contexts,
        block_table,
        block_size=4,
        max_model_len=8,
        clean_logits=True,
    )

    actual = dsa_paged_mqa_logits_decode_reference(
        q,
        cache,
        weights,
        contexts,
        block_table,
        block_size=4,
        max_model_len=8,
        clean_logits=True,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert torch.isneginf(actual[0]).all()
    assert not torch.isneginf(actual[1, :5]).any()
    assert torch.isneginf(actual[1, 5:]).all()


def test_nan_tail_explicitly_marks_undefined_production_positions():
    inputs = _inputs(next_n=1)
    q, cache, weights, contexts, block_table = inputs
    contexts = torch.tensor([[3], [5]], dtype=torch.int32)

    actual = dsa_paged_mqa_logits_decode_reference(
        q,
        cache,
        weights,
        contexts,
        block_table,
        block_size=4,
        max_model_len=8,
    )

    assert torch.isnan(actual[0, 3:]).all()
    assert torch.isnan(actual[1, 5:]).all()


def test_strided_q_and_weights_are_accepted():
    q, cache, weights, contexts, block_table = _inputs()
    _, keys, scales = _make_cache(4, 4, 8)
    q_base = torch.empty((3, *q.shape[:-1], q.shape[-1] * 2))
    q_base[1, ..., ::2] = q
    q_view = q_base[1, ..., ::2]
    weights_base = torch.empty((weights.shape[0] + 1, weights.shape[1] * 2))
    weights_base[1:, ::2] = weights
    weight_view = weights_base[1:, ::2]
    expected = _oracle(
        q_view,
        keys,
        scales,
        weight_view,
        contexts,
        block_table,
        block_size=4,
        max_model_len=8,
        clean_logits=False,
    )

    actual = dsa_paged_mqa_logits_decode_reference(
        q_view,
        cache,
        weight_view,
        contexts,
        block_table,
        block_size=4,
        max_model_len=8,
    )

    assert q_view.storage_offset() > 0 and not q_view.is_contiguous()
    assert weight_view.storage_offset() > 0 and not weight_view.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("position", range(5))
def test_rejects_non_tensor_inputs_atomically(position):
    values = list(_inputs())
    values[position] = object()
    _assert_rejected(*values, error=TypeError, match="must be a torch.Tensor")


@pytest.mark.parametrize(
    ("position", "replacement"),
    [
        (0, torch.ones((2, 4, 8))),
        (1, torch.ones((2, 4, 12), dtype=torch.uint8)),
        (2, torch.ones((4,))),
        (3, torch.ones((4,), dtype=torch.int32)),
        (4, torch.ones((2, 2, 1), dtype=torch.int32)),
    ],
)
def test_rejects_wrong_ranks_atomically(position, replacement):
    values = list(_inputs())
    values[position] = replacement
    _assert_rejected(*values, error=ValueError, match="must be rank")


@pytest.mark.parametrize(
    ("position", "replacement"),
    [
        (0, torch.empty((0, 2, 4, 8))),
        (1, torch.empty((0, 4, 1, 12), dtype=torch.uint8)),
        (2, torch.empty((0, 4))),
        (3, torch.empty((0, 2), dtype=torch.int32)),
        (4, torch.empty((0, 2), dtype=torch.int32)),
    ],
)
def test_rejects_empty_dimensions_atomically(position, replacement):
    values = list(_inputs())
    values[position] = replacement
    _assert_rejected(*values, error=ValueError, match="dimensions must be positive")


@pytest.mark.parametrize(
    ("position", "replacement", "match"),
    [
        (1, torch.zeros((4, 5, 1, 12), dtype=torch.uint8), "block dimension"),
        (1, torch.zeros((4, 4, 2, 12), dtype=torch.uint8), "cache must have shape"),
        (1, torch.zeros((4, 4, 1, 13), dtype=torch.uint8), "cache must have shape"),
        (2, torch.ones((3, 4)), "weights must have shape"),
        (3, torch.ones((2, 1), dtype=torch.int32), "context_lens must have shape"),
        (4, torch.ones((3, 2), dtype=torch.int32), "first dimension"),
    ],
)
def test_rejects_shape_mismatches_atomically(position, replacement, match):
    values = list(_inputs())
    values[position] = replacement
    _assert_rejected(*values, error=ValueError, match=match)


@pytest.mark.parametrize("dtype", [torch.float64, torch.int32])
def test_rejects_unsupported_q_dtype(dtype):
    values = list(_inputs())
    values[0] = values[0].to(dtype)
    _assert_rejected(*values, error=TypeError, match="q dtype")


@pytest.mark.parametrize(
    ("position", "replacement", "match"),
    [
        (1, torch.zeros((4, 4, 1, 12), dtype=torch.int8), "cache dtype"),
        (2, torch.ones((4, 4), dtype=torch.float16), "weights dtype"),
        (3, torch.ones((2, 2), dtype=torch.float32), "context_lens dtype"),
        (4, torch.ones((2, 2), dtype=torch.uint8), "block_table dtype"),
    ],
)
def test_rejects_auxiliary_dtype_errors(position, replacement, match):
    values = list(_inputs())
    values[position] = replacement
    _assert_rejected(*values, error=TypeError, match=match)


@pytest.mark.parametrize("position", [1, 3])
def test_rejects_required_noncontiguous_layouts(position):
    values = list(_inputs())
    if position == 1:
        values[position] = torch.zeros((4, 4, 1, 24), dtype=torch.uint8)[..., ::2]
    else:
        values[position] = torch.tensor([[0, 9, 5, 9], [3, 9, 8, 9]], dtype=torch.int32)[:, ::2]
    _assert_rejected(*values, error=ValueError, match="must be contiguous")


def test_rejects_block_table_without_contiguous_inner_dimension():
    values = list(_inputs())
    values[4] = torch.tensor([[2, 9, 0, 9], [2, 9, 3, 9]], dtype=torch.int32)[:, ::2]
    _assert_rejected(*values, error=ValueError, match="innermost dimension")


def test_rejects_misaligned_cache_storage_offset():
    values = list(_inputs())
    shape = values[1].shape
    backing = torch.empty(math.prod(shape) + 1, dtype=torch.uint8)
    values[1] = backing[1:].view(shape)
    _assert_rejected(*values, error=ValueError, match="aligned")


def test_rejects_device_mismatch_without_cuda():
    values = list(_inputs())
    values[2] = torch.empty(values[2].shape, dtype=torch.float32, device="meta")
    _assert_rejected(*values, error=ValueError, match="same device")


def test_rejects_meta_execution_even_when_devices_match():
    values = [tensor.to("meta") for tensor in _inputs()]
    _assert_rejected(*values, error=ValueError, match="meta tensors")


@pytest.mark.parametrize("position", [0, 2])
def test_rejects_nonfinite_math_inputs(position):
    values = list(_inputs())
    values[position] = values[position].clone()
    values[position].reshape(-1)[0] = float("nan")
    _assert_rejected(*values, error=ValueError, match="finite")


@pytest.mark.parametrize(
    ("contexts", "match"),
    [
        ([[-1, 1], [1, 1]], "nonnegative"),
        ([[9, 1], [1, 1]], "must not exceed"),
    ],
)
def test_rejects_invalid_context_bounds(contexts, match):
    values = list(_inputs())
    values[3] = torch.tensor(contexts, dtype=torch.int32)
    _assert_rejected(*values, error=ValueError, match=match)


def test_rejects_insufficient_block_table_width():
    values = list(_inputs())
    values[4] = torch.tensor([[2], [2]], dtype=torch.int32)
    _assert_rejected(*values, error=ValueError, match="needs at least")


@pytest.mark.parametrize("page_id", [-1, 4])
def test_rejects_referenced_page_ids_out_of_bounds(page_id):
    values = list(_inputs())
    values[4] = values[4].clone()
    values[4][0, 0] = page_id
    _assert_rejected(*values, error=ValueError, match="page IDs")


@pytest.mark.parametrize("bad_scale", [0.0, -1.0, float("nan"), float("inf")])
def test_rejects_invalid_referenced_scales(bad_scale):
    values = list(_inputs())
    cache = values[1].clone()
    physical_page = int(values[4][0, 0])
    scale_start = 4 * 8
    cache[physical_page].reshape(-1)[scale_start : scale_start + 4].copy_(
        torch.tensor([bad_scale], dtype=torch.float32).view(torch.uint8)
    )
    values[1] = cache
    _assert_rejected(*values, error=ValueError, match="scales")


@pytest.mark.parametrize(
    ("name", "value", "error", "match"),
    [
        ("block_size", 4.0, TypeError, "block_size"),
        ("block_size", True, TypeError, "block_size"),
        ("block_size", 0, ValueError, "positive"),
        ("max_model_len", 8.0, TypeError, "max_model_len"),
        ("max_model_len", True, TypeError, "max_model_len"),
        ("max_model_len", 0, ValueError, "positive"),
        ("clean_logits", 1, TypeError, "clean_logits"),
    ],
)
def test_rejects_invalid_keyword_arguments(name, value, error, match):
    kwargs = {"block_size": 4, "max_model_len": 8, "clean_logits": False}
    kwargs[name] = value
    _assert_rejected(*_inputs(), error=error, match=match, **kwargs)
