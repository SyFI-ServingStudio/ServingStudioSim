"""CPU contract tests for the DSA prefill MQA-logits reference."""

from __future__ import annotations

import pytest
import torch

from profiling.runners.attention.dsa_mqa_logits_prefill_reference import (
    dsa_mqa_logits_prefill_reference,
)

_DTYPES = [
    torch.float8_e4m3fn,
    torch.bfloat16,
    torch.float16,
    torch.float32,
]


def _values(shape: tuple[int, ...], dtype: torch.dtype, phase: float = 0.0) -> torch.Tensor:
    linear = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32)
    values = torch.sin(linear * 0.173 + phase) * 1.75
    return values.reshape(shape).to(dtype)


def _oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    *,
    clean_logits: bool,
) -> torch.Tensor:
    invalid = float("-inf") if clean_logits else float("nan")
    expected = torch.full((q.shape[0], k.shape[0]), invalid, dtype=torch.float32)
    for query in range(q.shape[0]):
        for key in range(int(starts[query]), int(ends[query])):
            reduced = 0.0
            for head in range(q.shape[1]):
                dot = float((q[query, head].float() * k[key].float()).sum(dtype=torch.float32))
                reduced += float(weights[query, head]) * max(dot, 0.0)
            expected[query, key] = reduced * float(k_scale[key])
    return expected


def _inputs(
    dtype: torch.dtype = torch.float32,
    *,
    m: int = 3,
    h: int = 4,
    d: int = 8,
    n: int = 7,
) -> tuple[torch.Tensor, ...]:
    q = _values((m, h, d), dtype, 0.1)
    k = _values((n, d), dtype, 0.7)
    k_scale = torch.linspace(0.125, 1.25, n, dtype=torch.float32)
    weights = _values((m, h), torch.float32, 1.2)
    starts = torch.tensor([0, 2, n][:m], dtype=torch.int32)
    ends = torch.tensor([n, n - 1, n][:m], dtype=torch.int32)
    return q, k, k_scale, weights, starts, ends


def _assert_unchanged(
    tensors: tuple[object, ...],
    snapshots: tuple[torch.Tensor | None, ...],
) -> None:
    for tensor, snapshot in zip(tensors, snapshots, strict=True):
        if isinstance(tensor, torch.Tensor) and snapshot is not None:
            torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0, equal_nan=True)


def _assert_rejected(
    q: object,
    k: object,
    scale: object,
    weights: object,
    starts: object,
    ends: object,
    *,
    error: type[Exception],
    match: str,
    clean_logits: object = False,
) -> None:
    tensors = (q, k, scale, weights, starts, ends)
    snapshots = tuple(
        tensor.clone()
        if isinstance(tensor, torch.Tensor) and tensor.device.type != "meta"
        else None
        for tensor in tensors
    )
    with pytest.raises(error, match=match):
        dsa_mqa_logits_prefill_reference(
            q,  # type: ignore[arg-type]
            k,  # type: ignore[arg-type]
            scale,  # type: ignore[arg-type]
            weights,  # type: ignore[arg-type]
            starts,  # type: ignore[arg-type]
            ends,  # type: ignore[arg-type]
            clean_logits=clean_logits,  # type: ignore[arg-type]
        )
    _assert_unchanged(tensors, snapshots)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("clean_logits", [False, True])
def test_correctness_dtypes_masking_shape_and_immutability(dtype, clean_logits):
    inputs = _inputs(dtype)
    snapshots = tuple(tensor.clone() for tensor in inputs)
    expected = _oracle(*inputs, clean_logits=clean_logits)

    actual = dsa_mqa_logits_prefill_reference(*inputs, clean_logits=clean_logits)

    assert actual.shape == expected.shape
    assert actual.dtype is torch.float32
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-5, equal_nan=True)
    _assert_unchanged(inputs, snapshots)


def test_exact_glm_heads_and_dimension_with_causal_tail_spans():
    m, h, d, n = 2, 64, 128, 5
    q = _values((m, h, d), torch.float8_e4m3fn, 0.1)
    k = _values((n, d), torch.float8_e4m3fn, 0.8)
    scales = torch.tensor([0.125, 0.25, 0.5, 1.0, 2.0])
    weights = torch.linspace(-0.75, 1.25, m * h).reshape(m, h)
    starts = torch.tensor([0, 2], dtype=torch.int32)
    ends = torch.tensor([3, 5], dtype=torch.int32)
    expected = _oracle(q, k, scales, weights, starts, ends, clean_logits=False)

    actual = dsa_mqa_logits_prefill_reference(q, k, scales, weights, starts, ends)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-5, equal_nan=True)
    assert torch.isnan(actual[0, 3:]).all()
    assert torch.isnan(actual[1, :2]).all()


def test_ragged_spans_include_empty_rows_and_clean_invalid_positions():
    q, k, scales, weights, _, _ = _inputs(m=3, n=8)
    starts = torch.tensor([1, 4, 6], dtype=torch.int64)
    ends = torch.tensor([6, 4, 8], dtype=torch.int64)
    expected = _oracle(q, k, scales, weights, starts, ends, clean_logits=True)

    actual = dsa_mqa_logits_prefill_reference(
        q, k, scales, weights, starts, ends, clean_logits=True
    )

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert torch.isneginf(actual[1]).all()


def test_nonzero_offset_and_strided_inputs_are_accepted():
    m, h, d, n = 3, 4, 8, 7
    q_base = _values((m + 1, h, d * 2), torch.float32, 0.2)
    k_base = _values((n + 1, d * 2), torch.float32, 0.5)
    scale_base = torch.linspace(0.1, 2.0, 2 * n + 1)
    weight_base = _values((m + 1, h * 2), torch.float32, 1.0)
    q = q_base[1:, :, ::2]
    k = k_base[1:, ::2]
    scales = scale_base[1::2]
    weights = weight_base[1:, ::2]
    starts = torch.tensor([0, 1, 7], dtype=torch.int64)
    ends = torch.tensor([7, 5, 7], dtype=torch.int64)
    expected = _oracle(q, k, scales, weights, starts, ends, clean_logits=False)

    actual = dsa_mqa_logits_prefill_reference(q, k, scales, weights, starts, ends)

    assert q.storage_offset() > 0 and not q.is_contiguous()
    assert k.storage_offset() > 0 and not k.is_contiguous()
    assert scales.storage_offset() > 0 and not scales.is_contiguous()
    assert weights.storage_offset() > 0 and not weights.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("position", range(6))
def test_rejects_non_tensor_inputs_atomically(position):
    values = list(_inputs())
    values[position] = object()
    _assert_rejected(*values, error=TypeError, match="must be a torch.Tensor")


@pytest.mark.parametrize(
    ("position", "replacement"),
    [
        (0, torch.ones((2, 4))),
        (1, torch.ones((2, 3, 4))),
        (2, torch.ones((1, 7))),
        (3, torch.ones((1, 3, 4))),
        (4, torch.ones((1, 3), dtype=torch.int32)),
        (5, torch.ones((1, 3), dtype=torch.int32)),
    ],
)
def test_rejects_wrong_ranks_atomically(position, replacement):
    values = list(_inputs())
    values[position] = replacement
    _assert_rejected(*values, error=ValueError, match="must be rank")


@pytest.mark.parametrize(
    ("position", "replacement"),
    [
        (0, torch.empty((0, 4, 8))),
        (1, torch.empty((0, 8))),
        (2, torch.empty((0,))),
        (3, torch.empty((0, 4))),
        (4, torch.empty((0,), dtype=torch.int32)),
        (5, torch.empty((0,), dtype=torch.int32)),
    ],
)
def test_rejects_empty_dimensions_atomically(position, replacement):
    values = list(_inputs())
    values[position] = replacement
    _assert_rejected(*values, error=ValueError, match="dimensions must be positive")


@pytest.mark.parametrize(
    ("position", "replacement", "match"),
    [
        (1, torch.ones((7, 9)), "head dimensions"),
        (2, torch.ones(6), "k_scale must have shape"),
        (3, torch.ones((3, 5)), "weights must have shape"),
        (4, torch.ones(2, dtype=torch.int32), "k_start must have shape"),
        (5, torch.ones(2, dtype=torch.int32), "k_end must have shape"),
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
    values[1] = values[1].to(dtype)
    _assert_rejected(*values, error=TypeError, match="q dtype")


def test_rejects_mismatched_k_dtype():
    values = list(_inputs(torch.float16))
    values[1] = values[1].float()
    _assert_rejected(*values, error=TypeError, match="k dtype must match")


@pytest.mark.parametrize(
    ("position", "replacement", "match"),
    [
        (2, torch.ones(7, dtype=torch.float16), "k_scale dtype"),
        (3, torch.ones((3, 4), dtype=torch.float16), "weights dtype"),
        (4, torch.ones(3, dtype=torch.float32), "k_start dtype"),
        (5, torch.ones(3, dtype=torch.uint8), "k_end dtype"),
    ],
)
def test_rejects_auxiliary_dtype_errors(position, replacement, match):
    values = list(_inputs())
    values[position] = replacement
    _assert_rejected(*values, error=TypeError, match=match)


@pytest.mark.parametrize("position", [4, 5])
def test_rejects_noncontiguous_span_vectors(position):
    values = list(_inputs())
    values[position] = torch.tensor([0, 9, 2, 9, 7, 9], dtype=torch.int64)[::2]
    _assert_rejected(*values, error=ValueError, match="must be contiguous")


def test_rejects_device_mismatch_without_cuda():
    values = list(_inputs())
    values[1] = torch.empty(values[1].shape, dtype=values[1].dtype, device="meta")
    _assert_rejected(*values, error=ValueError, match="same device")


def test_rejects_meta_execution_even_when_devices_match():
    q, k, scale, weights, starts, ends = _inputs()
    values = [tensor.to("meta") for tensor in (q, k, scale, weights, starts, ends)]
    _assert_rejected(*values, error=ValueError, match="meta tensors")


@pytest.mark.parametrize("position", range(4))
def test_rejects_nonfinite_math_inputs(position):
    values = list(_inputs())
    values[position] = values[position].clone()
    values[position].reshape(-1)[0] = float("nan")
    _assert_rejected(*values, error=ValueError, match="finite")


def test_rejects_nonpositive_scales():
    values = list(_inputs())
    values[2] = values[2].clone()
    values[2][1] = 0
    _assert_rejected(*values, error=ValueError, match="strictly positive")


@pytest.mark.parametrize(
    ("starts", "ends", "match"),
    [
        ([-1, 0, 0], [1, 1, 1], "nonnegative"),
        ([0, 0, 0], [8, 1, 1], "must not exceed"),
        ([2, 0, 0], [1, 1, 1], "less than or equal"),
    ],
)
def test_rejects_invalid_spans(starts, ends, match):
    values = list(_inputs())
    values[4] = torch.tensor(starts, dtype=torch.int32)
    values[5] = torch.tensor(ends, dtype=torch.int32)
    _assert_rejected(*values, error=ValueError, match=match)


def test_rejects_nonboolean_clean_logits():
    _assert_rejected(*_inputs(), error=TypeError, match="clean_logits", clean_logits=1)
