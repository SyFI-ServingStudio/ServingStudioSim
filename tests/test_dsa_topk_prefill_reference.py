from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

from profiling.runners.attention.dsa_topk_prefill_reference import (
    dsa_topk_prefill_reference,
)


def _manual_oracle(
    logits: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    expected = torch.full(
        (logits.shape[0], top_k),
        -1,
        dtype=torch.int32,
        device=logits.device,
    )
    for row in range(logits.shape[0]):
        start = int(starts[row])
        end = int(ends[row])
        length = end - start
        if length <= top_k:
            selected = list(range(length))
        else:
            values = logits[row, start:end].tolist()
            selected = sorted(range(length), key=lambda index: (-values[index], index))[:top_k]
        if selected:
            expected[row, : len(selected)] = torch.tensor(
                selected,
                dtype=torch.int32,
                device=logits.device,
            )
    return expected


def _base_case(
    *,
    num_rows: int = 3,
    num_keys: int = 8,
    top_k: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    logits = (
        torch.arange(num_rows * num_keys, dtype=torch.float32)
        .reshape(num_rows, num_keys)
        .remainder(11)
        .sub(5)
    )
    starts = torch.zeros(num_rows, dtype=torch.int32)
    ends = torch.full((num_rows,), num_keys, dtype=torch.int32)
    out = torch.full((num_rows, top_k), -777, dtype=torch.int32)
    return logits, starts, ends, out, top_k


def _snapshot(tensor: torch.Tensor) -> Any:
    if tensor.device.type == "meta":
        return ("meta", tensor.shape, tensor.dtype, tensor.layout)
    if tensor.layout is torch.sparse_coo:
        tensor = tensor.coalesce()
        return ("sparse", tensor.indices().clone(), tensor.values().clone())
    return tensor.clone()


def _assert_snapshot(tensor: torch.Tensor, snapshot: Any) -> None:
    if isinstance(snapshot, tuple) and snapshot[0] == "meta":
        assert ("meta", tensor.shape, tensor.dtype, tensor.layout) == snapshot
    elif isinstance(snapshot, tuple) and snapshot[0] == "sparse":
        tensor = tensor.coalesce()
        assert torch.equal(tensor.indices(), snapshot[1])
        assert torch.equal(tensor.values(), snapshot[2])
    elif tensor.dtype.is_floating_point:
        torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0, equal_nan=True)
    else:
        assert torch.equal(tensor, snapshot)


def _assert_atomic_failure(
    exc_type: type[Exception],
    match: str,
    call: Callable[[], Any],
    tensors: tuple[torch.Tensor, ...],
) -> None:
    snapshots = tuple(_snapshot(tensor) for tensor in tensors)
    with pytest.raises(exc_type, match=match):
        call()
    for tensor, snapshot in zip(tensors, snapshots, strict=True):
        _assert_snapshot(tensor, snapshot)


def test_glm_top_k_2048_contract_at_boundary_lengths() -> None:
    top_k = 2048
    num_keys = 2049
    logits = torch.empty((3, num_keys), dtype=torch.float32)
    logits[0] = torch.linspace(-3.0, 2.0, num_keys)
    logits[1] = torch.arange(num_keys, dtype=torch.float32).remainder(31).sub(15)
    logits[2] = -torch.arange(num_keys, dtype=torch.float32)
    starts = torch.zeros(3, dtype=torch.int32)
    ends = torch.tensor([2047, 2048, 2049], dtype=torch.int32)
    out = torch.full((3, top_k), -99, dtype=torch.int32)
    before = tuple(tensor.clone() for tensor in (logits, starts, ends))
    storage = out.untyped_storage().data_ptr()

    actual = dsa_topk_prefill_reference(
        logits,
        starts,
        ends,
        out,
        top_k=top_k,
    )

    assert actual is out
    assert actual.untyped_storage().data_ptr() == storage
    assert torch.equal(actual, _manual_oracle(logits, starts, ends, top_k))
    assert torch.equal(actual[0, :2047], torch.arange(2047, dtype=torch.int32))
    assert actual[0, -1].item() == -1
    assert torch.equal(actual[1], torch.arange(2048, dtype=torch.int32))
    for tensor, saved in zip((logits, starts, ends), before, strict=True):
        assert torch.equal(tensor, saved)


def test_causal_tail_ragged_empty_and_local_indices() -> None:
    logits = torch.tensor(
        [
            [9, 1, 4, 7, 3, 2, 8, 5, 6, 0],
            [-1, 5, 2, 9, 4, 8, 3, 7, 6, 0],
            [3, 1, 8, 2, 7, 0, 9, 4, 6, 5],
            [0, 2, 4, 6, 8, 1, 3, 5, 7, 9],
        ],
        dtype=torch.float32,
    )
    starts = torch.tensor([0, 2, 4, 7], dtype=torch.int32)
    ends = torch.tensor([3, 8, 10, 7], dtype=torch.int32)
    out = torch.full((4, 3), 123, dtype=torch.int32)

    actual = dsa_topk_prefill_reference(
        logits,
        starts,
        ends,
        out,
        top_k=3,
    )

    assert torch.equal(actual, _manual_oracle(logits, starts, ends, 3))
    assert actual[1].tolist() == [1, 3, 5]
    assert actual[3].tolist() == [-1, -1, -1]
    assert all(index < int(ends[1] - starts[1]) for index in actual[1].tolist())


def test_padded_offset_logits_and_offset_contiguous_output_are_accepted() -> None:
    num_rows, num_keys, top_k = 3, 7, 4
    row_stride = 12
    logits_storage = torch.full((50,), -999.0, dtype=torch.float32)
    logits = torch.as_strided(
        logits_storage,
        (num_rows, num_keys),
        (row_stride, 1),
        5,
    )
    logits.copy_(
        torch.tensor(
            [
                [3, -1, 8, 2, 7, 0, 4],
                [5, 5, 1, -2, 9, 3, 6],
                [-4, 2, 1, 8, 7, 0, 3],
            ],
            dtype=torch.float32,
        )
    )
    starts = torch.tensor([0, 1, 2], dtype=torch.int32)
    ends = torch.tensor([7, 6, 7], dtype=torch.int32)
    out_storage = torch.full((num_rows + 2, top_k), -313, dtype=torch.int32)
    out = out_storage[1 : num_rows + 1]

    assert logits.stride() == (12, 1)
    assert logits.storage_offset() == 5
    assert not logits.is_contiguous()
    assert out.is_contiguous()
    assert out.storage_offset() == top_k

    actual = dsa_topk_prefill_reference(
        logits,
        starts,
        ends,
        out,
        top_k=top_k,
    )

    assert actual is out
    assert torch.equal(actual, _manual_oracle(logits, starts, ends, top_k))
    assert torch.all(out_storage[0] == -313)
    assert torch.all(out_storage[-1] == -313)


def test_value_order_ties_and_repeated_values_use_ascending_local_tie_break() -> None:
    logits = torch.tensor(
        [[-9, -8, 4, 4, -1, 4, 2, 4, -7]],
        dtype=torch.float32,
    )
    starts = torch.tensor([2], dtype=torch.int32)
    ends = torch.tensor([8], dtype=torch.int32)
    out = torch.empty((1, 3), dtype=torch.int32)

    dsa_topk_prefill_reference(logits, starts, ends, out, top_k=3)

    assert out.tolist() == [[0, 1, 3]]


@pytest.mark.parametrize("name", ["logits", "row_starts", "row_ends", "out"])
def test_rejects_non_tensor_inputs_atomically(name: str) -> None:
    logits, starts, ends, out, top_k = _base_case()
    values: dict[str, Any] = {
        "logits": logits,
        "row_starts": starts,
        "row_ends": ends,
        "out": out,
    }
    values[name] = [1, 2, 3]
    _assert_atomic_failure(
        TypeError,
        f"{name} must be a torch.Tensor",
        lambda: dsa_topk_prefill_reference(
            values["logits"],
            values["row_starts"],
            values["row_ends"],
            values["out"],
            top_k=top_k,
        ),
        (logits, starts, ends, out),
    )


@pytest.mark.parametrize(
    ("name", "replacement", "rank"),
    [
        ("logits", torch.zeros(8, dtype=torch.float32), 2),
        ("row_starts", torch.zeros((3, 1), dtype=torch.int32), 1),
        ("row_ends", torch.zeros((3, 1), dtype=torch.int32), 1),
        ("out", torch.zeros(12, dtype=torch.int32), 2),
    ],
)
def test_rejects_wrong_ranks_atomically(
    name: str,
    replacement: torch.Tensor,
    rank: int,
) -> None:
    logits, starts, ends, out, top_k = _base_case()
    values = {"logits": logits, "row_starts": starts, "row_ends": ends, "out": out}
    values[name] = replacement
    tensors = tuple(value for value in values.values() if isinstance(value, torch.Tensor))
    _assert_atomic_failure(
        ValueError,
        f"{name} must be rank {rank}",
        lambda: dsa_topk_prefill_reference(
            values["logits"],
            values["row_starts"],
            values["row_ends"],
            values["out"],
            top_k=top_k,
        ),
        tensors,
    )


@pytest.mark.parametrize("name", ["row_starts", "row_ends", "out_rows", "out_width"])
def test_rejects_shape_mismatches_atomically(name: str) -> None:
    logits, starts, ends, out, top_k = _base_case()
    if name == "row_starts":
        starts = starts[:2]
    elif name == "row_ends":
        ends = ends[:2]
    elif name == "out_rows":
        out = out[:2]
    else:
        out = torch.full((3, top_k + 1), -777, dtype=torch.int32)
    _assert_atomic_failure(
        ValueError,
        "shape must be|out shape must be",
        lambda: dsa_topk_prefill_reference(logits, starts, ends, out, top_k=top_k),
        (logits, starts, ends, out),
    )


@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float16, torch.float64, torch.int32],
)
def test_rejects_unsupported_logits_dtypes_atomically(dtype: torch.dtype) -> None:
    logits, starts, ends, out, top_k = _base_case()
    logits = logits.to(dtype)
    _assert_atomic_failure(
        TypeError,
        "logits dtype must be torch.float32",
        lambda: dsa_topk_prefill_reference(logits, starts, ends, out, top_k=top_k),
        (logits, starts, ends, out),
    )


@pytest.mark.parametrize("name", ["row_starts", "row_ends", "out"])
def test_rejects_non_int32_metadata_or_output_atomically(name: str) -> None:
    logits, starts, ends, out, top_k = _base_case()
    values = {"row_starts": starts, "row_ends": ends, "out": out}
    values[name] = values[name].to(torch.int64)
    _assert_atomic_failure(
        TypeError,
        f"{name} dtype must be torch.int32",
        lambda: dsa_topk_prefill_reference(
            logits,
            values["row_starts"],
            values["row_ends"],
            values["out"],
            top_k=top_k,
        ),
        (logits, values["row_starts"], values["row_ends"], values["out"]),
    )


@pytest.mark.parametrize("name", ["row_starts", "row_ends", "out"])
def test_rejects_noncontiguous_metadata_or_output_atomically(name: str) -> None:
    logits, starts, ends, out, top_k = _base_case()
    if name == "row_starts":
        starts = torch.zeros(6, dtype=torch.int32)[::2]
    elif name == "row_ends":
        ends = torch.full((6,), 8, dtype=torch.int32)[::2]
    else:
        out = torch.full((3, top_k * 2), -777, dtype=torch.int32)[:, ::2]
    _assert_atomic_failure(
        ValueError,
        f"{name} must be contiguous",
        lambda: dsa_topk_prefill_reference(logits, starts, ends, out, top_k=top_k),
        (logits, starts, ends, out),
    )


def test_rejects_wrong_logits_inner_stride_atomically() -> None:
    logits, starts, ends, out, top_k = _base_case()
    logits = torch.empty((3, 16), dtype=torch.float32)[:, ::2]
    _assert_atomic_failure(
        ValueError,
        "logits innermost stride must be 1",
        lambda: dsa_topk_prefill_reference(logits, starts, ends, out, top_k=top_k),
        (logits, starts, ends, out),
    )


@pytest.mark.parametrize("case", ["zero_rows", "zero_keys", "empty_metadata", "empty_out"])
def test_rejects_empty_dimensions_atomically(case: str) -> None:
    logits, starts, ends, out, top_k = _base_case()
    if case == "zero_rows":
        logits = torch.empty((0, 8), dtype=torch.float32)
        starts = torch.empty(0, dtype=torch.int32)
        ends = torch.empty(0, dtype=torch.int32)
        out = torch.empty((0, top_k), dtype=torch.int32)
    elif case == "zero_keys":
        logits = torch.empty((3, 0), dtype=torch.float32)
        ends.zero_()
    elif case == "empty_metadata":
        starts = torch.empty(0, dtype=torch.int32)
    else:
        out = torch.empty((3, 0), dtype=torch.int32)
    _assert_atomic_failure(
        ValueError,
        "dimensions must be positive",
        lambda: dsa_topk_prefill_reference(logits, starts, ends, out, top_k=top_k),
        (logits, starts, ends, out),
    )


@pytest.mark.parametrize(
    ("top_k", "exc_type", "match"),
    [
        (True, TypeError, "top_k must be a Python int"),
        (1.0, TypeError, "top_k must be a Python int"),
        (0, ValueError, "top_k must be positive"),
        (-1, ValueError, "top_k must be positive"),
    ],
)
def test_rejects_invalid_top_k_atomically(
    top_k: Any,
    exc_type: type[Exception],
    match: str,
) -> None:
    logits, starts, ends, out, _ = _base_case()
    _assert_atomic_failure(
        exc_type,
        match,
        lambda: dsa_topk_prefill_reference(logits, starts, ends, out, top_k=top_k),
        (logits, starts, ends, out),
    )


def test_rejects_non_strided_logits_atomically() -> None:
    logits, starts, ends, out, top_k = _base_case()
    logits = logits.to_sparse()
    _assert_atomic_failure(
        ValueError,
        "logits must have torch.strided layout",
        lambda: dsa_topk_prefill_reference(logits, starts, ends, out, top_k=top_k),
        (logits, starts, ends, out),
    )


@pytest.mark.parametrize("name", ["logits", "row_starts", "row_ends", "out"])
def test_rejects_meta_or_device_mismatch_atomically(name: str) -> None:
    logits, starts, ends, out, top_k = _base_case()
    values = {"logits": logits, "row_starts": starts, "row_ends": ends, "out": out}
    values[name] = torch.empty_like(values[name], device="meta")
    _assert_atomic_failure(
        ValueError,
        "meta tensors are not supported",
        lambda: dsa_topk_prefill_reference(
            values["logits"],
            values["row_starts"],
            values["row_ends"],
            values["out"],
            top_k=top_k,
        ),
        (values["logits"], values["row_starts"], values["row_ends"], values["out"]),
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_rejects_nonfinite_logits_atomically(value: float) -> None:
    logits, starts, ends, out, top_k = _base_case()
    logits[1, 2] = value
    _assert_atomic_failure(
        ValueError,
        "logits must contain only finite values",
        lambda: dsa_topk_prefill_reference(logits, starts, ends, out, top_k=top_k),
        (logits, starts, ends, out),
    )


@pytest.mark.parametrize(
    ("starts_values", "ends_values", "match"),
    [
        ([-1, 0, 0], [3, 4, 5], "row_starts must be nonnegative"),
        ([4, 0, 0], [3, 4, 5], "start must be less than or equal"),
        ([0, 0, 0], [3, 9, 5], "row_ends must not exceed"),
    ],
)
def test_rejects_invalid_spans_atomically(
    starts_values: list[int],
    ends_values: list[int],
    match: str,
) -> None:
    logits, _starts, _ends, out, top_k = _base_case()
    starts = torch.tensor(starts_values, dtype=torch.int32)
    ends = torch.tensor(ends_values, dtype=torch.int32)
    _assert_atomic_failure(
        ValueError,
        match,
        lambda: dsa_topk_prefill_reference(logits, starts, ends, out, top_k=top_k),
        (logits, starts, ends, out),
    )


@pytest.mark.parametrize("name", ["logits", "row_starts", "row_ends", "out"])
def test_rejects_definite_internal_overlap_atomically(name: str) -> None:
    logits, starts, ends, out, top_k = _base_case()
    if name == "logits":
        logits = torch.arange(8, dtype=torch.float32).as_strided((3, 8), (0, 1))
    elif name == "row_starts":
        starts = torch.zeros(1, dtype=torch.int32).as_strided((3,), (0,))
    elif name == "row_ends":
        ends = torch.full((1,), 8, dtype=torch.int32).as_strided((3,), (0,))
    else:
        out = torch.full((top_k,), -777, dtype=torch.int32).as_strided(
            (3, top_k),
            (0, 1),
        )
    assert (
        int(
            torch._debug_has_internal_overlap(
                {"logits": logits, "row_starts": starts, "row_ends": ends, "out": out}[name]
            )
        )
        == 1
    )
    _assert_atomic_failure(
        ValueError,
        f"{name} must not have internal overlap",
        lambda: dsa_topk_prefill_reference(logits, starts, ends, out, top_k=top_k),
        (logits, starts, ends, out),
    )


@pytest.mark.parametrize("alias", ["logits", "row_starts", "row_ends"])
def test_rejects_output_storage_aliases_atomically(alias: str) -> None:
    num_rows = 2
    top_k = 3
    logits = torch.arange(6, dtype=torch.float32).reshape(num_rows, top_k)
    starts = torch.zeros(num_rows, dtype=torch.int32)
    ends = torch.full((num_rows,), top_k, dtype=torch.int32)
    out = torch.full((num_rows, top_k), -777, dtype=torch.int32)
    if alias == "logits":
        storage = torch.arange(6, dtype=torch.float32)
        logits = storage.view(num_rows, top_k)
        out = storage.view(torch.int32).view(num_rows, top_k)
    elif alias == "row_starts":
        storage = torch.zeros(num_rows * top_k, dtype=torch.int32)
        out = storage.view(num_rows, top_k)
        starts = storage[:num_rows]
    else:
        storage = torch.full((num_rows * top_k,), top_k, dtype=torch.int32)
        out = storage.view(num_rows, top_k)
        ends = storage[:num_rows]
    assert torch._C._overlaps(
        out, {"logits": logits, "row_starts": starts, "row_ends": ends}[alias]
    )
    _assert_atomic_failure(
        ValueError,
        f"out must not alias {alias}",
        lambda: dsa_topk_prefill_reference(logits, starts, ends, out, top_k=top_k),
        (logits, starts, ends, out),
    )
