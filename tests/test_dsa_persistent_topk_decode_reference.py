from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

from profiling.runners.attention.dsa_persistent_topk_decode_reference import (
    dsa_persistent_topk_decode_reference,
)


def _manual_oracle(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    expected = torch.full(
        (logits.shape[0], top_k),
        -1,
        dtype=torch.int32,
        device=logits.device,
    )
    for row, raw_length in enumerate(lengths.reshape(-1).tolist()):
        length = int(raw_length)
        if length <= top_k:
            selected = list(range(length))
        else:
            values = logits[row, :length].tolist()
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
    num_rows: int = 2,
    num_keys: int = 16,
    top_k: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    logits = (
        torch.arange(num_rows * num_keys, dtype=torch.float32)
        .reshape(num_rows, num_keys)
        .remainder(19)
        .sub(9)
    )
    lengths = torch.full((num_rows,), num_keys, dtype=torch.int32)
    out = torch.full((num_rows, top_k), -777, dtype=torch.int32)
    return logits, lengths, out, top_k, num_keys


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


def test_glm_top_k_2048_boundary_lengths_identity_and_immutability() -> None:
    top_k = 2048
    num_keys = 2049
    logits = torch.empty((3, num_keys), dtype=torch.float32)
    logits[0] = torch.linspace(-3.0, 2.0, num_keys)
    logits[1] = torch.arange(num_keys, dtype=torch.float32).remainder(31).sub(15)
    logits[2] = -torch.arange(num_keys, dtype=torch.float32)
    lengths = torch.tensor([2047, 2048, 2049], dtype=torch.int32)
    out = torch.full((3, top_k), -99, dtype=torch.int32)
    logits_before = logits.clone()
    lengths_before = lengths.clone()
    storage = out.untyped_storage().data_ptr()

    actual = dsa_persistent_topk_decode_reference(
        logits,
        lengths,
        out,
        top_k=top_k,
        max_seq_len=num_keys,
    )

    assert actual is out
    assert actual.untyped_storage().data_ptr() == storage
    assert torch.equal(actual, _manual_oracle(logits, lengths, top_k))
    assert torch.equal(actual[0, :2047], torch.arange(2047, dtype=torch.int32))
    assert actual[0, -1].item() == -1
    assert torch.equal(actual[1], torch.arange(2048, dtype=torch.int32))
    assert torch.equal(logits, logits_before)
    assert torch.equal(lengths, lengths_before)


def test_rank_one_mixed_lengths_zero_row_and_local_indices() -> None:
    logits = torch.tensor(
        [
            [9, 1, 4, 7, 3, 2],
            [-1, 5, 2, 9, 4, 8],
            [3, 1, 8, 2, 7, 0],
        ],
        dtype=torch.float32,
    )
    lengths = torch.tensor([0, 4, 6], dtype=torch.int32)
    out = torch.full((3, 512), 123, dtype=torch.int32)

    dsa_persistent_topk_decode_reference(
        logits,
        lengths,
        out,
        top_k=512,
        max_seq_len=6,
    )

    assert out[0].tolist() == [-1] * 512
    assert out[1, :5].tolist() == [0, 1, 2, 3, -1]
    assert out[2, :7].tolist() == [0, 1, 2, 3, 4, 5, -1]


def test_rank_two_next_n_two_flattens_in_row_major_order() -> None:
    logits = torch.arange(4 * 9, dtype=torch.float32).reshape(4, 9).neg()
    lengths = torch.tensor([[2, 5], [0, 9]], dtype=torch.int32)
    out = torch.full((4, 512), 77, dtype=torch.int32)
    before = (logits.clone(), lengths.clone())

    actual = dsa_persistent_topk_decode_reference(
        logits,
        lengths,
        out,
        top_k=512,
        max_seq_len=9,
    )

    assert torch.equal(actual, _manual_oracle(logits, lengths, 512))
    assert actual[:, 0].tolist() == [0, 0, -1, 0]
    assert actual[:, 4].tolist() == [-1, 4, -1, 4]
    assert torch.equal(logits, before[0])
    assert torch.equal(lengths, before[1])


def test_fully_padded_max_seq_len_zero_ignores_nonfinite_logits() -> None:
    logits = torch.tensor(
        [[float("nan"), float("inf")], [float("-inf"), float("nan")]],
        dtype=torch.float32,
    )
    lengths = torch.zeros((2, 1), dtype=torch.int32)
    out = torch.full((2, 512), 42, dtype=torch.int32)
    before = logits.clone()

    actual = dsa_persistent_topk_decode_reference(
        logits,
        lengths,
        out,
        top_k=512,
        max_seq_len=0,
    )

    assert torch.all(actual == -1)
    torch.testing.assert_close(logits, before, rtol=0, atol=0, equal_nan=True)


def test_padded_offset_logits_and_offset_contiguous_output() -> None:
    num_rows, num_keys, top_k = 3, 7, 512
    logits_storage = torch.full((50,), -999.0, dtype=torch.float32)
    logits = torch.as_strided(logits_storage, (num_rows, num_keys), (12, 1), 5)
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
    lengths = torch.tensor([7, 5, 3], dtype=torch.int32)
    out_storage = torch.full((num_rows + 2, top_k), -313, dtype=torch.int32)
    out = out_storage[1 : num_rows + 1]

    assert logits.stride() == (12, 1)
    assert logits.storage_offset() == 5
    assert not logits.is_contiguous()
    assert out.is_contiguous() and out.storage_offset() == top_k

    actual = dsa_persistent_topk_decode_reference(
        logits,
        lengths,
        out,
        top_k=top_k,
        max_seq_len=num_keys,
    )

    assert actual is out
    assert torch.equal(actual, _manual_oracle(logits, lengths, top_k))
    assert torch.all(out_storage[0] == -313)
    assert torch.all(out_storage[-1] == -313)


def test_long_row_signed_ties_and_repeats_use_ascending_index_tie_break() -> None:
    top_k = 512
    length = 513
    logits = -torch.arange(length, dtype=torch.float32).unsqueeze(0)
    logits[0, :8] = torch.tensor([4, 4, -1, 4, 2, 4, -7, 3], dtype=torch.float32)
    lengths = torch.tensor([length], dtype=torch.int32)
    out = torch.empty((1, top_k), dtype=torch.int32)

    dsa_persistent_topk_decode_reference(
        logits,
        lengths,
        out,
        top_k=top_k,
        max_seq_len=length,
    )

    assert torch.equal(out, _manual_oracle(logits, lengths, top_k))
    assert out[0, :4].tolist() == [0, 1, 3, 5]
    assert int(out.min()) >= 0
    assert int(out.max()) < length


def test_nonfinite_invalid_tails_are_accepted_preserved_and_unread() -> None:
    logits = torch.tensor(
        [
            [3.0, 2.0, float("nan"), float("inf")],
            [8.0, -1.0, 4.0, float("-inf")],
        ],
        dtype=torch.float32,
    )
    lengths = torch.tensor([2, 3], dtype=torch.int32)
    out = torch.empty((2, 512), dtype=torch.int32)
    before = logits.clone()

    dsa_persistent_topk_decode_reference(
        logits,
        lengths,
        out,
        top_k=512,
        max_seq_len=4,
    )

    assert out[0, :3].tolist() == [0, 1, -1]
    assert out[1, :4].tolist() == [0, 1, 2, -1]
    torch.testing.assert_close(logits, before, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_valid_prefix_is_rejected_atomically(bad_value: float) -> None:
    logits, lengths, out, top_k, max_seq_len = _base_case()
    logits[1, 3] = bad_value
    _assert_atomic_failure(
        ValueError,
        "valid prefix.*finite",
        lambda: dsa_persistent_topk_decode_reference(
            logits,
            lengths,
            out,
            top_k=top_k,
            max_seq_len=max_seq_len,
        ),
        (logits, lengths, out),
    )


@pytest.mark.parametrize("top_k", [512, 1024, 2048])
def test_all_supported_top_k_values(top_k: int) -> None:
    logits = torch.tensor([[2.0, -1.0, 3.0]], dtype=torch.float32)
    lengths = torch.tensor([3], dtype=torch.int32)
    out = torch.empty((1, top_k), dtype=torch.int32)

    dsa_persistent_topk_decode_reference(
        logits,
        lengths,
        out,
        top_k=top_k,
        max_seq_len=3,
    )

    assert out[0, :4].tolist() == [0, 1, 2, -1]


@pytest.mark.parametrize("name", ["logits", "lengths", "out"])
def test_rejects_non_tensor_inputs_atomically(name: str) -> None:
    logits, lengths, out, top_k, max_seq_len = _base_case()
    values: dict[str, Any] = {"logits": logits, "lengths": lengths, "out": out}
    values[name] = object()
    _assert_atomic_failure(
        TypeError,
        f"{name} must be a torch.Tensor",
        lambda: dsa_persistent_topk_decode_reference(
            values["logits"],
            values["lengths"],
            values["out"],
            top_k=top_k,
            max_seq_len=max_seq_len,
        ),
        (logits, lengths, out),
    )


@pytest.mark.parametrize(
    ("name", "replacement", "match"),
    [
        ("logits", torch.ones(16, dtype=torch.float32), "logits must be rank 2"),
        ("logits", torch.ones((1, 2, 8), dtype=torch.float32), "logits must be rank 2"),
        ("lengths", torch.tensor(16, dtype=torch.int32), "lengths must be rank 1 or 2"),
        ("lengths", torch.ones((1, 1, 2), dtype=torch.int32), "lengths must be rank 1 or 2"),
        ("out", torch.ones(1024, dtype=torch.int32), "out must be rank 2"),
        ("out", torch.ones((1, 2, 512), dtype=torch.int32), "out must be rank 2"),
    ],
)
def test_rejects_wrong_ranks_atomically(
    name: str,
    replacement: torch.Tensor,
    match: str,
) -> None:
    logits, lengths, out, top_k, max_seq_len = _base_case()
    values = {"logits": logits, "lengths": lengths, "out": out}
    values[name] = replacement
    _assert_atomic_failure(
        ValueError,
        match,
        lambda: dsa_persistent_topk_decode_reference(
            values["logits"],
            values["lengths"],
            values["out"],
            top_k=top_k,
            max_seq_len=max_seq_len,
        ),
        (logits, lengths, out, replacement),
    )


def test_rejects_empty_logits_dimensions_atomically() -> None:
    for logits in (
        torch.empty((0, 8), dtype=torch.float32),
        torch.empty((2, 0), dtype=torch.float32),
    ):
        lengths = torch.empty((logits.shape[0],), dtype=torch.int32)
        out = torch.empty((logits.shape[0], 512), dtype=torch.int32)
        _assert_atomic_failure(
            ValueError,
            "logits dimensions must be positive",
            lambda: dsa_persistent_topk_decode_reference(
                logits,
                lengths,
                out,
                top_k=512,
                max_seq_len=0,
            ),
            (logits, lengths, out),
        )


def test_rejects_length_count_and_output_shape_mismatches_atomically() -> None:
    logits, lengths, out, top_k, max_seq_len = _base_case()
    bad_lengths = torch.ones(3, dtype=torch.int32)
    _assert_atomic_failure(
        ValueError,
        "lengths must contain exactly 2 values",
        lambda: dsa_persistent_topk_decode_reference(
            logits,
            bad_lengths,
            out,
            top_k=top_k,
            max_seq_len=max_seq_len,
        ),
        (logits, bad_lengths, out),
    )
    bad_out = torch.empty((2, 1024), dtype=torch.int32)
    _assert_atomic_failure(
        ValueError,
        "out shape must be",
        lambda: dsa_persistent_topk_decode_reference(
            logits,
            lengths,
            bad_out,
            top_k=top_k,
            max_seq_len=max_seq_len,
        ),
        (logits, lengths, bad_out),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float64, torch.int32])
def test_rejects_unsupported_logits_dtypes_atomically(dtype: torch.dtype) -> None:
    _, lengths, out, top_k, max_seq_len = _base_case()
    logits = torch.ones((2, 16), dtype=dtype)
    _assert_atomic_failure(
        TypeError,
        "logits dtype must be torch.float32",
        lambda: dsa_persistent_topk_decode_reference(
            logits,
            lengths,
            out,
            top_k=top_k,
            max_seq_len=max_seq_len,
        ),
        (logits, lengths, out),
    )


@pytest.mark.parametrize("name", ["lengths", "out"])
def test_rejects_non_int32_index_tensors_atomically(name: str) -> None:
    logits, lengths, out, top_k, max_seq_len = _base_case()
    values = {"lengths": lengths, "out": out}
    values[name] = values[name].to(torch.int64)
    _assert_atomic_failure(
        TypeError,
        f"{name} dtype must be torch.int32",
        lambda: dsa_persistent_topk_decode_reference(
            logits,
            values["lengths"],
            values["out"],
            top_k=top_k,
            max_seq_len=max_seq_len,
        ),
        (logits, lengths, out, values[name]),
    )


def test_rejects_noncontiguous_lengths_and_output_atomically() -> None:
    logits, _, out, top_k, max_seq_len = _base_case()
    lengths = torch.tensor([16, 99, 16, 99], dtype=torch.int32)[::2]
    assert not lengths.is_contiguous()
    _assert_atomic_failure(
        ValueError,
        "lengths must be contiguous",
        lambda: dsa_persistent_topk_decode_reference(
            logits,
            lengths,
            out,
            top_k=top_k,
            max_seq_len=max_seq_len,
        ),
        (logits, lengths, out),
    )

    lengths = torch.full((2,), 16, dtype=torch.int32)
    bad_out = torch.empty((2, top_k * 2), dtype=torch.int32)[:, ::2]
    assert bad_out.shape == out.shape and not bad_out.is_contiguous()
    _assert_atomic_failure(
        ValueError,
        "out must be contiguous",
        lambda: dsa_persistent_topk_decode_reference(
            logits,
            lengths,
            bad_out,
            top_k=top_k,
            max_seq_len=max_seq_len,
        ),
        (logits, lengths, bad_out),
    )


def test_rejects_wrong_logits_inner_stride_atomically() -> None:
    storage = torch.arange(2 * 16 * 2, dtype=torch.float32)
    logits = torch.as_strided(storage, (2, 16), (32, 2))
    lengths = torch.full((2,), 16, dtype=torch.int32)
    out = torch.empty((2, 512), dtype=torch.int32)
    _assert_atomic_failure(
        ValueError,
        "logits innermost stride must be 1",
        lambda: dsa_persistent_topk_decode_reference(
            logits,
            lengths,
            out,
            top_k=512,
            max_seq_len=16,
        ),
        (logits, lengths, out),
    )


@pytest.mark.parametrize(
    ("top_k", "exc_type", "match"),
    [
        (True, TypeError, "top_k must be a Python int"),
        (512.0, TypeError, "top_k must be a Python int"),
        (0, ValueError, "top_k must be one of"),
        (256, ValueError, "top_k must be one of"),
        (4096, ValueError, "top_k must be one of"),
    ],
)
def test_rejects_invalid_top_k_atomically(
    top_k: object,
    exc_type: type[Exception],
    match: str,
) -> None:
    logits, lengths, out, _, max_seq_len = _base_case()
    _assert_atomic_failure(
        exc_type,
        match,
        lambda: dsa_persistent_topk_decode_reference(
            logits,
            lengths,
            out,
            top_k=top_k,
            max_seq_len=max_seq_len,
        ),
        (logits, lengths, out),
    )


@pytest.mark.parametrize(
    ("max_seq_len", "exc_type", "match"),
    [
        (True, TypeError, "max_seq_len must be a Python int"),
        (16.0, TypeError, "max_seq_len must be a Python int"),
        (-1, ValueError, "max_seq_len must be nonnegative"),
        (17, ValueError, "max_seq_len must not exceed"),
    ],
)
def test_rejects_invalid_max_seq_len_atomically(
    max_seq_len: object,
    exc_type: type[Exception],
    match: str,
) -> None:
    logits, lengths, out, top_k, _ = _base_case()
    _assert_atomic_failure(
        exc_type,
        match,
        lambda: dsa_persistent_topk_decode_reference(
            logits,
            lengths,
            out,
            top_k=top_k,
            max_seq_len=max_seq_len,
        ),
        (logits, lengths, out),
    )


@pytest.mark.parametrize(
    ("lengths", "match"),
    [
        (torch.tensor([-1, 2], dtype=torch.int32), "lengths must be nonnegative"),
        (torch.tensor([16, 17], dtype=torch.int32), "lengths must not exceed max_seq_len"),
    ],
)
def test_rejects_invalid_length_bounds_atomically(
    lengths: torch.Tensor,
    match: str,
) -> None:
    logits, _, out, top_k, max_seq_len = _base_case()
    _assert_atomic_failure(
        ValueError,
        match,
        lambda: dsa_persistent_topk_decode_reference(
            logits,
            lengths,
            out,
            top_k=top_k,
            max_seq_len=max_seq_len,
        ),
        (logits, lengths, out),
    )


def test_rejects_meta_tensors_atomically() -> None:
    logits = torch.empty((2, 16), dtype=torch.float32, device="meta")
    lengths = torch.full((2,), 16, dtype=torch.int32)
    out = torch.empty((2, 512), dtype=torch.int32)
    _assert_atomic_failure(
        ValueError,
        "meta tensors are not supported",
        lambda: dsa_persistent_topk_decode_reference(
            logits,
            lengths,
            out,
            top_k=512,
            max_seq_len=16,
        ),
        (logits, lengths, out),
    )


def test_rejects_non_strided_layout_atomically() -> None:
    indices = torch.tensor([[0, 1], [0, 1]])
    values = torch.tensor([1.0, 2.0])
    logits = torch.sparse_coo_tensor(indices, values, (2, 16))
    lengths = torch.full((2,), 16, dtype=torch.int32)
    out = torch.empty((2, 512), dtype=torch.int32)
    _assert_atomic_failure(
        ValueError,
        "logits must have torch.strided layout",
        lambda: dsa_persistent_topk_decode_reference(
            logits,
            lengths,
            out,
            top_k=512,
            max_seq_len=16,
        ),
        (logits, lengths, out),
    )


def test_rejects_definite_internal_overlap_atomically() -> None:
    logits_storage = torch.arange(16, dtype=torch.float32)
    logits = torch.as_strided(logits_storage, (2, 16), (0, 1))
    lengths = torch.full((2,), 16, dtype=torch.int32)
    out = torch.empty((2, 512), dtype=torch.int32)
    assert int(torch._debug_has_internal_overlap(logits)) == 1
    _assert_atomic_failure(
        ValueError,
        "logits must not have internal overlap",
        lambda: dsa_persistent_topk_decode_reference(
            logits,
            lengths,
            out,
            top_k=512,
            max_seq_len=16,
        ),
        (logits, lengths, out),
    )


def test_rejects_definite_output_internal_overlap_atomically() -> None:
    logits = torch.arange(32, dtype=torch.float32).reshape(2, 16)
    lengths = torch.full((2,), 16, dtype=torch.int32)
    out_storage = torch.full((512,), -9, dtype=torch.int32)
    out = torch.as_strided(out_storage, (2, 512), (0, 1))
    assert int(torch._debug_has_internal_overlap(out)) == 1
    _assert_atomic_failure(
        ValueError,
        "out must not have internal overlap",
        lambda: dsa_persistent_topk_decode_reference(
            logits,
            lengths,
            out,
            top_k=512,
            max_seq_len=16,
        ),
        (logits, lengths, out),
    )


def test_rejects_out_logits_storage_alias_atomically() -> None:
    logits = torch.arange(512, dtype=torch.float32).reshape(1, 512)
    lengths = torch.tensor([512], dtype=torch.int32)
    out = logits.view(torch.int32)
    _assert_atomic_failure(
        ValueError,
        "out must not alias logits",
        lambda: dsa_persistent_topk_decode_reference(
            logits,
            lengths,
            out,
            top_k=512,
            max_seq_len=512,
        ),
        (logits, lengths, out),
    )


def test_rejects_out_lengths_storage_alias_atomically() -> None:
    storage = torch.full((1024,), 7, dtype=torch.int32)
    lengths = storage[:2]
    out = storage.view(2, 512)
    logits = torch.arange(32, dtype=torch.float32).reshape(2, 16)
    _assert_atomic_failure(
        ValueError,
        "out must not alias lengths",
        lambda: dsa_persistent_topk_decode_reference(
            logits,
            lengths,
            out,
            top_k=512,
            max_seq_len=16,
        ),
        (logits, lengths, out),
    )
