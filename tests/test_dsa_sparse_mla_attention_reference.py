from __future__ import annotations

import json
import math
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import profiling.runners.attention.dsa_sparse_mla_attention_reference as reference_module
from profiling.runners.attention.dsa_sparse_mla_attention_reference import (
    dsa_sparse_mla_attention_reference,
)

# The subprocess must import `profiling` the way a caller outside pytest would:
# from the checkout root, not from whatever directory the test happens to run
# in. This was a hardcoded `/workspace`, which only exists in the container the
# test was written in.
_REPO_ROOT = Path(__file__).resolve().parents[1]

_SCORE_DIM = 576
_VALUE_DIM = 512
_SCALE = 0.0625


def _manual_oracle(
    q: torch.Tensor,
    cache: torch.Tensor,
    selected_indices: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Small independent Python-loop oracle, intentionally not vectorized."""
    result = torch.zeros(
        (q.shape[0], q.shape[1], _VALUE_DIM),
        dtype=torch.bfloat16,
    )
    for query_index in range(q.shape[0]):
        valid_indices = [
            int(index)
            for index in selected_indices[query_index, 0].tolist()
            if 0 <= int(index) < cache.shape[0]
        ]
        if not valid_indices:
            continue
        for head_index in range(q.shape[1]):
            scores = []
            for cache_index in valid_indices:
                dot = sum(
                    float(q[query_index, head_index, dim]) * float(cache[cache_index, 0, dim])
                    for dim in range(_SCORE_DIM)
                )
                scores.append(dot * softmax_scale)
            score_max = max(scores)
            exponentials = [math.exp(score - score_max) for score in scores]
            denominator = sum(exponentials)
            probabilities = [value / denominator for value in exponentials]
            row = [
                sum(
                    probability * float(cache[cache_index, 0, dim])
                    for probability, cache_index in zip(
                        probabilities,
                        valid_indices,
                        strict=True,
                    )
                )
                for dim in range(_VALUE_DIM)
            ]
            result[query_index, head_index].copy_(
                torch.tensor(row, dtype=torch.float32).to(torch.bfloat16)
            )
    return result


def _base_case(
    *,
    num_queries: int = 2,
    num_heads: int = 2,
    num_cache_tokens: int = 5,
    selected_k: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = (
        torch.arange(num_queries * num_heads * _SCORE_DIM, dtype=torch.float32)
        .reshape(num_queries, num_heads, _SCORE_DIM)
        .remainder(17)
        .sub(8)
        .div(16)
        .to(torch.bfloat16)
    )
    cache = (
        torch.arange(num_cache_tokens * _SCORE_DIM, dtype=torch.float32)
        .reshape(num_cache_tokens, 1, _SCORE_DIM)
        .remainder(19)
        .sub(9)
        .div(16)
        .to(torch.bfloat16)
    )
    rows = []
    for query_index in range(num_queries):
        rows.append([(query_index + offset) % num_cache_tokens for offset in range(selected_k)])
    indices = torch.tensor(rows, dtype=torch.int32).unsqueeze(1)
    return q, cache, indices


def _snapshot(tensor: torch.Tensor) -> Any:
    if tensor.device.type == "meta":
        return ("meta", tuple(tensor.shape), tensor.dtype, tensor.layout)
    if tensor.layout is torch.sparse_coo:
        tensor = tensor.coalesce()
        return ("sparse", tensor.indices().clone(), tensor.values().clone())
    return tensor.clone()


def _assert_snapshot(tensor: torch.Tensor, snapshot: Any) -> None:
    if isinstance(snapshot, tuple) and snapshot[0] == "meta":
        assert ("meta", tuple(tensor.shape), tensor.dtype, tensor.layout) == snapshot
    elif isinstance(snapshot, tuple) and snapshot[0] == "sparse":
        tensor = tensor.coalesce()
        assert torch.equal(tensor.indices(), snapshot[1])
        assert torch.equal(tensor.values(), snapshot[2])
    elif tensor.dtype.is_floating_point:
        torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0, equal_nan=True)
    else:
        assert torch.equal(tensor, snapshot)


def _assert_failure_without_mutation(
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


def test_signed_non_tied_values_match_independent_manual_oracle() -> None:
    q = torch.zeros((2, 2, _SCORE_DIM), dtype=torch.bfloat16)
    cache = torch.zeros((4, 1, _SCORE_DIM), dtype=torch.bfloat16)
    q[:, :, :4] = torch.tensor(
        [[[1, -2, 3, 1], [-1, 3, 2, -2]], [[2, 1, -3, 2], [3, -1, 1, -2]]],
        dtype=torch.bfloat16,
    )
    cache[:, 0, :4] = torch.tensor(
        [[2, -1, 1, 3], [-3, 2, 1, -1], [1, 4, -2, 2], [-2, -3, 2, 1]],
        dtype=torch.bfloat16,
    )
    cache[:, 0, 4:8] = torch.tensor(
        [[1, -2, 3, 4], [4, 3, -2, 1], [-1, 2, 5, -3], [3, -4, 2, 1]],
        dtype=torch.bfloat16,
    )
    indices = torch.tensor([[[3, 1, 0]], [[2, 0, 1]]], dtype=torch.int32)

    actual = dsa_sparse_mla_attention_reference(
        q,
        cache,
        indices,
        softmax_scale=_SCALE,
    )
    expected = _manual_oracle(q, cache, indices, _SCALE)

    torch.testing.assert_close(actual, expected, rtol=0, atol=torch.finfo(torch.bfloat16).eps)


def test_output_contract_input_identity_immutability_and_no_aliasing() -> None:
    q, cache, indices = _base_case()
    snapshots = tuple(tensor.clone() for tensor in (q, cache, indices))
    storage_ids = tuple(tensor.untyped_storage().data_ptr() for tensor in (q, cache, indices))

    output = dsa_sparse_mla_attention_reference(
        q,
        cache,
        indices,
        softmax_scale=_SCALE,
    )

    assert output.shape == (2, 2, _VALUE_DIM)
    assert output.dtype is torch.bfloat16
    assert output.device == q.device
    assert output.is_contiguous()
    assert output.untyped_storage().data_ptr() not in storage_ids
    for tensor, snapshot, storage_id in zip(
        (q, cache, indices), snapshots, storage_ids, strict=True
    ):
        assert tensor.untyped_storage().data_ptr() == storage_id
        assert torch.equal(tensor, snapshot)


def test_empty_one_short_and_full_valid_counts() -> None:
    q, cache, _ = _base_case(num_queries=4, num_heads=1, selected_k=4)
    indices = torch.tensor(
        [
            [[-1, -7, 5, 99]],
            [[2, -1, -1, -1]],
            [[0, 3, -1, -1]],
            [[0, 1, 2, 3]],
        ],
        dtype=torch.int32,
    )

    actual = dsa_sparse_mla_attention_reference(q, cache, indices, softmax_scale=_SCALE)
    expected = _manual_oracle(q, cache, indices, _SCALE)

    assert torch.count_nonzero(actual[0]).item() == 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=torch.finfo(torch.bfloat16).eps)


def test_arbitrary_negative_and_out_of_range_indices_are_masked() -> None:
    q, cache, _ = _base_case(num_queries=2, num_heads=1, num_cache_tokens=3)
    indices = torch.tensor(
        [[[-2_147_483_648, -123456, 0, 3]], [[2, 114514, 2_147_483_647, -1]]],
        dtype=torch.int32,
    )

    actual = dsa_sparse_mla_attention_reference(q, cache, indices, softmax_scale=_SCALE)

    torch.testing.assert_close(
        actual,
        _manual_oracle(q, cache, indices, _SCALE),
        rtol=0,
        atol=torch.finfo(torch.bfloat16).eps,
    )


def test_duplicate_indices_retain_repeated_softmax_mass_and_order() -> None:
    q = torch.zeros((1, 1, _SCORE_DIM), dtype=torch.bfloat16)
    cache = torch.zeros((2, 1, _SCORE_DIM), dtype=torch.bfloat16)
    q[0, 0, 0] = 1
    cache[0, 0, 0] = 0.5
    cache[1, 0, 0] = 2
    once = torch.tensor([[[0, 1, -1]]], dtype=torch.int32)
    duplicate = torch.tensor([[[0, 0, 1]]], dtype=torch.int32)
    reordered = torch.tensor([[[1, 0, 0]]], dtype=torch.int32)

    once_output = dsa_sparse_mla_attention_reference(q, cache, once, softmax_scale=1.0)
    duplicate_output = dsa_sparse_mla_attention_reference(q, cache, duplicate, softmax_scale=1.0)
    reordered_output = dsa_sparse_mla_attention_reference(q, cache, reordered, softmax_scale=1.0)

    assert not torch.equal(once_output, duplicate_output)
    assert torch.equal(duplicate_output, reordered_output)
    assert torch.equal(
        duplicate_output,
        _manual_oracle(q, cache, duplicate, 1.0),
    )


def test_unselected_nonfinite_rows_and_invalid_placeholder_do_not_contaminate() -> None:
    q = torch.ones((2, 2, _SCORE_DIM), dtype=torch.bfloat16)
    cache = torch.ones((3, 1, _SCORE_DIM), dtype=torch.bfloat16)
    cache[1].fill_(float("nan"))
    cache[2, :, :288].fill_(float("inf"))
    cache[2, :, 288:].fill_(float("-inf"))
    indices = torch.tensor([[[0, -1, 3]], [[-7, 99, -1]]], dtype=torch.int32)
    cache_before = cache.clone()

    output = dsa_sparse_mla_attention_reference(q, cache, indices, softmax_scale=_SCALE)

    assert bool(torch.isfinite(output).all())
    assert torch.count_nonzero(output[1]).item() == 0
    torch.testing.assert_close(cache, cache_before, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_selected_nonfinite_cache_rows_are_rejected(bad_value: float) -> None:
    q, cache, indices = _base_case(num_queries=1, num_heads=1)
    selected_row = int(indices[0, 0, 0])
    cache[selected_row, 0, 12] = bad_value
    _assert_failure_without_mutation(
        ValueError,
        "valid selected cache rows must contain only finite values",
        lambda: dsa_sparse_mla_attention_reference(q, cache, indices, softmax_scale=_SCALE),
        (q, cache, indices),
    )


def test_padded_outer_strides_and_nonzero_offsets_are_accepted() -> None:
    q_storage = torch.empty(4000, dtype=torch.bfloat16)
    q = torch.as_strided(q_storage, (2, 2, _SCORE_DIM), (1300, 620, 1), 7)
    cache_storage = torch.empty(4000, dtype=torch.bfloat16)
    cache = torch.as_strided(cache_storage, (4, 1, _SCORE_DIM), (700, 600, 1), 11)
    q.copy_(_base_case(num_queries=2, num_heads=2)[0])
    cache.copy_(_base_case(num_cache_tokens=4)[1])
    indices = torch.tensor([[[0, 3, -1]], [[2, 1, 4]]], dtype=torch.int32)

    assert q.stride() == (1300, 620, 1) and q.storage_offset() == 7
    assert cache.stride() == (700, 600, 1) and cache.storage_offset() == 11
    assert not q.is_contiguous() and not cache.is_contiguous()

    actual = dsa_sparse_mla_attention_reference(q, cache, indices, softmax_scale=_SCALE)
    torch.testing.assert_close(
        actual,
        _manual_oracle(q, cache, indices, _SCALE),
        rtol=0,
        atol=torch.finfo(torch.bfloat16).eps,
    )


def test_query_chunk_size_does_not_change_multi_head_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q, cache, indices = _base_case(
        num_queries=7,
        num_heads=3,
        num_cache_tokens=11,
        selected_k=6,
    )
    indices[1, 0, 2:] = torch.tensor([-1, 11, -7, 3], dtype=torch.int32)

    outputs = []
    for chunk_size in (1, 3, 64):
        monkeypatch.setattr(reference_module, "_QUERY_CHUNK_SIZE", chunk_size)
        outputs.append(dsa_sparse_mla_attention_reference(q, cache, indices, softmax_scale=_SCALE))

    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    torch.testing.assert_close(outputs[0], outputs[2], rtol=0, atol=0)


@pytest.mark.parametrize("name", ["q", "cache", "selected_indices"])
def test_non_tensor_inputs_fail_without_mutating_tensors(name: str) -> None:
    q, cache, indices = _base_case()
    values: dict[str, object] = {"q": q, "cache": cache, "selected_indices": indices}
    values[name] = object()
    _assert_failure_without_mutation(
        TypeError,
        f"{name} must be a torch.Tensor",
        lambda: dsa_sparse_mla_attention_reference(
            values["q"],  # type: ignore[arg-type]
            values["cache"],  # type: ignore[arg-type]
            values["selected_indices"],  # type: ignore[arg-type]
            softmax_scale=_SCALE,
        ),
        (q, cache, indices),
    )


@pytest.mark.parametrize(
    ("name", "replacement"),
    [
        ("q", torch.empty((1, _SCORE_DIM), dtype=torch.bfloat16)),
        ("cache", torch.empty((1, _SCORE_DIM), dtype=torch.bfloat16)),
        ("selected_indices", torch.empty((1, 4), dtype=torch.int32)),
    ],
)
def test_wrong_ranks_are_rejected(name: str, replacement: torch.Tensor) -> None:
    q, cache, indices = _base_case()
    values = {"q": q, "cache": cache, "selected_indices": indices}
    values[name] = replacement
    _assert_failure_without_mutation(
        ValueError,
        f"{name} must be rank 3",
        lambda: dsa_sparse_mla_attention_reference(
            values["q"], values["cache"], values["selected_indices"], softmax_scale=_SCALE
        ),
        tuple(values.values()),
    )


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("q_width", "q score width must be 576"),
        ("cache_width", "cache score width must be 576"),
        ("cache_heads", "cache must have exactly one MQA head"),
        ("query_mismatch", "selected_indices query dimension must match q"),
        ("index_heads", "selected_indices must have exactly one MQA head"),
        ("zero_queries", "q must contain at least one query"),
        ("zero_heads", "q must contain at least one head"),
        ("zero_cache", "cache must contain at least one token"),
        ("zero_k", "selected_indices K dimension must be positive"),
    ],
)
def test_shape_and_dimension_violations_are_rejected(case: str, match: str) -> None:
    q, cache, indices = _base_case()
    if case == "q_width":
        q = torch.empty((2, 2, 575), dtype=torch.bfloat16)
    elif case == "cache_width":
        cache = torch.empty((5, 1, 575), dtype=torch.bfloat16)
    elif case == "cache_heads":
        cache = torch.empty((5, 2, _SCORE_DIM), dtype=torch.bfloat16)
    elif case == "query_mismatch":
        indices = torch.empty((3, 1, 4), dtype=torch.int32)
    elif case == "index_heads":
        indices = torch.empty((2, 2, 4), dtype=torch.int32)
    elif case == "zero_queries":
        q = torch.empty((0, 2, _SCORE_DIM), dtype=torch.bfloat16)
        indices = torch.empty((0, 1, 4), dtype=torch.int32)
    elif case == "zero_heads":
        q = torch.empty((2, 0, _SCORE_DIM), dtype=torch.bfloat16)
    elif case == "zero_cache":
        cache = torch.empty((0, 1, _SCORE_DIM), dtype=torch.bfloat16)
    else:
        indices = torch.empty((2, 1, 0), dtype=torch.int32)
    _assert_failure_without_mutation(
        ValueError,
        match,
        lambda: dsa_sparse_mla_attention_reference(q, cache, indices, softmax_scale=_SCALE),
        (q, cache, indices),
    )


@pytest.mark.parametrize(
    ("name", "dtype", "match"),
    [
        ("q", torch.float32, "q dtype must be torch.bfloat16"),
        ("q", torch.float16, "q dtype must be torch.bfloat16"),
        ("cache", torch.float32, "cache dtype must be torch.bfloat16"),
        ("cache", torch.int32, "cache dtype must be torch.bfloat16"),
        ("selected_indices", torch.int64, "selected_indices dtype must be torch.int32"),
        ("selected_indices", torch.float32, "selected_indices dtype must be torch.int32"),
    ],
)
def test_dtype_violations_are_rejected(name: str, dtype: torch.dtype, match: str) -> None:
    q, cache, indices = _base_case()
    values = {"q": q, "cache": cache, "selected_indices": indices}
    values[name] = values[name].to(dtype)
    _assert_failure_without_mutation(
        TypeError,
        match,
        lambda: dsa_sparse_mla_attention_reference(
            values["q"], values["cache"], values["selected_indices"], softmax_scale=_SCALE
        ),
        tuple(values.values()),
    )


@pytest.mark.parametrize("name", ["q", "cache", "selected_indices"])
def test_noncontiguous_innermost_dimensions_are_rejected(name: str) -> None:
    q, cache, indices = _base_case()
    if name == "q":
        q = torch.empty((2, 2, _SCORE_DIM * 2), dtype=torch.bfloat16)[..., ::2]
    elif name == "cache":
        cache = torch.empty((5, 1, _SCORE_DIM * 2), dtype=torch.bfloat16)[..., ::2]
    else:
        indices = torch.empty((2, 1, 8), dtype=torch.int32)[..., ::2]
    _assert_failure_without_mutation(
        ValueError,
        f"{name} innermost stride must be 1",
        lambda: dsa_sparse_mla_attention_reference(q, cache, indices, softmax_scale=_SCALE),
        (q, cache, indices),
    )


def test_definite_internal_overlap_is_rejected() -> None:
    q = torch.ones((1, 1, _SCORE_DIM), dtype=torch.bfloat16).expand(2, 1, _SCORE_DIM)
    cache = torch.ones((2, 1, _SCORE_DIM), dtype=torch.bfloat16)
    indices = torch.zeros((2, 1, 1), dtype=torch.int32)

    with pytest.raises(ValueError, match="q must not have internal overlap"):
        dsa_sparse_mla_attention_reference(q, cache, indices, softmax_scale=_SCALE)


def test_meta_tensors_are_rejected_before_computation() -> None:
    q = torch.empty((1, 1, _SCORE_DIM), dtype=torch.bfloat16, device="meta")
    cache = torch.empty((1, 1, _SCORE_DIM), dtype=torch.bfloat16, device="meta")
    indices = torch.empty((1, 1, 1), dtype=torch.int32, device="meta")

    with pytest.raises(ValueError, match="meta tensors are not supported"):
        dsa_sparse_mla_attention_reference(q, cache, indices, softmax_scale=_SCALE)


def test_device_mismatch_is_rejected_with_cpu_constructible_fake_tensors() -> None:
    with FakeTensorMode():
        q = torch.empty((1, 1, _SCORE_DIM), dtype=torch.bfloat16, device="cpu")
        cache = torch.empty((1, 1, _SCORE_DIM), dtype=torch.bfloat16, device="cuda")
        indices = torch.empty((1, 1, 1), dtype=torch.int32, device="cpu")
        with pytest.raises(ValueError, match="cache must be on the same device as q"):
            dsa_sparse_mla_attention_reference(q, cache, indices, softmax_scale=_SCALE)


@pytest.mark.parametrize(
    ("scale", "exc_type", "match"),
    [
        (True, TypeError, "softmax_scale must be a Python float"),
        (1, TypeError, "softmax_scale must be a Python float"),
        (float("nan"), ValueError, "softmax_scale must be finite"),
        (float("inf"), ValueError, "softmax_scale must be finite"),
        (0.0, ValueError, "softmax_scale must be positive"),
        (-1.0, ValueError, "softmax_scale must be positive"),
    ],
)
def test_invalid_softmax_scale_is_rejected(
    scale: object,
    exc_type: type[Exception],
    match: str,
) -> None:
    q, cache, indices = _base_case()
    _assert_failure_without_mutation(
        exc_type,
        match,
        lambda: dsa_sparse_mla_attention_reference(
            q,
            cache,
            indices,
            softmax_scale=scale,  # type: ignore[arg-type]
        ),
        (q, cache, indices),
    )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_q_is_rejected(bad_value: float) -> None:
    q, cache, indices = _base_case()
    q[0, 0, 0] = bad_value
    _assert_failure_without_mutation(
        ValueError,
        "q must contain only finite values",
        lambda: dsa_sparse_mla_attention_reference(q, cache, indices, softmax_scale=_SCALE),
        (q, cache, indices),
    )


def test_import_is_semantic_only_and_has_no_registration_or_cuda_side_effect() -> None:
    script = """
import json
import sys
import torch
before = set(sys.modules)
import profiling.runners.attention.dsa_sparse_mla_attention_reference as module
new_modules = set(sys.modules) - before
print(json.dumps({
    "exports": module.__all__,
    "vllm": any(name == "vllm" or name.startswith("vllm.") for name in new_modules),
    "registry": "profiling.db.registry" in sys.modules,
    "kernels": "profiling.kernels" in sys.modules,
    "cuda_initialized": torch.cuda.is_initialized(),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    evidence = json.loads(completed.stdout)

    assert evidence == {
        "exports": ["dsa_sparse_mla_attention_reference"],
        "vllm": False,
        "registry": False,
        "kernels": False,
        "cuda_initialized": False,
    }


def test_rope_free_512_wide_layout_equals_the_576_layout_with_zero_rope() -> None:
    generator = torch.Generator().manual_seed(3)
    q = torch.randn((3, 2, _VALUE_DIM), generator=generator).to(torch.bfloat16)
    cache = torch.randn((9, 1, _VALUE_DIM), generator=generator).to(torch.bfloat16)
    indices = torch.tensor([[[4, 0, -1, 8]], [[-1, -1, -1, -1]], [[2, 2, 7, 1]]], dtype=torch.int32)
    padded_q = torch.zeros((3, 2, _SCORE_DIM), dtype=torch.bfloat16)
    padded_cache = torch.zeros((9, 1, _SCORE_DIM), dtype=torch.bfloat16)
    padded_q[..., :_VALUE_DIM] = q
    padded_cache[..., :_VALUE_DIM] = cache

    actual = dsa_sparse_mla_attention_reference(q, cache, indices, softmax_scale=_SCALE)
    expected = dsa_sparse_mla_attention_reference(
        padded_q, padded_cache, indices, softmax_scale=_SCALE
    )

    assert actual.shape == (3, 2, _VALUE_DIM)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_q_and_cache_widths_must_agree() -> None:
    q = torch.zeros((1, 1, _VALUE_DIM), dtype=torch.bfloat16)
    cache = torch.zeros((2, 1, _SCORE_DIM), dtype=torch.bfloat16)
    indices = torch.zeros((1, 1, 1), dtype=torch.int32)
    with pytest.raises(ValueError, match="cache score width must be 512"):
        dsa_sparse_mla_attention_reference(q, cache, indices, softmax_scale=_SCALE)
