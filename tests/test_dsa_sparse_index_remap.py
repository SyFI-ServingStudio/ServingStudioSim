"""Registration and Torch-runner tests for DSA sparse-index remapping."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import fields
from re import escape
from types import SimpleNamespace
from typing import Any, get_type_hints

import pytest
import torch

from profiling import perf_api
from profiling.db.args import DType
from profiling.db.batch import coerce_args
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import MetricFamily, find_kernel_profiler_spec, known_backends
from profiling.kernels.dsa_sparse_index_remap import KIND, DsaSparseIndexRemapArgs
from profiling.runners.attention.dsa_sparse_index_remap_reference import (
    dsa_sparse_index_remap_reference,
)
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented

_BASE_SPEC = {
    "num_queries": 4,
    "num_requests": 2,
    "selected_k": 2048,
    "block_size": 64,
    "max_blocks_per_request": 2048,
    "request_row_counts": "u:2x2",
    "local_span_lengths": "g:(0,1,64,130)x1",
    "valid_counts": "g:(0,1,2,4)x1",
    "index_distribution": "recent_contiguous",
    "page_table_mapping": "request_contiguous",
    "workspace_partition": "none",
    "return_valid_counts": False,
    "index_dtype": "int32",
}


def _small_validated(
    runner: Any,
    *,
    num_queries: int = 4,
    num_requests: int = 2,
    selected_k: int = 8,
    block_size: int = 4,
    max_blocks_per_request: int = 5,
    request_row_counts: tuple[int, ...] = (2, 2),
    local_span_lengths: tuple[int, ...] = (0, 1, 9, 17),
    valid_counts: tuple[int, ...] = (0, 1, 4, 8),
    index_distribution: str = "recent_contiguous",
    page_table_mapping: str = "request_contiguous",
    workspace_partition: Any = None,
    return_valid_counts: bool = False,
) -> Any:
    request_ids = tuple(
        request for request, count in enumerate(request_row_counts) for _ in range(count)
    )
    return runner._ValidatedArgs(
        num_queries=num_queries,
        num_requests=num_requests,
        selected_k=selected_k,
        block_size=block_size,
        max_blocks_per_request=max_blocks_per_request,
        request_row_counts=request_row_counts,
        request_ids=request_ids,
        local_span_lengths=local_span_lengths,
        valid_counts=valid_counts,
        index_distribution=index_distribution,
        page_table_mapping=page_table_mapping,
        workspace_partition=workspace_partition,
        return_valid_counts=return_valid_counts,
    )


def test_args_order_types_and_coercion() -> None:
    assert [field.name for field in fields(DsaSparseIndexRemapArgs)] == [
        "num_queries",
        "num_requests",
        "selected_k",
        "block_size",
        "max_blocks_per_request",
        "request_row_counts",
        "local_span_lengths",
        "valid_counts",
        "index_distribution",
        "page_table_mapping",
        "workspace_partition",
        "return_valid_counts",
        "index_dtype",
    ]
    assert list(get_type_hints(DsaSparseIndexRemapArgs).values()) == [
        int,
        int,
        int,
        int,
        int,
        str,
        str,
        str,
        str,
        str,
        str,
        bool,
        str,
    ]
    args = coerce_args(DsaSparseIndexRemapArgs, _BASE_SPEC)
    assert args == DsaSparseIndexRemapArgs(**_BASE_SPEC)


def test_registration_support_family_environment_runner_ref_and_facades() -> None:
    spec = find_kernel_profiler_spec(KIND, "torch")
    vllm_spec = find_kernel_profiler_spec(KIND, "vllm_triton")

    assert KIND == "dsa_sparse_index_remap"
    assert known_backends(KIND) == ["torch", "vllm_triton"]
    assert spec.kernel_kind == spec.table_name == KIND
    assert spec.args_schema is DsaSparseIndexRemapArgs
    assert spec.metric_family is MetricFamily.COMPUTE
    assert spec.batch_outlier_policy == BatchOutlierPolicy()
    assert spec.supports.compute is None
    assert spec.supports.kv is None
    assert spec.supports.gpus == frozenset({"NVIDIA H200"})
    assert spec.supports.allows(DType.FP16, gpu="NVIDIA H200")
    assert spec.supports.allows(DType.FP32, DType.FP8_E4M3, gpu="NVIDIA H200")
    assert not spec.supports.allows(DType.FP32, gpu="NVIDIA H100")
    assert spec.subprocess_env is None
    assert spec.runner_ref.module_name == ("profiling.runners.attention.dsa_sparse_index_remap")
    assert spec.runner_ref.function_name == "profile_dsa_sparse_index_remap_torch"
    assert vllm_spec.supports.compute is None
    assert vllm_spec.supports.gpus == frozenset({"NVIDIA B200"})
    assert vllm_spec.subprocess_env == "vllm_env"
    assert vllm_spec.runner_ref.function_name == ("profile_dsa_sparse_index_remap_vllm_triton")
    assert hasattr(perf_api, "get_dsa_sparse_index_remap_times")
    assert hasattr(perf_api, "count_missing_dsa_sparse_index_remap")


def test_registry_and_runner_loading_are_lazy() -> None:
    script = """
import sys
import profiling.kernels
from profiling.db.registry import find_kernel_profiler_spec

blocked = ('torch', 'vllm', 'triton')
print(any(name == item or name.startswith(item + '.') for item in blocked for name in sys.modules))
print('profiling.runners.attention.dsa_sparse_index_remap' in sys.modules)
print('profiling.runners.attention.dsa_sparse_index_remap_reference' in sys.modules)
runner = find_kernel_profiler_spec('dsa_sparse_index_remap', 'torch').runner_ref.load()
print(runner.__module__)
print(runner.__name__)
print(any(name == item or name.startswith(item + '.') for item in blocked for name in sys.modules))
print('profiling.runners.attention.dsa_sparse_index_remap_reference' in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == [
        "False",
        "False",
        "False",
        "profiling.runners.attention.dsa_sparse_index_remap",
        "profile_dsa_sparse_index_remap_torch",
        "False",
        "False",
    ]


@pytest.mark.parametrize(
    ("encoded", "name", "rows", "minimum", "maximum", "clipped", "expected"),
    [
        ("u:2x3", "request_row_counts", 3, 1, 8, False, (2, 2, 2)),
        ("r:1..3", "request_row_counts", 3, 1, 8, False, (1, 2, 3)),
        ("g:(1,3,2)x2", "request_row_counts", 6, 1, 8, False, (1, 3, 2) * 2),
        ("u:0x4", "local_span_lengths", 4, 0, 131072, False, (0, 0, 0, 0)),
        ("r:0..3", "local_span_lengths", 4, 0, 131072, False, (0, 1, 2, 3)),
        ("g:(0,64,7)x1", "local_span_lengths", 3, 0, 131072, False, (0, 64, 7)),
        ("u:2048x1", "valid_counts", 1, 0, 131072, True, (2048,)),
        ("r:1..128", "valid_counts", 128, 0, 131072, True, tuple(range(1, 129))),
        (
            "c:1922..2049@2048",
            "valid_counts",
            128,
            0,
            131072,
            True,
            tuple(min(value, 2048) for value in range(1922, 2050)),
        ),
        ("g:(0,2,1)x1", "valid_counts", 3, 0, 131072, True, (0, 2, 1)),
    ],
)
def test_canonical_vector_examples_round_trip(
    encoded: str,
    name: str,
    rows: int,
    minimum: int,
    maximum: int,
    clipped: bool,
    expected: tuple[int, ...],
) -> None:
    from profiling.runners.attention.dsa_sparse_index_remap import (
        _decode_vector,
        _encode_vector,
    )

    decoded = _decode_vector(
        encoded,
        name=name,
        num_rows=rows,
        minimum=minimum,
        maximum=maximum,
        allow_clipped=clipped,
        selected_k=2048 if clipped else None,
    )
    assert decoded == expected
    assert (
        _encode_vector(
            decoded,
            name=name,
            minimum=minimum,
            maximum=maximum,
            allow_clipped=clipped,
            selected_k=2048 if clipped else None,
        )
        == encoded
    )


@pytest.mark.parametrize(
    "encoded",
    [
        "",
        "u:1x1 ",
        " u:1x1",
        "u:+1x1",
        "u:-1x1",
        "u:01x1",
        "u:1x01",
        "u:1;1",
        "r:1-2",
        "r:1..",
        "g:()x1",
        "g:(1,)x1",
        "g:(1;2)x1",
        "g:(１)x1",
        "c:1..3@2;",
    ],
)
def test_vector_parser_rejects_malformed_compact_syntax(encoded: str) -> None:
    from profiling.runners.attention.dsa_sparse_index_remap import _decode_vector

    with pytest.raises(ValueError, match="vector"):
        _decode_vector(
            encoded,
            name="vector",
            num_rows=1,
            minimum=0,
            maximum=131072,
            allow_clipped=True,
            selected_k=2048,
        )


@pytest.mark.parametrize(
    ("encoded", "rows", "allow_clipped", "match"),
    [
        ("u:1x0", 1, False, "positive"),
        ("g:(1)x0", 1, False, "positive"),
        ("u:1x2", 1, False, "expands to 2"),
        ("u:1x999999999999", 1, False, "expands to 999999999999"),
        ("r:0..999999999999", 1, False, "expands to 1000000000000"),
        ("g:(1,2)x999999999999", 1, False, "expands to 1999999999998"),
        ("u:131073x1", 1, False, "0..131072"),
        ("r:2..2", 1, False, "strict"),
        ("r:3..2", 1, False, "strict"),
        ("c:1..3@2", 3, False, "u:, r:, or g:"),
        ("c:2048..2049@2048", 2, True, "first <"),
        ("c:1..2049@1024", 2049, True, "cap must equal"),
        ("c:1..131073@2048", 131073, True, "last <= 131072"),
    ],
)
def test_vector_parser_rejects_rows_ranges_overflow_and_bad_clipping(
    encoded: str,
    rows: int,
    allow_clipped: bool,
    match: str,
) -> None:
    from profiling.runners.attention.dsa_sparse_index_remap import _decode_vector

    with pytest.raises(ValueError, match=match):
        _decode_vector(
            encoded,
            name="vector",
            num_rows=rows,
            minimum=0,
            maximum=131072,
            allow_clipped=allow_clipped,
            selected_k=2048 if allow_clipped else None,
        )


@pytest.mark.parametrize(
    ("encoded", "rows", "canonical"),
    [
        ("g:(7)x3", 3, "u:7x3"),
        ("g:(1,2,3)x1", 3, "r:1..3"),
        ("g:(1,2,1,2)x1", 4, "g:(1,2)x2"),
        ("g:(1,2)x1", 2, "r:1..2"),
        ("g:(2047,2048,2048)x1", 3, "c:2047..2049@2048"),
    ],
)
def test_vector_parser_rejects_ambiguous_nonminimal_encodings(
    encoded: str,
    rows: int,
    canonical: str,
) -> None:
    from profiling.runners.attention.dsa_sparse_index_remap import _decode_vector

    with pytest.raises(ValueError, match=escape(canonical)):
        _decode_vector(
            encoded,
            name="vector",
            num_rows=rows,
            minimum=0,
            maximum=131072,
            allow_clipped=True,
            selected_k=2048,
        )


def test_vector_helpers_reject_non_string_and_non_integer_values() -> None:
    from profiling.runners.attention.dsa_sparse_index_remap import (
        _decode_vector,
        _encode_vector,
    )

    with pytest.raises(TypeError, match="vector must be a string"):
        _decode_vector(
            7,
            name="vector",
            num_rows=1,
            minimum=0,
            maximum=8,
        )
    with pytest.raises(TypeError, match="values must be integers"):
        _encode_vector([1, True], name="vector", minimum=0, maximum=8)


def test_arg_validation_enforces_vector_sum_and_rowwise_count_bounds() -> None:
    from profiling.runners.attention.dsa_sparse_index_remap import _validate_args

    with pytest.raises(ValueError, match="must sum to num_queries 4"):
        _validate_args(**(_BASE_SPEC | {"request_row_counts": "u:1x2"}))
    with pytest.raises(ValueError, match="valid_counts row 1"):
        _validate_args(
            **(
                _BASE_SPEC
                | {
                    "local_span_lengths": "g:(0,1,64,130)x1",
                    "valid_counts": "g:(0,2,2,4)x1",
                }
            )
        )


@pytest.mark.parametrize(
    ("encoded", "num_requests", "expected"),
    [
        ("none", 3, None),
        ("suffix:0@3", 3, (0, (3,))),
        ("suffix:1@2", 3, (1, (2,))),
        ("suffix:1@1,1", 3, (1, (1, 1))),
    ],
)
def test_workspace_partition_accepted_forms(
    encoded: str,
    num_requests: int,
    expected: tuple[int, tuple[int, ...]] | None,
) -> None:
    from profiling.runners.attention.dsa_sparse_index_remap import (
        _decode_workspace_partition,
    )

    actual = _decode_workspace_partition(encoded, num_requests=num_requests)
    if expected is None:
        assert actual is None
    else:
        assert actual is not None
        assert (actual.decode_requests, actual.chunk_sizes) == expected


@pytest.mark.parametrize(
    "encoded",
    [
        "",
        "none ",
        " none",
        "suffix:+0@1",
        "suffix:-1@1",
        "suffix:00@1",
        "suffix:0@01",
        "suffix:0@",
        "suffix:0@1,",
        "suffix:0@1;1",
        "suffix:０@1",
    ],
)
def test_workspace_partition_rejects_malformed_noncanonical_syntax(encoded: str) -> None:
    from profiling.runners.attention.dsa_sparse_index_remap import (
        _decode_workspace_partition,
    )

    with pytest.raises(ValueError, match="workspace_partition"):
        _decode_workspace_partition(encoded, num_requests=3)


@pytest.mark.parametrize(
    ("encoded", "match"),
    [
        ("suffix:3@1", "D must be in"),
        ("suffix:0@0,3", "chunk sizes must be positive"),
        ("suffix:1@1", "must sum"),
        ("suffix:1@3", "must sum"),
    ],
)
def test_workspace_partition_rejects_bounds_and_chunk_totals(
    encoded: str,
    match: str,
) -> None:
    from profiling.runners.attention.dsa_sparse_index_remap import (
        _decode_workspace_partition,
    )

    with pytest.raises(ValueError, match=match):
        _decode_workspace_partition(encoded, num_requests=3)


def test_workspace_ids_and_starts_follow_suffix_chunks_and_request_maxima() -> None:
    from profiling.runners.attention import dsa_sparse_index_remap as runner

    common = _BASE_SPEC | {
        "num_queries": 6,
        "num_requests": 3,
        "request_row_counts": "u:2x3",
        "local_span_lengths": "g:(10,20,30,40,50,60)x1",
        "valid_counts": "u:0x6",
    }
    one_chunk = runner._validate_args(**(common | {"workspace_partition": "suffix:1@2"}))
    split_chunks = runner._validate_args(**(common | {"workspace_partition": "suffix:1@1,1"}))
    pure_prefill = runner._validate_args(**(common | {"workspace_partition": "suffix:0@3"}))

    assert runner._derive_workspace_metadata(one_chunk) == (
        (-1, -1, 0, 0, 1, 1),
        (0, 40),
    )
    assert runner._derive_workspace_metadata(split_chunks) == (
        (-1, -1, 0, 0, 1, 1),
        (0, 0),
    )
    assert runner._derive_workspace_metadata(pure_prefill) == (
        (0, 0, 1, 1, 2, 2),
        (0, 20, 60),
    )


def test_spec5_rounded_8k_prefill_is_supported() -> None:
    from profiling.runners.attention import dsa_sparse_index_remap as runner

    args = runner._validate_args(**(_BASE_SPEC | {
        "num_queries": 8196,
        "num_requests": 1,
        "request_row_counts": "u:8196x1",
        "local_span_lengths": "r:1..8196",
        "valid_counts": "c:1..8196@2048",
    }))
    assert args.request_row_counts == (8196,)
    assert len(args.request_ids) == 8196
    assert args.local_span_lengths[-1] == 8196
    assert args.valid_counts[-1] == 2048


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"num_queries": 0}, "1..16384"),
        ({"num_queries": 16385}, "1..16384"),
        ({"num_requests": 0}, "1..min"),
        ({"num_requests": 5}, "1..min"),
        ({"num_requests": 257}, "1..min"),
        ({"selected_k": 1024}, "selected_k must be 2048"),
        ({"block_size": 128}, "block_size must be 64"),
        ({"max_blocks_per_request": 16385}, "max_blocks_per_request must be in 1..16384"),
        ({"index_dtype": "int64"}, "index_dtype must be int32"),
        ({"index_distribution": "random"}, "index_distribution"),
        ({"page_table_mapping": "random"}, "page_table_mapping"),
    ],
)
def test_unsupported_domain_fails_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
    match: str,
) -> None:
    from profiling.runners.attention import dsa_sparse_index_remap as runner

    allocated = False

    def fail_if_allocated(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal allocated
        allocated = True
        raise AssertionError("unsupported values must fail before allocation")

    monkeypatch.setattr(runner, "_build_operands", fail_if_allocated)
    spec = dict(_BASE_SPEC)
    spec.update(overrides)
    with pytest.raises(ProfilerNotImplemented, match=match):
        runner.profile_dsa_sparse_index_remap_torch(**spec)
    assert not allocated


def test_glm_full_context_block_table_width_is_supported() -> None:
    from profiling.runners.attention import dsa_sparse_index_remap as runner

    validated = runner._validate_args(
        **(
            _BASE_SPEC
            | {
                "num_queries": 1,
                "num_requests": 1,
                "max_blocks_per_request": 16384,
                "request_row_counts": "u:1x1",
                "local_span_lengths": "u:1048576x1",
                "valid_counts": "u:2048x1",
            }
        )
    )

    assert validated.max_blocks_per_request == 16384
    assert validated.local_span_lengths == (1048576,)


@pytest.mark.parametrize(
    ("name", "value", "match"),
    [
        ("num_queries", True, "num_queries must be an integer"),
        ("num_requests", 1.0, "num_requests must be an integer"),
        ("selected_k", "2048", "selected_k must be an integer"),
        ("return_valid_counts", 1, "return_valid_counts must be a Python bool"),
        ("index_distribution", 1, "index_distribution must be a string"),
        ("page_table_mapping", None, "page_table_mapping must be a string"),
    ],
)
def test_malformed_types_fail_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: Any,
    match: str,
) -> None:
    from profiling.runners.attention import dsa_sparse_index_remap as runner

    monkeypatch.setattr(
        runner,
        "_build_operands",
        lambda *_args, **_kwargs: pytest.fail("must fail before allocation"),
    )
    with pytest.raises(TypeError, match=match):
        runner.profile_dsa_sparse_index_remap_torch(**(_BASE_SPEC | {name: value}))


def test_cuda_and_gpu_support_failures_are_typed() -> None:
    from profiling.runners.attention.dsa_sparse_index_remap import (
        _require_b200,
        _require_h200,
    )

    no_cuda = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with pytest.raises(ProfilerNotImplemented, match="CUDA is required"):
        _require_h200(no_cuda)

    h100 = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_name=lambda _device: "NVIDIA H100",
        )
    )
    with pytest.raises(ProfilerNotImplemented, match="requires NVIDIA H200"):
        _require_h200(h100)
    with pytest.raises(ProfilerNotImplemented, match="requires NVIDIA B200"):
        _require_b200(h100)


@pytest.mark.parametrize(
    ("mapping", "expected"),
    [
        ("request_contiguous", [[0, 1, 2, 3], [4, 5, 6, 7]]),
        ("interleaved_requests", [[0, 2, 4, 6], [1, 3, 5, 7]]),
        ("reverse_within_request", [[3, 2, 1, 0], [7, 6, 5, 4]]),
    ],
)
def test_page_table_mappings_have_exact_values(mapping: str, expected: list[list[int]]) -> None:
    from profiling.runners.attention import dsa_sparse_index_remap as runner

    validated = _small_validated(
        runner,
        max_blocks_per_request=4,
        page_table_mapping=mapping,
    )
    table = runner._build_block_table(torch, validated, device=torch.device("cpu"))

    assert table.dtype is torch.int32 and table.is_contiguous()
    assert table.tolist() == expected


def test_fixed_permutation_page_table_is_deterministic_bijective() -> None:
    from profiling.runners.attention import dsa_sparse_index_remap as runner

    validated = _small_validated(
        runner,
        max_blocks_per_request=7,
        page_table_mapping="fixed_permutation",
    )
    first = runner._build_block_table(torch, validated, device=torch.device("cpu"))
    second = runner._build_block_table(torch, validated, device=torch.device("cpu"))

    assert torch.equal(first, second)
    assert sorted(first.flatten().tolist()) == list(range(14))


@pytest.mark.parametrize(
    "distribution",
    [
        "recent_contiguous",
        "unique_scattered_blocks",
        "clustered_blocks",
        "uniform_stride",
    ],
)
def test_local_index_distributions_are_deterministic_unique_and_in_span(
    distribution: str,
) -> None:
    from profiling.runners.attention.dsa_sparse_index_remap import _row_local_indices

    first = _row_local_indices(
        torch,
        row=3,
        count=31,
        span=67,
        distribution=distribution,
        block_size=64,
        device=torch.device("cpu"),
    )
    second = _row_local_indices(
        torch,
        row=3,
        count=31,
        span=67,
        distribution=distribution,
        block_size=64,
        device=torch.device("cpu"),
    )

    assert torch.equal(first, second)
    assert first.dtype is torch.int64
    assert first.numel() == torch.unique(first).numel() == 31
    assert int(first.min()) >= 0 and int(first.max()) < 67
    if distribution == "recent_contiguous":
        assert first.tolist() == list(range(36, 67))


def test_operand_layout_request_ids_valid_prefix_tails_and_workspace() -> None:
    from profiling.runners.attention import dsa_sparse_index_remap as runner

    partition = runner._WorkspacePartition(decode_requests=1, chunk_sizes=(1,))
    validated = _small_validated(
        runner,
        workspace_partition=partition,
        return_valid_counts=True,
        index_distribution="unique_scattered_blocks",
        page_table_mapping="interleaved_requests",
    )
    operands = runner._build_operands(torch, validated, device=torch.device("cpu"))

    assert operands.req_id.tolist() == [0, 0, 1, 1]
    assert operands.req_id.dtype is torch.int32 and operands.req_id.is_contiguous()
    assert operands.block_table.shape == (2, 5) and operands.block_table.is_contiguous()
    assert operands.token_indices.shape == (4, 8) and operands.token_indices.is_contiguous()
    assert operands.output.shape == (4, 8) and operands.output.is_contiguous()
    assert operands.counts is not None and operands.counts.shape == (4,)
    assert operands.prefill_workspace_request_ids.tolist() == [-1, -1, 0, 0]
    assert operands.prefill_workspace_starts.tolist() == [0]
    for row, (count, span) in enumerate(
        zip(validated.valid_counts, validated.local_span_lengths, strict=True)
    ):
        valid = operands.token_indices[row, :count]
        assert valid.numel() == torch.unique(valid).numel()
        if count:
            assert ((valid >= 0) & (valid < span)).all()
        assert operands.token_indices[row, count:].tolist() == [-1] * (8 - count)


def test_operand_construction_is_repeatable_for_every_mode() -> None:
    from profiling.runners.attention import dsa_sparse_index_remap as runner

    for distribution in sorted(runner._INDEX_DISTRIBUTIONS):
        for mapping in sorted(runner._PAGE_TABLE_MAPPINGS):
            validated = _small_validated(
                runner,
                index_distribution=distribution,
                page_table_mapping=mapping,
            )
            first = runner._build_operands(torch, validated, device=torch.device("cpu"))
            second = runner._build_operands(torch, validated, device=torch.device("cpu"))
            assert torch.equal(first.req_id, second.req_id)
            assert torch.equal(first.block_table, second.block_table)
            assert torch.equal(first.token_indices, second.token_indices)


def test_max_domain_builder_avoids_full_q_or_cache_sized_int64_ramps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from profiling.runners.attention import dsa_sparse_index_remap as runner

    validated = runner._ValidatedArgs(
        num_queries=4096,
        num_requests=256,
        selected_k=2048,
        block_size=64,
        max_blocks_per_request=2048,
        request_row_counts=(16,) * 256,
        request_ids=tuple(request for request in range(256) for _ in range(16)),
        local_span_lengths=(0,) * 4096,
        valid_counts=(0,) * 4096,
        index_distribution="uniform_stride",
        page_table_mapping="fixed_permutation",
        workspace_partition=None,
        return_valid_counts=False,
    )
    seen_arange_stops: list[int] = []
    original_arange = torch.arange

    def recording_arange(*args: Any, **kwargs: Any) -> Any:
        stop = args[0] if len(args) == 1 else args[1]
        seen_arange_stops.append(int(stop))
        return original_arange(*args, **kwargs)

    monkeypatch.setattr(torch, "arange", recording_arange)
    monkeypatch.setattr(runner, "_validate_operand_invariants", lambda *_args: None)
    operands = runner._build_operands(torch, validated, device=torch.device("cpu"))

    assert operands.req_id.shape == (4096,)
    assert operands.block_table.shape == (256, 2048)
    assert operands.token_indices.shape == operands.output.shape == (4096, 2048)
    assert max(seen_arange_stops) == 2048


def test_composite_matches_reference_global_and_preserves_duplicates_and_oob() -> None:
    from profiling.runners.attention import dsa_sparse_index_remap as runner

    req_id = torch.tensor([0, 1, 1], dtype=torch.int32)
    block_table = torch.tensor([[4, 7], [2, 9]], dtype=torch.int32)
    token_indices = torch.tensor(
        [[-1, -7, 0, 64, 128], [65, 65, 1, -1, 999], [0, 63, 64, 127, -2]],
        dtype=torch.int32,
    )
    operands = runner._Operands(
        req_id=req_id,
        block_table=block_table,
        token_indices=token_indices,
        prefill_workspace_request_ids=None,
        prefill_workspace_starts=None,
        output=torch.empty_like(token_indices),
        counts=None,
    )
    snapshots = tuple(tensor.clone() for tensor in (req_id, block_table, token_indices))

    actual = runner._torch_composite(torch, operands, block_size=64, row_chunk_size=2)
    expected = dsa_sparse_index_remap_reference(req_id, block_table, token_indices, block_size=64)

    assert torch.equal(actual, expected)
    assert actual.tolist()[1][:3] == [577, 577, 129]
    assert actual.is_contiguous() and actual.dtype is torch.int32
    for tensor, snapshot in zip((req_id, block_table, token_indices), snapshots, strict=True):
        assert torch.equal(tensor, snapshot)
        assert not torch._C._overlaps(actual, tensor)


def test_composite_matches_reference_workspace_and_count_modes() -> None:
    from profiling.runners.attention import dsa_sparse_index_remap as runner

    req_id = torch.tensor([0, 1, 1, 2], dtype=torch.int32)
    block_table = torch.tensor([[4, 7], [2, 9], [5, 8]], dtype=torch.int32)
    token_indices = torch.tensor(
        [[-1, -2, 128, 999], [0, 64, -1, 128], [1, 65, 65, -1], [0, 63, 127, -3]],
        dtype=torch.int32,
    )
    workspace_ids = torch.tensor([-1, 0, 0, 1], dtype=torch.int32)
    workspace_starts = torch.tensor([1000, 0], dtype=torch.int32)
    for return_counts in (False, True):
        operands = runner._Operands(
            req_id=req_id,
            block_table=block_table,
            token_indices=token_indices,
            prefill_workspace_request_ids=workspace_ids,
            prefill_workspace_starts=workspace_starts,
            output=torch.empty_like(token_indices),
            counts=(torch.empty((4,), dtype=torch.int32) if return_counts else None),
        )
        actual = runner._torch_composite(torch, operands, block_size=64, row_chunk_size=3)
        expected = dsa_sparse_index_remap_reference(
            req_id,
            block_table,
            token_indices,
            block_size=64,
            prefill_workspace_request_ids=workspace_ids,
            prefill_workspace_starts=workspace_starts,
            return_valid_counts=return_counts,
        )
        actual_values = actual if isinstance(actual, tuple) else (actual,)
        expected_values = expected if isinstance(expected, tuple) else (expected,)
        assert len(actual_values) == len(expected_values)
        assert all(
            torch.equal(actual_value, expected_value)
            for actual_value, expected_value in zip(actual_values, expected_values, strict=True)
        )
        if return_counts:
            assert actual_values[1].tolist() == [0, 2, 3, 3]


def test_composite_is_independent_of_row_chunking_and_handles_flattened_next_n2() -> None:
    from profiling.runners.attention import dsa_sparse_index_remap as runner

    req_id = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int32)
    block_table = torch.arange(12, dtype=torch.int32).reshape(3, 4).add(3)
    token_indices = torch.tensor(
        [
            [-1, -1, -1, -1],
            [0, -1, -1, -1],
            [0, 64, -1, -1],
            [1, 65, 129, -1],
            [2, 66, 130, 194],
            [3, 67, 131, 195],
        ],
        dtype=torch.int32,
    )

    def run(chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
        operands = runner._Operands(
            req_id=req_id,
            block_table=block_table,
            token_indices=token_indices,
            prefill_workspace_request_ids=None,
            prefill_workspace_starts=None,
            output=torch.empty_like(token_indices),
            counts=torch.empty((6,), dtype=torch.int32),
        )
        return runner._torch_composite(torch, operands, block_size=64, row_chunk_size=chunk)

    rowwise = run(1)
    chunked = run(4)
    all_rows = run(64)
    assert torch.equal(rowwise[0], chunked[0]) and torch.equal(rowwise[0], all_rows[0])
    assert torch.equal(rowwise[1], chunked[1]) and torch.equal(rowwise[1], all_rows[1])


def test_pre_timing_correctness_checks_reference_repeatability_and_aliasing() -> None:
    from profiling.runners.attention import dsa_sparse_index_remap as runner

    validated = _small_validated(runner, return_valid_counts=True)
    operands = runner._build_operands(torch, validated, device=torch.device("cpu"))
    runner._check_correctness(torch, validated, operands)

    aliased = runner._Operands(
        req_id=operands.req_id,
        block_table=operands.block_table,
        token_indices=operands.token_indices,
        prefill_workspace_request_ids=None,
        prefill_workspace_starts=None,
        output=operands.token_indices,
        counts=operands.counts,
    )
    with pytest.raises(AssertionError, match="aliases an input"):
        runner._check_correctness(torch, validated, aliased)


@pytest.mark.parametrize(
    ("workspace_ids", "return_counts"),
    [
        (None, False),
        (None, True),
        ((-1, 0), False),
        ((-1, 0), True),
        ((0, 1), False),
        ((0, 1), True),
    ],
)
def test_logical_accounting_exact_for_all_workspace_and_count_modes(
    workspace_ids: tuple[int, ...] | None,
    return_counts: bool,
) -> None:
    from profiling.runners.attention.dsa_sparse_index_remap import _logical_bytes

    q = 2
    k = 2048
    counts = (3, 4)
    tiles = q * k // 128
    expected = 4 * q * k + 4 * q * k + 4 * tiles
    expected += 4 * (
        7
        if workspace_ids is None
        else sum(
            count for count, workspace in zip(counts, workspace_ids, strict=True) if workspace == -1
        )
    )
    if workspace_ids is not None:
        expected += 4 * tiles
        expected += 4 * sum(value >= 0 for value in workspace_ids) * k // 128
    if return_counts:
        expected += 4 * q + 8 * tiles

    assert (
        _logical_bytes(
            num_queries=q,
            selected_k=k,
            valid_counts=counts,
            workspace_ids=workspace_ids,
            return_valid_counts=return_counts,
        )
        == expected
    )


def test_nominal_flops_are_zero() -> None:
    from profiling.runners.attention.dsa_sparse_index_remap import _logical_flops

    assert _logical_flops() == 0


def test_vllm_triton_launch_forwards_the_production_wrapper_contract() -> None:
    from profiling.runners.attention import dsa_sparse_index_remap as runner

    validated = _small_validated(runner, selected_k=8, block_size=4)
    operands = runner._Operands(
        req_id=object(),
        block_table=object(),
        token_indices=object(),
        prefill_workspace_request_ids=None,
        prefill_workspace_starts=None,
        output=object(),
        counts=None,
    )
    calls = []

    def callable_(*args, **kwargs):
        calls.append((args, kwargs))
        return "output"

    assert runner._launch_vllm_triton(callable_, operands, validated) == "output"
    assert calls == [
        (
            (operands.req_id, operands.block_table, operands.token_indices),
            {
                "BLOCK_SIZE": 4,
                "NUM_TOPK_TOKENS": 8,
                "HAS_PREFILL_WORKSPACE": False,
                "prefill_workspace_request_ids": None,
                "prefill_workspace_starts": None,
                "return_valid_counts": False,
            },
        )
    ]


def test_profile_times_complete_composite_and_returns_live_zero_flop_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from profiling.runners.attention import dsa_sparse_index_remap as runner

    live_output = object()
    operands = runner._Operands(
        req_id=object(),
        block_table=object(),
        token_indices=object(),
        prefill_workspace_request_ids=None,
        prefill_workspace_starts=None,
        output=object(),
        counts=None,
    )
    calls: list[Any] = []
    monkeypatch.setattr(runner, "_require_h200", lambda _torch: calls.append("gpu"))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        runner,
        "_build_operands",
        lambda *_args, **_kwargs: calls.append("build") or operands,
    )
    monkeypatch.setattr(
        runner,
        "_check_correctness",
        lambda *_args, **_kwargs: calls.append("correctness"),
    )
    monkeypatch.setattr(
        runner,
        "_derive_workspace_metadata",
        lambda _validated: (None, None),
    )

    def composite(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("composite")
        return live_output

    def timer(callable_: Any, **kwargs: Any) -> float:
        calls.append(("timer", kwargs))
        assert callable_() is live_output
        return 2.0

    def energy(callable_: Any, *, warmup: int, per_iter_time_ms: float) -> float:
        calls.append(("energy", warmup, per_iter_time_ms))
        assert callable_() is live_output
        return 0.25

    monkeypatch.setattr(runner, "_torch_composite", composite)
    monkeypatch.setattr(runner.Timer, "cupti", timer)
    monkeypatch.setattr(runner.Energy, "perf", energy)

    metrics = runner.profile_dsa_sparse_index_remap_torch(**_BASE_SPEC)

    logical_bytes = runner._logical_bytes(
        num_queries=4,
        selected_k=2048,
        valid_counts=(0, 1, 2, 4),
        workspace_ids=None,
        return_valid_counts=False,
    )
    assert metrics.time_ms == 2.0
    assert metrics.energy_j == 0.25
    assert metrics.tflops == 0.0
    assert metrics.memory_bandwidth_gbps == logical_bytes / 0.002 / 1e9
    assert calls == [
        "gpu",
        "build",
        "correctness",
        ("timer", {}),
        "composite",
        ("energy", 5, 2.0),
        "composite",
    ]


def test_profile_preserves_typed_unavailable_oom_and_execution_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from profiling.runners.attention import dsa_sparse_index_remap as runner

    def unavailable(_torch: Any) -> None:
        raise ProfilerNotImplemented("CUDA is required")

    monkeypatch.setattr(runner, "_require_h200", unavailable)
    with pytest.raises(ProfilerNotImplemented, match="CUDA is required"):
        runner.profile_dsa_sparse_index_remap_torch(**_BASE_SPEC)

    monkeypatch.setattr(runner, "_require_h200", lambda _torch: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    def oom(*_args: Any, **_kwargs: Any) -> Any:
        raise torch.OutOfMemoryError("capacity")

    monkeypatch.setattr(runner, "_build_operands", oom)
    with pytest.raises(OOMError, match="ran out of GPU memory") as oom_info:
        runner.profile_dsa_sparse_index_remap_torch(**_BASE_SPEC)
    assert isinstance(oom_info.value.__cause__, torch.OutOfMemoryError)

    def runtime(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("launch failed")

    monkeypatch.setattr(runner, "_build_operands", runtime)
    with pytest.raises(KernelLaunchFailed, match="semantic composite failed") as launch_info:
        runner.profile_dsa_sparse_index_remap_torch(**_BASE_SPEC)
    assert isinstance(launch_info.value.__cause__, RuntimeError)
