"""Registration and Torch-runner tests for selected sparse MLA attention."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import fields
from re import escape
from types import SimpleNamespace

import pytest
import torch

from profiling import perf_api
from profiling.db.args import DType
from profiling.db.batch import coerce_args
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import MetricFamily, find_kernel_profiler_spec, known_backends
from profiling.kernels.dsa_sparse_mla_attention import (
    KIND,
    DsaSparseMlaAttentionArgs,
)
from profiling.runners.attention.dsa_sparse_mla_attention_reference import (
    dsa_sparse_mla_attention_reference,
)
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented

_BASE_SPEC = {
    "num_queries": 1,
    "num_cache_tokens": 2048,
    "num_heads": 64,
    "num_kv_heads": 1,
    "selected_k": 2048,
    "latent_dim": 512,
    "rope_dim": 64,
    "value_dim": 512,
    "softmax_scale": 0.0625,
    "q_dtype": "bf16",
    "cache_dtype": "bf16",
    "index_dtype": "int32",
    "output_dtype": "bf16",
    "valid_counts": "u:2048x1",
    "index_distribution": "recent_contiguous",
    "cache_layout": "token_major_mqa_bf16_latent_rope",
}


def test_args_field_order_and_coercion() -> None:
    assert [field.name for field in fields(DsaSparseMlaAttentionArgs)] == [
        "num_queries",
        "num_cache_tokens",
        "num_heads",
        "num_kv_heads",
        "selected_k",
        "latent_dim",
        "rope_dim",
        "value_dim",
        "softmax_scale",
        "q_dtype",
        "cache_dtype",
        "index_dtype",
        "output_dtype",
        "valid_counts",
        "index_distribution",
        "cache_layout",
    ]
    args = coerce_args(DsaSparseMlaAttentionArgs, _BASE_SPEC)
    assert args == DsaSparseMlaAttentionArgs(
        num_queries=1,
        num_cache_tokens=2048,
        num_heads=64,
        num_kv_heads=1,
        selected_k=2048,
        latent_dim=512,
        rope_dim=64,
        value_dim=512,
        softmax_scale=0.0625,
        q_dtype=DType.BF16,
        cache_dtype=DType.BF16,
        index_dtype="int32",
        output_dtype=DType.BF16,
        valid_counts="u:2048x1",
        index_distribution="recent_contiguous",
        cache_layout="token_major_mqa_bf16_latent_rope",
    )


def test_registration_support_family_environment_and_facades() -> None:
    spec = find_kernel_profiler_spec(KIND, "torch")

    assert KIND == "dsa_sparse_mla_attention"
    assert known_backends(KIND) == ["torch"]
    assert spec.kernel_kind == spec.table_name == KIND
    assert spec.args_schema is DsaSparseMlaAttentionArgs
    assert spec.metric_family is MetricFamily.COMPUTE
    assert spec.batch_outlier_policy == BatchOutlierPolicy()
    assert spec.subprocess_env is None
    assert spec.supports.allows(DType.BF16, DType.BF16, gpu="NVIDIA H200")
    assert not spec.supports.allows(DType.FP32, DType.BF16, gpu="NVIDIA H200")
    assert not spec.supports.allows(DType.BF16, DType.FP32, gpu="NVIDIA H200")
    assert not spec.supports.allows(DType.BF16, DType.BF16, gpu="NVIDIA H100")
    assert spec.runner_ref.module_name == ("profiling.runners.attention.dsa_sparse_mla_attention")
    assert spec.runner_ref.function_name == "profile_dsa_sparse_mla_attention_torch"
    assert hasattr(perf_api, "get_dsa_sparse_mla_attention_times")
    assert hasattr(perf_api, "count_missing_dsa_sparse_mla_attention")


def test_registry_and_runner_ref_imports_are_lazy() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import profiling.kernels; "
                "from profiling.db.registry import find_kernel_profiler_spec; "
                "runner = find_kernel_profiler_spec("
                "'dsa_sparse_mla_attention', 'torch').runner_ref.load(); "
                "print(runner.__module__); print(runner.__name__); "
                "print('torch' in sys.modules); print('vllm' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == [
        "profiling.runners.attention.dsa_sparse_mla_attention",
        "profile_dsa_sparse_mla_attention_torch",
        "False",
        "False",
    ]


@pytest.mark.parametrize(
    ("encoded", "num_queries", "num_cache_tokens", "expected"),
    [
        ("u:2048x1", 1, 2048, (2048,)),
        ("g:(2047,2048)x16", 32, 2048, (2047, 2048) * 16),
        (
            "c:1922..2049@2048",
            128,
            2049,
            tuple(min(value, 2048) for value in range(1922, 2050)),
        ),
        ("r:1..128", 128, 2048, tuple(range(1, 129))),
        ("u:0x3", 3, 2048, (0, 0, 0)),
        ("g:(0,2,1)x1", 3, 2048, (0, 2, 1)),
    ],
)
def test_valid_counts_accepted_canonical_examples(
    encoded, num_queries, num_cache_tokens, expected
) -> None:
    from profiling.runners.attention.dsa_sparse_mla_attention import (
        _decode_valid_counts,
        _encode_valid_counts,
    )

    decoded = _decode_valid_counts(
        encoded,
        num_queries=num_queries,
        selected_k=2048,
        num_cache_tokens=num_cache_tokens,
    )
    assert decoded == expected
    assert (
        _encode_valid_counts(decoded, selected_k=2048, num_cache_tokens=num_cache_tokens) == encoded
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
        "c:1..3@",
        "g:()x1",
        "g:(1,)x1",
        "g:(1;2)x1",
        "g:(１)x1",
    ],
)
def test_valid_counts_rejects_malformed_compact_syntax(encoded) -> None:
    from profiling.runners.attention.dsa_sparse_mla_attention import (
        _decode_valid_counts,
    )

    with pytest.raises(ValueError, match="valid_counts"):
        _decode_valid_counts(encoded, num_queries=1, selected_k=2048, num_cache_tokens=2048)


@pytest.mark.parametrize(
    ("encoded", "num_queries", "num_cache_tokens", "match"),
    [
        ("u:1x0", 1, 2048, "positive"),
        ("g:(1)x0", 1, 2048, "positive"),
        ("u:1x2", 1, 2048, "expands to 2"),
        ("u:1x999999999999", 1, 2048, "expands to 999999999999"),
        ("r:0..999999999999", 1, 2048, "expands to 1000000000000"),
        ("g:(1,2)x999999999999", 1, 2048, "expands to 1999999999998"),
        ("u:2049x1", 1, 2048, "0..2048"),
        ("r:2..2", 1, 2048, "strict"),
        ("r:3..2", 1, 2048, "strict"),
        ("c:2048..2049@2048", 2, 2049, "first <"),
        ("c:1..2049@1024", 2049, 2049, "cap must equal"),
        ("c:1..2050@2048", 2050, 2049, "last <="),
    ],
)
def test_valid_counts_rejects_rows_ranges_and_clipping(
    encoded, num_queries, num_cache_tokens, match
) -> None:
    from profiling.runners.attention.dsa_sparse_mla_attention import (
        _decode_valid_counts,
    )

    with pytest.raises(ValueError, match=match):
        _decode_valid_counts(
            encoded,
            num_queries=num_queries,
            selected_k=2048,
            num_cache_tokens=num_cache_tokens,
        )


@pytest.mark.parametrize(
    ("encoded", "num_queries", "canonical"),
    [
        ("g:(7)x3", 3, "u:7x3"),
        ("g:(1,2,3)x1", 3, "r:1..3"),
        ("g:(1,2,1,2)x1", 4, "g:(1,2)x2"),
        ("g:(1,2)x1", 2, "r:1..2"),
    ],
)
def test_valid_counts_rejects_ambiguous_or_nonminimal_encodings(
    encoded, num_queries, canonical
) -> None:
    from profiling.runners.attention.dsa_sparse_mla_attention import (
        _decode_valid_counts,
    )

    with pytest.raises(ValueError, match=escape(canonical)):
        _decode_valid_counts(
            encoded,
            num_queries=num_queries,
            selected_k=2048,
            num_cache_tokens=4096,
        )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"num_queries": 0}, "1..4096"),
        ({"num_queries": 4097}, "1..4096"),
        ({"num_cache_tokens": 0}, "1..131072"),
        ({"num_cache_tokens": 131073}, "1..131072"),
        ({"num_heads": 63}, "num_heads must be 64"),
        ({"num_kv_heads": 2}, "num_kv_heads must be 1"),
        ({"selected_k": 1024}, "selected_k must be 2048"),
        ({"latent_dim": 256}, "latent_dim must be 512"),
        ({"rope_dim": 32}, "rope_dim must be 64"),
        ({"value_dim": 256}, "value_dim must be 512"),
        ({"softmax_scale": 0.1}, "softmax_scale must be exactly"),
        ({"q_dtype": "fp32"}, "q_dtype must be bf16"),
        ({"cache_dtype": "fp32"}, "cache_dtype must be bf16"),
        ({"index_dtype": "int64"}, "index_dtype must be int32"),
        ({"output_dtype": "fp32"}, "output_dtype must be bf16"),
        ({"index_distribution": "random"}, "index_distribution"),
        ({"cache_layout": "page_major"}, "cache_layout"),
    ],
)
def test_validation_rejects_unsupported_domain_before_allocation(
    monkeypatch, overrides, match
) -> None:
    from profiling.runners.attention import dsa_sparse_mla_attention as runner

    allocated = False

    def fail_if_allocated(*_args, **_kwargs):
        nonlocal allocated
        allocated = True
        raise AssertionError("unsupported inputs must fail before allocation")

    monkeypatch.setattr(runner, "_build_operands", fail_if_allocated)
    spec = dict(_BASE_SPEC)
    spec.update(overrides)
    with pytest.raises((TypeError, ValueError, ProfilerNotImplemented), match=match):
        runner.profile_dsa_sparse_mla_attention_torch(**spec)
    assert not allocated


def test_validation_rejects_boolean_integer_and_scale() -> None:
    from profiling.runners.attention.dsa_sparse_mla_attention import _validate_args

    with pytest.raises(TypeError, match="num_queries must be an integer"):
        _validate_args(**(_BASE_SPEC | {"num_queries": True}))
    with pytest.raises(TypeError, match="softmax_scale must be a real number"):
        _validate_args(**(_BASE_SPEC | {"softmax_scale": True}))


def test_cuda_and_gpu_support_failures_are_typed() -> None:
    from profiling.runners.attention.dsa_sparse_mla_attention import _require_h200

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


@pytest.mark.parametrize(
    "distribution",
    [
        "recent_contiguous",
        "unique_scattered_pages",
        "clustered_pages",
        "uniform_stride",
    ],
)
def test_index_distributions_are_deterministic_unique_and_in_range(distribution) -> None:
    from profiling.runners.attention.dsa_sparse_mla_attention import _row_indices

    first = _row_indices(
        torch,
        row=3,
        count=31,
        num_cache_tokens=67,
        distribution=distribution,
        device=torch.device("cpu"),
    )
    second = _row_indices(
        torch,
        row=3,
        count=31,
        num_cache_tokens=67,
        distribution=distribution,
        device=torch.device("cpu"),
    )
    assert torch.equal(first, second)
    assert first.dtype is torch.int64
    assert first.numel() == torch.unique(first).numel() == 31
    assert int(first.min()) >= 0
    assert int(first.max()) < 67
    if distribution == "recent_contiguous":
        assert first.tolist() == list(range(36, 67))


def test_operands_have_exact_layout_and_valid_then_sentinel_slots(monkeypatch) -> None:
    from profiling.runners.attention import dsa_sparse_mla_attention as runner

    monkeypatch.setattr(runner, "_NUM_HEADS", 3)
    monkeypatch.setattr(runner, "_SELECTED_K", 8)
    validated = runner._ValidatedArgs(
        num_queries=3,
        num_cache_tokens=17,
        valid_counts=(0, 3, 8),
        index_distribution="unique_scattered_pages",
    )
    operands = runner._build_operands(torch, validated, device=torch.device("cpu"))

    assert operands.q.shape == (3, 3, 576)
    assert operands.cache.shape == (17, 1, 576)
    assert operands.selected_indices.shape == (3, 1, 8)
    assert operands.q.dtype is operands.cache.dtype is torch.bfloat16
    assert operands.selected_indices.dtype is torch.int32
    assert operands.q.stride(-1) == operands.cache.stride(-1) == 1
    assert torch.isfinite(operands.q).all() and torch.isfinite(operands.cache).all()
    assert (operands.q < 0).any() and (operands.q > 0).any()
    assert operands.selected_indices[0, 0].tolist() == [-1] * 8
    for row, count in ((1, 3), (2, 8)):
        valid = operands.selected_indices[row, 0, :count]
        assert valid.numel() == torch.unique(valid).numel()
        assert ((valid >= 0) & (valid < 17)).all()
        assert operands.selected_indices[row, 0, count:].tolist() == [-1] * (8 - count)


def test_operand_sources_do_not_scale_an_arange_with_cache_size(monkeypatch) -> None:
    from profiling.runners.attention import dsa_sparse_mla_attention as runner

    monkeypatch.setattr(runner, "_NUM_HEADS", 2)
    monkeypatch.setattr(runner, "_SELECTED_K", 8)
    seen: list[int] = []
    original_arange = torch.arange

    def recording_arange(*args, **kwargs):
        stop = args[0] if len(args) == 1 else args[1]
        seen.append(int(stop))
        return original_arange(*args, **kwargs)

    monkeypatch.setattr(torch, "arange", recording_arange)
    validated = runner._ValidatedArgs(
        num_queries=2,
        num_cache_tokens=10000,
        valid_counts=(8, 8),
        index_distribution="uniform_stride",
    )
    runner._build_operands(torch, validated, device=torch.device("cpu"))
    assert max(seen) <= 64


def test_vectorized_composite_matches_reference_for_empty_short_and_full_rows(
    monkeypatch,
) -> None:
    from profiling.runners.attention import dsa_sparse_mla_attention as runner

    monkeypatch.setattr(runner, "_SELECTED_K", 5)
    q = torch.linspace(-0.8, 0.9, 3 * 2 * 576).reshape(3, 2, 576).to(torch.bfloat16)
    cache = torch.linspace(-1.0, 0.7, 7 * 576).reshape(7, 1, 576).to(torch.bfloat16)
    indices = torch.tensor(
        [[[-1, -2, 7, 8, -1]], [[0, 3, -1, 9, -4]], [[6, 1, 4, 0, 2]]],
        dtype=torch.int32,
    )
    snapshots = (q.clone(), cache.clone(), indices.clone())

    actual = runner._torch_composite(
        torch, q, cache, indices, softmax_scale=0.0625, query_chunk_size=2
    )
    expected = dsa_sparse_mla_attention_reference(q, cache, indices, softmax_scale=0.0625)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.count_nonzero(actual[0]) == 0
    assert actual.shape == (3, 2, 512)
    assert actual.dtype is torch.bfloat16
    assert torch.equal(q, snapshots[0])
    assert torch.equal(cache, snapshots[1])
    assert torch.equal(indices, snapshots[2])
    for source in (q, cache, indices):
        assert actual.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()


def test_composite_is_independent_of_query_chunking(monkeypatch) -> None:
    from profiling.runners.attention import dsa_sparse_mla_attention as runner

    monkeypatch.setattr(runner, "_SELECTED_K", 4)
    q = torch.linspace(-0.4, 0.6, 5 * 3 * 576).reshape(5, 3, 576).to(torch.bfloat16)
    cache = torch.linspace(-0.7, 0.5, 9 * 576).reshape(9, 1, 576).to(torch.bfloat16)
    indices = torch.tensor(
        [
            [[-1, -1, -1, -1]],
            [[0, -1, -1, -1]],
            [[2, 4, -1, -1]],
            [[7, 1, 5, -1]],
            [[8, 3, 6, 0]],
        ],
        dtype=torch.int32,
    )
    chunked = runner._torch_composite(
        torch, q, cache, indices, softmax_scale=0.0625, query_chunk_size=2
    )
    rowwise = runner._torch_composite(
        torch, q, cache, indices, softmax_scale=0.0625, query_chunk_size=1
    )
    assert torch.equal(chunked, rowwise)


def test_logical_accounting_matches_contract() -> None:
    from profiling.runners.attention.dsa_sparse_mla_attention import (
        _logical_bytes,
        _logical_flops,
    )

    assert _logical_flops(num_queries=2, num_heads=64, selected_k=2048) == (
        2 * 2 * 64 * 2048 * (576 + 512)
    )
    assert _logical_bytes(
        num_queries=2,
        num_heads=64,
        selected_k=2048,
        valid_counts=(0, 2),
    ) == (2 * 2 * 64 * 576 + 4 * 2 * 2048 + 2 * 2 * 576 + 2 * 2 * 64 * 512 + 8 * 2 * 64)


def test_profile_times_complete_composite_and_returns_compute_metrics(monkeypatch) -> None:
    from profiling.runners.attention import dsa_sparse_mla_attention as runner

    operands = runner._Operands(q=object(), cache=object(), selected_indices=object())
    calls: list[str] = []
    monkeypatch.setattr(runner, "_require_h200", lambda _torch: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(runner, "_build_operands", lambda *_args, **_kwargs: operands)
    monkeypatch.setattr(
        runner,
        "_check_correctness",
        lambda *_args, **_kwargs: calls.append("correctness"),
    )

    def composite(*_args, **_kwargs):
        calls.append("composite")
        return None

    def fake_timer(callable_):
        calls.append("timer")
        callable_()
        return 2.0

    def fake_energy(callable_, *, warmup, per_iter_time_ms):
        calls.extend([f"energy:{warmup}:{per_iter_time_ms}", "energy_call"])
        callable_()
        return 0.25

    monkeypatch.setattr(runner, "_torch_composite", composite)
    monkeypatch.setattr(runner.Timer, "cupti", fake_timer)
    monkeypatch.setattr(runner.Energy, "perf", fake_energy)
    metrics = runner.profile_dsa_sparse_mla_attention_torch(**_BASE_SPEC)

    assert metrics.time_ms == 2.0
    assert metrics.energy_j == 0.25
    assert metrics.tflops > 0
    assert metrics.memory_bandwidth_gbps > 0
    assert calls == [
        "correctness",
        "timer",
        "composite",
        "energy:5:2.0",
        "energy_call",
        "composite",
    ]


def test_profile_translates_runtime_failures(monkeypatch) -> None:
    from profiling.runners.attention import dsa_sparse_mla_attention as runner

    monkeypatch.setattr(runner, "_require_h200", lambda _torch: None)

    def fail(*_args, **_kwargs):
        raise RuntimeError("allocation failed")

    monkeypatch.setattr(runner, "_build_operands", fail)
    with pytest.raises(KernelLaunchFailed, match="composite failed") as exc_info:
        runner.profile_dsa_sparse_mla_attention_torch(**_BASE_SPEC)
    assert isinstance(exc_info.value.__cause__, RuntimeError)
