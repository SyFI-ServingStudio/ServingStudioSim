"""Contracts for GLM-5.2's B200 varlen sparse-MLA prefill L1."""

from dataclasses import fields

import pytest
import torch

from profiling.db.args import DType
from profiling.db.batch import coerce_args
from profiling.db.registry import find_kernel_profiler_spec, known_backends
from profiling.kernels.dsa_sparse_mla_prefill import (
    KIND,
    DsaSparseMlaPrefillArgs,
)
from profiling.runners.exceptions import ProfilerNotImplemented

_BASE_SPEC = {
    "query_context_pairs": ((2, 3), (1, 65)),
    "num_heads": 16,
    "num_kv_heads": 1,
    "selected_k": 2048,
    "latent_dim": 512,
    "rope_dim": 64,
    "value_dim": 512,
    "softmax_scale": 0.0625,
    "q_dtype": "fp8_e4m3",
    "cache_dtype": "fp8_e4m3",
    "index_dtype": "int32",
    "output_dtype": "bf16",
    "index_distribution": "recent_contiguous",
    "cache_layout": "hnd_paged_mqa_fp8_latent_rope",
}


def test_args_and_registration_preserve_varlen_request_boundaries():
    assert [field.name for field in fields(DsaSparseMlaPrefillArgs)] == list(_BASE_SPEC)
    args = coerce_args(DsaSparseMlaPrefillArgs, _BASE_SPEC)
    assert args.query_context_pairs == ((2, 3), (1, 65))
    assert args.q_dtype is args.cache_dtype is DType.FP8_E4M3
    assert args.output_dtype is DType.BF16

    assert known_backends(KIND) == ["flashinfer_trtllm_fp8"]
    spec = find_kernel_profiler_spec(KIND, "flashinfer_trtllm_fp8")
    assert spec.args_schema is DsaSparseMlaPrefillArgs
    assert spec.subprocess_env == "vllm_env"
    assert spec.supports.allows(
        DType.FP8_E4M3,
        kv_dtype=DType.FP8_E4M3,
        gpu="NVIDIA B200",
    )
    assert not spec.supports.allows(
        DType.FP8_E4M3,
        kv_dtype=DType.FP8_E4M3,
        gpu="NVIDIA H200",
    )
    assert spec.runner_ref.function_name == ("profile_dsa_sparse_mla_prefill_flashinfer_trtllm_fp8")


def test_validation_retains_boundaries_valid_counts_and_page_offsets():
    from profiling.runners.attention.dsa_sparse_mla_prefill import _validate_args

    shape = _validate_args(**_BASE_SPEC)
    assert shape.pairs == ((2, 3), (1, 65))
    assert shape.valid_counts == (2, 3, 65)
    assert shape.request_page_offsets == (0, 1)
    assert shape.num_pages == 3
    assert shape.num_queries == 3


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    [
        ({"query_context_pairs": ()}, ValueError, "nonempty tuple"),
        ({"query_context_pairs": ((0, 1),)}, ValueError, "0 < query <= context"),
        ({"query_context_pairs": ((2, 1),)}, ValueError, "0 < query <= context"),
        ({"query_context_pairs": ([1, 1],)}, TypeError, "integer.*pairs"),
        ({"num_heads": 64}, ProfilerNotImplemented, "model identity"),
        ({"selected_k": 1024}, ProfilerNotImplemented, "model identity"),
        ({"q_dtype": "bf16"}, ProfilerNotImplemented, "storage identity"),
        ({"cache_layout": "token_major"}, ProfilerNotImplemented, "storage identity"),
        ({"index_distribution": "random"}, ProfilerNotImplemented, "index_distribution"),
    ],
)
def test_validation_fails_closed(overrides, error, match):
    from profiling.runners.attention.dsa_sparse_mla_prefill import _validate_args

    with pytest.raises(error, match=match):
        _validate_args(**(_BASE_SPEC | overrides))


def test_operand_builder_keeps_requests_in_disjoint_physical_page_ranges(
    monkeypatch,
):
    from profiling.runners.attention import dsa_sparse_mla_prefill as runner

    monkeypatch.setattr(runner, "_SELECTED_K", 4)
    monkeypatch.setattr(runner, "_TRTLLM_WORKSPACE_BYTES", 16)
    shape = runner._Shape(
        num_heads=16,
        pairs=((2, 3), (1, 65)),
        valid_counts=(2, 3, 4),
        request_page_offsets=(0, 1),
        num_pages=3,
        index_distribution="recent_contiguous",
    )
    operands = runner._build_operands(torch, shape, device=torch.device("cpu"))

    assert operands.query.shape == (3, 1, 16, 576)
    assert operands.query.dtype is torch.float8_e4m3fn
    assert operands.cache.shape == (3, 1, 64, 576)
    assert operands.cache.dtype is torch.float8_e4m3fn
    assert operands.seq_lens.tolist() == [2, 3, 4]
    assert operands.block_tables[0, 0].tolist() == [0, 1, 0, 0]
    assert operands.block_tables[1, 0].tolist() == [0, 1, 2, 0]
    assert operands.block_tables[2, 0].tolist() == [125, 126, 127, 128]
    assert int(operands.block_tables[:2].max()) < 64
    assert int(operands.block_tables[2].min()) >= 64
    assert operands.workspace.numel() == 16


def test_runner_ref_loads_without_importing_frameworks():
    runner = find_kernel_profiler_spec(KIND, "flashinfer_trtllm_fp8").runner_ref.load()
    assert runner.__name__ == ("profile_dsa_sparse_mla_prefill_flashinfer_trtllm_fp8")
