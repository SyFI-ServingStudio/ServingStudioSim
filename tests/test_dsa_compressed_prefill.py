from types import SimpleNamespace

import pytest

from profiling.db.batch import coerce_args
from profiling.kernels.dsa_compressed_mqa_logits_prefill import (
    DsaCompressedMqaLogitsPrefillArgs,
)
from profiling.runners.attention.dsa_compressed_mqa_logits_prefill_deepgemm import (
    _launch as launch_mqa,
)
from profiling.runners.attention.dsa_compressed_mqa_logits_prefill_deepgemm import (
    _validate_args as validate_mqa,
)
from profiling.runners.attention.dsa_compressed_prefill_workload import (
    build_indexer_prefill_chunks,
)
from profiling.runners.attention.dsa_compressed_topk_prefill_cuda import (
    _launch as launch_topk,
)
from profiling.runners.attention.dsa_compressed_topk_prefill_cuda import (
    _required_logits_row_stride,
)
from profiling.runners.attention.dsa_compressed_topk_prefill_cuda import (
    _validate_args as validate_topk,
)
from profiling.runners.exceptions import ProfilerNotImplemented

MIB_512 = 512 * 1024 * 1024


def test_ragged_requests_produce_source_exact_causal_spans():
    chunks = build_indexer_prefill_chunks(((2, 8), (3, 13)), 65536, 8192, MIB_512, 4)
    assert len(chunks) == 1
    assert chunks[0].row_starts == (0, 0, 2, 2, 2)
    assert chunks[0].row_ends == (1, 2, 4, 5, 5)
    assert chunks[0].valid_key_pairs == 11


def test_one_large_request_is_query_sliced_without_creating_semantic_slots():
    chunks = build_indexer_prefill_chunks(((8192, 1_048_576),), 1_048_576, 8192, MIB_512, 4)
    assert len(chunks) == 16
    assert [(chunk.query_slice_start, chunk.query_slice_stop) for chunk in chunks] == [
        (start, start + 512) for start in range(0, 8192, 512)
    ]
    assert sum(chunk.num_queries for chunk in chunks) == 8192


def test_request_greedy_split_respects_physical_workspace_and_zero_key_boundary():
    chunks = build_indexer_prefill_chunks(((1, 65536),) * 64, 65536, 8192, MIB_512, 4)
    assert tuple(len(chunk.pairs) for chunk in chunks) == (40, 24)
    with pytest.raises(ProfilerNotImplemented, match="no compressed"):
        build_indexer_prefill_chunks(((1, 3),), 65536, 8192, MIB_512, 4)


def test_nested_pairs_are_canonicalized_by_the_public_args_boundary():
    coerced = coerce_args(
        DsaCompressedMqaLogitsPrefillArgs,
        {
            "query_context_pairs": [[2, 8], [3, 13]],
            "max_model_len": 65536,
            "max_num_batched_tokens": 8192,
            "max_logits_bytes": MIB_512,
            "compress_ratio": 4,
            "num_heads": 64,
            "head_dim": 128,
            "q_dtype": "fp8_e4m3",
            "k_dtype": "fp8_e4m3",
            "k_scale_dtype": "fp32",
            "weight_dtype": "fp32",
            "output_dtype": "fp32",
            "clean_logits": False,
        },
    )
    assert coerced.query_context_pairs == ((2, 8), (3, 13))


def test_topk_stride_matches_deepgemm_padded_output_layout():
    assert _required_logits_row_stride(1) == 512
    assert _required_logits_row_stride(1024) == 1280
    assert _required_logits_row_stride(2047) == 2304


def test_public_launches_preserve_production_argument_order():
    calls = []
    mqa_operands = SimpleNamespace(
        q="q",
        k="k",
        k_scale="scale",
        weights="weights",
        row_starts="starts",
        row_ends="ends",
    )
    launch_mqa(lambda *args, **kwargs: calls.append((args, kwargs)), mqa_operands)
    assert calls == [
        (
            (("q", None), ("k", "scale"), "weights", "starts", "ends"),
            {"clean_logits": False},
        )
    ]

    class Logits:
        shape = (3, 5)

        @staticmethod
        def stride(dimension):
            return (512, 1)[dimension]

    calls.clear()
    topk_operands = SimpleNamespace(
        logits=Logits(), row_starts="starts", row_ends="ends", output="output", top_k=512
    )
    launch_topk(lambda *args: calls.append(args), topk_operands)
    assert calls == [(topk_operands.logits, "starts", "ends", "output", 3, 512, 1, 512)]


def test_workload_caps_are_only_the_real_split_bounds():
    # More than 64 requests, a 2M context, a 64K batch and a non-default logits
    # budget all split the same way vLLM's splitter does.
    chunks = build_indexer_prefill_chunks(((1, 64),) * 100, 65536, 8192, MIB_512, 4)
    assert sum(len(chunk.pairs) for chunk in chunks) == 100
    chunks = build_indexer_prefill_chunks(((65536, 2_097_152),), 2_097_152, 65536, MIB_512, 4)
    assert sum(chunk.num_queries for chunk in chunks) == 65536
    chunks = build_indexer_prefill_chunks(((2, 8), (3, 13)), 65536, 8192, 256 * 1024 * 1024, 4)
    assert chunks[0].row_ends == (1, 2, 4, 5, 5)
    with pytest.raises(ValueError, match="at least one request"):
        build_indexer_prefill_chunks((), 65536, 8192, MIB_512, 4)
    with pytest.raises(ValueError, match="cover all queries"):
        build_indexer_prefill_chunks(((9, 16),), 65536, 8, MIB_512, 4)
    with pytest.raises(ValueError, match="max_logits_bytes"):
        build_indexer_prefill_chunks(((2, 8),), 65536, 8192, 0, 4)
    with pytest.raises(ProfilerNotImplemented, match="C4"):
        build_indexer_prefill_chunks(((2, 8),), 65536, 8192, MIB_512, 128)


def _mqa_args(num_heads, head_dim):
    return (
        ((2, 8), (3, 13)),
        65536,
        8192,
        MIB_512,
        4,
        num_heads,
        head_dim,
        "fp8_e4m3",
        "fp8_e4m3",
        "fp32",
        "fp32",
        "fp32",
        False,
    )


@pytest.mark.parametrize(
    ("num_heads", "head_dim"), [(64, 128), (32, 128), (16, 128), (8, 64), (64, 32)]
)
def test_mqa_logits_accepts_every_deepgemm_head_shape(num_heads, head_dim):
    assert validate_mqa(*_mqa_args(num_heads, head_dim))


@pytest.mark.parametrize(("num_heads", "head_dim"), [(128, 128), (48, 128), (64, 256)])
def test_mqa_logits_rejects_shapes_deepgemm_asserts_against(num_heads, head_dim):
    with pytest.raises(ProfilerNotImplemented, match="num_heads in"):
        validate_mqa(*_mqa_args(num_heads, head_dim))


@pytest.mark.parametrize("top_k", [512, 1024, 2048, 64])
def test_compressed_topk_accepts_any_positive_width(top_k):
    pairs = ((2, 8), (3, 13))
    assert validate_topk(pairs, 65536, 8192, MIB_512, 4, top_k, "fp32", "int32")


def test_compressed_topk_rejects_a_non_positive_width():
    with pytest.raises(ValueError, match="top_k"):
        validate_topk(((2, 8),), 65536, 8192, MIB_512, 4, 0, "fp32", "int32")
