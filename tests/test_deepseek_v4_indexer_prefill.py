from types import SimpleNamespace

import pytest

from profiling.db.batch import coerce_args
from profiling.kernels.deepseek_v4_indexer_mqa_logits_prefill import (
    DeepseekV4IndexerMqaLogitsPrefillArgs,
)
from profiling.runners.attention.deepseek_v4_indexer_mqa_logits_prefill_deepgemm import (
    _launch as launch_mqa,
)
from profiling.runners.attention.deepseek_v4_indexer_prefill_workload import (
    build_indexer_prefill_chunks,
)
from profiling.runners.attention.deepseek_v4_indexer_topk_prefill_cuda import (
    _launch as launch_topk,
)
from profiling.runners.attention.deepseek_v4_indexer_topk_prefill_cuda import (
    _required_logits_row_stride,
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
        DeepseekV4IndexerMqaLogitsPrefillArgs,
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
        logits=Logits(), row_starts="starts", row_ends="ends", output="output"
    )
    launch_topk(lambda *args: calls.append(args), topk_operands)
    assert calls == [(topk_operands.logits, "starts", "ends", "output", 3, 512, 1, 512)]
