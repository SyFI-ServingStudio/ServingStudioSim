import pytest

from profiling.runners.attention.deepseek_v4_sparse_mla_prefill_flashmla import (
    _chunks,
    _validate_args,
)


def _shape(
    *,
    pairs=((1, 1),),
    max_model_len=65536,
    max_num_batched_tokens=8192,
    ratio=4,
    selected_k=512,
):
    return _validate_args(
        pairs,
        max_model_len,
        max_num_batched_tokens,
        4,
        ratio,
        128,
        selected_k,
        "request_local_topk_plus_swa",
        64,
        1,
        512,
        512,
        512**-0.5,
        "bf16",
        "bf16",
        "int32",
        "bf16",
        "request_slot_major_flat_mqa_bf16_d512",
    )


def test_runtime_context_controls_physical_compressed_workspace():
    runtime_65k = _shape(max_model_len=65536, ratio=128)
    checkpoint_1m = _shape(max_model_len=1048576, ratio=128)
    assert runtime_65k.compressed_capacity == 512
    assert checkpoint_1m.compressed_capacity == 8192
    assert runtime_65k.request_slot_size < checkpoint_1m.request_slot_size


def test_one_semantic_prefill_operation_splits_requests_four_at_a_time():
    shape = _shape(pairs=((1, 8),) * 9)
    chunks = _chunks(shape)
    assert tuple(len(chunk.pairs) for chunk in chunks) == (4, 4, 1)
    assert sum(chunk.num_queries for chunk in chunks) == shape.num_queries


def test_swa_only_and_compressed_layers_have_distinct_physical_widths():
    swa = _shape(ratio=1, selected_k=0)
    c4 = _shape(ratio=4, selected_k=512)
    assert swa.padded_topk == 128
    assert c4.padded_topk == 640


def test_request_shape_and_runtime_limits_fail_closed():
    with pytest.raises(ValueError, match="query <= context"):
        _shape(pairs=((9, 8),))
    with pytest.raises(ValueError, match="exceeds max_model_len"):
        _shape(pairs=((1, 65537),), max_model_len=65536)
