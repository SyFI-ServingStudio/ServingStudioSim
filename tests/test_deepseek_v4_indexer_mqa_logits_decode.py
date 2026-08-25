import pytest
import torch

from profiling.runners.attention.deepseek_v4_indexer_mqa_logits_decode_deepgemm import (
    _allocate_aligned_cache,
    _build_operands,
    _check_output,
    _expected_logits_row,
    _max_ragged_context_lengths,
    _validate_args,
)
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented

PRODUCTION_SPEC = {
    "batch_size": 3,
    "context_len": 65,
    "next_n": 1,
    "max_model_len": 256,
    "num_heads": 64,
    "head_dim": 128,
    "block_size": 64,
    "q_dtype": "fp8_e4m3",
    "cache_dtype": "fp8_e4m3",
    "scale_dtype": "fp32",
    "weight_dtype": "fp32",
    "output_dtype": "fp32",
    "context_mode": "max_ragged",
    "page_mapping": "request_contiguous",
    "cache_format": "fp8_e4m3_ue8m0",
    "clean_logits": False,
}


def test_runner_accepts_only_the_canonical_cache_format_tag():
    assert _validate_args(**PRODUCTION_SPEC) == (65, 64, 63)
    with pytest.raises(ProfilerNotImplemented, match="storage identity"):
        _validate_args(**{**PRODUCTION_SPEC, "cache_format": "row_interleaved"})


def test_physical_cache_is_page_planar_with_576_aligned_pages():
    cache, key_view, scale_view, page_stride = _allocate_aligned_cache(
        torch, num_blocks=3, block_size=64, head_dim=128, device="cpu"
    )
    assert cache.shape == (3, 64, 1, 132)
    assert cache.stride() == (8640, 132, 132, 1)
    assert page_stride == 8640
    assert page_stride % 576 == 0
    assert key_view.shape == (3, 64, 1, 128)
    assert scale_view.shape == (3, 64, 1)
    assert key_view.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert scale_view.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert key_view.storage_offset() == 0
    assert scale_view.storage_offset() * scale_view.element_size() == 64 * 128


def test_max_ragged_topology_is_request_contiguous_and_correctness_is_independent():
    context_lengths = _max_ragged_context_lengths(3, 65)
    assert context_lengths == (65, 64, 63)
    operands = _build_operands(
        torch,
        context_lengths=context_lengths,
        next_n=1,
        num_heads=64,
        head_dim=128,
        block_size=64,
        device="cpu",
    )
    assert operands.context_lens.tolist() == [[65], [64], [63]]
    assert operands.block_table.tolist() == [[0, 1], [2, -1], [3, -1]]

    max_model_len = 96
    actual = torch.empty((3, max_model_len), dtype=torch.float32)
    for request_index in range(3):
        expected = _expected_logits_row(torch, operands, request_index)
        actual[request_index, : expected.numel()] = expected
    _check_output(torch, actual, operands, max_model_len)

    actual[1, 32] += 1.0
    with pytest.raises((AssertionError, KernelLaunchFailed)):
        _check_output(torch, actual, operands, max_model_len)
