import pytest

from profiling.runners.attention.deepseek_v4_sparse_attn_compress_store_cutedsl import (
    _logical_work,
    _padded_stride,
    _validate_args,
)
from profiling.runners.attention.deepseek_v4_sparse_attn_compress_store_triton import (
    _validate_args as validate_indexer_args,
)


def _shape(*, positions=(3, 4, 7), requests=(0, 0, 0), ratio=4, width=2):
    return _validate_args(
        positions,
        requests,
        width,
        ratio,
        1,
        512,
        64,
        256,
        1.0e-6,
        "fp32",
        "bf16",
        "fp8_ds_mla",
        "block_segregated_data_then_scales",
        "ue8m0",
    )


def test_active_rows_follow_real_compression_boundaries():
    c4 = _shape()
    assert c4.active_rows == (0, 2)
    assert c4.window == 8

    c128 = _shape(positions=(126, 127, 255), requests=(0, 0, 0), ratio=128, width=32)
    assert c128.active_rows == (1, 2)
    assert c128.window == 128


def test_partial_c4_window_is_a_supported_physical_shape():
    partial = _shape(positions=(3,), requests=(0,), ratio=4, width=1)
    assert partial.active_rows == (0,)
    flops, logical_bytes = _logical_work(partial)
    assert flops > 0
    assert logical_bytes > 584


def test_topology_and_table_capacity_fail_closed():
    with pytest.raises(ValueError, match="densely cover"):
        _shape(positions=(3, 7), requests=(0, 2), width=2)
    with pytest.raises(ValueError, match="does not cover"):
        _shape(positions=(127,), requests=(0,), ratio=128, width=15)


def test_production_packed_page_stride_includes_alignment_padding():
    assert _padded_stride(64) == 37440
    assert _padded_stride(2) == 1728


def test_indexer_backend_uses_its_real_132_byte_cache_identity():
    shape = validate_indexer_args(
        (3, 7),
        (0, 0),
        2,
        4,
        1,
        128,
        64,
        256,
        1.0e-6,
        "fp32",
        "bf16",
        "fp8_indexer",
        "block_segregated_data_then_scales",
        "fp32_per_token",
    )
    assert (
        shape.cache_row_bytes,
        shape.token_stride,
        shape.scale_dim,
        shape.page_alignment,
    ) == (132, 128, 4, 576)
    assert ((shape.kv_block_size * shape.cache_row_bytes + 575) // 576) * 576 == 8640
