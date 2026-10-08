import re
from pathlib import Path

import pytest

from profiling.kernels import kv_compress_store
from profiling.runners.attention.kv_compress_store_cutedsl import (
    _logical_work,
    _padded_stride,
    _validate_args,
)
from profiling.runners.attention.kv_compress_store_triton import (
    _validate_args as validate_indexer_args,
)
from profiling.runners.exceptions import ProfilerNotImplemented


def _shape(
    *,
    positions=(3, 4, 7),
    requests=(0, 0, 0),
    ratio=4,
    width=2,
    head_dim=512,
    block=256,
    eps=1.0e-6,
):
    return _validate_args(
        positions,
        requests,
        width,
        ratio,
        1,
        head_dim,
        64,
        block,
        eps,
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


def test_row_request_block_and_eps_limits_are_only_the_real_bounds():
    many = _shape(positions=tuple(range(9000)), requests=(0,) * 9000, width=2250)
    assert len(many.positions) == 9000
    requests = _shape(positions=(3,) * 100, requests=tuple(range(100)), width=1)
    assert max(requests.request_ids) == 99
    assert _shape(block=512).kv_block_size == 128
    assert _shape(eps=1.0e-5).rms_eps == 1.0e-5
    with pytest.raises(ValueError, match="logical_block_size"):
        _shape(block=258)
    with pytest.raises(ValueError, match="rms_eps"):
        _shape(eps=0.0)
    with pytest.raises(ProfilerNotImplemented, match="head_dim"):
        _shape(head_dim=576)


def test_indexer_block_size_and_eps_are_runtime_values():
    shape = validate_indexer_args(
        (3, 7),
        (0, 0),
        2,
        4,
        1,
        128,
        64,
        128,
        1.0e-5,
        "fp32",
        "bf16",
        "fp8_indexer",
        "block_segregated_data_then_scales",
        "fp32_per_token",
    )
    assert (shape.kv_block_size, shape.rms_eps) == (32, 1.0e-5)


def test_cache_row_bytes_match_the_simulators() -> None:
    rust = Path(__file__).resolve().parents[1] / "simulator/src/timing/kernels/kv_compress_store.rs"
    text = rust.read_text()
    for name in ("FP8_DS_MLA_ROW_BYTES", "FP8_INDEXER_ROW_BYTES"):
        (value,) = re.findall(rf"const {name}: u32 = (\d+);", text)
        assert getattr(kv_compress_store, name) == int(value), name
