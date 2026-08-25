import pytest

from profiling.db.args import DType
from profiling.runners.attention.deepseek_v4_packed_cache_gather_cutedsl import (
    _block_stride,
    _expected_rows,
    _logical_work,
    _validate_args,
)
from profiling.runners.exceptions import ProfilerNotImplemented

_IDENTITY = (
    1,
    512,
    448,
    64,
    "fp8_ds_mla",
    DType.BF16,
    "block_segregated_data_then_scales",
    "ue8m0",
)


def test_c128_page_stride_preserves_cutedsl_alignment():
    assert _block_stride(2) == 1184
    assert _block_stride(64) == 37_376


def test_ragged_suffix_uses_physical_block_translation():
    shape = _validate_args((5, 3), (2, 3), 8, 3, 2, 1, *_IDENTITY)
    table = ((4, 1, 3), (2, 0, -1))
    assert _expected_rows(shape, table) == ((7.0, 5.0), (1.0, 3.0, 1.0))


def test_empty_gather_tuple_means_full_sequence():
    shape = _validate_args((3,), (), 3, 2, 2, 0, *_IDENTITY)
    assert shape.gather_lens is None
    assert _expected_rows(shape, ((1, 0),)) == ((5.0, 7.0, 1.0),)


def test_logical_work_uses_only_gathered_rows():
    shape = _validate_args((129, 65), (64, 33), 256, 3, 64, 128, *_IDENTITY)
    assert _logical_work(shape) == (43_456, 156_380)


@pytest.mark.parametrize(
    "args",
    [
        ((), (), 1, 1, 64, 0, *_IDENTITY),
        ((3,), (4,), 4, 1, 64, 0, *_IDENTITY),
        ((65,), (), 65, 1, 64, 0, *_IDENTITY),
        ((3,), (), 2, 1, 64, 0, *_IDENTITY),
        ((3,), (), 3, 1, 4, 0, *_IDENTITY),
    ],
)
def test_invalid_physical_shapes_are_rejected(args):
    with pytest.raises((ValueError, ProfilerNotImplemented)):
        _validate_args(*args)


def test_other_model_identity_is_not_silently_reused():
    with pytest.raises(ProfilerNotImplemented, match="supports model identity"):
        _validate_args((1,), (), 1, 1, 64, 0, 2, *_IDENTITY[1:])
