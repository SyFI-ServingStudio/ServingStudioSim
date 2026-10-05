import pytest

from profiling.db.args import DType
from profiling.runners.attention.compressed_sparse_mla_decode_flashmla import (
    _logical_work,
    _selected_means,
    _validate_args,
)
from profiling.runners.exceptions import ProfilerNotImplemented

_MODEL_ARGS = (64, 1, 512, 512, 128)
_DTYPE_ARGS = (DType.BF16, DType.FP8_E4M3, DType.BF16)


def test_selected_means_preserve_ragged_row_boundaries():
    assert _selected_means((2, 1), (1.0, 3.0, 5.0), (1, 2), (9.0, 11.0, 13.0)) == (
        13.0 / 3.0,
        29.0 / 3.0,
    )


@pytest.mark.parametrize(
    "compress_ratio,extra_capacity,extra_counts",
    [(1, 0, (0, 0)), (4, 512, (512, 1)), (128, 8192, (8192, 17))],
)
def test_supported_ratios_keep_exact_ragged_work(compress_ratio, extra_capacity, extra_counts):
    shape = _validate_args(
        (128, 1),
        extra_counts,
        *_MODEL_ARGS,
        extra_capacity,
        compress_ratio,
        *_DTYPE_ARGS,
        "reused",
    )
    assert shape.swa_counts == (128, 1)
    assert shape.extra_counts == extra_counts


def test_logical_work_counts_only_valid_pairs():
    shape = _validate_args((128, 1), (512, 1), *_MODEL_ARGS, 512, 4, *_DTYPE_ARGS, "planned")
    flops, logical_bytes = _logical_work(shape)
    assert flops == 84_148_224
    assert logical_bytes == 640_424


@pytest.mark.parametrize(
    "args",
    [
        ((), (), *_MODEL_ARGS, 0, 1, *_DTYPE_ARGS, "reused"),
        ((1, 1), (0,), *_MODEL_ARGS, 0, 1, *_DTYPE_ARGS, "reused"),
        ((1,), (0,), 64, 1, 512, 512, 96, 0, 1, *_DTYPE_ARGS, "reused"),
        ((1,), (0,), *_MODEL_ARGS, 96, 4, *_DTYPE_ARGS, "reused"),
        ((129,), (0,), *_MODEL_ARGS, 0, 1, *_DTYPE_ARGS, "reused"),
        ((1,), (1,), *_MODEL_ARGS, 0, 1, *_DTYPE_ARGS, "reused"),
        ((1,), (513,), *_MODEL_ARGS, 512, 4, *_DTYPE_ARGS, "reused"),
        ((1,), (129,), *_MODEL_ARGS, 129, 128, *_DTYPE_ARGS, "reused"),
        ((1,), (0,), *_MODEL_ARGS, 512, 4, *_DTYPE_ARGS, "unknown"),
    ],
)
def test_invalid_aggregate_work_is_rejected(args):
    with pytest.raises((ValueError, TypeError, ProfilerNotImplemented)):
        _validate_args(*args)


def test_flashmla_head_counts_and_cache_shape_are_the_kernel_bounds():
    with pytest.raises(ProfilerNotImplemented, match="num_heads in"):
        _validate_args((1,), (0,), 32, 1, 512, 512, 128, 0, 1, *_DTYPE_ARGS, "reused")
    with pytest.raises(ProfilerNotImplemented, match="num_kv_heads, head_dim, value_dim"):
        _validate_args((1,), (0,), 64, 1, 576, 512, 128, 0, 1, *_DTYPE_ARGS, "reused")


@pytest.mark.parametrize(
    ("model_args", "extra_capacity", "compress_ratio", "num_rows"),
    [
        ((128, 1, 512, 512, 128), 512, 4, 2),
        ((64, 1, 512, 512, 256), 1024, 4, 2),
        ((64, 1, 512, 512, 128), 16384, 128, 2),
        ((64, 1, 512, 512, 128), 0, 1, 300),
    ],
)
def test_unmeasured_but_launchable_shapes_are_accepted(
    model_args, extra_capacity, compress_ratio, num_rows
):
    shape = _validate_args(
        (1,) * num_rows,
        (1 if compress_ratio > 1 else 0,) * num_rows,
        *model_args,
        extra_capacity,
        compress_ratio,
        *_DTYPE_ARGS,
        "reused",
    )
    assert shape.num_heads == model_args[0]
    assert len(shape.swa_counts) == num_rows
