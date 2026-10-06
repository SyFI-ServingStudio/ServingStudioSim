import pytest
import torch

from profiling.runners.embedding import engram_lookup as runner
from profiling.runners.exceptions import ProfilerNotImplemented

_ARGS = dict(
    num_tokens=48,
    local_heads=6,
    head_dim=256,
    quant_block_size=32,
    table_rows=96_000_564,
    residency="host_uva",
    weight_dtype="fp8_e4m3",
)


def test_reference_applies_one_ue8m0_scale_per_32_values():
    # Defect caught: applying the scale per row instead of per 32-value block,
    # or reading the exponent byte with the wrong bias (2**(e - 127)).
    weight = torch.full((2, 256), 0x38, dtype=torch.uint8)  # e4m3 0x38 == 1.0
    scales = torch.full((2, 8), 127, dtype=torch.uint8)
    scales[1, 3] = 130  # 2**3 on values 96..127 of row 1 only
    ids = torch.tensor([[1, 0]])
    out = runner.reference_rows(torch, weight, scales, ids, 32).float()
    assert out.shape == (1, 2, 256)
    assert torch.all(out[0, 1] == 1.0)
    assert torch.all(out[0, 0, 96:128] == 8.0)
    assert torch.all(out[0, 0, :96] == 1.0) and torch.all(out[0, 0, 128:] == 1.0)


def test_reference_zeroes_ids_outside_the_rank_slice():
    # Defect caught: gathering row 0 (or wrapping) for an id this rank does not
    # own, where the kernel writes zeros for the later head all-gather.
    weight = torch.full((4, 256), 0x38, dtype=torch.uint8)
    scales = torch.full((4, 8), 127, dtype=torch.uint8)
    out = runner.reference_rows(torch, weight, scales, torch.tensor([[-1, 4, 2]]), 32)
    assert torch.all(out[0, :2] == 0)
    assert torch.all(out[0, 2].float() == 1.0)


def test_id_pool_stays_inside_each_heads_bucket_range():
    # Defect caught: drawing ids for head h outside its own bucket range, which
    # the kernel would mask as unowned and skip the host read being timed.
    rows, heads = 1000, 6
    pool = runner.build_id_pool(torch, torch.device("cpu"), 64, heads, rows)
    for head, (lo, hi) in enumerate(runner.head_ranges(rows, heads)):
        column = pool[..., head]
        assert int(column.min()) >= lo and int(column.max()) < hi
    assert runner.head_ranges(rows, heads)[-1][1] == rows


def test_args_reject_shapes_the_kernel_is_not_profiled_for():
    runner._validate_args(**_ARGS)
    runner._validate_args(**dict(_ARGS, residency="device"))
    with pytest.raises(ValueError, match="residency"):
        runner._validate_args(**dict(_ARGS, residency="host"))
    with pytest.raises(ValueError, match="table_rows"):
        runner._validate_args(**dict(_ARGS, table_rows=5))
    with pytest.raises(ProfilerNotImplemented, match="head_dim"):
        runner._validate_args(**dict(_ARGS, head_dim=128))
    with pytest.raises(ProfilerNotImplemented, match="fp8_e4m3"):
        runner._validate_args(**dict(_ARGS, weight_dtype="bf16"))


def test_row_bytes_count_the_scale_bytes():
    # Defect caught: metric bandwidth that ignores the separate 8-byte scale row.
    assert runner.row_bytes(256, 32) == 264
    assert runner.logical_bytes(2, 6, 256, 32) == 12 * (4 + 264 + 512)
