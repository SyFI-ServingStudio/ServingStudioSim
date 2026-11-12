"""Behavioral tests for the summed RMSNorm with a selectable output set."""

import pytest
import torch

from profiling.runners.norm.multi_add_rms_norm_reference import multi_add_rms_norm_reference
from profiling.runners.norm.multi_add_rms_norm_vllm_inductor import _validate_args


def test_reference_normalizes_the_sum_of_every_addend() -> None:
    # A reference that dropped the recomputed third addend would normalize a
    # different row and accept a kernel that skipped one input read.
    addends = [torch.full((1, 4), value, dtype=torch.bfloat16) for value in (1.0, 2.0, -6.0)]
    weight = torch.full((4,), 2.0, dtype=torch.bfloat16)

    total, normalized, fp8 = multi_add_rms_norm_reference(
        torch, addends, weight, torch.tensor(0.5), 0.0
    )

    torch.testing.assert_close(total.float(), torch.full((1, 4), -3.0))
    torch.testing.assert_close(normalized.float(), torch.full((1, 4), -2.0))
    torch.testing.assert_close(fp8.float(), torch.full((1, 4), -4.0))


@pytest.mark.parametrize(
    ("num_addends", "store_sum", "store_normed", "store_fp8"),
    [(1, True, True, False), (2, True, False, False)],
)
def test_runner_rejects_output_sets_vllm_never_compiles(
    num_addends: int, store_sum: bool, store_normed: bool, store_fp8: bool
) -> None:
    # A one-addend "sum" is the input itself, and a norm with no output plane is
    # dead code Inductor deletes; profiling either would store a fake row.
    with pytest.raises(ValueError):
        _validate_args(8, 5120, num_addends, store_sum, store_normed, store_fp8, "bf16")
