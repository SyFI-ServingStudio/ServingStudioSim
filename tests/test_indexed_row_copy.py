from __future__ import annotations

import pytest

from profiling.db.args import DType


def test_operation_and_sizes_are_rejected_before_cuda() -> None:
    """An unknown operation or empty row must fail before any GPU work, not be
    silently timed as some other statement."""
    from profiling.runners.elementwise.indexed_row_copy_torch import profile_indexed_row_copy

    with pytest.raises(ValueError, match="operation"):
        profile_indexed_row_copy(num_rows=2, row_elements=8, dtype="fp32", operation="copy")
    with pytest.raises(ValueError, match="> 0"):
        profile_indexed_row_copy(num_rows=0, row_elements=8, dtype="fp32", operation="gather")


@pytest.mark.parametrize("operation", ["gather", "scatter", "masked_zero"])
def test_each_statement_moves_the_rows_it_names(operation: str) -> None:
    """Runs the timed statement once on CPU and its own check: catches a gather
    or scatter that reads or writes the wrong (adjacent) slots."""
    import torch

    from profiling.runners.elementwise.indexed_row_copy_torch import _build

    run, check = _build(torch, 3, 16, DType.FP32, operation, device=torch.device("cpu"))
    run()
    check()


def test_logical_bytes_count_read_and_write_except_for_zero() -> None:
    from profiling.runners.elementwise.indexed_row_copy_torch import logical_bytes, slot_indices

    assert slot_indices(3) == [2, 4, 6]
    assert logical_bytes(3, 10, DType.FP32, "gather") == 240
    assert logical_bytes(3, 10, DType.FP32, "masked_zero") == 120
