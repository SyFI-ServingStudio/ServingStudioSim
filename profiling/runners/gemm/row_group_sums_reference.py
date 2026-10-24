"""Independent value reference for row group sums."""

from typing import Any


def row_group_sums_reference(torch: Any, x: Any, group_size: int) -> Any:
    """(M, K) rows -> (M, K / group_size) fp32 sums of each group, summed in fp64."""
    m, k = x.shape
    return x.double().reshape(m, k // group_size, group_size).sum(-1).float()
