"""CPU tests for alignment-local NSYS normalization helpers.

Comparison, cost-tree folding, statistics, and rendering are tested with the
repository-level analyzer. GPU capture enters through
`python -m launcher alignment` and is not exercised here.
"""

from __future__ import annotations

from alignment.nsys import parse as nsys_parse


def test_nsys_kernel_category():
    assert nsys_parse.kernel_category("flash_fwd_kernel") == "attention"
    assert nsys_parse.kernel_category("nvjet_hgemm_128x256") == "gemm_or_cutlass"
    assert nsys_parse.kernel_category("triton_red_fused_rms") == "norm_reduce"
    assert nsys_parse.kernel_category("void act_and_mul_kernel") == "activation"
    assert nsys_parse.kernel_category("ncclDevKernel_AllReduce") == "nccl_collective"
    assert nsys_parse.kernel_category("some_random_kernel") == "other"


def test_merge_duration_union():
    # [0,10) ∪ [5,15) = 15 ; plus [20,25) = 5 → 20
    assert nsys_parse.merge_duration_ns([(0, 10), (5, 15), (20, 25)]) == 20
    assert nsys_parse.merge_duration_ns([]) == 0
