"""CPU tests for ``sliding_window_kv_append`` (TensorFold ``tensorfold_triton`` backend).

The GPU path is covered by the public profiling smoke, whose bit-exact check
runs this reference; these protect the reference and the argument gate.
"""

from __future__ import annotations

import pytest

from profiling.db.args import DType
from profiling.runners.attention import sliding_window_kv_append_tensorfold as runner
from profiling.runners.attention.sliding_window_kv_append_reference import (
    sliding_window_kv_append_reference,
)
from profiling.runners.exceptions import ProfilerNotImplemented

torch = pytest.importorskip("torch")


def test_reference_keeps_the_last_window_rows_of_context_then_new():
    # Catches a reference that keeps the first window rows, drops the new rows,
    # or puts them before the context: it would then accept a wrong kernel.
    old = torch.arange(6.0).view(1, 6, 1)
    new = torch.arange(100.0, 103.0).view(1, 3, 1)
    got = sliding_window_kv_append_reference(torch, old, new, window=4)
    assert got.flatten().tolist() == [5.0, 100.0, 101.0, 102.0]
    assert sliding_window_kv_append_reference(torch, None, new, window=8).shape == (1, 3, 1)
    assert sliding_window_kv_append_reference(torch, None, new, window=2).flatten().tolist() == [
        101.0,
        102.0,
    ]


_ARGS = dict(
    num_kv_heads=8,
    head_dim=128,
    dtype=DType.BF16,
    window=2047,
    batch_size=8,
    context_len=2047,
    num_tokens=16,
)


def test_kept_rows_clamp_at_the_window():
    # Catches a byte count (and so a bandwidth metric) that charges rows past the window.
    assert runner.validate_args(**_ARGS).keep == 2047
    assert runner.validate_args(**{**_ARGS, "context_len": 100}).keep == 116


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("dtype", DType.FP16, ProfilerNotImplemented),
        ("head_dim", 96, ProfilerNotImplemented),
        ("context_len", 2048, ValueError),
        ("num_tokens", 0, ValueError),
    ],
)
def test_calls_the_drafter_never_makes_are_refused(field, value, error):
    # Catches a row recorded for a dtype the pointer table cannot type, a head
    # dim tl.arange rejects, a context past the window, or an empty append.
    with pytest.raises(error):
        runner.validate_args(**{**_ARGS, field: value})
