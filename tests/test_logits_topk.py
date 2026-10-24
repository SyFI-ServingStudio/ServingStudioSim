"""Registration, validation, and GPU runner tests for ``logits_topk``."""

from __future__ import annotations

from dataclasses import fields

import pytest

from profiling.db.args import DType
from profiling.db.batch import coerce_args
from profiling.db.registry import MetricFamily, find_kernel_profiler_spec, known_backends
from profiling.kernels.logits_topk import KIND, LogitsTopkArgs
from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.sampling.logits_topk import (
    _check_unsorted_selection,
    _validate_args,
    profile_logits_topk_tensorfold_torch,
)


def test_all_backends_share_one_args_schema_and_table() -> None:
    # The Rust payload names these fields; a rename strands every profiled row.
    assert [field.name for field in fields(LogitsTopkArgs)] == [
        "num_rows",
        "num_columns",
        "top_k",
        "dtype",
    ]
    args = coerce_args(
        LogitsTopkArgs, {"num_rows": "8", "num_columns": "38720", "top_k": "16", "dtype": "bf16"}
    )
    assert args == LogitsTopkArgs(8, 38720, 16, DType.BF16)
    assert sorted(known_backends(KIND)) == ["flashinfer", "tensorfold_torch", "torch"]
    for backend in ("flashinfer", "tensorfold_torch", "torch"):
        spec = find_kernel_profiler_spec(KIND, backend)
        assert spec.args_schema is LogitsTopkArgs
        assert spec.table_name == KIND
        assert spec.metric_family is MetricFamily.COMPUTE


@pytest.mark.parametrize(
    ("num_rows", "num_columns", "top_k", "dtype"),
    [
        (8, 16, 17, "bf16"),  # more selected than there are columns
        (8, 16, 0, "bf16"),
        (0, 16, 4, "bf16"),
        (8, 16, 4, "fp8_e4m3"),  # flashinfer's radix select takes float scores only
        (True, 16, 4, "bf16"),
    ],
)
def test_invalid_shapes_are_rejected_before_any_cuda_work(
    num_rows: int, num_columns: int, top_k: int, dtype: str
) -> None:
    with pytest.raises(ValueError):
        _validate_args(num_rows, num_columns, top_k, dtype)


@pytest.mark.gpu
@pytest.mark.parametrize(("num_columns", "top_k"), [(38720, 16), (64, 16)])
def test_flashinfer_selects_what_torch_selects_at_both_call_site_widths(
    num_columns: int, top_k: int
) -> None:
    # The DFlash2 selector calls `_topk` twice: over a vocabulary shard, then over
    # the gathered `top_k * tp` candidates. The runner checks flashinfer against
    # torch.topk before timing, so a returned metric means the check passed.
    pytest.importorskip("flashinfer")
    from profiling.runners.sampling.logits_topk import profile_logits_topk_flashinfer

    metrics = profile_logits_topk_flashinfer(8, num_columns, top_k, "bf16")
    assert metrics.time_ms > 0


def test_tensorfold_torch_measures_only_the_fp32_select_tensorfold_makes() -> None:
    # Both TensorFold call sites select over `logits.float()`; a bf16 row would
    # time a select TensorFold never runs (and skip nothing it does run).
    with pytest.raises(ProfilerNotImplemented):
        profile_logits_topk_tensorfold_torch(8, 98592, 16, "bf16")


def test_unsorted_selection_check_compares_sets_not_order() -> None:
    # Catches a check that demands sorted output (sorted=False never sorts) or
    # one that would pass a select with a wrong k-th boundary.
    torch = pytest.importorskip("torch")
    scores = torch.randn(3, 500, generator=torch.Generator().manual_seed(0))
    values, indices = torch.topk(scores, 16, dim=-1, sorted=False)
    shuffled = torch.randperm(16)
    _check_unsorted_selection(torch, scores, 16, values[:, shuffled], indices[:, shuffled])
    wrong_values = values.clone()
    wrong_indices = indices.clone()
    smallest = scores.argmin(dim=-1)
    wrong_indices[:, 0] = smallest
    wrong_values[:, 0] = scores.gather(1, smallest[:, None])[:, 0]
    with pytest.raises(RuntimeError):
        _check_unsorted_selection(torch, scores, 16, wrong_values, wrong_indices)
