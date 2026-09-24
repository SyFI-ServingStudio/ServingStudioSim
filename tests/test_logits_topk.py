"""Registration, validation, and GPU runner tests for ``logits_topk``."""

from __future__ import annotations

from dataclasses import fields

import pytest

from profiling.db.args import DType
from profiling.db.batch import coerce_args
from profiling.db.registry import MetricFamily, find_kernel_profiler_spec, known_backends
from profiling.kernels.logits_topk import KIND, LogitsTopkArgs
from profiling.runners.sampling.logits_topk import _validate_args


def test_both_backends_share_one_args_schema_and_table() -> None:
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
    assert sorted(known_backends(KIND)) == ["flashinfer", "torch"]
    for backend in ("flashinfer", "torch"):
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
