"""A cache build's work list can be written here and measured elsewhere."""

from __future__ import annotations

import json

import pytest

from profiling import perf_api
from profiling.plan import WorkCollector, read_units, set_collector, write_units


def test_a_work_list_round_trips_and_folds_repeats(tmp_path):
    first = WorkCollector()
    first.record("single_gemm", "torch_linear", [{"m": 8, "n": 16, "k": 32}], gpu_name="NVIDIA H100")
    second = WorkCollector()
    second.record(
        "single_gemm",
        "torch_linear",
        [{"m": 8, "n": 16, "k": 32}, {"m": 4, "n": 16, "k": 32}],
        gpu_name="NVIDIA H100",
    )
    path = tmp_path / "work.json"
    write_units(first, path)
    write_units(second, path)  # a second cache key appends
    assert len(json.loads(path.read_text())["units"]) == 2
    collector = read_units(path)
    assert len(collector) == 2
    (unit,) = collector.units()
    assert (unit.kernel_kind, unit.backend, unit.gpu_name) == (
        "single_gemm",
        "torch_linear",
        "NVIDIA H100",
    )


def test_an_unknown_work_list_is_refused(tmp_path):
    path = tmp_path / "work.json"
    path.write_text(json.dumps({"schema_version": 99, "units": []}))
    with pytest.raises(ValueError, match="schema-1 work list"):
        read_units(path)


def test_issue_collected_exports_instead_of_measuring(tmp_path, monkeypatch):
    out = tmp_path / "work.json"
    monkeypatch.setenv(perf_api.COLLECT_EXPORT_ENV, str(out))
    collector = WorkCollector()
    collector.record("single_gemm", "torch_linear", [{"m": 8, "n": 16, "k": 32}])
    set_collector(collector)
    # No GPU is asked for: the export returns before `require_gpu`.
    monkeypatch.setattr(
        "profiling.gpu_policy.require_gpu",
        lambda reason: pytest.fail(f"asked for a GPU: {reason}"),
    )
    perf_api.issue_collected(expected_specs=1)
    assert len(read_units(out)) == 1
