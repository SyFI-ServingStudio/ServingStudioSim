"""Schema-6 parsed documents: kernel rows in a parquet sibling of parsed.json."""

from __future__ import annotations

import copy
import json

import pytest

from alignment.nsys.parsed_io import kernel_rows_path, read_parsed, write_parsed


def _kernel(ordinal: int, start: int, *, correlation_id: int | None = 5) -> dict:
    return {
        "ordinal": ordinal,
        "name_id": ordinal % 2,
        "category": "gemm_or_cutlass",
        "start_ns": start,
        "end_ns": start + 3,
        "stream_id": 19,
        "correlation_id": correlation_id,
        "track_index": 0,
    }


def _parsed() -> dict:
    def range_row(phase: str, kernels: list[dict]) -> dict:
        return {"device_id": 0, "phase": phase, "start_ns": 0, "end_ns": 99, "kernels": kernels}

    def range_with_metrics(phase: str, kernels: list[dict]) -> dict:
        return {**range_row(phase, kernels), "metrics": {"prefill_tokens": 3}}

    return {
        "schema_version": 5,
        "kernel_names": {"0": "a", "1": "b"},
        "iteration_details": [
            {
                "iteration": 7,
                "ranges": [
                    range_with_metrics(
                        "preprocess", [_kernel(1, 10), _kernel(2, 20, correlation_id=None)]
                    ),
                    range_row("forward", []),
                ],
            },
            {"iteration": 8, "ranges": [range_row("forward", [_kernel(1, 30)])]},
        ],
    }


def test_round_trip_restores_every_inline_kernel_in_order(tmp_path):
    parsed = _parsed()
    path = tmp_path / "parsed.json"

    write_parsed(path, parsed)

    document = json.loads(path.read_text())
    assert document["schema_version"] == 6
    assert document["kernel_rows"] == {
        "file": "parsed.kernels.parquet",
        "format": "parquet",
        "rows": 3,
    }
    assert all(
        "kernels" not in range_row
        for detail in document["iteration_details"]
        for range_row in detail["ranges"]
    )
    assert kernel_rows_path(path).exists()
    # The caller's document still carries its kernels and range metrics.
    assert len(parsed["iteration_details"][0]["ranges"][0]["kernels"]) == 2
    assert "metrics" in parsed["iteration_details"][0]["ranges"][0]

    restored = read_parsed(path)
    expected = {**copy.deepcopy(parsed), "schema_version": 6}
    del expected["iteration_details"][0]["ranges"][0]["metrics"]
    assert restored == expected


def test_metrics_only_read_skips_kernel_rows(tmp_path):
    path = tmp_path / "parsed.json"
    write_parsed(path, _parsed())
    kernel_rows_path(path).unlink()

    restored = read_parsed(path, kernels=False)

    assert "kernel_rows" not in restored
    assert "kernels" not in restored["iteration_details"][0]["ranges"][0]


def test_schema_five_documents_read_unchanged(tmp_path):
    path = tmp_path / "parsed.json"
    path.write_text(json.dumps(_parsed(), indent=2))

    assert read_parsed(path) == _parsed()


def test_row_count_mismatch_is_rejected(tmp_path):
    path = tmp_path / "parsed.json"
    write_parsed(path, _parsed())
    document = json.loads(path.read_text())
    document["kernel_rows"]["rows"] = 4
    path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="declares 4"):
        read_parsed(path)
