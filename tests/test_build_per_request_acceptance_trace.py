import csv
import json

import pytest

from scripts.build_per_request_acceptance_trace import build_trace


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text(
        "id,input_len,output_len,arrival_time,accept_rate,extra\n0,8,9,0,0.1,a\n1,16,3,1,0.1,b\n"
    )
    comparison = {
        "aggregate": {
            "measured_acceptance_effective": {
                "conditional_acceptance_rates": [0.6, 0.4],
            }
        },
        "requests_by_trace_order": [
            {
                "request_id": "0",
                "measured_acceptance": {
                    "conditional_acceptance_rates": [0.9, 0.8],
                },
            },
            {
                "request_id": "1",
                "measured_acceptance": {
                    "conditional_acceptance_rates": [0.7, None],
                },
            },
        ],
    }
    return source, comparison, tmp_path / "comparison.json", tmp_path / "out.csv"


@pytest.mark.parametrize("prefix", ["", "independent_"])
def test_trace_preserves_all_other_fields_and_records_fallback(inputs, prefix):
    source, comparison, comparison_path, output = inputs
    for row in comparison["requests_by_trace_order"]:
        row["request_id"] = prefix + row["request_id"]
    comparison_path.write_text(json.dumps(comparison))
    manifest = build_trace(
        source_trace=source, comparison_json=comparison_path, output_trace=output
    )
    with output.open() as handle:
        rows = list(csv.DictReader(handle))
    with source.open() as handle:
        original = list(csv.DictReader(handle))
    assert [json.loads(row.pop("accept_rate")) for row in rows] == [[0.9, 0.8], [0.7, 0.4]]
    for row in original:
        row.pop("accept_rate")
    assert rows == original
    assert manifest["predictive_alignment"] is False
    assert manifest["fallback_positions"] == [{"request_id": prefix + "1", "position": 1}]
    assert json.loads(output.with_suffix(".csv.manifest.json").read_text()) == manifest
    with pytest.raises(FileExistsError):
        build_trace(source_trace=source, comparison_json=comparison_path, output_trace=output)


@pytest.mark.parametrize("defect", ["duplicate", "missing", "nan", "depth", "no_fallback"])
def test_invalid_evidence_does_not_create_trace(inputs, defect):
    source, comparison, comparison_path, output = inputs
    rows = comparison["requests_by_trace_order"]
    if defect == "duplicate":
        rows.append(rows[0])
    elif defect == "missing":
        rows.pop()
    elif defect == "nan":
        rows[0]["measured_acceptance"]["conditional_acceptance_rates"][0] = float("nan")
    elif defect == "depth":
        rows[0]["measured_acceptance"]["conditional_acceptance_rates"].pop()
    else:
        comparison["aggregate"]["measured_acceptance_effective"]["conditional_acceptance_rates"][
            1
        ] = None
    comparison_path.write_text(json.dumps(comparison))
    with pytest.raises(ValueError):
        build_trace(source_trace=source, comparison_json=comparison_path, output_trace=output)
    assert not output.exists()
