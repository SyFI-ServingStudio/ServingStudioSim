import csv
import json
from pathlib import Path

import pytest

from scripts.build_per_request_acceptance_trace import build_trace
from scripts.compare_spec_acceptance_by_request import compare, simulate_request


@pytest.mark.parametrize(
    "case",
    json.loads((Path(__file__).parent / "fixtures/speculative_acceptance_chains.json").read_text()),
)
def test_python_chain_matches_rust_completion_fixture(case):
    result = simulate_request(
        dense_request_id=case["request_id"],
        input_len=8,
        output_len=case["output_len"],
        acceptance=case["rates"],
        seed=case["seed"],
    )
    assert [step["emitted_tokens"] for step in result["progress"]] == case["emitted"]
    assert sum(case["emitted"]) == case["output_len"] - 1


@pytest.mark.parametrize("output_len", [1, 10])
def test_comparison_handles_small_populations_and_feeds_trace_builder(tmp_path, output_len):
    trace = tmp_path / "trace.csv"
    trace.write_text(f"id,input_len,output_len,arrival_time,accept_rate\n0,8,{output_len},0,1\n")
    steps = simulate_request(
        dense_request_id=0,
        input_len=8,
        output_len=output_len,
        acceptance=[1] * 5,
        seed=73,
    )["progress"]
    metrics = tmp_path / "metrics.jsonl"
    records = [
        {
            "schema_version": 4,
            "iteration_index": index,
            "decode_request_progress": [
                {
                    **step,
                    "request_id": "independent_0",
                    "query_len": 6,
                    # The producer records sampled tokens before output-length truncation.
                    "emitted_tokens": 6,
                    "accepted_draft_tokens": 5,
                }
            ],
        }
        for index, step in enumerate(steps)
    ]
    metrics.write_text("".join(json.dumps(row) + "\n" for row in records))
    result = compare(metrics_jsonl=metrics, trace_csv=trace, draft_tokens=5, seed=73)
    aggregate = result["aggregate"]
    assert aggregate["decode_round_delta_pct"] == (0 if steps else None)
    assert sum(group["n"] for group in result["input_length_quartiles"]) == 1
    assert aggregate["measured_acceptance_effective"] == aggregate["simulated_acceptance_effective"]
    assert result["predictive_alignment"] is False
    if steps:
        comparison = tmp_path / "comparison.json"
        comparison.write_text(json.dumps(result))
        output = tmp_path / "observed.csv"
        build_trace(source_trace=trace, comparison_json=comparison, output_trace=output)
        with output.open() as handle:
            row = next(csv.DictReader(handle))
        assert json.loads(row["accept_rate"]) == [1, 1, 0.5, 1, 1]


def test_recompute_is_reported_separately_from_verify(tmp_path):
    trace = tmp_path / "trace.csv"
    trace.write_text("id,input_len,output_len,arrival_time,accept_rate\n0,8,2,0,0\n")
    metrics = tmp_path / "metrics.jsonl"
    progress = {
        "request_id": "0",
        "kv_len": 8,
        "query_len": 6,
        "output_tokens_before": 1,
        "drafted_tokens": 5,
        "emitted_tokens": 1,
        "accepted_draft_tokens": 0,
    }
    metrics.write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": 4,
                    "iteration_index": index,
                    "decode_request_progress": [step],
                }
            )
            + "\n"
            for index, step in enumerate(
                [
                    {**progress, "query_len": 8, "drafted_tokens": 0, "emitted_tokens": 0},
                    progress,
                ]
            )
        )
    )
    result = compare(metrics_jsonl=metrics, trace_csv=trace, draft_tokens=5, seed=73)
    assert result["aggregate"]["measured_recompute_query_rows_excluded"] == 8
    assert result["aggregate"]["measured_decode_rounds"] == 1


def test_finished_async_work_is_counted_without_inventing_output(tmp_path):
    trace = tmp_path / "trace.csv"
    trace.write_text("id,input_len,output_len,arrival_time,accept_rate\n0,8,2,0,0\n")
    progress = {
        "request_id": "0", "kv_len": 8, "query_len": 6,
        "output_tokens_before": 1, "drafted_tokens": 5,
        "emitted_tokens": 1, "accepted_draft_tokens": 0,
    }
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text("".join(json.dumps({
        "schema_version": 4, "decode_request_progress": [step],
    }) + "\n" for step in [progress, {
        **progress, "output_tokens_before": None, "request_finished_before": True,
    }]))
    result = compare(metrics_jsonl=metrics, trace_csv=trace, draft_tokens=5, seed=73)
    assert result["aggregate"]["measured_decode_rounds"] == 2
    assert result["aggregate"]["measured_checked_rows"] == 12
    assert result["aggregate"]["measured_finished_request_rounds"] == 1
    row = result["requests_by_trace_order"][0]
    assert row["measured_emitted_tokens_raw"] == 2
    assert row["measured_acceptance"] == row["simulated_acceptance"]
