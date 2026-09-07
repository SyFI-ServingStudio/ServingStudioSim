import csv
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from alignment.request_population import audit_request_population
from launcher.alignment_campaign.metrics import REPORT_LOCATIONS, measure_case


@pytest.mark.parametrize("mutation", [None, "output", "id"])
def test_dense_ids_are_resolved_before_population_comparison(tmp_path, mutation):
    reuse = tmp_path / "e2e_reuse"
    reuse.mkdir()
    rows = [
        {"id": "opaque-b", "input_len": 8, "output_len": 4, "arrival_time": 0},
        {"id": "opaque-a", "input_len": 12, "output_len": 6, "arrival_time": 10},
    ]
    with (reuse / "trace_observed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    replay = [
        {"source": {"data": {
            "id": row["id"], "input_len": row["input_len"],
            "output_len_target": row["output_len"], "arrival_time_ms": row["arrival_time"],
        }}, "outcome": {"status": "SUCCESS", "output_len_actual": row["output_len"]}}
        for row in reversed(rows)
    ]
    replay_path = tmp_path / "replay.jsonl"
    replay_path.write_text("\n".join(json.dumps(row) for row in replay))
    (tmp_path / "profile_result.json").write_text(json.dumps({
        "replay_result": str(replay_path),
    }))
    simulated = [
        {"request_id": index, "completed": True,
         "fresh_prompt_tokens": row["input_len"], "num_output_tokens": row["output_len"]}
        for index, row in enumerate(rows)
    ]
    if mutation == "output":
        simulated[0]["num_output_tokens"] = 3
    elif mutation == "id":
        simulated[0]["request_id"] = 5
    raw = tmp_path / "simulation/raw"
    raw.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(list(reversed(simulated))), raw / "request_slo.parquet")
    result = audit_request_population(
        trace_path=reuse / "trace_observed.csv", replay_path=replay_path,
        slo_path=raw / "request_slo.parquet",
    )
    assert result["all_ok"] == (mutation is None)
    assert result["identities"] == [
        {"request_id": 0, "source_id": "opaque-b"},
        {"request_id": 1, "source_id": "opaque-a"},
    ]
    if mutation is not None:
        assert "opaque-b" in result["mismatched_source_ids"]

    analysis = tmp_path / "analysis_e2e"
    analysis.mkdir()
    (analysis / "alignment_manifest.json").write_text(json.dumps({
        "simulation_log_dir": str(raw.parent), "replay_result": str(replay_path),
    }))
    (raw / "params.json").write_text(json.dumps({"workload": {
        "session_dependency": "independent",
        "trace_files": [str(reuse / "trace_observed.csv")],
    }}))
    report = tmp_path / REPORT_LOCATIONS["e2e"]
    report.parent.mkdir(exist_ok=True)
    report.write_text(json.dumps({"schema_version": 1, "meta": {
        "measured_successful_requests": 2, "simulated_completed_requests": 2,
    }}))
    measurement = measure_case(tmp_path)
    assert measurement.provenance["request_population_audit"]["all_ok"] == (mutation is None)
    assert ("request identity/length/completion audit failed" in measurement.issues) == (
        mutation is not None
    )
    if mutation is None:
        sidecar = reuse / "trace_observed.csv.manifest.json"
        sidecar.write_text(json.dumps({"output_trace_sha256": "stale"}))
        stale = measure_case(tmp_path)
        assert any("manifest does not match" in issue for issue in stale.issues)
