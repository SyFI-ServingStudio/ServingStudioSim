import csv
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.summarize_matrix_alignment import audit_request_population


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
    result = audit_request_population(tmp_path, {"workload_profile": str(tmp_path)})
    assert result["all_ok"] == (mutation is None)
    assert result["identities"] == [
        {"request_id": 0, "source_id": "opaque-b"},
        {"request_id": 1, "source_id": "opaque-a"},
    ]
    if mutation is not None:
        assert "opaque-b" in result["mismatched_source_ids"]
