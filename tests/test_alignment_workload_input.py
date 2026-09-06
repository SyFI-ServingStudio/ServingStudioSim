import csv
import json

import pytest

from alignment.workload_input import acceptance_counts, prepare_workload
from launcher.alignment import main


def evidence(tmp_path, depth=3):
    trace = tmp_path / "trace.csv"
    trace.write_text("id,input_len,output_len,arrival_time,extra\na,8,4,0,x\nb,12,4,10,y\n")
    replay = tmp_path / "replay.jsonl"
    replay.write_text("\n".join(json.dumps({
        "source": {"data": {"id": key, "input_len": length, "output_len_target": 4,
                             "arrival_time_ms": arrival}},
        "outcome": {"status": "SUCCESS", "output_len_actual": 4},
    }) for key, length, arrival in [("b", 12, 10), ("a", 8, 0)]))
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(json.dumps({"decode_request_progress": [
        {"request_id": "independent_a", "drafted_tokens": depth,
         "accepted_draft_tokens": depth},
        {"request_id": "independent_a", "drafted_tokens": depth,
         "accepted_draft_tokens": 0},
        {"request_id": "independent_a", "drafted_tokens": depth,
         "accepted_draft_tokens": depth, "request_finished_before": True},
    ]}))
    result = {"profile_kind": "workload_metrics", "drive_summary": {"reached_idle": True},
              "replay_result": str(replay), "metrics_jsonl": str(metrics)}
    (tmp_path / "profile_result.json").write_text(json.dumps(result))
    return {"source_trace": trace, "profile_dir": tmp_path,
            "output_trace": tmp_path / "observed.csv", "draft_tokens": depth,
            "request_id_prefix": "independent_", "missing_acceptance": "run-aggregate"}


@pytest.mark.parametrize("depth", [1, 3, 5])
def test_explicit_depth_preserves_trace_and_records_conditioning(tmp_path, depth):
    options = evidence(tmp_path, depth)
    original = options["source_trace"].read_bytes()
    manifest = prepare_workload(**options)
    with options["output_trace"].open() as stream:
        rows = list(csv.DictReader(stream))
    assert [json.loads(row.pop("accept_rate")) for row in rows] == [
        [0.5] + [1.0] * (depth - 1), [0.5] + [1.0] * (depth - 1),
    ]
    with options["source_trace"].open() as stream:
        assert rows == list(csv.DictReader(stream))
    assert options["source_trace"].read_bytes() == original
    assert manifest["predictive_alignment"] is False
    assert len(manifest["fallback_positions"]) == depth
    assert set(manifest["sources"]) == {"trace", "profile_result", "replay", "metrics"}
    with pytest.raises(FileExistsError):
        prepare_workload(**options)


@pytest.mark.parametrize("defect", ["prefix", "missing", "depth", "replay", "partial",
                                    "nsys", "invalid_count", "duplicate_trace", "unobserved"])
def test_invalid_or_missing_evidence_never_publishes_output(tmp_path, defect):
    options = evidence(tmp_path)
    if defect == "prefix":
        options["request_id_prefix"] = ""
    elif defect == "missing":
        options["missing_acceptance"] = "error"
    elif defect == "depth":
        options["draft_tokens"] = 0
    elif defect == "replay":
        replay = tmp_path / "replay.jsonl"
        replay.write_text(replay.read_text().replace('"SUCCESS"', '"FAILED"'))
    elif defect in {"partial", "nsys"}:
        path = tmp_path / "profile_result.json"
        result = json.loads(path.read_text())
        if defect == "partial":
            result["drive_summary"]["reached_idle"] = False
        else:
            result["profile_kind"] = "nsys"
        path.write_text(json.dumps(result))
    elif defect == "invalid_count":
        path = tmp_path / "metrics.jsonl"
        path.write_text(path.read_text().replace('"accepted_draft_tokens": 3',
                                                 '"accepted_draft_tokens": 4'))
    elif defect == "duplicate_trace":
        path = options["source_trace"]
        path.write_text(path.read_text() + "a,8,4,0,z\n")
    else:
        (tmp_path / "metrics.jsonl").write_text("{}")
    with pytest.raises(ValueError):
        prepare_workload(**options)
    assert not options["output_trace"].exists()
    assert not options["output_trace"].with_suffix(".csv.manifest.json").exists()


def test_prepare_workload_public_command(tmp_path):
    options = evidence(tmp_path)
    argv = ["prepare-workload"]
    for key, value in options.items():
        argv.extend(["--" + key.replace("_", "-"), str(value)])
    assert main(argv) == 0
    assert options["output_trace"].is_file()


def test_rejected_candidates_are_trials_and_finished_queue_is_not_output(tmp_path):
    evidence(tmp_path)
    assert acceptance_counts(tmp_path / "metrics.jsonl", {"a", "b"}, draft_tokens=3,
                             request_id_prefix="independent_") == {
        "a": [[1, 1, 1], [2, 1, 1]], "b": [[0, 0, 0], [0, 0, 0]],
    }
