"""Native EngineCore boundaries and request-population fidelity, without devices."""

import json
from types import SimpleNamespace as NS

import pytest

from alignment.neuron.vllm_records import RequestTimings
from alignment.neuron.vllm_runner import extract_stock_request_timings


def event(name, timestamp):
    return NS(type=NS(name=name), timestamp=timestamp)


def batch(timestamp, tokens=(), *, events=(), finish=None, finished=()):
    return {
        0: NS(
            timestamp=timestamp,
            outputs=[
                NS(
                    request_id="cmpl-case-0-1234abcd",
                    new_token_ids=list(tokens),
                    events=list(events),
                    finished=finish is not None,
                    finish_reason=NS(name=finish),
                )
            ],
            finished_requests=set(finished),
        )
    }


def complete(count=8):
    tracker = RequestTimings()
    assert tracker.observe(batch(10, [1], events=[event("QUEUED", 2), event("SCHEDULED", 4)])) == []
    return tracker.observe(batch(24, range(count - 1), finish="LENGTH"))[0]


def test_native_timestamps_arrive_with_first_output_and_empty_steps_do_not_move_tokens():
    tracker = RequestTimings()
    tracker.observe(batch(10, [1], events=[event("QUEUED", 2), event("SCHEDULED", 4)]))
    tracker.observe(batch(19))
    row = tracker.observe(batch(24, [2, 3], finish="LENGTH", finished={"cmpl-case-0-1234abcd"}))[0]
    assert row["engine_core_ttft_ms"] == 8000
    assert row["engine_queue_wait_ms"] == 2000
    assert row["engine_first_schedule_to_first_token_ms"] == 6000
    assert row["engine_core_decode_ms"] == 14000
    assert row["engine_core_tpot_ms"] == 7000
    assert row["num_output_tokens"] == 3
    with pytest.raises(ValueError, match="duplicate"):
        tracker.observe(batch(25, [3], finish="LENGTH"))


def test_one_token_and_empty_terminal_output_keep_actual_token_boundary():
    tracker = RequestTimings()
    tracker.observe(batch(10, [1], events=[event("QUEUED", 2), event("SCHEDULED", 4)]))
    row = tracker.observe(batch(30, finish="STOP"))[0]
    assert row["engine_core_ttft_ms"] == 8000
    assert row["engine_core_decode_ms"] == 0
    assert row["engine_core_tpot_ms"] is None
    assert (
        tracker.observe(
            {0: NS(timestamp=31, outputs=[], finished_requests={"cmpl-case-0-1234abcd"})}
        )
        == []
    )


@pytest.mark.parametrize(
    "mutation", ["missing_events", "backwards", "nan", "abort", "empty", "control_abort"]
)
def test_ineligible_or_missing_native_evidence_fails_closed(mutation):
    tracker = RequestTimings()
    first = batch(10, [1], events=[event("QUEUED", 2), event("SCHEDULED", 4)])
    if mutation == "missing_events":
        first[0].outputs[0].events = []
    elif mutation == "nan":
        first[0].timestamp = float("nan")
    elif mutation == "empty":
        first = batch(10, finish="LENGTH")
    elif mutation == "control_abort":
        first = {0: NS(timestamp=10, outputs=[], finished_requests={"aborted"})}
    if mutation in {"backwards", "abort"}:
        tracker.observe(first)
        first = batch(
            9 if mutation == "backwards" else 11,
            [2],
            finish="ABORT" if mutation == "abort" else "LENGTH",
        )
    with pytest.raises(ValueError):
        tracker.observe(first)


def test_generic_extractor_pairs_population_and_preserves_old_unavailable_receipts(tmp_path):
    log = tmp_path / "measurement.log"
    mapping = {"client_to_engine": {"case": "cmpl-case-0-1234abcd"}, "output_tokens": 8}
    log.write_text("legacy scheduler observations only\n")
    assert extract_stock_request_timings(log, tmp_path, mapping, required=False) is None
    with pytest.raises(ValueError, match="missing required"):
        extract_stock_request_timings(log, tmp_path, mapping, required=True)
    row = complete()
    line = "(EngineCore pid=42) VibeSimAlignmentRequestTiming " + json.dumps(row) + "\n"
    log.write_text(line)
    path = extract_stock_request_timings(log, tmp_path, mapping, required=True)
    parsed = json.loads(path.read_text())
    assert parsed["request_id"] == "case" and parsed["engine_core_tpot_ms"] == 2000
    log.write_text(line + line)
    with pytest.raises(ValueError):
        extract_stock_request_timings(log, tmp_path, mapping, required=True)
    log.write_text(line)
    with pytest.raises(ValueError):
        extract_stock_request_timings(
            log, tmp_path, {**mapping, "client_to_engine": {"missing": "x"}}, required=True
        )
    with pytest.raises(ValueError, match="token count"):
        extract_stock_request_timings(log, tmp_path, {**mapping, "output_tokens": 7}, required=True)


def test_stock_postprocess_hands_request_timings_to_analysis_manifest(tmp_path, monkeypatch):
    from alignment.neuron import vllm_runner as runner
    from alignment.neuron.vllm_records import STOCK_VERSIONS
    from launcher import alignment as launcher

    rid = "cmpl-case-0-1234abcd"
    (tmp_path / "server.log").write_text(
        "(EngineCore pid=42) VibeSimAlignmentRequestTiming " + json.dumps(complete()) + "\n"
    )
    replay = tmp_path / "replay.jsonl"
    replay.write_text("{}\n")
    frozen = {
        "files": {},
        "image": "pinned",
        "model_config": {},
        "checkpoint_path": "/model",
        "request_timing_schema": 2,
    }
    evidence = {
        "subject_continuations_by_prompt_sha256": {"hash": []},
        "http_generated_id_comparison": "unavailable",
    }
    for name, value in {
        "drive_summary.json": {"measurement_log_offset": 0, "log_path": str(replay)},
        "corpus-provenance.json": {"prompt_hashes": {"case": "hash"}},
        "accepted-forward-provenance.json": evidence,
        "source-provenance.json": frozen,
        "capture-source-after.json": frozen,
    }.items():
        (tmp_path / name).write_text(json.dumps(value))
    monkeypatch.setattr(runner, "verify_server_log", lambda _: None)
    monkeypatch.setattr(runner, "source_provenance", lambda _: frozen)
    monkeypatch.setattr(runner, "accepted_forward_evidence", lambda _: evidence)
    monkeypatch.setattr(
        runner, "extract_metrics_jsonl", lambda _, path, **kw: path.write_text("{}\n")
    )
    monkeypatch.setattr(
        runner,
        "validate_request_mapping",
        lambda *args: {"client_to_engine": {"case": rid}, "output_tokens": 8},
    )
    monkeypatch.setattr(
        runner,
        "_tagged",
        lambda path, tag: (
            [{"request_id": rid, "prompt_tokens": 504, "prompt_sha256": "hash"}]
            if tag == "VllmNeuronPrompt"
            else [
                {"rank": i, "pid": i + 10, "async_scheduling": True, "versions": STOCK_VERSIONS}
                for i in range(4)
            ]
        ),
    )
    result = runner.postprocess(
        NS(
            log_dir=tmp_path,
            server=NS(accepted_forward_path=tmp_path),
            gpu="AWS Trainium2 LNC2",
            profile_kind="workload_metrics",
        )
    )
    # Exercise the actual launcher handoff, not a reconstructed field-name assertion.
    monkeypatch.setattr(launcher, "_load_profile_result", lambda _: result)
    monkeypatch.setattr(launcher, "_load_simulation_params", lambda _: {})
    monkeypatch.setattr(launcher, "_simulation_target", lambda _: None)
    manifest = launcher._write_analysis_manifest(
        NS(
            log_dir=tmp_path / "analysis",
            iteration=NS(enabled=False),
            simulation_log_dir=tmp_path / "simulation",
            workload_profile_log_dir=tmp_path,
            e2e=NS(throughput_bins=20),
        )
    )
    assert json.loads(manifest.read_text())["request_timings_result"] == str(
        tmp_path / "request_timings.jsonl"
    )
