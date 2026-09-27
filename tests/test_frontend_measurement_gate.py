"""Exercise the real subprocess handshake without a serving process or GPU."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from alignment import runner as alignment_runner
from alignment.load_generator import runner
from alignment.load_generator.config import IndependentFrontendConfig, LoadGeneratorConfig
from alignment.profiler.config import ProfileConfig, ServerConfig


@pytest.mark.parametrize("mode", ["ready", "missing", "duplicate", "callback_error"])
def test_frontend_measurement_boundary(tmp_path, monkeypatch, mode):
    config = LoadGeneratorConfig(
        frontend=IndependentFrontendConfig(path="trace.csv"),
        text_file="corpus.txt", tokenizer="tokenizer", warmup=True,
    )
    prepared = runner.PreparedReplay(
        tmp_path / "trace.csv", tmp_path / "corpus.txt", "tokenizer",
        tmp_path / "replay.jsonl", tmp_path / "summary.json",
    )
    popen = subprocess.Popen
    calls = []
    program = (
        "import sys\n"
        "print('REQ_FRONTEND_MEASUREMENT_READY_V1', flush=True)\n"
        "assert input() == 'continue'\n"
    )
    if mode == "missing":
        program = "pass"
    if mode == "duplicate":
        program += "print('REQ_FRONTEND_MEASUREMENT_READY_V1', flush=True)\n"

    def launch(argv, **kwargs):
        assert "--warmup" in argv
        assert "--measurement-gate" in argv
        return popen([sys.executable, "-c", program], **kwargs)

    def ready():
        calls.append("measurement")
        if mode == "callback_error":
            raise RuntimeError("capture failed")

    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    if mode == "ready":
        result = runner.run_replay(
            config, prepared, base_url="http://localhost:8000", model="model",
            measurement_ready=ready,
        )
        assert result["warmup"] is True
        assert calls == ["measurement"]
    else:
        with pytest.raises(RuntimeError):
            runner.run_replay(
                config, prepared, base_url="http://localhost:8000", model="model",
                measurement_ready=ready,
            )
        assert len(calls) == (0 if mode == "missing" else 1)


def test_warmup_duplicate_ids_are_excluded_on_finalize_and_resume(tmp_path, monkeypatch):
    config = LoadGeneratorConfig(
        frontend=IndependentFrontendConfig(path="trace.csv"),
        text_file="corpus.txt", tokenizer="tokenizer", warmup=True,
    )
    cfg = ProfileConfig(
        name="warm", log_dir=str(tmp_path), gpu="B200", profile_kind="workload_metrics",
        workload=config, server=ServerConfig(model_path="model"),
    )
    prepared = runner.PreparedReplay(
        tmp_path / "trace.csv", tmp_path / "corpus.txt", "tokenizer",
        tmp_path / "replay.jsonl", tmp_path / "summary.json",
    )
    record = {
        "schema_version": 1, "engine_request_id": "cmpl-vibesim_7-0",
        "engine_core_ttft_ms": 12.5, "engine_queue_wait_ms": 3.0,
        "engine_first_schedule_to_first_token_ms": 9.5,
    }
    line = "INFO VibeSimAlignmentRequestTiming " + json.dumps(record) + "\n"
    source = tmp_path / "server.log"
    source.write_text(line + line)
    monkeypatch.setattr(alignment_runner, "_successful_replay_request_ids", lambda _: {"vibesim_7"})
    for _ in range(2):
        result = alignment_runner._finalize_profile(
            cfg, log_dir=tmp_path, engine_dir=tmp_path, server_log=source,
            out_rep=tmp_path / "unused", prepared_replay=prepared,
            drive_summary={"measurement_log_offset": len(line.encode())}, nsys_executable=None,
        )
        assert result["request_timing_count"] == 1
        assert source.read_text() == line + line
    with pytest.raises(ValueError, match="missing its persisted"):
        alignment_runner._finalize_profile(
            cfg, log_dir=tmp_path, engine_dir=tmp_path, server_log=source,
            out_rep=tmp_path / "unused", prepared_replay=prepared,
            drive_summary={}, nsys_executable=None,
        )


def test_profiler_and_counters_start_only_at_frontend_measurement_boundary(tmp_path, monkeypatch):
    config = LoadGeneratorConfig(
        frontend=IndependentFrontendConfig(path="trace.csv"),
        text_file="corpus.txt", tokenizer="tokenizer", warmup=True,
    )
    cfg = ProfileConfig(
        name="warm", log_dir=str(tmp_path), gpu="B200", workload=config,
        server=ServerConfig(model_path="model", enable_server_load_tracking=True),
    )
    events = []
    driver = alignment_runner.vllm_server
    monkeypatch.setattr(runner, "prepare_replay", lambda *_: None)
    monkeypatch.setattr(runner, "build_session_runner", lambda: None)
    monkeypatch.setattr(alignment_runner, "_resolve_fork_python", lambda _: "python")
    monkeypatch.setattr(alignment_runner, "_preflight_capture_environment", lambda *a, **k: None)
    monkeypatch.setattr(alignment_runner, "_shutdown", lambda _: None)
    monkeypatch.setattr(alignment_runner, "_finalize_profile", lambda _, **k: k["drive_summary"])
    monkeypatch.setattr(alignment_runner.time, "sleep", lambda _: None)
    monkeypatch.setattr(driver, "build_server_argv", lambda *a: ["python"])
    monkeypatch.setattr(driver, "build_server_env", lambda *a, **k: {})
    monkeypatch.setattr(driver, "write_launch_metadata", lambda *a: None)
    monkeypatch.setattr(driver, "wait_for_ready", lambda *a: None)
    monkeypatch.setattr(driver, "verify_server_started", lambda *a: None)
    monkeypatch.setattr(driver, "wait_for_idle", lambda *a: True)
    monkeypatch.setattr(driver, "speculative_decode_enabled", lambda _: True)
    monkeypatch.setattr(driver, "set_cuda_profile", lambda *a, active: events.append(active))
    monkeypatch.setattr(driver, "fetch_spec_decode_metrics", lambda _: list(events))
    monkeypatch.setattr(
        alignment_runner.runtime_artifacts, "prepare_python_runtime", lambda *a: None,
    )
    monkeypatch.setattr(
        alignment_runner.nsys_capture, "resolve_nsys_executable",
        lambda _: SimpleNamespace(provenance=lambda: {}),
    )
    monkeypatch.setattr(alignment_runner.nsys_capture, "build_nsys_prefix", lambda *a: [])

    def launch(*args, **kwargs):
        assert kwargs["env"]["VLLM_SERVER_DEV_MODE"] == "1"
        return object()

    def replay(*args, measurement_ready, **kwargs):
        assert events == []
        events.append("warmup_drained")
        measurement_ready()
        assert events == ["warmup_drained", True]
        events.append("measured_requests")
        return {"warmup": True}

    monkeypatch.setattr(alignment_runner.subprocess, "Popen", launch)
    monkeypatch.setattr(runner, "run_replay", replay)
    result = alignment_runner.run_profile(cfg)
    assert events == ["warmup_drained", True, "measured_requests", False]
    assert result["spec_decode_metrics_before"] == ["warmup_drained", True]
    assert result["spec_decode_metrics_after"] == ["warmup_drained", True, "measured_requests"]
    assert result["replay_end_monotonic_ns"] >= result["replay_start_monotonic_ns"]
    assert result["measurement_log_offset"] == 0
