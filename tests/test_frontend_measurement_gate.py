"""Exercise the real subprocess handshake without a serving process or GPU."""

import json
import subprocess
import sys
import time
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
    monkeypatch.setattr(alignment_runner.capture_facts, "record", lambda *a: {})
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


@pytest.mark.parametrize("gated", [True, False])
def test_replay_stops_when_the_server_fails(tmp_path, monkeypatch, gated):
    """A dead engine ends the replay within a poll, not at the stream idle timeout."""

    config = LoadGeneratorConfig(
        frontend=IndependentFrontendConfig(path="trace.csv"),
        text_file="corpus.txt", tokenizer="tokenizer", warmup=gated,
    )
    prepared = runner.PreparedReplay(
        tmp_path / "trace.csv", tmp_path / "corpus.txt", "tokenizer",
        tmp_path / "replay.jsonl", tmp_path / "summary.json",
    )
    popen = subprocess.Popen
    program = "import time\n"
    if gated:
        program += (
            "print('REQ_FRONTEND_MEASUREMENT_READY_V1', flush=True)\n"
            "assert input() == 'continue'\n"
        )
    program += "time.sleep(60)\n"  # requests the dead server never answers
    monkeypatch.setattr(
        runner.subprocess, "Popen",
        lambda argv, **kwargs: popen([sys.executable, "-c", program], **kwargs),
    )
    polls = []

    def server_failure():
        polls.append(None)
        return "Subprocess scheduler_0 (pid=1) crashed with exit code 1" if len(polls) > 1 else None

    started = time.monotonic()
    with pytest.raises(runner.ServerFailedDuringReplay, match="scheduler_0"):
        runner.run_replay(
            config, prepared, base_url="http://localhost:8000", model="model",
            measurement_ready=(lambda: None) if gated else None,
            server_failure=server_failure,
        )
    assert time.monotonic() - started < 30


def test_server_failure_reads_new_log_lines_and_the_process(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("Subprocess scheduler_0 (pid=1) crashed with exit code 1\n")  # before replay
    proc = SimpleNamespace(code=None, poll=lambda: proc.code)
    poll = alignment_runner._server_failure(proc, log)
    assert poll() is None
    with log.open("a") as out:
        out.write("[2026] Subprocess scheduler_0 (pid=432) crashed with exit")  # partial line
    assert poll() is None
    with log.open("a") as out:
        out.write(" code 1. Triggering SIGQUIT for cleanup...\n")
    assert "scheduler_0 (pid=432) crashed" in poll()
    with log.open("a") as out:
        out.write('VIBESIM_AUTOTUNE_FORBIDDEN {"source": "flashinfer.AutoTuner"}\n')
    assert poll().startswith("VIBESIM_AUTOTUNE_FORBIDDEN")
    proc.code = -9
    assert poll() == "server process exited with code -9"
