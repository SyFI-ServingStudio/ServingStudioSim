"""Native capture and clean workload passes through one NxDI token server."""

import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from alignment.load_generator import runner as frontend
from alignment.neuron.config import NxdiProfileConfig
from alignment.neuron.normalize import iteration_metric, write_normalized
from alignment.profiler.engine_records import NXDI_RECORDS
from alignment.profiler.record_extraction import (
    extract_metrics_jsonl,
    extract_request_timings_jsonl,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _http(base: str, path: str, *, post: bool = False) -> dict:
    request = urllib.request.Request(
        base + path, data=b"{}" if post else None, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def check_port(host: str, port: int) -> None:
    with socket.socket() as sock:
        try:
            sock.bind((host, port))
        except OSError as error:
            raise RuntimeError(f"NxDI alignment port {host}:{port} is occupied") from error


def _write_result(config: NxdiProfileConfig, result: dict) -> dict:
    result = {
        "producer_kind": "framework_capture",
        "engine": "nxdi",
        "profile_kind": config.profile_kind,
        "log_dir": str(Path(config.log_dir).resolve()),
        "gpu": config.gpu,
        "server_tp_size": 1,
        "server_dp_size": 1,
        "compiled_shapes": {"context_bucket": 128, "kv_bucket": 512},
        **result,
    }
    (Path(config.log_dir) / "profile_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def resume_capture(config: NxdiProfileConfig) -> dict:
    """Resume immutable native raw evidence without loading a model or device."""
    log_dir = Path(config.log_dir)
    trace, records = log_dir / "system-trace.json", log_dir / "forward-records.json"
    normalized = write_normalized(log_dir, trace, records)
    metrics = log_dir / "metrics.jsonl"
    rows = json.loads(records.read_text())["iterations"]
    metrics.write_text("".join(json.dumps(iteration_metric(row)) + "\n" for row in rows))
    return _write_result(
        config,
        {
            **normalized,
            "metrics_jsonl": str(metrics),
            "raw_trace": str(trace),
            "forward_records": str(records),
            "resumed_from_existing_capture": True,
        },
    )


def run_profile(config: NxdiProfileConfig, *, resume: bool = False) -> dict:
    config.validate()
    log_dir = Path(config.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        if config.profile_kind == "neuron":
            return resume_capture(config)
        result = log_dir / "profile_result.json"
        if not result.is_file():
            raise FileNotFoundError("no completed clean workload pass to resume")
        return json.loads(result.read_text())
    check_port(config.server.host, config.server.port)
    prepared = frontend.prepare_replay(config.workload, log_dir)
    frontend.build_session_runner()
    env = dict(os.environ)
    if "NEURON_RT_VISIBLE_CORES" in env:
        raise ValueError("unset NEURON_RT_VISIBLE_CORES; the NxDI server pool owns allocation")
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PATH"] = str(Path(config.fork_python).parent) + os.pathsep + env["PATH"]
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["OMP_NUM_THREADS"] = "4"
    env["PJRT_DEVICE"] = "NEURON"
    env["NXD_CPU_MODE"] = "0"
    command = [
        config.fork_python,
        "-u",
        "-m",
        "alignment.neuron.server",
        "--model-dir",
        config.server.model_path,
        "--compiled-dir",
        config.server.compiled_path,
        "--log-dir",
        str(log_dir),
        "--port",
        str(config.server.port),
        "--neuron-device",
        str(config.server.neuron_device),
    ]
    (log_dir / "launch.json").write_text(
        json.dumps(
            {
                "producer_kind": "framework_capture",
                "engine": "nxdi",
                "command": command,
                "profile_kind": config.profile_kind,
                "compiled_path": config.server.compiled_path,
                "model_path": config.server.model_path,
            },
            indent=2,
        )
        + "\n"
    )
    base = f"http://{config.server.host}:{config.server.port}"
    server_log = log_dir / "server.log"
    active_trace = False
    boundary = {}
    with server_log.open("w") as output:
        proc = subprocess.Popen(
            command,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            cwd=log_dir,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + config.server.startup_timeout
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"NxDI server exited {proc.returncode}; inspect {server_log}"
                    )
                try:
                    _http(base, "/health")
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"NxDI server readiness timed out; inspect {server_log}")
                    time.sleep(0.2)

            def measurement_ready():
                nonlocal active_trace
                if _http(base, "/load")["server_load"] != 0:
                    raise RuntimeError("NxDI preflight did not drain")
                boundary["measurement_log_offset"] = server_log.stat().st_size
                if config.profile_kind == "neuron":
                    _http(base, "/start_profile", post=True)
                    active_trace = True
                boundary["replay_start_monotonic_ns"] = time.monotonic_ns()

            drive = frontend.run_replay(
                config.workload,
                prepared,
                base_url=base,
                model="llama3.1-8b-nxdi",
                measurement_ready=measurement_ready,
            )
            boundary["replay_end_monotonic_ns"] = time.monotonic_ns()
            if _http(base, "/load")["server_load"] != 0:
                raise RuntimeError("NxDI workload did not drain")
            if active_trace:
                _http(base, "/stop_profile", post=True)
                active_trace = False
            drive.update(boundary)
            (log_dir / "drive_summary.json").write_text(json.dumps(drive, indent=2) + "\n")
        finally:
            if active_trace and proc.poll() is None:
                try:
                    _http(base, "/stop_profile", post=True)
                except (urllib.error.URLError, TimeoutError):
                    pass
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
    measured_log = log_dir / "measurement.log"
    measured_log.write_bytes(server_log.read_bytes()[boundary["measurement_log_offset"] :])
    replay_rows = [json.loads(line) for line in prepared.log_path.read_text().splitlines() if line]
    successful = {
        r["outcome"]["request_id"] for r in replay_rows if r["outcome"]["status"] == "SUCCESS"
    }
    if len(successful) != len(replay_rows):
        raise RuntimeError("NxDI replay contains failed or duplicated requests")
    metrics = log_dir / "metrics.jsonl"
    timing = log_dir / "request_timings.jsonl"
    extract_metrics_jsonl(measured_log, metrics, records=NXDI_RECORDS)
    count = extract_request_timings_jsonl(
        measured_log, timing, expected_request_ids=successful, records=NXDI_RECORDS
    )
    evidence = [
        json.loads(line.split("NxdiOutputEvidence ", 1)[1])
        for line in measured_log.read_text().splitlines()
        if "NxdiOutputEvidence " in line
    ]
    (log_dir / "output-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    result = {
        "metrics_jsonl": str(metrics),
        "request_timings_jsonl": str(timing),
        "request_timing_count": count,
        "server_log": str(server_log),
        "replay_result": str(prepared.log_path.resolve()),
        "drive_summary": drive,
        "output_evidence": str(log_dir / "output-evidence.json"),
        "runtime_provenance": str(log_dir / "runtime-provenance.json"),
    }
    if config.profile_kind == "neuron":
        result.update(
            write_normalized(
                log_dir,
                log_dir / "system-trace.json",
                log_dir / "forward-records.json",
                request_ids=successful,
            )
        )
        result["raw_trace"] = str(log_dir / "system-trace.json")
        result["forward_records"] = str(log_dir / "forward-records.json")
    return _write_result(config, result)
