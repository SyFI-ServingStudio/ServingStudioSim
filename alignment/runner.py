"""Coordinate one measured alignment run across TraceLab, vLLM, and Nsight.

The explicit `launcher alignment profile <profiling-config>` command calls this
module. It prepares TraceLab, launches one instrumented-fork vLLM server under
nsys, drives the workload, and exports the trace into the profiling config's
`log_dir`. It does not generate timing-predict inputs or run comparison analysis.

The server is spawned in its own session so a SIGINT to the group lets nsys
finalize the `.nsys-rep` cleanly on shutdown.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

from .load_generator import runner as load_generator
from .nsys.parse import parse_host_timeline, parse_trace, parsed_window_ns, write_kernel_sequences
from .profiler import nsys_capture, vllm_server
from .profiler.config import ProfileConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


def _successful_replay_request_ids(replay_jsonl: Path) -> set[str]:
    """Return the exact TraceLab request ids submitted successfully to vLLM."""
    request_ids: set[str] = set()
    for line_number, line in enumerate(replay_jsonl.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        outcome = row.get("outcome")
        if not isinstance(outcome, dict) or outcome.get("status") != "SUCCESS":
            continue
        request_id = outcome.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError(
                f"successful replay row {line_number} has no non-empty outcome.request_id"
            )
        if request_id in request_ids:
            raise ValueError(f"duplicate successful replay request id {request_id!r}")
        request_ids.add(request_id)
    return request_ids


def _resolve_fork_python(cfg: ProfileConfig) -> str:
    default_fork = REPO_ROOT / "alignment/profiler/vllm/.venv/bin/python"
    fork = Path(cfg.fork_python) if cfg.fork_python else default_fork
    if not fork.is_absolute():
        fork = REPO_ROOT / fork
    # normpath (not resolve) so the venv's python symlink is kept — resolving it
    # would point at the bare uv interpreter and lose the venv's site-packages.
    fork = Path(os.path.normpath(fork))
    if not fork.exists():
        raise FileNotFoundError(
            f"vLLM python not found: {fork}\n"
            "Point `fork_python` at a vLLM venv, or build one — "
            "see alignment/profiler/README.md."
        )
    return str(fork)


def run_profile(cfg: ProfileConfig) -> dict:
    """Run one explicit measured pass: NSYS, workload timing, or popularity."""
    log_dir = Path(cfg.log_dir)
    if not log_dir.is_absolute():
        log_dir = REPO_ROOT / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    # Build and normalize before nsys starts. This avoids capturing Rust compile
    # work and ensures a bad trace fails before allocating vLLM GPU memory.
    load_generator.build_session_runner()
    prepared_replay = load_generator.prepare_replay(cfg.workload, log_dir)
    fork_python = _resolve_fork_python(cfg)
    vllm_dir = log_dir / "vllm"
    nsys_dir = log_dir / "nsys"
    vllm_dir.mkdir(parents=True, exist_ok=True)
    nsys_dir.mkdir(parents=True, exist_ok=True)

    server_argv = vllm_server.build_server_argv(fork_python, cfg.server)
    is_expert_popularity = cfg.profile_kind == "expert_popularity"
    is_workload_metrics = cfg.profile_kind == "workload_metrics"
    is_nsys = cfg.profile_kind == "nsys"
    if not (is_nsys or is_expert_popularity or is_workload_metrics):
        raise ValueError(f"unsupported profile_kind: {cfg.profile_kind!r}")
    if is_nsys and cfg.nsys.capture_mode == "cuda_profiler_api":
        # This enables vLLM's /start_profile and /stop_profile routes. Those
        # routes fan CUDA profiler control into the actual GPU worker, which is
        # the reliable targeted-capture boundary for the spawned EngineCore.
        server_argv.append("--profiler-config.profiler=cuda")
    env = vllm_server.build_server_env(fork_python, cfg.cuda_visible_devices)
    if is_expert_popularity:
        # This pass measures routing counts, not phase timing.  NVTX construction
        # and NSYS are disabled so its deliberate EPLB all-reduce/D2H logging
        # overhead cannot be confused with the timing pass.
        env["VLLM_NVTX_SCOPES_FOR_PROFILING"] = "0"
    # Only the timing pass runs under NSYS. The other passes launch the same
    # server argv bare so profiler lifecycle work cannot enter their evidence.
    out_rep = nsys_dir / cfg.name
    nsys_executable = nsys_capture.resolve_nsys_executable(cfg.nsys.executable) if is_nsys else None
    full_argv = (
        nsys_capture.build_nsys_prefix(nsys_executable, cfg.nsys, out_rep) + server_argv
        if nsys_executable is not None
        else server_argv
    )
    vllm_server.write_launch_metadata(
        vllm_dir / f"{cfg.name}_launch.json",
        full_argv,
        server_argv,
        env,
        cfg,
        fork_python,
        nsys_executable.provenance() if nsys_executable is not None else None,
    )

    server_log = vllm_dir / f"{cfg.name}_server.log"
    base_url = f"http://{cfg.server.host}:{cfg.server.port}"
    model = cfg.server.served_model_name or cfg.server.model_path

    mode_by_kind = {
        "nsys": "external nsys",
        "workload_metrics": "bare workload metrics",
        "expert_popularity": "bare expert popularity",
    }
    mode = mode_by_kind[cfg.profile_kind]
    print(f"[profile] launching ({mode}): {' '.join(full_argv[:6])} ... (log: {server_log})")
    with server_log.open("w") as log_fh:
        proc = subprocess.Popen(
            full_argv,
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
            start_new_session=True,
        )
        drive_summary: dict = {}
        cuda_profile_active = False
        capture_timer_cancelled = threading.Event()
        capture_timer: threading.Thread | None = None
        capture_timer_errors: list[BaseException] = []
        try:
            vllm_server.wait_for_ready(base_url, proc, cfg.server.startup_timeout)
            if is_nsys and cfg.nsys.capture_mode == "cuda_profiler_api":
                vllm_server.set_cuda_profile(base_url, active=True)
                cuda_profile_active = True
                if cfg.nsys.capture_duration_seconds is not None:

                    def stop_bounded_capture() -> None:
                        nonlocal cuda_profile_active
                        if capture_timer_cancelled.wait(cfg.nsys.capture_duration_seconds):
                            return
                        try:
                            vllm_server.set_cuda_profile(base_url, active=False)
                        except BaseException as error:
                            capture_timer_errors.append(error)
                        else:
                            cuda_profile_active = False

                    capture_timer = threading.Thread(
                        target=stop_bounded_capture,
                        name="alignment-nsys-capture-timer",
                    )
                    capture_timer.start()
            print("[profile] server ready; driving workload")
            drive_summary = load_generator.run_replay(
                cfg.workload, prepared_replay, base_url=base_url, model=model
            )
            capture_timer_cancelled.set()
            if capture_timer is not None:
                capture_timer.join()
            if capture_timer_errors:
                raise RuntimeError("bounded NSYS capture stop failed") from capture_timer_errors[0]
            drive_summary["reached_idle"] = vllm_server.wait_for_idle(base_url, cfg.idle)
            print(f"[profile] workload done: {drive_summary}")
            if cuda_profile_active:
                vllm_server.set_cuda_profile(base_url, active=False)
                cuda_profile_active = False
            time.sleep(2)  # let the last iterations' kernels flush into the trace
        finally:
            capture_timer_cancelled.set()
            if capture_timer is not None:
                capture_timer.join()
            if cuda_profile_active:
                # Preserve the original workload error if the emergency stop
                # also fails; `_shutdown` still lets nsys finalize its report.
                try:
                    vllm_server.set_cuda_profile(base_url, active=False)
                except RuntimeError:
                    pass
            _shutdown(proc)

    metrics_jsonl = vllm_dir / f"{cfg.name}_metrics.jsonl"
    n_metrics = vllm_server.extract_metrics_jsonl(server_log, metrics_jsonl)

    if is_expert_popularity:
        # This pass deliberately disables VLLM_NVTX_SCOPES_FOR_PROFILING, which
        # gates both NVTX ranges and EngineCore request timing records in the
        # instrumented fork. Request timing belongs to the clean NSYS pass;
        # requiring it here would reject an otherwise valid popularity capture.
        expert_load_jsonl = vllm_dir / f"{cfg.name}_expert_load.jsonl"
        expert_popularity_json = log_dir / "expert_popularity.json"
        expert_parallel_size = cfg.server.tp_size * cfg.server.dp_size
        expert_record_count = vllm_server.extract_expert_popularity(
            server_log,
            expert_load_jsonl,
            expert_popularity_json,
            expert_parallel_size=expert_parallel_size,
        )
        result = {
            "profile_kind": cfg.profile_kind,
            "log_dir": str(log_dir),
            "metrics_jsonl": str(metrics_jsonl),
            "expert_load_jsonl": str(expert_load_jsonl),
            "expert_popularity_json": str(expert_popularity_json),
            "expert_record_count": expert_record_count,
            "server_log": str(server_log),
            "gpu": cfg.gpu,
            "server_tp_size": cfg.server.tp_size,
            "server_dp_size": cfg.server.dp_size,
            "cuda_visible_devices": cfg.cuda_visible_devices,
            "replay_result": str(prepared_replay.log_path.resolve()),
            "drive_summary": drive_summary,
        }
        (log_dir / "profile_result.json").write_text(json.dumps(result, indent=2))
        print(
            f"[profile] expert popularity: records={expert_record_count} "
            f"artifact={expert_popularity_json.name}"
        )
        return result

    request_timings_jsonl = vllm_dir / f"{cfg.name}_request_timings.jsonl"
    successful_request_ids = _successful_replay_request_ids(prepared_replay.log_path)
    n_request_timings = vllm_server.extract_request_timings_jsonl(
        server_log,
        request_timings_jsonl,
        expected_request_ids=successful_request_ids,
    )

    if is_workload_metrics:
        result = {
            "profile_kind": cfg.profile_kind,
            "log_dir": str(log_dir),
            "metrics_jsonl": str(metrics_jsonl),
            "request_timings_jsonl": str(request_timings_jsonl),
            "request_timing_count": n_request_timings,
            "server_log": str(server_log),
            "gpu": cfg.gpu,
            "server_tp_size": cfg.server.tp_size,
            "server_dp_size": cfg.server.dp_size,
            "cuda_visible_devices": cfg.cuda_visible_devices,
            "replay_result": str(prepared_replay.log_path.resolve()),
            "drive_summary": drive_summary,
        }
        (log_dir / "profile_result.json").write_text(json.dumps(result, indent=2))
        print(
            f"[profile] workload metrics: iterations={n_metrics} "
            f"request_timings={n_request_timings}"
        )
        return result

    assert nsys_executable is not None
    rep_alt = out_rep.with_suffix(".nsys-rep")
    rep = rep_alt if rep_alt.exists() else out_rep
    sqlite_path = nsys_capture.export_sqlite(nsys_executable, rep)
    validation = nsys_capture.validate_export(sqlite_path)
    print(
        f"[profile] export ok: sqlite={sqlite_path.name} "
        f"metrics_iters={n_metrics} request_timings={n_request_timings} "
        f"validation={validation}"
    )
    if not validation["ok"]:
        print("[warn] export validation failed — capture window may have missed target iterations")

    parsed_path = log_dir / "parsed.json"
    parsed = parse_trace(
        sqlite_path,
        metrics_jsonl,
        cfg.nsys.analyze_iteration_start,
        cfg.nsys.analyze_iteration_end,
        range_mode="phases",
    )
    expected_device_count = cfg.server.tp_size * cfg.server.dp_size
    if len(parsed["device_ids"]) != expected_device_count:
        raise RuntimeError(
            "normalized NSYS device population does not match server parallelism: "
            f"devices={parsed['device_ids']} tp_size={cfg.server.tp_size} "
            f"dp_size={cfg.server.dp_size}"
        )
    parsed_path.write_text(json.dumps(parsed, indent=2))
    kernel_sequences_path = log_dir / "kernel_sequences.json"
    write_kernel_sequences(kernel_sequences_path, parsed, parsed_path)
    print(
        f"[profile] parsed {len(parsed['iterations'])} iteration(s), "
        f"{parsed['scanned_kernel_rows']} kernel row(s) → {parsed_path.name}"
    )

    # The host side of the same window. Written as a sidecar rather than into
    # parsed.json because no kernel-attribution consumer reads a single row of
    # it, and it is comparable in size to parsed.json itself.
    host_timeline_path = log_dir / "host_timeline.json"
    window_start_ns, window_end_ns = parsed_window_ns(parsed)
    host_timeline = parse_host_timeline(sqlite_path, window_start_ns, window_end_ns)
    host_timeline_path.write_text(json.dumps(host_timeline, separators=(",", ":")))
    print(
        f"[profile] host {len(host_timeline['threads'])} thread(s), "
        f"{len(host_timeline['nvtx_ranges'])} nvtx range(s), "
        f"{len(host_timeline['api_calls'])} api call(s) → {host_timeline_path.name}"
    )

    result = {
        "profile_kind": cfg.profile_kind,
        # Resolved artifact root is the launcher→analyzer handoff. Keep it in the
        # result instead of making the launcher duplicate relative-path semantics.
        "log_dir": str(log_dir),
        "sqlite": str(sqlite_path),
        "metrics_jsonl": str(metrics_jsonl),
        "request_timings_jsonl": str(request_timings_jsonl),
        "request_timing_count": n_request_timings,
        "parsed_nsys": str(parsed_path),
        "kernel_sequences": str(kernel_sequences_path),
        "host_timeline": str(host_timeline_path),
        "server_log": str(server_log),
        "gpu": cfg.gpu,
        "server_tp_size": cfg.server.tp_size,
        "server_dp_size": cfg.server.dp_size,
        "cuda_visible_devices": cfg.cuda_visible_devices,
        "parsed_device_ids": parsed["device_ids"],
        "replay_result": str(prepared_replay.log_path.resolve()),
        "drive_summary": drive_summary,
        "validation": validation,
        "nsys_profiler": nsys_executable.provenance(),
    }
    (log_dir / "profile_result.json").write_text(json.dumps(result, indent=2))
    return result


def _shutdown(proc: subprocess.Popen) -> None:
    """SIGINT the process group (lets nsys finalize), escalate if it lingers."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
