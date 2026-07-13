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
import shutil
import signal
import subprocess
import time
from pathlib import Path

from .load_generator import runner as load_generator
from .nsys.parse import parse_trace, write_kernel_sequences
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
    """Launch vLLM under nsys, drive the workload, and export its SQLite trace."""
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
    if cfg.nsys.capture_mode == "cuda_profiler_api":
        # This enables vLLM's /start_profile and /stop_profile routes. Those
        # routes fan CUDA profiler control into the actual GPU worker, which is
        # the reliable targeted-capture boundary for the spawned EngineCore.
        server_argv.append("--profiler-config.profiler=cuda")
    env = vllm_server.build_server_env(fork_python, cfg.cuda_visible_devices)
    vllm_server.write_launch_metadata(
        vllm_dir / f"{cfg.name}_launch.json", server_argv, env, cfg, fork_python
    )

    # Two capture paths. Ray-nsight: Ray launches the model worker under nsys
    # itself (captures CUDA-graph replay), so we run the server bare and harvest
    # the worker `.nsys-rep` Ray wrote. External nsys: wrap the whole server.
    use_ray = cfg.server.use_ray_nsight
    out_rep = nsys_dir / cfg.name
    if use_ray:
        full_argv = server_argv
    else:
        full_argv = nsys_capture.build_nsys_prefix(cfg.nsys, out_rep) + server_argv

    server_log = vllm_dir / f"{cfg.name}_server.log"
    base_url = f"http://{cfg.server.host}:{cfg.server.port}"
    model = cfg.server.served_model_name or cfg.server.model_path

    mode = "ray-nsight" if use_ray else "external nsys"
    print(f"[profile] launching ({mode}): {' '.join(full_argv[:6])} ... (log: {server_log})")
    launch_ts = time.time()
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
        try:
            vllm_server.wait_for_ready(base_url, proc, cfg.server.startup_timeout)
            if cfg.nsys.capture_mode == "cuda_profiler_api":
                vllm_server.set_cuda_profile(base_url, active=True)
                cuda_profile_active = True
            print("[profile] server ready; driving workload")
            drive_summary = load_generator.run_replay(
                cfg.workload, prepared_replay, base_url=base_url, model=model
            )
            drive_summary["reached_idle"] = vllm_server.wait_for_idle(base_url, cfg.idle)
            print(f"[profile] workload done: {drive_summary}")
            if cuda_profile_active:
                vllm_server.set_cuda_profile(base_url, active=False)
                cuda_profile_active = False
            time.sleep(2)  # let the last iterations' kernels flush into the trace
        finally:
            if cuda_profile_active:
                # Preserve the original workload error if the emergency stop
                # also fails; `_shutdown` still lets nsys finalize its report.
                try:
                    vllm_server.set_cuda_profile(base_url, active=False)
                except RuntimeError:
                    pass
            _shutdown(proc)

    if use_ray:
        # Ray finalizes the worker trace on actor teardown; give it a moment, then
        # harvest the newest worker_process_*.nsys-rep into our nsys dir.
        time.sleep(5)
        ray_rep = nsys_capture.find_ray_nsight_trace(launch_ts)
        rep = out_rep.with_suffix(".nsys-rep")
        shutil.copy2(ray_rep, rep)
        print(f"[profile] harvested ray-nsight trace: {ray_rep} → {rep.name}")
    else:
        rep_alt = out_rep.with_suffix(".nsys-rep")
        rep = rep_alt if rep_alt.exists() else out_rep
    sqlite_path = nsys_capture.export_sqlite(rep)
    metrics_jsonl = vllm_dir / f"{cfg.name}_metrics.jsonl"
    n_metrics = vllm_server.extract_metrics_jsonl(server_log, metrics_jsonl)
    request_timings_jsonl = vllm_dir / f"{cfg.name}_request_timings.jsonl"
    successful_request_ids = _successful_replay_request_ids(prepared_replay.log_path)
    n_request_timings = vllm_server.extract_request_timings_jsonl(
        server_log,
        request_timings_jsonl,
        expected_request_ids=successful_request_ids,
    )
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
    parsed_path.write_text(json.dumps(parsed, indent=2))
    kernel_sequences_path = log_dir / "kernel_sequences.json"
    write_kernel_sequences(kernel_sequences_path, parsed, parsed_path)
    print(
        f"[profile] parsed {len(parsed['iterations'])} iteration(s), "
        f"{parsed['scanned_kernel_rows']} kernel row(s) → {parsed_path.name}"
    )

    result = {
        # Resolved artifact root is the launcher→analyzer handoff. Keep it in the
        # result instead of making the launcher duplicate relative-path semantics.
        "log_dir": str(log_dir),
        "sqlite": str(sqlite_path),
        "metrics_jsonl": str(metrics_jsonl),
        "request_timings_jsonl": str(request_timings_jsonl),
        "request_timing_count": n_request_timings,
        "parsed_nsys": str(parsed_path),
        "kernel_sequences": str(kernel_sequences_path),
        "server_log": str(server_log),
        "gpu": cfg.gpu,
        "server_tp_size": cfg.server.tp_size,
        "replay_result": str(prepared_replay.log_path.resolve()),
        "drive_summary": drive_summary,
        "validation": validation,
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
