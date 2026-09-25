"""Coordinate one measured alignment run across req-frontend, a serving engine, and Nsight.

The explicit `launcher alignment profile <profiling-config>` command calls this
module. It prepares req-frontend, launches one instrumented-fork server under nsys,
drives the workload, and exports the trace into the profiling config's
`log_dir`. It does not generate timing-predict inputs or run comparison analysis.

`cfg.engine` picks the launch driver -- `vllm_server` or `sglang_server`. Both
expose the same module-level surface (argv, env, launch metadata, ready/profile/
idle polling, worker ranks), and both forks emit the identical alignment records,
so everything after the capture is shared: this module names no engine below the
point where the driver and its `EngineRecords` descriptor are chosen.

The server is spawned in its own session so a SIGINT to the group lets nsys
finalize the `.nsys-rep` cleanly on shutdown.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

from .load_generator import runner as load_generator
from .nsys.parse import parse_host_timeline, parse_trace, parsed_window_ns, write_kernel_sequences
from .nsys.parsed_io import write_parsed
from .profiler import (
    nsys_capture,
    record_extraction,
    runtime_artifacts,
    sglang_server,
    spec_decode,
    token_corpus,
    vllm_server,
)
from .profiler.config import PROFILE_KINDS, ROUTING_PROFILE_KINDS, ProfileConfig
from .profiler.engine_records import SGLANG_RECORDS, VLLM_RECORDS

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Where a `token_corpus` replay persists one `.npy` of routed experts per
#: request, and where the packed corpus lands. Both are under the pass's own
#: `log_dir`, and the replay clears the first one so a re-run cannot pack a
#: previous capture's requests alongside its own.
ROUTED_EXPERTS_DIR = "routed_experts"
TOKEN_CORPUS_DIR = "token_corpus"

# One entry per supported engine: the launch driver, the log dialect its records
# arrive in, and the venv python a config that names no `fork_python` defaults to.
_ENGINES = {
    "vllm": (vllm_server, VLLM_RECORDS, "alignment/profiler/vllm/.venv/bin/python"),
    "sglang": (
        sglang_server,
        SGLANG_RECORDS,
        "alignment/profiler/sglang/python/.venv-sglang/bin/python",
    ),
}


def _successful_replay_request_ids(replay_jsonl: Path) -> set[str]:
    """Return the exact req-frontend request ids submitted successfully to vLLM."""
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


def _engine(cfg: ProfileConfig):
    """The launch driver, record dialect, and default venv for `cfg.engine`."""
    try:
        return _ENGINES[cfg.engine]
    except KeyError:
        raise ValueError(
            f"unsupported engine: {cfg.engine!r}; expected one of {sorted(_ENGINES)}"
        ) from None


def _resolve_fork_python(cfg: ProfileConfig) -> str:
    _, _, default_relative = _engine(cfg)
    fork = Path(cfg.fork_python) if cfg.fork_python else REPO_ROOT / default_relative
    if not fork.is_absolute():
        fork = REPO_ROOT / fork
    # normpath (not resolve) so the venv's python symlink is kept — resolving it
    # would point at the bare uv interpreter and lose the venv's site-packages.
    fork = Path(os.path.normpath(fork))
    if not fork.exists():
        raise FileNotFoundError(
            f"{cfg.engine} python not found: {fork}\n"
            f"Point `fork_python` at a {cfg.engine} venv, or build one — "
            "see alignment/profiler/README.md."
        )
    return str(fork)


def _extract_expert_load(
    cfg: ProfileConfig,
    measurement_log: Path,
    engine_dir: Path,
    log_dir: Path,
    records,
    *,
    speculative: bool,
    window: dict,
) -> dict:
    """EPLB's per-step expert-load stream, and the marginal aggregated from it.

    Required whenever the pass asked the engine for the stream, and whenever
    the marginal is the pass's own product. Returns `{}` only for the case that
    can produce neither: a corpus capture on a deployment with no expert
    parallelism, which still records the routes themselves in logical expert
    ids and needs no topology to describe them.
    """
    required = cfg.captures_expert_load or cfg.profile_kind == "expert_popularity"
    expert_parallel_size = cfg.server.expert_parallel_size
    reduction_group_size = cfg.server.expert_count_reduction_group_size
    if expert_parallel_size is None or reduction_group_size is None:
        if required:
            raise ValueError(
                f"{cfg.profile_kind} requires explicit server.expert_parallel_size and "
                "server.expert_count_reduction_group_size"
            )
        return {}

    shared = {
        "expert_parallel_size": expert_parallel_size,
        "reduction_group_size": reduction_group_size,
        "max_tokens_per_step": max(
            cfg.server.chunk_size,
            cfg.server.max_cudagraph_capture_size or 0,
        ),
        "dp_size": cfg.server.dp_size,
        "records": records,
        **window,
    }
    expert_load_jsonl = engine_dir / f"{cfg.name}_expert_load.jsonl"
    expert_popularity_json = log_dir / "expert_popularity.json"
    record_count = record_extraction.extract_expert_popularity(
        measurement_log,
        expert_load_jsonl,
        expert_popularity_json,
        model_role="target" if speculative else None,
        required=required,
        **shared,
    )
    if record_count is None:
        # The extractor opens its output before it can know the server logged
        # anything; an empty stream is litter, not an artifact.
        expert_load_jsonl.unlink(missing_ok=True)
        return {}

    artifacts = {
        "expert_load_jsonl": str(expert_load_jsonl),
        "expert_popularity_json": str(expert_popularity_json),
        "expert_record_count": record_count,
    }
    if speculative:
        # A drafter with experts routes over its own and is logged under its
        # own role, so it is a second population of the same stream, not a
        # slice. An n-gram or other expert-free drafter logs nothing, and no
        # consumer requires the draft marginal, so its absence is not an error.
        draft_load_jsonl = engine_dir / f"{cfg.name}_draft_expert_load.jsonl"
        draft_popularity_json = log_dir / "draft_expert_popularity.json"
        draft_count = record_extraction.extract_expert_popularity(
            measurement_log,
            draft_load_jsonl,
            draft_popularity_json,
            model_role="draft",
            required=False,
            **shared,
        )
        if draft_count is None:
            draft_load_jsonl.unlink(missing_ok=True)
        else:
            artifacts.update(
                draft_expert_load_jsonl=str(draft_load_jsonl),
                draft_expert_popularity_json=str(draft_popularity_json),
                draft_expert_record_count=draft_count,
            )
    return artifacts


def _server_option(extra_args: list[str], flag: str) -> str | None:
    """The value of `flag` in a server argv, in either argparse spelling."""
    value = None
    for index, arg in enumerate(extra_args):
        if arg == flag and index + 1 < len(extra_args):
            value = extra_args[index + 1]
        elif arg.startswith(f"{flag}="):
            value = arg.split("=", 1)[1]
    return value


def _checkpoint_config(server) -> Path:
    """The `config.json` of the checkpoint the server loaded.

    vLLM reads it from `--hf-config-path` when given and otherwise from
    `--model`, either a local directory or a hub repo id at `--revision`, which
    the server has already fetched into the cache.
    """
    source = _server_option(server.extra_args, "--hf-config-path") or server.model_path
    local = Path(source) / "config.json"
    if local.is_file():
        return local
    from huggingface_hub import hf_hub_download

    revision = _server_option(server.extra_args, "--revision")
    return Path(hf_hub_download(repo_id=source, filename="config.json", revision=revision))


def _checkpoint_text_config(cfg: ProfileConfig) -> tuple[Path, dict]:
    config_path = _checkpoint_config(cfg.server)
    document = json.loads(config_path.read_text())
    return config_path, document.get("text_config", document)


def _num_target_layers(cfg: ProfileConfig) -> int:
    """The checkpoint's own decoder depth, past which a capture holds MTP slots."""
    config_path, text_config = _checkpoint_text_config(cfg)
    value = text_config.get("num_hidden_layers")
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{config_path} names no num_hidden_layers")
    return value


def _num_routed_experts(cfg: ProfileConfig) -> int:
    """How many experts the checkpoint routes over, read from the checkpoint.

    The corpus needs this to range-check the ids it packs. Taking it from the
    model rather than from the expert-popularity marginal this pass also writes
    keeps the two artifacts independent: a corpus is a recording of the model,
    not of the other artifact, and one produced without the marginal must still
    be describable.
    """
    config_path, text_config = _checkpoint_text_config(cfg)
    for key in ("num_experts", "n_routed_experts", "num_local_experts"):
        value = text_config.get(key)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError(
        f"{config_path} names no routed expert count "
        "(num_experts / n_routed_experts / num_local_experts)"
    )


def _prepared_routes_dir(cfg: ProfileConfig, log_dir: Path) -> Path | None:
    """An empty directory for this replay's routes, or None for a pass with none.

    Emptied rather than merely created. The packer concatenates every `.npy` it
    finds, so a re-run into the same log directory -- or a run whose request set
    shrank -- would otherwise pack a previous capture's requests into a corpus
    that then checksums and provenances as if it were one recording.
    """
    if cfg.profile_kind != "token_corpus":
        return None
    routes = log_dir / ROUTED_EXPERTS_DIR
    shutil.rmtree(routes, ignore_errors=True)
    routes.mkdir(parents=True)
    return routes


# What a pass needs the engine to do, beyond what the profile config asks for.
# A capture pass should be one command: the operator names the kind, and the
# flags that kind cannot work without are this module's business, not theirs.
_PROFILE_KIND_SERVER_ARGS = {"token_corpus": (("--enable-return-routed-experts", None),)}

# EPLB's balancedness log is the only per-step view of what the engine actually
# ran: one rank-synchronized expert-load record per forward. A corpus says what
# a *token* routes to; that stream says what a *step* routed to, which is what
# a sampled fold has to be scored against. Both routing passes ask for it, so
# one capture yields the corpus, the referee, and the marginal. It is vLLM's
# only expert-load source and it requires expert parallelism, so it is asked
# for only when the deployment already has it. `rearrange: false` keeps it a
# recorder: a rearranging EPLB allocates a layer of expert weights per model as
# a transfer buffer and rehearses a transfer in `profile_run`, which on GLM-5.2
# MTP-5 TP4 cost ~16 GiB per GPU and left the timed passes' memory budget no KV
# cache at all. It would also move experts mid-capture.
_EXPERT_LOAD_SERVER_ARGS = (
    ("--enable-eplb", None),
    ("--eplb-config", '{"log_balancedness": true, "rearrange": false}'),
)


def _append_backend_server_args(server_argv: list[str], cfg: ProfileConfig) -> None:
    """Add backend- and pass-required server flags exactly once to the launch argv."""
    for argument in cfg.workload.backend.required_server_args:
        if argument not in server_argv:
            server_argv.append(argument)
    if cfg.profile_kind not in ROUTING_PROFILE_KINDS:
        return
    options = list(_PROFILE_KIND_SERVER_ARGS.get(cfg.profile_kind, ()))
    if cfg.captures_expert_load:
        options += _EXPERT_LOAD_SERVER_ARGS
    for flag, value in options:
        # argparse accepts both `--flag value` and `--flag=value`.
        present = [
            i for i, arg in enumerate(server_argv) if arg == flag or arg.startswith(f"{flag}=")
        ]
        if not present:
            server_argv.append(flag)
            if value is not None:
                server_argv.append(value)
            continue
        if value is None:
            continue
        # A valued option here carries a JSON object, which is a set of settings
        # rather than a switch. A preset that tuned one of them must not
        # silently drop the setting the pass cannot work without, so the two are
        # merged and the pass's own setting wins.
        index = present[-1]
        if server_argv[index] == flag:
            if index + 1 >= len(server_argv):
                raise ValueError(f"{flag} in server args carries no value")
            authored = server_argv.pop(index + 1)
        else:
            authored = server_argv[index].split("=", 1)[1]
        merged = {**json.loads(authored), **json.loads(value)}
        server_argv[index] = f"{flag}={json.dumps(merged, separators=(',', ':'))}"


def _preflight_capture_environment(
    driver,
    fork_python: str | None,
    env: dict[str, str],
    *,
    profile_kind: str,
    resume: bool,
) -> None:
    """Validate dependencies only when a new NSYS capture will be launched."""
    if profile_kind != "nsys" or resume:
        return
    if fork_python is None:
        raise ValueError("a new NSYS capture requires a serving-engine Python")
    driver.validate_nsys_capture_environment(fork_python, env=env, cwd=REPO_ROOT)


def run_profile(cfg: ProfileConfig, *, resume: bool = False) -> dict:
    """Run one explicit measured pass: NSYS, workload timing, or popularity.

    `resume=True` reuses an existing capture and redoes only the cheap
    post-capture work (record extraction, NSYS normalization, artifact
    manifest). The GPU capture is the expensive, non-reproducible part of this
    pipeline; a failure in extraction or parsing must never cost a re-capture.
    """
    driver, _, _ = _engine(cfg)
    if cfg.workload is not None and cfg.workload.warmup and (
        cfg.engine != "vllm"
        or (cfg.profile_kind == "nsys" and cfg.nsys.capture_mode != "cuda_profiler_api")
    ):
        raise ValueError("warmup requires vLLM and cuda_profiler_api for NSYS captures")
    if (
        cfg.workload is not None and cfg.workload.warmup
        and not cfg.server.enable_server_load_tracking
    ):
        raise ValueError("warmup requires server.enable_server_load_tracking")
    if (
        not resume
        and cfg.engine == "sglang"
        and (
            cfg.python_runtime is None
            or cfg.python_runtime.environment.get("FLASHINFER_DISABLE_JIT") != "1"
        )
    ):
        raise ValueError(
            "a new SGLang profile requires an explicit python_runtime "
            "prebuilt-artifact contract with FLASHINFER_DISABLE_JIT=1"
        )
    log_dir = Path(cfg.log_dir)
    if not log_dir.is_absolute():
        log_dir = REPO_ROOT / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    prepared_replay = load_generator.prepare_replay(cfg.workload, log_dir)
    if not resume:
        # Build and normalize before nsys starts. This avoids capturing Rust
        # compile work and ensures a bad trace fails before allocating vLLM GPU
        # memory.
        load_generator.build_session_runner()
    engine_dir = log_dir / cfg.engine
    nsys_dir = log_dir / "nsys"
    engine_dir.mkdir(parents=True, exist_ok=True)
    nsys_dir.mkdir(parents=True, exist_ok=True)

    # A resumed pass never launches the engine, so it must not require the fork
    # venv either — the capture it reuses is the evidence.
    fork_python = None if resume else _resolve_fork_python(cfg)
    server_argv = [] if resume else driver.build_server_argv(fork_python, cfg.server)
    if not resume:
        _append_backend_server_args(server_argv, cfg)
    is_routing = cfg.profile_kind in ROUTING_PROFILE_KINDS
    is_nsys = cfg.profile_kind == "nsys"
    if cfg.profile_kind not in PROFILE_KINDS:
        raise ValueError(f"unsupported profile_kind: {cfg.profile_kind!r}")
    if is_nsys and not resume:
        server_argv += list(driver.NSYS_CAPTURE_SERVER_ARGS)
    if is_nsys and cfg.nsys.capture_mode == "cuda_profiler_api" and not resume:
        server_argv += list(driver.CUDA_PROFILER_SERVER_ARGS)
    runtime_provenance = (
        None
        if resume
        else runtime_artifacts.prepare_python_runtime(
            fork_python,
            cfg.python_runtime,
        )
    )
    env = (
        {}
        if resume
        else driver.build_server_env(
            fork_python,
            cfg.cuda_visible_devices,
            driver_compat_lib_dir=cfg.driver_compat_lib_dir,
        )
    )
    if not resume and cfg.python_runtime is not None:
        env.update(cfg.python_runtime.environment)
    if not resume and cfg.workload.warmup:
        # vLLM exposes reset_prefix_cache through its development router.
        env["VLLM_SERVER_DEV_MODE"] = "1"
    if is_routing and not resume:
        # These passes measure routing, not phase timing.  NVTX construction and
        # NSYS are disabled so their deliberate EPLB all-reduce/D2H logging
        # overhead cannot be confused with the timing pass.
        env.update(driver.TIMING_INSTRUMENTATION_OFF_ENV)
    _preflight_capture_environment(
        driver,
        fork_python,
        env,
        profile_kind=cfg.profile_kind,
        resume=resume,
    )
    # Only the timing pass runs under NSYS. The other passes launch the same
    # server argv bare so profiler lifecycle work cannot enter their evidence.
    out_rep = nsys_dir / cfg.name
    nsys_executable = (
        nsys_capture.resolve_nsys_executable(cfg.nsys.executable)
        if is_nsys and not resume
        else None
    )
    server_log = engine_dir / f"{cfg.name}_server.log"
    drive_summary_path = engine_dir / f"{cfg.name}_drive_summary.json"

    if resume:
        if not server_log.is_file():
            raise FileNotFoundError(
                f"cannot resume: server log not found: {server_log}; run the capture first"
            )
        # Rebuilt from the config and the prepared replay, exactly as the live
        # path derives it. `reached_idle` is a liveness observation of the
        # finished server and is deliberately left out rather than invented.
        drive_summary = {
            "source_trace": str(prepared_replay.trace_path.resolve()),
            "frontend_type": cfg.workload.frontend.type,
            "backend_type": cfg.workload.backend.type,
            "log_path": str(prepared_replay.log_path),
            "summary_path": str(prepared_replay.summary_path),
            "resumed_from_existing_capture": True,
        }
        if drive_summary_path.is_file():
            drive_summary = {
                **json.loads(drive_summary_path.read_text()),
                "resumed_from_existing_capture": True,
            }
        print(f"[profile] resuming from existing capture (log: {server_log})")
        return _finalize_profile(
            cfg,
            log_dir=log_dir,
            engine_dir=engine_dir,
            server_log=server_log,
            out_rep=out_rep,
            prepared_replay=prepared_replay,
            drive_summary=drive_summary,
            nsys_executable=None,
        )

    full_argv = (
        nsys_capture.build_nsys_prefix(nsys_executable, cfg.nsys, out_rep) + server_argv
        if nsys_executable is not None
        else server_argv
    )
    driver.write_launch_metadata(
        engine_dir / f"{cfg.name}_launch.json",
        full_argv,
        server_argv,
        env,
        cfg,
        fork_python,
        nsys_executable.provenance() if nsys_executable is not None else None,
        runtime_provenance,
    )

    base_url = f"http://{cfg.server.host}:{cfg.server.port}"
    model = cfg.server.served_model_name or cfg.server.model_path

    mode_by_kind = {
        "nsys": "external nsys",
        "workload_metrics": "bare workload metrics",
        "expert_popularity": "bare expert popularity",
        "token_corpus": "bare token corpus",
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
            driver.wait_for_ready(base_url, proc, cfg.server.startup_timeout)
            speculative = driver.speculative_decode_enabled(cfg.server)
            counters_before = None
            replay_start_monotonic_ns = None
            measurement_log_offset = None

            def measurement_ready() -> None:
                nonlocal cuda_profile_active, capture_timer, counters_before
                nonlocal replay_start_monotonic_ns, measurement_log_offset
                if not driver.wait_for_idle(base_url, cfg.idle):
                    raise RuntimeError("server did not drain before measurement")
                measurement_log_offset = server_log.stat().st_size
                if is_nsys and cfg.nsys.capture_mode == "cuda_profiler_api":
                    driver.set_cuda_profile(base_url, active=True)
                    cuda_profile_active = True
                if cuda_profile_active and cfg.nsys.capture_duration_seconds is not None:

                    def stop_bounded_capture() -> None:
                        nonlocal cuda_profile_active
                        if capture_timer_cancelled.wait(cfg.nsys.capture_duration_seconds):
                            return
                        try:
                            driver.set_cuda_profile(base_url, active=False)
                        except BaseException as error:
                            capture_timer_errors.append(error)
                        else:
                            cuda_profile_active = False

                    capture_timer = threading.Thread(
                        target=stop_bounded_capture,
                        name="alignment-nsys-capture-timer",
                    )
                    capture_timer.start()
                counters_before = (
                    driver.fetch_spec_decode_metrics(base_url) if speculative else None
                )
                replay_start_monotonic_ns = time.monotonic_ns()
                print("[profile] frontend ready; measurement begins", flush=True)

            print("[profile] server ready; driving workload")
            drive_summary = load_generator.run_replay(
                cfg.workload, prepared_replay, base_url=base_url, model=model,
                measurement_ready=measurement_ready,
                # Only the corpus pass needs per-token routes, and it pays for
                # them with the replay's streaming timeline. An expert_popularity
                # pass reads the server's own counters and keeps its timeline.
                routed_experts_dir=_prepared_routes_dir(cfg, log_dir),
            )
            replay_end_monotonic_ns = time.monotonic_ns()
            # EngineCore metrics use the same host CLOCK_MONOTONIC domain. The
            # explicit window lets workload alignment exclude server-startup
            # and prefix-cache preflight iterations without consulting NSYS.
            drive_summary["replay_start_monotonic_ns"] = replay_start_monotonic_ns
            drive_summary["replay_end_monotonic_ns"] = replay_end_monotonic_ns
            drive_summary["measurement_log_offset"] = measurement_log_offset
            capture_timer_cancelled.set()
            if capture_timer is not None:
                capture_timer.join()
            if capture_timer_errors:
                raise RuntimeError("bounded NSYS capture stop failed") from capture_timer_errors[0]
            drive_summary["reached_idle"] = driver.wait_for_idle(base_url, cfg.idle)
            if speculative:
                if not drive_summary["reached_idle"]:
                    raise RuntimeError(
                        "spec-decode replay did not reach idle; counters are incomplete"
                    )
                drive_summary["spec_decode_metrics_before"] = counters_before
                drive_summary["spec_decode_metrics_after"] = driver.fetch_spec_decode_metrics(
                    base_url
                )
            drive_summary_path.write_text(json.dumps(drive_summary, indent=2))
            print(f"[profile] workload done: {drive_summary}")
            if cuda_profile_active:
                driver.set_cuda_profile(base_url, active=False)
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
                    driver.set_cuda_profile(base_url, active=False)
                except RuntimeError:
                    pass
            _shutdown(proc)

    return _finalize_profile(
        cfg,
        log_dir=log_dir,
        engine_dir=engine_dir,
        server_log=server_log,
        out_rep=out_rep,
        prepared_replay=prepared_replay,
        drive_summary=drive_summary,
        nsys_executable=nsys_executable,
    )


def _check_routes_cover_replay(summary_path: Path, routes: Path) -> None:
    """Refuse a corpus that silently lost some of the replay's requests.

    The load generator records a request whose response carried no routes as a
    failed step and still exits cleanly, and the packer reads whatever files
    exist. One file per successful step is the only evidence the two agree.
    """
    common = json.loads(Path(summary_path).read_text())["replay"]["common"]
    captured = len(list(routes.glob("*.npy")))
    if common["failed_steps"] or captured != common["success_steps"]:
        raise ValueError(
            f"token_corpus replay: {common['success_steps']} steps succeeded and "
            f"{common['failed_steps']} failed, but {captured} carried routes; a corpus "
            "missing requests would be packed as if it were the whole workload"
        )


def _finalize_profile(
    cfg: ProfileConfig,
    *,
    log_dir: Path,
    engine_dir: Path,
    server_log: Path,
    out_rep: Path,
    prepared_replay,
    drive_summary: dict,
    nsys_executable,
) -> dict:
    """Turn a finished capture into normalized artifacts and the result manifest.

    Split out of `run_profile` so a resume can redo exactly this — every step
    here reads only files the capture already wrote, so it is cheap and
    repeatable, unlike the capture itself.
    """
    is_routing = cfg.profile_kind in ROUTING_PROFILE_KINDS
    is_workload_metrics = cfg.profile_kind == "workload_metrics"
    driver, records, _ = _engine(cfg)

    speculative = driver.speculative_decode_enabled(cfg.server)
    spec_artifacts = {}
    if speculative:
        before = drive_summary.get("spec_decode_metrics_before")
        after = drive_summary.get("spec_decode_metrics_after")
        if before is None or after is None:
            raise ValueError("spec-decode capture requires persisted replay counter snapshots")
        spec_metrics_path = log_dir / "spec_decode_metrics.json"
        spec_metrics_path.write_text(json.dumps(spec_decode.replay_delta(before, after), indent=2))
        spec_artifacts["spec_decode_metrics_json"] = str(spec_metrics_path)

    measurement_log = server_log
    offset = drive_summary.get("measurement_log_offset")
    if cfg.workload.warmup and offset is None:
        raise ValueError("warmup capture is missing its persisted measurement log boundary")
    if offset is not None:
        if not isinstance(offset, int) or not 0 <= offset <= server_log.stat().st_size:
            raise ValueError("invalid measurement log offset")
        measurement_log = engine_dir / f"{cfg.name}_measurement.log"
        with server_log.open("rb") as source, measurement_log.open("wb") as target:
            source.seek(offset)
            shutil.copyfileobj(source, target)
    metrics_jsonl = engine_dir / f"{cfg.name}_metrics.jsonl"
    n_metrics = record_extraction.extract_metrics_jsonl(
        measurement_log, metrics_jsonl, dp_size=cfg.server.dp_size, records=records
    )

    if is_routing:
        # These passes deliberately disable the engine's timing instrumentation.
        # Request timing belongs to the clean NSYS pass; requiring it here
        # would reject an otherwise valid popularity capture.
        expert_load_artifacts = _extract_expert_load(
            cfg,
            measurement_log,
            engine_dir,
            log_dir,
            records,
            speculative=speculative,
            window={
                "replay_start_monotonic_ns": drive_summary.get("replay_start_monotonic_ns"),
                "replay_end_monotonic_ns": drive_summary.get("replay_end_monotonic_ns"),
            }
            if speculative
            else {},
        )
        corpus_artifacts = {}
        if cfg.profile_kind == "token_corpus":
            # Packed from the routes the replay persisted, and range-checked
            # against the checkpoint's own expert count.
            _check_routes_cover_replay(prepared_replay.summary_path, log_dir / ROUTED_EXPERTS_DIR)
            manifest = token_corpus.pack_token_corpus(
                log_dir / ROUTED_EXPERTS_DIR,
                log_dir / TOKEN_CORPUS_DIR,
                num_experts=_num_routed_experts(cfg),
                num_target_layers=_num_target_layers(cfg),
            )
            corpus_artifacts = {
                "token_corpus_manifest": str(log_dir / TOKEN_CORPUS_DIR / "manifest.json"),
                "token_corpus_tokens": manifest["num_tokens"],
                "token_corpus_layers": manifest["num_layers"],
            }
        result = {
            **spec_artifacts,
            **corpus_artifacts,
            **expert_load_artifacts,
            "profile_kind": cfg.profile_kind,
            "engine": cfg.engine,
            "log_dir": str(log_dir),
            "metrics_jsonl": str(metrics_jsonl),
            "server_log": str(server_log),
            "gpu": cfg.gpu,
            "server_tp_size": cfg.server.tp_size,
            "server_dp_size": cfg.server.dp_size,
            "cuda_visible_devices": cfg.cuda_visible_devices,
            "replay_result": str(prepared_replay.log_path.resolve()),
            "drive_summary": drive_summary,
        }
        (log_dir / "profile_result.json").write_text(json.dumps(result, indent=2))
        produced = []
        if corpus_artifacts:
            produced.append(f"corpus_tokens={corpus_artifacts['token_corpus_tokens']}")
        produced.append(
            f"expert_load_records={expert_load_artifacts['expert_record_count']}"
            if expert_load_artifacts
            else "expert_load=absent"
        )
        print(f"[profile] {cfg.profile_kind}: {' '.join(produced)}")
        return result

    request_timings_jsonl = engine_dir / f"{cfg.name}_request_timings.jsonl"
    successful_request_ids = _successful_replay_request_ids(prepared_replay.log_path)
    n_request_timings = record_extraction.extract_request_timings_jsonl(
        measurement_log,
        request_timings_jsonl,
        expected_request_ids=successful_request_ids,
        records=records,
    )

    if is_workload_metrics:
        result = {
            **spec_artifacts,
            "profile_kind": cfg.profile_kind,
            "engine": cfg.engine,
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

    rep_alt = out_rep.with_suffix(".nsys-rep")
    rep = rep_alt if rep_alt.exists() else out_rep
    exported_sqlite = rep.with_suffix(".sqlite")
    if nsys_executable is None:
        # Resume path: the SQLite export is a pure function of the immutable
        # `.nsys-rep`, so an existing one is reused rather than re-derived.
        if not exported_sqlite.is_file():
            raise FileNotFoundError(
                f"cannot resume: no SQLite export beside {rep}; re-run `alignment profile` "
                "without --resume, or export it with `nsys export --type sqlite`"
            )
        sqlite_path = exported_sqlite
    else:
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
    # Which device ran which DP rank. Engines say this two ways: some state it
    # outright per worker, the rest state pid ↔ rank and leave nsys's pid ↔
    # device knowledge to complete the join. Prefer the direct statement.
    dp_rank_by_device = record_extraction.extract_dp_rank_by_device(server_log, records=records)
    worker_ranks = driver.extract_worker_device_ranks(server_log)
    parsed = parse_trace(
        sqlite_path,
        metrics_jsonl,
        cfg.nsys.analyze_iteration_start,
        cfg.nsys.analyze_iteration_end,
        range_mode="phases",
        worker_ranks=worker_ranks,
        tp_size=cfg.server.tp_size,
        dp_rank_by_device=dp_rank_by_device,
    )
    expected_device_count = cfg.server.tp_size * cfg.server.dp_size
    if len(parsed["device_ids"]) != expected_device_count:
        raise RuntimeError(
            "normalized NSYS device population does not match server parallelism: "
            f"devices={parsed['device_ids']} tp_size={cfg.server.tp_size} "
            f"dp_size={cfg.server.dp_size}"
        )
    observed_dp_ranks = sorted(set(parsed["dp_rank_by_device"].values()))
    if observed_dp_ranks != list(range(cfg.server.dp_size)):
        raise RuntimeError(
            "normalized NSYS DP-rank population does not match server parallelism: "
            f"dp_ranks={observed_dp_ranks} dp_size={cfg.server.dp_size}"
        )
    write_parsed(parsed_path, parsed)
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
        **spec_artifacts,
        "profile_kind": cfg.profile_kind,
        "engine": cfg.engine,
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
        "parsed_dp_rank_by_device": parsed["dp_rank_by_device"],
        "replay_result": str(prepared_replay.log_path.resolve()),
        "drive_summary": drive_summary,
        "validation": validation,
        # The profiler that produced the capture, never the one this process can
        # see. A resume reads it back from the capture's own launch metadata and
        # reports null when that capture predates the field — an unrecorded fact
        # is left unrecorded rather than back-filled from the current host.
        "nsys_profiler": (
            nsys_executable.provenance()
            if nsys_executable is not None
            else _captured_nsys_provenance(engine_dir / f"{cfg.name}_launch.json")
        ),
    }
    (log_dir / "profile_result.json").write_text(json.dumps(result, indent=2))
    return result


def _captured_nsys_provenance(launch_metadata: Path) -> dict | None:
    """The NSYS provenance recorded when the capture ran, if it recorded any."""
    if not launch_metadata.is_file():
        return None
    metadata = json.loads(launch_metadata.read_text())
    provenance = metadata.get("nsys_profiler")
    return provenance if isinstance(provenance, dict) else None


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
