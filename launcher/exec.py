"""Execution wiring for build, simulator, and analyzer stages.

Per design §1.2.3 / §1.2.7:
- `cargo_build` is the single shared build per batch; on success it immediately
  runs `simulator list-params` and writes `deployment_schema.json` (INV-8:
  schema discovery is part of the build, not a separate step).
- Every child process enters through ``ProcessSupervisor``. Output is written to
  regular files and completion follows the reaped root PID, never pipe EOF.
- `_build_subprocess_env` wires the PyO3-embedded interpreter (the rust binary
  embeds Python to query profile.db) — ported unchanged from the reference
  launcher, the one piece design §1.2.3 says "survives unchanged".
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import sysconfig
import time
from pathlib import Path

from .process import ProcessResult, ProcessSpec, ProcessSupervisor
from .process.artifacts import (
    ArtifactValidationError,
    validate_alignment_analysis_artifacts,
    validate_json_outputs,
    validate_render_artifacts,
    validate_trace_artifacts,
)
from .process.journal import RunJournal, StageState
from .process.leases import LauncherLeases

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARALLELISM = 200
_PROCESS_SUPERVISOR = ProcessSupervisor()
_LAUNCHER_LEASES = LauncherLeases(REPO_ROOT)


def binary_path(build_type: str = "debug") -> Path:
    return REPO_ROOT / "target" / build_type / "simulator"


def analyzer_binary_path(build_type: str = "debug") -> Path:
    return REPO_ROOT / "target" / build_type / "analyze"


def schema_json_path(build_type: str = "debug") -> Path:
    return REPO_ROOT / "target" / build_type / "deployment_schema.json"


def _build_subprocess_env() -> dict[str, str]:
    """Clean environment for the rust subprocess that embeds Python via PyO3.

    The embedded interpreter must find (a) libpython (LD_LIBRARY_PATH), (b) its
    stdlib (PYTHONHOME = base prefix), and (c) on PYTHONPATH both the repo root
    (so `import profiling...` resolves the repo-local perf_api package, which is
    not installed into site-packages) and the venv site-packages (torch / the
    profiler stack).

    It also carries the measurement policy the launcher resolved. That child is
    a separate process running its own Python, so a policy settled in this one
    does not reach it; the environment is how a spawn configures a child, and
    writing it here -- at the spawn, from the resolved value -- keeps it visible
    rather than a side effect of whoever last called ``set_energy_enabled``.
    """
    from profiling.profilers.energy import (
        ENERGY_ENV,
        REQUIRE_ENERGY_ENV,
        energy_enabled,
        require_measured_energy,
    )

    env = os.environ.copy()
    env[ENERGY_ENV] = "1" if energy_enabled() else "0"
    env[REQUIRE_ENERGY_ENV] = "1" if require_measured_energy() else "0"
    env["PYTHON"] = sys.executable
    env["PYO3_PYTHON"] = sys.executable
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)

    libdir = sysconfig.get_config_var("LIBDIR")
    if libdir:
        current_ld_path = env.get("LD_LIBRARY_PATH", "")
        if libdir not in current_ld_path:
            env["LD_LIBRARY_PATH"] = f"{libdir}:{current_ld_path}" if current_ld_path else libdir

    if sys.base_prefix:
        env["PYTHONHOME"] = sys.base_prefix

    # Repo root first (repo-local `profiling` package), then venv site-packages.
    pythonpath_parts = [str(REPO_ROOT)]
    venv = os.environ.get("VIRTUAL_ENV")
    if not venv and sys.prefix != sys.base_prefix:
        venv = sys.prefix
    if venv and os.path.isdir(os.path.join(venv, "lib")):
        env["VIRTUAL_ENV"] = venv
    site_packages = sysconfig.get_path("purelib")
    if site_packages:
        pythonpath_parts.append(site_packages)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    return env


def _cargo_build_env() -> dict[str, str]:
    """Select the launcher's interpreter when PyO3 configures the Rust build.

    ``cargo build`` otherwise discovers Python from the ambient shell PATH. A
    launcher invoked as ``.venv/bin/python -m launcher`` can therefore compile
    against system Python while `_build_subprocess_env` later supplies the venv
    runtime, producing an ABI/stdlib mismatch before Python can import
    ``encodings``. Build-time PyO3 needs the interpreter selector, not the
    runtime's PYTHONHOME/PYTHONPATH, so remove inherited path overrides here.
    """

    env = os.environ.copy()
    python = _launcher_python()
    env["PYTHON"] = python
    env["PYO3_PYTHON"] = python
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    return env


def _launcher_python() -> str:
    """The launcher's interpreter under one stable spelling.

    A venv's ``python``, ``python3`` and ``python3.12`` are one interpreter behind
    links, but PyO3's and ``build.rs``'s build scripts fingerprint ``PYO3_PYTHON``
    as a string. ``uv run python`` reports ``.venv/bin/python3`` while pytest's
    workers report ``.venv/bin/python``, so alternating between them recompiled
    pyo3 and everything above it (~40 s) on the next launch. Follow only links
    that stay in the executable's directory: that collapses the spellings
    without leaving the venv for the base interpreter it links to.
    """

    path = Path(sys.executable)
    while path.is_symlink():
        target = path.parent / os.readlink(path)
        if target.parent.resolve() != path.parent.resolve():
            break
        path = target
    return str(path)


# ── perf profiling (skill `operate-profile-sim-speed`) ──────────────────────────────


def perf_available() -> bool:
    """Whether the `perf` CLI is on PATH (the wallclock profiler the launcher's
    `--profile` mode wraps the run with)."""
    return shutil.which("perf") is not None


def wrap_with_perf(argv: list[str], perf_data: Path, freq: int = 499) -> list[str]:
    """Wrap a run argv with `perf record`. `--call-graph dwarf` is required
    because the release profile omits frame pointers; `-F` is the sampling Hz
    (499/999 avoids lock-step with timers). Output is the binary `perf.data` at
    `perf_data` — read it with `perf report`, never `cat`. See skill
    `operate-profile-sim-speed`."""
    return [
        "perf",
        "record",
        "-F",
        str(freq),
        "--call-graph",
        "dwarf",
        "-o",
        str(perf_data),
        "--",
        *argv,
    ]


def _profile_env() -> dict[str, str]:
    """`_build_subprocess_env` plus single-threaded BLAS/OMP so the one-time L4
    model build's numpy/torch calls don't smear samples across worker threads
    (skill `operate-profile-sim-speed` Step 2)."""
    env = _build_subprocess_env()
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    return env


def _report_build_failure(stage: str, result: ProcessResult) -> None:
    reason = result.termination_reason or (
        "live descendants remained after root exit"
        if result.leaked_descendants
        else "nonzero exit status"
    )
    sys.stderr.write(
        f"{stage} failed: {reason}; exit_code={result.exit_code}, "
        f"pid={result.pid}, process_group_id={result.process_group_id}\n"
    )
    if result.output:
        sys.stderr.write(result.output + "\n")


def cargo_build(build_type: str = "debug", build_analyzer: bool = True) -> bool:
    """Build the simulator crate, then run schema discovery (INV-8). Returns
    True on success; on failure no schema is written (design §1.2.7 bootstrap
    edge — `load_schema` then raises `SchemaNotFound`)."""
    cmd = ["cargo", "build"]
    if build_type == "release":
        cmd.append("--release")
    elif build_type != "debug":
        cmd.extend(["--profile", build_type])
    build_env = _cargo_build_env()
    with _LAUNCHER_LEASES.build(build_type):
        build_result = _PROCESS_SUPERVISOR.run_sync(
            ProcessSpec(argv=cmd, cwd=REPO_ROOT, env=build_env, name="build-simulator")
        )
        if not build_result.succeeded:
            _report_build_failure("simulator compilation", build_result)
            return False

        # Schema discovery: list-params → deployment_schema.json.
        binary = binary_path(build_type)
        schema_result = _PROCESS_SUPERVISOR.run_sync(
            ProcessSpec(
                argv=[str(binary), "list-params"],
                cwd=REPO_ROOT,
                env=_build_subprocess_env(),
                capture_output=True,
                name="discover-schema",
            )
        )
        if not schema_result.succeeded:
            _report_build_failure("schema discovery", schema_result)
            return False
        schema_path = schema_json_path(build_type)
        temporary_schema = schema_path.with_name(f".{schema_path.name}.{os.getpid()}.tmp")
        try:
            temporary_schema.write_text(schema_result.output)
            os.replace(temporary_schema, schema_path)
        finally:
            temporary_schema.unlink(missing_ok=True)

        if not build_analyzer:
            return True

        # The analyzer is a standalone workspace crate (not a sim dep), so the
        # build above doesn't produce it. It remains best-effort.
        analyzer_cmd = _analyzer_build_command(build_type)
        analyzer_result = _PROCESS_SUPERVISOR.run_sync(
            ProcessSpec(
                argv=analyzer_cmd,
                cwd=REPO_ROOT,
                env=build_env,
                name="build-analyzer",
            )
        )
        if not analyzer_result.succeeded:
            _report_build_failure("analyzer compilation", analyzer_result)
            sys.stderr.write("[warn] analyzer build failed; runs will skip post-run analysis\n")
        return True


def cargo_build_analyzer(build_type: str = "debug") -> bool:
    """Build only the standalone analyzer used by post-run analysis.

    Alignment analysis consumes existing profile/prediction/simulation roots and
    does not need the simulator binary or deployment schema. Keeping this build
    boundary separate means `alignment analyze` cannot be blocked by unrelated
    simulator source drift.
    """
    analyzer_cmd = _analyzer_build_command(build_type)

    with _LAUNCHER_LEASES.build(build_type):
        return _PROCESS_SUPERVISOR.run_sync(
            ProcessSpec(
                argv=analyzer_cmd,
                cwd=REPO_ROOT,
                env=_cargo_build_env(),
                name="build-analyzer",
            )
        ).succeeded


def _analyzer_build_command(build_type: str) -> list[str]:
    analyzer_cmd = ["cargo", "build", "-p", "analyzer"]
    if build_type == "release":
        analyzer_cmd.append("--release")
    elif build_type != "debug":
        analyzer_cmd.extend(["--profile", build_type])
    return analyzer_cmd


async def _run_capture(argv: list[str]) -> tuple[int, str]:
    """Capture through a regular temp file under the shared supervisor."""

    result = await _PROCESS_SUPERVISOR.run(
        ProcessSpec(argv=argv, cwd=REPO_ROOT, capture_output=True, name="capture")
    )
    return _effective_exit_code(result), result.output


def _run_capture_sync(argv: list[str]) -> tuple[int, str]:
    """Synchronous capture through the same root-PID completion contract."""

    result = _PROCESS_SUPERVISOR.run_sync(
        ProcessSpec(argv=argv, cwd=REPO_ROOT, capture_output=True, name="capture")
    )
    return _effective_exit_code(result), result.output


def _effective_exit_code(result: ProcessResult) -> int:
    if result.succeeded:
        return 0
    return result.exit_code if result.exit_code != 0 else 1


async def run_logged_process(
    argv: list[str],
    log_dir: Path,
    *,
    env: dict[str, str] | None = None,
    name: str,
    append_log: bool = False,
) -> ProcessResult:
    """Run a stage with combined output written directly to ``stdout.log``."""

    log_dir.mkdir(parents=True, exist_ok=True)
    return await _PROCESS_SUPERVISOR.run(
        ProcessSpec(
            argv=argv,
            cwd=REPO_ROOT,
            env=env,
            log_path=log_dir / "stdout.log",
            append_log=append_log,
            name=name,
        )
    )


async def run_analysis(
    log_dir: Path, build_type: str = "debug", subjects: list[str] | None = None
) -> None:
    """Best-effort post-run analysis: Rust `analyze run` (parquet → report+payload
    JSON) then the Python renderer (payload JSON → PNGs). Failures warn and return
    — analysis must never fail an otherwise-successful run.

    Async: across a sweep, many runs' analyze+render overlap under the existing
    semaphore instead of serializing on a blocking call that stalls the event loop.

    `subjects` is the *intent* layer: which subjects to run (e.g. `["slo-general"]`).
    `None`/empty = all subjects applicable to the run's deployment (the analyzer
    self-selects via its applicability gate — the launcher never decides what is
    *applicable*, only what is *wanted*).

    Each step's combined stdout+stderr is appended to the run's `stdout.log` (the
    sim run already closed it, so analysis output would otherwise be lost) under a
    labeled section, keeping the full run record in one file."""
    analyzer = analyzer_binary_path(build_type)
    stdout_log = log_dir / "stdout.log"
    journal = RunJournal(log_dir)

    def _append(section: str, text: str, elapsed_ms: float) -> None:
        # Stamp each stage's wall time in the section header. `analyze run` also
        # writes a per-subject `analyzer_timing.json`; render/trace had no timing
        # at all before this, and trace is the pipeline's slowest stage.
        header = f"{section} [{elapsed_ms:.0f} ms]"
        with stdout_log.open("a") as fh:
            fh.write(f"\n=== {header} ===\n{text}")
            if text and not text.endswith("\n"):
                fh.write("\n")

    async def _timed_step(
        stage: str,
        section: str,
        argv: list[str],
        validate_artifacts=None,
    ) -> int:
        spec = ProcessSpec(
            argv=argv,
            cwd=REPO_ROOT,
            capture_output=True,
            name=stage,
        )
        journal.update(stage, StageState.RUNNING, spec=spec)
        t0 = time.perf_counter()
        try:
            result = await _PROCESS_SUPERVISOR.run(spec)
        except BaseException:
            journal.update(stage, StageState.CANCELLED, spec=spec)
            raise
        _append(section, result.output, (time.perf_counter() - t0) * 1e3)
        journal.update(stage, StageState.EXITED, spec=spec, result=result)
        if not result.succeeded:
            journal.update(
                stage,
                StageState.FAILED,
                spec=spec,
                result=result,
                error="process failed or left process-group descendants",
            )
            return _effective_exit_code(result)
        if validate_artifacts is not None:
            journal.update(stage, StageState.VALIDATING, spec=spec, result=result)
            try:
                validation = validate_artifacts()
            except ArtifactValidationError as error:
                journal.update(
                    stage,
                    StageState.FAILED,
                    spec=spec,
                    result=result,
                    error=str(error),
                )
                _append(f"{section} artifact validation", str(error), 0.0)
                return 1
            journal.update(
                stage,
                StageState.SUCCEEDED,
                spec=spec,
                result=result,
                artifacts=validation.paths,
            )
        else:
            journal.update(stage, StageState.SUCCEEDED, spec=spec, result=result)
        return 0

    if not analyzer.exists():
        print(f"[analyze] {analyzer} not built; skipping analysis for {log_dir}")
        return
    subjects = subjects or []
    # The analyzer owns variant completeness: a normal invocation generates both
    # unlocked and batch-locked optimality in one timing/report transaction.
    if (
        await _timed_step(
            "analyze_compute",
            "analyze compute",
            [str(analyzer), "run", str(log_dir), *subjects],
            lambda: validate_json_outputs(log_dir / "reports"),
        )
        != 0
    ):
        print(f"[analyze] compute failed for {log_dir}")
        return
    # Render and trace both read only compute's outputs and the raw logs, and the
    # workflow plan gives them no edge between them, so they run concurrently.
    # Always emit the per-kernel Perfetto timeline (traces/<prefix>.pftrace.gz).
    # A standalone verb, not a subject (different output contract: a binary trace
    # for ui.perfetto.dev, not report/payload JSON), so it runs here with CLI
    # defaults rather than through the subject catalog. Best-effort like the rest.
    render_rc, trace_rc = await asyncio.gather(
        _timed_step(
            "render",
            "analyze render",
            [
                sys.executable,
                str(REPO_ROOT / "analyzer" / "python"),
                "render",
                str(log_dir),
                *subjects,
            ],
            lambda: validate_render_artifacts(log_dir),
        ),
        _timed_step(
            "trace",
            "analyze trace",
            [str(analyzer), "trace", str(log_dir)],
            lambda: validate_trace_artifacts(log_dir),
        ),
    )
    if render_rc != 0:
        print(f"[analyze] render failed for {log_dir}")
    if trace_rc != 0:
        print(f"[analyze] trace failed for {log_dir}")


def run_alignment_analysis(
    log_dir: Path,
    build_type: str = "debug",
    subjects: list[str] | None = None,
) -> bool:
    """Synchronously compute and render selected alignment subjects.

    This explicit alignment command has no sweep-level parallelism to preserve;
    each phase must finish before the next consumes its artifacts.
    """
    analyzer = analyzer_binary_path(build_type)
    if not analyzer.exists():
        print(f"[analyze] {analyzer} not built; skipping alignment analysis for {log_dir}")
        return False

    stdout_log = log_dir / "stdout.log"

    def run_step(section: str, argv: list[str]) -> int:
        started = time.perf_counter()
        rc, out = _run_capture_sync(argv)
        elapsed_ms = (time.perf_counter() - started) * 1e3
        with stdout_log.open("a") as fh:
            fh.write(f"\n=== {section} [{elapsed_ms:.0f} ms] ===\n")
            fh.write(out)
            if out and not out.endswith("\n"):
                fh.write("\n")
        return rc

    selected = subjects or [
        "alignment-iteration",
        "alignment-timeline",
        "alignment-workload",
        "alignment-e2e",
    ]
    # The analyzer records subject failures even when the process exits zero.
    # Remove the prior outcome so a crashed recomputation cannot reuse it.
    (log_dir / "reports" / "analyzer_timing.json").unlink(missing_ok=True)
    rc = run_step(
        "analyze alignment compute",
        [str(analyzer), "alignment", str(log_dir), *selected],
    )
    if rc != 0:
        print(f"[analyze] alignment compute failed for {log_dir}")
        return False
    try:
        validate_alignment_analysis_artifacts(log_dir, selected)
    except ArtifactValidationError as error:
        print(f"[analyze] alignment compute failed for {log_dir}: {error}")
        return False
    # Timeline is an interactive payload served by the UI, with no PNG renderer.
    render_subjects = [subject for subject in selected if subject != "alignment-timeline"]
    if not render_subjects:
        return True
    if (
        run_step(
            "analyze alignment render",
            [
                sys.executable,
                str(REPO_ROOT / "analyzer" / "python"),
                "render",
                str(log_dir),
                *render_subjects,
            ],
        )
        != 0
    ):
        print(f"[analyze] alignment render failed for {log_dir}")
        return False
    return True


def run_sweep_analysis(experiment_dir: Path, build_type: str = "debug") -> None:
    """Synchronously collect and render one explicit launcher sweep.

    The simulation tasks and their per-run analyzers have already completed at
    this boundary. Rust reads `sweep_manifest.json` plus their small JSON reports;
    Python only renders the resulting experiment-level payload.
    """
    analyzer = analyzer_binary_path(build_type)
    if not analyzer.exists():
        print(f"[aggregate] {analyzer} not built; skipping sweep analysis for {experiment_dir}")
        return

    stdout_log = experiment_dir / "stdout.log"

    def run_step(section: str, argv: list[str]) -> int:
        started = time.perf_counter()
        returncode, output = _run_capture_sync(argv)
        elapsed_ms = (time.perf_counter() - started) * 1e3
        with stdout_log.open("a") as stream:
            stream.write(f"\n=== {section} [{elapsed_ms:.0f} ms] ===\n")
            stream.write(output)
            if output and not output.endswith("\n"):
                stream.write("\n")
        return returncode

    if (
        run_step(
            "analyze sweep compute",
            [str(analyzer), "sweep", str(experiment_dir)],
        )
        != 0
    ):
        print(f"[aggregate] sweep compute failed for {experiment_dir}")
        return
    if (
        run_step(
            "analyze sweep render",
            [
                sys.executable,
                str(REPO_ROOT / "analyzer" / "python"),
                "render",
                str(experiment_dir),
                "sweep",
            ],
        )
        != 0
    ):
        print(f"[aggregate] sweep render failed for {experiment_dir}")


async def run_iter_breakdown(log_dir: Path, build_type: str = "debug") -> None:
    """Best-effort `analyze gen-iter-breakdown` → `reports/iter_breakdown.ans`
    (human-readable cost tree). Wired ONLY into the timing-predict entry, not
    the shared `run_analysis`: a real run has thousands of iters, so auto-emitting
    a per-iter tree there would be a huge file — predict dirs have a handful of
    cases. The verb itself is general (`analyze gen-iter-breakdown <dir>` works on
    any artifact dir); only the *automatic* emission is predict-scoped. Failures
    warn and return (analysis never fails an otherwise-successful run)."""
    analyzer = analyzer_binary_path(build_type)
    if not analyzer.exists():
        print(f"[analyze] {analyzer} not built; skipping iter-breakdown for {log_dir}")
        return
    rc, out = await _run_capture([str(analyzer), "gen-iter-breakdown", str(log_dir)])
    stdout_log = log_dir / "stdout.log"
    if out:
        with stdout_log.open("a") as fh:
            fh.write(f"\n=== analyze gen-iter-breakdown ===\n{out}")
            if not out.endswith("\n"):
                fh.write("\n")
    if rc != 0:
        print(f"[analyze] gen-iter-breakdown failed for {log_dir}:\n{out}")
