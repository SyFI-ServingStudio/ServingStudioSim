"""Execution engine — cargo build + schema discovery, and the async subprocess
wrapper that runs the rust binary.

Per design §1.2.3 / §1.2.7:
- `cargo_build` is the single shared build per batch; on success it immediately
  runs `simulator list-params` and writes `deployment_schema.json` (INV-8:
  schema discovery is part of the build, not a separate step).
- `SimulationRunner` spawns one `run` / `build-cache-only` subprocess, streams
  stdout to `stdout.log`, and supports cooperative `cancel()` across a sweep.
- `_build_subprocess_env` wires the PyO3-embedded interpreter (the rust binary
  embeds Python to query profile.db) — ported unchanged from the reference
  launcher, the one piece design §1.2.3 says "survives unchanged".
"""

from __future__ import annotations

import asyncio
import os
import selectors
import shutil
import subprocess
import sys
import sysconfig
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARALLELISM = 200


async def _run_in_polling_thread[Result](function: Callable[[], Result]) -> Result:
    """Run blocking process work without registering an asyncio child watcher."""

    result: Future[Result] = Future()

    def invoke() -> None:
        try:
            result.set_result(function())
        except BaseException as error:
            result.set_exception(error)

    worker_thread = threading.Thread(target=invoke, daemon=True)
    worker_thread.start()
    while not result.done():
        await asyncio.sleep(0.05)
    worker_thread.join()
    return result.result()


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
    """
    env = os.environ.copy()
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
    env["PYTHON"] = sys.executable
    env["PYO3_PYTHON"] = sys.executable
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    return env


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
    if subprocess.run(cmd, cwd=REPO_ROOT, env=build_env).returncode != 0:
        return False

    # Schema discovery: list-params → deployment_schema.json.
    binary = binary_path(build_type)
    result = subprocess.run(
        [str(binary), "list-params"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=_build_subprocess_env(),
    )
    if result.returncode != 0:
        sys.stderr.write(f"schema discovery failed:\n{result.stderr}")
        return False
    schema_json_path(build_type).write_text(result.stdout)

    if not build_analyzer:
        return True

    # The analyzer is a standalone workspace crate (not a sim dep), so the build
    # above doesn't produce it — build it explicitly. Best-effort: a missing
    # analyzer must not block runs (the post-run analysis step is also optional).
    analyzer_cmd = ["cargo", "build", "-p", "analyzer"]
    if build_type == "release":
        analyzer_cmd.append("--release")
    elif build_type != "debug":
        analyzer_cmd.extend(["--profile", build_type])
    if subprocess.run(analyzer_cmd, cwd=REPO_ROOT, env=build_env).returncode != 0:
        sys.stderr.write("[warn] analyzer build failed; runs will skip post-run analysis\n")
    return True


def cargo_build_analyzer(build_type: str = "debug") -> bool:
    """Build only the standalone analyzer used by post-run analysis.

    Alignment analysis consumes existing profile/prediction/simulation roots and
    does not need the simulator binary or deployment schema. Keeping this build
    boundary separate means `alignment analyze` cannot be blocked by unrelated
    simulator source drift.
    """
    analyzer_cmd = ["cargo", "build", "-p", "analyzer"]
    if build_type == "release":
        analyzer_cmd.append("--release")
    elif build_type != "debug":
        analyzer_cmd.extend(["--profile", build_type])
    return (
        subprocess.run(
            analyzer_cmd,
            cwd=REPO_ROOT,
            env=_cargo_build_env(),
        ).returncode
        == 0
    )


def _run_capture_sync(argv: list[str]) -> tuple[int, str]:
    """Run one sequential launcher stage and capture its combined output.

    Completion belongs to the direct child. Profiler or renderer descendants may
    inherit stdout, so waiting for pipe EOF can hang after the requested command
    has already exited.
    """
    process = subprocess.Popen(
        argv,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    captured_output = bytearray()
    return_code = _consume_until_direct_child_exit(process, captured_output.extend)
    return return_code, captured_output.decode(errors="replace")


def _consume_until_direct_child_exit(
    process: subprocess.Popen[bytes],
    consume_chunk: Callable[[bytes], object],
) -> int:
    """Drain available output while treating only ``process`` as lifecycle owner."""

    assert process.stdout is not None
    output_selector = selectors.DefaultSelector()
    output_selector.register(process.stdout, selectors.EVENT_READ)
    try:
        while process.poll() is None:
            if not output_selector.get_map():
                return process.wait()
            for selector_key, _event_mask in output_selector.select(timeout=0.1):
                output_chunk = os.read(selector_key.fd, 64 * 1024)
                if output_chunk:
                    consume_chunk(output_chunk)
                else:
                    output_selector.unregister(selector_key.fileobj)

        # Drain bytes already queued by the direct child without waiting for an
        # inherited descriptor to reach EOF in a descendant.
        while output_selector.get_map():
            ready_events = output_selector.select(timeout=0)
            if not ready_events:
                break
            for selector_key, _event_mask in ready_events:
                output_chunk = os.read(selector_key.fd, 64 * 1024)
                if output_chunk:
                    consume_chunk(output_chunk)
                else:
                    output_selector.unregister(selector_key.fileobj)
        assert process.returncode is not None
        return process.returncode
    finally:
        output_selector.close()
        process.stdout.close()


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

    def _append(section: str, text: str, elapsed_ms: float) -> None:
        # Stamp each stage's wall time in the section header. `analyze run` also
        # writes a per-subject `analyzer_timing.json`; render/trace had no timing
        # at all before this, and trace is the pipeline's slowest stage.
        header = f"{section} [{elapsed_ms:.0f} ms]"
        with stdout_log.open("a") as fh:
            fh.write(f"\n=== {header} ===\n{text}")
            if text and not text.endswith("\n"):
                fh.write("\n")

    async def _timed_step(section: str, argv: list[str]) -> int:
        t0 = time.perf_counter()
        rc, out = await _run_in_polling_thread(lambda: _run_capture_sync(argv))
        _append(section, out, (time.perf_counter() - t0) * 1e3)
        return rc

    if not analyzer.exists():
        print(f"[analyze] {analyzer} not built; skipping analysis for {log_dir}")
        return
    subjects = subjects or []
    # The analyzer owns variant completeness: a normal invocation generates both
    # unlocked and batch-locked optimality in one timing/report transaction.
    if await _timed_step("analyze compute", [str(analyzer), "run", str(log_dir), *subjects]) != 0:
        print(f"[analyze] compute failed for {log_dir}")
        return
    if (
        await _timed_step(
            "analyze render",
            [
                sys.executable,
                str(REPO_ROOT / "analyzer" / "python"),
                "render",
                str(log_dir),
                *subjects,
            ],
        )
        != 0
    ):
        print(f"[analyze] render failed for {log_dir}")

    # Always emit the per-kernel Perfetto timeline (traces/<prefix>.pftrace.gz).
    # A standalone verb, not a subject (different output contract: a binary trace
    # for ui.perfetto.dev, not report/payload JSON), so it runs here with CLI
    # defaults rather than through the subject catalog. Best-effort like the rest.
    if await _timed_step("analyze trace", [str(analyzer), "trace", str(log_dir)]) != 0:
        print(f"[analyze] trace failed for {log_dir}")


def run_alignment_analysis(
    log_dir: Path,
    build_type: str = "debug",
    subjects: list[str] | None = None,
) -> None:
    """Synchronously compute and render selected alignment subjects.

    This explicit alignment command has no sweep-level parallelism to preserve;
    each phase must finish before the next consumes its artifacts.
    """
    analyzer = analyzer_binary_path(build_type)
    if not analyzer.exists():
        print(f"[analyze] {analyzer} not built; skipping alignment analysis for {log_dir}")
        return

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
    rc = run_step(
        "analyze alignment compute",
        [str(analyzer), "alignment", str(log_dir), *selected],
    )
    if rc != 0:
        print(f"[analyze] alignment compute failed for {log_dir}")
        return
    if (
        run_step(
            "analyze alignment render",
            [
                sys.executable,
                str(REPO_ROOT / "analyzer" / "python"),
                "render",
                str(log_dir),
                *selected,
            ],
        )
        != 0
    ):
        print(f"[analyze] alignment render failed for {log_dir}")


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
    rc, out = await _run_in_polling_thread(
        lambda: _run_capture_sync([str(analyzer), "gen-iter-breakdown", str(log_dir)])
    )
    stdout_log = log_dir / "stdout.log"
    if out:
        with stdout_log.open("a") as fh:
            fh.write(f"\n=== analyze gen-iter-breakdown ===\n{out}")
            if not out.endswith("\n"):
                fh.write("\n")
    if rc != 0:
        print(f"[analyze] gen-iter-breakdown failed for {log_dir}:\n{out}")


@dataclass
class SimulationRunner:
    """One subprocess run. `argv` comes from `schema.build_cli_command`."""

    argv: list[str]
    log_dir: Path
    env: dict[str, str] = field(default_factory=_build_subprocess_env)
    _proc: subprocess.Popen[bytes] | None = field(default=None, init=False, repr=False)

    async def run(self) -> bool:
        """Spawn and stream output while preserving sweep-level concurrency."""

        return await _run_in_polling_thread(self._run_sync)

    def _run_sync(self) -> bool:
        """Own the direct simulator child and return when that child exits."""

        self.log_dir.mkdir(parents=True, exist_ok=True)
        stdout_log = self.log_dir / "stdout.log"
        self._proc = subprocess.Popen(
            self.argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=REPO_ROOT,
            env=self.env,
        )
        with stdout_log.open("wb") as output_stream:

            def write_output(output_chunk: bytes) -> None:
                output_stream.write(output_chunk)
                output_stream.flush()

            return_code = _consume_until_direct_child_exit(self._proc, write_output)
        return return_code == 0

    def cancel(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.kill()
