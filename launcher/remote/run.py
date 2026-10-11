"""The sending side of `--remote`: plan here, run there, bring results back.

One invocation is one remote job and blocks until it ends:

1. Plan locally, as the command would (`plan_simulation`, the predictor's
   config), to learn where results go and to reject a bad command before
   anything is shipped. Outputs must be relative paths under ``logs/``.
2. Register a managed job here when the caller runs inside one, and seed the
   remote output directory with the identity files that registration produced
   (``experiment.meta.json``, ``prediction.meta.json``), which the remote run
   then keeps. The remote run itself never registers (`node.py`).
3. Ship the tree (`source.py`), give the remote its venv and binaries
   (`node.py build`), and start the job (`launch.py`) with ``--no-gpu
   --no-plot``.
4. Relay the job's output, with a heartbeat at least once a minute so a
   supervising agent does not take a long, quiet run for a hung one.
5. Copy the results back to the same relative path here, write
   ``remote-run.json`` beside them, and report the managed status.

The experiment stays on the remote in full; the copy leaves out the bulky
per-iteration logs (``raw/cost_log/``) and plots unless the remote is
``ephemeral``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from ..cli import GlobalOptions
from . import launch
from .config import Remote, RemoteConfigError, load_remote
from .layout import Layout
from .node import SHIPPED_MARKER
from .source import (
    CPU_GROUPS,
    REPO_ROOT,
    Source,
    SourceError,
    collect,
    source_stamp,
    venv_key,
)
from .transport import Transport, TransportError, q

LOGS = REPO_ROOT / "logs"
POLL_SECONDS = 10.0
HEARTBEAT_SECONDS = 60.0
# A job seen neither running nor finished this many polls in a row is lost
# (one missed `squeue` answer is not enough).
LOST_POLLS = 3
SIMULATION_COPY_EXCLUDES = ("raw/cost_log/", "plots/", "/experiment.meta.json")
REMOTE_RECORD = "remote-run.json"


class RemoteRunError(RuntimeError):
    """The remote run cannot proceed; the message says why."""


class _Interrupted(Exception):
    pass


@dataclass
class JobOutcome:
    job_id: str
    exit_code: int
    slurm_id: str | None
    started_at: str
    finished_at: str
    detail: str = ""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _say(message: str) -> None:
    print(f"[remote] {message}", flush=True)


def _warn(message: str) -> None:
    print(f"[remote] warning: {message}", file=sys.stderr, flush=True)


def _env_prefix(env: dict[str, str]) -> str:
    return "".join(f"export {key}={q(value)}; " for key, value in env.items())


class Session:
    """One remote run: the remote, how to reach it, and the shipped source."""

    def __init__(self, remote: Remote):
        self.remote = remote
        self.transport = Transport(remote)
        # `files: shared` moves files with a local copy; commands still run on
        # the access host.
        self.data = (
            Transport(dataclasses.replace(remote, ssh=None))
            if remote.files == "shared"
            else self.transport
        )
        self.layout = Layout(remote.workdir)
        self.source: Source | None = None
        self.bin_dir: PurePosixPath | None = None
        # The remote is itself a profiler image (`prepare_in_image`).
        self.in_image = False
        # The job's own venv, when it is not the version's (CPU) `.venv`.
        self.job_venv: PurePosixPath | None = None

    # ── code ──────────────────────────────────────────────────────────────

    def ship(self) -> Source:
        source = collect()
        self.source = source
        src = self.layout.src(source.version)
        shipped = self.transport.run(f"test -f {q(src / SHIPPED_MARKER)}", check=False)
        if shipped.returncode == 0:
            _say(f"code version {source.version} already on {self.remote.name}")
            return source
        src_root = self.layout.workdir / "src"
        previous = self.transport.run(
            f"cd {q(src_root)} 2>/dev/null && ls -1t | while read -r d; do "
            f'[ -f "$d/{SHIPPED_MARKER}" ] && echo "$d" && break; done',
            check=False,
        ).stdout.strip()
        incoming = src_root / f".incoming-{source.version}-{uuid.uuid4().hex[:8]}"
        self.transport.run(f"mkdir -p {q(incoming)} {q(self.layout.logs)}")
        options = ["--from0", "--files-from=-"]
        if previous:
            # Unchanged files become hard links to the previous version.
            options.append(f"--link-dest={src_root / previous}")
        _say(
            f"shipping code version {source.version} ({len(source.files)} files"
            + (f", linked against {previous}" if previous else "")
            + ")"
        )
        self.data.rsync(
            f"{source.root}/",
            self.data.remote_spec(incoming, directory=True),
            options=options,
            input="\0".join(source.files),
        )
        self.transport.write_text(
            incoming / ".source-stamp.json", json.dumps(source_stamp(source), indent=2) + "\n"
        )
        # A concurrent sender may have shipped the same version meanwhile.
        self.transport.run(
            f"cd {q(incoming)} && ln -sfn ../../logs logs && cd .. && "
            f"if [ -e {q(src)} ]; then rm -rf {q(incoming)}; "
            f"else mv {q(incoming)} {q(src)}; fi && touch {q(src / SHIPPED_MARKER)}"
        )
        return source

    def _venv_script(self, key: str, groups: tuple[str, ...]) -> str:
        """Create venv ``key`` with ``groups`` once per remote, under a host lock."""

        source = self.source
        assert source is not None
        venv = self.layout.venv(key)
        ready = venv / ".vibesim-ready"
        lock = self.remote.node_local / f"vibesim-remote-{key}-venv.lock"
        sync = f"UV_PROJECT_ENVIRONMENT={q(venv)} uv sync --frozen --no-default-groups " + " ".join(
            f"--group {group}" for group in groups
        )
        if "profiling" in groups:
            # As `just sync`: DeepGEMM builds without isolation against the env's
            # Torch, so everything else goes in first.
            sync = (
                f"export DG_USE_LOCAL_VERSION=0; {sync} --inexact "
                f"--no-install-package deep-gemm && {sync}"
            )
        sync = (
            f"if [ ! -f {q(ready)} ]; then cd {q(self.layout.src(source.version))} && "
            f"{sync} && touch {q(ready)}; fi"
        )
        return (
            f"if [ ! -f {q(ready)} ]; then mkdir -p {q(self.remote.node_local)}; "
            f"echo '[remote] creating venv {key} ({' '.join(groups)})'; "
            f"flock {q(lock)} bash -c {q(sync)}; fi; "
        )

    def prepare(self, job_groups: tuple[str, ...] | None = None) -> PurePosixPath:
        """The remote venv and binaries for the shipped version; returns the bin dir.

        The version's ``.venv`` is always the CPU venv, which builds and runs the
        job runner. ``job_groups`` gives the job a venv of its own with those groups.
        """

        source = self.source
        assert source is not None
        src = self.layout.src(source.version)
        script = (
            # The sender's venv must not leak in when the access host is this one.
            f"set -e; unset VIRTUAL_ENV; {_env_prefix(self.remote.env)}"
            + self._venv_script(source.venv_key, source.groups)
        )
        if job_groups is not None:
            key = venv_key(source.root, job_groups)
            self.job_venv = self.layout.venv(key)
            script += self._venv_script(key, job_groups)
        script += (
            f"ln -sfn ../../venvs/{source.venv_key} {q(src / '.venv')}; "
            f"cd {q(src)} && .venv/bin/python -m launcher.remote.node build "
            f"{q(self.layout.workdir)} {source.version} --node-local {q(self.remote.node_local)}"
        )
        if self.transport.stream(script) != 0:
            raise RemoteRunError(f"{self.remote.name}: preparing version {source.version} failed")
        resolved = self.transport.run(f"readlink -f {q(src / 'bin')}").stdout.strip()
        self.bin_dir = PurePosixPath(resolved)
        self.transport.run(
            f"cd {q(src)} && .venv/bin/python -m launcher.remote.node prune "
            f"{q(self.layout.workdir)} --keep {self.remote.keep_versions}",
            check=False,
        )
        return self.bin_dir

    # ── files ─────────────────────────────────────────────────────────────

    def remote_logs_path(self, local: Path) -> PurePosixPath:
        return self.layout.logs / local.resolve().relative_to(LOGS.resolve()).as_posix()

    def push_files(self, paths: list[Path]) -> None:
        """Copy local files under ``logs/`` to the same place under the remote's."""

        for path in paths:
            destination = self.remote_logs_path(path)
            self.transport.run(f"mkdir -p {q(destination.parent)}")
            self.data.rsync(str(path), self.data.remote_spec(destination))

    def fetch(self, local: Path, *, excludes: tuple[str, ...] = ()) -> None:
        local.mkdir(parents=True, exist_ok=True)
        if self.remote.ephemeral:
            # The remote may not exist tomorrow: keep everything but identity.
            excludes = tuple(e for e in excludes if e.startswith("/"))
        self.data.rsync(
            self.data.remote_spec(self.remote_logs_path(local), directory=True),
            f"{local}/",
            options=[f"--exclude={pattern}" for pattern in excludes],
        )

    # ── the job ───────────────────────────────────────────────────────────

    def run_job(
        self, argv: list[str], *, env: dict[str, str] | None = None, keep_db: bool = False
    ) -> JobOutcome:
        """Run ``python -m launcher ARGV`` there; ``keep_db`` keeps its profile.db copy."""

        source, bin_dir = self.source, self.bin_dir
        assert source is not None and (bin_dir is not None or self.in_image)
        job_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        jobdir = self.layout.job(job_id)
        src = self.layout.src(source.version)
        job = {
            "job_id": job_id,
            "version": source.version,
            "src": str(src),
            "argv": argv,
            "env": {**self.remote.env, **(env or {})},
            "keep_db": keep_db,
            "venv": str(self.job_venv or src / ".venv"),
            "node_local": str(self.remote.node_local),
            "bin_dir": "" if bin_dir is None else str(bin_dir),
            "sender": socket.gethostname(),
            "created_at": _now(),
        }
        self.transport.write_text(jobdir / "job.json", json.dumps(job, indent=2) + "\n")
        started = _now()
        submitted = launch.submit(
            self.transport,
            self.remote,
            job_id=job_id,
            jobdir=jobdir,
            cwd=src,
            remote_python=self.layout.python(source.version),
        )
        where = f"Slurm job {submitted.handle}" if submitted.slurm else f"pid {submitted.handle}"
        _say(f"job {job_id} started on {self.transport.host_label} ({where})")
        status = self._wait(submitted, job_id)
        return JobOutcome(
            job_id=job_id,
            exit_code=status.exit_code if status.exit_code is not None else 1,
            slurm_id=submitted.handle if submitted.slurm else None,
            started_at=started,
            finished_at=_now(),
            detail=status.detail,
        )

    def _wait(self, submitted: launch.Submitted, job_id: str) -> launch.Status:
        offset, lost_polls = 0, 0
        began = last_output = time.monotonic()
        previous = signal.signal(signal.SIGTERM, _raise_interrupted)
        try:
            delay = 2.0
            while True:
                time.sleep(delay)
                delay = POLL_SECONDS
                try:
                    status = launch.poll(self.transport, submitted, offset)
                except TransportError as exc:
                    _warn(f"poll failed, retrying: {exc}")
                    continue
                if status.log_chunk:
                    sys.stdout.write(status.log_chunk)
                    sys.stdout.flush()
                    last_output = time.monotonic()
                offset = status.log_size
                if status.log_total > offset:
                    delay = 0.2  # more output is already waiting
                    continue
                if status.exit_code is not None:
                    return status
                lost_polls = lost_polls + 1 if status.lost else 0
                if lost_polls >= LOST_POLLS:
                    _warn(f"job {job_id} vanished without an exit code ({status.detail})")
                    return status
                if time.monotonic() - last_output >= HEARTBEAT_SECONDS:
                    minutes = (time.monotonic() - began) / 60
                    state = f", {status.detail}" if status.detail else ""
                    _say(f"job {job_id} still running ({minutes:.0f} min{state})")
                    last_output = time.monotonic()
        except (KeyboardInterrupt, _Interrupted):
            _say(f"interrupted; stopping job {job_id}")
            launch.cancel(self.transport, submitted)
            raise KeyboardInterrupt from None
        finally:
            signal.signal(signal.SIGTERM, previous)

    def ensure_image(self, tag: str) -> str:
        """Make ``tag`` present on the remote, sending it from here if needed; its id."""

        inspect = "docker image inspect --format '{{.Id}}' "
        found = self.transport.run(inspect + q(tag), check=False)
        if found.returncode == 0:
            return found.stdout.strip()
        local = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}} {{.Size}}", tag],
            capture_output=True,
            text=True,
            check=False,
        )
        if local.returncode != 0:
            raise RemoteRunError(
                f"image {tag} is neither on {self.remote.name} nor here; build it first "
                "(python -m profiling.container.images build NAME)"
            )
        image_id, size = local.stdout.split()
        _say(f"sending image {tag} ({int(size) / 1e9:.1f} GB) to {self.remote.name}")
        save = subprocess.Popen(["docker", "save", tag], stdout=subprocess.PIPE)
        pack = subprocess.Popen(
            ["zstd", "-T0", "-3", "-q", "-c"], stdin=save.stdout, stdout=subprocess.PIPE
        )
        assert save.stdout is not None and pack.stdout is not None
        save.stdout.close()
        load = subprocess.run(
            self.transport.command_argv("zstd -dcq | docker load -q"),
            stdin=pack.stdout,
            capture_output=True,
            text=True,
            check=False,
        )
        pack.stdout.close()
        if save.wait() != 0 or pack.wait() != 0 or load.returncode != 0:
            raise RemoteRunError(f"sending image {tag} failed: {load.stderr.strip()}")
        sent = self.transport.run(inspect + q(tag)).stdout.strip()
        if sent != image_id:
            raise RemoteRunError(f"{self.remote.name} loaded {tag} as {sent}, not {image_id}")
        return sent

    def prepare_in_image(self, env_name: str, tag: str) -> str:
        """Run jobs on the interpreter of the profiler image the remote is; its inputs.

        The remote must be ``tag``, the image this checkout names for
        ``env_name``: its manifest (written by every profiler Dockerfile) has to
        carry the same inputs hash and serve the env. Nothing is built there.
        """

        from profiling.exec.env import IMAGE_MANIFEST

        source = self.source
        assert source is not None
        found = self.transport.run(f"cat {q(IMAGE_MANIFEST)}", check=False)
        try:
            manifest = json.loads(found.stdout)
        except ValueError:
            manifest = None
        if found.returncode != 0 or not isinstance(manifest, dict):
            raise RemoteRunError(
                f"{self.remote.name} has profile_runtime: in-image, but no profiler image "
                f"manifest at {IMAGE_MANIFEST}: it does not run a profiler image "
                "(or one built before the manifest existed)"
            )
        inputs = tag.rpartition(":")[2]
        if manifest.get("inputs") != inputs or env_name not in manifest.get("envs", ()):
            raise RemoteRunError(
                f"{env_name} needs image {tag}, but {self.remote.name} runs an image "
                f"with inputs {manifest.get('inputs')} serving {manifest.get('envs')}"
            )
        src = self.layout.src(source.version)
        venv = PurePosixPath(manifest["python"]).parents[1]
        self.transport.run(f"ln -sfn {q(venv)} {q(src / '.venv')}")
        self.in_image = True
        return inputs

    def fetch_job_file(self, job_id: str, name: str, local: Path) -> None:
        self.data.rsync(self.data.remote_spec(self.layout.job(job_id) / name), str(local))

    def record(
        self,
        local_root: Path,
        outcome: JobOutcome,
        command: list[str],
        *,
        copied_back: str | None = None,
        **extra: object,
    ) -> None:
        source = self.source
        assert source is not None
        record = {
            "schema_version": 1,
            "remote": self.remote.name,
            "host": self.transport.host_label,
            "remote_path": str(self.remote_logs_path(local_root)),
            "code_version": source.version,
            "bin_dir": None if self.bin_dir is None else str(self.bin_dir),
            "job_id": outcome.job_id,
            "slurm_job_id": outcome.slurm_id,
            "command": command,
            "started_at": outcome.started_at,
            "finished_at": outcome.finished_at,
            "exit_code": outcome.exit_code,
            "copied_back": copied_back
            or ("all" if self.remote.ephemeral else "without raw/cost_log and plots"),
            **extra,
        }
        local_root.mkdir(parents=True, exist_ok=True)
        (local_root / REMOTE_RECORD).write_text(json.dumps(record, indent=2) + "\n", "utf-8")


def _raise_interrupted(_signum, _frame):
    raise _Interrupted


def _report(job, status: str, **kwargs) -> None:
    """A managed status update that may not fail the run (e.g. an expired token)."""

    if job is None:
        return
    try:
        job.report(status, **kwargs)
    except Exception as exc:
        _warn(f"could not report managed status {status!r}: {exc}")


# ── inputs and outputs ────────────────────────────────────────────────────


def _require_repo_cwd() -> None:
    if Path.cwd().resolve() != REPO_ROOT.resolve():
        raise RemoteRunError(
            f"run --remote from the checkout root ({REPO_ROOT}): relative paths in the "
            "command must mean the same thing on the remote"
        )


def _under_logs(path: Path, what: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(LOGS.resolve()) or resolved == LOGS.resolve():
        raise RemoteRunError(f"{what} {path} must be a directory under logs/ for --remote")
    return resolved


def _input_files(paths: list[Path], source: Source) -> list[Path]:
    """Input files the remote lacks: the ones under ``logs/`` (which is not shipped).

    An input outside the checkout, or ignored by git outside ``logs/`` and
    ``trace/``, cannot reach the remote and is an error.
    """

    shipped = set(source.files)
    extra: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_relative_to(REPO_ROOT.resolve()):
            raise RemoteRunError(f"input {path} is outside the checkout; --remote cannot ship it")
        if resolved.is_relative_to(LOGS.resolve()):
            extra.append(resolved)
        elif resolved.relative_to(REPO_ROOT.resolve()).as_posix() not in shipped:
            raise RemoteRunError(f"input {path} is ignored by git, so --remote does not ship it")
    return extra


# ── commands ──────────────────────────────────────────────────────────────


def _simulation(session: Session, argv: list[str]) -> int:
    from ..__main__ import parse_simulation_args, plan_simulation
    from ..managed_run import EXPERIMENT_METADATA_FILENAME, prepare_experiment
    from ..sweep import _experiment_root, _sweep_axes, log_dir_of

    args = parse_simulation_args(argv)
    for flag, used in (
        ("--dry-run", args.dry_run),
        ("--cache-report", args.cache_report),
        ("--emit-backends", args.emit_backends is not None),
        ("--profile", args.profile),
    ):
        if used:
            raise RemoteRunError(f"{flag} runs locally; drop --remote")
    plan = plan_simulation(args)
    if isinstance(plan, int):
        return plan
    candidates = plan.candidates
    single = len(candidates) == 1
    root = Path(log_dir_of(candidates[0])) if single else _experiment_root(candidates)
    root = _under_logs(root, "the experiment root")

    source = session.ship()
    inputs = _input_files([Path(p) for p in args.presets], source)
    managed = prepare_experiment(
        root, run_count=len(candidates), axes=[] if single else _sweep_axes(candidates)
    )
    session.push_files([*inputs, root / EXPERIMENT_METADATA_FILENAME])
    session.prepare()
    _report(managed, "running")
    command = [*argv, "--no-gpu", "--no-plot"]
    try:
        outcome = session.run_job(command)
    except KeyboardInterrupt:
        _report(managed, "interrupted")
        raise
    _report(managed, "analysis_running")
    session.fetch(root, excludes=SIMULATION_COPY_EXCLUDES)
    session.record(root, outcome, command)
    _report(managed, "ready" if outcome.exit_code == 0 else "failed")
    _say(
        f"{'done' if outcome.exit_code == 0 else f'failed (exit {outcome.exit_code})'}: "
        f"{root.relative_to(REPO_ROOT)} (remote copy: "
        f"{session.transport.host_label}:{session.remote_logs_path(root)})"
    )
    return outcome.exit_code


def _timing_predict(session: Session, argv: list[str]) -> int:
    from ..managed_job import prepare_managed_job
    from ..timing_predict import (
        _build_argparse,
        _load_config,
        _predict_descriptor,
        _prediction_id,
    )

    args = _build_argparse().parse_intermixed_args(argv)
    if args.dry_run:
        raise RemoteRunError("--dry-run runs locally; drop --remote")
    if not args.configs:
        raise RemoteRunError("timing-predict: no config given")
    planned: list[tuple[Path, Path, dict]] = []
    inputs: list[Path] = []
    for name in args.configs:
        config_path = Path(name).resolve()
        if not config_path.is_file():
            raise RemoteRunError(f"{name}: not a file")
        cfg = _load_config(config_path)
        if not isinstance(cfg.get("log_dir"), str) or not cfg["log_dir"]:
            raise RemoteRunError(f"{name}: timing-predict config requires log_dir")
        # Relative to the checkout root, as `timing_predict.run_one` reads it.
        log_dir = _under_logs(REPO_ROOT / cfg["log_dir"], "log_dir")
        inputs.append(config_path)
        cases = cfg.get("cases_file")
        if isinstance(cases, str) and cases and not cases.startswith("hf://"):
            cases_path = Path(cases)
            if not cases_path.is_absolute():
                cases_path = config_path.parent / cases_path
            inputs.append(cases_path)
        planned.append((config_path, log_dir, cfg))

    source = session.ship()
    extra = _input_files(inputs, source)
    jobs = []
    seeds: list[Path] = []
    for config_path, log_dir, cfg in planned:
        prediction_id = _prediction_id(log_dir)
        job = prepare_managed_job(
            "timing_predict",
            log_dir,
            descriptor=_predict_descriptor(config_path, cfg),
            analyzer_resource_id=prediction_id,
        )
        jobs.append(job)
        if job is not None and not (log_dir / "prediction.meta.json").is_file():
            # The remote prediction keeps this id (`_prediction_id`).
            log_dir.mkdir(parents=True, exist_ok=True)
            seed = log_dir / "prediction.meta.json"
            seed.write_text(json.dumps({"prediction_id": prediction_id}) + "\n", "utf-8")
            seeds.append(seed)
    session.push_files([*extra, *seeds])
    # A seed is not a prediction's metadata; the remote writes the real file.
    for seed in seeds:
        seed.unlink()
    session.prepare()
    for job in jobs:
        _report(job, "running")
    command = ["timing-predict", *argv, "--no-gpu", "--no-plot"]
    try:
        outcome = session.run_job(command)
    except KeyboardInterrupt:
        for job in jobs:
            _report(job, "interrupted")
        raise
    for (config_path, log_dir, cfg), job in zip(planned, jobs, strict=True):
        session.fetch(log_dir)
        session.record(log_dir, outcome, command)
        metadata = log_dir / "prediction.meta.json"
        complete = metadata.is_file() and len(json.loads(metadata.read_text("utf-8"))) > 1
        if not complete:
            # A failed remote prediction leaves only the seed behind.
            metadata.unlink(missing_ok=True)
        if complete:
            _report(job, "ready", summary=_predict_descriptor(config_path, cfg))
        else:
            _report(job, "failed")
        _say(f"{'done' if complete else 'failed'}: {log_dir.relative_to(REPO_ROOT)}")
    return outcome.exit_code


def _merge_profile_db(job_db: Path, report: Path) -> str:
    """Fold a remote job's profile.db into this checkout's, under its write lease."""

    from profiling.db.merge import merge_profile_databases

    from ..process.leases import LauncherLeases

    target = Path(
        os.environ.get("VIBESIM_PROFILE_DB", REPO_ROOT / "profiling" / "profile.db")
    ).resolve()
    with LauncherLeases(REPO_ROOT).profile_database(write=True):
        merged = target.with_name(f".{target.name}.remote-{uuid.uuid4().hex[:8]}")
        result = merge_profile_databases(target, job_db, merged, report_path=report)
        if not result.published:
            raise RemoteRunError(
                f"the remote rows conflict with {target}; nothing was merged. "
                f"Review {report} (the job's DB is kept at {job_db})"
            )
        # The merge writes a private temporary file; keep the database's own mode.
        os.chmod(merged, target.stat().st_mode & 0o7777)
        os.replace(merged, target)
    inserted = sum(table.inserted_rows for table in result.tables)
    return f"{inserted} new row(s) merged into {target.relative_to(REPO_ROOT)}"


def _kernel_profile(session: Session, argv: list[str]) -> int:
    from profiling.artifacts import MEASUREMENT_METADATA_FILENAME, PROFILE_METADATA_FILENAME
    from profiling.cli import (
        _load_specs,
        _measurement_id,
        _profile_id,
        _resolve_profiler_spec,
        _spec_gpu_count,
        build_parser,
        measure_descriptor,
        measure_output_dir,
        run_descriptor,
    )
    from profiling.container.images import IMAGES_ENV
    from profiling.exec.env import PROFILE_RUNTIME_ENV, container_image

    from ..managed_job import prepare_managed_job

    args = build_parser(prog="python -m launcher kernel-profile").parse_args(argv)
    if args.command not in ("run", "measure"):
        raise RemoteRunError(f"kernel-profile {args.command} runs locally; drop --remote")
    if args.db is not None:
        raise RemoteRunError(
            "--db cannot be used with --remote: the remote job writes its own copy, "
            "merged back into this checkout's profile.db (VIBESIM_PROFILE_DB)"
        )
    measure = args.command == "measure"
    if args.output_dir is None and not measure:
        raise RemoteRunError("kernel-profile run --remote requires --output-dir under logs/")
    output_dir = _under_logs(
        measure_output_dir(args) if measure else args.output_dir, "--output-dir"
    )
    specs = _load_specs(args.spec, args.specs)
    profiler_spec = _resolve_profiler_spec(args.table, args.backend)
    image = container_image(profiler_spec.subprocess_env)
    if session.remote.slurm is not None:
        gpus = max(session.remote.slurm.gpus, _spec_gpu_count(profiler_spec, specs))
        session.remote = dataclasses.replace(
            session.remote, slurm=dataclasses.replace(session.remote.slurm, gpus=gpus)
        )

    source = session.ship()
    extra = _input_files([args.specs] if args.specs is not None else [], source)
    if measure:
        kind, metadata_name, id_key = (
            "kernel_measure",
            MEASUREMENT_METADATA_FILENAME,
            "measurement_id",
        )
        resource_id, descriptor = _measurement_id(output_dir), measure_descriptor(args)
    else:
        kind, metadata_name, id_key = "kernel_profile", PROFILE_METADATA_FILENAME, "profile_id"
        resource_id = _profile_id(output_dir)
        descriptor = run_descriptor(args, profiler_spec, specs)
    job = prepare_managed_job(
        kind, output_dir, descriptor=descriptor, analyzer_resource_id=resource_id
    )
    seeds: list[Path] = []
    if job is not None and not (output_dir / metadata_name).is_file():
        # The remote run keeps this id (`_profile_id` / `_measurement_id`).
        output_dir.mkdir(parents=True, exist_ok=True)
        seed = output_dir / metadata_name
        seed.write_text(json.dumps({id_key: resource_id}) + "\n", "utf-8")
        seeds.append(seed)
    session.push_files([*extra, *seeds])
    for seed in seeds:
        seed.unlink()
    env_name = profiler_spec.subprocess_env or "default_env"
    in_image = session.remote.profile_runtime == "in-image"
    if in_image:
        # The remote is the image: controller and workers run on its interpreter.
        image_id = session.prepare_in_image(env_name, image)
        _say(f"{args.table}/{args.backend} runs on {session.remote.name}, which is {image}")
    else:
        # The controller runs as it does here: on the project's profiling stack.
        session.prepare(job_groups=(*CPU_GROUPS, "profiling"))
        image_id = session.ensure_image(image)
        _say(f"{args.table}/{args.backend} runs in {image} ({image_id[:19]})")
    _report(job, "running")
    command = ["kernel-profile", *argv]
    keep_db = not measure and not args.fresh
    try:
        outcome = session.run_job(
            command,
            env={
                PROFILE_RUNTIME_ENV: "in-image" if in_image else "container",
                # The shipped tree has no git to compute the tag from.
                IMAGES_ENV: json.dumps({env_name: image}),
                "VIBESIM_PROFILE_CACHE_DIR": str(session.layout.workdir / "cache"),
            },
            keep_db=keep_db,
        )
    except KeyboardInterrupt:
        _report(job, "interrupted")
        raise
    _report(job, "analysis_running")
    session.fetch(output_dir)
    merged = "not merged (the run wrote no rows)"
    if keep_db and outcome.exit_code in (0, 1):
        # Exit 1 is a run with missing rows: the rows it did measure still count.
        held = Path(tempfile.mkdtemp(prefix="vibesim-remote-db-"))
        job_db = held / "profile.db"
        session.fetch_job_file(outcome.job_id, "profile.db", job_db)
        merged = _merge_profile_db(job_db, output_dir / "remote-profile-db.merge-report.json")
        job_db.unlink()
        held.rmdir()
        _say(merged)
    session.record(
        output_dir,
        outcome,
        command,
        copied_back="all",
        image=image,
        # A pod has no docker to name an image id; its manifest names the inputs.
        **({"image_inputs": image_id} if in_image else {"image_id": image_id}),
        profile_db=merged,
    )
    metadata = output_dir / metadata_name
    complete = metadata.is_file() and len(json.loads(metadata.read_text("utf-8"))) > 1
    if not complete:
        # A failed remote run leaves only the seed behind.
        metadata.unlink(missing_ok=True)
    ok = outcome.exit_code == 0 and complete
    _report(job, "ready" if ok else "failed")
    _say(
        f"{'done' if ok else f'failed (exit {outcome.exit_code})'}: "
        f"{output_dir.relative_to(REPO_ROOT)} (remote copy: "
        f"{session.transport.host_label}:{session.remote_logs_path(output_dir)})"
    )
    return outcome.exit_code


COMMANDS: dict[str | None, Callable[[Session, list[str]], int]] = {
    None: _simulation,
    "timing-predict": _timing_predict,
    "kernel-profile": _kernel_profile,
}


def run(subcommand: str | None, argv: list[str], options: GlobalOptions) -> int:
    if subcommand not in COMMANDS:
        print(
            f"python -m launcher: --remote {subcommand} is not implemented yet",
            file=sys.stderr,
        )
        return 2
    try:
        _require_repo_cwd()
        remote = load_remote(options.remote or "", options.remote_set)
        if remote.profile_runtime == "in-image" and subcommand != "kernel-profile":
            raise RemoteRunError(
                f"{remote.name} is a profiler image (profile_runtime: in-image), "
                "which runs kernel-profile only"
            )
        return COMMANDS[subcommand](Session(remote), argv)
    except (RemoteConfigError, RemoteRunError, SourceError, TransportError) as exc:
        print(f"python -m launcher --remote: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
