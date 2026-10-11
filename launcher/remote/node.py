"""The remote half of `--remote`: build a shipped version, run one job.

The sending launcher calls this module on the remote with the shipped tree's
own interpreter, ``src/<version>/.venv/bin/python -m launcher.remote.node``,
so it runs the same code as the job it serves:

``build WORKDIR VERSION``
    Give ``src/<version>`` its binaries. Cargo always builds in the fixed
    ``build/src`` checkout and one shared target dir, so a version that changed
    only Python is a no-op build, and a Rust edit recompiles only this
    repository's crates. The outputs are copied to ``bin/<build_hash>``, named
    by their own build inputs, so versions that compile the same sources share
    one set and a running job's binaries are never replaced.

``run-job JOBDIR``
    Run the launcher command recorded in ``JOBDIR/job.json`` and record its
    output and exit code there. The job gets a private copy of profile.db on
    node-local disk (SQLite must not run on a network filesystem), a node-local
    TMPDIR (the launcher's leases are flocks there), the prebuilt binaries, and
    no managed-job context: the sending machine registered the job and reports
    its status, so the remote run must not register a second one.

Builds are serialized by a lock on the build host's local disk. A remote's
builds always run on its access host (`run.py`), so the lock never has to
work across machines.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from ..binaries import (
    BIN_DIR_ENV,
    build_inputs_hash,
    check_prebuilt,
    dep_info_inputs,
    record_build,
)
from .layout import Layout

# The managed-job context variables (launcher/managed_client.py).
MANAGED_CONTEXT_VARIABLES = ("VIBESIM_MANAGED_JOB_CONTEXT", "VIBESIM_MANAGED_RUN_CONTEXT")
PROFILE_DB_ENV = "VIBESIM_PROFILE_DB"
_BUILD_EXCLUDES = ("/target", "/.venv", "/bin", "/logs")
_BUILT_FILES = ("simulator", "analyze")
# Written last when a version has been shipped completely (`run.py`).
SHIPPED_MARKER = ".shipped"


@contextmanager
def _host_lock(workdir: Path, purpose: str, node_local: Path):
    key = hashlib.sha256(str(workdir).encode()).hexdigest()[:12]
    node_local.mkdir(parents=True, exist_ok=True)
    path = node_local / f"vibesim-remote-{key}-{purpose}.lock"
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _relink(link: Path, target: Path) -> None:
    """Point ``link`` at ``target`` atomically (a rename over the old link)."""

    temporary = link.with_name(f".{link.name}.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(os.path.relpath(target, link.parent))
    temporary.replace(link)


def build(workdir: Path, version: str, node_local: Path) -> Path:
    layout = Layout(PurePosixPath(workdir))
    src = Path(layout.src(version))
    link = src / "bin"
    with _host_lock(workdir, "build", node_local):
        if link.exists():
            try:
                check_prebuilt(link.resolve(), src)
                print(f"[remote] {version}: binaries ready ({link.resolve().name})")
                return link.resolve()
            except Exception as exc:  # rebuild below
                print(f"[remote] {version}: rebuilding ({exc})")

        build_src, target = Path(layout.build_src), Path(layout.build_target)
        build_src.mkdir(parents=True, exist_ok=True)
        target.mkdir(parents=True, exist_ok=True)
        # rsync -a keeps the sending machine's mtimes, so cargo sees an
        # unchanged file as unchanged and the shared target stays incremental.
        excludes = [f"--exclude={pattern}" for pattern in _BUILD_EXCLUDES]
        subprocess.run(
            ["rsync", "-a", "--delete", *excludes, f"{src}/", f"{build_src}/"], check=True
        )
        _relink(build_src / "target", target)
        _relink(build_src / ".venv", (src / ".venv").resolve())

        env = {k: v for k, v in os.environ.items() if k != BIN_DIR_ENV}
        env["VIRTUAL_ENV"] = str((src / ".venv").resolve())
        compiled = subprocess.run(
            [
                str(build_src / ".venv" / "bin" / "python"),
                "-c",
                "import sys; from launcher.exec import cargo_build; "
                "sys.exit(0 if cargo_build('release') else 1)",
            ],
            cwd=build_src,
            env=env,
            check=False,
        )
        if compiled.returncode != 0:
            raise SystemExit(f"[remote] {version}: build failed")

        release = target / "release"
        dep_info = [release / f"{name}.d" for name in _BUILT_FILES if (release / name).is_file()]
        inputs = dep_info_inputs(build_src, dep_info)
        if not any(name.endswith(".rs") for name in inputs):
            raise SystemExit(f"[remote] {version}: no Rust sources in {dep_info}; cannot pin")
        build_hash = build_inputs_hash(build_src, inputs)
        destination = Path(layout.bin(build_hash))
        if not destination.is_dir():
            staging = destination.with_name(f".{build_hash}.{os.getpid()}")
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir(parents=True)
            for name in (*_BUILT_FILES, "deployment_schema.json"):
                if (release / name).is_file():
                    shutil.copy2(release / name, staging / name)
            record_build(staging, build_src, inputs)
            staging.rename(destination)
        _relink(link, destination)
        check_prebuilt(destination, src)
        print(f"[remote] {version}: binaries {build_hash[:16]}")
        return destination


def _job_environment(job: dict, scratch: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in MANAGED_CONTEXT_VARIABLES}
    env.update(job.get("env", {}))
    tmp = scratch / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(tmp)
    env[BIN_DIR_ENV] = job["bin_dir"]
    env["VIRTUAL_ENV"] = str((Path(job["src"]) / ".venv").resolve())
    env[PROFILE_DB_ENV] = str(scratch / "profile.db")
    return env


def run_job(jobdir: Path) -> int:
    job = json.loads((jobdir / "job.json").read_text("utf-8"))
    (jobdir / "pid").write_text(f"{os.getpid()}\n")
    src = Path(job["src"])
    scratch = Path(job["node_local"]) / f"vibesim-job-{job['job_id']}"
    scratch.mkdir(parents=True, exist_ok=True)
    exit_code = 1
    try:
        shutil.copy2(src / "profiling" / "profile.db", scratch / "profile.db")
        argv = [str(src / ".venv" / "bin" / "python"), "-m", "launcher", *job["argv"]]
        with (jobdir / "stdout.log").open("ab") as log:
            child = subprocess.Popen(
                argv,
                cwd=src,
                env=_job_environment(job, scratch),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

            def forward(signum, _frame):
                try:
                    os.killpg(child.pid, signum)
                except ProcessLookupError:
                    pass

            for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                signal.signal(signum, forward)
            exit_code = child.wait()
            if exit_code < 0:
                exit_code = 128 - exit_code
        if job.get("keep_db"):
            shutil.copy2(scratch / "profile.db", jobdir / "profile.db")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        temporary = jobdir / "exit_code.tmp"
        temporary.write_text(f"{exit_code}\n")
        temporary.replace(jobdir / "exit_code")
    return exit_code


def prune(workdir: Path, keep: int) -> None:
    """Delete all but the ``keep`` newest shipped versions, and orphaned binaries.

    A version with an unfinished job is kept regardless. Experiments under
    ``logs/`` are never touched.
    """

    layout = Layout(PurePosixPath(workdir))
    src_root = Path(layout.workdir / "src")
    shipped = sorted(
        (path for path in src_root.iterdir() if (path / SHIPPED_MARKER).is_file()),
        key=lambda path: (path / SHIPPED_MARKER).stat().st_mtime,
        reverse=True,
    )
    busy = set()
    jobs_root = Path(layout.workdir / "jobs")
    for jobdir in jobs_root.iterdir() if jobs_root.is_dir() else ():
        if not (jobdir / "exit_code").exists() and (jobdir / "job.json").is_file():
            try:
                busy.add(json.loads((jobdir / "job.json").read_text("utf-8"))["version"])
            except (OSError, ValueError, KeyError):
                continue
    for path in shipped[keep:]:
        if path.name not in busy:
            print(f"[remote] pruning version {path.name}")
            shutil.rmtree(path, ignore_errors=True)
    in_use = {
        (path / "bin").resolve() for path in src_root.iterdir() if (path / "bin").is_symlink()
    }
    bin_root = Path(layout.workdir / "bin")
    for directory in bin_root.iterdir() if bin_root.is_dir() else ():
        if directory.resolve() not in in_use and not directory.name.startswith("."):
            print(f"[remote] pruning binaries {directory.name[:16]}")
            shutil.rmtree(directory, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m launcher.remote.node")
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("workdir", type=Path)
    build_parser.add_argument("version")
    build_parser.add_argument("--node-local", type=Path, default=Path("/tmp"))
    run_parser = commands.add_parser("run-job")
    run_parser.add_argument("jobdir", type=Path)
    prune_parser = commands.add_parser("prune")
    prune_parser.add_argument("workdir", type=Path)
    prune_parser.add_argument("--keep", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        build(args.workdir, args.version, args.node_local)
        return 0
    if args.command == "prune":
        prune(args.workdir, args.keep)
        return 0
    return run_job(args.jobdir)


if __name__ == "__main__":
    sys.exit(main())
