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
import subprocess
import sys
import sysconfig
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARALLELISM = 200


def binary_path(build_type: str = "debug") -> Path:
    return REPO_ROOT / "target" / build_type / "simulator"


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
            env["LD_LIBRARY_PATH"] = (
                f"{libdir}:{current_ld_path}" if current_ld_path else libdir
            )

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


def cargo_build(build_type: str = "debug") -> bool:
    """Build the simulator crate, then run schema discovery (INV-8). Returns
    True on success; on failure no schema is written (design §1.2.7 bootstrap
    edge — `load_schema` then raises `SchemaNotFound`)."""
    cmd = ["cargo", "build"]
    if build_type == "release":
        cmd.append("--release")
    if subprocess.run(cmd, cwd=REPO_ROOT).returncode != 0:
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
    return True


@dataclass
class SimulationRunner:
    """One subprocess run. `argv` comes from `schema.build_cli_command`."""

    argv: list[str]
    log_dir: Path
    env: dict[str, str] = field(default_factory=_build_subprocess_env)
    _proc: asyncio.subprocess.Process | None = field(default=None, init=False, repr=False)

    async def run(self) -> bool:
        """Spawn, stream stdout/stderr to `stdout.log`, return success."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stdout_log = self.log_dir / "stdout.log"
        self._proc = await asyncio.create_subprocess_exec(
            *self.argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=REPO_ROOT,
            env=self.env,
        )
        assert self._proc.stdout is not None
        with stdout_log.open("w") as fh:
            async for line_bytes in self._proc.stdout:
                fh.write(line_bytes.decode(errors="replace"))
                fh.flush()
        return await self._proc.wait() == 0

    def cancel(self) -> None:
        if self._proc and self._proc.returncode is None:
            self._proc.kill()
