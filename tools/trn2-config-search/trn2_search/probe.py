"""Classify configs before a sweep: buildable, invalid, or needing profiling.

The launcher aborts a whole sweep when any cache key fails its prebuild (and
under `--no-gpu` a missing profile.db row is such a failure), so every config
is probed alone first: its one-run preset is expanded by the launcher's own
`expand_preset` and the simulator's `dry-run` builds the arch (layout checks,
HBM check, every profile.db lookup) and counts the missing rows. Nothing is
profiled and no device is touched.

Status per config:
- `ok`: builds, 0 missing rows;
- `needs_profiling`: builds, but the named kernels have missing rows;
- `invalid`: the build refused it (the simulator's error, last line).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .presets import render


def _launcher():
    # Imported lazily: the launcher pulls in its schema loader and asyncio.
    from launcher.__main__ import expand_preset
    from launcher.exec import _build_subprocess_env, binary_path
    from launcher.schema import build_cli_command
    from launcher.schema.loader import load_schema

    return expand_preset, _build_subprocess_env, binary_path, build_cli_command, load_schema


def probe_configs(
    configs: list[dict],
    workload: tuple[str, str],
    work_dir: Path,
    jobs: int = 32,
    build_type: str = "release",
) -> list[dict]:
    """Probe each config with one workload; returns `{label, status, detail}`."""
    expand_preset, sub_env, binary_path, build_cli_command, load_schema = _launcher()
    schema = load_schema(build_type)
    binary = binary_path(build_type)
    env = sub_env()
    env["SERVINGSTUDIO_NO_GPU"] = "1"
    work_dir.mkdir(parents=True, exist_ok=True)

    def one(cfg: dict) -> dict:
        d = work_dir / cfg["label"]
        d.mkdir(parents=True, exist_ok=True)
        preset = render(cfg["family"], [cfg], [workload], str(d / "run"))
        try:
            (cand,) = expand_preset(preset, schema, str(d / "preset.yaml"))
        except Exception as exc:  # the launcher's own validation refused it
            return {"label": cfg["label"], "status": "invalid", "detail": f"preset: {exc}"}
        report = d / "dry_run_report.json"
        argv = build_cli_command(cand, binary, d / "run_config.yaml", subcommand="dry-run")
        argv += ["--report-json", str(report)]
        res = subprocess.run(argv, env=env, capture_output=True, text=True, cwd=os.getcwd())
        (d / "dry_run.log").write_text(res.stdout + res.stderr)
        if res.returncode != 0:
            lines = [ln.strip() for ln in (res.stdout + res.stderr).splitlines() if ln.strip()]
            return {
                "label": cfg["label"],
                "status": "invalid",
                "detail": lines[-1] if lines else "?",
            }
        rep = json.loads(report.read_text())
        missing = [k for k in rep["kernels"] if k["missing"]]
        if missing:
            detail = ", ".join(f"{k['kind']}({k['missing']}/{k['total']})" for k in missing)
            return {"label": cfg["label"], "status": "needs_profiling", "detail": detail}
        return {"label": cfg["label"], "status": "ok", "detail": ""}

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(one, configs))


def default_work_dir() -> Path:
    return Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "trn2_search_probe"
