"""The GLM-5.3-Flash PP4 prefill presets (`presets/glm53_flash_prefill/`): every
member expands against the simulator's schema and builds its deployment
through the binary's ``dry-run`` (the flow is built, nothing simulated)."""

from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from launcher.__main__ import expand_preset
from launcher.exec import ERROR_JSON, _build_subprocess_env, binary_error
from launcher.schema import strip_internal
from launcher.schema.argv import build_cli_command
from launcher.schema.loader import _load_preset, load_schema

REPO_ROOT = Path(__file__).resolve().parent.parent
PRESET_DIR = REPO_ROOT / "presets/glm53_flash_prefill"
PRESETS = sorted(path for path in PRESET_DIR.glob("*.yaml") if not path.stem.startswith("backends"))
JOBS = min(32, os.cpu_count() or 8)


def _members(path: Path) -> list[dict]:
    return expand_preset(_load_preset(path), load_schema("release"), str(path))


def _missing_traces(member: dict) -> list[str]:
    return [
        trace
        for trace in member["workload"]["trace_files"]
        if not (REPO_ROOT / trace).exists()
    ]


def test_the_presets_are_there():
    names = {path.stem for path in PRESETS}
    assert {"ssengine", "check_scheduler_order", "check_tier_read", "v3_m6_full"} <= names


@pytest.mark.presets
@pytest.mark.needs_binary
@pytest.mark.parametrize("path", PRESETS, ids=[path.stem for path in PRESETS])
def test_every_member_builds(path, sim_bin, tmp_path):
    members = _members(path)
    assert members
    runnable = [member for member in members if not _missing_traces(member)]
    if not runnable:
        # The v3 session traces are not committed (their manifests are).
        pytest.skip(f"trace not provisioned: {_missing_traces(members[0])}")
    env = {**_build_subprocess_env(), "RUST_LOG": "warn"}

    def build(index: int) -> str | None:
        scratch = tmp_path / str(index)
        scratch.mkdir()
        error = scratch / ERROR_JSON
        argv = build_cli_command(
            strip_internal(runnable[index]),
            sim_bin,
            scratch / "run_config.yaml",
            "dry-run",
            error_json=error,
        )
        result = subprocess.run(
            [*argv, "--report-json", str(scratch / "report.json")],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return None
        return f"{runnable[index]['io']['log_dir']}: {binary_error(error) or result.stderr[-2000:]}"

    with ThreadPoolExecutor(max_workers=min(JOBS, len(runnable))) as pool:
        failures = [failure for failure in pool.map(build, range(len(runnable))) if failure]
    assert not failures, "\n".join(failures)
