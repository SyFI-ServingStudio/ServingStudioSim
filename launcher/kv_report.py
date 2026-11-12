"""KV cache report over a preset — `launcher <preset> --kv-report[-json]`.

Runs the binary's `kv-cache` subcommand, which builds the flow structure only
(no Python, profile.db or GPU) and prints each KV pool's layout, the charge of
one request at `max_model_len`, and the resulting concurrency. A sweep expands
to many candidates, but the workload and io blocks never change KV capacity, so
candidates that differ only there share one run of the binary; candidates
whose reports come out identical are then printed once, under every run's
log_dir.
"""

from __future__ import annotations

import json
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .exec import REPO_ROOT, _build_subprocess_env, binary_path
from .process import ProcessSpec, ProcessSupervisor
from .schema import build_cli_command, log_dir_of, strip_internal

SCHEMA_VERSION = 1
# Blocks that never reach KV capacity: what arrives, and where logs go.
_KV_IRRELEVANT_BLOCKS = ("workload", "io")
_PARALLELISM = 16

_PROCESS_SUPERVISOR = ProcessSupervisor()


def kv_report_key(config: dict) -> str:
    """The part of a config that can change its KV report, as a stable string."""
    rest = {
        key: value
        for key, value in strip_internal(config).items()
        if key not in _KV_IRRELEVANT_BLOCKS
    }
    return json.dumps(rest, sort_keys=True, default=str)


def group_by_kv_report(param_sets: list[dict]) -> list[list[dict]]:
    """Candidates grouped by `kv_report_key`, in first-seen order."""
    groups: dict[str, list[dict]] = {}
    for config in param_sets:
        groups.setdefault(kv_report_key(config), []).append(config)
    return list(groups.values())


def report_kv_cache(
    param_sets: list[dict], build_type: str = "release", *, as_json: bool = False
) -> int:
    """Print the KV report of every distinct candidate (JSON with `as_json`).
    Returns a process exit code: 0 iff every report was built."""
    binary = binary_path(build_type)
    env = _build_subprocess_env()
    groups = group_by_kv_report(param_sets)
    with tempfile.TemporaryDirectory(prefix="kv-report-") as scratch:

        def build(index: int):
            argv = build_cli_command(
                groups[index][0],
                binary,
                Path(scratch) / f"{index}.yaml",
                subcommand="kv-cache",
            )
            if as_json:
                argv.append("--json")
            return _PROCESS_SUPERVISOR.run_sync(
                ProcessSpec(
                    argv=argv,
                    cwd=REPO_ROOT,
                    env=env,
                    capture_output=True,
                    separate_stderr=True,
                    name="kv-report",
                )
            )

        with ThreadPoolExecutor(max_workers=min(_PARALLELISM, len(groups) or 1)) as pool:
            results = list(pool.map(build, range(len(groups))))

    rc = 0
    # Configs that differ elsewhere (a scheduling knob) can still print the
    # same report; show each distinct report once, listing every run behind it.
    by_output: dict[str, list[str]] = {}
    for members, result in zip(groups, results):
        log_dirs = [log_dir_of(config) for config in members]
        if not result.succeeded:
            rc = result.exit_code or 1
            print(
                f"kv-report failed for {log_dirs[0]}:\n{result.stderr or result.output}",
                file=sys.stderr,
            )
            continue
        by_output.setdefault(result.output, []).extend(log_dirs)
    if as_json:
        reports = [
            {"log_dirs": log_dirs, "kv_cache": json.loads(output)}
            for output, log_dirs in by_output.items()
        ]
        print(json.dumps({"schema_version": SCHEMA_VERSION, "reports": reports}, indent=2))
    else:
        for output, log_dirs in by_output.items():
            print(_run_heading(log_dirs))
            print(output.rstrip("\n"))
            print()
    return rc


def _run_heading(log_dirs: list[str]) -> str:
    if len(log_dirs) == 1:
        return f"== {log_dirs[0]}"
    shown = ", ".join(log_dirs[:3]) + (", ..." if len(log_dirs) > 3 else "")
    return f"== {len(log_dirs)} runs share this report: {shown}"
