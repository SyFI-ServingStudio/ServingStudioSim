"""Register the kernel configs a run or prediction asks profile.db for.

The simulator writes them with `--kernel-configs-out FILE` (`build-cache-only`,
`dry-run`, `timing-predict`); this module adds what the simulator does not know
-- which preset or predict config, pool and arch built them -- and hands both
to `profiling.db.kernel_config.register_kernel_configs`.

Registration happens where rows are measured, so the rows and the configs that
asked for them arrive together:

- after a cache prebuild that profiled missing rows (`cache_build`);
- after a timing-predict run, which JIT-profiles what it misses.

`--register-kernel-configs` (presets), `timing-predict
--register-kernel-configs` (predict configs) and
`--register-supported-kernel-configs` (every `#[supported]` arch deployment)
register without measuring anything: they run the simulator's `dry-run` or
`supported-cost-trees`, which need no GPU. That is how configs whose rows were
measured before the registry existed get registered, so they skip configs with
no measured row at all.
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from profiling.db.kernel_config import RegisterReport, register_kernel_configs
from profiling.perf_api import DB_PATH

from .corpus import restore_hf_references
from .exec import REPO_ROOT, _build_subprocess_env, binary_path
from .process import ProcessSpec, ProcessSupervisor
from .process.leases import LauncherLeases
from .schema import build_cli_command

_PROCESS_SUPERVISOR = ProcessSupervisor()
_LAUNCHER_LEASES = LauncherLeases(REPO_ROOT)


def run_sources(config: Mapping[str, Any], preset: str | None) -> dict[str, dict[str, Any]]:
    """What built each pool of a run config: the preset, and the pool's GPU and
    arch per group. Pool `""` is the deployment itself, for kernels built
    outside any pool (AFD's attention-to-FFN transfer). An artifact the preset
    names by `hf://` reference is recorded by that reference, not by the local
    file it resolved to."""
    config = restore_hf_references(config)
    base = {"preset": _repo_relative(preset), "deployment": config["deployment"]}
    sources: dict[str, dict[str, Any]] = {"": {**base, "pool": None}}
    for pool, body in config["pools"].items():
        sources[pool] = {
            **base,
            "pool": pool,
            "groups": [{"gpu": group["gpu"], "arch": group["arch"]} for group in body["groups"]],
        }
    return sources


def predict_source(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    """What built a prediction's configs: the predict config, its GPU and arch."""
    return {
        "timing_predict": _repo_relative(str(config_path)),
        "gpu": config["gpu"],
        "arch": config["arch"],
    }


def supported_source(build: Mapping[str, Any]) -> dict[str, Any]:
    """What built a supported deployment's configs: the `#[supported]` row's
    arch, GPU and params, as `simulator supported-cost-trees` reports them."""
    return {"supported": {"arch": build["arch"], "gpu": build["gpu"], "params": build["params"]}}


def register_file(
    records: Path,
    sources: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    source: Mapping[str, Any] | None = None,
    measured_only: bool = False,
) -> RegisterReport:
    """Register a `--kernel-configs-out` file in profile.db.

    Pass `sources` per pool, or one `source` for every pool the file names (a
    prediction builds one model, so all of its configs share one source).
    `measured_only`: see `register_kernel_configs`.
    """
    document = json.loads(records.read_text())
    if source is not None:
        pools = {use["pool"] for config in document["configs"] for use in config["uses"]}
        sources = {pool: source for pool in pools}
    if sources is None:
        raise ValueError("register_file needs `sources` or `source`")
    return register_kernel_configs(DB_PATH, document, sources, measured_only=measured_only)


def describe(report: RegisterReport) -> str:
    unmeasured = (
        f", {report.configs_unmeasured} skipped with no measured row"
        if report.configs_unmeasured
        else ""
    )
    if not report.written:
        return f"{report.configs} kernel configs, nothing new to register{unmeasured}"
    return (
        f"{report.configs} kernel configs: {report.configs_added} new, "
        f"{report.configs_regridded} with a changed grid, {report.uses_added} new uses"
        f"{unmeasured}"
    )


def register_run_configs(candidates: list[tuple[str, dict]], build_type: str) -> int:
    """`--register-kernel-configs`: dry-run each run config and register what
    it asks for. `candidates` pairs each expanded config with its preset.
    Returns a process exit code."""
    binary = binary_path(build_type)
    env = _build_subprocess_env()
    rc = 0
    with tempfile.TemporaryDirectory(prefix="register-kernel-configs-") as scratch:
        for i, (preset, config) in enumerate(candidates):
            workdir = Path(scratch) / str(i)
            records = workdir / "kernel_configs.json"
            argv = build_cli_command(config, binary, workdir / "run_config.yaml", "dry-run")
            argv += ["--kernel-configs-out", str(records)]
            label = f"{preset} ({config['io']['log_dir']})"
            if not _run(argv, env, label):
                rc = 1
                continue
            with _LAUNCHER_LEASES.profile_database(write=True):
                report = register_file(records, run_sources(config, preset), measured_only=True)
            print(f"[kernel-configs] {label}: {describe(report)}")
    return rc


def register_supported_configs(build_type: str) -> int:
    """`--register-supported-kernel-configs`: register the kernel configs of
    every `#[supported]` arch deployment. Returns a process exit code."""
    argv = [str(binary_path(build_type)), "supported-cost-trees", "--kernel-configs"]
    result = _PROCESS_SUPERVISOR.run_sync(
        ProcessSpec(
            argv=argv,
            cwd=REPO_ROOT,
            env={"RUST_LOG": "warn", **_build_subprocess_env()},
            capture_output=True,
            separate_stderr=True,
            name="register-kernel-configs",
        )
    )
    if not result.succeeded:
        print(f"[kernel-configs] supported-cost-trees failed\n{result.stderr}", file=sys.stderr)
        return 1
    rc = 0
    for build in json.loads(result.output):
        label = f"{build['arch']} {build['gpu']} {json.dumps(build['params'])}"
        if build["error"]:
            print(f"[kernel-configs] {label}: {build['error']}", file=sys.stderr)
            rc = 1
            continue
        document = build["kernel_configs"]
        pools = {use["pool"] for config in document["configs"] for use in config["uses"]}
        source = supported_source(build)
        with _LAUNCHER_LEASES.profile_database(write=True):
            report = register_kernel_configs(
                DB_PATH, document, dict.fromkeys(pools, source), measured_only=True
            )
        print(f"[kernel-configs] {label}: {describe(report)}")
    return rc


def register_predict_configs(
    config_path: Path, config: dict, resolved: dict | None, build_type: str
) -> bool:
    """`timing-predict --register-kernel-configs`: dry-run one predict config
    and register what it asks for. `config` is the file as written, which the
    source records; `resolved` is a copy with hub references fetched, or None
    when the binary can read the file itself."""
    with tempfile.TemporaryDirectory(prefix="register-kernel-configs-") as scratch:
        binary_config = config_path
        if resolved is not None:
            binary_config = Path(scratch) / "predict.json"
            binary_config.write_text(json.dumps(resolved))
        records = Path(scratch) / "kernel_configs.json"
        argv = [
            str(binary_path(build_type)),
            "timing-predict",
            "--dry-run",
            "--kernel-configs-out",
            str(records),
            str(binary_config),
        ]
        if not _run(argv, _build_subprocess_env(), str(config_path)):
            return False
        with _LAUNCHER_LEASES.profile_database(write=True):
            report = register_file(
                records, source=predict_source(config, config_path), measured_only=True
            )
    print(f"[kernel-configs] {config_path}: {describe(report)}")
    return True


def _run(argv: list[str], env: dict[str, str], label: str) -> bool:
    result = _PROCESS_SUPERVISOR.run_sync(
        ProcessSpec(
            argv=argv,
            cwd=REPO_ROOT,
            env={"RUST_LOG": "warn", **env},
            capture_output=True,
            name="register-kernel-configs",
        )
    )
    if not result.succeeded:
        print(f"[kernel-configs] {label}: dry-run failed\n{result.output}", file=sys.stderr)
    return result.succeeded


def _repo_relative(path: str | None) -> str | None:
    """A path as the repository names it, so sources match across checkouts."""
    if path is None:
        return None
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)
