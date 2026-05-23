"""Profile.db cache prebuild — the SQLite / GPU-contention fix (design §1.2.2).

A cache-miss `run` JIT-profiles the missing kernel on the GPU and writes it to
`profile.db` (SQLite). If N concurrent sweep runs share a cache key and the
kernel is missing, they (a) launch real CUDA kernels at once → wrong timing
from contention, and (b) open profile.db for write at once → SQLite lock
errors. So before any parallel launch we prebuild each unique `cache_key`
**sequentially** with `build-cache-only` (INV-3). A single distinct key is a
no-op of one prebuild.

**Which params form the key is Rust-authoritative.** The set of
kernel-determining params is L1 knowledge (only Rust knows which params flow
into `*KernelInput` / profile.db lookup keys), so it is NOT hardcoded here:
`cache_key` takes the field names the Rust schema tags with `affects_cache`
(`DeploymentSchema.cache_key_fields`). When Rust adds a kernel-shaping param it
tags it, and the launcher picks it up with no Python edit. See `param_def.rs`
for the safe-direction rule (when unsure, tag it — over-tagging only costs extra
prebuild passes; under-tagging reintroduces the contention bug).
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from pathlib import Path

from .exec import SimulationRunner, _build_subprocess_env, binary_path
from .schema import build_cli_command
from .schema.loader import Schema


def cache_key(params: dict, cache_fields: Iterable[str]) -> tuple:
    """The kernel-determining subset of `params`, hashable for de-duplication.
    `cache_fields` comes from `DeploymentSchema.cache_key_fields` (Rust-tagged)."""
    return tuple(params.get(field) for field in cache_fields)


def _unique_by_cache_key(
    param_sets: list[dict], schema: Schema
) -> list[dict]:
    """One representative param set per distinct cache key — the dedup shared by
    the cache prebuild and the coverage report. Key fields are Rust-tagged
    (`affects_cache`), so runs differing only in non-kernel params collapse."""
    seen_keys: set[tuple] = set()
    representatives: list[dict] = []
    for params in param_sets:
        cache_fields = schema.deployment_schemas[params["deployment"]].cache_key_fields
        key = cache_key(params, cache_fields)
        if key not in seen_keys:
            seen_keys.add(key)
            representatives.append(params)
    return representatives


def report_cache_coverage(
    param_sets: list[dict], schema: Schema, build_type: str = "debug"
) -> int:
    """Run the Rust `dry-run` subcommand once per unique cache key and stream its
    per-kernel missing-spec report to the console (no caches built, no sim run).
    Mirrors `prebuild_caches`'s cache-key dedup so a sweep reports each distinct
    kernel set once. Returns a process exit code (0 iff every probe succeeded)."""
    binary = binary_path(build_type)
    env = _build_subprocess_env()
    rc = 0
    for params in _unique_by_cache_key(param_sets, schema):
        argv = build_cli_command(params, binary, subcommand="dry-run")
        result = subprocess.run(argv, env=env)
        if result.returncode != 0:
            rc = result.returncode
    return rc


def _prebuild_log_dir(base_dir: Path, params: dict) -> Path:
    """Where a prebuild's stdout lands — under a `.cache_build/` subdir of the
    experiment's base logging directory (`base_dir`), named after the run's log
    folder (its path relative to `base_dir`, with separators flattened) instead
    of an opaque hash. This keeps prebuild output human-readable and grouped
    under `base_dir` while staying unique per run (sweep log_dirs are unique by
    INV) and not colliding with the real run output. `base_dir` is the run's
    log_dir for a single run and the sweep's `_experiment_root` for a sweep."""
    log_dir = Path(str(params.get("log_dir", "logs")))
    try:
        rel = log_dir.resolve().relative_to(base_dir.resolve())
        label = "_".join(rel.parts)
    except ValueError:
        label = ""
    label = label or log_dir.name or "run"
    return base_dir / ".cache_build" / label


async def prebuild_caches(
    param_sets: list[dict],
    schema: Schema,
    build_type: str = "debug",
    base_dir: Path | None = None,
) -> bool:
    """Run one `build-cache-only` per unique cache key, sequentially. Returns
    True iff every prebuild succeeded. The key fields come from the Rust schema
    (`affects_cache`), so two runs differing only in non-kernel params (rate,
    server count, log_dir, ...) share a single prebuild.

    `base_dir` is the experiment's base logging directory under which prebuild
    output subdirs are placed; the caller passes the single run's log_dir or the
    sweep's `_experiment_root`. If omitted it falls back to the parent of the
    first param set's log_dir."""
    if base_dir is None:
        first = Path(str(param_sets[0].get("log_dir", "logs"))) if param_sets else Path("logs")
        base_dir = first.parent
    binary = binary_path(build_type)
    env = _build_subprocess_env()

    for params in _unique_by_cache_key(param_sets, schema):
        argv = build_cli_command(params, binary, subcommand="build-cache-only")
        runner = SimulationRunner(
            argv=argv, log_dir=_prebuild_log_dir(base_dir, params), env=env
        )
        if not await runner.run():
            return False
    return True
