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

from collections.abc import Iterable
from pathlib import Path

from .exec import SimulationRunner, _build_subprocess_env, binary_path
from .schema import build_cli_command
from .schema.loader import Schema


def cache_key(params: dict, cache_fields: Iterable[str]) -> tuple:
    """The kernel-determining subset of `params`, hashable for de-duplication.
    `cache_fields` comes from `DeploymentSchema.cache_key_fields` (Rust-tagged)."""
    return tuple(params.get(field) for field in cache_fields)


def _prebuild_log_dir(params: dict, key: tuple) -> Path:
    """Where a prebuild's stdout lands — a sibling of the run's log_dir so it
    does not collide with the real run output."""
    base = Path(str(params.get("log_dir", "logs")))
    return base.parent / f".cache_build_{abs(hash(key))}"


async def prebuild_caches(
    param_sets: list[dict], schema: Schema, build_type: str = "debug"
) -> bool:
    """Run one `build-cache-only` per unique cache key, sequentially. Returns
    True iff every prebuild succeeded. The key fields come from the Rust schema
    (`affects_cache`), so two runs differing only in non-kernel params (rate,
    server count, log_dir, ...) share a single prebuild."""
    binary = binary_path(build_type)
    env = _build_subprocess_env()

    seen_keys: set[tuple] = set()
    representatives: list[tuple[dict, tuple]] = []
    for params in param_sets:
        cache_fields = schema.deployment_schemas[params["deployment"]].cache_key_fields
        key = cache_key(params, cache_fields)
        if key not in seen_keys:
            seen_keys.add(key)
            representatives.append((params, key))

    for params, key in representatives:
        argv = build_cli_command(params, binary, subcommand="build-cache-only")
        runner = SimulationRunner(
            argv=argv, log_dir=_prebuild_log_dir(params, key), env=env
        )
        if not await runner.run():
            return False
    return True
