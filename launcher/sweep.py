"""Sweep orchestration — the top-level run/sweep flow (design §1.2.5).

`run_single` is the plan-then-build flow for one param set: prebuild its cache,
write metadata BEFORE spawn (INV-2), then run. `run_sweep` expands that across
many param sets: reject log_dir collisions, prebuild all unique cache keys
sequentially (INV-3), launch the runs in parallel under a semaphore, persist
the shared preset at the experiment root, and best-effort aggregate.

File layout: shared helpers, then the single-run flow, the sweep flow, and the
aggregator contract — each public entry point (`run_single` / `run_sweep`) sits
at the top of its section, with its private async machinery below it.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from . import metadata
from .cache_build import prebuild_caches
from .exec import (
    DEFAULT_PARALLELISM,
    SimulationRunner,
    _build_subprocess_env,
    _profile_env,
    binary_path,
    run_analysis,
    wrap_with_perf,
)
from .schema import build_cli_command, log_dir_of, validate_unique_log_dirs
from .schema.loader import Registry

# Resume marker (INV-5): the launcher writes this into a run's log_dir only
# after a zero-exit run. On the default resume path a run whose log_dir already
# carries it is skipped; `--refresh` ignores it and re-runs.
COMPLETE_MARKER = ".complete"


# ── shared helpers (resume markers) ─────────────────────────────────────────


def _is_complete(log_dir: Path) -> bool:
    return (log_dir / COMPLETE_MARKER).is_file()


def _mark_complete(log_dir: Path) -> None:
    (log_dir / COMPLETE_MARKER).write_text(datetime.now(UTC).isoformat() + "\n")


async def _launch_one(
    params: dict,
    build_type: str,
    refresh: bool = False,
    profile: bool = False,
    profile_freq: int = 499,
    analyze: bool = True,
    analyze_subjects: list[str] | None = None,
) -> bool:
    """Metadata-then-spawn for a single (already normalized) param set. On resume
    (the default) a run whose log_dir already has a `.complete` marker is skipped;
    `refresh=True` forces a re-run. The marker is written only on a zero exit.
    Shared by both the single-run and sweep flows.

    `profile=True` wraps the run argv with `perf record` (output `<log_dir>/
    perf.data`) under a single-threaded BLAS/OMP env — the launcher's wallclock
    profiling mode (skill `profile-sim-speed`).

    `analyze=True` runs the post-run analyzer (Rust compute → Python plots) after
    a successful run; best-effort, so analysis failures never fail the run.
    `analyze_subjects` narrows which subjects run (None/empty = all applicable)."""
    log_dir = Path(log_dir_of(params))

    if not refresh and _is_complete(log_dir):
        print(f"[skip] {log_dir} already complete (use --refresh to re-run)")
        return True

    binary = binary_path(build_type)
    # The concrete config the binary reads lives alongside the run's metadata.
    argv = build_cli_command(
        params, binary, metadata.raw_dir(log_dir) / "run_config.yaml", subcommand="run"
    )

    # INV-2: all metadata lands before the subprocess starts (record the bare run
    # argv, before any perf wrapping, so metadata reflects the simulated run).
    metadata.write_run_metadata(log_dir, params, argv)
    # Clear any stale marker so a crash mid-run never leaves a false 'complete'.
    (log_dir / COMPLETE_MARKER).unlink(missing_ok=True)

    env = _build_subprocess_env()
    if profile:
        # perf runs with cwd=REPO_ROOT, so write to an absolute path.
        perf_data = log_dir.resolve() / "perf.data"
        argv = wrap_with_perf(argv, perf_data, profile_freq)
        env = _profile_env()

    runner = SimulationRunner(argv=argv, log_dir=log_dir, env=env)
    ok = await runner.run()
    if ok:
        _mark_complete(log_dir)
        if profile:
            print(
                f"[profile] wrote {log_dir.resolve() / 'perf.data'} — read with "
                f"`perf report -i {log_dir.resolve() / 'perf.data'} --stdio` (not cat)"
            )
        if analyze:
            await run_analysis(log_dir, build_type, analyze_subjects)
    return ok


# ── single-run flow ──────────────────────────────────────────────────────────


def run_single(
    params: dict,
    preset: dict | None,
    schema: Registry,
    build_type: str = "debug",
    refresh: bool = False,
    profile: bool = False,
    profile_freq: int = 499,
    analyze: bool = True,
    analyze_subjects: list[str] | None = None,
) -> bool:
    """Prebuild → metadata → run, for one param set. Synchronous entry. The
    caller must pass the already-loaded Rust schema from `load_schema()`; the
    sweep layer never loads a schema implicitly. Resumes by default (skips a run
    already marked `.complete`); `refresh=True` re-runs. `profile=True` wraps the
    run with `perf record` (skill `profile-sim-speed`). `analyze=True` runs the
    post-run analyzer (best-effort); `analyze_subjects` narrows which subjects."""
    if schema is None:
        raise TypeError("run_single requires a loaded Schema; call load_schema() first")
    return asyncio.run(
        _run_single_async(
            params, preset, schema, build_type, refresh, profile, profile_freq,
            analyze, analyze_subjects,
        )
    )


async def _run_single_async(
    params: dict,
    preset: dict | None,
    schema: Registry,
    build_type: str,
    refresh: bool = False,
    profile: bool = False,
    profile_freq: int = 499,
    analyze: bool = True,
    analyze_subjects: list[str] | None = None,
) -> bool:
    log_dir = Path(log_dir_of(params))
    if not refresh and _is_complete(log_dir):
        print(f"[skip] {log_dir} already complete (use --refresh to re-run)")
        return True
    metadata.write_shared_metadata(log_dir, preset or params)
    if not await prebuild_caches([params], schema, build_type, base_dir=log_dir):
        return False
    return await _launch_one(
        params, build_type, refresh, profile, profile_freq, analyze, analyze_subjects
    )


# ── sweep flow ─────────────────────────────────────────────────────────────


def run_sweep(
    param_sets: list[dict],
    original_preset: dict,
    schema: Registry,
    build_type: str = "debug",
    parallelism: int = DEFAULT_PARALLELISM,
    refresh: bool = False,
    analyze: bool = True,
    analyze_subjects: list[str] | None = None,
) -> int:
    """Expand-then-launch a full sweep. Returns a process exit code. The caller
    must pass the already-loaded Rust schema from `load_schema()`; the sweep
    layer never loads a schema implicitly. Resumes by default (skips runs marked
    `.complete`); `refresh=True` re-runs all. `analyze=True` runs the per-run
    analyzer after each successful run (best-effort); `analyze_subjects` narrows
    which subjects."""
    if schema is None:
        raise TypeError("run_sweep requires a loaded Schema; call load_schema() first")
    return asyncio.run(
        _run_sweep_async(
            param_sets, original_preset, schema, build_type, parallelism, refresh,
            analyze, analyze_subjects,
        )
    )


async def _run_sweep_async(
    param_sets: list[dict],
    original_preset: dict,
    schema: Registry,
    build_type: str,
    parallelism: int,
    refresh: bool = False,
    analyze: bool = True,
    analyze_subjects: list[str] | None = None,
) -> int:
    if not validate_unique_log_dirs(param_sets):
        return 2

    base_dir = _experiment_root(param_sets)
    base_dir.mkdir(parents=True, exist_ok=True)
    metadata.write_shared_metadata(base_dir, original_preset)

    # Resume (default): only prebuild + launch runs that aren't already complete.
    # `--refresh` re-runs everything. Aggregation still spans all runs so the
    # summary covers previously-completed ones too.
    if refresh:
        pending = param_sets
    else:
        pending = [
            p for p in param_sets if not _is_complete(Path(log_dir_of(p)))
        ]
    skipped = len(param_sets) - len(pending)
    if skipped:
        print(
            f"[resume] {skipped} run(s) already complete; {len(pending)} to run "
            "(use --refresh to force all)"
        )

    if pending:
        if not await prebuild_caches(pending, schema, build_type, base_dir=base_dir):
            print("[error] cache prebuild failed; aborting sweep")
            return 1

        sem = asyncio.Semaphore(parallelism)

        async def _bounded(params: dict) -> bool:
            async with sem:
                return await _launch_one(
                    params, build_type, refresh,
                    analyze=analyze, analyze_subjects=analyze_subjects,
                )

        results = await asyncio.gather(*(_bounded(p) for p in pending))
    else:
        results = []

    _aggregate(param_sets, base_dir, original_preset)

    return 0 if all(results) else 1


def _experiment_root(param_sets: list[dict]) -> Path:
    """Common parent of all run log_dirs — the sweep base_dir."""
    dirs = [Path(log_dir_of(p)).resolve() for p in param_sets]
    if not dirs:
        return Path("logs").resolve()
    common = dirs[0].parent
    for d in dirs[1:]:
        # Walk up until `common` is a prefix of d's parent.
        while common not in d.parents and common != d.parent:
            common = common.parent
    return common


# ── aggregator contract (sweep axes) ────────────────────────────────────────


def _axis_value_key(value):
    """Hashable representation for distinct-value detection (lists → tuples)."""
    return tuple(value) if isinstance(value, list) else value


def _sweep_axes(param_sets: list[dict]) -> list[str]:
    """Sweep/derived/compound-group names that take more than one distinct value
    across the runs — the sweep axes. Read from each run's resolved `_env` (the
    bindings stashed by expansion), so list / dict sweeps, varying `derived`, and
    `compound` groups all surface uniformly. Constant bindings are excluded.

    `compound` *members* are folded out: a group's members co-vary, so the group
    name (also in `_env`, valued by the row label) is the single axis — listing the
    members as independent axes would imply a cross-product that does not exist."""
    members: set[str] = set()
    for p in param_sets:
        members |= set(p.get("_compound_members", ()))
    keys: set[str] = set()
    for p in param_sets:
        keys |= set(p.get("_env", {}))
    axes = []
    for k in sorted(keys - members):
        if len({_axis_value_key(p.get("_env", {}).get(k)) for p in param_sets}) > 1:
            axes.append(k)
    return axes


def _aggregate(param_sets: list[dict], base_dir: Path, original_preset: dict) -> None:
    """Best-effort sweep aggregation. The aggregator is analyzer-owned and may
    not be present yet; skip silently if unavailable (design §1.2.5).

    Example payload after expansion (sweep coords come from each run's `_env`):
        run_infos = [
            {"log_dir": "/abs/logs/tp2", "sweep": {"ptp": 2}, "labels": {}},
            {"log_dir": "/abs/logs/tp4", "sweep": {"ptp": 4}, "labels": {}},
        ]
    Independent `sweep` dims form a cartesian grid; correlated columns come from a
    `compound` group (one named axis whose members co-vary) or a `derived` name
    moving with its inputs — both surface as ordinary `_env` axes, so no separate
    group-zip hint is needed in the structured interface.
    """
    try:
        from analyze_aggregator.aggregator import aggregate_sweep
    except Exception:
        print("[aggregate] aggregator unavailable; skipping sweep summary")
        return

    # Each run_info is one row of the axis-coordinate → output-folder map. Use
    # ABSOLUTE log_dir so it is unambiguous against the resolved base_dir.
    axes = _sweep_axes(param_sets)
    run_infos = [
        {
            "log_dir": str(Path(log_dir_of(p)).resolve()),
            "sweep": {k: p.get("_env", {}).get(k) for k in axes},
            "labels": p.get("_sweep_labels", {}),
        }
        for p in param_sets
    ]
    try:
        aggregate_sweep(run_infos, base_dir, groups={})
    except Exception as exc:  # aggregation failure must not fail the sweep
        print(f"[aggregate] skipped: {exc}")
