"""`python -m launcher timing-predict CONFIG.{yaml,yml,json} [...]` — offline
per-building-block timing prediction.

A predict config is NOT a deployment `RunConfig` (it has no workload / pools /
sweep — just one arch selector `{iter|attn|ffn}` + gpu + a cases file), so this
entry bypasses the schema expansion the normal run flow does. It performs the
shared `cargo_build`, then for each config runs the predictor subcommand (which
writes the standard `raw/cost_log` + `cost_manifest` artifacts, one row per
case/section) followed by the post-run analyzer — `run_analysis` emits the
Perfetto trace plus the cost reports from exactly those artifacts.

The `arch` selector picks one of `{iter|attn|ffn}` — an iter-wise whole-iteration
arch, or one half of an AFD split (attn / ffn), each costed independently.

Unlike a real `run` (whose launcher snapshots the expanded config into `log_dir`),
the predict path previously left no provenance behind; we now copy the config + its
cases file into `log_dir` so a predict result is self-describing.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

from .exec import (
    REPO_ROOT,
    SimulationRunner,
    _build_subprocess_env,
    binary_path,
    cargo_build,
    run_analysis,
    run_iter_breakdown,
)
from .managed_job import prepare_managed_job


def _load_config(path: Path) -> dict:
    """Parse the minimal predict config (JSON or YAML) — only `log_dir` /
    `cases_file` are needed launcher-side; the binary re-parses the whole thing."""
    text = path.read_text()
    if path.suffix == ".json":
        return json.loads(text)
    import yaml

    return yaml.safe_load(text)


def _snapshot_inputs(config_path: Path, cfg: dict, log_dir: Path) -> None:
    """Copy the predict config + its cases file into `log_dir` for provenance, so a
    predict result is self-describing (mirrors a real run's config snapshot). The
    cases file is resolved relative to the config's directory, the same way the
    binary resolves it. Best-effort: a copy failure warns but does not fail the run.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Alignment generates its config directly in the predict root. Treat
        # that as an already-complete snapshot instead of asking shutil to copy
        # a file onto itself and obscuring the rest of the provenance step.
        config_snapshot = log_dir / config_path.name
        if config_path.resolve() != config_snapshot.resolve():
            shutil.copy2(config_path, config_snapshot)
        cases_file = cfg.get("cases_file")
        if cases_file:
            cases_path = Path(cases_file)
            if not cases_path.is_absolute():
                cases_path = config_path.parent / cases_path
            if cases_path.is_file():
                cases_snapshot = log_dir / cases_path.name
                if cases_path.resolve() != cases_snapshot.resolve():
                    shutil.copy2(cases_path, cases_snapshot)
            else:
                print(
                    f"[warn] cases_file {cases_path} not found; not snapshotted",
                    file=sys.stderr,
                )
    except OSError as e:
        print(f"[warn] failed to snapshot predict inputs into {log_dir}: {e}", file=sys.stderr)


def _predict_descriptor(config_path: Path, cfg: dict) -> dict:
    """Build the small catalog descriptor without interpreting case semantics."""
    arch = cfg.get("arch")
    selector = next(iter(arch)) if isinstance(arch, dict) and len(arch) == 1 else None
    descriptor = {
        "selector": selector,
        "configName": config_path.name,
        "caseCount": _predict_case_count(config_path, cfg),
    }
    return {key: value for key, value in descriptor.items() if value is not None}


def _predict_case_count(config_path: Path, cfg: dict) -> int | None:
    cases_file = cfg.get("cases_file")
    if not isinstance(cases_file, str) or not cases_file:
        return None
    cases_path = Path(cases_file)
    if not cases_path.is_absolute():
        cases_path = config_path.parent / cases_path
    try:
        payload = _load_config(cases_path)
    except (OSError, json.JSONDecodeError):
        return None
    return len(payload) if isinstance(payload, list) else None


async def run_one(config_path: Path, build_type: str, analyze: bool) -> bool:
    """Run one already-built predictor config.

    Public so the alignment timing-predict stage can reuse the exact predictor
    execution/snapshot path.
    """
    cfg = _load_config(config_path)
    log_dir = Path(cfg["log_dir"])
    if not log_dir.is_absolute():
        # The binary writes `raw/` relative to its cwd (REPO_ROOT); anchor the
        # launcher-side log_dir to the same root so stdout.log + analysis line up.
        log_dir = REPO_ROOT / log_dir

    descriptor = _predict_descriptor(config_path, cfg)
    managed_job = prepare_managed_job(
        "timing_predict",
        log_dir,
        descriptor=descriptor,
    )

    try:
        if managed_job is not None:
            managed_job.report("running")
        _snapshot_inputs(config_path, cfg, log_dir)

        binary = binary_path(build_type)
        argv = [str(binary), "timing-predict", str(config_path)]
        runner = SimulationRunner(argv=argv, log_dir=log_dir, env=_build_subprocess_env())
        ok = await runner.run()
        if not ok:
            if managed_job is not None:
                managed_job.report("failed")
            return False
        if analyze:
            if managed_job is not None:
                managed_job.report("analysis_running")
            # Best-effort: `analyze trace` (the Perfetto tree) + `analyze run`. Cost
            # subjects apply; request/throughput subjects self-skip on a predict dir.
            await run_analysis(log_dir, build_type)
            # Predict-only: the human-readable cost tree (reports/iter_breakdown.ans).
            # Not in the shared run_analysis — a real run's thousands of iters would
            # make that file enormous; a predict dir has only a few cases.
            await run_iter_breakdown(log_dir, build_type)
        if managed_job is not None:
            managed_job.report("ready", summary=descriptor)
        return True
    except (asyncio.CancelledError, KeyboardInterrupt):
        if managed_job is not None:
            managed_job.report("interrupted")
        raise
    except Exception:
        if managed_job is not None:
            managed_job.report("failed")
        raise


def main(argv: list[str]) -> int:
    build_type = "release"
    analyze = True
    configs: list[str] = []
    it = iter(argv)
    for tok in it:
        if tok == "--build-type":
            build_type = next(it, "")
            if not build_type:
                sys.exit("timing-predict: --build-type needs a value")
        elif tok == "--no-analyze":
            analyze = False
        elif tok.startswith("-"):
            sys.exit(f"timing-predict: unknown flag {tok}")
        else:
            configs.append(tok)

    if not configs:
        sys.exit(
            "timing-predict: no config given; usage: "
            "python -m launcher timing-predict CONFIG.yaml|json [...]"
        )

    # INV-8: the single shared build (also produces the analyzer binary).
    if not cargo_build(build_type, build_analyzer=analyze):
        sys.exit("build failed (see errors above)")

    rc = 0
    for c in configs:
        config_path = Path(c).resolve()
        if not config_path.is_file():
            print(f"[invalid] {c}: not a file", file=sys.stderr)
            rc = 2
            continue
        if not asyncio.run(run_one(config_path, build_type, analyze)):
            rc = rc or 1
    return rc
