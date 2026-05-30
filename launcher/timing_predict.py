"""`python -m launcher iter-timing-predict <config> [...]` — offline whole-iteration
timing prediction.

A predict config is NOT a deployment `RunConfig` (it has no workload / pools /
sweep — just one iter-wise arch + gpu + a cases file), so this entry bypasses the
schema expansion the normal run flow does. It performs the shared `cargo_build`,
then for each config runs the `iter-timing-predict` subcommand (which writes the
standard `raw/cost_log` + `cost_manifest` artifacts, one row per case) followed by
the post-run analyzer — `run_analysis` emits the Perfetto `iter → layers →
kernels` trace plus the cost reports from exactly those artifacts.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from .exec import (
    REPO_ROOT,
    SimulationRunner,
    _build_subprocess_env,
    binary_path,
    cargo_build,
    run_analysis,
)


def _load_config(path: Path) -> dict:
    """Parse the minimal predict config (JSON or YAML) — only `log_dir` is needed
    launcher-side; the binary re-parses the whole thing."""
    text = path.read_text()
    if path.suffix == ".json":
        return json.loads(text)
    import yaml

    return yaml.safe_load(text)


async def _run_one(config_path: Path, build_type: str, analyze: bool) -> bool:
    cfg = _load_config(config_path)
    log_dir = Path(cfg["log_dir"])
    if not log_dir.is_absolute():
        # The binary writes `raw/` relative to its cwd (REPO_ROOT); anchor the
        # launcher-side log_dir to the same root so stdout.log + analysis line up.
        log_dir = REPO_ROOT / log_dir

    binary = binary_path(build_type)
    argv = [str(binary), "iter-timing-predict", str(config_path)]
    runner = SimulationRunner(argv=argv, log_dir=log_dir, env=_build_subprocess_env())
    ok = await runner.run()
    if ok and analyze:
        # Best-effort: `analyze trace` (the Perfetto tree) + `analyze run`. Cost
        # subjects apply; request/throughput subjects self-skip on a predict dir.
        await run_analysis(log_dir, build_type)
    return ok


def main(argv: list[str]) -> int:
    build_type = "release"
    analyze = True
    configs: list[str] = []
    it = iter(argv)
    for tok in it:
        if tok == "--build-type":
            build_type = next(it, "")
            if not build_type:
                sys.exit("iter-timing-predict: --build-type needs a value")
        elif tok == "--no-analyze":
            analyze = False
        elif tok.startswith("-"):
            sys.exit(f"iter-timing-predict: unknown flag {tok}")
        else:
            configs.append(tok)

    if not configs:
        sys.exit(
            "iter-timing-predict: no config given; usage: "
            "python -m launcher iter-timing-predict <config.json> [...]"
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
        if not asyncio.run(_run_one(config_path, build_type, analyze)):
            rc = rc or 1
    return rc
