"""Four explicit alignment phases under ``python -m launcher alignment``.

Each command accepts exactly its own YAML/JSON contract. Cross-stage commands
consume completed artifact directories, never earlier config files. Launcher
code owns parsing and orchestration; alignment modules own measured profiling,
NSYS normalization, and timing-predict input conversion.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from alignment.runner import run_profile
from alignment.timing_predict_input import BuildRequest, build_inputs
from alignment.timing_predict_input.builder import INPUT_MANIFEST_NAME

from .alignment_config import (
    AnalyzePhaseConfig,
    TimingPredictPhaseConfig,
    load_analyze_config,
    load_labeled_kernel_sequences,
    load_profile_config,
    load_timing_predict_config,
)
from .schema.loader import PresetError, _load_preset

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SUFFIXES = frozenset({".json", ".yaml", ".yml"})
ALIGNMENT_MANIFEST_NAME = "alignment_manifest.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m launcher alignment",
        description="Run one explicit stage of the VibeSim↔vLLM alignment workflow.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    sim = commands.add_parser("sim", help="Run the ordinary VibeSim simulation.")
    sim.add_argument("config", type=Path, help="Simulation preset YAML/JSON")
    sim.add_argument("--dry-run", action="store_true", help="Validate and expand only.")
    sim.add_argument("--build-type", default="release", help="Cargo profile (default: release).")
    sim.add_argument("--refresh", action="store_true", help="Ignore completed-run markers.")
    sim.add_argument(
        "--no-analyze",
        action="store_true",
        help="Skip the simulation's ordinary post-run analyzer.",
    )

    profile = commands.add_parser(
        "profile", help="Run vLLM under NSYS and write normalized measured artifacts."
    )
    profile.add_argument("config", type=Path, help="Profile phase YAML/JSON")
    profile.add_argument("--dry-run", action="store_true", help="Validate only.")

    predict = commands.add_parser(
        "timing-predict",
        help="Build measured iteration cases and run offline timing prediction.",
    )
    predict.add_argument("config", type=Path, help="Timing-predict phase YAML/JSON")
    predict.add_argument("--build-type", default="release", help="Cargo profile.")

    analyze = commands.add_parser(
        "analyze", help="Compare completed profile, prediction, and simulation artifacts."
    )
    analyze.add_argument("config", type=Path, help="Analyze phase YAML/JSON")
    analyze.add_argument("--build-type", default="release", help="Cargo profile.")
    return parser


def _load_simulation_preset(path: Path) -> dict[str, Any]:
    """Validate one ordinary simulation preset without introducing a wrapper."""
    if path.suffix.lower() not in CONFIG_SUFFIXES:
        raise ValueError(f"simulation config must be YAML or JSON: {path}")
    try:
        return _load_preset(path)
    except PresetError as exc:
        raise ValueError(str(exc)) from exc


def _load_simulation_params(log_dir: Path) -> dict[str, Any]:
    params_path = log_dir / "raw" / "params.json"
    if not params_path.is_file():
        raise ValueError(
            f"completed simulation params not found: {params_path}; run alignment sim first"
        )
    params = json.loads(params_path.read_text())
    if not isinstance(params, dict):
        raise ValueError(f"simulation params must be a JSON object: {params_path}")
    return params


def _simulation_target(params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Resolve the one direct unified target supported by the v1 input builder."""
    if params.get("deployment") != "unified":
        raise ValueError("alignment v1 timing input supports deployment: unified only")
    pools = params.get("pools")
    main = pools.get("main") if isinstance(pools, dict) else None
    groups = main.get("groups") if isinstance(main, dict) else None
    if not isinstance(groups, list) or len(groups) != 1:
        raise ValueError("alignment v1 requires exactly one pools.main.groups entry")
    group = groups[0]
    if not isinstance(group, dict) or group.get("replicas", 1) != 1:
        raise ValueError("alignment v1 requires one main-pool replica")
    gpu = group.get("gpu")
    arch = group.get("arch")
    if not isinstance(gpu, str) or not isinstance(arch, dict):
        raise ValueError("simulation main group must define gpu and arch")
    return gpu, arch


def _simulation_backends(params: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    """Return the normalized run's backend policy for predictor parity.

    A completed launcher run always records a validated mapping in params.json.
    Keep the check here because alignment consumes an artifact boundary rather
    than trusting that an arbitrary directory was produced by this launcher.
    """
    backends = params.get("backends", {})
    if not isinstance(backends, dict):
        raise ValueError("completed simulation backends must be a mapping")
    for pool, roles in backends.items():
        if not isinstance(pool, str) or not isinstance(roles, dict):
            raise ValueError("completed simulation backends must map pool names to role maps")
        for role, candidates in roles.items():
            if (
                not isinstance(role, str)
                or not isinstance(candidates, list)
                or not candidates
                or not all(isinstance(candidate, str) for candidate in candidates)
            ):
                raise ValueError(
                    "completed simulation backend roles must map to non-empty string lists"
                )
    return backends


def _load_profile_result(profile_log_dir: Path) -> dict[str, Any]:
    result_path = profile_log_dir / "profile_result.json"
    if not result_path.is_file():
        raise ValueError(f"profile result not found: {result_path}; run alignment profile first")
    result = json.loads(result_path.read_text())
    if not isinstance(result, dict):
        raise ValueError(f"profile result must be a JSON object: {result_path}")
    recorded = result.get("log_dir")
    if not isinstance(recorded, str) or Path(recorded).resolve() != profile_log_dir.resolve():
        raise ValueError(f"profile result does not identify configured directory: {result_path}")
    return result


def _profile_artifact(result: dict[str, Any], key: str) -> Path:
    value = result.get(key)
    if not isinstance(value, str):
        raise ValueError(f"profile_result.json has no {key!r}")
    path = Path(value).resolve()
    if not path.is_file():
        raise ValueError(f"profile artifact not found: {path}")
    return path


def _optional_profile_artifact(result: dict[str, Any], key: str) -> Path | None:
    """Resolve a versioned profile artifact that older captures may not own."""
    if result.get(key) is None:
        return None
    return _profile_artifact(result, key)


def _warn_if_trace_mismatch(params: dict[str, Any], profile_result: dict[str, Any]) -> None:
    workload = params.get("workload")
    traces = workload.get("trace_files") if isinstance(workload, dict) else None
    drive_summary = profile_result.get("drive_summary")
    profile_trace = drive_summary.get("source_trace") if isinstance(drive_summary, dict) else None
    resolved_simulation = []
    if isinstance(traces, list) and all(isinstance(item, str) for item in traces):
        resolved_simulation = [
            (Path(item) if Path(item).is_absolute() else REPO_ROOT / item).resolve()
            for item in traces
        ]
    aligned = (
        len(resolved_simulation) == 1
        and isinstance(profile_trace, str)
        and resolved_simulation[0] == Path(profile_trace).resolve()
    )
    if not aligned:
        print(
            "[warn] alignment trace mismatch: "
            f"simulation={list(map(str, resolved_simulation))!r}, profile={profile_trace!r}; "
            "timing-predict will continue, but request populations may not align",
            file=sys.stderr,
        )


def _read_input_manifest(timing_predict_log_dir: Path) -> dict[str, Any]:
    path = timing_predict_log_dir / INPUT_MANIFEST_NAME
    if not path.is_file():
        raise ValueError(f"timing-predict input manifest not found: {path}")
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError(f"unsupported timing-predict input manifest: {path}")
    return manifest


def _manifest_path(manifest: dict[str, Any], key: str) -> Path:
    value = manifest.get(key)
    if not isinstance(value, str):
        raise ValueError(f"timing-predict input manifest has no {key!r}")
    return Path(value).resolve()


def _require_manifest_dir(manifest: dict[str, Any], key: str, expected: Path) -> None:
    recorded = _manifest_path(manifest, key)
    if recorded != expected.resolve():
        raise ValueError(f"timing-predict provenance mismatch for {key}: {recorded} != {expected}")


def _snapshot_config(source: Path, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{name}{source.suffix.lower()}"
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


def _write_analysis_manifest(config: AnalyzePhaseConfig) -> Path:
    """Assemble the analyzer envelope only after all prior phases completed."""
    simulation_params = _load_simulation_params(config.simulation_log_dir)
    _simulation_target(simulation_params)
    profile_result = _load_profile_result(config.profile_log_dir)
    request_timings_result = _optional_profile_artifact(
        profile_result, "request_timings_jsonl"
    )
    input_manifest = _read_input_manifest(config.timing_predict_log_dir)
    _require_manifest_dir(input_manifest, "simulation_log_dir", config.simulation_log_dir)
    _require_manifest_dir(input_manifest, "profile_log_dir", config.profile_log_dir)
    _require_manifest_dir(input_manifest, "predict_log_dir", config.timing_predict_log_dir)

    cost_manifest = config.timing_predict_log_dir / "raw" / "cost_manifest"
    cost_log = config.timing_predict_log_dir / "raw" / "cost_log"
    if not cost_manifest.is_dir() or not cost_log.is_dir():
        raise ValueError(
            f"timing-predict output is incomplete under {config.timing_predict_log_dir}"
        )

    labeled_sequences = None
    if config.iteration.enabled:
        assert config.iteration.labeled_kernel_sequences_file is not None
        sequences = load_labeled_kernel_sequences(config.iteration.labeled_kernel_sequences_file)
        labeled_sequences = config.log_dir / "kernel_sequences_labeled.json"
        config.log_dir.mkdir(parents=True, exist_ok=True)
        labeled_sequences.write_text(json.dumps(sequences, indent=2))

    config.log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.log_dir / ALIGNMENT_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "profile_log_dir": str(config.profile_log_dir),
                "simulation_log_dir": str(config.simulation_log_dir),
                "analysis_log_dir": str(config.log_dir),
                "parsed_nsys": str(_manifest_path(input_manifest, "parsed_nsys")),
                "replay_result": str(_profile_artifact(profile_result, "replay_result")),
                "request_timings_result": (
                    str(request_timings_result) if request_timings_result else None
                ),
                "predict_log_dir": str(config.timing_predict_log_dir),
                "timing_predict_case_map": str(
                    _manifest_path(input_manifest, "timing_predict_case_map")
                ),
                "labeled_kernel_sequences": (str(labeled_sequences) if labeled_sequences else None),
                "iteration": {
                    "enabled": config.iteration.enabled,
                },
                "workload": {
                    "enabled": config.workload.enabled,
                },
                "e2e": {
                    "enabled": config.e2e.enabled,
                    "throughput_bins": config.e2e.throughput_bins,
                },
            },
            indent=2,
        )
    )
    return manifest_path


def _launch_simulation(
    path: Path,
    *,
    dry_run: bool,
    build_type: str,
    refresh: bool,
    no_analyze: bool,
) -> int:
    from .__main__ import main as launcher_main

    argv = [str(path), "--build-type", build_type]
    if dry_run:
        argv.append("--dry-run")
    if refresh:
        argv.append("--refresh")
    if no_analyze:
        argv.append("--no-analyze")
    try:
        return launcher_main(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2


def _launch_timing_predict(config_path: Path, *, build_type: str) -> int:
    from .timing_predict import main as timing_predict_main

    try:
        return timing_predict_main([str(config_path), "--build-type", build_type, "--no-analyze"])
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2


def _launch_alignment_analysis(log_dir: Path, *, build_type: str, subjects: list[str]) -> bool:
    from .exec import analyzer_binary_path, cargo_build, run_alignment_analysis

    if not cargo_build(build_type, build_analyzer=True):
        return False
    if not analyzer_binary_path(build_type).is_file():
        print("[alignment] analyzer build failed", file=sys.stderr)
        return False
    asyncio.run(run_alignment_analysis(log_dir, build_type, subjects))
    return True


def _run_sim(args: argparse.Namespace) -> int:
    _load_simulation_preset(args.config)
    print(f"[alignment] simulation: {args.config}")
    return _launch_simulation(
        args.config,
        dry_run=args.dry_run,
        build_type=args.build_type,
        refresh=args.refresh,
        no_analyze=args.no_analyze,
    )


def _run_profile(args: argparse.Namespace) -> int:
    config = load_profile_config(args.config)
    if args.dry_run:
        print(f"[alignment] profile validated: {args.config} (log_dir={config.log_dir})")
        return 0
    _snapshot_config(args.config, Path(config.log_dir), "profile")
    print(f"[alignment] profiling: {args.config}")
    result = run_profile(config)
    print(
        f"[alignment] profiling complete: {result['parsed_nsys']}\n"
        "[alignment] inspect parsed.json, then create timing_predict.yaml"
    )
    return 0


def _run_timing_predict(args: argparse.Namespace) -> int:
    config: TimingPredictPhaseConfig = load_timing_predict_config(args.config)
    params = _load_simulation_params(config.simulation_log_dir)
    gpu, arch = _simulation_target(params)
    profile_result = _load_profile_result(config.profile_log_dir)
    if profile_result.get("server_tp_size") != 1:
        raise ValueError("alignment v1 requires profile server_tp_size: 1")
    _warn_if_trace_mismatch(params, profile_result)
    build_result = build_inputs(
        BuildRequest(
            simulation_log_dir=config.simulation_log_dir,
            profile_log_dir=config.profile_log_dir,
            parsed_nsys=_profile_artifact(profile_result, "parsed_nsys"),
            output_dir=config.log_dir,
            gpu=gpu,
            arch=arch,
            backends=_simulation_backends(params),
            input_spec=config.input_builder,
        )
    )
    _snapshot_config(args.config, config.log_dir, "timing_predict")
    print(f"[alignment] timing-predict: {build_result.predict_config}")
    return _launch_timing_predict(build_result.predict_config, build_type=args.build_type)


def _run_analyze(args: argparse.Namespace) -> int:
    config = load_analyze_config(args.config)
    manifest = _write_analysis_manifest(config)
    _snapshot_config(args.config, config.log_dir, "analyze")
    print(f"[alignment] analyzer manifest: {manifest}")
    return (
        0
        if _launch_alignment_analysis(
            config.log_dir, build_type=args.build_type, subjects=config.subjects
        )
        else 1
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "sim":
            return _run_sim(args)
        if args.command == "profile":
            return _run_profile(args)
        if args.command == "timing-predict":
            return _run_timing_predict(args)
        if args.command == "analyze":
            return _run_analyze(args)
        raise AssertionError(f"unhandled alignment command {args.command!r}")
    except (KeyError, OSError, ValueError) as exc:
        print(f"[invalid] alignment {args.command}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
