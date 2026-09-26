"""Four explicit alignment phases plus report comparison under ``launcher alignment``.

Each command accepts exactly its own YAML/JSON contract. Cross-stage commands
consume completed artifact directories, never earlier config files. Launcher
code owns parsing and orchestration; alignment modules own measured profiling,
NSYS normalization, and timing-predict input conversion.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import sys
import tempfile
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
from .artifact_kind import ArtifactKind, write_artifact_kind
from .corpus import CorpusError, resolve_hf_references
from .schema.loader import PresetError, SchemaNotFound, _load_preset, load_schema

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SUFFIXES = frozenset({".json", ".yaml", ".yml"})
ALIGNMENT_MANIFEST_NAME = "alignment_manifest.json"
# Bumped when either typed manifest shape changes; must match the analyzer's
# SCHEMA_VERSION in analyzer/rust/src/alignment_input.rs.
ALIGNMENT_MANIFEST_SCHEMA_VERSION = 9


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m launcher alignment",
        description="Run one explicit stage of the ServingStudioSim↔vLLM alignment workflow.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare-workload", help="Build an explicitly observed-conditioned acceptance trace."
    )
    prepare.add_argument("--source-trace", type=Path, required=True)
    prepare.add_argument("--profile-dir", type=Path, required=True)
    prepare.add_argument("--output-trace", type=Path, required=True)
    prepare.add_argument("--draft-tokens", type=int, required=True)
    prepare.add_argument("--missing-acceptance", choices=["error", "run-aggregate"], required=True)
    prepare.add_argument("--request-id-prefix", default="")

    sim = commands.add_parser("sim", help="Run the ordinary ServingStudioSim simulation.")
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
    profile.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse the existing capture and redo only record extraction, NSYS "
            "normalization, and the result manifest. Use after a post-capture "
            "failure so the GPU capture is never repeated."
        ),
    )

    predict = commands.add_parser(
        "timing-predict",
        help="Build measured iteration cases and run offline timing prediction.",
    )
    predict.add_argument("config", type=Path, help="Timing-predict phase YAML/JSON")
    predict.add_argument("--build-type", default="release", help="Cargo profile.")
    predict.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate the cases in a scratch directory; predict and write nothing.",
    )

    analyze = commands.add_parser(
        "analyze", help="Compare completed profile, prediction, and simulation artifacts."
    )
    analyze.add_argument("config", type=Path, help="Analyze phase YAML/JSON")
    analyze.add_argument("--build-type", default="release", help="Cargo profile.")

    compare = commands.add_parser(
        "compare", help="Compare aggregate errors from two completed kernel-align reports."
    )
    compare.add_argument("baseline", type=Path, help="Baseline analysis dir or report JSON")
    compare.add_argument("candidate", type=Path, help="Candidate analysis dir or report JSON")
    compare.add_argument(
        "--limit", type=int, default=20, help="Maximum operation rows to print (default: 20)."
    )
    compare.add_argument("--json", action="store_true", help="Emit the complete comparison JSON.")
    return parser


def _load_simulation_preset(path: Path) -> dict[str, Any]:
    """Validate one ordinary simulation preset without introducing a wrapper.

    Hub references are resolved here, as the ordinary launcher resolves them,
    because the arch this returns reaches the binary through timing-predict.
    """
    if path.suffix.lower() not in CONFIG_SUFFIXES:
        raise ValueError(f"simulation config must be YAML or JSON: {path}")
    try:
        return resolve_hf_references(_load_preset(path))
    except (PresetError, CorpusError) as exc:
        raise ValueError(str(exc)) from exc


def _normalize_simulation_preset(path: Path, *, build_type: str) -> dict[str, Any]:
    """The sim preset as `launcher sim` hands it to the binary: `backends_file`
    folded into a nested `backends` map, schema defaults filled (an omitted
    `fp8` becomes `false`), and hub references resolved.

    timing-predict forwards this arch and backend policy to the predictor, whose
    config has no defaults of its own, so it must not read the raw preset.
    """
    from .__main__ import _expand_preset
    from .exec import cargo_build

    preset = _load_simulation_preset(path)
    # The schema comes from `list-params`, which the build writes; building
    # first keeps it in step with the binary that will run the prediction.
    if not cargo_build(build_type, build_analyzer=False):
        raise ValueError("simulator build failed (see error above)")
    try:
        schema = load_schema(build_type)
    except SchemaNotFound as exc:
        raise ValueError(str(exc)) from exc
    candidates = _expand_preset(preset, schema, str(path))
    if candidates is None:
        raise ValueError(f"invalid simulation preset: {path} (see errors above)")
    if len(candidates) != 1:
        raise ValueError(
            f"simulation preset {path} expands to {len(candidates)} runs; "
            "alignment timing-predict needs exactly one"
        )
    return candidates[0]


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
    """The normalized preset's ``{pool: {role: [candidates]}}`` backend map, the
    form the timing predictor expects. The backend policy is part of the
    simulated CostTree identity, so this must match what the simulation will use.
    """
    nested = params.get("backends", {})
    if not isinstance(nested, dict):
        raise ValueError("simulation preset backends must be a mapping")
    for pool, roles in nested.items():
        for role, candidates in roles.items():
            if (
                not isinstance(candidates, list)
                or not candidates
                or not all(isinstance(candidate, str) for candidate in candidates)
            ):
                raise ValueError(
                    f"simulation preset backend '{pool}/{role}' must map to a non-empty string list"
                )
    return nested


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


def _same_trace_source(left: Any, right: Any) -> bool:
    """Accept copied trace artifacts only when their bytes are identical."""
    if not isinstance(left, str) or not isinstance(right, str):
        return left == right
    left_path = Path(left).resolve()
    right_path = Path(right).resolve()
    if left_path == right_path:
        return True
    try:
        return (
            left_path.is_file()
            and right_path.is_file()
            and filecmp.cmp(left_path, right_path, shallow=False)
        )
    except OSError:
        return False


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
        and _same_trace_source(str(resolved_simulation[0]), profile_trace)
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
    """Write the typed analyzer manifest for one analysis phase.

    An analyze config is exactly one phase (enforced by load_analyze_config):
    kernel-align (iteration) writes a kernel-only manifest with no simulation;
    e2e-align (workload/e2e) writes a simulation-consuming manifest. The two
    shapes are deliberately disjoint: NSYS belongs only to kernel alignment,
    while request/workload alignment reads only the full workload-metrics run.
    """
    config.log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.log_dir / ALIGNMENT_MANIFEST_NAME
    common = {
        "schema_version": ALIGNMENT_MANIFEST_SCHEMA_VERSION,
        "analysis_log_dir": str(config.log_dir),
    }

    if config.iteration.enabled:
        assert config.profile_log_dir is not None  # config validation
        profile_result = _load_profile_result(config.profile_log_dir)
        assert config.timing_predict_log_dir is not None  # config validation
        input_manifest = _read_input_manifest(config.timing_predict_log_dir)
        _require_manifest_dir(input_manifest, "profile_log_dir", config.profile_log_dir)
        _require_manifest_dir(input_manifest, "predict_log_dir", config.timing_predict_log_dir)
        cost_manifest = config.timing_predict_log_dir / "raw" / "cost_manifest"
        cost_log = config.timing_predict_log_dir / "raw" / "cost_log"
        if not cost_manifest.is_dir() or not cost_log.is_dir():
            raise ValueError(
                f"timing-predict output is incomplete under {config.timing_predict_log_dir}"
            )
        assert config.iteration.labeled_kernel_sequences_file is not None
        sequences = load_labeled_kernel_sequences(config.iteration.labeled_kernel_sequences_file)
        labeled_sequences = config.log_dir / "kernel_sequences_labeled.json"
        labeled_sequences.write_text(json.dumps(sequences, indent=2))
        # Captures taken before the host sidecar existed simply have no host
        # lane; the subject degrades to the device-only view it always drew.
        host_timeline = _optional_profile_artifact(profile_result, "host_timeline")
        # A schema-6 parse is unreadable without its kernel-row sibling; fail here,
        # naming the file, rather than inside the analyzer. Older captures name none.
        _optional_profile_artifact(profile_result, "parsed_kernel_rows")
        manifest = {
            **common,
            "profile_log_dir": str(config.profile_log_dir),
            "parsed_nsys": str(_manifest_path(input_manifest, "parsed_nsys")),
            "predict_log_dir": str(config.timing_predict_log_dir),
            "timing_predict_case_map": str(
                _manifest_path(input_manifest, "timing_predict_case_map")
            ),
            "labeled_kernel_sequences": str(labeled_sequences),
            "host_timeline": str(host_timeline) if host_timeline else None,
        }
    else:
        assert config.simulation_log_dir is not None  # enforced by load_analyze_config
        assert config.workload_profile_log_dir is not None  # config validation
        _simulation_target(_load_simulation_params(config.simulation_log_dir))
        workload_profile_result = _load_profile_result(config.workload_profile_log_dir)
        if workload_profile_result.get("profile_kind") != "workload_metrics":
            raise ValueError(
                "workload_profile_log_dir must contain a workload_metrics profile, "
                f"got {workload_profile_result.get('profile_kind')!r}"
            )
        request_timings_result = _optional_profile_artifact(
            workload_profile_result, "request_timings_jsonl"
        )
        drive_summary = workload_profile_result.get("drive_summary")
        if not isinstance(drive_summary, dict):
            drive_summary = {}
        manifest = {
            **common,
            "workload_profile_log_dir": str(config.workload_profile_log_dir),
            "simulation_log_dir": str(config.simulation_log_dir),
            "metrics_jsonl": str(_profile_artifact(workload_profile_result, "metrics_jsonl")),
            "replay_result": str(_profile_artifact(workload_profile_result, "replay_result")),
            "request_timings_result": (
                str(request_timings_result) if request_timings_result else None
            ),
            "replay_start_monotonic_ns": drive_summary.get("replay_start_monotonic_ns"),
            "replay_end_monotonic_ns": drive_summary.get("replay_end_monotonic_ns"),
            "throughput_bins": config.e2e.throughput_bins,
        }

    manifest_path.write_text(json.dumps(manifest, indent=2))
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


def _launch_timing_predict(config_path: Path, *, build_type: str, dry_run: bool = False) -> int:
    from .timing_predict import main as timing_predict_main

    argv = [str(config_path), "--build-type", build_type]
    if dry_run:
        argv.append("--dry-run")
    try:
        # A prediction is not complete until Analyzer has materialized its cost
        # subjects. In particular, unlocked optimality needs the cached grid-peak
        # sidecar; skipping analysis silently collapses R3 onto R2 in the UI.
        return timing_predict_main(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2


def _launch_alignment_analysis(log_dir: Path, *, build_type: str, subjects: list[str]) -> bool:
    from .exec import analyzer_binary_path, cargo_build_analyzer, run_alignment_analysis

    if not cargo_build_analyzer(build_type):
        return False
    if not analyzer_binary_path(build_type).is_file():
        print("[alignment] analyzer build failed", file=sys.stderr)
        return False
    return run_alignment_analysis(log_dir, build_type, subjects)


def _run_sim(args: argparse.Namespace) -> int:
    preset = _load_simulation_preset(args.config)
    print(f"[alignment] simulation: {args.config}")
    if not args.dry_run:
        io_config = preset.get("io")
        log_dir = io_config.get("log_dir") if isinstance(io_config, dict) else None
        if not isinstance(log_dir, str) or not log_dir:
            raise ValueError("alignment simulation preset requires io.log_dir")
        simulation_log_dir = Path(log_dir)
        if not simulation_log_dir.is_absolute():
            simulation_log_dir = REPO_ROOT / simulation_log_dir
        write_artifact_kind(simulation_log_dir.parent, ArtifactKind.ALIGNMENT_BUNDLE)
    return _launch_simulation(
        args.config,
        dry_run=args.dry_run,
        build_type=args.build_type,
        refresh=args.refresh,
        no_analyze=args.no_analyze,
    )


def _run_profile(args: argparse.Namespace) -> int:
    config = load_profile_config(
        args.config,
        require_python_runtime=not args.resume,
    )
    if args.dry_run:
        print(f"[alignment] profile validated: {args.config} (log_dir={config.log_dir})")
        return 0
    write_artifact_kind(Path(config.log_dir).parent, ArtifactKind.ALIGNMENT_BUNDLE)
    _snapshot_config(args.config, Path(config.log_dir), "profile")
    print(f"[alignment] {'resuming' if args.resume else 'profiling'}: {args.config}")
    result = run_profile(config, resume=args.resume)
    if result.get("profile_kind", "nsys") == "nsys":
        print(
            f"[alignment] profiling complete: {result['parsed_nsys']}\n"
            "[alignment] inspect parsed.json, then create timing_predict.yaml"
        )
    else:
        print(
            f"[alignment] profiling complete: {result['metrics_jsonl']} "
            f"(kind={result['profile_kind']})"
        )
    return 0


def _run_timing_predict(args: argparse.Namespace) -> int:
    config: TimingPredictPhaseConfig = load_timing_predict_config(args.config)
    if args.dry_run:
        # The same inputs, built where nothing reads them and removed afterwards.
        with tempfile.TemporaryDirectory(prefix="alignment-timing-predict-") as scratch:
            predict_config = _build_timing_predict_inputs(config, Path(scratch))
            print(f"[alignment] timing-predict dry run: {args.config}")
            return _launch_timing_predict(predict_config, build_type=args.build_type, dry_run=True)
    write_artifact_kind(config.log_dir.parent, ArtifactKind.ALIGNMENT_BUNDLE)
    predict_config = _build_timing_predict_inputs(config, config.log_dir)
    _snapshot_config(args.config, config.log_dir, "timing_predict")
    print(f"[alignment] timing-predict: {predict_config}")
    return _launch_timing_predict(predict_config, build_type=args.build_type)


def _build_timing_predict_inputs(config: TimingPredictPhaseConfig, output_dir: Path) -> Path:
    """Write the measured cases and predictor config under `output_dir`; return the config."""
    # Read the sim *preset* (not a completed run): timing-predict is kernel-only,
    # so it needs the gpu / arch / backends but never a finished simulation.
    preset = _normalize_simulation_preset(config.simulation_preset, build_type=args.build_type)
    gpu, arch = _simulation_target(preset)
    profile_result = _load_profile_result(config.profile_log_dir)
    # No profile/simulation parallelism pre-check here: every arch spells its
    # sharding differently, so a launcher-side guess is brittle. A genuine
    # mismatch surfaces on its own downstream (e.g. the per-rank device-symmetry
    # fold and the labeled-slot coverage), which is where it is meaningful.
    _warn_if_trace_mismatch(preset, profile_result)
    build_result = build_inputs(
        BuildRequest(
            simulation_preset=config.simulation_preset,
            profile_log_dir=config.profile_log_dir,
            parsed_nsys=_profile_artifact(profile_result, "parsed_nsys"),
            output_dir=output_dir,
            gpu=gpu,
            arch=arch,
            backends=_simulation_backends(preset),
            input_spec=config.input_builder,
        )
    )
    return build_result.predict_config


def _run_analyze(args: argparse.Namespace) -> int:
    config = load_analyze_config(args.config)
    write_artifact_kind(config.log_dir.parent, ArtifactKind.ALIGNMENT_BUNDLE)
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


def _run_compare(args: argparse.Namespace) -> int:
    from .alignment_compare import compare_reports, render_comparison

    if args.limit < 0:
        raise ValueError("compare --limit must be nonnegative")
    comparison = compare_reports(args.baseline, args.candidate)
    if args.json:
        print(json.dumps(comparison, indent=2))
    else:
        print(render_comparison(comparison, args.limit))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "prepare-workload":
            from alignment.workload_input import prepare_workload

            options = vars(args).copy()
            options.pop("command")
            print(json.dumps(prepare_workload(**options), indent=2))
            return 0
        if args.command == "sim":
            return _run_sim(args)
        if args.command == "profile":
            return _run_profile(args)
        if args.command == "timing-predict":
            return _run_timing_predict(args)
        if args.command == "analyze":
            return _run_analyze(args)
        if args.command == "compare":
            return _run_compare(args)
        raise AssertionError(f"unhandled alignment command {args.command!r}")
    except (KeyError, OSError, ValueError) as exc:
        print(f"[invalid] alignment {args.command}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
