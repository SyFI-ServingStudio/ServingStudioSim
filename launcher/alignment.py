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
from .schema.loader import PresetError, _load_preset

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SUFFIXES = frozenset({".json", ".yaml", ".yml"})
ALIGNMENT_MANIFEST_NAME = "alignment_manifest.json"
# Bumped when either typed manifest shape changes; must match the analyzer's
# SCHEMA_VERSION in analyzer/rust/src/alignment_input.rs.
ALIGNMENT_MANIFEST_SCHEMA_VERSION = 8


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
    sim.add_argument(
        "--gpu-time-multiplier-from",
        type=Path,
        default=None,
        metavar="KERNEL_ALIGN_DIR",
        help=(
            "Kernel-align analysis dir; auto-injects its derived "
            "recommended_gpu_time_multiplier into the worker as an override."
        ),
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


def _preset_backends(preset: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    """Re-nest the sim preset's flat ``"pool/role": [candidates]`` backend map
    into the ``{pool: {role: [candidates]}}`` form the timing predictor expects —
    the same shape the launcher resolves into a completed run's params.json. The
    backend policy is part of the simulated CostTree identity, so this must match
    what the simulation will use.
    """
    flat = preset.get("backends", {})
    if not isinstance(flat, dict):
        raise ValueError("simulation preset backends must be a mapping")
    nested: dict[str, dict[str, list[str]]] = {}
    for key, candidates in flat.items():
        pool, separator, role = str(key).partition("/")
        if not separator or not pool or not role:
            raise ValueError(f"simulation preset backend key {key!r} must be 'pool/role'")
        if (
            not isinstance(candidates, list)
            or not candidates
            or not all(isinstance(candidate, str) for candidate in candidates)
        ):
            raise ValueError(
                f"simulation preset backend {key!r} must map to a non-empty string list"
            )
        nested.setdefault(pool, {})[role] = list(candidates)
    return nested


def _read_recommended_multiplier(analysis_log_dir: Path) -> float:
    """Read the duty-cycle multiplier the kernel-align pass derived from the
    measured side (Σ measured_gpu_cycle_ms / Σ measured_ms)."""
    report_path = analysis_log_dir / "reports" / "alignment_iteration_report.json"
    if not report_path.is_file():
        raise ValueError(
            f"kernel-align report not found: {report_path}; run the kernel-align analyze first"
        )
    report = json.loads(report_path.read_text())
    meta = report.get("meta") if isinstance(report, dict) else None
    multiplier = meta.get("recommended_gpu_time_multiplier") if isinstance(meta, dict) else None
    if not isinstance(multiplier, (int, float)) or isinstance(multiplier, bool) or multiplier < 1.0:
        raise ValueError(
            f"kernel-align report has no valid recommended_gpu_time_multiplier: {report_path}"
        )
    return float(multiplier)


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


def _validate_e2e_profile_pair(
    nsys_result: dict[str, Any],
    workload_result: dict[str, Any],
    *,
    require_same_trace: bool = True,
) -> None:
    """Reject an E2E profile pair that cannot describe the same serving run.

    Device ordinals may differ between captures, but the logical topology and
    replay source are part of the comparison identity and must remain equal.
    """
    nsys_kind = nsys_result.get("profile_kind")
    if nsys_kind not in {None, "nsys"}:
        raise ValueError(f"profile_log_dir must contain an nsys profile, got {nsys_kind!r}")
    workload_kind = workload_result.get("profile_kind")
    if workload_result is not nsys_result and workload_kind != "workload_metrics":
        raise ValueError(
            "workload_profile_log_dir must contain a workload_metrics profile, "
            f"got {workload_kind!r}"
        )

    identity_fields = ("gpu", "server_tp_size", "server_dp_size")
    for field in identity_fields:
        if nsys_result.get(field) != workload_result.get(field):
            raise ValueError(
                f"E2E profile provenance mismatch for {field}: "
                f"{nsys_result.get(field)!r} != {workload_result.get(field)!r}"
            )
    nsys_drive = nsys_result.get("drive_summary")
    workload_drive = workload_result.get("drive_summary")
    if isinstance(nsys_drive, dict) and isinstance(workload_drive, dict):
        fields = ("source_trace", "frontend_type") if require_same_trace else ("frontend_type",)
        for field in fields:
            matches = (
                _same_trace_source(nsys_drive.get(field), workload_drive.get(field))
                if field == "source_trace"
                else nsys_drive.get(field) == workload_drive.get(field)
            )
            if not matches:
                raise ValueError(
                    f"E2E profile provenance mismatch for drive_summary.{field}: "
                    f"{nsys_drive.get(field)!r} != {workload_drive.get(field)!r}"
                )


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
    shapes share only the measured-profile anchor.
    """
    profile_result = _load_profile_result(config.profile_log_dir)

    config.log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.log_dir / ALIGNMENT_MANIFEST_NAME
    common = {
        "schema_version": ALIGNMENT_MANIFEST_SCHEMA_VERSION,
        "analysis_log_dir": str(config.log_dir),
        "profile_log_dir": str(config.profile_log_dir),
    }

    if config.iteration.enabled:
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
        manifest = {
            **common,
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
        _simulation_target(_load_simulation_params(config.simulation_log_dir))
        workload_profile_result = _load_profile_result(config.workload_profile_log_dir)
        if config.workload_profile_log_dir.resolve() == config.profile_log_dir.resolve():
            workload_profile_result = profile_result
        uses_nsys = config.workload.enabled or config.e2e.server_gpu_throughput
        _validate_e2e_profile_pair(
            profile_result,
            workload_profile_result,
            require_same_trace=uses_nsys,
        )
        request_timings_result = _optional_profile_artifact(
            workload_profile_result, "request_timings_jsonl"
        )
        manifest = {
            **common,
            "workload_profile_log_dir": str(config.workload_profile_log_dir),
            "parsed_nsys": (
                str(_profile_artifact(profile_result, "parsed_nsys")) if uses_nsys else None
            ),
            "simulation_log_dir": str(config.simulation_log_dir),
            "metrics_jsonl": str(_profile_artifact(workload_profile_result, "metrics_jsonl")),
            "replay_result": str(_profile_artifact(workload_profile_result, "replay_result")),
            "request_timings_result": (
                str(request_timings_result) if request_timings_result else None
            ),
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
    overrides: list[str] | None = None,
) -> int:
    from .__main__ import main as launcher_main

    argv = [str(path), "--build-type", build_type]
    for override in overrides or []:
        argv.extend(["--override", override])
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
        # A prediction is not complete until Analyzer has materialized its cost
        # subjects. In particular, unlocked optimality needs the cached grid-peak
        # sidecar; skipping analysis silently collapses R3 onto R2 in the UI.
        return timing_predict_main([str(config_path), "--build-type", build_type])
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2


def _launch_alignment_analysis(log_dir: Path, *, build_type: str, subjects: list[str]) -> bool:
    from .exec import analyzer_binary_path, cargo_build_analyzer, run_alignment_analysis

    if not cargo_build_analyzer(build_type):
        return False
    if not analyzer_binary_path(build_type).is_file():
        print("[alignment] analyzer build failed", file=sys.stderr)
        return False
    run_alignment_analysis(log_dir, build_type, subjects)
    return True


def _run_sim(args: argparse.Namespace) -> int:
    preset = _load_simulation_preset(args.config)
    overrides: list[str] = []
    if args.gpu_time_multiplier_from is not None:
        multiplier = _read_recommended_multiplier(args.gpu_time_multiplier_from)
        overrides.append(f"pools.main.groups.0.worker.gpu_time_multiplier={multiplier}")
        print(
            f"[alignment] injecting worker.gpu_time_multiplier={multiplier} "
            f"from kernel-align {args.gpu_time_multiplier_from}"
        )
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
        overrides=overrides,
    )


def _run_profile(args: argparse.Namespace) -> int:
    config = load_profile_config(args.config)
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
    write_artifact_kind(config.log_dir.parent, ArtifactKind.ALIGNMENT_BUNDLE)
    # Read the sim *preset* (not a completed run): timing-predict is kernel-only,
    # so it needs the gpu / arch / backends but never a finished simulation. This
    # lets it run before the sim, so kernel-align can derive the multiplier the
    # sim then bakes in.
    preset = _load_simulation_preset(config.simulation_preset)
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
            output_dir=config.log_dir,
            gpu=gpu,
            arch=arch,
            backends=_preset_backends(preset),
            input_spec=config.input_builder,
        )
    )
    _snapshot_config(args.config, config.log_dir, "timing_predict")
    print(f"[alignment] timing-predict: {build_result.predict_config}")
    return _launch_timing_predict(build_result.predict_config, build_type=args.build_type)


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
