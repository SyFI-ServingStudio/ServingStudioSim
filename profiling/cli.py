"""Command line facade for existing L1 profiling entries.

This module is intentionally thin: it parses human/agent inputs, then calls the
generated public functions on ``profiling.perf_api``. It must not call runners
or ``run_profile_batch`` directly, preserving the L1 entry-point invariant.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from launcher.managed_job import ManagedJob
from profiling import perf_api
from profiling.artifacts import (
    MEASUREMENT_METADATA_FILENAME,
    PROFILE_CURVE_FILENAME,
    PROFILE_METADATA_FILENAME,
    complete_profile_artifacts,
    measurement_artifact_names,
    prepare_profile_artifacts,
    write_measurement_metadata,
    write_profile_metadata,
)
from profiling.db.args import KernelArgs
from profiling.db.batch import ProfileProvenance
from profiling.db.registry import KernelProfilerSpec, iter_kernel_profiler_specs
from profiling.db.table import MissingEntry
from profiling.facade import run_kind_times
from profiling.gpu_catalog import resolve_gpu_spec
from profiling.runners.metrics import CommMetrics, ComputeMetrics, Metrics


def main(
    argv: Sequence[str] | None = None,
    *,
    prog: str = "python -m profiling",
) -> int:
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)
    try:
        return int(args.command_fn(args))
    except Exception as exc:
        if getattr(args, "json", False):
            print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2


def build_parser(*, prog: str = "python -m profiling") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run/query existing VibeSim L1 profile entries through profiling.perf_api.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List registered kernel/backend profilers.")
    list_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    list_parser.set_defaults(command_fn=_cmd_list)

    for command, help_text in (
        ("query", "Read profile rows without JIT or refresh."),
        ("count-missing", "Count missing rows without running profilers."),
        ("run", "Fill missing rows or force-refresh rows through perf_api."),
    ):
        command_parser = subparsers.add_parser(command, help=help_text)
        _add_common_profile_args(command_parser)
        if command == "run":
            command_parser.add_argument(
                "--force",
                action="store_true",
                help="Refresh all specs even when profile.db already has rows.",
            )
            command_parser.add_argument(
                "--output-dir",
                type=Path,
                help=(
                    "Write an immutable request/results/curve snapshot here. "
                    "Required when invoked from a managed Agent turn."
                ),
            )
        command_parser.set_defaults(
            command_fn={
                "query": _cmd_query,
                "count-missing": _cmd_count_missing,
                "run": _cmd_run,
            }[command]
        )

    measure_parser = subparsers.add_parser(
        "measure",
        help="Sustained per-launch trend + NVML telemetry for one CUPTI kernel spec.",
    )
    _add_common_profile_args(measure_parser)
    measure_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Where to write CSV / summary.json / plots. Defaults to ./measure_<table>_<backend>.",
    )
    measure_parser.add_argument("--duration-s", type=float, default=10.0)
    measure_parser.add_argument("--telemetry-hz", type=float, default=20.0)
    measure_parser.add_argument(
        "--no-telemetry",
        dest="telemetry",
        action="store_false",
        help="Capture launch durations only (no NVML telemetry).",
    )
    measure_parser.add_argument(
        "--no-clear-l2",
        dest="clear_l2",
        action="store_false",
        help="Warm continuous window (no per-launch L2 displacement); reveals power/clock drift.",
    )
    measure_parser.set_defaults(command_fn=_cmd_measure, clear_l2=True, telemetry=True)
    return parser


def _add_common_profile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("table", help="Registry table stem, e.g. single_gemm.")
    parser.add_argument("--backend", required=True, help="Registered backend, e.g. torch.")
    parser.add_argument(
        "--spec",
        action="append",
        default=[],
        help="One JSON object spec. May be repeated.",
    )
    parser.add_argument(
        "--specs",
        type=Path,
        help="Path to JSON/JSONL specs. JSON may be an object, a list, or {'specs': [...]}.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        help="profile.db path. Defaults to profiling.perf_api.DB_PATH.",
    )
    parser.add_argument("--gpu-name", help="DB gpu_name key. Defaults to CUDA device 0 name.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")


def _cmd_list(args: argparse.Namespace) -> int:
    rows = [_profiler_spec_summary(profiler_spec) for profiler_spec in iter_kernel_profiler_specs()]
    if args.json:
        print(json.dumps({"ok": True, "profilers": rows}, indent=2, sort_keys=True))
        return 0

    if not rows:
        print("No profiler specs registered.")
        return 0
    _print_table(
        rows,
        columns=[
            "table",
            "kernel_kind",
            "backend",
            "args",
            "metric_family",
            "subprocess_env",
            "get_fn",
            "count_fn",
        ],
    )
    return 0


def _cmd_count_missing(args: argparse.Namespace) -> int:
    specs = _load_specs(args.spec, args.specs)
    _set_db_path(args.db)
    count_fn = _count_facade(args.table)
    missing_count = count_fn(specs, backend=args.backend, gpu_name=args.gpu_name)
    payload = _base_payload(args, specs) | {"missing_count": missing_count}
    if args.json:
        print(json.dumps({"ok": True, **payload}, indent=2, sort_keys=True))
    else:
        _print_run_summary(payload)
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    specs = _load_specs(args.spec, args.specs)
    _set_db_path(args.db)
    results = _get_facade(args.table)(
        specs,
        backend=args.backend,
        gpu_name=args.gpu_name,
        force=False,
    )
    missing_count = _count_facade(args.table)(specs, backend=args.backend, gpu_name=args.gpu_name)
    payload = _result_payload(args, specs, results, mode="query", missing_count=missing_count)
    if args.json:
        print(json.dumps({"ok": True, **payload}, indent=2, sort_keys=True))
    else:
        _print_result_payload(payload)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    specs = _load_specs(args.spec, args.specs)
    _set_db_path(args.db)
    mode = "force-refresh" if args.force else "jit-fill"
    profiler_spec = _resolve_profiler_spec(args.table, args.backend)
    output_dir = args.output_dir.resolve() if args.output_dir is not None else None
    managed_job = ManagedJob.from_environment("kernel_profile")
    if managed_job is not None and output_dir is None:
        raise ValueError("managed kernel-profile run requires --output-dir")

    # Resource identity is generated before registration so the managed job carries
    # the Analyzer resource id, and before artifacts so the job can never outlive its
    # discovery identity. Direct development runs without a managed context still write
    # the same metadata (origin kind=development) — Analyzer discovery never depends on
    # a conversation backend.
    create_time = _utc_now()
    profile_id = _profile_id(output_dir)
    descriptor = {
        "table": args.table,
        "kernelKind": str(profiler_spec.kernel_kind),
        "backend": args.backend,
        "metricFamily": profiler_spec.metric_family.value,
        "pointCount": len(specs),
    }
    if managed_job is not None:
        assert output_dir is not None
        managed_job.register(
            output_dir,
            descriptor=descriptor,
            analyzer_resource_id=profile_id,
        )

    try:
        if managed_job is not None:
            managed_job.report("running")
        if output_dir is not None:
            prepare_profile_artifacts(
                output_dir,
                request_payload={
                    "schemaVersion": 1,
                    "jobKind": "kernel_profile",
                    "resourceId": profile_id,
                    "mode": mode,
                    **_base_payload(args, specs),
                },
                job_metadata=_profile_job_metadata(managed_job, descriptor, profile_id),
            )

        if args.force:
            outcome = run_kind_times(
                profiler_spec.kernel_kind,
                specs,
                backend=args.backend,
                gpu_name=args.gpu_name,
                db_path=perf_api.DB_PATH,
                jit_enabled=False,
                force=True,
            )
        else:
            perf_api.enable_jit_profiling()
            try:
                outcome = run_kind_times(
                    profiler_spec.kernel_kind,
                    specs,
                    backend=args.backend,
                    gpu_name=args.gpu_name,
                    db_path=perf_api.DB_PATH,
                    jit_enabled=True,
                    force=False,
                )
            finally:
                perf_api.disable_jit_profiling()
        results = outcome.results
        run_provenance = outcome.provenance

        missing_count = _count_facade(args.table)(
            specs,
            backend=args.backend,
            gpu_name=args.gpu_name,
        )
        payload = _result_payload(
            args,
            specs,
            results,
            mode=mode,
            missing_count=missing_count,
        )
        if output_dir is not None:
            if managed_job is not None:
                managed_job.report("analysis_running")
            curve_payload = complete_profile_artifacts(
                output_dir,
                profiler_spec=profiler_spec,
                specs=specs,
                result_payload=payload,
            )
            request_identifier, observed_gpu, _gpu_count = _outcome_provenance(
                run_provenance, args.gpu_name, forced=args.force
            )
            _validate_measured_gpu_identity(request_identifier, observed_gpu)
            resolved_canonical = (
                resolve_gpu_spec(request_identifier).canonical_name
                if request_identifier and resolve_gpu_spec(request_identifier) is not None
                else None
            )
            _ = write_profile_metadata(
                output_dir,
                profile_id=profile_id,
                profiler_spec=profiler_spec,
                specs=specs,
                requested_gpu_name=request_identifier,
                observed_gpu_name=observed_gpu,
                provenance_source="measurement" if observed_gpu else "cache_key",
                resolved_canonical_name=resolved_canonical,
                mode=mode,
                created_at=create_time,
            )
            payload["resource_id"] = profile_id
            payload["artifact_root"] = str(output_dir)
            payload["curve_path"] = str(output_dir / PROFILE_CURVE_FILENAME)
            payload["metadata_path"] = str(output_dir / PROFILE_METADATA_FILENAME)
        else:
            curve_payload = None

        command_ok = missing_count == 0
        if managed_job is not None:
            managed_job.report(
                "ready" if command_ok else "failed",
                summary=descriptor
                | {
                    "resourceId": profile_id,
                    "missingCount": missing_count,
                    "axes": [axis["key"] for axis in (curve_payload or {}).get("axes", [])],
                },
            )
        if args.json:
            print(json.dumps({"ok": command_ok, **payload}, indent=2, sort_keys=True))
        else:
            _print_result_payload(payload)
            if output_dir is not None:
                print(f"profile: {output_dir / PROFILE_METADATA_FILENAME}")
                print(f"curve: {output_dir / PROFILE_CURVE_FILENAME}")
        return 0 if command_ok else 1
    except KeyboardInterrupt:
        if managed_job is not None:
            managed_job.report("interrupted")
        raise
    except Exception:
        if managed_job is not None:
            managed_job.report("failed")
        raise


def _cmd_measure(args: argparse.Namespace) -> int:
    specs = _load_specs(args.spec, args.specs)
    if len(specs) != 1:
        raise ValueError(f"measure takes exactly one spec, got {len(specs)}")
    output_dir = args.output_dir or Path(f"measure_{args.table}_{args.backend}")
    profiler_spec = _resolve_profiler_spec(args.table, args.backend)
    create_time = _utc_now()
    measurement_id = _measurement_id(output_dir)
    descriptor = {
        "table": args.table,
        "backend": args.backend,
        "pointCount": 1,
        "durationSeconds": args.duration_s,
        "clearL2": args.clear_l2,
    }
    managed_job = ManagedJob.from_environment("kernel_measure")
    if managed_job is not None:
        managed_job.register(
            output_dir,
            descriptor=descriptor,
            analyzer_resource_id=measurement_id,
        )
        managed_job.report("running")
    try:
        result = perf_api.measure_kernel(
            args.table,
            specs[0],
            backend=args.backend,
            gpu_name=args.gpu_name,
            output_dir=output_dir,
            duration_s=args.duration_s,
            telemetry_hz=args.telemetry_hz,
            telemetry=args.telemetry,
            clear_l2=args.clear_l2,
        )
        observed_gpu = result.get("observed_gpu_name")
        if not observed_gpu:
            raise ValueError("kernel measurement did not report an observed physical GPU")
        _validate_measured_gpu_identity(args.gpu_name, observed_gpu)
        artifacts = list(result.get("artifacts", []))
        artifact_names = measurement_artifact_names(output_dir, artifacts)
        summary_file = "summary.json"
        if summary_file not in artifact_names:
            raise ValueError("kernel measurement did not produce summary.json")
        resolved_plots = sorted(
            name
            for name in artifact_names
            if name in ("runtime_trend.png", "runtime_telemetry.png")
        )
        _ = write_measurement_metadata(
            output_dir,
            measurement_id=measurement_id,
            profiler_spec=profiler_spec,
            shape=_strip_render_kwargs(specs[0]),
            requested_gpu_name=args.gpu_name,
            observed_gpu_name=observed_gpu,
            gpu_count=_spec_gpu_count(profiler_spec, specs),
            duration_s=args.duration_s,
            telemetry=args.telemetry,
            created_at=create_time,
            summary_file=summary_file,
            plots=resolved_plots,
            artifacts=artifact_names,
        )
        if managed_job is not None:
            managed_job.report(
                "ready",
                summary=descriptor
                | {
                    "resourceId": measurement_id,
                    "timeMs": result.get("time_ms"),
                },
            )
    except KeyboardInterrupt:
        if managed_job is not None:
            managed_job.report("interrupted")
        raise
    except Exception:
        if managed_job is not None:
            managed_job.report("failed")
        raise
    if args.json:
        payload = {"ok": True, **result, "resource_id": measurement_id}
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_measure_result(result)
        print(f"measurement: {output_dir / MEASUREMENT_METADATA_FILENAME}")
    return 0


def _print_measure_result(result: Mapping[str, Any]) -> None:
    print(f"kernel: {result['kernel_kind']}:{result['backend']}")
    print(f"gpu_index: {result['gpu_index']}")
    print(f"output_dir: {result['output_dir']}")
    if result.get("time_ms") is not None:
        print(f"trend_median_ms: {float(result['time_ms']):.6f}")
    if result.get("runner_error"):
        print(f"runner_error: {result['runner_error']}")
    print("artifacts:")
    for artifact in result.get("artifacts", []):
        print(f"  {artifact}")


def _load_specs(spec_values: Sequence[str], specs_path: Path | None) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for spec_value in spec_values:
        specs.extend(_coerce_specs_payload(json.loads(spec_value), label="--spec"))
    if specs_path is not None:
        if str(specs_path) == "-":
            text = sys.stdin.read()
        else:
            text = specs_path.read_text(encoding="utf-8")
        specs.extend(_parse_specs_text(text, label=str(specs_path)))
    if not specs:
        raise ValueError("provide at least one --spec JSON object or --specs file")
    return specs


def _parse_specs_text(text: str, *, label: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        raise ValueError(f"{label} is empty")
    try:
        return _coerce_specs_payload(json.loads(stripped), label=label)
    except json.JSONDecodeError:
        specs = []
        for line_number, line in enumerate(stripped.splitlines(), start=1):
            if line.strip():
                specs.extend(
                    _coerce_specs_payload(
                        json.loads(line),
                        label=f"{label}:{line_number}",
                    )
                )
        return specs


def _coerce_specs_payload(payload: Any, *, label: str) -> list[dict[str, Any]]:
    if isinstance(payload, Mapping) and "specs" in payload:
        return _coerce_specs_payload(payload["specs"], label=label)
    if isinstance(payload, Mapping):
        return [dict(payload)]
    if isinstance(payload, list):
        specs = []
        for index, item in enumerate(payload):
            if not isinstance(item, Mapping):
                raise ValueError(f"{label}[{index}] must be a JSON object")
            specs.append(dict(item))
        return specs
    raise ValueError(f"{label} must be a JSON object, list, or object with a 'specs' list")


def _set_db_path(db_path: Path | None) -> None:
    if db_path is not None:
        perf_api.DB_PATH = db_path


def _get_facade(table_name: str):
    name = f"get_{_validate_table_name(table_name)}_times"
    try:
        return getattr(perf_api, name)
    except AttributeError as exc:
        raise ValueError(f"no generated perf_api facade {name!r}") from exc


def _count_facade(table_name: str):
    name = f"count_missing_{_validate_table_name(table_name)}"
    try:
        return getattr(perf_api, name)
    except AttributeError as exc:
        raise ValueError(f"no generated perf_api facade {name!r}") from exc


def _validate_table_name(table_name: str) -> str:
    known_tables = {profiler_spec.table_name for profiler_spec in iter_kernel_profiler_specs()}
    if table_name in known_tables:
        return table_name
    raise ValueError(f"unknown profiler table {table_name!r}; known tables: {sorted(known_tables)}")


def _resolve_profiler_spec(table_name: str, backend: str) -> KernelProfilerSpec:
    matches = [
        profiler_spec
        for profiler_spec in iter_kernel_profiler_specs()
        if profiler_spec.table_name == table_name and profiler_spec.backend == backend
    ]
    if len(matches) == 1:
        return matches[0]
    known_backends = sorted(
        profiler_spec.backend
        for profiler_spec in iter_kernel_profiler_specs()
        if profiler_spec.table_name == table_name
    )
    if known_backends:
        raise ValueError(
            f"unknown backend {backend!r} for {table_name!r}; known backends: {known_backends}"
        )
    _validate_table_name(table_name)
    raise AssertionError("validated profiler table did not resolve")


def _profile_job_metadata(
    managed_job: ManagedJob | None,
    descriptor: dict[str, Any],
    profile_id: str,
) -> dict[str, Any]:
    origin: dict[str, Any]
    if managed_job is None:
        origin = {"kind": "development"}
    else:
        origin = {
            "kind": "managed",
            "jobId": managed_job.job_id,
            "jobResourceId": managed_job.resource_id or profile_id,
        }
    return {
        "schemaVersion": 1,
        "jobKind": "kernel_profile",
        "resourceId": profile_id,
        "descriptor": descriptor,
        "origin": origin,
    }


def _outcome_provenance(
    provenance: ProfileProvenance,
    requested_gpu_name: str | None,
    *,
    forced: bool = False,
) -> tuple[str | None, str | None, int | None]:
    """Project one typed per-invocation ``ProfileProvenance`` onto the artifact
    fields. Observed GPU is reported only for a genuine measurement (``source ==
    "measurement"``); a forced measurement that produced no physical GPU
    observation must fail loudly rather than silently downgrade to cache-only."""
    if forced and provenance.source != "measurement":
        raise ValueError(
            "force-refresh produced no worker-observed physical GPU; "
            "refusing to stamp cache-only provenance for a forced measurement"
        )
    requested = provenance.requested_gpu_name or requested_gpu_name
    if provenance.source == "measurement":
        if provenance.observed_gpu_name is None:
            raise ValueError(
                "measured batch recorded no worker-observed physical GPU; cannot stamp "
                "measurement provenance for the artifact"
            )
        return requested, provenance.observed_gpu_name, provenance.gpu_count
    return requested, None, None


def _validate_measured_gpu_identity(
    requested_gpu_name: str | None,
    observed_gpu_name: str | None,
) -> None:
    """Fail a measured job when the requested cache key and the worker-observed
    physical GPU cannot be canonicalized to the same SKU via ``gpu/spec.json`` —
    never infer a default GPU. Cache-only jobs (no observed GPU) skip validation."""
    if not requested_gpu_name or not observed_gpu_name:
        return
    if requested_gpu_name == observed_gpu_name:
        return
    from profiling.gpu_catalog import same_canonical_sku

    if not same_canonical_sku(requested_gpu_name, observed_gpu_name):
        raise ValueError(
            f"GPU identity mismatch: requested cache key {requested_gpu_name!r} and "
            f"observed physical GPU {observed_gpu_name!r} do not resolve to the same "
            "canonical SKU in gpu/spec.json"
        )


def _profile_id(output_dir: Path | None) -> str:
    """Stable ``kp_<uuid>`` identity for a profile artifact: reuses an existing
    valid id so re-runs never rotate identity behind a live resource."""
    if output_dir is not None:
        metadata_path = output_dir / PROFILE_METADATA_FILENAME
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = None
        if isinstance(metadata, dict) and _valid_profile_id(metadata.get("profile_id")):
            return metadata["profile_id"]
    return f"kp_{uuid.uuid4().hex}"


def _measurement_id(output_dir: Path | None) -> str:
    """Stable ``km_<uuid>`` identity for a measurement artifact."""
    if output_dir is not None:
        metadata_path = output_dir / MEASUREMENT_METADATA_FILENAME
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = None
        if isinstance(metadata, dict) and _valid_measurement_id(metadata.get("measurement_id")):
            return metadata["measurement_id"]
    return f"km_{uuid.uuid4().hex}"


def _valid_profile_id(value: object) -> bool:
    return _valid_resource_id(value, "kp_")


def _valid_measurement_id(value: object) -> bool:
    return _valid_resource_id(value, "km_")


def _valid_resource_id(value: object, prefix: str) -> bool:
    if not isinstance(value, str) or not value.startswith(prefix):
        return False
    suffix = value.removeprefix(prefix)
    return (
        1 <= len(suffix) <= 64
        and suffix.isascii()
        and all(
            character.islower() or character.isdigit() or character == "_" for character in suffix
        )
    )


def _spec_gpu_count(profiler_spec: KernelProfilerSpec, specs: list[dict[str, Any]]) -> int:
    if profiler_spec.gpu_count_fn is None:
        return 1
    counts = {int(profiler_spec.gpu_count_fn(spec)) for spec in specs}
    return max(counts) if counts and 0 not in counts else 1


def _strip_render_kwargs(spec: dict[str, Any]) -> dict[str, Any]:
    """The measurement metadata ``shape`` excludes routing-only keys like ``backend``."""
    return {key: value for key, value in spec.items() if key != "backend"}


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _profiler_spec_summary(profiler_spec: KernelProfilerSpec) -> dict[str, str]:
    args_fields = ",".join(field.name for field in fields(profiler_spec.args_schema))
    stem = profiler_spec.table_name
    return {
        "table": stem,
        "kernel_kind": profiler_spec.kernel_kind,
        "backend": profiler_spec.backend,
        "args": args_fields,
        "metric_family": profiler_spec.metric_family.value,
        "subprocess_env": profiler_spec.subprocess_env or "default_env",
        "get_fn": f"get_{stem}_times",
        "count_fn": f"count_missing_{stem}",
    }


def _base_payload(args: argparse.Namespace, specs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "table": args.table,
        "backend": args.backend,
        "db_path": str(perf_api.DB_PATH),
        "gpu_name": args.gpu_name,
        "spec_count": len(specs),
        "specs": specs,
    }


def _result_payload(
    args: argparse.Namespace,
    specs: list[dict[str, Any]],
    results: Sequence[Metrics | MissingEntry],
    *,
    mode: str,
    missing_count: int,
) -> dict[str, Any]:
    return _base_payload(args, specs) | {
        "mode": mode,
        "missing_count": missing_count,
        "results": [_result_to_payload(index, result) for index, result in enumerate(results)],
    }


def _result_to_payload(index: int, result: Metrics | MissingEntry) -> dict[str, Any]:
    if isinstance(result, MissingEntry):
        return {
            "index": index,
            "status": "missing",
            "kernel_kind": result.kernel_kind,
            "backend": result.backend,
            "gpu_name": result.gpu_name,
            "args": _jsonable(result.args),
        }
    if isinstance(result, ComputeMetrics):
        return {
            "index": index,
            "status": "ok",
            "metric_family": "compute",
            "metrics": _jsonable(result),
        }
    if isinstance(result, CommMetrics):
        return {
            "index": index,
            "status": "ok",
            "metric_family": "comm",
            "metrics": _jsonable(result),
        }
    raise TypeError(f"unexpected result type {type(result).__name__}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, KernelArgs):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _print_run_summary(payload: Mapping[str, Any]) -> None:
    print(f"table: {payload['table']}")
    print(f"backend: {payload['backend']}")
    print(f"db_path: {payload['db_path']}")
    if payload.get("gpu_name"):
        print(f"gpu_name: {payload['gpu_name']}")
    print(f"spec_count: {payload['spec_count']}")
    print(f"missing_count: {payload['missing_count']}")


def _print_result_payload(payload: Mapping[str, Any]) -> None:
    _print_run_summary(payload)
    print(f"mode: {payload['mode']}")
    rows = []
    for result in payload["results"]:
        row = {
            "index": str(result["index"]),
            "status": result["status"],
            "args": json.dumps(result.get("args", {}), sort_keys=True),
        }
        metrics = result.get("metrics", {})
        row.update({key: _format_value(value) for key, value in metrics.items()})
        rows.append(row)
    if rows:
        metric_columns = sorted({key for row in rows for key in row} - {"index", "status", "args"})
        _print_table(rows, columns=["index", "status", *metric_columns, "args"])


def _print_table(rows: Sequence[Mapping[str, Any]], *, columns: Sequence[str]) -> None:
    widths = {
        column: max(len(column), *(len(str(row.get(column, ""))) for row in rows))
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns))


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
