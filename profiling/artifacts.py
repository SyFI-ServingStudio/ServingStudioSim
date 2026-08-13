"""Immutable, UI-facing snapshots for explicit kernel profiling batches.

``profile.db`` remains the mutable L1 cache authority.  These JSON files record
exactly which submitted points and returned metrics belonged to one operator
job, so reopening that job does not substitute rows from a newer DB state.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from launcher.artifact_kind import ArtifactKind, write_artifact_kind
from profiling.db.registry import KernelProfilerSpec

PROFILE_REQUEST_FILENAME = "request.json"
PROFILE_RESULTS_FILENAME = "results.json"
PROFILE_CURVE_FILENAME = "curve.json"
PROFILE_JOB_METADATA_FILENAME = "job.meta.json"
# First-class Analyzer discovery metadata: the source of truth for the
# ``kernel_profile`` resource. Written by the CLI artifact path so direct development
# runs generate resource identity without any conversation backend.
PROFILE_METADATA_FILENAME = "kernel-profile.meta.json"
# First-class Analyzer discovery metadata for ``kernel_measure`` resources.
MEASUREMENT_METADATA_FILENAME = "kernel-measurement.meta.json"


def prepare_profile_artifacts(
    output_dir: Path,
    *,
    request_payload: dict[str, Any],
    job_metadata: dict[str, Any],
) -> None:
    """Create a fresh artifact root and persist inputs before GPU execution."""
    protected_names = (
        PROFILE_REQUEST_FILENAME,
        PROFILE_RESULTS_FILENAME,
        PROFILE_CURVE_FILENAME,
        PROFILE_JOB_METADATA_FILENAME,
    )
    existing = [name for name in protected_names if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(f"profile artifact root already contains immutable files: {existing}")
    write_artifact_kind(output_dir, ArtifactKind.KERNEL_PROFILE)
    _write_json_atomic(output_dir / PROFILE_REQUEST_FILENAME, request_payload)
    _write_json_atomic(output_dir / PROFILE_JOB_METADATA_FILENAME, job_metadata)


def complete_profile_artifacts(
    output_dir: Path,
    *,
    profiler_spec: KernelProfilerSpec,
    specs: list[dict[str, Any]],
    result_payload: dict[str, Any],
) -> dict[str, Any]:
    """Persist exact results and the canonical curve/grid visualization payload."""
    curve_payload = build_curve_payload(profiler_spec, specs, result_payload)
    _write_json_atomic(output_dir / PROFILE_RESULTS_FILENAME, result_payload)
    _write_json_atomic(output_dir / PROFILE_CURVE_FILENAME, curve_payload)
    return curve_payload


def build_curve_payload(
    profiler_spec: KernelProfilerSpec,
    specs: list[dict[str, Any]],
    result_payload: dict[str, Any],
) -> dict[str, Any]:
    """Derive ordered varying axes from KernelArgs declaration order."""
    argument_names = [field.name for field in fields(profiler_spec.args_schema)]
    axis_values: dict[str, list[Any]] = {}
    fixed_args: dict[str, Any] = {}
    for argument_name in argument_names:
        values = _unique_values([spec.get(argument_name) for spec in specs])
        if len(values) > 1:
            axis_values[argument_name] = values
        elif values:
            fixed_args[argument_name] = values[0]

    result_rows = result_payload.get("results", [])
    rows = []
    for index, spec in enumerate(specs):
        result = (
            result_rows[index]
            if index < len(result_rows)
            else {
                "index": index,
                "status": "missing",
            }
        )
        rows.append(
            {
                "index": index,
                "coordinates": {
                    argument_name: spec.get(argument_name) for argument_name in axis_values
                },
                "args": spec,
                "status": result.get("status", "missing"),
                "metrics": result.get("metrics"),
            }
        )

    metric_family = profiler_spec.metric_family.value
    series = _metric_series(metric_family)
    axes = [
        {"key": argument_name, "values": values} for argument_name, values in axis_values.items()
    ]
    return {
        "schemaVersion": 1,
        "resourceKind": "kernel_profile_curve",
        "kernelKind": str(profiler_spec.kernel_kind),
        "table": profiler_spec.table_name,
        "backend": profiler_spec.backend,
        "metricFamily": metric_family,
        "axes": axes,
        "fixedArgs": fixed_args,
        "layout": {
            "xAxis": axes[0]["key"] if axes else None,
            "yAxis": axes[1]["key"] if len(axes) > 1 else None,
            "facets": [axis["key"] for axis in axes[2:]],
        },
        "series": series,
        "rows": rows,
    }


def _metric_series(metric_family: str) -> list[dict[str, Any]]:
    if metric_family == "compute":
        return [
            {"metric": "time_ms", "unit": "ms", "lowerIsBetter": True},
            {"metric": "tflops", "unit": "TFLOP/s", "lowerIsBetter": False},
            {
                "metric": "memory_bandwidth_gbps",
                "unit": "GB/s",
                "lowerIsBetter": False,
            },
            {"metric": "energy_j", "unit": "J", "lowerIsBetter": True},
        ]
    if metric_family == "comm":
        return [
            {"metric": "time_ms", "unit": "ms", "lowerIsBetter": True},
            {"metric": "algbw_gbps", "unit": "GB/s", "lowerIsBetter": False},
            {"metric": "busbw_gbps", "unit": "GB/s", "lowerIsBetter": False},
            {"metric": "energy_j", "unit": "J", "lowerIsBetter": True},
        ]
    raise ValueError(f"unsupported metric family {metric_family!r}")


def build_profile_metadata(
    *,
    profile_id: str,
    profiler_spec: KernelProfilerSpec,
    specs: list[dict[str, Any]],
    requested_gpu_name: str,
    observed_gpu_name: str | None,
    provenance_source: str,
    resolved_canonical_name: str | None,
    mode: str,
    created_at: str,
) -> dict[str, Any]:
    """The immutable ``kernel-profile.meta.json`` body — the Analyzer discovery
    source of truth. GPU provenance is explicit: the requested DB cache key, the
    observed physical name (only when a worker actually ran), and the source that tied
    them together. ``observed_gpu_name`` stays ``None`` for a cached-only job."""
    return {
        "schema_version": 1,
        "profile_id": profile_id,
        "kernel": {
            "kind": str(profiler_spec.kernel_kind),
            "table": profiler_spec.table_name,
            "backend": profiler_spec.backend,
            "metric_family": profiler_spec.metric_family.value,
        },
        "gpu": {
            "cache_key": requested_gpu_name,
            "observed_name": observed_gpu_name,
            "count": int(_spec_gpu_count(profiler_spec, specs)),
        },
        "provenance": {
            "source": provenance_source,
            "resolved_canonical_name": resolved_canonical_name,
        },
        "mode": mode,
        "args": specs,
        "created_at": created_at,
        "artifacts": {
            "request": PROFILE_REQUEST_FILENAME,
            "results": PROFILE_RESULTS_FILENAME,
            "curve": PROFILE_CURVE_FILENAME,
            "job_metadata": PROFILE_JOB_METADATA_FILENAME,
        },
    }


def write_profile_metadata(output_dir: Path, **fields_kwargs: Any) -> dict[str, Any]:
    """Atomically persist ``kernel-profile.meta.json`` after a job completes, before
    the managed job reports ready."""
    metadata = build_profile_metadata(**fields_kwargs)
    _write_json_atomic(output_dir / PROFILE_METADATA_FILENAME, metadata)
    return metadata


def build_measurement_metadata(
    *,
    measurement_id: str,
    profiler_spec: KernelProfilerSpec,
    shape: dict[str, Any],
    requested_gpu_name: str | None,
    observed_gpu_name: str | None,
    gpu_count: int,
    duration_s: float,
    telemetry: bool,
    created_at: str,
    summary_file: str,
    plots: list[str],
    artifacts: list[str],
) -> dict[str, Any]:
    """The immutable ``kernel-measurement.meta.json`` body."""
    return {
        "schema_version": 1,
        "measurement_id": measurement_id,
        "kernel": {
            "kind": str(profiler_spec.kernel_kind),
            "table": profiler_spec.table_name,
            "backend": profiler_spec.backend,
            "metric_family": profiler_spec.metric_family.value,
        },
        "gpu": {
            "cache_key": requested_gpu_name,
            "observed_name": observed_gpu_name,
            "count": gpu_count,
        },
        "shape": shape,
        "duration_s": duration_s,
        "telemetry": telemetry,
        "created_at": created_at,
        "summary_file": summary_file,
        "plots": sorted(plots),
        "artifacts": sorted(artifacts),
    }


def write_measurement_metadata(output_dir: Path, **fields_kwargs: Any) -> dict[str, Any]:
    """Atomically persist ``kernel-measurement.meta.json`` after a capture, before
    the managed job reports ready."""
    write_artifact_kind(output_dir, ArtifactKind.KERNEL_MEASUREMENT)
    metadata = build_measurement_metadata(**fields_kwargs)
    _write_json_atomic(output_dir / MEASUREMENT_METADATA_FILENAME, metadata)
    return metadata


def measurement_artifact_names(output_dir: Path, artifact_paths: list[str]) -> list[str]:
    """Return unique direct-child basenames for measurement metadata.

    The worker returns absolute paths for compatibility with CLI callers. Analyzer
    metadata is deliberately filesystem-opaque, so each declaration must resolve to
    exactly one direct child of the measurement root before its basename is stored.
    """

    artifact_root = output_dir.resolve()
    names: list[str] = []
    for artifact_path in artifact_paths:
        candidate = Path(artifact_path)
        resolved_candidate = (
            candidate.resolve()
            if candidate.is_absolute()
            else (artifact_root / candidate).resolve()
        )
        try:
            relative = resolved_candidate.relative_to(artifact_root)
        except ValueError as error:
            raise ValueError(
                f"measurement artifact is outside output directory: {artifact_path!r}"
            ) from error
        if len(relative.parts) != 1 or relative.name.startswith("."):
            raise ValueError(
                f"measurement artifact must be a direct-child basename: {artifact_path!r}"
            )
        if relative.name in names:
            raise ValueError(f"duplicate measurement artifact: {relative.name!r}")
        names.append(relative.name)
    return sorted(names)


def _spec_gpu_count(profiler_spec: KernelProfilerSpec, specs: list[dict[str, Any]]) -> int:
    """GPU count for the artifact metadata — the configured chunk shape for these
    specs (comm kinds declare a ``gpu_count_fn``; compute kinds default to 1)."""
    if profiler_spec.gpu_count_fn is None:
        return 1
    counts = {int(profiler_spec.gpu_count_fn(spec)) for spec in specs}
    if not counts or 0 in counts:
        return 1
    return max(counts)


def _unique_values(values: list[Any]) -> list[Any]:
    unique: list[Any] = []
    encoded_values: set[str] = set()
    for value in values:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
        if encoded not in encoded_values:
            unique.append(value)
            encoded_values.add(encoded)
    return unique


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
