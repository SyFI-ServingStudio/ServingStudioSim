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

from profiling.db.registry import KernelProfilerSpec

PROFILE_REQUEST_FILENAME = "request.json"
PROFILE_RESULTS_FILENAME = "results.json"
PROFILE_CURVE_FILENAME = "curve.json"
PROFILE_JOB_METADATA_FILENAME = "job.meta.json"


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
    output_dir.mkdir(parents=True, exist_ok=True)
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
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
