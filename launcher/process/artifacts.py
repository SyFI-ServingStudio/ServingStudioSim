"""Stage artifact validation independent of child-process exit codes."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ArtifactValidationError(RuntimeError):
    """A stage exited successfully but did not publish a valid contract."""


@dataclass(frozen=True, slots=True)
class ArtifactValidation:
    paths: tuple[Path, ...]


def validate_simulation_artifacts(log_dir: Path) -> ArtifactValidation:
    summary_path = log_dir / "summary.json"
    run_meta_path = log_dir / "raw" / "run_meta.json"
    summary = _read_json_object(summary_path)
    run_meta = _read_json_object(run_meta_path)

    required_summary_fields = {
        "cause",
        "requests_total",
        "requests_finished",
        "total_tokens",
        "sim_ms",
    }
    _require_fields(summary_path, summary, required_summary_fields)
    _require_fields(run_meta_path, run_meta, {"schema_version", "workers"})

    paths = [summary_path, run_meta_path]
    requests_total = summary.get("requests_total")
    if not isinstance(requests_total, int) or isinstance(requests_total, bool):
        raise ArtifactValidationError(
            f"{summary_path} field requests_total must be an integer"
        )
    if requests_total > 0:
        request_slo = log_dir / "raw" / "request_slo.parquet"
        _validate_parquet(request_slo, {"request_id", "completed"})
        paths.append(request_slo)
        # A run whose arrived requests are never admitted has no state rows; the
        # streaming writer intentionally creates no empty parquet file.
        request_state = log_dir / "raw" / "request_state.parquet"
        if request_state.exists():
            _validate_parquet(request_state, {"logging_time", "n_admitted"})
            paths.append(request_state)
    return ArtifactValidation(paths=tuple(paths))


def validate_trace_artifacts(log_dir: Path) -> ArtifactValidation:
    traces = tuple(sorted((log_dir / "traces").glob("*.pftrace.gz")))
    if not traces:
        raise ArtifactValidationError(f"no Perfetto trace under {log_dir / 'traces'}")
    for trace_path in traces:
        try:
            with gzip.open(trace_path, "rb") as stream:
                while stream.read(1024 * 1024):
                    pass
        except (OSError, EOFError) as error:
            raise ArtifactValidationError(f"invalid gzip trace {trace_path}: {error}") from error
    return ArtifactValidation(paths=traces)


def validate_json_outputs(directory: Path) -> ArtifactValidation:
    paths = tuple(sorted(directory.glob("*.json")))
    if not paths:
        raise ArtifactValidationError(f"no JSON artifacts under {directory}")
    for path in paths:
        _read_json(path)
    return ArtifactValidation(paths=paths)


def validate_render_artifacts(log_dir: Path) -> ArtifactValidation:
    payloads = tuple(sorted((log_dir / "payloads").glob("*.json")))
    plots = tuple(sorted((log_dir / "plots").glob("*.png")))
    if payloads and not plots:
        raise ArtifactValidationError(
            f"renderer produced no PNG artifacts for {len(payloads)} payload(s) under {log_dir}"
        )
    for path in plots:
        try:
            if path.stat().st_size == 0:
                raise ArtifactValidationError(f"empty rendered artifact {path}")
        except OSError as error:
            raise ArtifactValidationError(
                f"unreadable rendered artifact {path}: {error}"
            ) from error
    return ArtifactValidation(paths=plots)


def validate_timing_prediction_artifacts(log_dir: Path) -> ArtifactValidation:
    manifests = tuple(sorted((log_dir / "raw" / "cost_manifest").glob("*.json")))
    cost_logs = tuple(sorted((log_dir / "raw" / "cost_log").glob("*.parquet")))
    if not manifests or not cost_logs:
        raise ArtifactValidationError(
            "timing prediction needs cost_manifest JSON and cost_log parquet "
            f"under {log_dir / 'raw'}"
        )
    for path in manifests:
        _read_json_object(path)
    for path in cost_logs:
        _validate_parquet(path, {"total_time_ms"})
    return ArtifactValidation(paths=(*manifests, *cost_logs))


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ArtifactValidationError(f"{path} must contain a JSON object")
    return payload


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactValidationError(f"invalid JSON artifact {path}: {error}") from error


def _require_fields(path: Path, payload: dict[str, Any], fields: set[str]) -> None:
    missing = sorted(fields - payload.keys())
    if missing:
        raise ArtifactValidationError(f"{path} is missing fields: {', '.join(missing)}")


def _validate_parquet(path: Path, required_columns: set[str]) -> None:
    try:
        import pyarrow.parquet as parquet

        columns = set(parquet.ParquetFile(path).schema_arrow.names)
    except Exception as error:
        raise ArtifactValidationError(f"invalid parquet artifact {path}: {error}") from error
    missing = sorted(required_columns - columns)
    if missing:
        raise ArtifactValidationError(
            f"{path} is missing parquet columns: {', '.join(missing)}"
        )
