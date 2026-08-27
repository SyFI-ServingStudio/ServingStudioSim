"""Compare two completed kernel-alignment reports without reopening raw inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPORT_RELATIVE_PATH = Path("reports/alignment_iteration_report.json")
METRICS = (
    "measured_ms",
    "simulated_ms",
    "signed_error_ms",
    "signed_error_pct",
    "absolute_error_ms",
    "absolute_error_pct",
)
NULLABLE_METRICS = frozenset({"signed_error_pct", "absolute_error_pct"})


def _report_path(path: Path) -> Path:
    return path / REPORT_RELATIVE_PATH if path.is_dir() else path


def _load_report(path: Path) -> tuple[Path, dict[str, Any]]:
    report_path = _report_path(path).resolve()
    if not report_path.is_file():
        raise ValueError(f"kernel-align report not found: {report_path}")
    report = json.loads(report_path.read_text())
    if not isinstance(report, dict) or report.get("available") is not True:
        raise ValueError(f"kernel-align report is unavailable: {report_path}")
    if not isinstance(report.get("comparison"), dict):
        raise ValueError(
            f"kernel-align report has no aggregate comparison fields: {report_path}; "
            "rerun alignment analyze with the current Analyzer"
        )
    return report_path, report


def _aggregate(value: Any, context: str) -> dict[str, float | int | None]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    n = value.get("n")
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError(f"{context}.n must be a nonnegative integer")
    out: dict[str, float | int | None] = {"n": n}
    for metric in METRICS:
        item = value.get(metric)
        if item is None and metric in NULLABLE_METRICS:
            out[metric] = None
            continue
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise ValueError(f"{context}.{metric} must be numeric")
        out[metric] = float(item)
    return out


def _scopes(report: dict[str, Any], context: str) -> dict[str, dict[str, float | int | None]]:
    comparison = report["comparison"]
    by_stage = comparison.get("by_stage")
    if not isinstance(by_stage, dict):
        raise ValueError(f"{context}.comparison.by_stage must be an object")
    scopes = {"all": _aggregate(comparison.get("all"), f"{context}.comparison.all")}
    for stage, value in by_stage.items():
        if not isinstance(stage, str) or not stage:
            raise ValueError(f"{context}.comparison.by_stage has an invalid stage name")
        scopes[stage] = _aggregate(value, f"{context}.comparison.by_stage.{stage}")
    return scopes


def _operations(report: dict[str, Any], context: str) -> dict[str, dict[str, float | int | None]]:
    rows = report.get("operations")
    if not isinstance(rows, list):
        raise ValueError(f"{context}.operations must be an array")
    operations: dict[str, dict[str, float | int | None]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("operation"), str):
            raise ValueError(f"{context}.operations[{index}] has no operation name")
        operation = row["operation"]
        if operation in operations:
            raise ValueError(f"{context}.operations repeats {operation!r}")
        aggregate = _aggregate(row.get("comparison"), f"{context}.operations[{index}].comparison")
        for field in ("missing_measured", "missing_simulated"):
            count = row.get(field)
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError(f"{context}.operations[{index}].{field} must be nonnegative")
            aggregate[field] = count
        operations[operation] = aggregate
    return operations


def _paired_change(
    baseline: dict[str, float | int | None] | None,
    candidate: dict[str, float | int | None] | None,
) -> dict[str, float | int | None]:
    row: dict[str, float | int | None] = {}
    for metric in ("n", "missing_measured", "missing_simulated", *METRICS):
        row[f"baseline_{metric}"] = baseline.get(metric) if baseline else None
        row[f"candidate_{metric}"] = candidate.get(metric) if candidate else None
    change_metrics = (
        "signed_error_ms",
        "signed_error_pct",
        "absolute_error_ms",
        "absolute_error_pct",
    )
    for metric in change_metrics:
        before = baseline.get(metric) if baseline else None
        after = candidate.get(metric) if candidate else None
        row[f"change_{metric}"] = (
            float(after) - float(before) if (before is not None and after is not None) else None
        )
    return row


def compare_reports(baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    baseline_report_path, baseline_report = _load_report(baseline_path)
    candidate_report_path, candidate_report = _load_report(candidate_path)
    baseline_scopes = _scopes(baseline_report, "baseline")
    candidate_scopes = _scopes(candidate_report, "candidate")
    scope_names = ["all", *sorted((baseline_scopes.keys() | candidate_scopes.keys()) - {"all"})]
    scopes = [
        {
            "scope": scope,
            **_paired_change(baseline_scopes.get(scope), candidate_scopes.get(scope)),
        }
        for scope in scope_names
    ]

    baseline_operations = _operations(baseline_report, "baseline")
    candidate_operations = _operations(candidate_report, "candidate")
    operations = [
        {
            "operation": operation,
            **_paired_change(
                baseline_operations.get(operation), candidate_operations.get(operation)
            ),
        }
        for operation in baseline_operations.keys() | candidate_operations.keys()
    ]
    operations.sort(
        key=lambda row: (
            -abs(float(row["change_absolute_error_ms"]))
            if row["change_absolute_error_ms"] is not None
            else float("inf"),
            str(row["operation"]),
        )
    )
    return {
        "schema_version": 1,
        "baseline_report": str(baseline_report_path),
        "candidate_report": str(candidate_report_path),
        "scopes": scopes,
        "operations": operations,
    }


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):+.2f}%"


def render_comparison(comparison: dict[str, Any], limit: int) -> str:
    lines = [
        "Scope          Signed baseline  Signed candidate  Change    "
        "Absolute baseline  Absolute candidate  Change",
    ]
    for row in comparison["scopes"]:
        lines.append(
            f"{row['scope']:<14}"
            f"{_pct(row['baseline_signed_error_pct']):>16}  "
            f"{_pct(row['candidate_signed_error_pct']):>16}  "
            f"{_pct(row['change_signed_error_pct']):>8}  "
            f"{_pct(row['baseline_absolute_error_pct']):>17}  "
            f"{_pct(row['candidate_absolute_error_pct']):>18}  "
            f"{_pct(row['change_absolute_error_pct']):>8}"
        )
    lines.extend(
        [
            "",
            "Operations sorted by absolute-error change",
            "Operation                                      Baseline abs ms  "
            "Candidate abs ms  Change abs ms",
        ]
    )
    for row in comparison["operations"][:limit]:
        baseline = row["baseline_absolute_error_ms"]
        candidate = row["candidate_absolute_error_ms"]
        change = row["change_absolute_error_ms"]
        lines.append(
            f"{str(row['operation'])[:46]:<46}"
            f"{('n/a' if baseline is None else f'{float(baseline):.3f}'):>17}  "
            f"{('n/a' if candidate is None else f'{float(candidate):.3f}'):>16}  "
            f"{('n/a' if change is None else f'{float(change):+.3f}'):>13}"
        )
    return "\n".join(lines)
