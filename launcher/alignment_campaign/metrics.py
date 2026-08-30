"""The formula table, and reading it off completed Analyzer reports.

Formulas live in the engine, not in a pack. They depend only on the Analyzer
report schema, which is model-independent, so copying them into every pack would
produce N copies to keep in sync — and `extract` without a pack still needs a
default set. What a pack owns is the *policy*: tolerances and the rationale for
each exception (`acceptance.yaml`). Each metric's formula string travels with
its value into the output and into a recorded golden's provenance, so the
derivation stays visible in the report.

Metrics are grouped and emitted in the tight -> loose order that
`skills/top-align-with-framework` prescribes, because a failure at a tight level
explains the loose ones: kernel timing, then duty cycle, then workload
structure, then end-to-end latency and throughput.

## Why `iteration` schema 1 is refused rather than read

`alignment_iteration_report` v1 had no `comparison` block: its kernel error was
`total_iteration.abs_relative_error_pct`, an unweighted mean over iterations,
where v2 reports a duration-weighted error over all of them. Those are different
statistics, so reading both would mean either two metric names to keep aligned
forever, or one name whose meaning turns over with the schema — and a golden key
that silently switches statistic is worse than one that goes missing.

The analyzer settled the question: `ALIGNMENT_ITERATION_SCHEMA_VERSION` is 2 and
nothing in the tree emits 1, so the v1 path was reachable only by a report old
enough that its numbers predate the current mapping. `measure_case` now rejects
it by name. `e2e` and `workload` are still v1 — that is the version the analyzer
writes today (`analyzer/rust/src/io.rs`), not a legacy one.

Only three small reports are read (iteration 132 KB - 1 MB, e2e ~6 KB, workload
~4 KB). `alignment_kernel_inventory.jsonl` reaches 338 MB and is never opened.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

METRICS_SCHEMA_VERSION = 1

#: Report schema versions the formula table knows how to read.
SUPPORTED_SCHEMAS = {"iteration": (2,), "e2e": (1,), "workload": (1,)}

#: Where each report sits inside a case run directory.
REPORT_LOCATIONS = {
    "iteration": Path("analysis_kernel/reports/alignment_iteration_report.json"),
    "e2e": Path("analysis_e2e/reports/alignment_e2e_report.json"),
    "workload": Path("analysis_e2e/reports/alignment_workload_report.json"),
}

#: Emission order, tight -> loose.
GROUP_ORDER = ("kernel", "duty_cycle", "workload", "e2e")
GROUP_TITLES = {
    "kernel": "Check 1 — per-iteration kernel timing",
    "duty_cycle": "Check 2 — GPU duty cycle",
    "workload": "Workload structure",
    "e2e": "Check 3 — TTFT / TPOT / throughput",
}


#: How a tolerance reads for a metric. `abs` bounds |value| (a signed error);
#: `min` is a floor (mapping coverage, where higher is better); `max` is a
#: ceiling (the duty-cycle multiplier, where a large value means host bubbles
#: that deserve an explanation rather than silent absorption).
BOUNDS = ("abs", "min", "max")


@dataclass(frozen=True)
class MetricSpec:
    name: str
    group: str
    report: str
    formula: str
    read: Callable[[dict[str, Any]], float | None]
    bound: str = "abs"

    def exceeds(self, value: float, tolerance: float) -> bool:
        if self.bound == "min":
            return value < tolerance
        if self.bound == "max":
            return value > tolerance
        return abs(value) > tolerance

    def margin(self, value: float, tolerance: float) -> float:
        """How much of the tolerance this value uses, as a fraction. Used only
        to decide WARN (close to the edge) versus OK."""
        if self.bound == "min":
            return 0.0 if tolerance == 0 else (100.0 - value) / max(100.0 - tolerance, 1e-9)
        if self.bound == "max":
            return 0.0 if tolerance == 0 else value / tolerance
        return 0.0 if tolerance == 0 else abs(value) / tolerance


def _path(document: Any, *keys: str) -> Any:
    node = document
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _ratio_pct(simulated: Any, measured: Any) -> float | None:
    """`(simulated / measured - 1) * 100`, the sign convention every alignment
    table in this repo uses: positive means the simulator is higher/slower."""
    if not isinstance(simulated, (int, float)) or not isinstance(measured, (int, float)):
        return None
    if measured == 0:
        return None
    return (simulated / measured - 1.0) * 100.0


def _scalar(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _metric_mean_ratio(report: dict, block: str, metric: str, sim: str, meas: str):
    return _ratio_pct(
        _path(report, block, metric, sim, "mean"), _path(report, block, metric, meas, "mean")
    )


# ── iteration report (kernel + duty cycle) ───────────────────────────────────

_ITERATION = (
    MetricSpec(
        "kernel_abs_pct", "kernel", "iteration",
        "iteration.comparison.all.absolute_error_pct",
        lambda d: _scalar(_path(d, "comparison", "all", "absolute_error_pct")),
    ),
    MetricSpec(
        "kernel_signed_pct", "kernel", "iteration",
        "iteration.comparison.all.signed_error_pct",
        lambda d: _scalar(_path(d, "comparison", "all", "signed_error_pct")),
    ),
)

_ITERATION += (
    MetricSpec(
        "map_coverage_pct", "kernel", "iteration",
        "iteration.mapping.coverage.measured_duration_fraction * 100",
        lambda d: (
            None
            if _scalar(_path(d, "mapping", "coverage", "measured_duration_fraction")) is None
            else _path(d, "mapping", "coverage", "measured_duration_fraction") * 100.0
        ),
        bound="min",
    ),
    MetricSpec(
        "gpu_time_multiplier", "duty_cycle", "iteration",
        "iteration.meta.recommended_gpu_time_multiplier",
        lambda d: _scalar(_path(d, "meta", "recommended_gpu_time_multiplier")),
        bound="max",
    ),
)

# ── workload report (scheduler structure + duty cycle landing) ───────────────

_WORKLOAD = (
    MetricSpec(
        "iteration_cycle_mean_pct", "duty_cycle", "workload",
        "ratio(workload.metrics.iteration_cycle_ms.{simulated,measured}.mean)",
        lambda d: _metric_mean_ratio(d, "metrics", "iteration_cycle_ms", "simulated", "measured"),
    ),
    MetricSpec(
        "iterations_pct", "workload", "workload",
        "ratio(workload.meta.{simulated,measured}_iterations)",
        lambda d: _ratio_pct(
            _path(d, "meta", "simulated_iterations"), _path(d, "meta", "measured_iterations")
        ),
    ),
    MetricSpec(
        "prefill_tokens_mean_pct", "workload", "workload",
        "ratio(workload.metrics.prefill_tokens.{simulated,measured}.mean)",
        lambda d: _metric_mean_ratio(d, "metrics", "prefill_tokens", "simulated", "measured"),
    ),
    MetricSpec(
        "decode_batch_mean_pct", "workload", "workload",
        "ratio(workload.metrics.decode_batch_size.{simulated,measured}.mean)",
        lambda d: _metric_mean_ratio(d, "metrics", "decode_batch_size", "simulated", "measured"),
    ),
    MetricSpec(
        "scheduled_kv_mean_pct", "workload", "workload",
        "ratio(workload.metrics.scheduled_kv_tokens.{simulated,measured}.mean)",
        lambda d: _metric_mean_ratio(d, "metrics", "scheduled_kv_tokens", "simulated", "measured"),
    ),
)

# ── e2e report (loosest) ─────────────────────────────────────────────────────

_E2E = (
    MetricSpec(
        "output_tps_pct", "e2e", "e2e",
        "ratio(e2e.throughput.simulated_completion_tps, "
        "e2e.throughput.measured_client_completion_tps)",
        lambda d: _ratio_pct(
            _path(d, "throughput", "simulated_completion_tps"),
            _path(d, "throughput", "measured_client_completion_tps"),
        ),
    ),
    MetricSpec(
        "server_ttft_mean_pct", "e2e", "e2e",
        "ratio(e2e.latency.server_ttft.{simulated,measured}_ms.mean)",
        lambda d: _metric_mean_ratio(d, "latency", "server_ttft", "simulated_ms", "measured_ms"),
    ),
    MetricSpec(
        "server_tpot_mean_pct", "e2e", "e2e",
        "ratio(e2e.latency.server_tpot.{simulated,measured}_ms.mean)",
        lambda d: _metric_mean_ratio(d, "latency", "server_tpot", "simulated_ms", "measured_ms"),
    ),
    MetricSpec(
        "e2e_mean_pct", "e2e", "e2e",
        "ratio(e2e.latency.e2e.{simulated,measured}_ms.mean)",
        lambda d: _metric_mean_ratio(d, "latency", "e2e", "simulated_ms", "measured_ms"),
    ),
)


def metric_specs(schemas: dict[str, int] | None = None) -> tuple[MetricSpec, ...]:
    """The metrics computable against these report schema versions.

    Defaults to the newest supported version of each report, which is what a
    fresh run produces and what a pack that declares no `analyzer_schema` gets.
    """
    for name, version in (schemas or {}).items():
        if name in SUPPORTED_SCHEMAS and version not in SUPPORTED_SCHEMAS[name]:
            raise ValueError(
                f"{name} report schema_version {version} is not supported; this engine "
                f"reads {list(SUPPORTED_SCHEMAS[name])}. A pack declaring an unsupported "
                "version would be judged by formulas its reports cannot produce."
            )
    specs = _ITERATION + _WORKLOAD + _E2E
    return tuple(sorted(specs, key=lambda spec: (GROUP_ORDER.index(spec.group), spec.name)))


def metric_names(schemas: dict[str, int] | None = None) -> tuple[str, ...]:
    return tuple(spec.name for spec in metric_specs(schemas))


def formula_table(schemas: dict[str, int] | None = None) -> dict[str, str]:
    return {spec.name: spec.formula for spec in metric_specs(schemas)}


# ── reading one case ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CaseMeasurement:
    """One case's numbers plus enough provenance to find them again."""

    slug: str
    directory: Path
    available: bool
    metrics: dict[str, float]
    schemas: dict[str, int]
    reports: dict[str, str]
    provenance: dict[str, Any]
    issues: tuple[str, ...]

    def as_json(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "directory": str(self.directory),
            "analyzer_schema": self.schemas,
            "reports": self.reports,
            "metrics": self.metrics,
            "provenance": self.provenance,
            "issues": list(self.issues),
        }


def _load_report(directory: Path, kind: str) -> tuple[dict[str, Any] | None, str | None]:
    path = directory / REPORT_LOCATIONS[kind]
    if not path.is_file():
        return None, None
    try:
        return json.loads(path.read_text()), str(path)
    except json.JSONDecodeError:
        return None, str(path)


def is_case_directory(directory: Path) -> bool:
    """A directory holds a case when at least one Analyzer report sits in it."""
    return any((directory / location).is_file() for location in REPORT_LOCATIONS.values())


def measure_case(directory: Path, *, expected_requests: int | None = None) -> CaseMeasurement:
    """Read one case run directory into metrics.

    Eligibility comes first: a number computed from a run whose two sides did
    not execute the same request population is not a timing result. Anything
    that fails lands in `issues` and clears `available`, so `compare --record`
    can refuse rather than baking a broken case into a baseline.
    """
    directory = Path(directory).resolve()
    issues: list[str] = []
    documents: dict[str, dict[str, Any]] = {}
    reports: dict[str, str] = {}
    schemas: dict[str, int] = {}

    for kind in REPORT_LOCATIONS:
        document, path = _load_report(directory, kind)
        if path is None:
            issues.append(f"{kind} report missing ({REPORT_LOCATIONS[kind]})")
            continue
        reports[kind] = path
        if document is None:
            issues.append(f"{kind} report is not valid JSON")
            continue
        version = document.get("schema_version")
        if version not in SUPPORTED_SCHEMAS[kind]:
            issues.append(
                f"{kind} report schema_version {version!r} is not one of "
                f"{list(SUPPORTED_SCHEMAS[kind])}"
            )
            continue
        if document.get("available") is False:
            issues.append(f"{kind} report declares available=false")
        schemas[kind] = version
        documents[kind] = document

    specs = metric_specs(schemas) if schemas else metric_specs()
    values: dict[str, float] = {}
    for spec in specs:
        document = documents.get(spec.report)
        if document is None:
            continue
        value = spec.read(document)
        if value is None:
            issues.append(f"{spec.name}: no value at `{spec.formula}`")
            continue
        values[spec.name] = value

    provenance = _provenance(documents)
    issues += _eligibility(documents, expected_requests)
    available = not issues and len(documents) == len(REPORT_LOCATIONS)
    return CaseMeasurement(
        slug=directory.name,
        directory=directory,
        available=available,
        metrics=values,
        schemas=schemas,
        reports=reports,
        provenance=provenance,
        issues=tuple(issues),
    )


def _provenance(documents: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The artifact directories each report says it read. Recorded so a golden
    can be traced back to the capture that produced it."""
    found: dict[str, Any] = {}
    for kind, document in documents.items():
        meta = document.get("meta", {})
        if not isinstance(meta, dict):
            continue
        for key, value in meta.items():
            if key.endswith("_log_dir") and isinstance(value, str):
                found.setdefault(key, value)
    for kind in ("iteration",):
        document = documents.get(kind)
        if document is not None:
            iterations = _path(document, "meta", "iterations")
            if iterations is not None:
                found["iteration_count"] = iterations
    return found


def _eligibility(
    documents: dict[str, dict[str, Any]], expected_requests: int | None
) -> list[str]:
    """The mechanizable half of `operate-run-alignment`'s completion checklist.

    Deliberately *not* checked: `request_id_audit.shared_ids`. Every accepted
    case in this pipeline reports 0 shared ids because the client and the
    simulator label requests in different id spaces; gating on it would fail all
    fifteen accepted runs. What is checkable is that the two sides ran the same
    number of requests, and that the number matches the trace when a pack says
    how many there should be.
    """
    issues: list[str] = []
    e2e = documents.get("e2e")
    if e2e is not None:
        meta = e2e.get("meta", {})
        measured = meta.get("measured_successful_requests")
        simulated = meta.get("simulated_completed_requests")
        if isinstance(measured, int) and isinstance(simulated, int):
            if measured != simulated:
                issues.append(
                    f"request populations differ: {measured} measured vs "
                    f"{simulated} simulated completions"
                )
            elif expected_requests is not None and measured != expected_requests:
                issues.append(
                    f"both sides completed {measured} requests but the pack's trace "
                    f"declares {expected_requests}"
                )
    iteration = documents.get("iteration")
    if iteration is not None:
        unmapped = _path(iteration, "mapping", "unmapped_measured_kernels")
        slots = _path(iteration, "mapping", "unmapped_simulated_slots")
        if unmapped is None or slots is None:
            issues.append(
                "iteration report is missing a coverage side "
                "(unmapped_measured_kernels / unmapped_simulated_slots)"
            )
    return issues
