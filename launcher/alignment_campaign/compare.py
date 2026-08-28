"""`alignment-campaign compare` — judge an extraction, two ways at once.

The two judgements are kept separate because they answer different questions and
carry different authority:

**Acceptance** asks "is this alignment good enough?" against the tolerances the
pack declares in `acceptance.yaml`. A breach is a FAIL and the process exits
non-zero, because that threshold was written down and reviewed in advance.
Statuses follow `analyzer/rust/src/conservation/workload.rs`: OK / WARN / FAIL.

**Drift** asks "did anything move since the accepted baseline?" against the
recorded golden. It only warns, matching `skills/dev-run-tests`' position that
goldens are monitors — a cost-model change that improves a number is still a
change worth seeing, but it is not a failure.

Tolerances are not stored in the golden. Re-recording measured values is routine;
changing the standard the values are judged against should be a reviewed edit to
a tracked file. Keeping them in one store would let `--record` quietly move the
goalposts along with the numbers.

Output is ordered tight -> loose (`skills/top-align-with-framework`), and
out-of-tolerance rows are listed separately at the end so a single bad case is
not hidden by an aggregate that looks fine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..golden import Golden
from .metrics import GROUP_ORDER, GROUP_TITLES, MetricSpec, metric_specs
from .pack import REPO_ROOT, Pack

#: Fraction of a tolerance a value may use before it is called WARN rather than
#: OK. One knob rather than a second per-metric threshold: nobody has the data to
#: set fifteen warn bands, and "close to the declared edge" is the actionable
#: signal. Overridable per pack via `acceptance.warn_fraction`.
DEFAULT_WARN_FRACTION = 0.8

#: A recorded value may move by this fraction of itself before drift is reported.
DEFAULT_DRIFT_RELATIVE = 0.20

#: Absolute floor on the drift band, in the metric's own units. Alignment errors
#: sit near zero, where a purely relative band would fire on noise: 0.10% moving
#: to 0.13% is a 30% relative change and means nothing.
DEFAULT_DRIFT_FLOOR = 0.5


@dataclass(frozen=True)
class Judgement:
    """One (case, metric) verdict."""

    case_key: str
    metric: str
    group: str
    value: float
    tolerance: float | None
    bound: str
    status: str  # OK | WARN | FAIL | UNJUDGED
    rationale: str  # why this case has a loosened tolerance, if it does
    golden: float | None
    drifted: bool

    @property
    def drift(self) -> float | None:
        return None if self.golden is None else self.value - self.golden


@dataclass(frozen=True)
class Comparison:
    pack_name: str | None
    judgements: tuple[Judgement, ...]
    unavailable: tuple[str, ...]
    missing_metrics: tuple[str, ...]

    @property
    def failures(self) -> tuple[Judgement, ...]:
        return tuple(item for item in self.judgements if item.status == "FAIL")

    @property
    def warnings(self) -> tuple[Judgement, ...]:
        return tuple(item for item in self.judgements if item.status == "WARN")

    @property
    def drifts(self) -> tuple[Judgement, ...]:
        return tuple(item for item in self.judgements if item.drifted)

    @property
    def accepted(self) -> bool:
        return not self.failures and not self.unavailable


def _tolerances(pack: Pack | None) -> tuple[dict[str, float], dict[str, dict[str, Any]], float]:
    if pack is None:
        return {}, {}, DEFAULT_WARN_FRACTION
    block = pack.acceptance.get("tolerances", {}) or {}
    default = dict(block.get("default", {}) or {})
    per_case = dict(block.get("per_case", {}) or {})
    warn_fraction = float(pack.acceptance.get("warn_fraction", DEFAULT_WARN_FRACTION))
    return default, per_case, warn_fraction


def _drift_band(pack: Pack | None) -> tuple[float, float]:
    if pack is None:
        return DEFAULT_DRIFT_RELATIVE, DEFAULT_DRIFT_FLOOR
    block = pack.acceptance.get("drift", {}) or {}
    return (
        float(block.get("relative", DEFAULT_DRIFT_RELATIVE)),
        float(block.get("floor", DEFAULT_DRIFT_FLOOR)),
    )


def golden_store(pack: Pack, variant_name: str, *, update: bool = False) -> Golden:
    """The golden file for one variant's device.

    Grouped by GPU rather than by pack because the store is already per-device;
    a pack that later spans two devices gets two files without a key migration.
    """
    variant = pack.variants[variant_name]
    return Golden(f"alignment_{pack.name}", variant.gpu, update)


def golden_key(case_key: str, metric: str) -> str:
    """`<variant>/<slug>@<metric>` — the variant prefix means a DP+EP topology
    is a new set of keys beside the old ones, never a migration of them."""
    return f"{case_key}@{metric}"


def compare(
    extraction: dict[str, Any], pack: Pack | None = None
) -> Comparison:
    """Judge an extraction document against the pack's policy and its golden."""
    default, per_case, warn_fraction = _tolerances(pack)
    drift_relative, drift_floor = _drift_band(pack)
    schemas = None
    if pack is not None:
        declared = pack.acceptance.get("analyzer_schema")
        if isinstance(declared, dict):
            schemas = {key: int(value) for key, value in declared.items()}
    specs: dict[str, MetricSpec] = {spec.name: spec for spec in metric_specs(schemas)}

    stores: dict[str, Golden] = {}
    judgements: list[Judgement] = []
    unavailable: list[str] = []
    missing: list[str] = []

    for case_key, body in sorted(extraction.get("cases", {}).items()):
        if not body.get("available"):
            unavailable.append(case_key)
        variant_name, _, slug = case_key.partition("/")
        overrides = per_case.get(slug, {}) or {}
        rationale = str(overrides.get("rationale", ""))

        store: Golden | None = None
        if pack is not None and variant_name in pack.variants:
            store = stores.get(variant_name)
            if store is None:
                store = golden_store(pack, variant_name)
                stores[variant_name] = store

        values = body.get("metrics", {}) or {}
        for name, spec in specs.items():
            if name not in values:
                missing.append(f"{case_key}@{name}")
                continue
            value = float(values[name])
            tolerance = overrides.get(name, default.get(name))
            tolerance = None if tolerance is None else float(tolerance)
            status = "UNJUDGED"
            if tolerance is not None:
                if spec.exceeds(value, tolerance):
                    status = "FAIL"
                elif spec.margin(value, tolerance) > warn_fraction:
                    status = "WARN"
                else:
                    status = "OK"
            recorded = None if store is None else store.get(golden_key(case_key, name))
            drifted = recorded is not None and abs(value - recorded) > max(
                drift_relative * abs(recorded), drift_floor
            )
            judgements.append(
                Judgement(
                    case_key=case_key,
                    metric=name,
                    group=spec.group,
                    value=value,
                    tolerance=tolerance,
                    bound=spec.bound,
                    status=status,
                    rationale=rationale if name in overrides else "",
                    golden=recorded,
                    drifted=drifted,
                )
            )
    return Comparison(
        pack_name=None if pack is None else pack.name,
        judgements=tuple(judgements),
        unavailable=tuple(unavailable),
        missing_metrics=tuple(missing),
    )


# ── recording ────────────────────────────────────────────────────────────────

class RecordRefused(RuntimeError):
    """`--record` would bake a number that should not become a baseline."""


def record(
    extraction: dict[str, Any],
    pack: Pack,
    *,
    accept_provisional: bool = False,
) -> list[Path]:
    """Write the accepted numbers into the golden store.

    Refuses when any case is unavailable, when a report schema disagrees with
    what the pack declares, or when a calibrated input is still provisional. The
    provisional refusal is overridable: GLM case 09's rate knee is provisional in
    the accepted run, so a blanket refusal would make the real matrix
    unrecordable. Overriding is explicit and the field names are written into the
    provenance, so the baseline says what was uncalibrated when it was taken.
    """
    unavailable = sorted(
        key for key, body in extraction.get("cases", {}).items() if not body.get("available")
    )
    if unavailable:
        raise RecordRefused(
            "refusing to record: these cases are not complete — "
            + ", ".join(unavailable)
        )

    declared = pack.acceptance.get("analyzer_schema")
    if isinstance(declared, dict):
        for case_key, body in extraction.get("cases", {}).items():
            actual = body.get("analyzer_schema", {})
            mismatched = {
                name: (actual.get(name), int(version))
                for name, version in declared.items()
                if actual.get(name) != int(version)
            }
            if mismatched:
                raise RecordRefused(
                    f"refusing to record: {case_key} reports Analyzer schema "
                    f"{ {k: v[0] for k, v in mismatched.items()} } but acceptance.yaml "
                    f"declares { {k: v[1] for k, v in mismatched.items()} }; a golden "
                    "recorded across a schema change compares different statistics"
                )

    provisional = pack.provisional_fields
    if provisional and not accept_provisional:
        raise RecordRefused(
            "refusing to record: these calibrated inputs are still provisional — "
            + ", ".join(provisional)
            + ". Run the preflight calibration, or pass --accept-provisional to "
            "record anyway (the field names are stored in the baseline's provenance)."
        )

    stores: dict[str, Golden] = {}
    for case_key, body in sorted(extraction.get("cases", {}).items()):
        variant_name, _, _ = case_key.partition("/")
        if variant_name not in pack.variants:
            continue
        store = stores.get(variant_name)
        if store is None:
            store = golden_store(pack, variant_name, update=True)
            stores[variant_name] = store
        for metric, value in (body.get("metrics", {}) or {}).items():
            store.data[golden_key(case_key, metric)] = float(value)

    written: list[Path] = []
    for store in stores.values():
        store.flush()
        written.append(store.path)
        written.append(_write_provenance(store.path, extraction, pack, provisional))
    return written


def _portable(value: str) -> str:
    """A run path as provenance rather than as a location on one machine.

    What a later reader needs from these is *which experiment* produced a
    number -- the dated `logs/<experiment>/cases/<case>/...` tail, which is also
    what `skills/operate-run-alignment` asks a report to name. The machine
    prefix in front of it carries no information and is a liability: it is
    somebody's home directory, it names a worktree that will be deleted, and it
    is committed. This file is the only thing the engine writes into the tree
    from a real run, so this is the one place that has to strip it.

    Under the repo, the repo-relative path says everything. Outside it, keep the
    tail from the run root's own name so the experiment is still identifiable
    and no absolute prefix survives.
    """
    path = Path(value)
    if not path.is_absolute():
        return value
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        parts = path.parts
        return str(Path(".../", *parts[-5:])) if len(parts) > 5 else str(path.name)


def _write_provenance(
    golden_path: Path, extraction: dict[str, Any], pack: Pack, provisional: list[str]
) -> Path:
    """The story behind the numbers, beside them.

    The golden file itself is a flat `{key: float}` map shared with the pytest
    tier, so it cannot hold the derivation. This sidecar carries what a later
    reader needs to judge whether a drift is meaningful: which formula produced
    each key, which Analyzer schema it was read from, which run directories it
    came from, and which calibrated inputs were still provisional when it was
    taken.

    Every path is written through `_portable`: this is committed, and a recorded
    baseline must not name the machine it happened to be recorded on.
    """
    path = golden_path.with_suffix(".provenance.json")
    document = {
        "recorded_from": [_portable(root) for root in extraction.get("roots", [])],
        "pack": pack.name,
        "analyzer_schema": {
            case_key: body.get("analyzer_schema", {})
            for case_key, body in sorted(extraction.get("cases", {}).items())
        },
        "case_reports": {
            case_key: {kind: _portable(value) for kind, value in body.get("reports", {}).items()}
            for case_key, body in sorted(extraction.get("cases", {}).items())
        },
        "formulas": extraction.get("formulas", {}),
        "provisional_inputs": provisional,
    }
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return path


# ── rendering ────────────────────────────────────────────────────────────────

def _format(value: float | None, spec_bound: str) -> str:
    """Signed for an error, unsigned for a level.

    A leading `+` says "the simulator is higher than measured". Mapping coverage
    and the duty-cycle multiplier are levels, not differences, so signing them
    would read as a 98% error rather than 98% covered.
    """
    if value is None:
        return "—"
    if spec_bound == "max":  # a bare multiplier, not a percentage
        return f"{value:.4f}"
    if spec_bound == "min":  # a level: how much is covered, not how far off
        return f"{value:.2f}%"
    return f"{value:+.2f}%" if abs(value) < 1000 else f"{value:+.1f}%"


def render_text(comparison: Comparison) -> str:
    """Human-readable report, grouped tight -> loose."""
    lines: list[str] = []
    scope = comparison.pack_name or "(no pack — engine defaults, no golden)"
    lines.append(f"alignment-campaign compare — {scope}")
    if comparison.unavailable:
        lines.append("")
        lines.append("incomplete cases (excluded from acceptance):")
        for key in comparison.unavailable:
            lines.append(f"  {key}")

    by_group: dict[str, list[Judgement]] = {group: [] for group in GROUP_ORDER}
    for item in comparison.judgements:
        by_group.setdefault(item.group, []).append(item)

    for group in GROUP_ORDER:
        rows = by_group.get(group) or []
        if not rows:
            continue
        lines.append("")
        lines.append(GROUP_TITLES[group])
        width = max(len(row.case_key) for row in rows)
        for row in sorted(rows, key=lambda item: (item.case_key, item.metric)):
            drift = "" if not row.drifted else f"  drift {row.value - row.golden:+.2f} vs golden"
            tolerance = "" if row.tolerance is None else f" (tol {row.tolerance:g})"
            lines.append(
                f"  {row.case_key:<{width}}  {row.metric:<24} "
                f"{_format(row.value, row.bound):>10}  {row.status}{tolerance}{drift}"
            )

    if comparison.failures:
        lines.append("")
        lines.append("out of tolerance:")
        for row in comparison.failures:
            note = f" — {row.rationale}" if row.rationale else ""
            lines.append(
                f"  {row.case_key} {row.metric} = {_format(row.value, row.bound)} "
                f"exceeds {row.tolerance:g}{note}"
            )
    if comparison.missing_metrics:
        lines.append("")
        lines.append(f"metrics with no value: {len(comparison.missing_metrics)}")
        for key in comparison.missing_metrics[:10]:
            lines.append(f"  {key}")
    lines.append("")
    lines.append(
        f"{len(comparison.judgements)} judgements — "
        f"{len(comparison.failures)} FAIL, {len(comparison.warnings)} WARN, "
        f"{len(comparison.drifts)} drifted"
    )
    return "\n".join(lines) + "\n"


def render_markdown(comparison: Comparison, specs: tuple[MetricSpec, ...]) -> str:
    """The tracked matrix table, replacing the hand-transcribed `RESULTS.md`."""
    order = [spec for spec in specs if spec.group in ("kernel", "e2e")]
    cases = sorted({item.case_key for item in comparison.judgements})
    values = {(item.case_key, item.metric): item for item in comparison.judgements}

    lines = [
        f"# {comparison.pack_name or 'alignment'} — case matrix",
        "",
        "Generated by `python -m launcher alignment-campaign compare --markdown`.",
        "Every number is read from a completed Analyzer report by the formula table in",
        "`launcher/alignment_campaign/metrics.py`; none is transcribed by hand.",
        "",
        "| Case | " + " | ".join(spec.name for spec in order) + " |",
        "|---|" + "---:|" * len(order),
    ]
    for case_key in cases:
        cells = []
        for spec in order:
            item = values.get((case_key, spec.name))
            cells.append("—" if item is None else _format(item.value, item.bound))
        lines.append(f"| {case_key} | " + " | ".join(cells) + " |")

    exceptions = [item for item in comparison.judgements if item.rationale]
    if exceptions:
        lines += ["", "## Declared exceptions", ""]
        seen: set[tuple[str, str]] = set()
        for item in sorted(exceptions, key=lambda row: (row.case_key, row.metric)):
            if (item.case_key, item.metric) in seen:
                continue
            seen.add((item.case_key, item.metric))
            lines.append(
                f"- `{item.case_key}` **{item.metric}** "
                f"{_format(item.value, item.bound)} (tolerance {item.tolerance:g}): "
                f"{item.rationale}"
            )
    if comparison.failures:
        lines += ["", "## Out of tolerance", ""]
        for item in comparison.failures:
            lines.append(
                f"- `{item.case_key}` **{item.metric}** "
                f"{_format(item.value, item.bound)} exceeds {item.tolerance:g}"
            )
    return "\n".join(lines) + "\n"


def as_json(comparison: Comparison) -> dict[str, Any]:
    return {
        "pack": comparison.pack_name,
        "accepted": comparison.accepted,
        "unavailable": list(comparison.unavailable),
        "judgements": [
            {
                "case": item.case_key,
                "metric": item.metric,
                "group": item.group,
                "value": item.value,
                "tolerance": item.tolerance,
                "status": item.status,
                "rationale": item.rationale,
                "golden": item.golden,
                "drifted": item.drifted,
            }
            for item in comparison.judgements
        ],
    }
