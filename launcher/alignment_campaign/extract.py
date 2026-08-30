"""`alignment-campaign extract` — completed reports to one metrics document.

This replaces transcribing Analyzer JSON into a markdown table by hand. Every
number comes from `metrics.py`'s formula table; none is typed in. The output is
the only input `compare` accepts, so there is no path by which a hand-edited
number reaches an acceptance decision or a golden.

`--pack` is optional. Without it, any set of run directories is readable with
the engine's default formula set — which is what makes the tool useful during a
one-off alignment, before a pack exists. With it, each case is additionally
labeled with its variant (so golden keys are stable across topologies) and its
completed request count is checked against the trace the pack declares.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .metrics import (
    METRICS_SCHEMA_VERSION,
    CaseMeasurement,
    formula_table,
    is_case_directory,
    measure_case,
)
from .pack import Pack

#: Golden and comparison key: `<variant>/<slug>@<metric>`. Without a pack there
#: is no variant, so the engine uses this placeholder — a pack-less comparison is
#: never recorded, so the key never reaches the store.
UNSCOPED_VARIANT = "-"


@dataclass(frozen=True)
class Extraction:
    pack_name: str | None
    schemas: dict[str, int]
    cases: dict[str, CaseMeasurement]  # key: "<variant>/<slug>"
    roots: tuple[Path, ...]

    def as_json(self) -> dict[str, Any]:
        return {
            "schema_version": METRICS_SCHEMA_VERSION,
            "generated_by": "python -m launcher alignment-campaign extract",
            "pack": self.pack_name,
            "roots": [str(root) for root in self.roots],
            "formulas": formula_table(self.schemas or None),
            "cases": {key: value.as_json() for key, value in sorted(self.cases.items())},
        }

    @property
    def unavailable(self) -> list[str]:
        return sorted(key for key, value in self.cases.items() if not value.available)


def discover_case_dirs(root: Path) -> list[Path]:
    """Case run directories under one root.

    A root may itself be one case (a one-off alignment) or a directory of them
    (a campaign). Sorted so the output order is the case order, not the order the
    filesystem happens to return.
    """
    root = Path(root).resolve()
    if is_case_directory(root):
        return [root]
    if not root.is_dir():
        return []
    return sorted(child for child in root.iterdir() if child.is_dir() and is_case_directory(child))


def extract(roots: list[Path], pack: Pack | None = None) -> Extraction:
    """Read every case under `roots` into one document."""
    by_slug: dict[str, Path] = {}
    resolved_roots: list[Path] = []
    for root in roots:
        resolved_roots.append(Path(root).resolve())
        for directory in discover_case_dirs(root):
            # A later root wins: the usual layout splits one matrix across two
            # dated campaigns, and re-running a case moves it to the newer one.
            by_slug[directory.name] = directory

    schemas: dict[str, int] = {}
    if pack is not None:
        declared = pack.acceptance.get("analyzer_schema")
        if isinstance(declared, dict):
            schemas = {key: int(value) for key, value in declared.items()}

    cases: dict[str, CaseMeasurement] = {}
    for slug, directory in sorted(by_slug.items()):
        expected: int | None = None
        variant = UNSCOPED_VARIANT
        if pack is not None:
            case = pack.case_named(slug)
            if case is None:
                continue  # a directory the pack does not declare is not this matrix
            variant = case.variant
            expected = case.workload_trace.request_count
        measurement = measure_case(directory, expected_requests=expected)
        cases[f"{variant}/{slug}"] = measurement

    if pack is not None:
        for case in pack.cases:
            key = f"{case.variant}/{case.slug}"
            if key not in cases:
                cases[key] = CaseMeasurement(
                    slug=case.slug,
                    directory=Path(""),
                    available=False,
                    metrics={},
                    schemas={},
                    reports={},
                    provenance={},
                    issues=("no run directory found under the given roots",),
                )
    return Extraction(
        pack_name=None if pack is None else pack.name,
        schemas=schemas,
        cases=cases,
        roots=tuple(resolved_roots),
    )


def write_extraction(extraction: Extraction, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(extraction.as_json(), indent=2, sort_keys=True) + "\n")
    return path


def load_extraction(path: Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    version = document.get("schema_version")
    if version != METRICS_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: metrics schema_version must be {METRICS_SCHEMA_VERSION}, got {version!r}"
        )
    return document
