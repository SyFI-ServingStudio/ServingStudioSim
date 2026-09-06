"""Generate an observed-conditioned trace without replaying scheduler decisions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _rates(value: Any, *, allow_missing: bool) -> list[float | None]:
    if not isinstance(value, list) or not value:
        raise ValueError("acceptance rates must be a nonempty list")
    for rate in value:
        if allow_missing and rate is None:
            continue
        if (
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not math.isfinite(rate)
            or not 0 <= rate <= 1
        ):
            raise ValueError("acceptance rates must be finite probabilities")
    return value


def build_trace(*, source_trace: Path, comparison_json: Path, output_trace: Path) -> dict[str, Any]:
    comparison = json.loads(comparison_json.read_text())
    aggregate = _rates(
        comparison["aggregate"]["measured_acceptance_effective"]["conditional_acceptance_rates"],
        allow_missing=True,
    )
    requests = {}
    for request in comparison["requests_by_trace_order"]:
        request_id = request["request_id"]
        if request_id in requests:
            raise ValueError(f"duplicate comparison request: {request_id!r}")
        requests[request_id] = request
    with source_trace.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        required = {"id", "input_len", "output_len", "arrival_time", "accept_rate"}
        if fields is None or not required.issubset(fields):
            raise ValueError(f"source trace requires columns {sorted(required)}")
        rows = list(reader)
    ids = [row["id"] for row in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("source trace must contain distinct request IDs")
    populations = [
        prefix
        for prefix in ("", "independent_")
        if {prefix + request_id for request_id in ids} == set(requests)
    ]
    if len(populations) != 1:
        raise ValueError("trace and comparison request populations must match exactly")
    prefix = populations[0]
    fallback_positions = []
    for row in rows:
        request_id = prefix + row["id"]
        measured = _rates(
            requests[request_id]["measured_acceptance"]["conditional_acceptance_rates"],
            allow_missing=True,
        )
        if len(measured) != len(aggregate):
            raise ValueError(f"acceptance depth mismatch for {request_id!r}")
        resolved = []
        for position, rate in enumerate(measured):
            if rate is None:
                rate = aggregate[position]
                fallback_positions.append({"request_id": request_id, "position": position})
            resolved.append(rate)
        _rates(resolved, allow_missing=False)
        row["accept_rate"] = json.dumps(resolved, separators=(",", ":"))

    manifest = {
        "schema_version": 1,
        "kind": "observed_conditioned_per_request_acceptance",
        "predictive_alignment": False,
        "source_trace": str(source_trace),
        "source_comparison": str(comparison_json),
        "output_trace": str(output_trace),
        "requests": len(rows),
        "aggregate_fallback_rates": aggregate,
        "fallback_positions": fallback_positions,
        "preserved_fields": [field for field in fields if field != "accept_rate"],
        "conditioned_field": "accept_rate",
        "excluded_observed_decisions": [
            "scheduler placement",
            "batch membership",
            "prefill chunking",
            "per-round accept/reject outcomes",
        ],
    }
    manifest_path = output_trace.with_suffix(output_trace.suffix + ".manifest.json")
    if output_trace.exists() or manifest_path.exists():
        raise FileExistsError("output trace or its manifest already exists")
    output_trace.parent.mkdir(parents=True, exist_ok=True)
    with output_trace.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with manifest_path.open("x") as handle:
        handle.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-trace", type=Path, required=True)
    parser.add_argument("--comparison-json", type=Path, required=True)
    parser.add_argument("--output-trace", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_trace(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
