"""Extract the stock vLLM Neuron frontend hand-off and planning distributions.

The simulator's opt-in frontend hand-off model and the async-lookahead worker
offset are calibrated from stock vLLM Neuron 0.24 burst captures (16
simultaneous requests, async scheduling on). Each capture directory holds the
alignment profiler's ``request_timings.jsonl`` (frontend arrival and EngineCore
receive per request, one monotonic clock) and ``metrics.jsonl`` (one record per
scheduler step, ``observed_start_monotonic_ns`` = the instant the step was
planned).

Quantities written, all in milliseconds:

``ingest_delay_ms``
    Per capture: first frontend arrival to the first EngineCore receive. The
    engine is idle, so that receive is the instant the first request became
    visible to the scheduler.
``handoff_gap_ms``
    Consecutive API-server send gaps: the ``api_send_monotonic_ns`` stamps
    (arrival timing sub-version 2, from vLLM's public request log) sorted per
    capture. A capture without a stamp on every request falls back to its
    frontend inter-arrival gaps (arrivals sorted). ``provenance.handoff_gap_source``
    records which one each capture used. The simulator serializes simultaneous
    arrivals by these gaps.
``first_plan_offset_ms``
    Per capture: step 1 planned minus step 0 planned, i.e. how long after the
    first step after an idle engine starts the next step is planned.
``steady_plan_offset_ms``
    Step N+1 planned minus step N-1 finished (N >= 2), recorded for reference.
``extra_solo_decode``
    Per capture: whether step 1 was a decode of request 0 (the race outcome).

Usage::

    uv run --no-sync python -m alignment.neuron.vllm_handoff_distribution \
        "$TMPDIR/neuron-context-ladder/race" \
        --out alignment/neuron/data/vllm_neuron_handoff.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

SCHEMA_VERSION = 1
_NS_PER_MS = 1e6


def _load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _capture_sort_key(path: Path) -> tuple[int, int]:
    context, _, index = path.name.removeprefix("c").partition("-")
    return int(context), int(index)


def capture_dirs(root: Path) -> tuple[list[Path], dict[str, str]]:
    """Capture directories with both record files, plus skipped names and why."""
    included: list[Path] = []
    skipped: dict[str, str] = {}
    for path in sorted((p for p in root.iterdir() if p.is_dir()), key=_capture_sort_key):
        missing = [
            name
            for name in ("request_timings.jsonl", "metrics.jsonl")
            if not (path / name).is_file()
        ]
        if missing:
            skipped[path.name] = "missing " + ", ".join(missing)
        else:
            included.append(path)
    return included, skipped


def extract_capture(path: Path) -> dict:
    timings = _load_jsonl(path / "request_timings.jsonl")
    steps = _load_jsonl(path / "metrics.jsonl")
    arrivals = sorted(row["frontend_arrival_monotonic_ns"] for row in timings)
    receives = sorted(row["engine_receive_monotonic_ns"] for row in timings)
    sends = [row.get("api_send_monotonic_ns") for row in timings]
    if all(send is not None for send in sends):
        gap_source, stamps = "api_send", sorted(sends)
    else:
        gap_source, stamps = "frontend_arrival", arrivals
    plans = [step["observed_start_monotonic_ns"] for step in steps]
    ends = [step["observed_end_monotonic_ns"] for step in steps]
    return {
        "capture": path.name,
        "steps": len(steps),
        "extra_solo_decode": len(steps) > 1 and steps[1]["phase"] == "decode",
        "ingest_delay_ms": (receives[0] - arrivals[0]) / _NS_PER_MS,
        "handoff_gap_source": gap_source,
        "handoff_gap_ms": [(b - a) / _NS_PER_MS for a, b in zip(stamps, stamps[1:])],
        "first_plan_offset_ms": (plans[1] - plans[0]) / _NS_PER_MS,
        "steady_plan_offset_ms": [
            (plans[n + 1] - ends[n - 1]) / _NS_PER_MS for n in range(2, len(steps) - 1)
        ],
    }


def _rounded(values: list[float]) -> list[float]:
    return [round(value, 6) for value in values]


def build_distribution(root: Path) -> dict:
    captures, skipped = capture_dirs(root)
    if not captures:
        raise SystemExit(f"no capture directories with records under {root}")
    rows = [extract_capture(path) for path in captures]
    gaps = sorted(gap for row in rows for gap in row["handoff_gap_ms"])
    first_offsets = [row["first_plan_offset_ms"] for row in rows]
    steady = sorted(value for row in rows for value in row["steady_plan_offset_ms"])
    return {
        "schema_version": SCHEMA_VERSION,
        "engine": "stock vLLM Neuron 0.24, async scheduling on, Llama 3.1 8B TP4 LNC2",
        "workload": "16 simultaneous (context-8)+8 requests per capture, C512-C8192",
        "provenance": {
            "source_root": "tmp/neuron-context-ladder/race (workspace-local)",
            "extraction": "alignment/neuron/vllm_handoff_distribution.py",
            "captures": [row["capture"] for row in rows],
            "skipped": skipped,
            "handoff_gap_source": {row["capture"]: row["handoff_gap_source"] for row in rows},
        },
        "ingest_delay_ms": _rounded([row["ingest_delay_ms"] for row in rows]),
        "handoff_gap_ms": _rounded(gaps),
        "first_plan_offset_ms": _rounded(first_offsets),
        "first_plan_offset_median_ms": round(statistics.median(first_offsets), 6),
        "steady_plan_offset_median_ms": round(statistics.median(steady), 6),
        "outcome": {
            "captures": len(rows),
            "extra_solo_decode": sum(row["extra_solo_decode"] for row in rows),
            "steps": {row["capture"]: row["steps"] for row in rows},
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("race_root", type=Path, help="directory of c<C>-<i>/ captures")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    distribution = build_distribution(args.race_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(distribution, indent=1) + "\n")


if __name__ == "__main__":
    main()
