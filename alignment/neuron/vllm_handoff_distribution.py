"""Extract the stock vLLM Neuron frontend hand-off and planning distributions.

The simulator's opt-in frontend hand-off model and the async-lookahead first
plan offset are calibrated from stock vLLM Neuron 0.24 burst captures (16
simultaneous (context-8)+8 requests, async scheduling on). Each capture
directory ``c<context>-<i>/`` holds the alignment profiler's
``request_timings.jsonl`` (per request, one monotonic clock: frontend arrival,
API-server send ``api_send_monotonic_ns`` when the capture ran with
``server.handoff_stamp: true``, and EngineCore receive) and ``metrics.jsonl``
(one record per scheduler step; ``observed_start_monotonic_ns`` is the instant
the step was planned).

The simulator composes, for a burst into an idle server (request ``k`` in send
order)::

    send_0    = arrival + ingest
    send_k    = send_{k-1} + gap_k        (gap_1: first gap; k >= 2: later gap)
    visible_k = max(visible_{k-1}, send_k + send_to_visible_k)
    step 1 planned at visible_0 + first_plan_offset

Every input below is a measured duration, never fitted to the outcome. Only
stamped captures (every request has a send stamp) supply model samples.
Unstamped captures contribute only their race outcome, reported for comparison.
Quantities, all in milliseconds, per stamped capture:

``ingest_ms``
    First frontend arrival to the first API-server send.
``first_gap_ms``
    Second send minus first send.
``later_gap_ms``
    Consecutive send gaps from the third request on.
``send_to_visible_ms``
    First send to its EngineCore receive. The engine is idle and blocks on its
    input queue, so that receive is the instant the request became visible.
    A later request is received only when the busy loop next drains its queue,
    so its receive bounds its visibility from above and is not a sample;
    receives at a drain before step 1 is planned are recorded as
    ``later_send_to_receive_upper_bound_ms``.
``first_plan_offset_ms``
    Step 1 planned minus the first request's EngineCore receive: the simulator
    starts step 0 the instant request 0 becomes visible to the idle engine.

Context keying: a quantity is keyed by context when some pair of contexts
clearly differs, otherwise it is pooled across contexts (``keying`` records the
decision and the per-context summaries it rests on). Per-capture quantities
(one sample per capture) differ clearly when two contexts' sample ranges are
disjoint; per-gap quantities (many correlated samples per capture) when two
contexts' interquartile ranges are disjoint. Every context entry lists every
quantity; a pooled quantity repeats the pooled samples.

Usage::

    uv run --no-sync python -m alignment.neuron.vllm_handoff_distribution \
        "$TMPDIR/neuron-context-ladder/race2" "$TMPDIR/neuron-context-ladder/race" \
        --out alignment/neuron/data/vllm_neuron_handoff.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from itertools import combinations
from pathlib import Path

SCHEMA_VERSION = 2
_NS_PER_MS = 1e6

# Per-capture quantities take one sample per capture; per-gap quantities many.
PER_CAPTURE = ("ingest_ms", "first_gap_ms", "send_to_visible_ms", "first_plan_offset_ms")
PER_GAP = ("later_gap_ms",)
QUANTITIES = PER_CAPTURE + PER_GAP


def _load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _capture_key(path: Path) -> tuple[int, int]:
    context, _, index = path.name.removeprefix("c").partition("-")
    return int(context), int(index)


def capture_dirs(root: Path) -> tuple[list[Path], dict[str, str]]:
    """Capture directories with both record files, plus skipped names and why."""
    included: list[Path] = []
    skipped: dict[str, str] = {}
    for path in sorted((p for p in root.iterdir() if p.is_dir()), key=_capture_key):
        missing = [
            name
            for name in ("request_timings.jsonl", "metrics.jsonl")
            if not (path / name).is_file()
        ]
        if missing:
            skipped[f"{root.name}/{path.name}"] = "missing " + ", ".join(missing)
        else:
            included.append(path)
    return included, skipped


def _ms(delta_ns: int) -> float:
    return delta_ns / _NS_PER_MS


def capture_outcome(steps: list[dict]) -> dict:
    return {
        "steps": len(steps),
        "extra_solo_decode": len(steps) > 1 and steps[1]["phase"] == "decode",
    }


def extract_capture(path: Path) -> dict:
    """Model samples of one stamped capture, or only its outcome if unstamped."""
    timings = _load_jsonl(path / "request_timings.jsonl")
    steps = _load_jsonl(path / "metrics.jsonl")
    row = {"capture": path.name, "context": _capture_key(path)[0], **capture_outcome(steps)}
    if not all(timing.get("api_send_monotonic_ns") is not None for timing in timings):
        return {**row, "stamped": False}
    by_send = sorted(timings, key=lambda timing: timing["api_send_monotonic_ns"])
    sends = [timing["api_send_monotonic_ns"] for timing in by_send]
    first = by_send[0]
    first_arrival = min(timing["frontend_arrival_monotonic_ns"] for timing in timings)
    first_receive = first["engine_receive_monotonic_ns"]
    plans = [step["observed_start_monotonic_ns"] for step in steps]
    if min(timing["engine_receive_monotonic_ns"] for timing in timings) != first_receive:
        raise ValueError(f"{path}: the first request sent is not the first received")
    if not first_receive <= plans[0]:
        raise ValueError(f"{path}: step 0 is planned before the first request is received")
    gaps = [_ms(b - a) for a, b in zip(sends, sends[1:])]
    return {
        **row,
        "stamped": True,
        "ingest_ms": _ms(sends[0] - first_arrival),
        "first_gap_ms": gaps[0],
        "later_gap_ms": gaps[1:],
        "send_to_visible_ms": _ms(first_receive - sends[0]),
        "first_plan_offset_ms": _ms(plans[1] - first_receive),
        "later_send_to_receive_upper_bound_ms": [
            _ms(timing["engine_receive_monotonic_ns"] - timing["api_send_monotonic_ns"])
            for timing in by_send[1:]
            if timing["engine_receive_monotonic_ns"] <= plans[1]
        ],
    }


def _samples(rows: list[dict], quantity: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row[quantity]
        values.extend(value if isinstance(value, list) else [value])
    return values


def _summary(values: list[float]) -> dict:
    ordered = sorted(values)
    quartiles = statistics.quantiles(ordered, n=4) if len(ordered) > 1 else ordered * 3
    return {
        "n": len(ordered),
        "min": round(ordered[0], 6),
        "q1": round(quartiles[0], 6),
        "median": round(statistics.median(ordered), 6),
        "q3": round(quartiles[2], 6),
        "max": round(ordered[-1], 6),
    }


def keying_decision(by_context: dict[int, list[dict]]) -> dict:
    """Per quantity: ``context`` when some pair of contexts clearly differs."""
    decision = {}
    for quantity in QUANTITIES:
        summaries = {
            context: _summary(_samples(rows, quantity)) for context, rows in by_context.items()
        }
        low, high = ("min", "max") if quantity in PER_CAPTURE else ("q1", "q3")
        separated = [
            [a, b]
            for a, b in combinations(sorted(summaries), 2)
            if summaries[a][high] < summaries[b][low] or summaries[b][high] < summaries[a][low]
        ]
        decision[quantity] = {
            "key": "context" if separated else "pooled",
            "test": "disjoint ranges"
            if quantity in PER_CAPTURE
            else "disjoint interquartile ranges",
            "separated_contexts": separated,
            "by_context": {str(context): summary for context, summary in summaries.items()},
        }
    return decision


def _rounded(values: list[float]) -> list[float]:
    return [round(value, 6) for value in values]


def build_distribution(roots: list[Path]) -> dict:
    rows: list[dict] = []
    skipped: dict[str, str] = {}
    for root in roots:
        captures, root_skipped = capture_dirs(root)
        skipped.update(root_skipped)
        rows.extend({**extract_capture(path), "root": root.name} for path in captures)
    stamped = [row for row in rows if row["stamped"]]
    if not stamped:
        raise SystemExit(f"no stamped capture with records under {[str(r) for r in roots]}")
    by_context: dict[int, list[dict]] = {}
    for row in stamped:
        by_context.setdefault(row["context"], []).append(row)
    keying = keying_decision(by_context)

    contexts = {}
    for context, context_rows in sorted(by_context.items()):
        entry = {"captures": [f"{row['root']}/{row['capture']}" for row in context_rows]}
        for quantity in QUANTITIES:
            source = context_rows if keying[quantity]["key"] == "context" else stamped
            values = _samples(source, quantity)
            entry[quantity] = _rounded(sorted(values) if quantity in PER_GAP else values)
        entry["later_send_to_receive_upper_bound_ms"] = _rounded(
            _samples(context_rows, "later_send_to_receive_upper_bound_ms")
        )
        entry["outcome"] = {
            "captures": len(context_rows),
            "extra_solo_decode": sum(row["extra_solo_decode"] for row in context_rows),
        }
        contexts[str(context)] = entry

    outcome: dict[str, dict] = {}
    for row in rows:
        group = outcome.setdefault(str(row["context"]), {"stamped": {}, "unstamped": {}})
        group["stamped" if row["stamped"] else "unstamped"][f"{row['root']}/{row['capture']}"] = (
            row["steps"]
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "engine": "stock vLLM Neuron 0.24, async scheduling on, Llama 3.1 8B TP4 LNC2",
        "workload": "16 simultaneous (context-8)+8 requests per capture",
        "composition": (
            "send_0 = arrival + ingest; send_k = send_{k-1} + gap_k (k=1 first gap, "
            "k>=2 later gap); visible_k = max(visible_{k-1}, send_k + send_to_visible_k); "
            "step 1 planned first_plan_offset after visible_0"
        ),
        "keying": keying,
        "contexts": contexts,
        "outcome": {
            "note": "steps per capture; 24 = extra solo decode of request 0, 23 = none",
            "by_context": outcome,
        },
        "provenance": {
            "sources": [
                f"tmp/neuron-context-ladder/{root.name} (workspace-local)" for root in roots
            ],
            "extraction": "alignment/neuron/vllm_handoff_distribution.py",
            "model_captures": [f"{row['root']}/{row['capture']}" for row in stamped],
            "outcome_only_captures": [
                f"{row['root']}/{row['capture']}" for row in rows if not row["stamped"]
            ],
            "skipped": skipped,
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("roots", type=Path, nargs="+", help="directories of c<C>-<i>/ captures")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    distribution = build_distribution(args.roots)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(distribution, indent=1) + "\n")


if __name__ == "__main__":
    main()
