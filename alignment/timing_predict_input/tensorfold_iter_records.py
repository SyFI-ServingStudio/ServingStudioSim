"""Convert TensorFold ``TENSORFOLD_ITER_RECORDS`` JSONL into speculative predictor cases.

TensorFold (``tensorfold.cuda.nvtx.record``) appends one JSON line per engine step,
with monotonic ``start_ns`` / ``end_ns`` around it. Two engines write them:

- ``MultiDecoder.round`` (``families/qwen3_5/cuda/multi.py``) writes ``multi_round``:
  one prefill step, then one verify over every decoding stream.

  - ``prefill``: ``[[sid, start, stop], ...]`` or null.
  - ``streams``: ``[{sid, pos, rows, parents, mode, accepted}]`` or null. ``pos`` is
    the committed length before the round; ``mode`` is ``COPY, TREE, ONE = 0, 1, 2``.

  Its prefill step also writes nested ``prefill_chunk`` lines, which are sub-steps
  of that round and do not become cases of their own.
- The single-stream decoder (``decode.py`` / ``prefill.py``) writes ``round``
  (``pos``, ``rows``, ``parents``, ``source`` in copy/tree/serial, ``accepted``) and
  ``prefill_chunk`` (``pos``, ``rows``, ``last``), one forward each.

Each kept record becomes one speculative timing-predict case with one group:

- ``prefill_chunk_pairs``: ``[[start, stop - start]]`` per prefill piece;
- ``decode_requests``: ``[[pos + rows, rows]]`` per stream (``kv_len`` counts every
  verified row, the root included);
- ``decode_outcomes``: one entry per ``decode_requests`` row, see
  :func:`decode_outcome`.

A sidecar entry per case carries the measured step (wall, timestamps, capture
level, round index) and the shape counts a host-time fit needs; the NVTX range
the step ran inside is named so a comparison can join it to an nsys capture.

A record that cannot be lowered is listed in ``excluded`` with its reason, never
repaired.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from profiling.runners.attention.gdn_tree_verify_recurrence_reference import (
    slot_count_for_parents,
)

#: ``multi.py``: ``COPY, TREE, ONE = 0, 1, 2`` -- a stream's window this round.
MULTI_MODES: dict[int, str] = {0: "copy", 1: "tree", 2: "one"}
#: ``decode.py`` ``round`` record ``source``: a serial round has no drafter (ONE).
SINGLE_SOURCES: dict[str, str] = {"copy": "copy", "tree": "tree", "serial": "one"}

ITERATION_RANGE = "tensorfold.engine.iteration"
PREFILL_CHUNK_RANGE = "tensorfold.prefill.chunk"

CASES_NAME = "timing_predict_cases.json"
SIDECAR_NAME = "tensorfold_round_sidecar.json"
SIDECAR_SCHEMA_VERSION = 1


def decode_outcome(accepted: int, tree_slots: int, mode: str) -> list[Any]:
    """One ``decode_outcomes`` wire entry, parallel to a ``decode_requests`` row.

    Wire format of ``SpeculativePredictGroup.decode_outcomes`` (design §5e option 1,
    landed from ``tf-predsched``): ``[accepted, tree_slots, mode]``, ``accepted``
    excluding the root, ``mode`` in ``copy`` / ``tree`` / ``one``. This is the only
    place that shapes the entry.
    """
    return [accepted, tree_slots, mode]


@dataclass(frozen=True)
class Phase:
    """One capture window from ``phases.tsv``: a load level and its monotonic span."""

    level: int
    t0_ns: int
    t1_ns: int


def read_phases(path: Path) -> list[Phase]:
    """Read ``level<TAB>t0_ns<TAB>t1_ns`` lines (same clock as the records)."""
    phases = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"{path}:{number}: expected 'level t0_ns t1_ns', got {line!r}")
        level, t0, t1 = (int(field) for field in fields)
        if t1 <= t0:
            raise ValueError(f"{path}:{number}: window ends before it starts")
        phases.append(Phase(level, t0, t1))
    return phases


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read the JSONL; a torn last line (a killed server) is dropped, any other bad line fails."""
    lines = path.read_text().splitlines()
    records = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if number == len(lines):
                break
            raise ValueError(f"{path}:{number}: not a JSON record") from None
    return records


def _int(value: Any) -> bool:
    return type(value) is int


def _stream_shape(
    pos: Any, rows: Any, parents: Any, accepted: Any, mode: str
) -> tuple[list[int], list[Any]]:
    """Lower one verified window to its ``decode_requests`` row and outcome."""
    if not (_int(pos) and pos >= 0 and _int(rows) and rows >= 1):
        raise ValueError(f"invalid pos/rows {pos!r}/{rows!r}")
    if not isinstance(parents, list) or len(parents) != rows:
        raise ValueError(f"parents must list one entry per row ({rows})")
    if not (_int(accepted) and 0 <= accepted < rows):
        raise ValueError(f"accepted {accepted!r} outside 0..{rows - 1}")
    slots = slot_count_for_parents(parents)
    # The lowering's rules (tf-predsched): only a tree window holds live slots, and
    # a ONE window is the root alone.
    if slots and mode != "tree":
        raise ValueError(f"a {mode} window needs {slots} tree slots")
    if mode == "one" and rows != 1:
        raise ValueError(f"a one window has {rows} rows")
    return [pos + rows, rows], decode_outcome(accepted, slots, mode)


def _multi_round_case(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    prefill = record.get("prefill")
    streams = record.get("streams")
    if prefill is None and streams is None:
        raise ValueError("round has neither prefill nor streams")
    pairs = []
    for piece in prefill or []:
        if (
            not (isinstance(piece, list) and len(piece) == 3 and all(_int(v) for v in piece))
            or not 0 <= piece[1] < piece[2]
        ):
            raise ValueError(f"invalid prefill piece {piece!r}")
        pairs.append([piece[1], piece[2] - piece[1]])
    requests, outcomes = [], []
    for stream in streams or []:
        mode = MULTI_MODES.get(stream.get("mode"))
        if mode is None:
            raise ValueError(f"unknown stream mode {stream.get('mode')!r}")
        request, outcome = _stream_shape(
            stream.get("pos"),
            stream.get("rows"),
            stream.get("parents"),
            stream.get("accepted"),
            mode,
        )
        requests.append(request)
        outcomes.append(outcome)
    return _group(pairs, requests, outcomes), {"nvtx_range": ITERATION_RANGE}


def _single_round_case(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    mode = SINGLE_SOURCES.get(record.get("source"))
    if mode is None:
        raise ValueError(f"unknown round source {record.get('source')!r}")
    request, outcome = _stream_shape(
        record.get("pos"), record.get("rows"), record.get("parents"), record.get("accepted"), mode
    )
    return _group([], [request], [outcome]), {"nvtx_range": ITERATION_RANGE}


def _prefill_chunk_case(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    pos, rows = record.get("pos"), record.get("rows")
    if not (_int(pos) and pos >= 0 and _int(rows) and rows >= 1):
        raise ValueError(f"invalid pos/rows {pos!r}/{rows!r}")
    return _group([[pos, rows]], [], []), {"nvtx_range": PREFILL_CHUNK_RANGE}


def _group(pairs: list, requests: list, outcomes: list) -> dict[str, Any]:
    return {
        "prefill_chunk_pairs": pairs,
        "decode_requests": requests,
        "decode_outcomes": outcomes,
    }


def _phase_of(record: dict[str, Any], phases: Sequence[Phase]) -> Phase | None | str:
    """The window holding the whole record; ``"partial"`` when it straddles one."""
    start, end = record["start_ns"], record["end_ns"]
    for phase in phases:
        if phase.t0_ns <= start and end <= phase.t1_ns:
            return phase
        if start < phase.t1_ns and phase.t0_ns < end:
            return "partial"
    return None


def build_cases(
    records: Iterable[dict[str, Any]], phases: Sequence[Phase] | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(cases, sidecar, excluded)`` in record order.

    With ``phases``, only records wholly inside a capture window are kept; a record
    straddling a window edge is excluded, and one outside every window is skipped
    silently (it was never captured). ``round_index`` counts kept cases per level.
    """
    records = list(records)
    kinds = {record.get("kind") for record in records}
    if "multi_round" in kinds and "round" in kinds:
        raise ValueError("records mix multi_round and single-stream round lines")
    multi = "multi_round" in kinds
    builders = (
        {"multi_round": _multi_round_case}
        if multi
        else {"round": _single_round_case, "prefill_chunk": _prefill_chunk_case}
    )

    cases: list[dict[str, Any]] = []
    sidecar: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    per_level: dict[int | None, int] = {}
    for line, record in enumerate(records, start=1):
        kind = record.get("kind")
        start, end = record.get("start_ns"), record.get("end_ns")
        if not (_int(start) and _int(end) and start <= end):
            excluded.append({"record_line": line, "kind": kind, "reason": "invalid start/end_ns"})
            continue
        level = None
        if phases is not None:
            phase = _phase_of(record, phases)
            if phase is None:
                continue
            if phase == "partial":
                excluded.append(
                    {"record_line": line, "kind": kind, "reason": "straddles a capture window edge"}
                )
                continue
            level = phase.level
        if kind == "prefill_chunk" and multi:
            excluded.append(
                {"record_line": line, "kind": kind, "reason": "sub-step of a multi_round prefill"}
            )
            continue
        builder = builders.get(kind)
        if builder is None:
            excluded.append({"record_line": line, "kind": kind, "reason": "unknown record kind"})
            continue
        try:
            group, extra = builder(record)
        except ValueError as error:
            excluded.append({"record_line": line, "kind": kind, "reason": str(error)})
            continue

        round_index = per_level.get(level, 0)
        per_level[level] = round_index + 1
        outcomes = group["decode_outcomes"]
        sidecar.append(
            {
                "case_index": len(cases),
                "record_line": line,
                "kind": kind,
                "level": level,
                "round_index": round_index,
                "start_ns": start,
                "end_ns": end,
                "wall_ns": end - start,
                **extra,
                "streams": len(outcomes),
                "tree_streams": sum(outcome[2] == "tree" for outcome in outcomes),
                "verify_rows": sum(request[1] for request in group["decode_requests"]),
                "prefill_pieces": len(group["prefill_chunk_pairs"]),
                "prefill_rows": sum(pair[1] for pair in group["prefill_chunk_pairs"]),
            }
        )
        cases.append({"groups": [group]})
    return cases, sidecar, excluded


def write_outputs(
    output_dir: Path,
    cases: list[dict[str, Any]],
    sidecar: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    *,
    records_path: Path,
    phases_path: Path | None,
) -> tuple[Path, Path]:
    """Write the cases file and its sidecar under ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / CASES_NAME
    sidecar_path = output_dir / SIDECAR_NAME
    cases_path.write_text(json.dumps(cases, indent=1))
    sidecar_path.write_text(
        json.dumps(
            {
                "schema_version": SIDECAR_SCHEMA_VERSION,
                "input_builder": {"type": "tensorfold_iter_records"},
                "iter_records": str(records_path.resolve()),
                "phases": None if phases_path is None else str(phases_path.resolve()),
                "timing_predict_cases": str(cases_path.resolve()),
                "cases": sidecar,
                "excluded_records": excluded,
            },
            indent=1,
        )
    )
    return cases_path, sidecar_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records", type=Path, help="TENSORFOLD_ITER_RECORDS JSONL")
    parser.add_argument("--phases", type=Path, help="phases.tsv: level t0_ns t1_ns windows")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    phases = read_phases(args.phases) if args.phases else None
    cases, sidecar, excluded = build_cases(read_records(args.records), phases)
    if not cases:
        raise SystemExit("no record can be converted to a timing-predict case")
    cases_path, sidecar_path = write_outputs(
        args.output_dir,
        cases,
        sidecar,
        excluded,
        records_path=args.records,
        phases_path=args.phases,
    )
    levels: dict[Any, int] = {}
    for entry in sidecar:
        levels[entry["level"]] = levels.get(entry["level"], 0) + 1
    reasons: dict[str, int] = {}
    for entry in excluded:
        reasons[entry["reason"]] = reasons.get(entry["reason"], 0) + 1
    print(f"wrote {cases_path} ({len(cases)} cases) and {sidecar_path}")
    for level, count in levels.items():
        print(f"  level {level}: {count} cases")
    for reason, count in reasons.items():
        print(f"  excluded {count}: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
