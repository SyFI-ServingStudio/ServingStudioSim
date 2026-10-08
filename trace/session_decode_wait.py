"""Fold decode time into a session trace's tool waits, for prefill-only engines.

A pipeline head with `external_decode: true` completes a round at its first
token and retains the round's whole context, as if a decode instance ran the
remaining output and handed the KV back. The decode time then has to appear
somewhere, or the next round would arrive too early. This script adds it to the
round's `tool_wait_after_ms`:

    tool_wait_after_ms += (output_len - 1) / decode_tok_s * 1000

The first output token is produced by the prefill itself, so it costs no
decode time. Everything else is kept: sessions, rounds, the arrival timeline,
`prefix_len`, `input_len` and `output_len` (the simulator retains the outputs'
KV for the next round). A round whose prefix covers its whole prompt (zero
fresh tokens) becomes one fresh token after a prefix one token shorter, as in
`session_to_prefill_only.py`, since an engine still computes the last token.

`--sessions N` keeps the first N sessions by arrival, so the Poisson session
timeline is a prefix of the source's, and a preset's `workload.request_rate`
scales it to sessions per second. `--repeat K` lays K copies of the kept
sessions end to end (copy k's sessions start `k * span` later, span being the
last session start plus one mean inter-arrival gap), so a high session rate
still has arrivals for a long run.

A manifest beside the output records the source, the selection, and the totals.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

COLUMNS = (
    "request_id",
    "session_id",
    "round_idx",
    "arrival_time_ms",
    "prefix_len",
    "input_len",
    "output_len",
    "tool_wait_after_ms",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path, help="session-execution-v2 trace")
    parser.add_argument("out", type=Path)
    parser.add_argument("--decode-tok-s", type=float, required=True, help="decode speed per request")
    parser.add_argument("--sessions", type=int, default=None, help="keep the first N sessions by arrival")
    parser.add_argument("--repeat", type=int, default=1, help="copies of the kept sessions, end to end")
    args = parser.parse_args()
    if args.decode_tok_s <= 0:
        raise SystemExit("--decode-tok-s must be positive")
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")

    sessions: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise SystemExit(f"{args.source}: missing columns {sorted(missing)}")
        for row in reader:
            sessions[row["session_id"]].append(row)
    start = {s: min(float(r["arrival_time_ms"]) for r in rows) for s, rows in sessions.items()}
    order = sorted(sessions, key=start.get)
    kept = order[: args.sessions] if args.sessions is not None else order
    first, last = start[kept[0]], start[kept[-1]]
    span_ms = last + (last - first) / max(len(kept) - 1, 1)
    session_stride = max(int(s) for s in kept) + 1

    totals = dict(rounds=0, prefix_tokens=0, input_tokens=0, output_tokens=0, decode_wait_ms=0.0, tool_wait_ms=0.0)
    with args.out.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(COLUMNS)
        for copy, session in ((c, s) for c in range(args.repeat) for s in kept):
            for row in sorted(sessions[session], key=lambda r: int(r["round_idx"])):
                prefix_len, input_len = int(row["prefix_len"]), int(row["input_len"])
                if input_len == 0:
                    prefix_len, input_len = prefix_len - 1, 1
                output_len = int(row["output_len"])
                tool_wait = float(row["tool_wait_after_ms"])
                decode_wait = max(output_len - 1, 0) / args.decode_tok_s * 1000.0
                session_id = int(row["session_id"]) + copy * session_stride
                writer.writerow(
                    (
                        row["request_id"] if copy == 0 else f"session_{session_id}_round_{int(row['round_idx']):06d}",
                        session_id,
                        row["round_idx"],
                        f"{float(row['arrival_time_ms']) + copy * span_ms:.6f}" if copy else row["arrival_time_ms"],
                        prefix_len,
                        input_len,
                        output_len,
                        f"{tool_wait + decode_wait:.6f}",
                    )
                )
                totals["rounds"] += 1
                totals["prefix_tokens"] += prefix_len
                totals["input_tokens"] += input_len
                totals["output_tokens"] += output_len
                totals["decode_wait_ms"] += decode_wait
                totals["tool_wait_ms"] += tool_wait

    manifest = {
        "source": str(args.source),
        "source_sha256": _sha256(args.source),
        "decode_tok_s": args.decode_tok_s,
        "sessions": len(kept),
        "sessions_in_source": len(order),
        "repeat": args.repeat,
        "span_ms": span_ms,
        **totals,
    }
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
