"""Order a session trace for a closed loop of N concurrent sessions, starting in steady state.

A preset with `arrival_mode: saturated` and `workload.max_concurrency: N`
keeps N sessions active: the first N sessions of the file start at once, and
each one that finishes lets in the next. Started from N fresh sessions,
though, the mix is far from steady. In steady state a slot holds a session
with probability proportional to its duration, and this corpus's durations
are heavy-tailed (half of all session-time sits in sessions longer than
3 h), so the mix would take many hours to settle.

So the file starts in the renewal process's equilibrium instead:

- The first N sessions are drawn with probability proportional to their
  duration (the sum of their rounds' waits, except the last round's). Each
  one joins at a uniformly random instant of its life, at the first round
  after that instant. Earlier rounds are dropped.
- Such a session starts with a one-token placeholder round (round 0), whose
  wait is the rest of the wait the instant fell in, so the joining rounds
  arrive spread as they would in steady state rather than all at t = 0. The
  joining round keeps its declared prefix: a server without the context
  recomputes it, and one with `prefix_tier_warm_start` reads it from its
  slowest tier, where a long-lived session's context would sit.
  `--cold-at-once` restores the old start instead: no placeholder, and the
  joining round is round 0 with its whole context as fresh tokens.
- Every later session is a whole session drawn uniformly, with replacement.

The source should already carry decode time in its waits
(`session_decode_wait.py`). Every row's arrival is 0, since saturated mode
releases sessions in file order. A manifest beside the output records the
draw.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import itertools
import json
import random
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
    parser.add_argument("source", type=Path, help="session-execution-v2 trace with decode in its waits")
    parser.add_argument("out", type=Path)
    parser.add_argument("--concurrency", type=int, required=True, help="sessions started in equilibrium")
    parser.add_argument("--after", type=int, required=True, help="whole sessions drawn after them")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cold-at-once",
        action="store_true",
        help="no placeholder round: join rounds arrive at t = 0 with their context as fresh tokens",
    )
    args = parser.parse_args()
    if args.concurrency < 1 or args.after < 0:
        raise SystemExit("--concurrency must be positive and --after non-negative")

    sessions: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise SystemExit(f"{args.source}: missing columns {sorted(missing)}")
        for row in reader:
            sessions[row["session_id"]].append(row)
    names = sorted(sessions, key=int)
    for name in names:
        sessions[name].sort(key=lambda r: int(r["round_idx"]))
    # A round's wait follows it; the next round arrives after it.
    waits = {s: [float(r["tool_wait_after_ms"]) for r in sessions[s][:-1]] for s in names}
    duration = {s: sum(waits[s]) for s in names}
    weighted = [s for s in names if duration[s] > 0]
    cumulative = list(itertools.accumulate(duration[s] for s in weighted))

    rng = random.Random(args.seed)
    totals = dict(rows=0, cold_tokens=0, dropped_rounds=0, join_delay_ms_sum=0.0, join_delay_ms_max=0.0)
    out_sessions = 0
    with args.out.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(COLUMNS)

        def emit(rows: list[dict[str, str]], first_prefix: int, delay_ms: float | None = None) -> None:
            nonlocal out_sessions
            sid = out_sessions
            out_sessions += 1
            if delay_ms is not None:
                # Placeholder round: one token in, one out, then the rest of the wait.
                rows = [
                    dict(prefix_len="0", input_len="1", output_len="1", tool_wait_after_ms=f"{delay_ms:.6f}"),
                    *rows,
                ]
                totals["join_delay_ms_sum"] += delay_ms
                totals["join_delay_ms_max"] = max(totals["join_delay_ms_max"], delay_ms)
                totals["cold_tokens"] += first_prefix
            for idx, row in enumerate(rows):
                prefix_len, input_len = int(row["prefix_len"]), int(row["input_len"])
                if idx == 0 and first_prefix and delay_ms is None:
                    prefix_len, input_len = 0, prefix_len + input_len
                    totals["cold_tokens"] += input_len
                writer.writerow(
                    (
                        f"session_{sid}_round_{idx:06d}",
                        sid,
                        idx,
                        "0.000000",
                        prefix_len,
                        input_len,
                        row["output_len"],
                        row["tool_wait_after_ms"],
                    )
                )
                totals["rows"] += 1

        for _ in range(args.concurrency):
            pick = bisect.bisect_right(cumulative, rng.random() * cumulative[-1])
            name = weighted[min(pick, len(weighted) - 1)]
            instant = rng.random() * duration[name]
            # Join at the first round arriving after `instant`, the rest of
            # that wait later.
            elapsed, join = 0.0, len(sessions[name]) - 1
            for k, wait in enumerate(waits[name]):
                elapsed += wait
                if elapsed >= instant:
                    join = k + 1
                    break
            totals["dropped_rounds"] += join
            rows = sessions[name][join:]
            emit(rows, int(rows[0]["prefix_len"]), None if args.cold_at_once else elapsed - instant)
        for _ in range(args.after):
            emit(sessions[rng.choice(names)], 0)

    manifest = {
        "source": str(args.source),
        "source_sha256": _sha256(args.source),
        "concurrency": args.concurrency,
        "after": args.after,
        "seed": args.seed,
        "cold_at_once": args.cold_at_once,
        "sessions": out_sessions,
        **totals,
    }
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
