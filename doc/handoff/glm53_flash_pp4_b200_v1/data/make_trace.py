"""Build the benchmark trace: coding-agent sessions arriving at a fixed rate for a fixed time.

Input: the session trace with decode time folded into each round's wait
(trace/session_decode_wait.py --decode-tok-s 80 over trace/tracelab_preserving.csv). Its sessions
arrive as a Poisson process at 1 session/s, from their first round.
Output: a session-execution-v2 CSV that req-frontend and the simulator replay the same way
(`arrival_mode: trace-timed`, no rate rescaling, no concurrency cap), plus a manifest.

- Arrivals: each session's arrival time is divided by the rate, so sessions arrive at
  --sessions-per-s. Only sessions that arrive before --minutes are kept. A session starts at its
  first round and runs its rounds in order; round k+1 is sent one tool wait after round k ends.
- No KV hand-back: the server computes one output token per round, so the other output tokens
  enter the next round as fresh prompt tokens. Per round, the prefix is capped at what the
  server holds (previous prompt + 1 output token), the rest moves to input_len, and
  output_len becomes 1. Every prompt keeps its length. req-frontend replays this as is: the next
  prompt is the previous prompt, its one real output token, then fresh tokens.
- The replay stops at --minutes (see 1_contract.md); later rounds are never sent.

usage: make_trace.py <decode_folded.csv> <out.csv> --sessions-per-s R [--minutes 20]
"""
import argparse, collections, csv, hashlib, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("src", type=Path)
ap.add_argument("dst", type=Path)
ap.add_argument("--sessions-per-s", type=float, required=True)
ap.add_argument("--minutes", type=float, default=20.0)
a = ap.parse_args()
end = a.minutes * 60e3

sessions = collections.OrderedDict()
with open(a.src) as f:
    for row in csv.DictReader(f):
        sessions.setdefault(row["session_id"], []).append(row)

kept = []
for rows in sessions.values():
    arrival = float(rows[0]["arrival_time_ms"]) / a.sessions_per_s
    if arrival >= end:
        break
    out, held = [], None  # held: tokens the server holds after the previous round
    for r in rows:
        prefix, inp = int(r["prefix_len"]), int(r["input_len"])
        new_prefix = prefix if held is None else min(prefix, held)
        new_input = inp + prefix - new_prefix
        out.append(dict(prefix_len=new_prefix, input_len=new_input, tool_wait_after_ms=float(r["tool_wait_after_ms"])))
        held = new_prefix + new_input + 1
    kept.append((arrival, out))
assert len(kept) < len(sessions), "the source ran out of sessions before the end; lower the rate or repeat the source"

cols = ["request_id", "session_id", "round_idx", "arrival_time_ms", "prefix_len", "input_len", "output_len",
        "tool_wait_after_ms"]
a.dst.parent.mkdir(parents=True, exist_ok=True)
with open(a.dst, "w", newline="") as f:
    w = csv.writer(f); w.writerow(cols)
    for sid, (arrival, rows) in enumerate(kept):
        for k, r in enumerate(rows):
            w.writerow([f"session_{sid}_round_{k:06d}", sid, k, f"{arrival:.6f}", r["prefix_len"], r["input_len"], 1,
                        f"{r['tool_wait_after_ms']:.6f}"])

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = dict(source=a.src.name, source_sha256=sha(a.src), sessions_per_s=a.sessions_per_s, minutes=a.minutes,
                sessions=len(kept), rounds=sum(len(r) for _, r in kept), sha256=sha(a.dst))
a.dst.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest))
