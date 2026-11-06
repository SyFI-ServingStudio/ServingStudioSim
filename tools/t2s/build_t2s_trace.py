"""Text2SQL slime rollouts -> one ServingStudio Sim session trace per rollout.

    uv run python tools/t2s/build_t2s_trace.py <per_turn.csv[.gz]> <out_dir> --arm NAME

Input is the handoff's per-turn export (`t2s_per_turn_export.py`): one row per
turn of every trajectory. Output is `<out_dir>/r{rollout}.csv` in the
independent text-generation format with the `session` and `placement` tags,
plus `<out_dir>/manifest.json` pinning the input's sha256 (of the uncompressed
CSV, as the handoff's Q&A 6 pins it) and per-rollout totals.

One session per trajectory: a dispatch row, then one row per turn.

  id                  r{rollout}_s{sample_index}_t{turn_idx} (dispatch row: _d)
  session_id          s{sample_index}
  arrival_time        0 on every row. Later rows are released by their
                      predecessor's completion plus its tool_wait_after_ms.
  dispatch row        input_len 1, output_len 1, no pin; its tool_wait_after_ms
                      is when turn 0 reached the engine: turn_t_recv - turn_server_s
                      of turn 0, minus the rollout's earliest turn-0 send. The
                      engines' own logs show no request before ~8.5 s although
                      the driver's sends start at 0 (handoff per-turn columns +
                      sglang_metrics), so the whole client-side gap is ahead of
                      the engine. A row is the only place a session can carry a
                      delay before its first engine request: an arrival column
                      would have to be non-decreasing in file order, and that
                      order must keep a GRPO group's five sessions back to back.
                      Cost: one 1-token prefill and one decode step per
                      trajectory at t = 0, before any real request arrives.
  input_len           turn_prompt_tokens: the whole context this turn sends
  prefix_kv           0 (see below)
  prefix_len          turn_cached_tokens, the measured radix hit, capped at
                      input_len - 1 (SGLang always computes at least one token)
  output_len          model_tokens (>= 1)
  tool_wait_after_ms  tool_time_s of this turn (the driver's inter-turn wait,
                      92% of it the 64-worker env pool's queueing, handoff Q&A 1)
                      plus the NEXT turn's driver-side gap turn_client_s -
                      turn_server_s, which likewise lies ahead of the engine;
                      0 on the last turn. `--no-driver-gap` drops the second
                      term and the dispatch delay (dispatch row then waits the
                      measured send offset only).
  target_worker       (group_index - 256 * rollout_id) mod 8: the streaming
                      router's group round-robin (handoff Q&A 2)

Why prefix_kv = 0 and the measured hit pinned: with prefix_kv = 0 the request's
first-admission hit is exactly `prefix_len` (the simulator's resident prefix is
max(pin, retained session hit), and a session hit is capped by prefix_kv), so
the replay reproduces the measured hit turn by turn, including the 0-hit head of
each step's burst, the 579-token shared system prefix, the sibling hits a GRPO
group gets on one engine and the later-turn misses the measured runs show.
The simulator cannot produce those cross-session hits itself (sibling and
system-prefix sharing are not session KV). A re-prefill (retraction or
migration) drops the pin and recomputes the whole context.

Sessions are written in send order (first turn_t_send, then sample_index), all
rows of one session together. Both arms send a GRPO group's five trajectories
back to back (verified: every block of five consecutive first sends is one
group), so dense session ids come out in group blocks of five -- what
`group_by: session-block` with `group_size: 5` reads as one prompt group.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path

ENGINES = 8
GROUPS_PER_ROLLOUT = 256
HEADER = ["id", "input_len", "output_len", "arrival_time", "session_id", "prefix_kv",
          "tool_wait_after_ms", "prefix_len", "target_worker"]


def read_rows(path: Path) -> tuple[list[dict[str, str]], str]:
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    digest = hashlib.sha256(raw).hexdigest()
    return list(csv.DictReader(io.StringIO(raw.decode()))), digest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("per_turn", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--arm", required=True, help="measured arm the per-turn file was exported from")
    ap.add_argument("--no-driver-gap", action="store_true",
                    help="replay tool_time_s only; no client-side gap ahead of the engine")
    args = ap.parse_args()

    rows, digest = read_rows(args.per_turn)
    by_rollout: dict[int, dict[int, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_rollout[int(row["rollout_id"])][int(row["sample_index"])].append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    totals = {}
    for rollout in sorted(by_rollout):
        trajectories = by_rollout[rollout]
        starts = {}
        for sample, turns in trajectories.items():
            turns.sort(key=lambda r: int(r["turn_idx"]))
            assert [int(r["turn_idx"]) for r in turns] == list(range(len(turns))), (rollout, sample)
            assert len(turns) == int(turns[0]["n_turns"]), (rollout, sample)
            starts[sample] = float(turns[0]["turn_t_send"])
        t0 = min(starts.values())
        order = sorted(trajectories, key=lambda s: (starts[s], s))
        groups = [int(trajectories[s][0]["group_index"]) for s in order]
        assert all(len(set(groups[i:i + 5])) == 1 for i in range(0, len(groups), 5)), rollout

        out_rows = []
        tot = defaultdict(float)
        for sample in order:
            turns = trajectories[sample]
            group = int(turns[0]["group_index"])
            target = (group - GROUPS_PER_ROLLOUT * rollout) % ENGINES
            gap = [0.0 if args.no_driver_gap else float(t["turn_client_s"]) - float(t["turn_server_s"])
                   for t in turns]
            dispatch_ms = (starts[sample] - t0 + gap[0]) * 1e3
            out_rows.append([f"r{rollout}_s{sample}_d", 1, 1, "0", f"s{sample}", 0,
                             f"{dispatch_ms:.3f}", 0, target])
            tot["driver_gap_s"] += sum(gap)
            for k, turn in enumerate(turns):
                prompt = int(turn["turn_prompt_tokens"])
                cached = min(int(turn["turn_cached_tokens"]), prompt - 1)
                output = max(1, int(turn["model_tokens"]))
                last = k == len(turns) - 1
                wait_ms = 0.0 if last else (float(turn["tool_time_s"]) + gap[k + 1]) * 1e3
                out_rows.append([f"r{rollout}_s{sample}_t{k}", prompt, output, "0",
                                 f"s{sample}", 0, f"{wait_ms:.3f}", cached, target])
                tot["rounds"] += 1
                tot["prompt_tokens"] += prompt
                tot["pinned_prefix_tokens"] += cached
                tot["output_tokens"] += output
                tot["tool_wait_s"] += float(turn["tool_time_s"]) if not last else 0.0
        tot["sessions"] = len(order)
        tot["burst_span_ms"] = round((max(starts.values()) - t0) * 1e3, 3)
        totals[rollout] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in tot.items()}
        with open(args.out_dir / f"r{rollout}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(HEADER)
            w.writerows(out_rows)

    manifest = {
        "generator": "tools/t2s/build_t2s_trace.py",
        "arm": args.arm,
        "source": args.per_turn.name,
        "source_sha256_uncompressed": digest,
        "driver_gap": not args.no_driver_gap,
        "format": {"input_file_format": "text-generation-independent",
                   "input_file_tags": ["session", "placement"]},
        "rollouts": totals,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {len(totals)} rollouts to {args.out_dir} (source sha256 {digest[:12]})")


if __name__ == "__main__":
    main()
