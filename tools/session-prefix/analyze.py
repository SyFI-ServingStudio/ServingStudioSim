#!/usr/bin/env python3
"""What session-aware prefix reuse achieved, per arm of a placement x cache sweep.

Reads the sweep directory `presets/session_placed_prefix_sweep.yaml` writes and
reports, per arm, the one number the whole design exists to produce:

    achieved hit rate = sum(prefix_cache_hit_tokens) / sum(declared_prefix_tokens)

`declared_prefix_tokens` is what the trace SAYS is reusable -- the planned hit
rate, an upper bound the manifest also records. `prefix_cache_hit_tokens` is
what the modelled KV actually had when the round arrived. The gap between them
is finite memory and eviction, which is the part a simulator has to predict.

The comparisons that make the arms worth running:

  recorded vs least_queued   how much of the hit rate came from the PLACEMENT.
                             If ignoring the trace's engine column does not
                             lower it, something else is feeding the hits and
                             the number does not mean what it looks like.
  opportunistic vs disabled  `disabled` must be exactly 0.0. It is the control,
                             and a nonzero there means the mode knob is not
                             reaching the KV layer.
  hit rate vs TTFT           a hit that does not shorten prefill has not been
                             wired through to the cost model. If hit rate moves
                             between arms and TTFT does not, stop and look.

Usage:

    uv run python tools/session-prefix/analyze.py logs/session_placed_prefix_sweep
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow.parquet as pq

# Arm directories only: a sweep root also holds `payloads/`, `plots/`,
# `reports/` and `git_snapshot/`, none of which are runs.
SLO = Path("raw") / "request_slo.parquet"


def arm_rows(run: Path) -> dict[str, float]:
    table = pq.read_table(
        run / SLO,
        columns=[
            "declared_prefix_tokens",
            "prefix_cache_hit_tokens",
            "fresh_prompt_tokens",
            "ttft_ms",
            "finish_decode_time_ms",
            "stage_worker_ids",
        ],
    )
    declared = sum(table["declared_prefix_tokens"].to_pylist())
    hit = sum(table["prefix_cache_hit_tokens"].to_pylist())
    fresh = sum(table["fresh_prompt_tokens"].to_pylist())
    ttfts = [t for t in table["ttft_ms"].to_pylist() if t is not None]
    ends = [t for t in table["finish_decode_time_ms"].to_pylist() if t is not None]
    # Where each round actually ran: its first recorded stage. A round that
    # moved (migration) still started somewhere, and that is the placement.
    first_worker = [
        stages[0] for stages in table["stage_worker_ids"].to_pylist() if stages
    ]
    spread = {worker: first_worker.count(worker) for worker in sorted(set(first_worker))}
    return {
        "rounds": table.num_rows,
        "declared": declared,
        "hit": hit,
        # The tokens prefill actually had to compute. `hit + computed ==
        # fresh + declared` is the KV layer's own identity, so this is the
        # honest denominator for "what did reuse save".
        "computed": fresh + declared - hit,
        "achieved": hit / declared if declared else 0.0,
        "ttft_mean_ms": sum(ttfts) / len(ttfts) if ttfts else 0.0,
        "makespan_ms": max(ends) if ends else 0.0,
        "spread": spread,
    }


def main(root: Path) -> int:
    runs = sorted(d for d in root.iterdir() if (d / SLO).exists())
    if not runs:
        print(f"no runs with {SLO} under {root}", file=sys.stderr)
        return 1

    arms = {run.name: arm_rows(run) for run in runs}
    width = max(len(name) for name in arms)
    header = (
        f"{'arm':<{width}}  {'rounds':>7}  {'declared':>12}  {'hit':>12}  "
        f"{'computed':>12}  {'achieved':>9}  {'ttft ms':>9}  {'makespan s':>11}"
    )
    print(header)
    print("-" * len(header))
    for name, row in arms.items():
        print(
            f"{name:<{width}}  {row['rounds']:>7}  {row['declared']:>12,}  "
            f"{row['hit']:>12,}  {row['computed']:>12,}  "
            f"{row['achieved']:>8.2%}  {row['ttft_mean_ms']:>9.1f}  "
            f"{row['makespan_ms'] / 1000:>11.2f}"
        )

    print("\nrounds per engine (first stage):")
    for name, row in arms.items():
        print(f"  {name:<{width}}  {row['spread']}")

    for name, row in arms.items():
        if name.endswith("disabled") and row["hit"]:
            print(
                f"\nFAIL {name}: prefix_cache_mode=disabled scored "
                f"{row['hit']:,} hit tokens; the control arm is not a control",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(Path(sys.argv[1])))
