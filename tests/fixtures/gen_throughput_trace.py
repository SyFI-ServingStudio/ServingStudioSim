"""Deterministic generator for the throughput/sim-speed regression trace.

A fixed seed + shape distribution produces a reproducible single-round trace
(`id,input_len,output_len,arrival_time`) without committing a large CSV. The
throughput golden and the sim-speed bench both run this trace; the golden key
encodes `n` so changing the size (or this generator) cleanly invalidates it.

Larger than a smoke trace on purpose: the sim-speed bench measures the tick-loop
wall time (`summary.json.wall_s`, timed from inside `run_sim` — startup excluded),
so the loop must run long enough (~1-2k requests) to give a stable number.

Library use:  `write_trace(path, n=1500, seed=0)`
CLI (inspect): `python tests/fixtures/gen_throughput_trace.py out.csv --n 1500`
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

# Realistic-ish shape mix (weighted). Prefill capped at 4096 to bound per-request
# KV/compute so the run stays a few seconds; decode lengths span short/medium.
_INPUT_CHOICES = [256, 512, 768, 1024, 1536, 2048, 3072, 4096]
_INPUT_WEIGHTS = [3, 4, 3, 4, 3, 3, 2, 1]
_OUTPUT_CHOICES = [32, 64, 128, 256]
_OUTPUT_WEIGHTS = [2, 4, 3, 1]
# Mean inter-arrival in rate-1-normalized ms. At request_rate=150 the effective
# spacing is this/150 (~0.33ms), so arrivals outpace service → a real backlog,
# which is what makes throughput a meaningful (contention-sensitive) signal.
_MEAN_INTERARRIVAL_MS = 50.0

DEFAULT_N = 1500
DEFAULT_SEED = 0


def generate_rows(n: int = DEFAULT_N, seed: int = DEFAULT_SEED) -> list[tuple[int, int, int, float]]:
    """`n` rows of `(id, input_len, output_len, arrival_time)`, fully deterministic
    for a given `(n, seed)`. ids are sequential; arrival_time is non-decreasing
    (cumulative exponential inter-arrivals) — both required by the trace loader."""
    rng = random.Random(seed)
    rows: list[tuple[int, int, int, float]] = []
    t = 0.0
    for i in range(n):
        input_len = rng.choices(_INPUT_CHOICES, weights=_INPUT_WEIGHTS)[0]
        output_len = rng.choices(_OUTPUT_CHOICES, weights=_OUTPUT_WEIGHTS)[0]
        rows.append((i, input_len, output_len, round(t, 3)))
        t += rng.expovariate(1.0 / _MEAN_INTERARRIVAL_MS)
    return rows


def write_trace(path: Path, n: int = DEFAULT_N, seed: int = DEFAULT_SEED) -> Path:
    lines = ["id,input_len,output_len,arrival_time"]
    lines += [f"{i},{inp},{out},{at}" for i, inp, out, at in generate_rows(n, seed)]
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path, help="output CSV path")
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    write_trace(args.out, args.n, args.seed)
    print(f"wrote {args.n} rows (seed={args.seed}) -> {args.out}")


if __name__ == "__main__":
    main()
