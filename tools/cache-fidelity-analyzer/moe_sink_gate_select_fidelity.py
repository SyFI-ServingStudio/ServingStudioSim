"""`moe_sink_gate_select` cache fidelity: a Recipe-A caller with serving-shaped probes.

The Rust Input is `num_tokens` and the config dims are exactly the Python args,
so the generic ground truth applies. The grid holds every vLLM CUDA-graph
capture size up to 2048, so graph-padded calls read a profiled row; this caller
probes what falls between rows: cell midpoints, unpadded decode and
mixed-batch sizes (eager or differently padded deployments), the gap between
2048 and the first extrapolation row, and sizes past the 16384-token grid edge.

Pre-fill the grid rows first with `python -m profiling run` (the
`vllm_upstream_fork_env` backend should be profiled from a Python process, not
JIT-filled through the `kernel-query` bridge). Off-grid ground truth is
JIT-profiled, so point `VIBESIM_PROFILE_DB` at a scratch copy. On a B200:

    uv run python tools/cache-fidelity-analyzer/moe_sink_gate_select_fidelity.py \
        --config cfg.json --log-dir logs/<dated>/fidelity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cache_fidelity as cf  # the generic core (sibling file)

KIND = "moe_sink_gate_select"


def probes(grid_axis: list[float]) -> list[tuple[tuple, str]]:
    """Cell midpoints, unpadded decode/mixed sizes, the 2048-3072 gap, beyond-grid."""
    on_grid = {round(t) for t in grid_axis}
    out: list[tuple[tuple, str]] = []
    for lo, hi in zip(grid_axis, grid_axis[1:]):
        mid = round(cf.geomean(lo, hi))
        if lo < mid < hi:
            region = "decode" if hi <= 256 else ("mixed" if hi <= 2048 else "gap")
            out.append(((mid,), region))
    for t in (3, 5, 7, 13, 25, 29, 31, 57, 121, 149):
        out.append(((t,), "decode"))
    for t in (285, 345, 651, 1054, 1405, 1702, 2018, 2040):
        out.append(((t,), "mixed"))
    for t in (2300, 2600, 3500):
        out.append(((t,), "gap"))
    for t in (20000, 24576, 32768):
        out.append(((t,), "extrap"))
    seen: set = set()
    return [
        p for p in out if p[0][0] not in on_grid and not (p[0] in seen or seen.add(p[0]))
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--sim-bin", default="target/release/simulator")
    ap.add_argument("--log-dir", type=Path, required=True)
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    grid = cf.query_grid(args.sim_bin, KIND, config)
    cf.run_fidelity(
        args.sim_bin,
        KIND,
        config,
        probes=probes(grid["grid_axes"][0]),
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
