#!/usr/bin/env python3
"""Cache fidelity for `mega_moe` (DeepGEMM fused EP MoE).

The cache is 1D over the per-rank token count, but the Config carries
`max_tokens_per_rank`, which bounds the grid and is not a profiler arg, so the
generic CLI (which sends every non-identity config field to perf_api) cannot
build ground truth. This caller drops it from the dims and probes every grid
cell's geometric midpoint plus the tile-size boundaries of
`get_block_m_for_mega_moe` (expected rows per expert crossing 10, 24 and the
192/240-row block choices), where the time steps rather than bends.

    uv run python tools/cache-fidelity-analyzer/mega_moe_fidelity.py \
        --config cfg.json --log-dir logs/<date>_mega_moe_fidelity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cache_fidelity as cf  # the generic core (sibling file)


def block_m_boundaries(config: dict) -> list[int]:
    """Token counts just below and above each `get_block_m_for_mega_moe` switch:
    the expected rows per local expert `n·ep·k/E` crossing 10 (16 -> 32 rows) and
    24 (32 -> 64+ rows)."""
    per_token = config["ep_size"] * config["top_k"] / config["num_experts"]
    points = []
    for rows in (10, 24):
        edge = rows / per_token
        points += [int(edge), int(edge) + 2]
    return points


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="full MegaMoeKernelConfig JSON")
    ap.add_argument("--sim-bin", default="target/release/simulator")
    ap.add_argument("--log-dir", type=Path)
    args = ap.parse_args()
    config = json.loads(Path(args.config).read_text())

    cf._NON_DIM_KEYS = (*cf._NON_DIM_KEYS, "max_tokens_per_rank")
    axis = cf.query_grid(args.sim_bin, "mega_moe", config)["grid_axes"][0]
    probes = [((round(cf.geomean(a, b)),), "interior") for a, b in zip(axis, axis[1:])]
    probes = [p for p in probes if p[0][0] not in axis]
    probes += [((n,), "block_m_edge") for n in block_m_boundaries(config)]
    cf.run_fidelity(args.sim_bin, "mega_moe", config, probes=probes, log_dir=args.log_dir)


if __name__ == "__main__":
    main()
