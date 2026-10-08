"""`inkling_qkvr_prep` cache fidelity: a Recipe-A caller with switch-aware probes.

The Rust Input is `num_tokens` and the config dims are exactly the Python args,
so the generic ground truth applies. This caller only places probes where the
call changes implementation: the 128-token fork, and the rel-projection variant
at `8 * num_tokens` = 8192 rows (`rel_extent` 512) or 2048 rows (otherwise),
plus decode-sized shapes, mixed-batch shapes and beyond-grid extrapolation.

Pre-fill the grid rows first with `launcher kernel-profile run` (the
`vllm_upstream_fork_env` backend should be profiled from a Python process, not
JIT-filled through the `kernel-query` bridge). Off-grid ground truth is
JIT-profiled, so point `VIBESIM_PROFILE_DB` at a scratch copy. On a B200:

    uv run python tools/cache-fidelity-analyzer/inkling_qkvr_prep_fidelity.py \
        --config cfg.json --log-dir logs/<dated>/fidelity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cache_fidelity as cf  # the generic core (sibling file)

KIND = "inkling_qkvr_prep"


def probes(grid_axis: list[float], rel_extent: int, num_q_heads: int) -> list[tuple[tuple, str]]:
    """Cell midpoints, decode sizes, both switches from each side, beyond-grid."""
    variant = -(-(8192 if rel_extent == 512 else 2048) // num_q_heads)
    out: list[tuple[tuple, str]] = []
    for lo, hi in zip(grid_axis, grid_axis[1:]):
        mid = round(cf.geomean(lo, hi))
        if lo < mid < hi:
            out.append(((mid,), "decode" if hi <= 128 else "interior"))
    for t in (5, 10, 25, 28, 31, 44, 100, 125):
        out.append(((t,), "decode"))
    for t in (126, 129, 130, 140):
        out.append(((t,), "fork"))
    for t in (variant - 8, variant - 2, variant + 1, variant + 8):
        out.append(((t,), "rel_variant"))
    for t in (1900, 2000, 2018, 2040):
        out.append(((t,), "mixed"))
    for t in (10000, 12288, 16384):
        out.append(((t,), "extrap"))
    seen: set = set()
    return [p for p in out if not (p[0] in seen or seen.add(p[0]))]


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
        probes=probes(grid["grid_axes"][0], config["rel_extent"], config["num_q_heads"]),
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
