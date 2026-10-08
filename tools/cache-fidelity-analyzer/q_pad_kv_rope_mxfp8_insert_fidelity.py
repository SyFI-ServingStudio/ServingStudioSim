"""`q_pad_kv_rope_mxfp8_insert` cache fidelity: a Recipe-B caller of `cache_fidelity.py`.

Why a dedicated caller: the Rust Input is `num_tokens` alone, and `enumerate`
derives `num_insert_tokens = num_tokens` (production passes one slot per row).
The generic harness builds perf_api specs from config dims plus input fields,
so it would omit `num_insert_tokens`. This caller adds it to every ground-truth
spec and places probes around the op's ReducedGrid cutoff (1024 tokens).

Pre-fill the grid rows first with `launcher kernel-profile run` (the
`vllm_upstream_fork_env` backend should be profiled from a Python process, not JIT-filled
through the `kernel-query` bridge). Then, on a B200:

    uv run python tools/cache-fidelity-analyzer/q_pad_kv_rope_mxfp8_insert_fidelity.py \
        --config cfg.json --log-dir logs/<dated>/fidelity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cache_fidelity as cf  # the generic core (sibling file)

KIND = "q_pad_kv_rope_mxfp8_insert"


def probes(grid_axis: list[float]) -> list[tuple[tuple, str]]:
    """Off-grid cell midpoints, the variant cutoff, small T, and beyond-grid."""
    out: list[tuple[tuple, str]] = []
    for lo, hi in zip(grid_axis, grid_axis[1:]):
        mid = round(cf.geomean(lo, hi))
        if lo < mid < hi:
            out.append(((mid,), "small" if hi <= 64 else "interior"))
    for t in (3, 5, 12, 24, 40):
        out.append(((t,), "small"))
    for t in (900, 1000, 1022, 1023, 1024, 1025, 1100, 1280):
        out.append(((t,), "cutoff"))
    for t in (20000, 24576, 28000):
        out.append(((t,), "interior"))
    for t in (40000, 49152, 65536):
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

    def ground_truth(kind, cfg, input_fields, shapes):
        """One batched perf_api call per backend, with the derived insert count."""
        assert input_fields == ["num_tokens"], input_fields
        cf.perf_api.enable_jit_profiling()
        fn = getattr(cf.perf_api, f"get_{kind}_times")
        dims = {k: v for k, v in cfg.items() if k not in cf._NON_DIM_KEYS}
        specs = [{**dims, "num_tokens": int(t), "num_insert_tokens": int(t)} for (t,) in shapes]
        best = [float("inf")] * len(specs)
        for backend in cfg["backends"]:
            res = fn(specs, backend=backend, gpu_name=cfg["gpu_name"])
            best = [min(b, float(getattr(r, "time_ms", float("inf")))) for b, r in zip(best, res)]
        return best

    cf.ground_truth = ground_truth
    grid = cf.query_grid(args.sim_bin, KIND, config)
    cf.run_fidelity(args.sim_bin, KIND, config, probes=probes(grid["grid_axes"][0]),
                    log_dir=args.log_dir)


if __name__ == "__main__":
    main()
