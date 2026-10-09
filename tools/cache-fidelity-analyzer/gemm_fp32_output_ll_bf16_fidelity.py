"""`gemm_fp32_output` / `vllm_ll_bf16_auto` cache fidelity: switch-aware probes.

The Rust Input is `m` and the config dims are exactly the Python args, so the
generic ground truth applies. This caller only places probes where the router
call changes kernel: the ll_bf16 dot-product to split-K switch (m 4 -> 5) and
the ll_bf16 to cuBLAS switch (m 64 -> 65), plus eager decode sizes between the
CUDA-graph points, mixed-batch sizes and beyond-grid extrapolation.

Pre-fill the grid rows first with `launcher kernel-profile run` (the
`vllm_upstream_fork_env` backend should be profiled from a Python process, not
JIT-filled through the `kernel-query` bridge). Off-grid ground truth is
JIT-profiled, so point `VIBESIM_PROFILE_DB` at a scratch copy. On a B200:

    uv run python tools/cache-fidelity-analyzer/gemm_fp32_output_ll_bf16_fidelity.py \
        --config cfg.json --log-dir logs/<dated>/fidelity

with `cfg.json` = `{"backends": ["vllm_ll_bf16_auto"], "gpu_name":
"NVIDIA B200", "n": 264, "k": 4096, "input_dtype": "bf16"}`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cache_fidelity as cf  # the generic core (sibling file)

KIND = "gemm_fp32_output"


def probes(grid_axis: list[float]) -> list[tuple[tuple, str]]:
    """Cell midpoints, both sides of each switch, decode, mixed, beyond-grid."""
    out: list[tuple[tuple, str]] = []
    for lo, hi in zip(grid_axis, grid_axis[1:]):
        mid = round(cf.geomean(lo, hi))
        if lo < mid < hi:
            out.append(((mid,), "decode" if hi <= 128 else "interior"))
    for m in (6, 7, 9, 12, 25, 28, 31, 50, 60, 63):
        out.append(((m,), "ll_bf16"))
    for m in (66, 70, 77, 100, 125):
        out.append(((m,), "cublas_small"))
    for m in (1900, 2000, 2018, 2040):
        out.append(((m,), "mixed"))
    for m in (98304, 131072):
        out.append(((m,), "extrap"))
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
        probes=probes(grid["grid_axes"][0]),
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
