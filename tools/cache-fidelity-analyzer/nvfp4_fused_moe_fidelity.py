"""`nvfp4_fused_moe` cache fidelity: a Recipe-B caller of `cache_fidelity.py`.

Why a dedicated caller: the kernel's config carries `expert_demand` and
`folded_rank_position`, which are *not* profiler args. Rust turns them into a
`per_expert_batches` histogram per token count, so the generic harness (which
forwards config fields verbatim as spec dims) would send perf_api the wrong
keys and no histogram at all.

Ground truth must be measured on exactly the histograms the Rust cache was
built from. Rather than re-implement corpus / popularity sampling in Python,
this caller runs the `#[ignore]` Rust test
`timing::kernels::nvfp4_fused_moe::tests::dump_enumerate_payloads`, which calls
the real `Nvfp4FusedMoeSpec::enumerate` on a one-axis grid of the probe token
counts and prints each payload. Those payloads, minus `backend`, are the
perf_api specs. The Rust cache side is the ordinary `kernel-query eval`.

Run from the repo root under uv (GPU needed to fill grid rows and ground truth):

    uv run python tools/cache-fidelity-analyzer/nvfp4_fused_moe_fidelity.py \
        --config cfg.json --tokens 24,48,100,700,1500,3000 \
        --log-dir logs/<dated>/fidelity
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from pathlib import Path

import cache_fidelity as cf  # the generic core (sibling file)

KIND = "nvfp4_fused_moe"
_DUMP_TEST = "timing::kernels::nvfp4_fused_moe::tests::dump_enumerate_payloads"


def rust_payloads(config_path: Path, tokens: list[int]) -> list[dict]:
    """The exact `enumerate` payloads for `tokens`, one per (backend, T)."""
    env = cf._build_subprocess_env()
    env["NVFP4_MOE_DUMP_CONFIG"] = str(config_path.resolve())
    env["NVFP4_MOE_DUMP_TOKENS"] = ",".join(str(t) for t in tokens)
    proc = subprocess.run(
        ["cargo", "test", "--release", "-p", "simulator", "--lib", _DUMP_TEST,
         "--", "--ignored", "--nocapture", "--exact"],
        capture_output=True, text=True, env=env, cwd=cf._REPO,
    )
    if proc.returncode != 0:
        raise SystemExit(f"payload dump failed:\n{proc.stderr[-4000:]}")
    payloads = [json.loads(line[len("PAYLOAD "):])
                for line in proc.stdout.splitlines() if line.startswith("PAYLOAD ")]
    if not payloads:
        raise SystemExit("payload dump printed nothing (env vars not seen?)")
    return payloads


def region(t: int, grid: list[float]) -> str:
    if t in grid:
        return "on_grid"
    if t > 8192:
        return "above_autotune_8192"
    if t < 64:
        return "small_decode"
    return "interior"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, type=Path,
                    help="full Nvfp4FusedMoeKernelConfig JSON")
    ap.add_argument("--tokens", required=True, help="comma list of probe num_tokens")
    ap.add_argument("--sim-bin", default="target/release/simulator")
    ap.add_argument("--log-dir", type=Path, required=True)
    args = ap.parse_args()

    config = json.loads(args.config.read_text())
    tokens = [int(t) for t in args.tokens.split(",")]
    args.log_dir.mkdir(parents=True, exist_ok=True)

    grid = cf.query_grid(args.sim_bin, KIND, config)
    grid_axis = grid["grid_axes"][0]
    grid_tokens = [int(t) for t in grid_axis]
    grid_payloads = rust_payloads(args.config, grid_tokens)
    payloads = rust_payloads(args.config, tokens)
    (args.log_dir / "payloads.json").write_text(json.dumps(payloads))

    cf.perf_api.enable_jit_profiling()
    fn = getattr(cf.perf_api, f"get_{KIND}_times")

    # Pre-fill the grid rows from this process. `kernel-query eval` would JIT
    # them through the PyO3 bridge, but that process's PYTHONPATH carries the
    # project venv's site-packages into the worker, which shadows the vLLM
    # fork's own Torch for `vllm_fork_env` backends.
    for backend in config["backends"]:
        fn([{k: v for k, v in p.items() if k != "backend"}
            for p in grid_payloads if p["backend"] == backend],
           backend=backend, gpu_name=config["gpu_name"])

    # Rust cache (best-of-N over config backends).
    evals = cf.eval_kind(args.sim_bin, KIND, config, [{"num_tokens": t} for t in tokens])

    # Ground truth on the same histograms, best-of-N over backends.
    truth = {t: math.inf for t in tokens}
    for backend in config["backends"]:
        specs = [{k: v for k, v in p.items() if k != "backend"}
                 for p in payloads if p["backend"] == backend]
        results = fn(specs, backend=backend, gpu_name=config["gpu_name"])
        for spec, res in zip(specs, results):
            t = spec["num_tokens"]
            truth[t] = min(truth[t], float(getattr(res, "time_ms", math.inf)))

    rows = []
    for t, ev in zip(tokens, evals):
        p = next(p for p in payloads if p["num_tokens"] == t)
        local = p["per_expert_batches"][: config["num_local_experts"]]
        tr = truth[t]
        rows.append({
            "region": region(t, grid_axis),
            "num_tokens": t,
            "local_rows": sum(local),
            "local_active": sum(1 for x in local if x > 0),
            "local_max": max(local),
            "truth_ms": round(tr, 5),
            "base_ms": round(ev["time_ms"], 5),
            "base_ratio": round(ev["time_ms"] / tr if tr > 0 else math.inf, 4),
            "coverage": ev["coverage"],
        })

    out = args.log_dir / "fidelity_base.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n=== cache fidelity: {KIND} {config['backends']} on {config['gpu_name']} ===")
    print(f"grid points={len(grid_axis)}  probes={len(rows)}  bar=+/-15%")
    for r in rows:
        print(f"  T={r['num_tokens']:>6} {r['region']:<20} rows={r['local_rows']:>6} "
              f"active={r['local_active']:>2} max={r['local_max']:>5}  "
              f"truth={r['truth_ms']:.4f} cache={r['base_ms']:.4f} ratio={r['base_ratio']:.3f}")
    med, win, n, worst = cf.stats([r["base_ratio"] for r in rows])
    print(f"ALL n={n} med={med:.3f} within={win}/{n} worst={worst:.3f}")
    print(f"per-probe rows -> {out}")
    print(f"environment VIBESIM_PROFILE_DB={os.environ.get('VIBESIM_PROFILE_DB')}")


if __name__ == "__main__":
    main()
