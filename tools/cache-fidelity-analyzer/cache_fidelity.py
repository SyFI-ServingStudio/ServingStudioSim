"""Generic L1 kernel-cache **interpolation-fidelity** harness (kernel-agnostic).

Measures how faithfully a kernel's *cache* reproduces the true kernel time at
**off-grid** shapes. Works for ANY registered `KernelSpec` via two exposed,
kernel-agnostic interfaces — no per-kernel Rust or Python wiring:

  - **Rust `simulator kernel-query`** (stdin/stdout JSON), the authoritative
    interpolation + grid metadata:
      * `grid` op  → `{describe_config, input_fields, grid_axes}` (bridge-free).
      * `eval` op  → best-of-N interpolated metrics at a batch of `query_points`
        (builds the kernel, JIT-profiling missing grid rows).
  - **Python `perf_api.get_{kind}_times(specs, backend=, gpu_name=)`** ground
    truth (the facade name is `get_{KIND}_times` by convention).

Division of labor: Python owns ONE config and feeds it to **both** the Rust
interp and the perf_api truth, so they can't describe different kernels. Rust
owns the authoritative bilinear/linear interp; we never reimplement it here.

For each off-grid shape: `interp = kernel-query(best-of-N)` vs
`truth = min_b perf_api.get_{kind}_times(shape, backend=b)` (best-of-N, matching
the sim's `Kernel::eval` argmin); report `ratio = interp / truth`.

## Two ways to use it

1. **Simple kernels** (cache grid == query space, e.g. `single_gemm`, `rms_norm`,
   `flashinfer_attn_decode`): run this file directly — `generic_probes` places
   probes straight off the kernel's own `grid_axes`:

       uv run python tools/cache-fidelity-analyzer/cache_fidelity.py \
           --kind single_gemm --config cfg.json --log-dir logs/<dated>/fidelity

   where `cfg.json` is the full `KernelConfig` JSON (`backends`, `gpu_name`, dims).

2. **Re-axis / domain-specific kernels** (cache grid is NOT the query space — e.g.
   `flashinfer_attn_prefill`, whose grid is `(A=k+q/2, B=q)` while queries are
   physical `(prefix_len, append_len)`): write a thin caller that builds the
   config and physical probes, then calls `run_fidelity(..., probes=...)`. See
   `flashinfer_attn_fidelity.py` in this folder. The harness sends your physical
   `query_points` to `kernel-query` (which projects them) and physical `specs` to
   perf_api, so the re-axis stays invisible here.

See skill `validate-kernel-cache` for the full workflow + gotchas.

Run from the repo root under uv (pins the PyO3 / venv interpreter).
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

# `profiling` / `launcher` live at the repo root; this file is
# `<repo>/tools/cache-fidelity-analyzer/`, so go up three. Put the repo on the
# path so the imports below resolve when run from anywhere under uv.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# perf_api + the launcher's PyO3 env wiring (run under uv so these import).
import profiling.perf_api as perf_api  # noqa: E402
from launcher.exec import _build_subprocess_env  # noqa: E402

WITHIN = 1.15  # the validated-model bar: within +/-15%.

# Config keys that are cache identity / call args, NOT per-spec kernel dims. The
# remaining config fields ARE the perf_api spec dims (must match `KernelArgs`).
_NON_DIM_KEYS = ("backends", "gpu_name")


# ─── exposed interfaces (kernel-agnostic) ────────────────────────────────────

def kernel_query(sim_bin: str, req: dict) -> dict:
    """Invoke `simulator kernel-query` under the PyO3 env; return parsed JSON.
    `req` is `{"op": "grid"|"eval", "kind": ..., "config": ..., [query_points]}`."""
    proc = subprocess.run(
        [sim_bin, "kernel-query"],
        input=json.dumps(req),
        capture_output=True,
        text=True,
        env=_build_subprocess_env(),
    )
    if proc.returncode != 0:
        sys.exit(f"kernel-query failed (exit {proc.returncode}):\n{proc.stderr}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        sys.exit(f"kernel-query exit 0 but stdout was not JSON (len={len(proc.stdout)}):\n"
                 f"--- stdout ---\n{proc.stdout!r}\n--- stderr ---\n{proc.stderr}")


def query_grid(sim_bin: str, kind: str, config: dict) -> dict:
    """`grid` op: fitted `grid_axes`, the `input_fields` that label the
    query-point keys, and the resolved `describe_config`. No bridge/GPU."""
    return kernel_query(sim_bin, {"op": "grid", "kind": kind, "config": config})


def eval_kind(sim_bin: str, kind: str, config: dict,
              query_points: list[dict]) -> list[dict]:
    """`eval` op: best-of-N interpolated result per query point. Each result is
    `{input, time_ms, flops, bytes, energy_j, coverage}` (coverage bits:
    EXTRAPOLATED=1, JIT=2, NO_COVERAGE=4)."""
    resp = kernel_query(sim_bin, {"op": "eval", "kind": kind, "config": config,
                                  "query_points": query_points})
    return resp["results"]


def ground_truth(kind: str, config: dict, input_fields: list[str],
                 shapes: list[tuple]) -> list[float]:
    """Best-of-N true time per shape via `perf_api.get_{kind}_times`. The spec for
    each point is `{**dims, **dict(zip(input_fields, shape))}` where `dims` is the
    config minus the non-dim keys — exactly the kernel's `KernelArgs` fields.
    Profiles each backend (JIT fills off-grid points on the GPU); min over
    backends matches the sim's argmin."""
    perf_api.enable_jit_profiling()
    fn = getattr(perf_api, f"get_{kind}_times")
    dims = {k: v for k, v in config.items() if k not in _NON_DIM_KEYS}
    specs = [{**dims, **dict(zip(input_fields, shape))} for shape in shapes]
    per_backend = {}
    for b in config["backends"]:
        res = fn(specs, backend=b, gpu_name=config["gpu_name"])
        per_backend[b] = [float(getattr(r, "time_ms", math.inf)) for r in res]
    return [min(per_backend[b][i] for b in config["backends"])
            for i in range(len(shapes))]


# ─── generic probe generation (cache grid == query space) ────────────────────

def geomean(a: float, b: float) -> float:
    return math.sqrt(a * b)


def generic_probes(grid_axes: list[list[float]]) -> list[tuple[tuple, str]]:
    """Dimension-general off-grid probes straight off a kernel's own `grid_axes`
    (valid only when the grid axes ARE the query coordinates). Returns
    `[(coords_tuple, region), ...]` with `coords` in `input_fields` order.

    Regions: per-axis cell midpoints (`axis{d}`, holding the others at a mid grid
    value), the cross-axis diagonal (`diag`), and beyond-grid `extrap`. Bounded in
    O(Σ axis lengths), so it stays cheap for 1D/2D/3D kernels alike. Callers with
    domain structure (a causal diagonal, a re-axis, memory-bound corners) should
    pass explicit probes to `run_fidelity` instead."""
    ndim = len(grid_axes)
    ref = [ax[len(ax) // 2] for ax in grid_axes]  # hold-other-axes-here baseline
    probes: list[tuple[tuple, str]] = []

    # per-axis interior midpoints (one axis swept, others at ref)
    for d, ax in enumerate(grid_axes):
        for i in range(len(ax) - 1):
            pt = list(ref)
            pt[d] = round(geomean(ax[i], ax[i + 1]))
            probes.append((tuple(pt), f"axis{d}"))

    # cross-axis diagonal cell midpoints (all axes move together)
    if ndim > 1:
        for i in range(min(len(ax) for ax in grid_axes) - 1):
            probes.append((tuple(round(geomean(ax[i], ax[i + 1])) for ax in grid_axes),
                           "diag"))

    # extrapolation beyond each axis max (others at ref)
    for d, ax in enumerate(grid_axes):
        for mult in (1.5, 2.0):
            pt = list(ref)
            pt[d] = round(ax[-1] * mult)
            probes.append((tuple(pt), "extrap"))

    return probes


# ─── analysis + reporting ─────────────────────────────────────────────────────

def stats(ratios: list[float]) -> tuple[float, int, int, float]:
    """(median, within±15%, n, worst-by-log-ratio) over finite ratios."""
    fin = [x for x in ratios if math.isfinite(x)]
    if not fin:
        return math.nan, 0, 0, math.nan
    within = sum(1 for x in fin if 1 / WITHIN <= x <= WITHIN)
    worst = max(fin, key=lambda x: abs(math.log(x)) if x > 0 else math.inf)
    return statistics.median(fin), within, len(fin), worst


def run_fidelity(sim_bin: str, kind: str, config: dict, *,
                 probes: list[tuple[tuple, str]] | None = None,
                 compare: str | None = None,
                 truth_kind: str | None = None,
                 log_dir: Path | None = None) -> list[dict]:
    """End-to-end fidelity run for one kernel `kind`, built from `config`.

    `probes`: explicit `[(coords, region), ...]` (coords in `input_fields` order);
      if None, generated generically off the kernel's own `grid_axes`.
    `compare`: optional second registry `kind` to eval on the SAME query points
      (e.g. an old vs new cache variant).
    `truth_kind`: perf_api facade to call for ground truth (defaults to `kind`;
      set when a cache variant reuses another kind's `profile_kind` table).

    Prints a per-region summary + (if comparing) a per-probe `extrap` table, and
    writes per-probe rows to `<log_dir>/fidelity_{base|compare}.csv`. Returns the
    rows."""
    grid = query_grid(sim_bin, kind, config)
    input_fields = grid["input_fields"]
    if probes is None:
        probes = generic_probes(grid["grid_axes"])
    shapes = [p for p, _ in probes]
    regions = [r for _, r in probes]
    query_points = [dict(zip(input_fields, s)) for s in shapes]

    base = eval_kind(sim_bin, kind, config, query_points)
    cmp = eval_kind(sim_bin, compare, config, query_points) if compare else None
    truth = ground_truth(truth_kind or kind, config, input_fields, shapes)

    rows = []
    for i, (shape, region) in enumerate(probes):
        tr = truth[i]
        row = {"region": region}
        row.update({f: int(v) for f, v in zip(input_fields, shape)})
        row["truth_ms"] = round(tr, 5)
        row["base_ms"] = round(base[i]["time_ms"], 5)
        row["base_ratio"] = round(base[i]["time_ms"] / tr if tr > 0 else math.inf, 4)
        if cmp is not None:
            row["cmp_ms"] = round(cmp[i]["time_ms"], 5)
            row["cmp_ratio"] = round(cmp[i]["time_ms"] / tr if tr > 0 else math.inf, 4)
        rows.append(row)

    title = kind + (f" vs {compare}" if compare else "")
    print(f"\n=== cache fidelity: {title} on {config['gpu_name']} ===")
    print(grid["describe_config"])
    print(f"backends={config['backends']}  bar=+/-{(WITHIN-1)*100:.0f}%  "
          f"({len(rows)} off-grid probes)\n")

    def line(label: str, rs: list[dict]) -> None:
        if not rs:
            return
        bmed, bwin, bn, bworst = stats([r["base_ratio"] for r in rs])
        out = (f"{label:<12} n={bn:>3}  base: med={bmed:.3f} "
               f"within={bwin}/{bn} worst={bworst:.3f}")
        if cmp is not None:
            cmed, cwin, cn, cworst = stats([r["cmp_ratio"] for r in rs])
            out += f"   |  cmp: med={cmed:.3f} within={cwin}/{cn} worst={cworst:.3f}"
        print(out)

    line("ALL", rows)
    # first-seen region order, so domain probes keep their meaningful ordering.
    seen: list[str] = []
    for r in regions:
        if r not in seen:
            seen.append(r)
    for region in seen:
        line(region, [r for r in rows if r["region"] == region])

    extrap = [r for r in rows if r["region"] == "extrap"]
    if extrap and cmp is not None:
        print(f"\nextrap per-probe ({', '.join(input_fields)}): truth | base | cmp")
        for r in extrap:
            coords = "  ".join(f"{r[f]:>7}" for f in input_fields)
            print(f"  {coords}  truth={r['truth_ms']:.4f}  "
                  f"base={r['base_ratio']:.3f}  cmp={r['cmp_ratio']:.3f}")

    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        out = log_dir / f"fidelity_{'compare' if compare else 'base'}.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nper-probe rows -> {out}")

    return rows


# ─── generic CLI (simple kernels whose grid == query space) ──────────────────

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Generic kernel-cache fidelity harness.")
    ap.add_argument("--kind", required=True, help="registry KIND to validate")
    ap.add_argument("--config", required=True, type=Path,
                    help="full KernelConfig JSON (backends, gpu_name, + dims)")
    ap.add_argument("--compare", default=None, help="optional second KIND to compare")
    ap.add_argument("--truth-kind", default=None,
                    help="perf_api facade kind for ground truth (default: --kind)")
    ap.add_argument("--sim-bin", default="target/release/simulator")
    ap.add_argument("--log-dir", type=Path, default=None)
    args = ap.parse_args()

    if not Path(args.sim_bin).exists():
        sys.exit(f"{args.sim_bin} not found — build it first:\n"
                 f"  uv run cargo build --release -p simulator")
    config = json.loads(args.config.read_text())
    run_fidelity(args.sim_bin, args.kind, config, compare=args.compare,
                 truth_kind=args.truth_kind, log_dir=args.log_dir)


if __name__ == "__main__":
    main()
