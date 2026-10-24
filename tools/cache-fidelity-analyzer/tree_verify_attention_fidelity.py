"""`tree_verify_attention` cache fidelity: a Recipe-B caller.

Why a dedicated caller: the Rust Input is the exact per-stream
`query_context_pairs` list (window rows, committed keys) that the cache
projects to (streams, mean rows, mean committed keys), and the backend
(`tensorfold_triton`) runs in `tensorfold_env`. So ground truth comes from the
public profiling CLI with `--fresh` (measured, never written to a DB), and
this file only places physical probes and compares.

Probe regions, all off the grid's uniform cells:

- `single_chain`: one stream, off-knot rows (3-96) and committed keys;
- `uniform_multi`: off-knot stream counts and rows, every stream alike;
- `spread_ctx`: 16-row trees, committed keys spread evenly 5k-60k
  (the brief's mixed-context probe) at 3-16 streams;
- `ragged`: seeded random rows 1-16 and committed 0-131072 per stream;
- `one_long`: one long stream beside many short ones;
- `small_work`: many streams over almost no committed keys;
- `pieces`: more streams than the grid (`off_grid` splits the launch).

Two stages, run from the repo root under uv:

    uv run python tools/cache-fidelity-analyzer/tree_verify_attention_fidelity.py \
        probes --config cfg.json --out-dir DIR
    uv run python -m profiling run tree_verify_attention --backend tensorfold_triton \
        --specs DIR/probe_specs.json --fresh --gpu-name "NVIDIA GB10" --json > DIR/truth.json
    uv run python tools/cache-fidelity-analyzer/tree_verify_attention_fidelity.py \
        report --config cfg.json --out-dir DIR --truth DIR/truth.json

`report` evaluates the Rust cache through `simulator kernel-query eval`, which
reads the grid rows from profile.db; profile the grid first.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path

import cache_fidelity as cf  # the generic core (sibling file)

KIND = "tree_verify_attention"
MAX_COMMITTED = 131072


def probes() -> list[dict]:
    out: list[dict] = []

    def add(pairs: list[list[int]], region: str) -> None:
        out.append({"pairs": pairs, "region": region})

    for rows in (3, 10, 14, 24, 48, 96):
        for committed in (300, 3000, 20000, 100000):
            add([[rows, committed]], "single_chain")
    for streams, rows, committed in [
        (3, 16, 1500),
        (3, 16, 20000),
        (6, 16, 90000),
        (12, 16, 1500),
        (12, 16, 20000),
        (12, 16, 90000),
        (5, 6, 7000),
        (5, 14, 50000),
        (10, 6, 50000),
        (10, 14, 7000),
    ]:
        add([[rows, committed]] * streams, "uniform_multi")
    for streams in (3, 4, 8, 12, 16):
        add(
            [[16, round(5000 + 55000 * i / (streams - 1))] for i in range(streams)],
            "spread_ctx",
        )
    rng = random.Random(23)
    for streams in (2, 3, 5, 8, 8, 11, 16, 16):
        add(
            [[rng.randint(1, 16), rng.randint(0, MAX_COMMITTED - 16)] for _ in range(streams)],
            "ragged",
        )
    rng = random.Random(29)
    for streams in (4, 8, 13):  # production-like: rows 8-16, committed 2k-60k log-spread
        add(
            [[rng.randint(8, 16), round(2000 * 30 ** rng.random())] for _ in range(streams)],
            "ragged",
        )
    add([[16, 131000]] + [[16, 200]] * 7, "one_long")
    add([[16, 100000]] + [[4, 500]] * 15, "one_long")
    add([[1, 120000]] + [[16, 3000]] * 5, "one_long")
    add([[1, 100]] * 16, "small_work")
    add([[1, 800]] * 16, "small_work")
    add([[16, 300]] * 8, "small_work")
    add([[16, 5000]] * 20, "pieces")
    add([[8, 30000]] * 24, "pieces")
    return out


def _dims(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if k not in cf._NON_DIM_KEYS}


def stage_probes(out: Path, cfg: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    generated = probes()
    specs = [{**_dims(cfg), "query_context_pairs": p["pairs"]} for p in generated]
    (out / "probes.json").write_text(json.dumps(generated))
    (out / "probe_specs.json").write_text(json.dumps({"specs": specs}))
    print(len(specs), "probe specs ->", out / "probe_specs.json")


def _stat(ratios: list[float]) -> str:
    fin = [x for x in ratios if math.isfinite(x) and x > 0]
    if not fin:
        return "n=0"
    within = sum(1 for x in fin if 1 / cf.WITHIN <= x <= cf.WITHIN)
    worst = max(fin, key=lambda x: abs(math.log(x)))
    return (
        f"n={len(fin)} med={statistics.median(fin):.3f} "
        f"within={within}/{len(fin)} worst={worst:.3f}"
    )


def _describe(pairs: list[list[int]]) -> str:
    rows = [w for w, _ in pairs]
    keys = [p for _, p in pairs]
    if len(pairs) == 1:
        return f"1x({rows[0]},{keys[0]})"
    return (
        f"{len(pairs)}x rows[{min(rows)}..{max(rows)}] "
        f"keys[{min(keys)}..{max(keys)}] mean{sum(keys) // len(keys)}"
    )


def stage_report(out: Path, cfg: dict, truth_path: Path, sim_bin: str) -> None:
    generated = json.loads((out / "probes.json").read_text())
    truth = json.loads(truth_path.read_text())["results"]
    assert len(truth) == len(generated), (len(truth), len(generated))
    points = [{"query_context_pairs": p["pairs"]} for p in generated]
    results = cf.eval_kind(sim_bin, KIND, cfg, points)
    rows = []
    for probe, t, r in zip(generated, truth, results):
        tms = t.get("metrics", {}).get("time_ms") if t.get("status") == "ok" else None
        rows.append(
            {
                "region": probe["region"],
                "batch": _describe(probe["pairs"]),
                "truth_us": round(tms * 1e3, 3) if tms else "",
                "cache_us": round(r["time_ms"] * 1e3, 3),
                "ratio_cache_truth": round(r["time_ms"] / tms, 4) if tms else "",
                "coverage": r["coverage"],
                "status": t.get("status"),
                "pairs": json.dumps(probe["pairs"]),
            }
        )
    with (out / "fidelity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["ratio_cache_truth"] != ""]
    bar = f"bar=+/-{(cf.WITHIN - 1) * 100:.0f}%"
    lines = [f"ALL {_stat([r['ratio_cache_truth'] for r in ok])}  {bar}"]
    for region in dict.fromkeys(r["region"] for r in ok):
        sel = [r["ratio_cache_truth"] for r in ok if r["region"] == region]
        lines.append(f"{region:<14} {_stat(sel)}")
    for r in rows:
        lines.append(
            f"{r['region']:<14} {r['batch']:<44} truth={r['truth_us']} cache={r['cache_us']} "
            f"ratio={r['ratio_cache_truth']} cov={r['coverage']} {r['status']}"
        )
    text = "\n".join(lines)
    (out / "fidelity_summary.txt").write_text(text + "\n")
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["probes", "report"])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--truth", type=Path)
    ap.add_argument("--sim-bin", default="target/release/simulator")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.stage == "probes":
        stage_probes(args.out_dir, cfg)
    else:
        stage_report(args.out_dir, cfg, args.truth, args.sim_bin)


if __name__ == "__main__":
    main()
