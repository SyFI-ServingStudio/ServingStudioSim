"""`gdn_state_replay` cache fidelity: a Recipe-B caller.

Why a dedicated caller: the Rust Input is the commit's totals (`batch_size`
streams, `num_tokens` accepted rows) that the cache projects to (streams, mean
rows a stream), and the backend (`tensorfold_cuda`) runs in `tensorfold_env`.
A profile row always splits its rows evenly, but a real commit is ragged, so
the probes carry each stream's rows:

- even probes (rows split as the profiler splits them) take their truth from
  the public profiling CLI with `--fresh` (measured, never written to a DB);
- ragged probes take theirs from the runner's own timing function
  (`measure_counts`, the same boundary a profile row uses) under the
  TensorFold interpreter.

Probe regions:

- `recheck`: grid knots, re-measured in this session (drift check);
- `single_chain`: one stream, off-knot rows (5-112);
- `uniform_multi`: off-knot stream counts and rows, every stream alike;
- `ragged`: seeded random rows 1-16 per stream, and half-1 / half-16 batches;
- `one_long`: one 16-row stream beside many 1-row ones, and the reverse;
- `pieces`: more streams than the grid (`off_grid` splits the launch).

Stages, from the repo root (the truths in one GPU-lock hold, one session):

    uv run python tools/cache-fidelity-analyzer/gdn_state_replay_fidelity.py \
        probes --config cfg.json --out-dir DIR
    uv run python -m profiling run gdn_state_replay --backend tensorfold_cuda \
        --specs DIR/probe_specs.json --fresh --gpu-name "NVIDIA GB10" --json > DIR/truth.json
    PYTHONPATH=$PWD:$PWD/alignment/profiler/tensorfold/src \
        alignment/profiler/tensorfold/.venv/bin/python \
        tools/cache-fidelity-analyzer/gdn_state_replay_fidelity.py \
        ragged-truth --config cfg.json --out-dir DIR
    uv run python tools/cache-fidelity-analyzer/gdn_state_replay_fidelity.py \
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

KIND = "gdn_state_replay"
WITHIN = 1.15  # cache_fidelity.WITHIN; kept here so ragged-truth runs without it.


def even_split(total: int, streams: int) -> list[int]:
    base, extra = divmod(total, streams)
    return [base + 1 if s < extra else base for s in range(streams)]


def probes() -> list[dict]:
    out: list[dict] = []

    def add(counts: list[int], region: str) -> None:
        out.append({"counts": counts, "region": region})

    for streams, rows in [(1, 1), (1, 16), (1, 128), (4, 8), (8, 16), (16, 1), (16, 16)]:
        add([rows] * streams, "recheck")
    for rows in (5, 10, 14, 20, 24, 40, 48, 80, 96, 112):
        add([rows], "single_chain")
    for streams, rows in [
        (3, 16),
        (3, 5),
        (5, 10),
        (6, 14),
        (7, 1),
        (10, 7),
        (12, 3),
        (14, 16),
        (16, 5),
        (16, 10),
    ]:
        add([rows] * streams, "uniform_multi")
    rng = random.Random(31)
    for streams in (2, 3, 5, 8, 8, 11, 16, 16):
        add([rng.randint(1, 16) for _ in range(streams)], "ragged")
    for streams in (2, 4, 8, 16):
        add([1, 16] * (streams // 2), "ragged")
    add([16] + [1] * 15, "one_long")
    add([1] + [16] * 7, "one_long")
    add([4] * 7 + [16], "one_long")
    add([16] * 20, "pieces")
    add([4] * 24, "pieces")
    return out


def is_even(counts: list[int]) -> bool:
    return counts == even_split(sum(counts), len(counts))


def _dims(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if k not in ("backends", "gpu_name")}


def stage_probes(out: Path, cfg: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    generated = probes()
    specs = [
        {**_dims(cfg), "batch_size": len(p["counts"]), "num_tokens": sum(p["counts"])}
        for p in generated
        if is_even(p["counts"])
    ]
    (out / "probes.json").write_text(json.dumps(generated))
    (out / "probe_specs.json").write_text(json.dumps({"specs": specs}))
    print(len(specs), "even probe specs ->", out / "probe_specs.json")
    print(len(generated) - len(specs), "ragged probes for ragged-truth")


def stage_ragged_truth(out: Path, cfg: dict) -> None:
    """Measure the ragged probes; run under the TensorFold interpreter."""

    import importlib

    import torch

    from profiling.runners.attention import gdn_state_replay_tensorfold as runner

    module = importlib.import_module("tensorfold.cuda.kernels.gdn")
    heads = dict(num_qk_heads=cfg["num_qk_heads"], num_value_heads=cfg["num_value_heads"])
    runner.check_correctness(torch, module, **heads)
    truth = {}
    for index, probe in enumerate(json.loads((out / "probes.json").read_text())):
        if is_even(probe["counts"]):
            continue
        metrics = runner.measure_counts(
            torch, module, counts=probe["counts"], num_layers=cfg["num_layers"], **heads
        )
        truth[index] = metrics.time_ms
        torch.cuda.empty_cache()
        print(index, probe["counts"], round(metrics.time_ms, 4), flush=True)
    (out / "ragged_truth.json").write_text(json.dumps(truth))


def _stat(ratios: list[float]) -> str:
    fin = [x for x in ratios if math.isfinite(x) and x > 0]
    if not fin:
        return "n=0"
    within = sum(1 for x in fin if 1 / WITHIN <= x <= WITHIN)
    worst = max(fin, key=lambda x: abs(math.log(x)))
    return (
        f"n={len(fin)} med={statistics.median(fin):.3f} "
        f"within={within}/{len(fin)} worst={worst:.3f}"
    )


def _describe(counts: list[int]) -> str:
    if len(set(counts)) == 1:
        return f"{len(counts)}x{counts[0]}"
    return f"{len(counts)}x rows[{min(counts)}..{max(counts)}] sum{sum(counts)}"


def stage_report(out: Path, cfg: dict, truth_path: Path, sim_bin: str) -> None:
    import cache_fidelity as cf  # the generic core (sibling file)

    generated = json.loads((out / "probes.json").read_text())
    even = iter(json.loads(truth_path.read_text())["results"])
    ragged = json.loads((out / "ragged_truth.json").read_text())
    points = [{"batch_size": len(p["counts"]), "num_tokens": sum(p["counts"])} for p in generated]
    results = cf.eval_kind(sim_bin, KIND, cfg, points)
    rows = []
    for index, (probe, r) in enumerate(zip(generated, results)):
        if is_even(probe["counts"]):
            t = next(even)
            tms = t.get("metrics", {}).get("time_ms") if t.get("status") == "ok" else None
            source = "cli"
        else:
            tms = ragged.get(str(index))
            source = "runner"
        rows.append(
            {
                "region": probe["region"],
                "batch": _describe(probe["counts"]),
                "truth_ms": round(tms, 4) if tms else "",
                "cache_ms": round(r["time_ms"], 4),
                "ratio_cache_truth": round(r["time_ms"] / tms, 4) if tms else "",
                "coverage": r["coverage"],
                "truth_source": source,
                "counts": json.dumps(probe["counts"]),
            }
        )
    with (out / "fidelity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["ratio_cache_truth"] != ""]
    bar = f"bar=+/-{(WITHIN - 1) * 100:.0f}%"
    lines = [f"ALL {_stat([r['ratio_cache_truth'] for r in ok])}  {bar}"]
    for region in dict.fromkeys(r["region"] for r in ok):
        sel = [r["ratio_cache_truth"] for r in ok if r["region"] == region]
        lines.append(f"{region:<14} {_stat(sel)}")
    for r in rows:
        lines.append(
            f"{r['region']:<14} {r['batch']:<28} truth={r['truth_ms']} cache={r['cache_ms']} "
            f"ratio={r['ratio_cache_truth']} cov={r['coverage']} {r['truth_source']}"
        )
    text = "\n".join(lines)
    (out / "fidelity_summary.txt").write_text(text + "\n")
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["probes", "ragged-truth", "report"])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--truth", type=Path)
    ap.add_argument("--sim-bin", default="target/release/simulator")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.stage == "probes":
        stage_probes(args.out_dir, cfg)
    elif args.stage == "ragged-truth":
        stage_ragged_truth(args.out_dir, cfg)
    else:
        stage_report(args.out_dir, cfg, args.truth, args.sim_bin)


if __name__ == "__main__":
    main()
