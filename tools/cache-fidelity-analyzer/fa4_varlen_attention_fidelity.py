"""`fa4_varlen_attention` cache fidelity: a Recipe-B caller.

Why a dedicated caller: the Rust Input is a ragged `query_context_pairs` batch
(plus the FULL-graph `padded_requests`) that the cache projects to (decode
requests, prefill tokens, equivalent context), and the backend (`vllm_fa4`)
runs in the vLLM environment. So ground truth comes from the public profiling
CLI with `--fresh` (measured, never written to a DB), and this file only places
physical probes and compares.

The layer comes from `--config` (a full KernelConfig JSON: backends, gpu_name
and the Fa4VarlenAttentionArgs dims); several configs may be given. Probes are
placed off the grid relative to the config's `max_model_len`,
`full_cudagraph_max_size` and `--capture-sizes` (the graph sizes a decode step
pads to) and `--chunk` (the step's prefill token budget):

- `graph_decode_uniform` / `graph_decode_ragged`: decode-only steps within the
  graph limit, padded to the next capture size;
- `eager_decode`: decode-only steps past the graph limit;
- `mixed`: one prefill chunk (off-grid length and prefix) beside ragged decodes;
- `two_chunks`: two prefill chunks in one step;
- `chunk`: a chunk alone at a long prefix;
- optionally `--capture-probes FILE`: exact batches from a vLLM capture (a JSON
  list of `{"label", "pairs", "padded_requests"?}`), labelled `capture_<label>`.

Two stages, run from the repo root under uv:

    uv run python tools/cache-fidelity-analyzer/fa4_varlen_attention_fidelity.py \
        probes --config GA.json --config SWA.json --capture-sizes 1,2,4,8,16,...,200 \
        --chunk 2048 --out-dir DIR
    uv run python -m profiling run fa4_varlen_attention --backend vllm_fa4 \
        --specs DIR/probe_specs.json --fresh --output-dir OUT --json > DIR/truth.json
    uv run python tools/cache-fidelity-analyzer/fa4_varlen_attention_fidelity.py \
        report --out-dir DIR --truth DIR/truth.json

`report` evaluates the Rust cache through `simulator kernel-query eval`, which
reads the grid rows from profile.db; profile the grid first.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import cache_fidelity as cf  # the generic core (sibling file)

KIND = "fa4_varlen_attention"


def _log_spread(count: int, lo: int, hi: int) -> list[list[int]]:
    """Decode contexts spread geometrically over [lo, hi]."""
    if count == 1:
        return [[1, hi]]
    return [[1, round(lo * (hi / lo) ** (i / (count - 1)))] for i in range(count)]


def _padded(batch: int, capture_sizes: list[int]) -> int:
    return min((size for size in capture_sizes if size >= batch), default=0)


def probes_for(cfg: dict, capture_sizes: list[int], chunk: int) -> list[dict]:
    """Physical probes off the config's grid."""
    mml = cfg["max_model_len"]
    cap = cfg["full_cudagraph_max_size"]
    out: list[dict] = []

    def add(pairs: list[list[int]], region: str) -> None:
        decode_only = all(q == 1 for q, _ in pairs)
        padded = _padded(len(pairs), capture_sizes) if decode_only and len(pairs) <= cap else 0
        out.append({"pairs": pairs, "padded_requests": padded, "region": region})

    graph_batches = [b for b in (3, 12, 24, 48, 100, 180) if b <= cap] or [1]
    for batch in graph_batches:
        for context in (700, 3000, 20000, 90000):
            if context <= mml:
                add([[1, context]] * batch, "graph_decode_uniform")
    for batch in graph_batches[1:]:
        add(_log_spread(batch, 300, mml - 1000), "graph_decode_ragged")
    for batch in (cap + 20, cap + 90):
        for context in (1500, 30000):
            add([[1, context]] * batch, "eager_decode")
    for size, prefix, decodes in [
        (100, 0, 3),
        (300, 1500, 20),
        (700, 0, 50),
        (1500, 5000, 10),
        (chunk - 32, 20000, 32),
        (chunk - 32, 64000, 32),
        (chunk - 32, 120000, 16),
        (512, 100000, 8),
        (128, 60000, 16),
    ]:
        if prefix + size > mml:
            continue
        pairs = _log_spread(decodes, 300, min(mml - 1000, 60000)) if decodes else []
        pairs.append([size, prefix + size])
        add(pairs, "mixed")
    add([[1, 2000]] * 20 + [[600, 600], [chunk - 640, 3000 + chunk]], "two_chunks")
    for size, prefix in [(chunk, 45000), (chunk, 90000), (1000, 120000), (1900, 0)]:
        if prefix + size <= mml:
            add([[size, prefix + size]], "chunk")
    return out


def _dims(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if k not in cf._NON_DIM_KEYS}


def _name(cfg: dict) -> str:
    return f"w{cfg['window']}_kv{cfg['num_kv_heads']}"


def stage_probes(
    out: Path, configs: list[dict], capture_sizes: list[int], chunk: int, capture: Path | None
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    extra = json.loads(capture.read_text()) if capture is not None else []
    probes, specs = [], []
    for index, cfg in enumerate(configs):
        generated = probes_for(cfg, capture_sizes, chunk)
        for entry in extra:
            generated.append(
                {
                    "pairs": entry["pairs"],
                    "padded_requests": entry.get("padded_requests", 0),
                    "region": f"capture_{entry['label']}",
                }
            )
        for probe in generated:
            probes.append({"config": index, **probe})
            specs.append(
                {
                    **_dims(cfg),
                    "query_context_pairs": probe["pairs"],
                    "padded_requests": probe["padded_requests"],
                }
            )
    (out / "configs.json").write_text(json.dumps(configs))
    (out / "probes.json").write_text(json.dumps(probes))
    (out / "probe_specs.json").write_text(json.dumps(specs))
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


def _describe(pairs: list[list[int]], padded: int) -> str:
    decodes = [c for q, c in pairs if q == 1]
    prefill = [f"({q},{c})" for q, c in pairs if q > 1]
    parts = []
    if decodes:
        parts.append(f"{len(decodes)}dec[{min(decodes)}..{max(decodes)}]")
    text = "+".join(parts + prefill)
    return f"{text} pad{padded}" if padded else text


def stage_report(out: Path, truth_path: Path, sim_bin: str) -> None:
    configs = json.loads((out / "configs.json").read_text())
    probes = json.loads((out / "probes.json").read_text())
    truth = json.loads(truth_path.read_text())["results"]
    assert len(truth) == len(probes), (len(truth), len(probes))
    rows = []
    for index, cfg in enumerate(configs):
        idx = [i for i, p in enumerate(probes) if p["config"] == index]
        points = [
            {
                "query_context_pairs": probes[i]["pairs"],
                "padded_requests": probes[i]["padded_requests"],
            }
            for i in idx
        ]
        results = cf.eval_kind(sim_bin, KIND, cfg, points)
        for i, r in zip(idx, results):
            t = truth[i]
            tms = t.get("metrics", {}).get("time_ms") if t.get("status") == "ok" else None
            rows.append(
                {
                    "layer": _name(cfg),
                    "region": probes[i]["region"],
                    "batch": _describe(probes[i]["pairs"], probes[i]["padded_requests"]),
                    "truth_us": round(tms * 1e3, 3) if tms else "",
                    "cache_us": round(r["time_ms"] * 1e3, 3),
                    "ratio_cache_truth": round(r["time_ms"] / tms, 4) if tms else "",
                    "coverage": r["coverage"],
                    "status": t.get("status"),
                }
            )
    with (out / "fidelity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["ratio_cache_truth"] != ""]
    bar = f"bar=+/-{(cf.WITHIN - 1) * 100:.0f}%"
    lines = [f"ALL {_stat([r['ratio_cache_truth'] for r in ok])}  {bar}"]
    for layer in dict.fromkeys(r["layer"] for r in ok):
        for region in dict.fromkeys(r["region"] for r in ok):
            sel = [
                r["ratio_cache_truth"] for r in ok if r["layer"] == layer and r["region"] == region
            ]
            if sel:
                lines.append(f"{layer:<10} {region:<22} {_stat(sel)}")
    for r in rows:
        lines.append(
            f"{r['layer']:<10} {r['region']:<22} {r['batch']:<40} truth={r['truth_us']} "
            f"cache={r['cache_us']} ratio={r['ratio_cache_truth']} "
            f"cov={r['coverage']} {r['status']}"
        )
    text = "\n".join(lines)
    (out / "fidelity_summary.txt").write_text(text + "\n")
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["probes", "report"])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, action="append", default=[])
    ap.add_argument("--capture-sizes", default="", help="comma-separated FULL graph sizes")
    ap.add_argument("--chunk", type=int, default=2048, help="prefill tokens per step")
    ap.add_argument("--truth", type=Path)
    ap.add_argument("--capture-probes", type=Path)
    ap.add_argument("--sim-bin", default="target/release/simulator")
    args = ap.parse_args()
    if args.stage == "probes":
        configs = [json.loads(path.read_text()) for path in args.config]
        if not configs:
            ap.error("probes needs at least one --config")
        sizes = sorted(int(s) for s in args.capture_sizes.split(",") if s)
        stage_probes(args.out_dir, configs, sizes, args.chunk, args.capture_probes)
    else:
        stage_report(args.out_dir, args.truth, args.sim_bin)


if __name__ == "__main__":
    main()
