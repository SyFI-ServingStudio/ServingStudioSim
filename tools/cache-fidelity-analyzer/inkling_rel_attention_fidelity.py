"""`inkling_rel_attention` cache fidelity: a Recipe-B caller.

Why a dedicated caller: the Rust Input is a ragged `query_context_pairs` batch
that the cache projects to (decode requests, prefill tokens, equivalent
context), and the backend (`tml_fa4`) runs in the vLLM fork environment. So
ground truth comes from the public profiling CLI with `--fresh` (measured,
never written to a DB), and this file only places physical probes and compares.

Probes, for the Inkling-Small TP4 config (8 q / 2 KV heads, local 512-window
and global layers) at `--max-model-len` (default 8192):

- `decode_uniform`: off-grid batch sizes and contexts, every request alike;
- `decode_ragged`: contexts spread over a range, as a serving step has them;
- `mixed`: one prefill chunk (off-grid length and prefix) beside decodes;
- `two_chunks`: two prefill chunks in one step;
- `extrapolate`: past the grid (more decodes, a longer chunk);
- past 8192 the set is the long-context one instead (`long_context_probes`):
  `long_decode_uniform` at off-grid contexts, `long_decode_ragged`,
  `long_mixed` (a chunk at a long prefix beside ragged decodes) and
  `long_chunk` (a chunk alone at a long prefix);
- optionally `--capture-probes FILE`: exact batches taken from a vLLM capture
  (a JSON list of `{"label", "pairs", ...}`), labelled `capture_<label>`.

Two stages, run from the repo root under uv:

    uv run python tools/cache-fidelity-analyzer/inkling_rel_attention_fidelity.py \
        probes --out-dir DIR [--max-model-len N] [--capture-probes FILE]
    uv run python -m profiling run inkling_rel_attention --backend tml_fa4 \
        --specs DIR/probe_specs.json --fresh --output-dir OUT --json > DIR/truth.json
    uv run python tools/cache-fidelity-analyzer/inkling_rel_attention_fidelity.py \
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

KIND = "inkling_rel_attention"
BACKEND = "tml_fa4"
GPU = "NVIDIA B200"
_DIMS = {
    "num_qo_heads": 8,
    "num_kv_heads": 2,
    "head_dim": 128,
    "page_size": 32,
    "kv_layout": "lbnhc",
    "q_dtype": "bf16",
    "kv_dtype": "bf16",
}


def config(is_local: bool, max_model_len: int) -> dict:
    return {
        "backends": [BACKEND],
        "gpu_name": GPU,
        "is_local": is_local,
        "rel_extent": 512 if is_local else 1024,
        "max_model_len": max_model_len,
        **_DIMS,
    }


def _spread(count: int, lo: int, hi: int) -> list[list[int]]:
    if count == 1:
        return [[1, hi]]
    return [[1, round(lo + (hi - lo) * i / (count - 1))] for i in range(count)]


def synthetic_probes() -> list[tuple[list[list[int]], str]]:
    p: list[tuple[list[list[int]], str]] = []
    for batch, context in [
        (2, 3000),
        (3, 700),
        (12, 1500),
        (24, 300),
        (24, 6000),
        (48, 2500),
        (100, 700),
        (200, 3000),
    ]:
        p.append(([[1, context]] * batch, "decode_uniform"))
    for batch, lo, hi in [(5, 300, 4300), (24, 256, 4300), (60, 100, 8000), (150, 500, 3000)]:
        p.append((_spread(batch, lo, hi), "decode_ragged"))
    for chunk, prefix, decodes, lo, hi in [
        (40, 2000, 20, 300, 3000),
        (100, 0, 3, 1000, 1000),
        (300, 1500, 20, 256, 4300),
        (700, 0, 50, 300, 2000),
        (1500, 5000, 10, 1000, 6000),
        (1900, 0, 0, 0, 0),
        (2000, 3000, 30, 256, 4300),
    ]:
        pairs = _spread(decodes, lo, hi) if decodes else []
        pairs.append([chunk, prefix + chunk])
        p.append((pairs, "mixed"))
    p.append(([[1, 2000]] * 20 + [[600, 600], [1200, 3000]], "two_chunks"))
    p.append(([[1, 2000]] * 400, "extrapolate"))
    p.append(([[4096, 4096]], "extrapolate"))
    return p


def _log_spread(count: int, lo: int, hi: int) -> list[list[int]]:
    """Decode contexts spread geometrically over [lo, hi], as a long-context
    serving step holds many short requests and a few long ones."""
    if count == 1:
        return [[1, hi]]
    return [[1, round(lo * (hi / lo) ** (i / (count - 1)))] for i in range(count)]


def long_context_probes(max_model_len: int) -> list[tuple[list[list[int]], str]]:
    """Probes between the long-context grid points (8256, 32832, max_model_len)."""
    p: list[tuple[list[list[int]], str]] = []
    for context in (20000, 45000, 90000, 120000):
        for batch in (1, 4, 32, 100):
            p.append(([[1, context]] * batch, "long_decode_uniform"))
    for batch in (8, 32, 64):
        p.append((_log_spread(batch, 300, max_model_len - 1000), "long_decode_ragged"))
    for chunk, prefix, decodes in [
        (2016, 20000, 32),
        (2016, 64000, 32),
        (2016, 100000, 32),
        (2016, 128000, 32),
        (512, 100000, 8),
        (128, 60000, 16),
    ]:
        pairs = _log_spread(decodes, 300, max_model_len - 1000)
        pairs.append([chunk, prefix + chunk])
        p.append((pairs, "long_mixed"))
    for chunk, prefix in [(2048, 45000), (2048, 90000), (1000, 120000)]:
        p.append(([[chunk, prefix + chunk]], "long_chunk"))
    return p


def stage_probes(out: Path, capture: Path | None, max_model_len: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if max_model_len > 8192:
        generated = long_context_probes(max_model_len)
    else:
        generated = synthetic_probes()
    if capture is not None:
        for entry in json.loads(capture.read_text()):
            generated.append((entry["pairs"], f"capture_{entry['label']}"))
    probes, specs = [], []
    for is_local in (True, False):
        dims = {
            k: v
            for k, v in config(is_local, max_model_len).items()
            if k not in cf._NON_DIM_KEYS
        }
        for pairs, region in generated:
            probes.append(
                {
                    "is_local": is_local,
                    "max_model_len": max_model_len,
                    "region": region,
                    "pairs": pairs,
                }
            )
            specs.append({**dims, "query_context_pairs": pairs})
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


def _describe(pairs: list[list[int]]) -> str:
    decodes = [c for q, c in pairs if q == 1]
    prefill = [f"({q},{c})" for q, c in pairs if q > 1]
    parts = []
    if decodes:
        parts.append(f"{len(decodes)}dec[{min(decodes)}..{max(decodes)}]")
    return "+".join(parts + prefill)


def stage_report(out: Path, truth_path: Path, sim_bin: str) -> None:
    probes = json.loads((out / "probes.json").read_text())
    truth = json.loads(truth_path.read_text())["results"]
    assert len(truth) == len(probes), (len(truth), len(probes))
    rows = []
    for is_local in (True, False):
        idx = [i for i, p in enumerate(probes) if p["is_local"] == is_local]
        if not idx:
            continue
        max_model_len = probes[idx[0]].get("max_model_len", 8192)
        points = [{"query_context_pairs": probes[i]["pairs"]} for i in idx]
        results = cf.eval_kind(sim_bin, KIND, config(is_local, max_model_len), points)
        for i, r in zip(idx, results):
            t = truth[i]
            tms = t.get("metrics", {}).get("time_ms") if t.get("status") == "ok" else None
            rows.append(
                {
                    "layer": "local" if is_local else "global",
                    "region": probes[i]["region"],
                    "batch": _describe(probes[i]["pairs"]),
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
    for layer in ("local", "global"):
        for region in dict.fromkeys(r["region"] for r in ok):
            sel = [
                r["ratio_cache_truth"] for r in ok if r["layer"] == layer and r["region"] == region
            ]
            if sel:
                lines.append(f"{layer:<6} {region:<22} {_stat(sel)}")
    for r in rows:
        lines.append(
            f"{r['layer']:<6} {r['region']:<22} {r['batch']:<34} truth={r['truth_us']} "
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
    ap.add_argument("--truth", type=Path)
    ap.add_argument("--capture-probes", type=Path)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--sim-bin", default="target/release/simulator")
    args = ap.parse_args()
    if args.stage == "probes":
        stage_probes(args.out_dir, args.capture_probes, args.max_model_len)
    else:
        stage_report(args.out_dir, args.truth, args.sim_bin)


if __name__ == "__main__":
    main()
