"""`paged_attn_batch_decode` / `paged_attn_batch_prefill` cache fidelity: a Recipe-B caller.

Why a dedicated caller: both caches are built on derived axes (decode:
batch, longest context and fill; prefill: query tiles and keys per tile) that
the Rust Input projects to, and the backend runs in
the vLLM environment. Ground truth comes from the public profiling CLI with
`--fresh` (measured, never written to a DB) at the canonical batch of each
off-grid probe (prefill: an even split of new tokens over requests sharing a
prefix, so partial tiles and uneven tile counts are probed); this file only
places probes and compares.

`--ragged FILE` adds batches measured elsewhere whose context is spread unlike
the canonical one (JSON lines of `{"op": "decode", "lengths": [...], "ms"}` or
`{"op": "context", "prefixes": [...], "appends": [...], "ms"}`). They show the
projection error on top of the interpolation error.

Two stages, run from the repo root under uv:

    uv run python tools/cache-fidelity-analyzer/paged_attn_batch_fidelity.py \
        probes --config CFG.json --out-dir DIR
    uv run python -m launcher kernel-profile run paged_attn_batch_decode \
        --backend flashinfer_trtllm_gen --specs DIR/decode_specs.json --fresh --json \
        > DIR/decode_truth.json
    (the same for prefill)
    uv run python tools/cache-fidelity-analyzer/paged_attn_batch_fidelity.py \
        report --config CFG.json --out-dir DIR [--ragged FILE]

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

DECODE = "paged_attn_batch_decode"
PREFILL = "paged_attn_batch_prefill"
# The runners' cap on one batch's KV bytes.
MAX_KV_BYTES = 32 << 30

# Off-grid knots, each between two grid knots of its axis.
DECODE_BATCHES = (3, 11, 15, 21, 28, 45, 80, 150, 220)
DECODE_CONTEXTS = (600, 2500, 10000, 50000, 100000)
DECODE_FILLS = (0.1, 0.7)
PREFILL_SEQS = (1, 2, 5, 10, 24)
PREFILL_TOKENS = (100, 700, 1300, 2500, 5000, 7000)
PREFILL_PREFIXES = (0, 1000, 5000, 20000, 60000, 110000)


def _kv_bytes_per_token(cfg: dict) -> int:
    return 2 * cfg["num_kv_heads"] * cfg["head_dim"]  # FP8 cache


def _decode_input(lengths: list[int]) -> dict:
    return {"batch_size": len(lengths), "max_kv_len": max(lengths), "total_kv_len": sum(lengths)}


def _prefill_input(prefixes: list[int], appends: list[int]) -> dict:
    return {"requests": [[p, a] for p, a in zip(prefixes, appends)]}


def _canonical_appends(seqs: int, tokens: int) -> list[int]:
    base, extra = divmod(tokens, seqs)
    return [base + (1 if i < extra else 0) for i in range(seqs)]


def decode_probes(cfg: dict) -> list[dict]:
    out = []
    for batch in DECODE_BATCHES:
        for context in DECODE_CONTEXTS:
            for fill in DECODE_FILLS if batch > 1 else (0.0,):
                total = context + round(fill * (batch - 1) * context)
                if total * _kv_bytes_per_token(cfg) > MAX_KV_BYTES:
                    continue
                spec = {"batch_size": batch, "max_kv_len": context, "total_kv_len": total}
                out.append({"spec": spec, "input": spec, "region": "decode_offgrid"})
    return out


def prefill_probes(cfg: dict) -> list[dict]:
    out = []
    # Every other (tokens, prefix) pair per request count keeps the set small
    # while covering each knot interval.
    for i, seqs in enumerate(PREFILL_SEQS):
        for j, tokens in enumerate(PREFILL_TOKENS):
            for k, prefix in enumerate(PREFILL_PREFIXES):
                if (i + j + k) % 2 or seqs > tokens:
                    continue
                if (seqs * prefix + tokens) * _kv_bytes_per_token(cfg) > MAX_KV_BYTES:
                    continue
                appends = _canonical_appends(seqs, tokens)
                point = _prefill_input([prefix] * seqs, appends)
                spec = point
                out.append({"spec": spec, "input": point, "region": "prefill_offgrid"})
    return out


def _dims(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if k not in cf._NON_DIM_KEYS}


def stage_probes(out: Path, cfg: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, probes in (("decode", decode_probes(cfg)), ("prefill", prefill_probes(cfg))):
        (out / f"{name}_probes.json").write_text(json.dumps(probes))
        specs = [{**_dims(cfg), **probe["spec"]} for probe in probes]
        (out / f"{name}_specs.json").write_text(json.dumps(specs))
        print(len(specs), f"{name} probe specs ->", out / f"{name}_specs.json")


def _ragged(path: Path) -> dict[str, list[dict]]:
    found: dict[str, list[dict]] = {"decode": [], "prefill": []}
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("op") == "decode" and rec.get("lengths"):
            # Zero-length rows are CUDA-graph padding; they count as batch rows.
            found["decode"].append(
                {
                    "input": _decode_input(rec["lengths"]),
                    "truth_ms": rec["ms"],
                    "region": f"decode_ragged_{rec.get('tag', '')}",
                    "label": str(rec["lengths"])[:60],
                }
            )
        elif rec.get("op") == "context" and rec.get("appends"):
            found["prefill"].append(
                {
                    "input": _prefill_input(rec["prefixes"], rec["appends"]),
                    "truth_ms": rec["ms"],
                    "region": f"prefill_ragged_{rec.get('tag', '')}",
                    "label": f"p{rec['prefixes'][:4]} a{rec['appends'][:4]} n{len(rec['appends'])}",
                }
            )
    return found


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


def stage_report(out: Path, cfg: dict, ragged: Path | None, sim_bin: str) -> None:
    extra = _ragged(ragged) if ragged else {"decode": [], "prefill": []}
    rows = []
    for name, kind in (("decode", DECODE), ("prefill", PREFILL)):
        probes = json.loads((out / f"{name}_probes.json").read_text())
        truth = json.loads((out / f"{name}_truth.json").read_text())["results"]
        assert len(truth) == len(probes), (len(truth), len(probes))
        for probe, t in zip(probes, truth):
            ok = t.get("status") == "ok"
            probe["truth_ms"] = t["metrics"]["time_ms"] if ok else None
            probe["label"] = json.dumps(probe["spec"])
        entries = probes + extra[name]
        results = cf.eval_kind(sim_bin, kind, cfg, [e["input"] for e in entries])
        for entry, result in zip(entries, results):
            truth_ms = entry["truth_ms"]
            rows.append(
                {
                    "region": entry["region"],
                    "probe": entry["label"],
                    "truth_us": round(truth_ms * 1e3, 3) if truth_ms else "",
                    "cache_us": round(result["time_ms"] * 1e3, 3),
                    "ratio_cache_truth": round(result["time_ms"] / truth_ms, 4) if truth_ms else "",
                    "coverage": result["coverage"],
                }
            )
    with (out / "fidelity.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ok = [r for r in rows if r["ratio_cache_truth"] != ""]
    lines = [f"bar=+/-{(cf.WITHIN - 1) * 100:.0f}%"]
    for region in dict.fromkeys(r["region"] for r in ok):
        sel = [r["ratio_cache_truth"] for r in ok if r["region"] == region]
        lines.append(f"{region:<34} {_stat(sel)}")
    worst = sorted(ok, key=lambda r: -abs(math.log(r["ratio_cache_truth"])))[:15]
    lines.append("worst probes:")
    for r in worst:
        lines.append(
            f"  {r['region']:<30} {r['probe']:<70} truth={r['truth_us']} "
            f"cache={r['cache_us']} ratio={r['ratio_cache_truth']} cov={r['coverage']}"
        )
    text = "\n".join(lines)
    (out / "fidelity_summary.txt").write_text(text + "\n")
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["probes", "report"])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ragged", type=Path)
    ap.add_argument("--sim-bin", default="target/release/simulator")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.stage == "probes":
        stage_probes(args.out_dir, cfg)
    else:
        stage_report(args.out_dir, cfg, args.ragged, args.sim_bin)


if __name__ == "__main__":
    main()
