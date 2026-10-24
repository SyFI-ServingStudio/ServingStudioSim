"""Cache fidelity for TensorFold's DFlash2 glue: a Recipe-B caller.

Covers three kinds whose backends run in `tensorfold_env`:

- `sliding_window_kv_append` (`tensorfold_triton`): Input = call totals
  (streams, context rows, new rows), cached over (streams, mean kept rows);
- `sliding_block_attention` (`tensorfold_triton`): Input = each stream's
  context keys, cached over the dominant uniform batch (streams, keys);
- `logits_topk` (`tensorfold_torch`): Input = rows, at the drafter head's and
  target sampling's (columns, k);
- `rms_norm` (`tensorfold_torch`): Input = rows, at the drafter's hidden 5120.

A profile row gives every stream the same lengths, but a real call is ragged,
so each probe carries the per-stream lengths. Truths are measured in one
process under the TensorFold interpreter, with the runners' own operands and
timing boundary (`Timer.cupti` filtered to the one launch, or every launch for
top-k); uniform probes go through the registered runner function itself.

Probe regions: `recheck` (grid knots, re-measured in this session: drift),
`uniform` (off-knot, every stream alike), `ragged`, `prefill` (append only:
one or several streams with a fresh 2,048-row prefill tap), `pieces` (more
streams than the grid).

Stages, from the repo root:

    uv run python tools/cache-fidelity-analyzer/draftglue_fidelity.py probes --out-dir DIR
    flock .../gpu.lock env PYTHONPATH=$PWD:$PWD/alignment/profiler/tensorfold/src \
        alignment/profiler/tensorfold/.venv/bin/python \
        tools/cache-fidelity-analyzer/draftglue_fidelity.py truth --out-dir DIR
    uv run python tools/cache-fidelity-analyzer/draftglue_fidelity.py report --out-dir DIR

`report` evaluates the Rust cache through `simulator kernel-query eval`, which
reads the grid rows from profile.db; profile the grids first.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path

WITHIN = 1.15  # cache_fidelity.WITHIN; kept here so `truth` runs without it.
WINDOW = 2047

APPEND_CFG = {
    "backends": ["tensorfold_triton"],
    "gpu_name": "NVIDIA GB10",
    "num_kv_heads": 8,
    "head_dim": 128,
    "dtype": "bf16",
    "window": WINDOW,
}
BATTN_CFG = {
    "backends": ["tensorfold_triton"],
    "gpu_name": "NVIDIA GB10",
    "num_qo_heads": 32,
    "num_kv_heads": 8,
    "head_dim": 128,
    "dtype": "bf16",
    "block_rows": 16,
    "window": WINDOW,
    "causal": False,
}
RMSNORM_CFG = {
    "backends": ["tensorfold_torch"],
    "gpu_name": "NVIDIA GB10",
    "hidden": 5120,
    "dtype": "bf16",
}
TOPK_CFGS = {
    "drafter": {
        "backends": ["tensorfold_torch"],
        "gpu_name": "NVIDIA GB10",
        "num_columns": 98592,
        "top_k": 16,
        "dtype": "fp32",
    },
    "sampling": {
        "backends": ["tensorfold_torch"],
        "gpu_name": "NVIDIA GB10",
        "num_columns": 248320,
        "top_k": 28,
        "dtype": "fp32",
    },
}


def probes() -> list[dict]:
    out: list[dict] = []
    rng = random.Random(7)

    def append(ctx: list[int], new: list[int], region: str) -> None:
        out.append({"kind": "sliding_window_kv_append", "ctx": ctx, "new": new, "region": region})

    def battn(ctx: list[int], region: str) -> None:
        out.append({"kind": "sliding_block_attention", "ctx": ctx, "region": region})

    # sliding_window_kv_append: per stream (old context rows, new rows).
    for s, kept in [(1, 2047), (8, 2047), (16, 256), (4, 64), (2, 1024)]:
        new = min(kept, 16)
        append([kept - new] * s, [new] * s, "recheck")
    for s, kept, new in [
        (3, 2047, 1),
        (5, 2047, 9),
        (6, 1800, 16),
        (10, 700, 4),
        (14, 1300, 16),
        (3, 100, 16),
        (7, 30, 8),
        (12, 2047, 2),
        (16, 1800, 16),
        (9, 400, 12),
    ]:
        append([kept - new] * s, [new] * s, "uniform")
    for s in (4, 8, 16):
        append([2047 - 6, 300] * (s // 2), [6, 6] * (s // 2), "ragged")
    for s in (3, 8, 13):
        ctx = [rng.randint(0, 2047) for _ in range(s)]
        append([min(c, 2047 - 1) for c in ctx], [rng.randint(1, 16) for _ in range(s)], "ragged")
    append([0], [2048], "prefill")
    append([0, 0, 0], [1000, 2048, 300], "prefill")
    append([2047 - 16] * 7 + [0], [16] * 7 + [2048], "prefill")
    append([2031] * 20, [16] * 20, "pieces")

    # sliding_block_attention: per stream context keys.
    for s, ctx in [(1, 2047), (8, 2047), (16, 64), (4, 512), (6, 1024)]:
        battn([ctx] * s, "recheck")
    for s, ctx in [
        (3, 2047),
        (5, 1300),
        (7, 700),
        (10, 2047),
        (14, 128),
        (3, 30),
        (11, 1800),
        (16, 1700),
        (2, 400),
        (13, 2047),
    ]:
        battn([ctx] * s, "uniform")
    for s in (4, 8, 16):
        battn([2047, 256] * (s // 2), "ragged")
    for s in (3, 8, 13):
        battn([rng.randint(1, 2047) for _ in range(s)], "ragged")
    battn([2047] + [16] * 7, "ragged")
    battn([2047] * 20, "pieces")

    # logits_topk at both call sites: rows.
    for site in TOPK_CFGS:
        for rows in (1, 16, 128, 256):
            out.append({"kind": "logits_topk", "site": site, "rows": rows, "region": "recheck"})
        for rows in (3, 12, 24, 45, 88, 120, 200, 300, 600, 1000, 2500):
            out.append({"kind": "logits_topk", "site": site, "rows": rows, "region": "uniform"})

    # rms_norm at the drafter's rows (16 a stream; 15 a stream for the final norm).
    for rows in (32, 64, 256):
        out.append({"kind": "rms_norm", "rows": rows, "region": "recheck"})
    for rows in (1, 15, 16, 45, 80, 112, 120, 200, 240):
        out.append({"kind": "rms_norm", "rows": rows, "region": "uniform"})
    return out


def _config(probe: dict) -> dict:
    if probe["kind"] == "sliding_window_kv_append":
        return APPEND_CFG
    if probe["kind"] == "sliding_block_attention":
        return BATTN_CFG
    if probe["kind"] == "rms_norm":
        return RMSNORM_CFG
    return TOPK_CFGS[probe["site"]]


def _input(probe: dict) -> dict:
    if probe["kind"] == "sliding_window_kv_append":
        return {
            "batch_size": len(probe["ctx"]),
            "context_len": sum(probe["ctx"]),
            "num_tokens": sum(probe["new"]),
        }
    if probe["kind"] == "sliding_block_attention":
        return {"context_lens": probe["ctx"]}
    if probe["kind"] == "rms_norm":
        return {"m": probe["rows"]}
    return {"num_rows": probe["rows"]}


def _describe(probe: dict) -> str:
    if probe["kind"] == "logits_topk":
        return f"{probe['site']} rows={probe['rows']}"
    if probe["kind"] == "rms_norm":
        return f"rows={probe['rows']}"
    ctx = probe["ctx"]
    text = f"{len(ctx)}x ctx" + (f"={ctx[0]}" if len(set(ctx)) == 1 else f"[{min(ctx)}..{max(ctx)}]")
    if "new" in probe:
        new = probe["new"]
        text += " new" + (f"={new[0]}" if len(set(new)) == 1 else f"[{min(new)}..{max(new)}]")
    return text


def stage_probes(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    generated = probes()
    (out / "probes.json").write_text(json.dumps(generated, indent=1))
    print(len(generated), "probes ->", out / "probes.json")


def _measure_append(torch, append, ctx: list[int], new: list[int]) -> float:
    from profiling.profilers.timer import Timer

    heads, dim = APPEND_CFG["num_kv_heads"], APPEND_CFG["head_dim"]
    gen = torch.Generator(device="cuda").manual_seed(0)
    fresh = torch.randn(heads, sum(new), dim, generator=gen, device="cuda").to(torch.bfloat16)
    olds = [
        None
        if n == 0
        else torch.randn(heads, n, dim, generator=gen, device="cuda").to(torch.bfloat16)
        for n in ctx
    ]
    torch.cuda.synchronize()
    return Timer.cupti(lambda: append(fresh, olds, new, WINDOW), kernel_name="_append")


def _measure_battn(torch, block_attention, ctx: list[int]) -> float:
    from profiling.profilers.timer import Timer

    cfg = BATTN_CFG
    rows = len(ctx) * cfg["block_rows"]
    gen = torch.Generator(device="cuda").manual_seed(0)

    def bf16(*shape):
        return torch.randn(*shape, generator=gen, device="cuda").to(torch.bfloat16)

    q = bf16(cfg["num_qo_heads"], rows, cfg["head_dim"])
    k = bf16(cfg["num_kv_heads"], rows, cfg["head_dim"])
    v = bf16(cfg["num_kv_heads"], rows, cfg["head_dim"])
    keys = [bf16(cfg["num_kv_heads"], n, cfg["head_dim"]) for n in ctx]
    values = [bf16(cfg["num_kv_heads"], n, cfg["head_dim"]) for n in ctx]
    scale = cfg["head_dim"] ** -0.5
    torch.cuda.synchronize()
    return Timer.cupti(
        lambda: block_attention(
            q, k, v, keys, values, cfg["block_rows"], WINDOW, scale, cfg["causal"]
        ),
        kernel_name="_block_attention",
    )


def stage_truth(out: Path) -> None:
    """Measure every probe in one session; run under the TensorFold interpreter."""

    import torch
    from tensorfold.families.qwen3_5.cuda.draft_attention import append, block_attention

    from profiling.runners.norm.rms_norm_tensorfold_torch import profile_rms_norm_tensorfold_torch
    from profiling.runners.sampling.logits_topk import profile_logits_topk_tensorfold_torch

    truth = {}
    for index, probe in enumerate(json.loads((out / "probes.json").read_text())):
        if probe["kind"] == "sliding_window_kv_append":
            ms = _measure_append(torch, append, probe["ctx"], probe["new"])
        elif probe["kind"] == "sliding_block_attention":
            ms = _measure_battn(torch, block_attention, probe["ctx"])
        elif probe["kind"] == "rms_norm":
            ms = profile_rms_norm_tensorfold_torch(
                probe["rows"], RMSNORM_CFG["hidden"], RMSNORM_CFG["dtype"]
            ).time_ms
        else:
            cfg = TOPK_CFGS[probe["site"]]
            ms = profile_logits_topk_tensorfold_torch(
                probe["rows"], cfg["num_columns"], cfg["top_k"], cfg["dtype"]
            ).time_ms
        truth[index] = ms
        torch.cuda.empty_cache()
        print(index, probe["kind"], _describe(probe), round(ms, 4), flush=True)
    (out / "truth.json").write_text(json.dumps(truth))


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


def stage_report(out: Path, sim_bin: str) -> None:
    import cache_fidelity as cf  # the generic core (sibling file)

    generated = json.loads((out / "probes.json").read_text())
    truth = json.loads((out / "truth.json").read_text())
    cache: dict[int, dict] = {}
    groups: dict[str, list[int]] = {}
    for index, probe in enumerate(generated):
        groups.setdefault(json.dumps([probe["kind"], _config(probe)]), []).append(index)
    for key, indices in groups.items():
        kind, config = json.loads(key)
        results = cf.eval_kind(sim_bin, kind, config, [_input(generated[i]) for i in indices])
        cache.update(zip(indices, results))
    rows = []
    for index, probe in enumerate(generated):
        tms = truth.get(str(index))
        r = cache[index]
        rows.append(
            {
                "kind": probe["kind"],
                "region": probe["region"],
                "probe": _describe(probe),
                "truth_ms": round(tms, 4) if tms else "",
                "cache_ms": round(r["time_ms"], 4),
                "ratio_cache_truth": round(r["time_ms"] / tms, 4) if tms else "",
                "coverage": r["coverage"],
            }
        )
    with (out / "fidelity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["ratio_cache_truth"] != ""]
    lines = [f"bar=+/-{(WITHIN - 1) * 100:.0f}%"]
    for kind in dict.fromkeys(r["kind"] for r in ok):
        lines.append(f"{kind}: ALL {_stat([r['ratio_cache_truth'] for r in ok if r['kind'] == kind])}")
        for region in dict.fromkeys(r["region"] for r in ok if r["kind"] == kind):
            sel = [r["ratio_cache_truth"] for r in ok if r["kind"] == kind and r["region"] == region]
            lines.append(f"  {region:<9} {_stat(sel)}")
    for r in rows:
        lines.append(
            f"{r['kind']:<25} {r['region']:<8} {r['probe']:<36} truth={r['truth_ms']} "
            f"cache={r['cache_ms']} ratio={r['ratio_cache_truth']} cov={r['coverage']}"
        )
    text = "\n".join(lines)
    (out / "fidelity_summary.txt").write_text(text + "\n")
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["probes", "truth", "report"])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--sim-bin", default="target/release/simulator")
    args = ap.parse_args()
    if args.stage == "probes":
        stage_probes(args.out_dir)
    elif args.stage == "truth":
        stage_truth(args.out_dir)
    else:
        stage_report(args.out_dir, args.sim_bin)


if __name__ == "__main__":
    main()
