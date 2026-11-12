"""`compressed_sparse_mla_rope_cast` long-context cache fidelity: a Recipe-B caller.

Why a dedicated caller: the Rust Input is a ragged `query_context_pairs` list,
which `run_fidelity` cannot put in its integer CSV columns, and the backend
(`flashmla_mega`) runs in the vLLM fork environment. So ground truth comes from
the public profiling CLI with `--fresh` (measured, never written to a DB), and
this file only places physical probes and compares.

The probes target the DeepSeek-V4.1 default `max_model_len` (1048576): prefill
chunks at contexts between the gather-area planes (200K..900K), a two-request
chunk, chunks at 1M, and decode at 1M context (decode keys saturate, so these
check that the cache is flat in context).

Two stages, run from the repo root under uv on a B200 node:

    uv run python tools/cache-fidelity-analyzer/compressed_sparse_mla_rope_cast_fidelity.py \
        probes --out-dir DIR
    uv run python -m launcher kernel-profile run compressed_sparse_mla_rope_cast \
        --backend flashmla_mega --specs DIR/probe_specs.json --gpu-name "NVIDIA B200" \
        --fresh --json > DIR/truth.json
    uv run python tools/cache-fidelity-analyzer/compressed_sparse_mla_rope_cast_fidelity.py \
        report --out-dir DIR --truth DIR/truth.json

`--backend swapab_cutedsl` checks the swap-AB decode rows instead (16 live
heads, decode on ratio 1 and 2, at the capture's 131072 and at 1048576): off-grid
row counts, contexts between key points, ragged verify-shaped batches (runs of
consecutive positions, as DSpark's 6-row verify writes them) and saturated
long contexts. Pass the same flag to `probes` and `report`, and
`--backend swapab_cutedsl` to `kernel-profile run`.

`report` evaluates the Rust cache through `simulator kernel-query eval`, which
reads the grid rows from profile.db; profile them first (the grid must be
complete, or the eval JIT-profiles through the bridge).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import cache_fidelity as cf  # the generic core (sibling file)

KIND = "compressed_sparse_mla_rope_cast"
BACKEND = "flashmla_mega"
GPU = "NVIDIA B200"
MAX_MODEL_LEN = 1_048_576

# The V4.1 attention config, as the arch builds it (simulator/src/arch/deepseek_v41_vllm.rs).
_DIMS = {
    "window_size": 128,
    "index_topk": 512,
    "num_heads": 64,
    "head_dim": 512,
    "rope_dim": 64,
    "max_model_len": MAX_MODEL_LEN,
    "max_num_batched_tokens": 2048,
    "prefill_chunk_size": 4,
    "q_dtype": "bf16",
    "swa_cache_format": "mxfp8",
    "output_dtype": "fp8_e4m3",
}


SWAPAB = "swapab_cutedsl"
SWAPAB_HEADS = 16


def config(
    mode: str, ratio: int, backend: str = BACKEND, max_model_len: int = MAX_MODEL_LEN
) -> dict:
    swapab = backend == SWAPAB
    return {
        "backends": [backend],
        "gpu_name": GPU,
        "mode": mode,
        "compress_ratio": ratio,
        "compressed_cache_format": "none" if ratio == 0 else "nvfp4",
        **_DIMS,
        "num_heads": SWAPAB_HEADS if swapab else _DIMS["num_heads"],
        "max_model_len": max_model_len,
    }


def verify_rows(requests: int, context: int, width: int = 6) -> list[list[int]]:
    """``requests`` verify blocks of ``width`` one-row decodes at consecutive positions."""
    return [[1, context - width + 1 + i] for _ in range(requests) for i in range(width)]


def swapab_probes(ratio: int, max_model_len: int) -> list[tuple[list[list[int]], str]]:
    p = [([[1, 4096]] * rows, "rows_off") for rows in (3, 5, 7, 9, 11, 13, 15)]
    # Contexts whose keys sit between the key-axis points.
    p += [([[1, c]] * 6, "keys_off") for c in (40, 150, 300, 700)]
    p += [(verify_rows(1, c), "verify_r1") for c in (1000, 20_000, 120_000)]
    p += [(verify_rows(2, 60_000), "verify_r2")]
    p.append(([[1, 200]] * 5 + [[1, 90_000]], "ragged"))
    if max_model_len > 131072:
        p += [(verify_rows(1, c), "verify_long") for c in (500_000, max_model_len)]
        p.append((verify_rows(2, max_model_len), "verify_long"))
    return p


def swapab_configs() -> list[tuple[int, int]]:
    return [(r, m) for r in (1, 2) for m in (131072, MAX_MODEL_LEN)]


def prefill_probes() -> list[tuple[list[list[int]], str]]:
    p = [([[2048, c]], "chunk") for c in (200_000, 400_000, 700_000, 900_000)]
    p.append(([[512, 300_000]], "chunk"))
    p.append(([[1024, 300_000], [1024, 600_000]], "two_request"))
    p.append(([[1024, MAX_MODEL_LEN]], "chunk_1m"))
    p.append(([[2048, MAX_MODEL_LEN]], "chunk_1m"))
    return p


def decode_probes(ratio: int) -> list[tuple[list[list[int]], str]]:
    p = [([[1, MAX_MODEL_LEN]] * 8, "decode_1m")]
    if ratio == 1:
        p.append(([[1, MAX_MODEL_LEN]] * 200, "decode_1m"))
    return p


def configs() -> list[tuple[str, int]]:
    return [("prefill", 1), ("prefill", 2), ("decode", 1), ("decode", 2), ("decode", 0)]


def stage_probes(out: Path, backend: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    probes, specs = [], []
    if backend == SWAPAB:
        for ratio, mml in swapab_configs():
            cfg = config("decode", ratio, SWAPAB, mml)
            dims = {k: v for k, v in cfg.items() if k not in cf._NON_DIM_KEYS}
            for pairs, region in swapab_probes(ratio, mml):
                probes.append(
                    {"mode": "decode", "ratio": ratio, "mml": mml, "region": region, "pairs": pairs}
                )
                specs.append({**dims, "query_context_pairs": pairs})
        (out / "probes.json").write_text(json.dumps(probes))
        (out / "probe_specs.json").write_text(json.dumps(specs))
        print(len(specs), "probe specs ->", out / "probe_specs.json")
        return
    for mode, ratio in configs():
        dims = {k: v for k, v in config(mode, ratio).items() if k not in cf._NON_DIM_KEYS}
        gen = prefill_probes() if mode == "prefill" else decode_probes(ratio)
        for pairs, region in gen:
            probes.append({"mode": mode, "ratio": ratio, "region": region, "pairs": pairs})
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
    med = statistics.median(fin)
    return f"n={len(fin)} med={med:.3f} within={within}/{len(fin)} worst={worst:.3f}"


def _describe(pairs: list[list[int]]) -> str:
    runs: list[list] = []
    for pair in pairs:
        if runs and runs[-1][1] == pair:
            runs[-1][0] += 1
        else:
            runs.append([1, pair])
    return "+".join(f"{n}x({q},{c})" if n > 1 else f"({q},{c})" for n, (q, c) in runs)


def stage_report(out: Path, truth_path: Path, sim_bin: str, backend: str) -> None:
    probes = json.loads((out / "probes.json").read_text())
    truth = json.loads(truth_path.read_text())["results"]
    assert len(truth) == len(probes), (len(truth), len(probes))
    rows = []
    groups = (
        [("decode", r, m) for r, m in swapab_configs()]
        if backend == SWAPAB
        else [(mode, ratio, MAX_MODEL_LEN) for mode, ratio in configs()]
    )
    for mode, ratio, mml in groups:
        idx = [
            i
            for i, p in enumerate(probes)
            if p["mode"] == mode and p["ratio"] == ratio and p.get("mml", MAX_MODEL_LEN) == mml
        ]
        points = [{"query_context_pairs": probes[i]["pairs"]} for i in idx]
        results = cf.eval_kind(sim_bin, KIND, config(mode, ratio, backend, mml), points)
        for i, r in zip(idx, results):
            t = truth[i]
            tms = t.get("metrics", {}).get("time_ms") if t.get("status") == "ok" else None
            rows.append(
                {
                    "mode": mode,
                    "ratio": ratio,
                    "region": f"{probes[i]['region']}@{mml}",
                    "pairs": _describe(probes[i]["pairs"]),
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
    for r in rows:
        lines.append(
            f"{r['mode']:<7} r{r['ratio']} {r['region']:<20} {r['pairs']:<26} "
            f"truth={r['truth_us']} cache={r['cache_us']} ratio={r['ratio_cache_truth']} "
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
    ap.add_argument("--sim-bin", default="target/release/simulator")
    ap.add_argument("--backend", default=BACKEND, choices=[BACKEND, SWAPAB])
    args = ap.parse_args()
    if args.stage == "probes":
        stage_probes(args.out_dir, args.backend)
    else:
        stage_report(args.out_dir, args.truth, args.sim_bin, args.backend)


if __name__ == "__main__":
    main()
