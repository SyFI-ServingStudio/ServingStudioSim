"""Collect the all-stacked experiment into results.json + table.md.

Usage: python tools/all-stacked/collect.py <experiment dir>

Reads <exp>/{k7,hs}/<strategy>/c<conc>/ (summary.json, reports/slo_general_report.json,
reports/kernel_time_share_report.json). Point fields match
wt-flash-moe-exchange tools/fast-moe-exchange/collect.py (results_idxfix.json).
The cumulative path picks, per load, the best attention-TP degree (KDA-only TP
b in {1,2,4,8}) for step 4 and keeps that b for step 5 (f_b/g_b merged).
"""
import json
import sys
from pathlib import Path

exp = Path(sys.argv[1])
LOADS = [("k7-c1", "k7", 1), ("k7-c8", "k7", 8), ("k7-c16", "k7", 16), ("k7-c32", "k7", 32),
         ("k7-c64", "k7", 64), ("k7-c128", "k7", 128), ("k7-c256", "k7", 256),
         ("hs-k3-c768", "hs", 768), ("hs-k3-c1280", "hs", 1280)]
COMM_KINDS = {"moe_ep_quantized_all_gather", "moe_ep_reduce_scatter", "moe_alltoall", "moe_alltoall_prepare",
              "ep_comm_analytic", "all_reduce", "all_reduce_fusion"}

MEAS = "measured rows"
STRATEGIES = [
    ("S0_base", "base: indexer-fixed DP8 attention + EP8, pynccl all_gatherv/reduce_scatterv (M0)", "M0", 1, False, False,
     "all measured rows (MoE expert GEMM rows from the October nvfp4_fused_moe batch)"),
    ("S1_M2", "+ M2 fixed-pattern direct-write all-gather + reduce-scatter (MNNVL)", "M2", 1, False, False,
     "EXTRAPOLATED: dispatch/combine from the moe_alltoall MNNVL fit (a 4.11 us, L 638 GB/s), not measured rows"),
    ("S2_M3", "+ M3 fused exchange (dispatch write in the NVFP4 quant, combine read in the next op)", "M3", 1, False, False,
     "EXTRAPOLATED: MNNVL fit minus its 4.11 us fixed cost; kernels do not exist"),
    ("S3_M3_mhc", "+ mHC post/pre fusion (fused boundary minus 11.09 us fixed)", "M3+mHC", 1, True, False,
     "EXTRAPOLATED: mhc_fused_post_pre_rms_norm fit (a 11.09 us + 17.1 ns/row) fixed part removed"),
    ("S4_M3_mhc_ko2", "+ KDA-only TP2", "M3+mHC", 2, True, False,
     "TP2 all-reduce priced from 4-GPU rows (EXTRAPOLATED); KDA kernels at TP2 measured"),
    ("S4_M3_mhc_ko4", "+ KDA-only TP4", "M3+mHC", 4, True, False, "measured 4-GPU all-reduce rows"),
    ("S4_M3_mhc_ko8", "+ KDA-only TP8", "M3+mHC", 8, True, False,
     "measured 8-GPU all-reduce rows (MNNVL <=128 rows, else nccl/nvshmem)"),
    ("S5_M3_mhc_gbm", "+ f_b/g_b as one batched GEMM (TP1)", "M3+mHC", 1, True, True,
     "measured batched_gemm torch_bmm_packed_lhs rows"),
    ("S5_M3_mhc_ko2_gbm", "+ f_b/g_b merged, KDA-only TP2", "M3+mHC", 2, True, True, "as S4 ko2 + measured bmm rows"),
    ("S5_M3_mhc_ko4_gbm", "+ f_b/g_b merged, KDA-only TP4", "M3+mHC", 4, True, True, "as S4 ko4 + measured bmm rows"),
    ("S5_M3_mhc_ko8_gbm", "+ f_b/g_b merged, KDA-only TP8", "M3+mHC", 8, True, True, "as S4 ko8 + measured bmm rows"),
    ("C_M2_mhc", "control: M2 + mHC fusion (regression vs wt-flash-moe-exchange)", "M2+mHC", 1, True, False, "control"),
    ("C_M0_ko2", "control: M0 + KDA-only TP2 (indexer-fixed baseline)", "M0", 2, False, False, "control"),
    ("C_M0_ko4", "control: M0 + KDA-only TP4", "M0", 4, False, False, "control"),
    ("C_M0_ko8", "control: M0 + KDA-only TP8", "M0", 8, False, False, "control"),
]


def point(d, load):
    s = json.loads((d / "summary.json").read_text())
    metrics = json.loads((d / "reports/slo_general_report.json").read_text())["metrics"]
    slo, ttft = metrics["tpot"], metrics["ttft"]
    share = None
    ks = d / "reports/kernel_time_share_report.json"
    if ks.exists():
        pos = json.loads(ks.read_text())["positions"]
        share = sum(p["overall_share_pct"] for p in pos if p["kind"] in COMM_KINDS)
    return {"load": load, "tok_s": round(s["decode_tok_s"], 1), "tpot_ms": round(slo["mean"], 3),
            "tpot_p50_ms": round(slo["p50"], 3), "tpot_p90_ms": round(slo["p90"], 3), "tpot_p99_ms": round(slo["p99"], 3),
            "ttft_p50_ms": round(ttft["p50"], 3), "ttft_p90_ms": round(ttft["p90"], 3), "ttft_p99_ms": round(ttft["p99"], 3),
            "comm_share_pct": None if share is None else round(share, 2),
            "log_dir": str(d.resolve())}


def pct(a, b):
    return round((a / b - 1) * 100, 2)


strategies, by = [], {}
for sid, name, moe, tp, mhc, gbm, note in STRATEGIES:
    pts = {}
    for load, fam, conc in LOADS:
        d = exp / fam / sid / f"c{conc}"
        if (d / "summary.json").exists() and (d / "reports/slo_general_report.json").exists():
            pts[load] = point(d, load)
    by[sid] = pts
    strategies.append({"id": sid, "name": name, "moe": moe, "kda": "K0" if tp == 1 else f"kda_only TP{tp}",
                       "attn_tp": tp, "mhc_fuse": mhc, "kda_gate_b_merged": gbm, "source": "wt-flash-all-stacked",
                       "note": note, "run_label": sid, "points": list(pts.values())})
for s in strategies:
    for p in s["points"]:
        b = by["S0_base"].get(p["load"])
        if b:
            p["tok_s_vs_baseline_pct"] = pct(p["tok_s"], b["tok_s"])
            p["tpot_vs_baseline_pct"] = pct(p["tpot_ms"], b["tpot_ms"])

STEP_IDS = [("0 base (indexer fix)", ["S0_base"]), ("1 + M2", ["S1_M2"]), ("2 M2 -> M3", ["S2_M3"]),
            ("3 + mHC fuse", ["S3_M3_mhc"]),
            ("4 + KDA-only TP (best b)", ["S3_M3_mhc", "S4_M3_mhc_ko2", "S4_M3_mhc_ko4", "S4_M3_mhc_ko8"])]
GBM = {"S3_M3_mhc": "S5_M3_mhc_gbm", "S4_M3_mhc_ko2": "S5_M3_mhc_ko2_gbm",
       "S4_M3_mhc_ko4": "S5_M3_mhc_ko4_gbm", "S4_M3_mhc_ko8": "S5_M3_mhc_ko8_gbm"}
cumulative = []
for load, _, _ in LOADS:
    rows, prev = [], None
    chosen = None
    for step, ids in STEP_IDS:
        cands = [(by[i][load], i) for i in ids if load in by[i]]
        if not cands:
            continue
        p, sid = max(cands, key=lambda c: c[0]["tok_s"])
        if step.startswith("4"):
            chosen = sid
        rows.append(dict(p, step=step, strategy=sid))
    if chosen and load in by.get(GBM[chosen], {}):
        rows.append(dict(by[GBM[chosen]][load], step="5 + f_b/g_b merged (same b)", strategy=GBM[chosen]))
    base = rows[0] if rows else None
    for r in rows:
        r["tok_s_vs_baseline_pct"] = pct(r["tok_s"], base["tok_s"])
        r["tpot_vs_baseline_pct"] = pct(r["tpot_ms"], base["tpot_ms"])
        r["tok_s_vs_prev_pct"] = 0.0 if prev is None else pct(r["tok_s"], prev["tok_s"])
        r["tpot_vs_prev_pct"] = 0.0 if prev is None else pct(r["tpot_ms"], prev["tpot_ms"])
        prev = r
    tp_options = {i: by[i][load]["tok_s"] for i in STEP_IDS[-1][1] if load in by[i]}
    cumulative.append({"load": load, "kda_tp_options_tok_s": tp_options, "steps": rows})

result = {"experiment": str(exp.resolve()),
          "label": "all stacked: indexer fix -> M2 -> M3 -> mHC fuse -> KDA-only TP -> f_b/g_b merge "
                   "(MoE expert GEMM rows from the October nvfp4_fused_moe batch)",
          "strategies": strategies, "cumulative": cumulative, "baseline": "S0_base"}
(exp / "results.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))

f = lambda v, n=1: "-" if v is None else f"{v:,.{n}f}"
sg = lambda v: "-" if v is None else f"{v:+.1f}%"
L = ["# All stacked: GLM-5.3-Flash NVFP4 + DFlash2, 8x B200, attention DP8 + MoE EP8 (sim, --no-gpu)", "",
     f"Source: `{(exp / 'results.json').resolve()}`. Corpus routing. Absolute tok/s uses the October nvfp4_fused_moe rows.", "",
     "## Cumulative path", "",
     "| load | step | strategy | tok/s | vs base | vs prev | TPOT mean ms | p50 | p90 | p99 | TTFT p50 ms | p90 | p99 | comm % |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for c in cumulative:
    for r in c["steps"]:
        L.append(f"| {c['load']} | {r['step']} | {r['strategy']} | {f(r['tok_s'])} | {sg(r['tok_s_vs_baseline_pct'])} | "
                 f"{sg(r['tok_s_vs_prev_pct'])} | {f(r['tpot_ms'],2)} | {f(r['tpot_p50_ms'],2)} | {f(r['tpot_p90_ms'],2)} | "
                 f"{f(r['tpot_p99_ms'],2)} | {f(r['ttft_p50_ms'],1)} | {f(r['ttft_p90_ms'],1)} | {f(r['ttft_p99_ms'],1)} | {f(r['comm_share_pct'])} |")
L += ["", "## Every strategy (tok/s / mean TPOT ms)", "",
      "| id | " + " | ".join(l for l, _, _ in LOADS) + " |", "|---|" + "---|" * len(LOADS)]
for s in strategies:
    pts = {p["load"]: p for p in s["points"]}
    L.append(f"| {s['id']} | " + " | ".join(
        f"{f(pts[l]['tok_s'])} / {f(pts[l]['tpot_ms'],2)}" if l in pts else "-" for l, _, _ in LOADS) + " |")
L += ["", "## Pricing source per step", ""] + [f"- {s['id']}: {s['note']}" for s in strategies]
(exp / "table.md").write_text("\n".join(L) + "\n")
print("\n".join(L[:60]))
