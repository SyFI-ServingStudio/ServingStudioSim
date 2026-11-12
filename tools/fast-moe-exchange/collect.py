"""Collect the fast MoE exchange experiment into table.md + results.json.

Usage: python collect.py <experiment dir> <strategies.json>
strategies.json: [{id, name, moe, ex (run label), note}]. For every strategy it
reads dp8/<ex>/c{128,256} and hs/<ex>/c{768,1280} summaries and SLO reports,
the comm share (kernel_time_share_report positions), and the per-step times of
one KDA+MoE layer at k7 c128 (iterations where all 8 ranks hold 128 rows,
median over them, max over each slot's rank copies).
"""
import json, sys, statistics
from pathlib import Path
import pyarrow.parquet as pq

exp = Path(sys.argv[1])
strategies = json.loads(Path(sys.argv[2]).read_text())
LINK = 900.0  # B200 NVLink5 GB/s per direction (gpu/spec.json 1800 is bidirectional)
H, TOPK, N = 4096, 8, 8
LOADS = [("k7-c128", "dp8", 128), ("k7-c256", "dp8", 256), ("hs-k3-c768", "hs", 768), ("hs-k3-c1280", "hs", 1280)]
COMM_KINDS = {"moe_ep_quantized_all_gather", "moe_ep_reduce_scatter", "moe_alltoall", "moe_alltoall_prepare",
              "ep_comm_analytic", "all_reduce", "all_reduce_fusion"}
FIT = {"mnnvl": "moe_alltoall MNNVL fit (a 4.11 us, L 638 GB/s)", "nvshmem": "nvshmem fit (a 15.0 us, L 729 GB/s)",
       "lamport": "lamport fit (a 14.9 us, L 495 GB/s)"}


def run_dir(load, ex, conc):
    return exp / load / ex / f"c{conc}"


def point(d, load):
    s = json.loads((d / "summary.json").read_text())
    slo = json.loads((d / "reports/slo_general_report.json").read_text())["metrics"]["tpot"]
    share = None
    ks = d / "reports/kernel_time_share_report.json"
    if ks.exists():
        pos = json.loads(ks.read_text())["positions"]
        share = sum(p["overall_share_pct"] for p in pos if p["kind"] in COMM_KINDS)
        exch = sum(p["overall_share_pct"] for p in pos if ".moe." in p["name"] and p["kind"] == "elementwise"
                   and any(k in p["name"] for k in ("swizzle", "combine_output_fill", "combine_reduce", "combine_sum")))
    return {"load": load, "tok_s": round(s["decode_tok_s"], 1), "tpot_ms": round(slo["mean"], 3),
            "tpot_p50_ms": round(slo["p50"], 3), "tpot_p90_ms": round(slo["p90"], 3), "tpot_p99_ms": round(slo["p99"], 3),
            "comm_share_pct": None if share is None else round(share, 2),
            "exchange_elementwise_share_pct": None if share is None else round(exch, 2),
            "log_dir": str(d.resolve())}


def steps(d):
    man = json.loads(next((d / "raw/cost_manifest").glob("*.json")).read_text())["sections"][0]["slots"]
    t = pq.read_table(next((d / "raw/cost_log").glob("*.parquet")),
                      columns=["groups", "slot_time_ms", "slot_coverage", "slot_input", "slot_backend"])
    idx = {}
    for i, s in enumerate(man):
        if s["name"].startswith("unified.kda_moe."):
            idx.setdefault(s["name"], []).append(i)
    picked = []
    for batch in t.to_batches(20000):
        for r in batch.to_pylist():
            g = r["groups"]
            if len(g) == N and all(x["batch_tokens"] == 128 and x["prefill_tokens"] == 0 for x in g):
                picked.append(r)
        if len(picked) >= 400:
            break
    out = []
    for name, slots in idx.items():
        kind = man[slots[0]]["kind"]
        times, cov, inp, be = [], 0, None, None
        for r in picked:
            vals = [(r["slot_time_ms"][i], i) for i in slots]
            v, i = max(vals)
            times.append(v * 1e3)
            cov |= r["slot_coverage"][i]
            inp = r["slot_input"][i]
            be = r["slot_backend"][i]
        us = statistics.median(times)
        cfg = man[slots[0]]["kernel_config"]
        backends = cfg.get("backends") or []
        backend = backends[be] if backends and be is not None and be < len(backends) else None
        short = name.split(".", 2)[2]
        rec = {"name": short, "kind": kind, "us": round(us, 2), "copies": len(slots), "extrapolated": bool(cov & 1)}
        group = "compute"
        try:
            j = json.loads(inp)
        except Exception:
            j = {}
        if kind in COMM_KINDS:
            group = "prepare" if "prepare" in short else "comm"
        if kind == "ep_comm_analytic":
            fast = cfg.get("fast") or {}
            op = cfg.get("op")
            backend = "analytic: " + (FIT[fast["backend"]] if fast else f"ep_comm {cfg.get('kind')}")
            rec["extrapolated"] = True
            send, recv, nbytes = j.get("max_send_rows", 0), j.get("max_recv_rows", 0), j.get("bytes", 0)
            rows = max(send, recv)
            if fast and op == "combine" and fast.get("combine") == "all_reduce":
                rows = recv  # two legs, reported as one
            bpr = nbytes / rows if rows else 0
            rec.update(rows=round(rows, 1), bytes_per_row=round(bpr, 1), send_bytes=round(send * bpr),
                       recv_bytes=round(recv * bpr), latency_us=j.get("latency_us"))
        elif kind == "moe_ep_quantized_all_gather":
            bpr = H // 2 + H // 16 + 8 * TOPK
            rec.update(rows=7 * 128, bytes_per_row=bpr, send_bytes=7 * 128 * bpr, recv_bytes=7 * 128 * bpr)
        elif kind == "moe_ep_reduce_scatter":
            rec.update(rows=7 * 128, bytes_per_row=2 * H, send_bytes=7 * 128 * 2 * H, recv_bytes=7 * 128 * 2 * H)
        elif kind == "all_reduce":
            msg = j["message_size_bytes"]
            rec.update(rows=msg // (2 * H), bytes_per_row=2 * H, send_bytes=round(2 * 7 / 8 * msg),
                       recv_bytes=round(2 * 7 / 8 * msg), note="two-shot bytes 2(N-1)/N x message")
        elif kind == "all_reduce_fusion":
            msg = j["num_tokens"] * 2 * H
            rec.update(rows=j["num_tokens"], bytes_per_row=2 * H, send_bytes=round(2 * 7 / 8 * msg),
                       recv_bytes=round(2 * 7 / 8 * msg), note="two-shot bytes 2(N-1)/N x message")
        elif kind in ("moe_alltoall", "moe_alltoall_prepare"):
            rec.update(input=j)
        if group != "compute" and "send_bytes" in rec:
            moved = max(rec["send_bytes"], rec["recv_bytes"])
            rec["gb_per_s"] = round(moved / (us * 1e3), 1) if us else None
            rec["link_util_pct"] = round(100 * moved / (us * 1e3) / LINK, 1) if us else None
            rec["bw_floor_us"] = round(moved / (LINK * 1e3), 2)
        rec.update(group=group, backend=backend)
        out.append(rec)
    return out, len(picked)


result = {"experiment": str(exp.resolve()), "link_gb_per_s_one_way": LINK, "strategies": [], "comm_utilization": []}
for st in strategies:
    pts = []
    for load, fam, conc in LOADS:
        d = run_dir(fam, st["ex"], conc)
        if (d / "summary.json").exists() and (d / "reports/slo_general_report.json").exists():
            pts.append(point(d, load))
    stp, n = steps(run_dir("dp8", st["ex"], 128)) if (run_dir("dp8", st["ex"], 128) / "raw/cost_log").exists() else ([], 0)
    entry = {"id": f"K0+{st['moe']}", "name": st["name"], "moe": st["moe"], "kda": "K0", "source": "sim",
             "note": st.get("note", ""), "run_label": st["ex"], "steps_iterations": n,
             "steps": [{k: s.get(k) for k in ("name", "group", "us", "rows", "bytes_per_row", "backend", "extrapolated")} for s in stp],
             "points": pts}
    result["strategies"].append(entry)
    for s in stp:
        if s["group"] != "compute" and "send_bytes" in s:
            result["comm_utilization"].append({"strategy": entry["id"], "variant": st["ex"], **{k: s.get(k) for k in (
                "name", "rows", "bytes_per_row", "send_bytes", "recv_bytes", "us", "gb_per_s", "link_util_pct",
                "bw_floor_us", "backend", "extrapolated", "latency_us", "note")}})
(exp / "results.json").write_text(json.dumps(result, indent=1))

L = [f"# Fast MoE exchange — GLM-5.3-Flash NVFP4 DP8+EP8 DFlash2 (sim, --no-gpu)\n",
     f"Source: `{(exp / 'results.json').resolve()}`. Link = {LINK:.0f} GB/s per direction.\n",
     "## Throughput / mean TPOT\n",
     "| id | variant | " + " | ".join(f"{l} tok/s | TPOT ms | comm %" for l, _, _ in LOADS) + " |",
     "|" + "---|" * (2 + 3 * len(LOADS))]
for e in result["strategies"]:
    cells = []
    by = {p["load"]: p for p in e["points"]}
    for l, _, _ in LOADS:
        p = by.get(l)
        cells += [f"{p['tok_s']:,.1f}", f"{p['tpot_ms']:.2f}", "-" if p["comm_share_pct"] is None else f"{p['comm_share_pct']:.1f}"] if p else ["-"] * 3
    L.append(f"| {e['id']} | {e['run_label']} | " + " | ".join(cells) + " |")
L += ["\n## Comm steps, k7 c128, one KDA+MoE layer, busiest rank\n",
      "| variant | step | rows | B/row | send B | recv B | us | GB/s | link % | BW floor us | extrap | backend |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for c in result["comm_utilization"]:
    L.append(f"| {c['variant']} | {c['name']} | {c['rows']} | {c['bytes_per_row']} | {c['send_bytes']:,} | {c['recv_bytes']:,} | "
             f"{c['us']} | {c['gb_per_s']} | {c['link_util_pct']} | {c['bw_floor_us']} | {c['extrapolated']} | {c['backend']} |")
L += ["\n## All steps per variant (k7 c128, KDA+MoE layer, us)\n"]
for e in result["strategies"]:
    L.append(f"### {e['id']} ({e['run_label']}) — {e['name']}\n")
    L.append("| step | group | us | extrap | backend |\n|---|---|---|---|---|")
    for s in e["steps"]:
        L.append(f"| {s['name']} | {s['group']} | {s['us']} | {s['extrapolated']} | {s['backend']} |")
    tot = {g: round(sum(s["us"] for s in e["steps"] if s["group"] == g), 2) for g in ("compute", "comm", "prepare")}
    L.append(f"\nSum of leaf times by group (not the critical path; the shared expert overlaps): {tot}\n")
(exp / "table.md").write_text("\n".join(L) + "\n")
print("\n".join(L[:12 + len(result['strategies'])]))
