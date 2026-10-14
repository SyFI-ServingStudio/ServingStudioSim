"""Steady-state table for closed-loop runs: arrivals in [lo, hi) hours, plus per-hour drift.

drift columns: max - min over the hourly buckets of the window (p50 ms, hit points, rounds/s %).
"""
import json, sys
from pathlib import Path
import pyarrow.parquet as pq

root = Path(sys.argv[1]); lo, hi = float(sys.argv[2]), float(sys.argv[3])
H = 3600e3
rows = []
for summary in sorted(root.glob("**/summary.json")):
    run = summary.parent
    d = pq.read_table(run / "raw/request_slo.parquet", columns=["arrival_time_ms", "completed", "ttft_ms", "declared_prefix_tokens", "prefix_cache_hit_tokens", "fresh_prompt_tokens"]).to_pydict()
    hours = {}
    for i, a in enumerate(d["arrival_time_ms"]):
        if lo * H <= a < hi * H and d["completed"][i]:
            h = hours.setdefault(int(a // H), dict(k=0, dec=0, fr=0, hit=0, t=[]))
            h["k"] += 1; h["dec"] += d["declared_prefix_tokens"][i]; h["fr"] += d["fresh_prompt_tokens"][i]
            h["hit"] += d["prefix_cache_hit_tokens"][i]; h["t"].append(d["ttft_ms"][i])
    pick = lambda q, v: sorted(v)[min(len(v) - 1, int(q * len(v)))]
    allt = [t for h in hours.values() for t in h["t"]]
    dec = sum(h["dec"] for h in hours.values()); fr = sum(h["fr"] for h in hours.values()); hit = sum(h["hit"] for h in hours.values())
    per = [(h["k"] / 3600, 100 * h["hit"] / (h["dec"] + h["fr"]), pick(.5, h["t"])) for h in hours.values()]
    tier, load = run.parent.name, run.name
    rows.append(dict(tier=tier, conc=int(load[1:]), rounds_s=sum(h["k"] for h in hours.values()) / ((hi - lo) * 3600),
        ideal=100 * dec / (dec + fr), hit=100 * hit / (dec + fr), p50=pick(.5, allt), p90=pick(.9, allt), p99=pick(.99, allt),
        computed_ktok_s=(dec + fr - hit) / ((hi - lo) * 3600) / 1e3,
        served_ktok_s=(dec + fr) / ((hi - lo) * 3600) / 1e3,
        d_p50=max(p[2] for p in per) - min(p[2] for p in per), d_hit=max(p[1] for p in per) - min(p[1] for p in per),
        d_rate=100 * (max(p[0] for p in per) - min(p[0] for p in per)) / (sum(p[0] for p in per) / len(per))))
rows.sort(key=lambda r: (r["conc"], ["hbm", "dram", "dram_ssd", "dram_ssd20"].index(r["tier"]) if r["tier"] in ("hbm", "dram", "dram_ssd", "dram_ssd20") else 9))
print(f"{'tier':9}{'conc':>6}{'rounds/s':>9}{'ideal':>7}{'hit':>7}{'gap':>6}{'p50':>6}{'p90':>6}{'p99':>7}{'ktok/s':>8}{'served':>8}{'dp50':>6}{'dhit':>6}{'drate%':>7}")
for r in rows:
    print(f"{r['tier']:9}{r['conc']:>6}{r['rounds_s']:9.1f}{r['ideal']:7.2f}{r['hit']:7.2f}{r['ideal']-r['hit']:6.2f}{r['p50']:6.0f}{r['p90']:6.0f}{r['p99']:7.0f}{r['computed_ktok_s']:8.1f}{r['served_ktok_s']:8.0f}{r['d_p50']:6.0f}{r['d_hit']:6.2f}{r['d_rate']:7.1f}")
if len(sys.argv) > 4:
    Path(sys.argv[4]).write_text(json.dumps(rows, indent=1))
