"""HBM KV that cannot be evicted (running requests + read holds = active - retained prefixes) over [lo, hi) h.

Time-weighted mean and percentiles of that need as a share of the pool, and of the retained prefixes; plus the
read-hold share from the tier log (read_ms -> admit_ms, declared_tokens).
"""
import sys, json, glob, csv
import pyarrow.parquet as pq
run, lo, hi = sys.argv[1], float(sys.argv[2]) * 3.6e6, float(sys.argv[3]) * 3.6e6
meta = json.load(open(f"{run}/raw/run_meta.json"))
caps = {p["group_id"]: p["capacity_tokens"] for w in meta["workers"] for p in w["kv_pools"]}
f = glob.glob(f"{run}/raw/kv_snapshot/*.parquet")[0]
t = pq.read_table(f).to_pydict()
by = {}
for g, tm, a, r in zip(t["group_id"], t["time_ms"], t["active_kv"], t["retained_prefix_kv"]):
    by.setdefault(g, []).append((tm, a - r, r))
need_w, ret_w, span = [], 0.0, 0.0
for g, s in by.items():
    s.sort(); cap = caps[g]
    for (t0, n, r), (t1, *_) in zip(s, s[1:]):
        a, b = max(t0, lo), min(t1, hi)
        if b > a: need_w.append((n / cap, b - a)); ret_w += r / cap * (b - a); span += b - a
need_w.sort()
def pct(q):
    acc = 0
    for v, w in need_w:
        acc += w
        if acc >= q * span: return v
held = 0.0
for f in glob.glob(f"{run}/prefix_tiers_w0*.csv"):
    for row in csv.DictReader(open(f)):
        if row["tier"] == "none": continue
        a, b = max(float(row["read_ms"]), lo), min(float(row["admit_ms"]), hi)
        if b > a: held += int(row["declared_tokens"]) * (b - a)
held_share = held / (hi - lo) / sum(caps.values())
mean = sum(v * w for v, w in need_w) / span
full = sum(w for v, w in need_w if v > 0.95) / span
print(f"{run.split('logs/')[-1]:45s} need mean {100*mean:5.1f}% (holds {100*held_share:4.1f}%), p50 {100*pct(.5):5.1f}%, p99 {100*pct(.99):5.1f}%, max {100*need_w[-1][0]:5.1f}%; retained {100*ret_w/span:5.1f}%; time with need>95% {100*full:5.1f}%")
