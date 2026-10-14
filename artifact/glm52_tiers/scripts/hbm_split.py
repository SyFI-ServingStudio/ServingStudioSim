"""Time-mean HBM KV per DP rank over [lo, hi) h: running requests, read holds, retained prefixes, free.

kv_snapshot gives active = resident + held + retained and retained; the read holds come from the tier log
(read_ms -> admit_ms, holding the request's context, declared_tokens). running = active - retained - held.
"""
import sys, csv, glob
import pyarrow.parquet as pq
run, lo, hi = sys.argv[1], float(sys.argv[2]) * 3.6e6, float(sys.argv[3]) * 3.6e6
cap = 1277698
t = pq.read_table(f"{run}/raw/kv_snapshot/worker_main_0.parquet").to_pydict()
by = {}
for g, tm, a, r in zip(t["group_id"], t["time_ms"], t["active_kv"], t["retained_prefix_kv"]):
    by.setdefault(g, []).append((tm, a, r))
def tmean(series):  # step function, value holds until the next sample
    s = 0.0; span = 0.0
    for (t0, *v), (t1, *_) in zip(series, series[1:]):
        a, b = max(t0, lo), min(t1, hi)
        if b > a: s = [x + y * (b - a) for x, y in zip(s, v)] if span else [y * (b - a) for y in v]; span += b - a
    return [x / span for x in s]
act = ret = 0.0
for g, s in by.items():
    s.sort(); a, r = tmean(s); act += a; ret += r
n = len(by); act /= n; ret /= n
held = 0.0
for f in glob.glob(f"{run}/prefix_tiers_w0_p*.csv"):
    for row in csv.DictReader(open(f)):
        if row["tier"] == "none": continue
        a, b = max(float(row["read_ms"]), lo), min(float(row["admit_ms"]), hi)
        if b > a: held += int(row["declared_tokens"]) * (b - a)
held /= (hi - lo) * n
run_ = act - ret - held
B = 55224 / 1e9
print(f"{run}: ranks {n}, per GPU (capacity {cap:,} tok = {cap*B:.1f} GB)")
for k, v in (("running requests", run_), ("read holds", held), ("retained prefixes", ret), ("free", cap - act)):
    print(f"  {k:18s} {v/1e3:8.0f}k tok  {v*B:5.1f} GB  {100*v/cap:5.1f}%")
