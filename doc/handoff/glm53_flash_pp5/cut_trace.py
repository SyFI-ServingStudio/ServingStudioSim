"""Cut a closed-loop session trace (trace/session_closed_loop.py) to what a run of `--minutes` can reach.

Keeps, per session, every round whose earliest possible arrival (sum of the session's earlier tool waits) is before
the cutoff, plus the first round at or after it, so no session can end, and free its slot, earlier than in the full
file. Keeps only sessions in file order up to `--sessions` (the full run shows how many start before the cutoff).
usage: cut_trace.py <in.csv> <out.csv> --minutes 16 --sessions K
"""
import argparse, csv
ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--minutes", type=float, required=True); ap.add_argument("--sessions", type=int, required=True)
a = ap.parse_args(); cut = a.minutes * 60e3
order = {}; wait = {}; done = set(); kept = 0
with open(a.src) as fi, open(a.dst, "w", newline="") as fo:
    r = csv.DictReader(fi); w = csv.DictWriter(fo, r.fieldnames); w.writeheader()
    for row in r:
        s = row["session_id"]
        if s not in order:
            if len(order) >= a.sessions: continue
            order[s] = len(order); wait[s] = 0.0
        if s in done: continue
        w.writerow(row); kept += 1
        if wait[s] >= cut: done.add(s)
        wait[s] += float(row["tool_wait_after_ms"])
print(f"{a.dst}: {len(order)} sessions, {kept} rows")
