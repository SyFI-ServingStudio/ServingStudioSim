"""DRAM capacity a rank needs: replay each rank's tier log as an LRU over session contexts.

Each request touches its session's context (declared_tokens) at admission. The LRU stack distance (bytes of the
distinct sessions touched since this session's last touch, plus its own) says whether a DRAM of a given size still
holds it. First touches (sessions joined mid-life or new) are compulsory misses and listed apart.
"""
import sys
from tier_log import admissions
B = 55224  # bytes per token per GPU (GLM-5.2 DP8)
run = sys.argv[1]
caps_gb = [150, 300, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000]
tot = first = 0
hit = {c: 0 for c in caps_gb}
dists = []
by_rank = {}
for r in admissions(run):
    if r["declared_tokens"] > 0:
        by_rank.setdefault(r["partition"], []).append(r)
for _, rows in sorted(by_rank.items()):
    rows.sort(key=lambda r: r["admit_ms"])
    n = len(rows); tree = [0] * (n + 1)
    def add(i, v):
        i += 1
        while i <= n: tree[i] += v; i += i & -i
    def pre(i):
        s = 0
        while i > 0: s += tree[i]; i -= i & -i
        return s
    last = {}
    for i, r in enumerate(rows):
        s, d = r["session_id"], r["declared_tokens"]
        tot += d
        if s in last:
            j, size = last[s]
            dist = pre(n) - pre(j + 1) + d  # sessions touched after j, plus this context
            add(j, -size)
            gb = dist * B / 1e9
            for c in caps_gb:
                if gb <= c: hit[c] += d
        else:
            first += d
        add(i, d); last[s] = (i, d)
print(f"{run}: prefix tokens {tot/1e9:.2f}B, first-touch (cold) {100*first/tot:.1f}%")
for c in caps_gb:
    print(f"  DRAM {c:5d} GB/GPU: {100*hit[c]/tot:5.1f}% of prefix tokens hit DRAM; {100*hit[c]/(tot-first):5.1f}% of the re-touched")
