"""Export the tier bring-up targets: tier_targets.json beside this file.

Each argument is <setting>=<sim run dir> of bringup/tier_sessions.csv (7_offloading.md §1) with a KV pool of
1,234,944 tokens: hbm (no DRAM or SSD), dram (DRAM only), ssd (SSD only), each with balanced storage, dram_unbal /
ssd_unbal with each GPU storing and reading its own stage's layers, and hbm_hit with the full 61.3M-token pool and no
tiers, so that round 1 hits HBM. For each prefix length P: round 0's TTFT, and round 1's TTFT, prefix hit, the tier it
read from and the read time (from the read's start to its landing), each the median of the 5 sessions (all 5 agree to
0.1 ms).

usage (ServingStudioSim root, at the frozen commit):
  PYTHONPATH=$PWD uv run python doc/handoff/glm53_flash_pp4_b200_v1/data/tier_targets.py \\
      hbm_hit=logs/<run>/tiercheck/hbm_hit hbm=logs/<run>/tiercheck/hbm dram=... ssd=... dram_unbal=... ssd_unbal=...
"""
import csv, json, sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

here = Path(__file__).parent
trace = list(csv.DictReader(open(here / "bringup/tier_sessions.csv")))
out = {}
for arg in sys.argv[1:]:
    setting, run = arg.split("=", 1)
    run = Path(run)
    t = pq.read_table(run / "raw/request_slo.parquet", columns=["request_id", "ttft_ms", "prefix_cache_hit_tokens"])
    ttft = dict(zip(t.column("request_id").to_pylist(), t.column("ttft_ms").to_pylist()))
    hit = dict(zip(t.column("request_id").to_pylist(), t.column("prefix_cache_hit_tokens").to_pylist()))
    reads = {}
    if (run / "prefix_tiers_w0.csv").exists():
        for row in csv.DictReader(open(run / "prefix_tiers_w0.csv")):
            reads[int(row["request_id"])] = (row["tier"], float(row["ready_ms"]) - float(row["read_ms"]))
    rows = {}
    for i, r in enumerate(trace):
        if r["round_idx"] != "1":
            continue
        p = int(r["prefix_len"]) - 1
        tier, read_ms = reads.get(i, ("none", 0.0))
        rows.setdefault(p, []).append((ttft[i - 1], ttft[i], hit[i] or 0, tier, read_ms))
    res = []
    for p, v in sorted(rows.items()):
        a = np.array([x[:3] + (x[4],) for x in v], dtype=float)
        assert np.ptp(a[:, 1]) < 0.1 and np.ptp(a[:, 3]) < 0.1 and len({x[3] for x in v}) == 1, (setting, p, v)
        med = np.median(a, axis=0)
        res.append(dict(prefix_tokens=p, round0_ttft_ms=round(med[0], 1), round1_ttft_ms=round(med[1], 1),
                        round1_hit_tokens=int(med[2]), round1_tier=v[0][3], read_ms=round(med[3], 2)))
        print(f"{setting:11} P={p:7}  round0 {med[0]:7.1f}  round1 {med[1]:7.1f} ms  hit {int(med[2]):7}  "
              f"{v[0][3]:4} read {med[3]:6.2f} ms")
    out[setting] = res
(here / "tier_targets.json").write_text(json.dumps(out, indent=1) + "\n")
