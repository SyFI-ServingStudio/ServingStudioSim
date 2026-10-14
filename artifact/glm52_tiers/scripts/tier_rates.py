"""Per-GPU DRAM/SSD reads, writes and end-of-run occupancy of one run, from its prefix_tiers_w*.json.

Whole-run means at perfect balance (cluster total / GPUs). Bytes per token is per GPU for a DP rank's whole model;
a pipeline's tiers count whole-model tokens, so the same per-token bytes over the same GPU count apply.
usage: tier_rates.py <run dir> [bytes_per_token=55224] [gpus=8] [hours=8]
"""
import json, sys
from pathlib import Path

run = Path(sys.argv[1])
B = float(sys.argv[2]) if len(sys.argv) > 2 else 55224
G = int(sys.argv[3]) if len(sys.argv) > 3 else 8
T = (float(sys.argv[4]) if len(sys.argv) > 4 else 8) * 3600
declared = hbm = 0
tiers = {}
for f in sorted(run.glob("prefix_tiers_w*.json")):
    d = json.load(open(f))
    declared += d["declared_tokens"]; hbm += d["hbm_hit_tokens"]
    for t in d["tiers"]:
        a = tiers.setdefault(t["tier"], dict(hit=0, cap=0, used=0, written=0, evicted=0))
        a["hit"] += t["hit_tokens"]; a["cap"] += t["capacity_tokens"]; a["used"] += t["used_tokens"]
        a["written"] += t["written_tokens"]; a["evicted"] += t["evicted_tokens"]
gbs = lambda tok: tok * B / T / G / 1e9
print(f"HBM hit {100 * hbm / declared:.1f}% of declared prefix")
for name, a in tiers.items():
    print(f"{name}: hit {100 * a['hit'] / declared:.1f}%, read {gbs(a['hit']):.2f} GB/s/GPU, write {gbs(a['written']):.2f}, "
          f"evict {gbs(a['evicted']):.2f}, used {a['used'] * B / G / 1e9:.0f} of {a['cap'] * B / G / 1e9:.0f} GB/GPU at the end")
