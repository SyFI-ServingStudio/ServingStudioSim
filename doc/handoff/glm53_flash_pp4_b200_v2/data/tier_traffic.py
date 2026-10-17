"""Tier traffic of a session run with DRAM and SSD (7_offloading.md §4), per GPU.

For each <name>=<sim run dir>: each tier's write and read rate averaged over the 20 minutes, its space used at the
end, the busiest 10-s window of SSD reads and of each tier's writes, and the longest wait of a read for its tier's channel.
Bytes per GPU are the balanced store's: `tier_kv_bytes_per_token` per token (the tier's capacity in tokens over its
bytes) plus one KDA state per stored or read context (`tier_state_bytes_per_request` in the run's stdout).
- Writes: the tokens each store adds to the session's entry, plus one state per store (DRAM: what HBM evicts; SSD:
  what DRAM evicts).
- Reads: the tokens a read moves (what HBM lacks), plus one state; a read's window is the one it lands in.
- Write windows: the run's own count of each tier's writes per 10 s (`window_writes` in prefix_tiers_w0.json).
- Read wait: from when the read is issued to when it lands, minus its transfer time.

usage (ServingStudioSim root):
  PYTHONPATH=$PWD uv run python doc/handoff/glm53_flash_pp4_b200_v2/data/tier_traffic.py <name>=<run dir> ...
"""
import csv, json, math, re, sys
from collections import defaultdict
from pathlib import Path


SECONDS = 1200.0
BW = {"dram": 50e9, "ssd": 10e9}

for arg in sys.argv[1:]:
    name, run = arg.split("=", 1)
    run = Path(run)
    summary = json.loads((run / "prefix_tiers_w0.json").read_text())
    tiers = {t["tier"]: t for t in summary["tiers"]}
    state_bytes = int(re.search(r"tier_state_bytes_per_request=(\d+)", (run / "stdout.log").read_text()).group(1))
    cfg = (run / "raw/run_config.yaml").read_text()
    dram_gb = float(re.search(r"dram_tier_gb: ([0-9.]+)", cfg).group(1))
    per_token = round(dram_gb * 1e9 / tiers["dram"]["capacity_tokens"])
    state_tokens = -(-state_bytes // per_token)
    out = {}
    for tier, t in tiers.items():
        out[tier] = dict(
            write_gb_s=(t["written_tokens"] * per_token + t["stored_entries"] * state_bytes) / SECONDS / 1e9,
            read_gb_s=(t["loaded_tokens"] + t["loads"] * state_tokens) * per_token / SECONDS / 1e9,
            used_gb=t["used_tokens"] * per_token / 1e9,
            capacity_gb=t["capacity_tokens"] * per_token / 1e9,
            evicted_entries=t["evicted_entries"])
    reads = defaultdict(float)
    max_wait = 0.0
    for row in csv.DictReader(open(run / "prefix_tiers_w0.csv")):
        if row["tier"] not in BW:
            continue
        nbytes = (int(row["tier_tokens"]) - int(row["hbm_tokens"]) + state_tokens) * per_token
        ready = float(row["ready_ms"])
        max_wait = max(max_wait, (ready - float(row["read_ms"])) / 1e3 - nbytes / BW[row["tier"]])
        if row["tier"] == "ssd":
            reads[int(ready // 10e3)] += nbytes / 10.0
    for tier, t in tiers.items():
        # Each tier's writes per 10-s window of the run: (entries stored, tokens written).
        peak = max(((n * state_bytes + tokens * per_token) / 10.0 for n, tokens in t["window_writes"][:120]),
                   default=0.0)
        out[f"{tier}_write_peak_10s_gb_s"] = peak / 1e9
    out["ssd_read_peak_10s_gb_s"] = max(reads.values(), default=0.0) / 1e9
    out["max_read_wait_s"] = max_wait
    print(name, json.dumps(out, default=lambda x: round(x, 3)))
