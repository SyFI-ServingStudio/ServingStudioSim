"""Export the session-benchmark targets: session_targets.json beside this file.

Each argument is <setting>=<sim run dir> of a 20-minute run of sessions_2ps.csv (1_contract.md §2.2), with the
run's session cap read from its config. For each run: the output of metrics.py (rounds/s, new, prompt and computed
tokens/s, prefix hit share, TTFT p50/p90/p99/max, rounds over 60 s), each stage's busy fraction over the run, and the
prefix read share (1_contract.md §3): the prefix tokens served from HBM, DRAM and SSD over the prefix tokens of the
rounds that arrive in [10, 20 min) (the rest were recomputed). A tier read moves only what HBM lacks, so its HBM part
counts as HBM.

usage (ServingStudioSim root, at the frozen commit):
  PYTHONPATH=$PWD uv run python doc/handoff/glm53_flash_pp4_b200_v1/data/session_targets.py \
      plain_hbm=logs/<sched run>/plain/c400 ...
"""
import csv, json, subprocess, sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

here = Path(__file__).parent
path = here / "session_targets.json"
out = {(r["setting"], r["max_sessions"]): r for r in json.loads(path.read_text())} if path.exists() else {}
for arg in sys.argv[1:]:
    setting, run = arg.split("=", 1)
    run = Path(run)
    cfg = yaml.safe_load((run / "raw/run_config.yaml").read_text())
    n = cfg["workload"]["max_concurrency"]
    metrics = json.loads(subprocess.run([sys.executable, str(here / "metrics.py"), str(run)], check=True,
                                        capture_output=True, text=True).stdout)
    util = json.loads((run / "reports/utilization_report.json").read_text())["totals"]["per_worker"]
    busy = [w["avg_util"] for w in sorted(util, key=lambda w: w["worker_id"])]
    t = pq.read_table(run / "raw/request_slo.parquet",
                      columns=["request_id", "arrival_time_ms", "declared_prefix_tokens", "prefix_cache_hit_tokens"])
    arrival = dict(zip(t.column("request_id").to_pylist(), t.column("arrival_time_ms").to_pylist()))
    share = np.zeros(3)
    if (run / "prefix_tiers_w0.csv").exists():
        declared = 0
        for row in csv.DictReader(open(run / "prefix_tiers_w0.csv")):
            if not 600e3 <= arrival.get(int(row["request_id"]), -1) < 1200e3:
                continue
            declared += int(row["declared_tokens"])
            if row["tier"] in ("dram", "ssd"):
                share[0] += int(row["hbm_tokens"])
                share[1 if row["tier"] == "dram" else 2] += int(row["tier_tokens"]) - int(row["hbm_tokens"])
            elif row["tier"] == "hbm":
                share[0] += int(row["resident_tokens"])
    else:
        at = t.column("arrival_time_ms").to_numpy()
        m = (at >= 600e3) & (at < 1200e3)
        declared = t.column("declared_prefix_tokens").to_numpy()[m].sum()
        share[0] = np.nan_to_num(t.column("prefix_cache_hit_tokens").to_numpy(zero_copy_only=False)[m].astype(float)).sum()
    read_share = dict(zip(("hbm", "dram", "ssd"), (share / max(declared, 1)).tolist()))
    out[(setting, n)] = dict(setting=setting, max_sessions=n, **metrics, stage_busy=busy, prefix_read_share=read_share)
    print(f"{setting:10} N={n:4}  {metrics['rounds_per_s']:5.1f} r/s  new {metrics['new_tokens_per_s'] / 1e3:6.1f}k  "
          f"computed {metrics['computed_tokens_per_s'] / 1e3:6.1f}k  hit {metrics['prefix_hit_share']:.4f}  TTFT "
          + " / ".join(f"{v:.2f}" for v in metrics["ttft_s"].values()) + f"  >60s {metrics['rounds_ttft_over_60s']}" + "  busy " + " ".join(f"{b:.2f}" for b in busy)
          + "  reads " + " / ".join(f"{100 * v:.0f}" for v in read_share.values()))
path.write_text(json.dumps(sorted(out.values(), key=lambda r: (r["setting"], r["max_sessions"])), indent=1) + "\n")
print(path)
