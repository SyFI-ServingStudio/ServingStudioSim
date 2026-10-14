"""Export the force-schedule check targets: force_check_targets.json beside this file.

From runs of bringup/force_check.csv (a 32k prompt at t = 0 behind a stream of 1k prompts that the pipeline cannot
keep up with), one per scheduler setting: the 32k prompt's TTFT, the 1k prompts' TTFT p50 / p90 / p99 / max, the time the
last token comes out, and the microbatch count.

usage (ServingStudioSim root, at the frozen commit):
  PYTHONPATH=$PWD uv run python doc/handoff/glm53_flash_pp4_b200_v1/data/force_check_targets.py <setting>=<run dir> ...
"""
import csv, json, sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

here = Path(__file__).parent
names = [r["id"] for r in csv.DictReader(open(here / "bringup/force_check.csv"))]
out = {}
for arg in sys.argv[1:]:
    setting, run = arg.split("=", 1)
    rows = pq.read_table(Path(run) / "raw/request_slo.parquet",
                         columns=["request_id", "arrival_time_ms", "ttft_ms"]).to_pylist()
    long_ms, = [r["ttft_ms"] for r in rows if names[r["request_id"]] == "force_long"]
    short = np.array([r["ttft_ms"] for r in rows if names[r["request_id"]] != "force_long"]) / 1e3
    out[setting] = dict(
        long_ttft_s=round(long_ms / 1e3, 3),
        short_ttft_s={k: round(float(v), 3) for k, v in
                      (("p50", np.percentile(short, 50)), ("p90", np.percentile(short, 90)),
                       ("p99", np.percentile(short, 99)), ("max", short.max()))},
        last_token_s=round(max(r["arrival_time_ms"] + r["ttft_ms"] for r in rows) / 1e3, 3),
        microbatches=pq.read_metadata(Path(run) / "raw/cost_log/worker_stage_0.parquet").num_rows)
    print(setting, out[setting])
(here / "force_check_targets.json").write_text(json.dumps(out, indent=1) + "\n")
print(here / "force_check_targets.json")
