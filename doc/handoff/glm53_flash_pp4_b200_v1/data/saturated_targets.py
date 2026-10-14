"""Export the saturated bring-up targets: saturated_targets.json beside this file.

For each prompt length, from the saturated run of bringup/uni_<L>.csv (all requests at t = 0, microbatches filled to
8,192 tokens in arrival order, longer prompts cut into 8k chunks):
  run_s            first stage-0 start to the last stage's last finish
  tokens_per_s     prompt tokens of the microbatches that start on stage 0 inside the middle 80% of the run, over that
                   window (1_contract.md §3); requests_per_s is the same divided by the prompt length
  stage_busy       the fraction of the middle 80% each stage computes
  ttft_s           first, p50, p90, p99 and max over all requests
  ttft_ms_by_request   every request's TTFT, in the order of the trace

usage (ServingStudioSim root, at the frozen commit):
  PYTHONPATH=$PWD uv run python doc/handoff/glm53_flash_pp4_b200_v1/data/saturated_targets.py <saturated run dir>
"""
import json, re, sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

run = Path(sys.argv[1])
here = Path(__file__).parent
out = []
cases = sorted((p for p in run.iterdir() if (p / "raw/cost_log").is_dir()),
               key=lambda p: int(re.search(r"(\d+)k", p.name).group(1)))
for case in cases:
    length = int(re.search(r"(\d+)k", case.name).group(1)) * 1024
    stages = [pq.read_table(case / f"raw/cost_log/worker_stage_{s}.parquet",
                            columns=["wall_start_ms", "total_time_ms", "groups"]).to_pylist() for s in range(4)]
    start = stages[0][0]["wall_start_ms"]
    end = max(r["wall_start_ms"] + r["total_time_ms"] for r in stages[3])
    lo, hi = start + 0.1 * (end - start), start + 0.9 * (end - start)
    tokens = sum(g["batch_tokens"] for r in stages[0] if lo <= r["wall_start_ms"] < hi for g in r["groups"])
    busy = [sum(max(0.0, min(r["wall_start_ms"] + r["total_time_ms"], hi) - max(r["wall_start_ms"], lo)) for r in s)
            / (hi - lo) for s in stages]
    slo = pq.read_table(case / "raw/request_slo.parquet", columns=["request_id", "completed", "ttft_ms"])
    assert all(slo.column("completed").to_pylist()), case.name
    ttft = np.array(slo.column("ttft_ms").to_pylist(), dtype=float)[np.argsort(slo.column("request_id").to_numpy())]
    tokens_per_s = tokens / ((hi - lo) / 1e3)
    out.append(dict(
        prompt_tokens=length, requests=len(ttft), microbatches=len(stages[0]), run_s=(end - start) / 1e3,
        tokens_per_s=tokens_per_s, requests_per_s=tokens_per_s / length, stage_busy=busy,
        ttft_s=dict(first=ttft.min() / 1e3, **{f"p{q}": np.percentile(ttft, q) / 1e3 for q in (50, 90, 99)},
                    max=ttft.max() / 1e3),
        ttft_ms_by_request=[round(float(x), 3) for x in ttft]))
    r = out[-1]
    print(f"{case.name:9} {r['requests']:6} req  {r['run_s']:7.2f} s  {r['tokens_per_s'] / 1e3:6.1f}k tok/s  "
          f"TTFT p50 {r['ttft_s']['p50']:.2f} p99 {r['ttft_s']['p99']:.2f} s  busy "
          + " ".join(f"{100 * b:.1f}" for b in busy))
(here / "saturated_targets.json").write_text(json.dumps(out) + "\n")
print(here / "saturated_targets.json")
