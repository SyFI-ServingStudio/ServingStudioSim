"""Export the prefix bring-up targets: prefix_targets.json beside this file.

From the run of bringup/prefix_sessions.csv (two-round sessions, one at a time; plain scheduler, 8,192-token
microbatches), for each (prefix P, computed A):
  round0        TTFT of the P-token prompt, in 8k chunks, and its microbatch count
  round1        TTFT, hit tokens, computed tokens, and per microbatch the stage compute times and the sends between
                stages (the gap from one stage's end to the next one's start)
The 5 sessions of a case must give the same times; the script checks it.

usage (ServingStudioSim root, at the frozen commit):
  PYTHONPATH=$PWD uv run python doc/handoff/glm53_flash_pp4_b200_v1/data/prefix_targets.py <prefix bring-up run dir>
"""
import collections, csv, json, sys
from pathlib import Path

import pyarrow.parquet as pq

run = Path(sys.argv[1])
here = Path(__file__).parent
trace = list(csv.DictReader(open(here / "bringup/prefix_sessions.csv")))
slo = {r["request_id"]: r for r in pq.read_table(run / "raw/request_slo.parquet").to_pylist()}
stages = [pq.read_table(run / f"raw/cost_log/worker_stage_{s}.parquet",
                        columns=["wall_start_ms", "total_time_ms", "groups"]).to_pylist() for s in range(4)]


def microbatches(start, end):
    """The microbatches stage 0 starts in [start, end), each followed through the 4 stages (FIFO per stage)."""
    first = [i for i, r in enumerate(stages[0]) if start <= r["wall_start_ms"] < end]
    out = []
    for i in first:
        rows = [stages[s][i] for s in range(4)]
        out.append(dict(
            tokens=sum(g["batch_tokens"] for g in rows[0]["groups"]),
            context=[p for g in rows[0]["groups"] for p in g["prefill_prefix_lens"]],
            compute_ms=[round(r["total_time_ms"], 3) for r in rows],
            send_ms=[round(rows[s + 1]["wall_start_ms"] - rows[s]["wall_start_ms"] - rows[s]["total_time_ms"], 3)
                     for s in range(3)]))
    return out


assert all(len(s) == len(stages[0]) for s in stages), "every microbatch passes all 4 stages"
cases = collections.defaultdict(list)
for i in range(0, len(trace), 2):
    r0, r1 = trace[i], trace[i + 1]
    s0, s1 = slo[int(r0["session_id"]) * 2], slo[int(r1["session_id"]) * 2 + 1]
    prefix, computed = int(r0["input_len"]), int(r1["input_len"]) + 1
    cases[(prefix, computed)].append(dict(
        round0=dict(ttft_ms=round(s0["ttft_ms"], 3),
                    microbatches=len(microbatches(s0["arrival_time_ms"], s0["arrival_time_ms"] + s0["ttft_ms"]))),
        round1=dict(ttft_ms=round(s1["ttft_ms"], 3), hit_tokens=s1["prefix_cache_hit_tokens"],
                    computed_tokens=s1["prefill_processed"],
                    microbatches=microbatches(s1["arrival_time_ms"], s1["arrival_time_ms"] + s1["ttft_ms"]))))
out = []
for (prefix, computed), runs in sorted(cases.items()):
    assert all(r == runs[0] for r in runs), (prefix, computed)
    r = runs[0]
    assert r["round1"]["hit_tokens"] == prefix and r["round1"]["computed_tokens"] == computed, (prefix, computed)
    out.append(dict(prefix_tokens=prefix, computed_tokens=computed, sessions=len(runs), **r))
    mb = r["round1"]["microbatches"]
    print(f"P {prefix:7} A {computed:5}  round0 {r['round0']['ttft_ms']:9.3f} ms ({r['round0']['microbatches']} mb)"
          f"  round1 {r['round1']['ttft_ms']:8.3f} ms  stages " + " ".join(f"{x:7.3f}" for x in mb[0]["compute_ms"])
          + f"  sends {mb[0]['send_ms'][0]:.3f}  ({len(mb)} mb)")
(here / "prefix_targets.json").write_text(json.dumps(out, indent=1) + "\n")
print(here / "prefix_targets.json")
