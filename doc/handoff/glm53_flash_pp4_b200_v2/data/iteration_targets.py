"""Export the single-iteration targets of the one-request bring-up: iteration_targets.json beside this file.

For each prompt length: when each stage starts and how long it computes (raw/cost_log), the send between stages
(the gap from one stage's end to the next one's start), TTFT (raw/request_slo.parquet), and each stage's compute
split by layer type from layer_targets.jsonl. Each stage's split is checked against its compute time, and TTFT
against the last stage's end.

usage (ServingStudioSim root, at the frozen commit, after layer_targets.py on the same run):
  PYTHONPATH=$PWD uv run python doc/handoff/glm53_flash_pp4_b200_v2/data/iteration_targets.py <one-request run dir>
"""
import collections, json, re, sys
from pathlib import Path

import pyarrow.parquet as pq

run = Path(sys.argv[1])
here = Path(__file__).parent
ACTIVATION_BYTES_PER_TOKEN = 4 * 4096 * 2  # the 4-wide mHC residual stream, BF16

split = collections.defaultdict(dict)  # (tokens, stage) -> layer type -> {count, layer_ms, total_ms}
for line in (here / "layer_targets.jsonl").read_text().splitlines():
    r = json.loads(line)
    split[(r["prompt_tokens"], r["stage"])][r["layer_type"]] = dict(
        count=r["count"], layer_ms=r["layer_ms"], total_ms=r["count"] * r["layer_ms"])

out = []
cases = sorted((p for p in run.iterdir() if (p / "raw/cost_log").is_dir()),
               key=lambda p: int(re.search(r"(\d+)k", p.name).group(1)))
for case in cases:
    tokens = int(re.search(r"(\d+)k", case.name).group(1)) * 1024
    rows = [pq.read_table(case / f"raw/cost_log/worker_stage_{s}.parquet",
                          columns=["wall_start_ms", "total_time_ms"]).to_pylist()
            for s in range(4)]
    assert all(len(r) == 1 for r in rows), "one request is one iteration per stage"
    rows = [r[0] for r in rows]
    ttft, = pq.read_table(case / "raw/request_slo.parquet", columns=["ttft_ms"]).column("ttft_ms").to_pylist()
    stages = []
    for s, r in enumerate(rows):
        layers = split[(tokens, s)]
        assert abs(sum(v["total_ms"] for v in layers.values()) - r["total_time_ms"]) < 1e-3, (case.name, s)
        stages.append(dict(stage=s, start_ms=r["wall_start_ms"], compute_ms=r["total_time_ms"], layers=layers))
    sends = [dict(after_stage=s, bytes=tokens * ACTIVATION_BYTES_PER_TOKEN,
                  ms=rows[s + 1]["wall_start_ms"] - rows[s]["wall_start_ms"] - rows[s]["total_time_ms"])
             for s in range(3)]
    end = rows[3]["wall_start_ms"] + rows[3]["total_time_ms"]
    assert abs(end - ttft) < 1e-3, (case.name, end, ttft)
    out.append(dict(prompt_tokens=tokens, ttft_ms=ttft, stages=stages, sends=sends))
    print(f"{case.name:9} TTFT {ttft:9.3f}  compute " + " ".join(f"{s['compute_ms']:8.3f}" for s in stages)
          + f"  send {sends[0]['ms']:.3f} x3")
(here / "iteration_targets.json").write_text(json.dumps(out, indent=1) + "\n")
print(here / "iteration_targets.json")
