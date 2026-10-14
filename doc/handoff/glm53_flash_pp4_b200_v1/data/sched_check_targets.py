"""Export the scheduling-check targets: sched_check_targets.json beside this file.

From runs of bringup/sched_check.csv (26 requests at t = 0), one per scheduler setting: every microbatch stage 0
forms, in order, with its budget-relevant facts (tokens, start time) and its chunks as (request id, first token,
tokens), and every request's TTFT. A chunk is matched to its request by where it starts: a chunk starting at token 0
opens a prompt, a later one continues the prompt that has computed exactly that many tokens and ran most recently.
Each prompt is then named by its length and the TTFT its last microbatch gives (the end of that microbatch on the
last stage), in trace order among prompts that share both; the script checks that every name is used once.

usage (ServingStudioSim root, at the frozen commit):
  PYTHONPATH=$PWD uv run python doc/handoff/glm53_flash_pp4_b200_v1/data/sched_check_targets.py <setting>=<run dir> ...
"""
import collections, csv, json, sys
from pathlib import Path

import pyarrow.parquet as pq

here = Path(__file__).parent
trace = list(csv.DictReader(open(here / "bringup/sched_check.csv")))
trace_at = {r["id"]: r["arrival_time"] for r in trace}
ids_by_length = collections.defaultdict(list)
for r in trace:
    ids_by_length[int(r["input_len"])].append(r["id"])

out = {}
for arg in sys.argv[1:]:
    setting, run = arg.split("=", 1)
    rows = pq.read_table(Path(run) / "raw/cost_log/worker_stage_0.parquet",
                         columns=["wall_start_ms", "total_time_ms", "groups"]).to_pylist()
    last = pq.read_table(Path(run) / "raw/cost_log/worker_stage_3.parquet",
                         columns=["wall_start_ms", "total_time_ms"]).to_pylist()
    assert len(last) == len(rows)
    slo = pq.read_table(Path(run) / "raw/request_slo.parquet", columns=["request_id", "ttft_ms"]).to_pylist()
    ttft = {trace[r["request_id"]]["id"]: r["ttft_ms"] for r in slo}
    prompts = []  # [computed tokens, [(microbatch, first token, tokens)]]
    for i, r in enumerate(rows):
        for g in r["groups"]:
            for start, tokens in zip(g["prefill_prefix_lens"], g["prefill_append_lens"]):
                if start == 0:
                    prompts.append([0, []])
                    prompt = prompts[-1]
                else:
                    prompt = max((p for p in prompts if p[0] == start), key=lambda p: p[1][-1][0])
                prompt[0] += tokens
                prompt[1].append((i, start, tokens))
    names = {}
    for p in prompts:
        end = last[p[1][-1][0]]
        end_ms = end["wall_start_ms"] + end["total_time_ms"]
        used = set(names.values())
        names[id(p)] = next(n for n in ids_by_length[p[0]]
                            if n not in used and abs(ttft[n] + float(trace_at[n]) - end_ms) < 1e-2)
    assert sorted(names.values()) == sorted(ttft), "every request is one prompt"
    microbatches = [dict(index=i, start_ms=round(r["wall_start_ms"], 3), compute_ms=round(r["total_time_ms"], 3),
                         tokens=sum(g["batch_tokens"] for g in r["groups"]), chunks=[]) for i, r in enumerate(rows)]
    for p in prompts:
        for i, start, tokens in p[1]:
            microbatches[i]["chunks"].append([names[id(p)], start, tokens])
    for mb in microbatches:
        mb["chunks"].sort(key=lambda c: int(c[0].split("_")[1]))
    out[setting] = dict(microbatches=microbatches, ttft_ms={n: round(t, 3) for n, t in ttft.items()})
    t = out[setting]["ttft_ms"]
    print(f"{setting:8} {len(rows):3} microbatches  last TTFT {max(t.values()):8.1f} ms  "
          f"128k {t['sched_0']:7.1f}  32k {t['sched_1']:7.1f}  4k {t['sched_2']:6.1f}..{t['sched_9']:6.1f}  "
          f"1k {t['sched_10']:6.1f}..{t['sched_25']:6.1f}  alone 8k {t['sched_26']:6.1f} 1k {t['sched_27']:5.1f}")
(here / "sched_check_targets.json").write_text(json.dumps(out, indent=1) + "\n")
print(here / "sched_check_targets.json")
