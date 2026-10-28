#!/usr/bin/env python3
"""The colocate baseline of the same 20-rollout sweep. Runs on coriander.

`results_fixed_B_sweep_20roll/colocate_baseline` is the matched arm for the five
`batch_thresh_agg_*` arms, and it is matched in the strongest possible way: it
ran FIRST with `natural_generation: true` and
`record_lengths_path: .../lengths_20roll.json`, and every streaming arm then
replayed that file. Same box, same model, same 8 GPUs, same group size, same
prompts -- and, by construction, the exact same 20 rollouts of sample lengths.
The only difference is the execution mode.

Colocate has no `report.json`: generation and training never overlap, so there
is no phase split to report. `rollout_timing.jsonl` carries one `end` record per
rollout with the three spans laid end to end.

    inference_s + train_s  is the like-for-like counterpart of the streaming
    arms' `training_end` (= inference_time - overlap + training_time): both are
    "rollout start -> last gradient computed", both exclude the weight update.

`duration_s` is larger than the sum of the three fields -- 130 s on rollout 0
(router and engine warm-up), 7-15 s afterwards (engine sleep/wake around the
optimizer step). Compare the spans, not `duration_s`.

**One confound to state whenever this baseline is used:** colocate routes with
sgl-router `cache_aware` while the streaming arms route `group_index % 8`, so
its generation is not the same schedule, only the same work. On rollout 0 it
spends 443.8 s generating against streaming's 421.7 s. Part of the measured
speedup is that routing difference rather than the overlap itself.

    scp tools/slime-b-sweep/extract_colocate_times.py coriander:<remote>/
    ssh coriander 'python3 <remote>/extract_colocate_times.py' > data/colocate_times.csv
"""

import json

PATH = (
    "/m-coriander/coriander/mjacob2/slime/experiments/long_rl_training/"
    "deepseek_r1_8b/results_fixed_B_sweep_20roll/colocate_baseline/rollout_timing.jsonl"
)

FIELDS = ("duration_s", "inference_s", "train_s", "weight_update_s")

print("rollout," + ",".join(FIELDS))
for line in open(PATH):
    record = json.loads(line)
    if record.get("event") != "end":
        continue
    print("%d,%s" % (record["rollout"], ",".join("%.6f" % record[f] for f in FIELDS)))
