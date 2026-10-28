#!/usr/bin/env python3
"""Per-rollout training phase timing for every fixed-B arm. Runs on coriander.

Two sources, one row per rollout.

`report.json` is the only place the phase split is recorded, and its five fields
close exactly:

    total_rollout_time = inference_time + training_time - overlap_time
                         + gradient_sync_time + weight_update_time

`training_time` spans the first chunk of the earliest-freed train pair to the
last chunk of the last one, and `overlap_time` is the part of that which ran
while generation was still going — which is the quantity streaming RL exists to
maximise, and therefore the one a simulated release policy has to reproduce.

The `tuner_decision` record in `train_metrics/*.jsonl` adds the work accounting
behind that span, in GPU-seconds over the 8 training GPUs (so halve for the
TP=2 pair-seconds a simulated block is charged in). `busy_gpu_s` is the one to
calibrate a chunk cost against: it is larger than the sum of the same rollout's
`chunk_total_s` records, because a pair is held for more than the fwd/bwd chunks
it logs.

    scp tools/slime-b-sweep/extract_training_times.py coriander:<remote>/
    ssh coriander 'python3 <remote>/extract_training_times.py' > data/train_times.csv
"""

import glob
import json

ROOT = (
    "/m-coriander/coriander/mjacob2/slime/experiments/long_rl_training/"
    "deepseek_r1_8b/results_fixed_B_sweep_20roll"
)

REPORT_FIELDS = (
    "inference_time_s",
    "training_time_s",
    "overlap_time_s",
    "gradient_sync_time_s",
    "weight_update_time_s",
    "total_rollout_time_s",
)

TUNER_FIELDS = (
    "training_span_gpu_s",
    "busy_gpu_s",
    "interior_idle_gpu_s",
    "trailing_idle_gpu_s",
)

print("B,rollout," + ",".join(REPORT_FIELDS + TUNER_FIELDS))
for b in (16, 32, 64, 96, 128):
    arm = "%s/batch_thresh_agg_%d_mc0" % (ROOT, b)
    tuner = {}
    for path in glob.glob(arm + "/train_metrics/*.jsonl"):
        for line in open(path):
            record = json.loads(line)
            if record.get("phase") == "tuner_decision":
                tuner[record["rollout_id"]] = record
    for entry in json.load(open(arm + "/report.json"))["rollouts"]:
        rollout = entry["rollout_id"]
        row = [entry[f] for f in REPORT_FIELDS]
        row += [tuner[rollout][f] for f in TUNER_FIELDS]
        print("%d,%d,%s" % (b, rollout, ",".join("%.6f" % v for v in row)))
