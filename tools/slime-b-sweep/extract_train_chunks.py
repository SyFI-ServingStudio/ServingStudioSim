#!/usr/bin/env python3
"""Every training chunk of every fixed-B arm. Runs on coriander.

One row per `phase: train_chunk` record in `train_metrics/*.jsonl`. This is what
the simulator's chunk cost model is calibrated against:

    chunk_total_s = total_tokens / 10_176 + 0.915   (r = 0.9884 over 6053 chunks)

`chunk_total_s`, not `fwd_bwd_s`, is the quantity to fit: it is how long the GPU
pair is actually held, and the gap between the two is the per-chunk framework
overhead a streaming trainer pays once per grab.

**That fit is not what a preset should carry**, for two separate reasons:

* **This file is incomplete.** A rollout makes 64 grabs and only ~60 chunk
  records get written, so ~6% of the tokens are missing -- which is why a
  rollout's `total_tokens` sums to 94.0% +- 1.4% of the trace's
  `input_len + output_len`. The coverage tracks the record count (60 records ->
  93.6%, 78 -> 100.0%, r = 0.80) and three B=128 cells match the trace to the
  exact token. The trace is right; this log is short.
* **`chunk_total_s` is not how long the pair is held.** `busy_gpu_s / 2` from
  the same rollout's `tuner_decision` record (see `extract_training_times.py`)
  is 118.4% +- 2.0% of the sum here, and spans the whole rollout rather than
  the records that survived.

So the preset fits the pair-seconds against trace tokens instead:

    busy_pair_s = trace_tokens / 9_296 + 1.227 per chunk

which lands the 100 measured cells' training time at +0.1% mean error
(|mean| 1.9%). Use this file for the shape of a chunk, not for a total.

`timestamp` is wall-clock at chunk END (`time.time()` after `_process_chunk`
returns), so a chunk's start is `timestamp - chunk_total_s`. Rollouts are
separated by a weight update, so make a per-rollout frame before comparing:
absolute timestamps across rollouts are not a timeline the simulator has.

    scp tools/slime-b-sweep/extract_train_chunks.py coriander:<remote>/
    ssh coriander 'python3 <remote>/extract_train_chunks.py' > data/train_chunks.csv
"""

import glob
import json

ROOT = (
    "/m-coriander/coriander/mjacob2/slime/experiments/long_rl_training/"
    "deepseek_r1_8b/results_fixed_B_sweep_20roll"
)

print("B,rollout,train_group,chunk_id,total_tokens,num_microbatches,fwd_bwd_s,chunk_total_s,end_ts")
for b in (16, 32, 64, 96, 128):
    # Every GPU writes the file and a TP=2 pair reports identical rows, so the
    # rank in the filename is the DP rank; keying on (rollout, train_group)
    # already de-duplicates. Ranks that produced nothing leave a short file.
    for path in sorted(glob.glob(f"{ROOT}/batch_thresh_agg_{b}_mc0/train_metrics/*.jsonl")):
        for line in open(path):
            record = json.loads(line)
            if record.get("phase") != "train_chunk":
                continue
            print(
                f"{b},{record['rollout_id']},{record['train_group']},{record['chunk_id']},"
                f"{record['total_tokens']},{record['num_microbatches']},"
                f"{record['fwd_bwd_s']:.4f},{record['chunk_total_s']:.4f},"
                f"{record['timestamp']:.4f}"
            )
