"""Every threshold firing of the slime fixed-B sweep, with its rollout.

Runs ON CORIANDER. Emits CSV on stdout:

    B,rollout,train_group,abs_ts,kind      # kind = fire | end

One `fire` row per prompt group moved -- the router logs one
`[MIGRATION] aborting N rid(s)` line per group, about 0.7 s apart, so a release
of 15 groups spans ~10 s of wall clock during which the source engines are still
decoding. Grouping those rows by `(B, rollout, train_group)` gives both the
firing instant (the first one) and that abort loop's duration (last - first).

Two things to know about the source data:

* the raw `[MIGRATION]` lines carry no rollout id, so rollout attribution comes
  from buffering them until the next `[ROLLOUT] Migration summary for rollout N`
  line. A rollout that somehow logged no summary would push its firings onto the
  next rollout; validate with the summary counts (the arms total 60 / 175 / 407 /
  624 / 866 groups, handoff SS2).
* timestamps are whole seconds. Fine for a 15-group release, useless for the
  B=16 arm where a release is a single group.

    scp tools/slime-b-sweep/extract_firings.py coriander:/m-coriander/coriander/kanzhu/MLSim_workspace/wt-sgl-fa3-profile/tools/slime-b-sweep/
    ssh coriander 'python3 /m-coriander/coriander/kanzhu/MLSim_workspace/wt-sgl-fa3-profile/tools/slime-b-sweep/extract_firings.py' > data/fires.csv
"""

import gzip
import re

ROOT = (
    "/m-coriander/coriander/mjacob2/slime/experiments/long_rl_training/"
    "deepseek_r1_8b/results_fixed_B_sweep_20roll"
)
ABORT = re.compile(
    r"\[(2026-\d\d-\d\d \d\d:\d\d:\d\d)\].*aborting \d+ rid\(s\) on engine (\d+)"
    r" -> dst engine (\d+) \(batch_threshold: group=(\d+)"
)
SUMMARY = re.compile(
    r"\[(2026-\d\d-\d\d \d\d:\d\d:\d\d)\].*Migration summary for rollout (\d+): (\d+) group"
)

print("B,rollout,train_group,abs_ts,kind")
for b in (16, 32, 64, 96, 128):
    pending = []
    with gzip.open(f"{ROOT}/batch_thresh_agg_{b}_mc0/run.log.gz", "rt", errors="replace") as f:
        for line in f:
            hit = ABORT.search(line)
            if hit:
                pending.append((hit.group(1), hit.group(4)))
                continue
            done = SUMMARY.search(line)
            if done:
                rollout = done.group(2)
                for ts, group in pending:
                    print(f"{b},{rollout},{group},{ts},fire")
                print(f"{b},{rollout},-1,{done.group(1)},end")
                pending = []
