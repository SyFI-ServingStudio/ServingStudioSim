"""Per-engine inference windows for every (arm, rollout) of the slime fixed-B sweep.

Runs ON CORIANDER, where the measured arms live. Emits CSV on stdout:

    B,rollout,engine,dur_s

`dur` is the engine's own `inference` slice: it starts when the rollout's work is
dispatched and ends when that engine goes idle, NOT when its GPU is handed back
to training (a train group is only returned once both of its engines are done).
So `max` over the 8 engines is the rollout's makespan and `sum` is active GPU
seconds -- the second one is what a simulator's per-worker activity windows are
comparable to. Charging each engine to its block's release instead inflates the
low-B arms by ~9%.

    scp tools/slime-b-sweep/extract_engine_windows.py coriander:/m-coriander/coriander/kanzhu/MLSim_workspace/wt-sgl-fa3-profile/tools/slime-b-sweep/
    ssh coriander 'python3 /m-coriander/coriander/kanzhu/MLSim_workspace/wt-sgl-fa3-profile/tools/slime-b-sweep/extract_engine_windows.py' > data/all_arms.csv
"""

import json

ROOT = (
    "/m-coriander/coriander/mjacob2/slime/experiments/long_rl_training/"
    "deepseek_r1_8b/results_fixed_B_sweep_20roll"
)

print("B,rollout,engine,dur_s")
for b in (16, 32, 64, 96, 128):
    events = json.load(open(f"{ROOT}/batch_thresh_agg_{b}_mc0/trace.json"))
    if isinstance(events, dict):
        events = events["traceEvents"]
    for e in events:
        if e.get("name") != "inference":
            continue
        args = e.get("args") or {}
        if "rollout_id" not in args:
            continue
        print(f"{b},{args['rollout_id']},{args['engine_idx']},{e['dur'] / 1e6:.6f}")
