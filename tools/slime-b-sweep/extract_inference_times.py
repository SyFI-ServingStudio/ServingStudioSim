import json
from pathlib import Path

root = Path("/m-coriander/coriander/mjacob2/slime/experiments/long_rl_training/"
            "deepseek_r1_8b/results_fixed_B_sweep_20roll")
for b in (16, 32, 64, 96, 128):
    report = root / f"batch_thresh_agg_{b}_mc0" / "report.json"
    rows = json.loads(report.read_text())["rollouts"]
    for r in rows:
        print(b, r["rollout_id"], round(r["inference_time_s"], 3))
