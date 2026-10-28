"""Streaming Text2SQL arm (bt_tuner_B8_step8): each measured rollout vs its simulated cell at the same B.

    uv run python tools/t2s/align_streaming.py <grid_dir> [--arm-dir DIR]

The measured arm's tuner never starved, so its B was 8, 16, ..., 256, 256, ...:
rollout r ran at a known fixed B_r. A simulated rollout does not depend on the
ones before it, so cell `r{r}_{B_r}` of the grid is that rollout's replay.

Measured per rollout (report.json, same clock origin = rollout dispatch):
  inference_time_s                       last engine idle
  training_end = inference + training - overlap
  overlap_time_s                         training that ran while inference did
Simulated: last request finish, last chunk end, first chunk start.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t2s_cells as tc  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
ARM = DATA / "arms" / "StreamTrainer_RollPackerFaithfulQueue_35rollout_bt_tuner_B8_step8"


def measured(arm: Path) -> pd.DataFrame:
    rep = pd.read_csv(arm / "report.csv").set_index("rollout_id")
    tun = pd.read_csv(arm / "tuner.csv").set_index("rollout_id")
    out = pd.DataFrame({
        "B": tun.b_before.astype(int),
        "inference": rep.inference_time_s,
        "training": rep.training_time_s,
        "overlap": rep.overlap_time_s,
        "busy_pair_s": tun.busy_gpu_s / 2,
        "interior_ratio": tun.interior_idle_ratio,
    })
    out["training_end"] = out.inference + out.training - out.overlap
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("grid", type=Path)
    ap.add_argument("--arm-dir", type=Path, default=ARM)
    args = ap.parse_args()
    meas = measured(args.arm_dir)
    rows = []
    for r, m in meas.iterrows():
        cell = args.grid / f"r{r}_{int(m.B)}"
        if not (cell / "summary.json").exists():
            continue
        t_start, t_end, chunks = tc.training_window(cell)
        inf = tc.inference_end(cell)
        obs = tc.observation(cell)
        rows.append({"r": r, "B": int(m.B),
                     "inf_meas": m.inference, "inf_sim": inf,
                     "end_meas": m.training_end, "end_sim": t_end,
                     "ovl_meas": m.overlap, "ovl_sim": max(0.0, inf - t_start),
                     "busy_meas": m.busy_pair_s, "busy_sim": obs["busy"] / 2,
                     "idle_meas": m.interior_ratio, "idle_sim": obs["ratio"], "chunks_sim": chunks})
    df = pd.DataFrame(rows)
    for k in ("inf", "end", "busy"):
        df[f"{k}_err"] = df[f"{k}_sim"] / df[f"{k}_meas"] - 1
    pd.set_option("display.width", 220)
    print(df.round(3).to_string(index=False))
    sel = df[df.r > 0]  # rollout 0 is startup-inflated (slime's own tuner skips it)
    print(f"\nrollouts 1-34: inference sum sim/meas {sel.inf_sim.sum() / sel.inf_meas.sum():.4f} "
          f"|err| {sel.inf_err.abs().mean():.4f};  training_end sum {sel.end_sim.sum() / sel.end_meas.sum():.4f} "
          f"|err| {sel.end_err.abs().mean():.4f};  busy {sel.busy_sim.sum() / sel.busy_meas.sum():.4f};  "
          f"overlap sum sim {sel.ovl_sim.sum():.0f} meas {sel.ovl_meas.sum():.0f}")
    df.to_csv(args.grid / "align_streaming.csv", index=False)


if __name__ == "__main__":
    main()
