"""Per-B speedup and the adaptive tuners, as CSV, for DeepSeek-R1-8B DAPO and Text2SQL.

    uv run python tools/t2s/b_speedup_tables.py \
        --dapo-grid logs/dapo/runs/grid --dapo-nomig logs/dapo/runs/nomig \
        --t2s-grid logs/t2s/runs/streaming_grid --t2s-nomig logs/t2s/runs/streaming_nomig \
        --out-dir logs/t2s/b_speedup

Speedup = simulated colocated wall / simulated training_end, both on the same
trace (t2s_cells.colocated_end on the migration-off cells). Two numbers per
policy: `speedup_mean` (mean of the per-rollout ratios) and `speedup_total` (sum
colocated / sum training_end). `speedup_vs_measured` divides the measured colocate
arm's total `inference_s + train_s` instead (DAPO's simulated seconds scaled by
analyze_sweep.K first, as its speedup figures do). `rel_oracle` = oracle total
training_end / policy total training_end, the b_vs_oracle.png bar height.

Both workloads use the whole grid, [8, 256], as the oracle and tuner range, as
plot_t2s_vs_oracle.py does, and the same tuner settings as the measured Text2SQL
bt_tuner_B8_step8 arm: B0 = 8, linear step +-8.

Why the simulated denominator: the DAPO colocate arm replayed the same lengths as
its streaming grid, but the Text2SQL colocate arm (colocate_idle_measurement) is a
separate run whose rollouts drew different samples -- its long-tail chains land in
other steps (rollout 1: 153 s of generation there, 50 s in the streaming arm).
On its own trace the simulated colocated generation matches it (122.6 vs 122.7 s
mean), so the gap is the sample draw, not the model.

Two baseline rows come from each workload's migration-off cells: "colocated
(sim)" (t2s_cells.colocated_end) and "no migration" (blocks train as their own
engines drain).

Writes `<out-dir>/{per_b,policies,per_rollout}.csv`.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "slime-b-sweep"))
import analyze_sweep as sw  # noqa: E402
import replay_cubic as rc  # noqa: E402
import t2s_cells as tc  # noqa: E402

B0, LO, HI, STEP = 8, 8, 256, 8
T2S_REPORT = (HERE / "data" / "arms" / "StreamTrainer_RollPackerFaithfulQueue_35rollout_colocate_idle_measurement"
              / "report.csv")


def t2s_colocate(rollouts: int) -> np.ndarray:
    with open(T2S_REPORT) as f:
        rows = {int(r["rollout_id"]): float(r["inference_s"]) + float(r["train_s"]) for r in csv.DictReader(f)}
    return np.array([rows[r] for r in range(rollouts)])


def replay(grid: Path, make, rollouts: int) -> list[tuple[int, float]]:
    """(B, training_end) along the tuner's own path on the grid."""
    tuner, out = make(), []
    for r in range(rollouts):
        b = tuner.current
        cell = grid / f"r{r}_{b}"
        queue = int(tc.engine_queues(cell)[1]) if getattr(tuner, "needs_queue", False) else None
        tuner.update(r, tc.observation(cell)["ratio"], queue)
        out.append((b, tc.training_window(cell)[1]))
    return out


def workloads(args):
    # DAPO's simulated seconds reach wall clock through slime-b-sweep's end-to-end
    # factor K (its preset leaves the worker at gpu_time_multiplier 1); Text2SQL's
    # preset carries its fitted multiplier, so its seconds are used as is.
    yield dict(name="deepseek_r1_8b_dapo", grid=args.dapo_grid, nomig=args.dapo_nomig,
               colocate=sw.measured_colocate(), scale=sw.K)
    yield dict(name="text2sql_qwen25_coder_7b", grid=args.t2s_grid, nomig=args.t2s_nomig,
               colocate=t2s_colocate(35), scale=1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dapo-grid", type=Path, required=True)
    ap.add_argument("--dapo-nomig", type=Path, required=True)
    ap.add_argument("--t2s-grid", type=Path, required=True)
    ap.add_argument("--t2s-nomig", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_b, policies, per_rollout = [], [], []
    for w in workloads(args):
        colo, grid, k = w["colocate"], w["grid"], w["scale"]
        bs = sorted(int(p.name.split("_")[1]) for p in grid.glob("r0_*") if p.is_dir())
        n = len(colo)
        end = k * np.array([[tc.training_window(grid / f"r{r}_{b}")[1] for b in bs] for r in range(n)])
        rng = [i for i, b in enumerate(bs) if LO <= b <= HI]
        oracle_end = end[:, rng].min(1)
        oracle_b = [bs[rng[i]] for i in end[:, rng].argmin(1)]

        cells = [w["nomig"] / f"r{r}" for r in range(n)]
        sim_colo = k * np.array([tc.colocated_end(c) for c in cells])

        def stats(ends):
            return {
                "speedup_mean": round(float((sim_colo / ends).mean()), 4),
                "speedup_total": round(float(sim_colo.sum() / ends.sum()), 4),
                "speedup_vs_measured": round(float(colo.sum() / ends.sum()), 4),
                "rel_oracle": round(float(oracle_end.sum() / ends.sum()), 4),
                "training_end_total_s": round(float(ends.sum()), 1),
            }

        def row(policy, b_path, ends, b_label=""):
            ends = np.asarray(ends)
            if not policy.startswith(("fixed", "best", "oracle")):
                # per_b.csv also carries the baselines and the tuners, so one file
                # holds every bar: colocated, B = 0 (no migration), each B, adaptive.
                per_b.append({"workload": w["name"], "policy": policy, "B": b_label,
                              "mean_B": round(float(np.mean(b_path)), 1)} | stats(ends))
            policies.append({
                "workload": w["name"], "policy": policy, "B": b_label,
                "mean_B": round(float(np.mean(b_path)), 1),
                "speedup_mean": round(float((sim_colo / ends).mean()), 4),
                "speedup_total": round(float(sim_colo.sum() / ends.sum()), 4),
                "speedup_vs_measured": round(float(colo.sum() / ends.sum()), 4),
                "rel_oracle": round(float(oracle_end.sum() / ends.sum()), 4),
                "training_end_total_s": round(float(ends.sum()), 1),
            })
            for r, (b, e) in enumerate(zip(b_path, ends)):
                per_rollout.append({"workload": w["name"], "policy": policy, "rollout": r, "B": b,
                                    "training_end_s": round(float(e), 2),
                                    "colocated_sim_s": round(float(sim_colo[r]), 2),
                                    "colocate_measured_s": round(float(colo[r]), 2),
                                    "speedup": round(float(sim_colo[r] / e), 4)})

        row("colocated (sim)", [0] * n, sim_colo, "colocated")
        row("no migration", [0] * n, [k * tc.training_window(c)[1] for c in cells], 0)
        for j, b in enumerate(bs):
            per_b.append({"workload": w["name"], "policy": f"fixed B={b}", "B": b,
                          "mean_B": b} | stats(end[:, j]))
        for j, b in enumerate(bs):
            if LO <= b <= HI:
                row(f"fixed B={b}", [b] * n, end[:, j], b)
        best = max(rng, key=lambda j: -end[:, j].sum())
        row(f"best fixed (B={bs[best]})", [bs[best]] * n, end[:, best], bs[best])
        row(f"oracle [{LO},{HI}]", oracle_b, oracle_end)
        tuners = {
            "CUBIC": lambda: rc.CubicTuner(initial=B0, b_min=LO, b_max=HI, snap_quantum=STEP),
            f"idle ±{STEP}": lambda: rc.LinearTuner(initial=B0, b_min=LO, b_max=HI, step=STEP),
            f"KV cap + ±{STEP}": lambda: rc.KvCappedTuner(initial=B0, b_min=LO, b_max=HI, step=STEP),
        }
        for name, make in tuners.items():
            path = replay(grid, make, n)
            row(f"adaptive {name}", [b for b, _ in path], [k * e for _, e in path], "adaptive")

    for name, rows in (("per_b", per_b), ("policies", policies), ("per_rollout", per_rollout)):
        with open(args.out_dir / f"{name}.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print(f"wrote {args.out_dir / name}.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
