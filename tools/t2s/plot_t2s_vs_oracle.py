"""Text2SQL: fixed-B arms and the closed-loop tuners, relative to the per-rollout oracle.

    uv run python tools/t2s/plot_t2s_vs_oracle.py <grid> [--out file.png]

<grid> is the simulated streaming sweep (`r{0..34}_{B}`, B on a lattice of 8,
presets/t2s/streaming_grid.yaml). Full KV only, one bar series.

Performance = oracle total training_end / policy total training_end over the 35
rollouts, where training_end is the rollout's last training chunk end (the
rollout's wall, dispatch at t = 0). The oracle takes the best B of the grid per
rollout; the tuners are clamped to the same range. A simulated rollout does not
depend on the rollouts before it, so walking the grid cell by cell with the
tuner's own B is the closed loop (tools/slime-b-sweep/replay_cubic.py).

With --nomig (the migration-off cells of the same workload, `r{N}`), two
baselines lead the bars: "Colocated" (t2s_cells.colocated_end: generation to the
last engine, then every block trains) and "No migration" (a block trains once its
own engines drain; nothing is moved).

Tuners are slime's (ported in replay_cubic.py) with the measured bt_tuner_B8_step8
arm's settings: B0 = 8, b_min = 8, b_max = 256, linear step 8, rollout 0
skipped. CUBIC snaps to multiples of 8 -- the grid lattice and the measured
arm's step -- where slime would snap to n_samples_per_prompt (5 here); the
linear laws move by 8 and stay on the lattice. KV cap + +-step keeps the
slime-b-sweep ceiling rule unchanged (migrated-in queue above 20 on one engine).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "slime-b-sweep"))
import replay_cubic as rc  # noqa: E402
import t2s_cells as tc  # noqa: E402

ROLLOUTS = 35
B0, B_MIN, B_MAX, LATTICE = 8, 8, 256, 8
FIXED = [8, 16, 32, 64, 128, 256]


def adaptive(b0: int, step: int) -> dict:
    return {
        "CUBIC": lambda: rc.CubicTuner(initial=b0, b_min=B_MIN, b_max=B_MAX, snap_quantum=LATTICE),
        f"idle ±{step}": lambda: rc.LinearTuner(initial=b0, b_min=B_MIN, b_max=B_MAX, step=step),
        f"KV cap + ±{step}": lambda: rc.KvCappedTuner(initial=b0, b_min=B_MIN, b_max=B_MAX, step=step),
    }


COLOR = "#0072B2"


def grid_bs(grid: Path) -> list[int]:
    return sorted(int(p.name.split("_")[1]) for p in grid.glob("r0_*") if p.is_dir())


def fixed_ends(grid: Path, bs: list[int]) -> np.ndarray:
    return np.array([[tc.training_window(grid / f"r{r}_{b}")[1] for b in bs] for r in range(ROLLOUTS)])


def replay(grid: Path, bs: list[int], make) -> list[tuple[int, int, float, float, str]]:
    """(rollout, B, training_end, idle ratio, reason) along the tuner's own path."""
    tuner, out = make(), []
    for r in range(ROLLOUTS):
        b = tuner.current
        assert b in bs, f"B={b} is not a column of {grid}"
        cell = grid / f"r{r}_{b}"
        obs = tc.observation(cell)
        queue = int(tc.engine_queues(cell)[1]) if getattr(tuner, "needs_queue", False) else None
        tuner.update(r, obs["ratio"], queue)
        out.append((r, b, tc.training_window(cell)[1], obs["ratio"], tuner.reason))
    return out


def main() -> None:
    global ROLLOUTS
    ap = argparse.ArgumentParser()
    ap.add_argument("grid", type=Path)
    ap.add_argument("--out", type=Path, default=Path("t2s_b_vs_oracle.png"))
    # Defaults are Text2SQL's; the DAPO grid (logs/dapo/presets/grid.yaml) is
    # --rollouts 20 --label "DeepSeek-R1-8B DAPO, full KV".
    ap.add_argument("--rollouts", type=int, default=ROLLOUTS)
    ap.add_argument("--b0", type=int, default=B0)
    ap.add_argument("--step", type=int, default=LATTICE)
    ap.add_argument("--label", default="Text2SQL, full KV")
    ap.add_argument("--only-kv-cap", action="store_true", help="drop the CUBIC and idle bars")
    ap.add_argument("--nomig", type=Path, help="migration-off cells r{N}: adds Colocated and No migration")
    args = ap.parse_args()
    ROLLOUTS = args.rollouts

    bs = [b for b in grid_bs(args.grid) if B_MIN <= b <= B_MAX]
    end = fixed_ends(args.grid, bs)
    oracle = end.min(1).sum()
    best = [bs[i] for i in end.argmin(1)]
    print(f"oracle total {oracle:.1f} s; best B per rollout {best}")
    values = {}
    if args.nomig:
        cells = [args.nomig / f"r{r}" for r in range(ROLLOUTS)]
        values["Colocated"] = oracle / sum(tc.colocated_end(c) for c in cells)
        values["No\nmigration"] = oracle / sum(tc.training_window(c)[1] for c in cells)
    baselines = len(values)
    values |= {f"Fixed\nB={b}": oracle / end[:, bs.index(b)].sum() for b in FIXED}
    tuners = adaptive(args.b0, args.step)
    if args.only_kv_cap:
        tuners = {k: v for k, v in tuners.items() if k.startswith("KV cap")}
    for name, make in tuners.items():
        path = replay(args.grid, bs, make)
        values[f"Adaptive\n{name}"] = oracle / sum(e for _, _, e, _, _ in path)
        print(f"{name}: B path {[b for _, b, *_ in path]}")
    k = int(np.argmin([end[:, i].sum() for i in range(len(bs))]))
    print(f"best single fixed B = {bs[k]}: {oracle / end[:, k].sum():.4f}")
    cats = list(values)
    print({c.replace("\n", " "): round(v, 4) for c, v in values.items()})

    fig, ax = plt.subplots(figsize=(1.0 * len(cats) + 1.2, 3.0))
    x = np.arange(len(cats))
    ax.axvspan(baselines + len(FIXED) - 0.5, len(cats) - 0.5, color="0.94", zorder=0)
    if baselines:
        ax.axvline(baselines - 0.5, color="0.6", lw=0.8, zorder=1)
    v = [values[c] for c in cats]
    ax.bar(x, v, 0.6, color=COLOR, label=args.label, zorder=2)
    # 0.90 is the Llama3-8B b_vs_oracle.png axis; lower only when a bar would be cut.
    lo = min(0.90, np.floor((min(v) - 0.02) * 20) / 20)
    ax.set_ylim(lo, 1.02)
    inside = 0.04 * (1.02 - lo) / 0.12  # room a rotated label needs, scaled to the axis
    for p, val in zip(x, v):
        if val - lo > inside:
            ax.text(p, val - 0.004, f"{val:.3f}", rotation=90, ha="center", va="top",
                    color="white", fontsize=8, zorder=3)
        else:
            ax.text(p, val + 0.004, f"{val:.3f}", rotation=90, ha="center", va="bottom",
                    color="black", fontsize=8, zorder=3)
    ax.axhline(1.0, color="black", ls="--", lw=1.2, zorder=1)
    ax.text(-0.45, 1.002, "oracle (best B per rollout) = 1.00", fontsize=8, va="bottom")
    ax.set_xticks(x, cats)
    ax.set_xlim(-0.5, len(cats) - 0.5)
    ax.set_ylabel("Performance\nrelative to oracle", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9", zorder=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=1, frameon=False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
