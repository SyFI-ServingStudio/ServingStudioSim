"""Fixed-B arms and the closed-loop tuners, relative to the per-rollout oracle.

    uv run python tools/slime-b-sweep/plot_vs_oracle.py <grid> [<grid> ...] [--out file.png]

Each grid is one simulated B sweep (`r{0..19}_{B}`), one bar series per grid.
Performance = oracle total training_end / policy total training_end over the 20
rollouts; the oracle takes the best B in [16, 128] per rollout, the same range
the tuners are clamped to. The tuners are replayed closed loop on the grid
(`replay_cubic.replay`).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import replay_cubic as rc

FIXED = [16, 32, 64, 96, 128]
ADAPTIVE = {"CUBIC": rc.CubicTuner, "idle ±16": rc.LinearTuner, "KV cap + ±16": rc.KvCappedTuner}
COLORS = ["#0072B2", "#D55E00", "#009E73"]


def relative(grid: Path) -> dict[str, float]:
    bs, end = rc.fixed_ends(grid)
    oracle = end.min(1).sum()
    out = {f"Fixed\nB={b}": oracle / end[:, bs.index(b)].sum() for b in FIXED}
    for name, make in ADAPTIVE.items():
        out[f"Adaptive\n{name}"] = oracle / sum(o["end"] for _, _, o, _, _ in rc.replay(grid, make))
    return out


def main() -> None:
    args = sys.argv[1:]
    out = Path("b_vs_oracle.png")
    if "--out" in args:
        i = args.index("--out")
        out = Path(args[i + 1])
        del args[i:i + 2]
    grids = [Path(a.split("=", 1)[-1]) for a in args]
    labels = [a.split("=", 1)[0] if "=" in a else Path(a).name for a in args]
    series = [relative(g) for g in grids]
    cats = list(series[0])

    fig, ax = plt.subplots(figsize=(1.0 * len(cats) + 1.2, 3.0))
    width = 0.8 / len(series)
    x = np.arange(len(cats))
    first_adaptive = len(FIXED)
    ax.axvspan(first_adaptive - 0.5, len(cats) - 0.5, color="0.94", zorder=0)
    for k, (label, values) in enumerate(zip(labels, series)):
        pos = x - 0.4 + width * (k + 0.5)
        v = [values[c] for c in cats]
        ax.bar(pos, v, width * 0.92, color=COLORS[k % len(COLORS)], label=label, zorder=2)
        for p, val in zip(pos, v):
            ax.text(p, val - 0.004, f"{val:.3f}", rotation=90, ha="center", va="top",
                    color="white", fontsize=8, zorder=3)
        print(label, {c.replace("\n", " "): round(values[c], 4) for c in cats})
    ax.axhline(1.0, color="black", ls="--", lw=1.2, zorder=1)
    ax.text(-0.45, 1.002, "oracle (best B per rollout) = 1.00", fontsize=8, va="bottom")
    lo = min(min(s.values()) for s in series)
    ax.set_ylim(np.floor((lo - 0.02) * 20) / 20, 1.02)
    ax.set_xticks(x, cats)
    ax.set_xlim(-0.5, len(cats) - 0.5)
    ax.set_ylabel("Performance\nrelative to oracle", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9", zorder=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=len(series), frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
