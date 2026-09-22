"""Compare a simulated fixed-B sweep against the measured slime arms.

    uv run python tools/slime-b-sweep/analyze_sweep.py logs/<sweep_dir>

Reads `<sweep_dir>/r{0..19}_{16,32,64,96,128}/raw/request_slo.parquet`, pairs
each cell with `data/all_arms.csv`, and writes three figures next to the repo
root plus the summary tables on stdout.

Two metrics, and the difference between them is the finding:

* **makespan** -- the slowest engine. Depends on which engine ends up holding the
  tail, so it degrades as B rises and consolidation concentrates that tail.
* **active GPU-seconds** -- the sum of the engines' own activity windows. A
  conservation quantity: right token count times right cost per token. It stays
  accurate across the whole sweep, which is how we know the residual is
  distribution and not pricing.

The third figure sorts the engines by their own window before comparing, so
rank is matched rather than identity. Its mean strip shows a sign flip at rank
4/5 that neither aggregate metric reveals.

`K` converts simulated GPU time to wall clock. It is an end-to-end fit on the
no-migration arm, not an alignment-derived duty cycle, and the shape is known to
be wrong -- framework overhead is per iteration, not proportional to GPU time
(see progress.md). Kept here so the figures reproduce; replace it when the
worker gains a per-iteration overhead knob.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from matplotlib import gridspec

K = 1.1623
BS = [16, 32, 64, 96, 128]
ROLLOUTS = 20
HERE = Path(__file__).parent


def measured() -> dict[tuple[int, int], list[float]]:
    out = defaultdict(list)
    with open(HERE / "data" / "all_arms.csv") as f:
        for row in csv.DictReader(f):
            out[(int(row["B"]), int(row["rollout"]))].append(float(row["dur_s"]))
    return out


def windows(path: Path) -> np.ndarray:
    """Each worker's last moment of activity, which is its `inference` slice."""
    table = pq.read_table(path)
    if table.num_rows != 1024:
        # A Rust panic still leaves a partial parquet and no summary.json, and a
        # truncated run reads as a suspiciously fast cell rather than a failure.
        raise SystemExit(f"{path}: {table.num_rows} rows, expected 1024 -- the run died")
    last = defaultdict(float)
    for workers, times in zip(
        table["stage_worker_ids"].to_pylist(), table["stage_times_ms"].to_pylist()
    ):
        for worker, at in zip(workers, times):
            last[worker] = max(last[worker], at / 1000.0)
    return np.array([last[i] for i in range(8)]) * K


def collect(sweep: Path):
    meas = measured()
    sim_w = np.zeros((ROLLOUTS, 5, 8))
    meas_w = np.zeros((ROLLOUTS, 5, 8))
    for j, b in enumerate(BS):
        for r in range(ROLLOUTS):
            sim_w[r, j] = windows(sweep / f"r{r}_{b}" / "raw" / "request_slo.parquet")
            meas_w[r, j] = np.array(meas[(b, r)])
    return sim_w, meas_w


def _grid(ax, values, labels, title, sub, diverge, vmin, vmax, fmt, mark_min=False):
    shown = values if diverge else values / values[:, :1]
    im = ax.imshow(
        shown, aspect="auto", cmap="coolwarm" if diverge else "RdYlGn_r", vmin=vmin, vmax=vmax
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(ROLLOUTS))
    ax.set_yticklabels([f"r{i}" for i in range(ROLLOUTS)], fontsize=7)
    ax.set_title(title, fontsize=11, pad=14)
    ax.text(0.5, 1.013, sub, transform=ax.transAxes, ha="center", fontsize=8, color="0.35")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            hot = abs(values[i, j]) > 0.6 * vmax if diverge else not 0.90 < shown[i, j] < 1.14
            ax.text(
                j,
                i,
                fmt.format(values[i, j]),
                ha="center",
                va="center",
                fontsize=6.4,
                color="white" if hot else "black",
            )
    if mark_min:
        for i, j in enumerate(values.argmin(1)):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", lw=1.6))
    return im


def absolute_figure(sim_w, out: Path):
    mk, busy = sim_w.max(2), sim_w.sum(2)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 10))
    labels = [f"B={b}" for b in BS]
    im = _grid(ax[0], mk, labels, "Slowest GPU (makespan, s)",
               "lower = rollout finishes sooner", False, 0.82, 1.28, "{:.0f}", True)
    _grid(ax[1], busy, labels, "Active GPU-seconds",
          "lower = less GPU spent on inference", False, 0.82, 1.28, "{:.0f}", True)
    fig.suptitle(
        f"slime fixed-B sweep, simulated (x{K})\n"
        "cell = absolute value; colour = ratio to that rollout's B=16; box = row minimum",
        fontsize=11,
    )
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label("vs B=16")
    fig.savefig(out, dpi=140, bbox_inches="tight")


def error_figure(sim_w, meas_w, out: Path):
    e_mk = (sim_w.max(2) / meas_w.max(2) - 1) * 100
    e_gpu = (sim_w.sum(2) / meas_w.sum(2) - 1) * 100
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 10))
    labels = [f"B={b}" for b in BS]
    im = _grid(ax[0], e_mk, labels, "Slowest GPU (makespan)",
               f"sim x{K} vs measured, % error", True, -15, 15, "{:+.1f}")
    _grid(ax[1], e_gpu, labels, "Active GPU-seconds",
          f"sim x{K} vs measured, % error", True, -15, 15, "{:+.1f}")
    fig.suptitle(
        "slime fixed-B sweep - simulation error\n"
        "red = sim too slow / too expensive, blue = sim too fast / too cheap",
        fontsize=11,
    )
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label("% error")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    return e_mk, e_gpu


def rank_figure(sim_w, meas_w, out: Path):
    s = -np.sort(-sim_w, axis=2)
    m = -np.sort(-meas_w, axis=2)
    err = (s / m - 1) * 100
    flat = err.reshape(ROLLOUTS, 40)

    fig = plt.figure(figsize=(17, 9))
    gs = gridspec.GridSpec(2, 1, height_ratios=[20, 1.8], hspace=0.06)
    ax, strip = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    im = ax.imshow(flat, aspect="auto", cmap="coolwarm", vmin=-15, vmax=15)
    for i in range(ROLLOUTS):
        for j in range(40):
            ax.text(j, i, f"{flat[i, j]:.0f}", ha="center", va="center", fontsize=5.0,
                    color="white" if abs(flat[i, j]) > 9 else "black")
    for b in range(1, 5):
        ax.axvline(b * 8 - 0.5, color="black", lw=1.8)
        strip.axvline(b * 8 - 0.5, color="black", lw=1.8)
    for axis in (ax, strip):
        axis.set_xticks(range(40))
        axis.set_xticklabels([f"{i % 8 + 1}" for i in range(40)], fontsize=6.5)
    ax.set_yticks(range(ROLLOUTS))
    ax.set_yticklabels([f"r{i}" for i in range(ROLLOUTS)], fontsize=7)
    ax.set_ylabel("rollout")
    for b, value in enumerate(BS):
        ax.text(b * 8 + 3.5, -1.2, f"B={value}", ha="center", fontsize=11, fontweight="bold")

    col = flat.mean(0)[None, :]
    strip.imshow(col, aspect="auto", cmap="coolwarm", vmin=-15, vmax=15)
    for j in range(40):
        strip.text(j, 0, f"{col[0, j]:+.1f}", ha="center", va="center", fontsize=5.6,
                   color="white" if abs(col[0, j]) > 9 else "black")
    strip.set_yticks([0])
    strip.set_yticklabels(["mean"], fontsize=8)
    strip.set_xlabel(
        "engine rank within the run (1 = slowest / the makespan holder ... 8 = first to go idle)"
    )
    fig.suptitle(
        f"Per-rank engine error: sim (x{K}) vs measured, %\n"
        "engines sorted by their own active window, so rank is matched not identity; "
        "red = sim too slow, blue = sim too fast",
        fontsize=12,
    )
    fig.colorbar(im, ax=[ax, strip], fraction=0.018, pad=0.015).set_label("% error")
    fig.savefig(out, dpi=145, bbox_inches="tight")
    return err


def main() -> None:
    sweep = Path(sys.argv[1])
    root = Path(__file__).resolve().parents[2]
    sim_w, meas_w = collect(sweep)

    absolute_figure(sim_w, root / "b_sweep_heatmaps.png")
    e_mk, e_gpu = error_figure(sim_w, meas_w, root / "b_sweep_error_heatmaps.png")
    e_rank = rank_figure(sim_w, meas_w, root / "b_sweep_rank_error.png")

    print(f"{'B':>5} {'makespan':>9} {'GPU-s':>9}   (simulated means, s)")
    for j, b in enumerate(BS):
        print(f"{b:>5} {sim_w[:, j].max(1).mean():9.1f} {sim_w[:, j].sum(1).mean():9.1f}")

    print(f"\n{'B':>5} {'mk mean':>8} {'mk RMS':>7} {'gpu mean':>9} {'gpu RMS':>8}   (% error)")
    for j, b in enumerate(BS):
        print(f"{b:>5} {e_mk[:, j].mean():+8.2f} {np.sqrt((e_mk[:, j] ** 2).mean()):7.2f} "
              f"{e_gpu[:, j].mean():+9.2f} {np.sqrt((e_gpu[:, j] ** 2).mean()):8.2f}")
    print(f"\noverall mean |error|: makespan {np.abs(e_mk).mean():.2f}%  "
          f"GPU-s {np.abs(e_gpu).mean():.2f}%")

    print(f"\nper-rank mean error (%)\n{'B':>5} " + " ".join(f"{'#' + str(i + 1):>7}" for i in range(8)))
    for j, b in enumerate(BS):
        print(f"{b:>5} " + " ".join(f"{e_rank[:, j, i].mean():+7.2f}" for i in range(8)))

    # The means hide which rollouts carry the RMS. A handful of cells at ±10%
    # is a different problem from every cell at ±3%, and only the listing tells
    # them apart.
    bad = sorted(
        ((abs(e_mk[i, j]), i, j) for i in range(e_mk.shape[0]) for j in range(len(BS))),
        reverse=True,
    )[:10]
    print(f"\nworst makespan cells\n{'rollout':>7} {'B':>5} {'sim':>8} {'meas':>8} {'err%':>7}")
    for _, i, j in bad:
        print(f"{i:>7} {BS[j]:>5} {sim_w[i, j].max():8.1f} {meas_w[i, j].max():8.1f} "
              f"{e_mk[i, j]:+7.2f}")


if __name__ == "__main__":
    main()
