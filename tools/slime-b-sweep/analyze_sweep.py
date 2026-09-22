"""Compare a simulated fixed-B sweep against the measured slime arms.

    uv run python tools/slime-b-sweep/analyze_sweep.py logs/<sweep_dir>

Reads `<sweep_dir>/r{0..19}_{16,32,64,96,128}/raw/request_slo.parquet`, pairs
each cell with `data/all_arms.csv`, and writes five figures next to the repo
root plus the summary tables on stdout. The last two appear only when the runs
carry a `train` section, and are paired with `data/train_times.csv` and
`data/colocate_times.csv` instead.

Two metrics, and the difference between them is the finding:

* **makespan** -- the slowest engine. Depends on which engine ends up holding the
  tail, so it degrades as B rises and consolidation concentrates that tail.
* **active GPU-seconds** -- the sum of the engines' own activity windows. A
  conservation quantity: right token count times right cost per token. It stays
  accurate across the whole sweep, which is how we know the residual is
  distribution and not pricing.

With training simulated, three more, read off the `train` section of each cell's
`cost_log` and paired with `data/train_times.csv`:

* **training_time** -- first chunk start to last chunk end. Mostly determined by
  the rollout's token count, so it is the check on the chunk cost.
* **overlap** -- how much of that ran while generation was still going. This is
  the benefit streaming RL exists to produce and the quantity most sensitive to
  the release policy, so it is the check on WHEN blocks are freed rather than
  what they then cost.
* **training_end** -- the thing an RL run actually waits for.

If overlap matches and training_time does not, re-calibrate the rate; if
training_time matches and overlap does not, the release model is wrong.

The fifth figure drops the error framing and asks what the experiment was run to
find out: `training_end` against the colocate arm of the same sweep, measured and
simulated side by side. That arm recorded the very lengths the streaming arms
replayed, so it is a matched baseline; it does route generation differently
(sgl-router `cache_aware` vs `group_index % 8`), which is a confound in the
generation half of the span and is why the figure is captioned, not just plotted.

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


def measured_training() -> dict[tuple[int, int], dict[str, float]]:
    out = {}
    with open(HERE / "data" / "train_times.csv") as f:
        for row in csv.DictReader(f):
            out[(int(row["B"]), int(row["rollout"]))] = {
                k: float(v) for k, v in row.items() if k not in ("B", "rollout")
            }
    return out


def measured_colocate() -> np.ndarray:
    """The colocate arm's `inference_s + train_s`, one entry per rollout.

    Same sweep directory, same box, same 20 rollouts: the colocate arm ran first
    with natural generation and RECORDED `lengths_20roll.json`, which every
    `batch_thresh_agg_*` arm then replayed. So this is a matched baseline on the
    identical workload, not a nearby run.

    `inference_s + train_s` is the like-for-like counterpart of `training_end`:
    both run from rollout start to the last gradient computed and both leave the
    weight update out. Colocate has no overlap to subtract -- that is the point.
    """
    with open(HERE / "data" / "colocate_times.csv") as f:
        rows = {int(r["rollout"]): float(r["inference_s"]) + float(r["train_s"])
                for r in csv.DictReader(f)}
    return np.array([rows[r] for r in range(ROLLOUTS)])


def training_window(cell: Path) -> tuple[float, float, int]:
    """(start, end, chunks) of a cell's training, in wall seconds.

    The `train` section of the per-block `cost_log` is the record -- one row per
    chunk, `wall_start_ms` + `total_time_ms`. Scaled by K like every other
    simulated time: the chunk constants are configured in the same (fast) clock,
    so one factor puts generation and training on the measured axis together.
    """
    starts, ends, chunks = [], [], 0
    for path in sorted((cell / "raw" / "cost_log").glob("worker_main.train_*.parquet")):
        table = pq.read_table(path)
        if table.num_rows == 0:
            continue
        start = np.array(table["wall_start_ms"])
        end = start + np.array(table["total_time_ms"])
        starts.append(start.min())
        ends.append(end.max())
        chunks += table.num_rows
    if not starts:
        raise SystemExit(f"{cell}: no train rows -- was the preset's `training` left off?")
    return min(starts) / 1000.0 * K, max(ends) / 1000.0 * K, chunks


def collect(sweep: Path):
    meas = measured()
    sim_w = np.zeros((ROLLOUTS, 5, 8))
    meas_w = np.zeros((ROLLOUTS, 5, 8))
    for j, b in enumerate(BS):
        for r in range(ROLLOUTS):
            sim_w[r, j] = windows(sweep / f"r{r}_{b}" / "raw" / "request_slo.parquet")
            meas_w[r, j] = np.array(meas[(b, r)])
    return sim_w, meas_w


def _grid(ax, values, labels, title, sub, diverge, vmin, vmax, fmt, mark_min=False, center=None):
    """One 20 x 5 grid plus a mean row.

    `diverge` picks what the colour means: True paints the value itself against a
    scale centred on `center` (0 for an error, 1 for a speedup), False paints
    each row's ratio to its own B=16 cell while the text stays absolute.

    The column mean rides along as a 21st row rather than a separate strip: it
    has to share the colour scale to be readable against the cells above it, and
    a rule under r19 keeps it from being mistaken for a rollout.
    """
    values = np.vstack([values, values.mean(0)])
    center = 0.0 if center is None else center
    shown = values if diverge else values / values[:, :1]
    im = ax.imshow(
        shown,
        aspect="auto",
        cmap="coolwarm" if diverge else "RdYlGn_r",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(ROLLOUTS + 1))
    ax.set_yticklabels([f"r{i}" for i in range(ROLLOUTS)] + ["mean"], fontsize=7)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.axhline(ROLLOUTS - 0.5, color="black", lw=1.4)
    ax.set_title(title, fontsize=11, pad=14)
    ax.text(0.5, 1.013, sub, transform=ax.transAxes, ha="center", fontsize=8, color="0.35")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            hot = (
                abs(values[i, j] - center) > 0.6 * (vmax - center)
                if diverge
                else not 0.90 < shown[i, j] < 1.14
            )
            ax.text(
                j,
                i,
                fmt.format(values[i, j]),
                ha="center",
                va="center",
                fontsize=6.4,
                color="white" if hot else "black",
                fontweight="bold" if i == ROLLOUTS else "normal",
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


TRAINING_NAMES = ("training_time", "overlap", "training_end")


def training_errors(sweep: Path, sim_w: np.ndarray):
    """Simulated vs measured training_time / overlap / training_end, per cell.

    Returns `(sim, meas, chunks)`, each `[rollout, B, metric]` with `metric` in
    [`TRAINING_NAMES`] order.
    """
    meas = measured_training()
    span = np.zeros((ROLLOUTS, len(BS), 3))
    ref = np.zeros_like(span)
    chunks = np.zeros((ROLLOUTS, len(BS)))
    for j, b in enumerate(BS):
        for r in range(ROLLOUTS):
            start, end, n = training_window(sweep / f"r{r}_{b}")
            generation_end = sim_w[r, j].max()
            span[r, j] = (end - start, max(0.0, generation_end - start), end)
            m = meas[(b, r)]
            ref[r, j] = (
                m["training_time_s"],
                m["overlap_time_s"],
                # The measured report gives no absolute end, but the identity
                # `total = inference + training - overlap + gsync + wupdate`
                # fixes it: training ends `training_time` after it started, and
                # it started `overlap` before generation finished.
                m["inference_time_s"] - m["overlap_time_s"] + m["training_time_s"],
            )
            chunks[r, j] = n
    return span, ref, chunks


def training_figure(span: np.ndarray, ref: np.ndarray, out: Path) -> np.ndarray:
    """One 20 x 5 error grid per training metric, same shape as the other figures.

    On a tighter colour scale than `b_sweep_error_heatmaps.png` (+-8% vs +-15%):
    the training residual is several times smaller than the makespan one, and on
    the generation figure's scale every cell would read as white. The three read
    together -- overlap says whether blocks were freed at the right moment,
    training_time whether a chunk was priced right, and training_end is what the
    RL run actually waits for.
    """
    err = (span - ref) / ref * 100.0
    fig, ax = plt.subplots(1, 3, figsize=(15, 10))
    labels = [f"B={b}" for b in BS]
    subs = (
        "first chunk -> last chunk (chunk cost)",
        "training hidden behind generation (release timing)",
        "when the last chunk lands (the goal)",
    )
    for k, (name, sub) in enumerate(zip(TRAINING_NAMES, subs)):
        im = _grid(ax[k], err[:, :, k], labels, name, sub, True, -8, 8, "{:+.1f}")
        # Three narrow panels put the title right on top of its own subtitle
        # (which `_grid` anchors in axes coords, so it does not move with it).
        ax[k].set_title(name, fontsize=11, pad=24)
    fig.suptitle(
        "slime fixed-B sweep - training-phase simulation error\n"
        f"sim x{K} vs measured report.json, % error; "
        "red = sim too slow / too late, blue = sim too fast / too early",
        fontsize=11,
    )
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02).set_label("% error")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    return err


def speedup_figure(span: np.ndarray, ref: np.ndarray, out: Path):
    """What streaming actually buys, measured and simulated, against colocate.

    The error figures answer "does the simulator agree with the measurement";
    this one answers the question the measurement was taken to settle, and the
    simulator only earns the right to be on the same page because it does.

    All three panels share one scale centred on parity, which is the comparison
    worth making: the first two are the streaming gain (~1.07-1.16x) and the
    third is the migration threshold's own contribution on top of it (~1.08x at
    best). Streaming is the big lever; B is the trim.

    Panels 1 and 2 divide the SAME measured colocate baseline by the measured and
    the simulated `training_end`, so their difference is exactly the
    `training_end` residual -- which is why this figure can be read as a claim
    about the design and not about the fit.
    """
    colocate = measured_colocate()[:, None]
    meas, sim = ref[:, :, 2], span[:, :, 2]
    panels = (
        (colocate / meas, "measured speedup vs colocate", "same box, same 20 rollouts of lengths"),
        (colocate / sim, "simulated speedup vs colocate", "measured colocate / simulated training_end"),
        (meas[:, :1] / meas, "measured speedup vs B=16", "the threshold alone, streaming already on"),
    )
    fig, ax = plt.subplots(1, 3, figsize=(15, 10))
    labels = [f"B={b}" for b in BS]
    for k, (values, name, sub) in enumerate(panels):
        im = _grid(ax[k], values, labels, name, sub, True, 0.65, 1.35, "{:.2f}", center=1.0)
        ax[k].set_title(name, fontsize=11, pad=24)
    fig.suptitle(
        "slime fixed-B sweep - time to the last gradient, relative\n"
        "colocate `inference_s + train_s` vs streaming `training_end`; "
        "red = streaming finishes sooner, blue = slower",
        fontsize=11,
    )
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02).set_label("x faster")
    fig.savefig(out, dpi=140, bbox_inches="tight")

    print(f"\n{'B':>5} {'colocate':>9} {'meas end':>9} {'sim end':>9} "
          f"{'meas x':>7} {'sim x':>7} {'vs B=16':>8}   (s, means over 20 rollouts)")
    for j, b in enumerate(BS):
        print(f"{b:>5} {colocate.mean():9.1f} {meas[:, j].mean():9.1f} {sim[:, j].mean():9.1f} "
              f"{(colocate[:, 0] / meas[:, j]).mean():7.3f} "
              f"{(colocate[:, 0] / sim[:, j]).mean():7.3f} "
              f"{(meas[:, 0] / meas[:, j]).mean():8.3f}")
    # A per-cell win rate, because a mean above 1 is compatible with losing half
    # the rollouts and the colocate arm's own generation is noisy.
    won = (colocate / meas > 1).sum(0)
    print("rollouts where streaming won: " + "  ".join(
        f"B={b} {won[j]}/{ROLLOUTS}" for j, b in enumerate(BS)))


def training_table(span: np.ndarray, ref: np.ndarray, chunks: np.ndarray, err: np.ndarray) -> None:
    print(f"\n{'B':>5} " + "  ".join(
        f"{n + ' sim':>16} {'meas':>7} {'err%':>7}" for n in TRAINING_NAMES
    ))
    for j, b in enumerate(BS):
        cells = "  ".join(
            f"{span[:, j, k].mean():16.1f} {ref[:, j, k].mean():7.1f} {err[:, j, k].mean():+7.2f}"
            for k in range(3)
        )
        print(f"{b:>5} {cells}")
    print("\noverall mean |error|: " + "  ".join(
        f"{n} {np.abs(err[:, :, k]).mean():.2f}%" for k, n in enumerate(TRAINING_NAMES)
    ))
    # The measured side has no honest counterpart: `train_metrics` logs about 60
    # of the 64 grabs a rollout makes, so its chunk count is a floor, not a count.
    print(f"chunks per rollout: sim {chunks.mean():.1f} (the grab policy's own count)")

    # Same listing the makespan side gets: a handful of cells at +-6% is a
    # different problem from every cell at +-2%.
    end = err[:, :, 2]
    bad = sorted(
        ((abs(end[i, j]), i, j) for i in range(ROLLOUTS) for j in range(len(BS))), reverse=True
    )[:10]
    print(f"\nworst training_end cells\n{'rollout':>7} {'B':>5} {'sim':>8} {'meas':>8} {'err%':>7}")
    for _, i, j in bad:
        print(f"{i:>7} {BS[j]:>5} {span[i, j, 2]:8.1f} {ref[i, j, 2]:8.1f} {end[i, j]:+7.2f}")


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

    if (sweep / "r0_16" / "raw" / "cost_log").exists():
        span, ref, chunks = training_errors(sweep, sim_w)
        err = training_figure(span, ref, root / "b_sweep_training_error.png")
        training_table(span, ref, chunks, err)
        speedup_figure(span, ref, root / "b_sweep_speedup.png")


if __name__ == "__main__":
    main()
