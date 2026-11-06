"""Compare a simulated fixed-B sweep against the measured slime arms.

    uv run python tools/slime-b-sweep/analyze_sweep.py logs/<sweep_dir> [tag]

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
# The B values slime was measured at. A sweep over any other grid gets only the
# simulated speedup figure: there is nothing to compare the rest against.
MEASURED_BS = [16, 32, 64, 96, 128]
BS = MEASURED_BS
ROLLOUTS = 20
GPUS = 8
# The preset's `workers_per_train_group`; block k is GPUs 2k and 2k+1.
GPUS_PER_BLOCK = 2
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


def measured_cubic() -> tuple[np.ndarray, np.ndarray] | None:
    """slime's online CUBIC tuner arm: (B in effect, measured training_end) per rollout.

    `cubic_B16/` in the same sweep directory, replaying the same
    `lengths_20roll.json`, starting at B=16 and re-picking B after every rollout
    from the rollout's idle signal, clamped to [16, 128]. B is the threshold IN
    EFFECT during the rollout (`b_per_step.csv`'s `B`), which is what a grid cell
    at that B simulates. training_end is closed from report.json the same way
    as for the fixed arms.
    """
    path = HERE / "data" / "cubic_B16.csv"
    if not path.exists():
        return None
    with open(path) as f:
        rows = {int(r["rollout"]): r for r in csv.DictReader(f)}
    b = np.array([int(rows[r]["B"]) for r in range(ROLLOUTS)])
    end = np.array([
        float(rows[r]["inference_time_s"]) - float(rows[r]["overlap_time_s"])
        + float(rows[r]["training_time_s"])
        for r in range(ROLLOUTS)
    ])
    return b, end


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
    sim_w = np.zeros((ROLLOUTS, len(BS), 8))
    meas_w = np.zeros((ROLLOUTS, len(BS), 8))
    for j, b in enumerate(BS):
        for r in range(ROLLOUTS):
            sim_w[r, j] = windows(sweep / f"r{r}_{b}" / "raw" / "request_slo.parquet")
            meas_w[r, j] = np.array(meas[(b, r)])
    return sim_w, meas_w


def _grid(ax, values, labels, title, sub, diverge, vmin, vmax, fmt, mark_min=False, center=None,
          cmap=None):
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
        cmap=cmap or ("coolwarm" if diverge else "RdYlGn_r"),
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


def grid_speedup_figure(sweep: Path, out: Path, closed_loop: bool = False) -> None:
    """The simulated panel of `speedup_figure`, over a B grid finer than the arms.

    Same measured colocate denominator and colour scale; the measured B columns
    are bolded so the grid can be read against `b_sweep_speedup.png`.

    The tuner columns and dots are, by default, the measured CUBIC arm's own B
    trajectory read off this grid. With `closed_loop` they are instead the
    tuners replayed against THIS grid's own idle signal (`replay_cubic.py`):
    CUBIC and the linear +-16 law, one colour each -- the right reading for a
    counterfactual grid, where the measured trajectory answers a different run.
    """
    end = np.zeros((ROLLOUTS, len(BS)))
    for j, b in enumerate(BS):
        for r in range(ROLLOUTS):
            windows(sweep / f"r{r}_{b}" / "raw" / "request_slo.parquet")  # refuses a dead run
            end[r, j] = training_window(sweep / f"r{r}_{b}")[1]
    speedup = measured_colocate()[:, None] / end
    # The per-rollout oracle: each row's best B, as if the threshold could be
    # picked after seeing the rollout. Its mean row is the ceiling a fixed B is
    # measured against.
    pick = speedup.argmax(1)
    oracle = speedup.max(1)
    # The online tuners, read off this same grid: the cell at the B each had in
    # effect for each rollout. Only when every B used is a grid column.
    cubic = None if closed_loop else measured_cubic()
    if cubic is not None and not set(cubic[0]) <= set(BS):
        cubic = None
    tuners = []  # (column label, B per rollout, dot colour, dot x offset)
    if closed_loop:
        import replay_cubic

        for bs, label, color, dx in zip(
            replay_cubic.trajectories(sweep).values(),
            ("cubic", "linear", "+16/-32", "last 5", "all past"),
            ("black", "#00a651", "#8e24aa", "#ff9800", "#00acc1"), (-0.32, -0.16, 0.0, 0.16, 0.32)
        ):
            tuners.append((label, bs, color, dx))
    elif cubic is not None:
        tuners.append(("cubic", cubic[0], "black", 0.0))
    extra = [oracle]
    extra_labels = ["best"]
    for label, bs, _, _ in tuners:
        extra.append(np.array([speedup[r, BS.index(b)] for r, b in enumerate(bs)]))
        extra_labels.append(label)
    if cubic is not None:
        tuned = extra[1]
    fig, ax = plt.subplots(figsize=(0.42 * (len(BS) + len(extra)) + 2.5, 10))
    im = _grid(ax, np.column_stack([speedup, *extra]), [str(b) for b in BS] + extra_labels,
               "simulated speedup vs colocate",
               "measured colocate / simulated training_end; box = best B per row; "
               "bold B = measured arm; best = per-rollout oracle"
               + ("; black dot / cubic = measured CUBIC run's B" if cubic is not None else "")
               + ("; closed loop on this grid: black dot / cubic = CUBIC, "
                  "green = +-16, purple = +16/-32, orange / cyan = best B of the last 5 / all past rollouts" if closed_loop else ""),
               True, 0.65, 1.35, "{:.2f}", center=1.0)
    ax.set_title("simulated speedup vs colocate", fontsize=11, pad=24)
    ax.set_xlabel("migration threshold B")
    for label, b in zip(ax.get_xticklabels(), BS):
        label.set_fontweight("bold" if b in MEASURED_BS else "normal")
    for label in ax.get_xticklabels()[len(BS):]:
        label.set_fontweight("bold")
        label.set_fontsize(7.5)
        label.set_rotation(35)
        label.set_ha("right")
        label.set_rotation_mode("anchor")
    ax.axvline(len(BS) - 0.5, color="black", lw=1.4)
    best = np.vstack([speedup, speedup.mean(0)]).argmax(1)
    for i, j in enumerate(best):
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", lw=1.6))
    # Each extra column says which B it used, under its value.
    notes = {len(BS): [f"B={BS[j]}" for j in pick] + [f"B~{np.mean([BS[j] for j in pick]):.0f}"]}
    for k, (_, bs, color, dx) in enumerate(tuners):
        notes[len(BS) + 1 + k] = [f"B={b}" for b in bs] + [f"B~{bs.mean():.0f}"]
        for r, b in enumerate(bs):
            ax.plot(BS.index(b) + dx, r - 0.33, marker="o", ms=3.2 if len(tuners) < 5 else 2.6,
                    color=color)
        ax.get_xticklabels()[len(BS) + 1 + k].set_color(color)
    for text in list(ax.texts):
        x, y = text.get_position()
        if x in notes:
            text.set_position((x, y - 0.17))
            ax.text(x, y + 0.24, notes[x][int(y)], ha="center", va="center", fontsize=5,
                    color=text.get_color())
    fig.suptitle(
        f"slime B sweep, simulated ({sweep.name}) - time to the last gradient vs colocate\n"
        "colocate `inference_s + train_s` / streaming `training_end` (x"
        f"{K}); red = streaming finishes sooner",
        fontsize=11,
    )
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label("x faster")
    fig.savefig(out, dpi=140, bbox_inches="tight")

    mean = speedup.mean(0)
    print(f"{'B':>5} {'sim end':>8} {'sim x':>7}   (means over {ROLLOUTS} rollouts)")
    for j, b in enumerate(BS):
        print(f"{b:>5} {end[:, j].mean():8.1f} {mean[j]:7.3f}{'  *measured arm' if b in MEASURED_BS else ''}")
    print(f"best mean B = {BS[mean.argmax()]} ({mean.max():.3f}x)")
    print(f"per-rollout oracle = {oracle.mean():.3f}x (mean picked B {np.mean([BS[j] for j in pick]):.0f})")
    if cubic is None:
        return
    b_choice_figure(speedup, cubic[0], out.with_name(out.stem.replace("speedup_grid", "b_choice") + ".png"))
    colocate = measured_colocate()
    fixed = measured_training()
    print(f"\n{'policy':<26} {'sim x':>7} {'meas x':>7}   (mean over {ROLLOUTS} rollouts of colocate / training_end)")
    for b in MEASURED_BS:
        meas_end = np.array([fixed[(b, r)]["inference_time_s"] - fixed[(b, r)]["overlap_time_s"]
                             + fixed[(b, r)]["training_time_s"] for r in range(ROLLOUTS)])
        print(f"{'fixed B=' + str(b):<26} {mean[BS.index(b)]:7.3f} {(colocate / meas_end).mean():7.3f}")
    print(f"{'fixed B=' + str(BS[mean.argmax()]) + ' (sim best)':<26} {mean.max():7.3f} {'-':>7}")
    print(f"{'per-rollout oracle':<26} {oracle.mean():7.3f} {'-':>7}")
    print(f"{'online CUBIC, B0=16':<26} {tuned.mean():7.3f} {(colocate / cubic[1]).mean():7.3f}")
    ramp = slice(4, ROLLOUTS)  # after the slow-start ramp 16 -> 32 -> 64 -> 128
    print(f"  rollouts 4-19 only: cubic sim {tuned[ramp].mean():.3f}, sim best fixed "
          f"{speedup[ramp, mean.argmax()].mean():.3f}, oracle {oracle[ramp].mean():.3f}, "
          f"cubic meas {(colocate / cubic[1])[ramp].mean():.3f}")


def b_choice_figure(speedup: np.ndarray, tuned_b: np.ndarray, out: Path, tolerance=0.01) -> None:
    """Per rollout: the simulated best B, the near-best range, and the B the tuner used.

    The range is every grid B whose simulated speedup is within `tolerance` of
    that rollout's best -- drawn min to max, so it can span a dip. Its width is
    the point: a rollout whose band covers 64-200 barely cares what B it gets.
    """
    x = np.arange(ROLLOUTS)
    b = np.array(BS)
    best = b[speedup.argmax(1)]
    near = speedup >= speedup.max(1, keepdims=True) * (1 - tolerance)
    lo = np.array([b[row].min() for row in near])
    hi = np.array([b[row].max() for row in near])
    fixed = BS[speedup.mean(0).argmax()]

    blue, orange, ink, muted = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
    fig, ax = plt.subplots(figsize=(10, 4.8))
    # One bar per rollout, not a band: the rollouts are separate runs, and a
    # band would invent a range between them.
    ax.vlines(x, lo, hi, color=blue, alpha=0.22, lw=14, label=f"within {tolerance:.0%} of best (sim)")
    ax.plot(x, best, color=blue, lw=2, marker="o", ms=6, label="best B (sim)")
    ax.plot(x, tuned_b, color=orange, lw=2, marker="s", ms=6, label="CUBIC selected B (measured run)")
    ax.axhline(fixed, color=muted, lw=1.2, ls="--")
    ax.text(-0.6, fixed - 3, f"best fixed B = {fixed}", ha="left", va="top", fontsize=8, color=muted)
    ax.axhspan(16, 128, color="0.5", alpha=0.06, lw=0)
    ax.text(-0.6, 130, "tuner range [16, 128]", fontsize=8, color=muted, va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([f"r{i}" for i in x], fontsize=8)
    ax.set_yticks(range(0, max(BS) + 1, 32))
    ax.set_xlim(-0.8, ROLLOUTS - 0.2)
    ax.set_ylim(0, max(BS) + 8)
    ax.set_xlabel("rollout", color=ink)
    ax.set_ylabel("migration threshold B", color=ink)
    ax.grid(axis="y", color="0.9", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", fontsize=8, frameon=False, ncol=3)
    ax.set_title("slime rollouts - best B vs the B the online tuner picked", fontsize=11, color=ink)
    fig.savefig(out, dpi=140, bbox_inches="tight")

    inside = ((tuned_b >= lo) & (tuned_b <= hi) & near[x, [BS.index(v) for v in tuned_b]]).sum()
    print(f"\nCUBIC's B was within {tolerance:.0%} of the best in {inside}/{ROLLOUTS} rollouts; "
          f"near-best band width mean {np.mean(hi - lo):.0f}")


def _union_length(start: np.ndarray, end: np.ndarray) -> float:
    """Total length covered by a set of intervals, overlaps counted once."""
    order = np.argsort(start, kind="stable")
    start, reach = start[order], np.maximum.accumulate(end[order])
    first = np.flatnonzero(np.r_[True, start[1:] > reach[:-1]])
    last = np.r_[first[1:] - 1, len(start) - 1]
    return float((reach[last] - start[first]).sum())


def _intervals(table) -> tuple[np.ndarray, np.ndarray]:
    start = np.array(table["wall_start_ms"])
    return start, start + np.array(table["total_time_ms"])


def idle_gpu_seconds(cell: Path) -> tuple[np.ndarray, np.ndarray]:
    """(inference idle, training idle) per GPU of a cell, in wall seconds.

    Each GPU's timeline splits where its two-GPU block transitions to training:
    the moment both engines are empty, i.e. the later of the pair's last
    inference iterations. Before that the GPU is on the inference side, and its
    idle is the gaps between its own iterations -- mostly an engine that ran dry
    waiting for its partner. After it the GPU is on the training side up to
    `training_end`, and its idle is the gaps between its block's chunks -- a freed
    block waiting on the group queue, or one done while another still trains.
    """
    log = cell / "raw" / "cost_log"
    inference = [_intervals(pq.read_table(log / f"worker_main_{gpu}.parquet",
                                          columns=["wall_start_ms", "total_time_ms"]))
                 for gpu in range(GPUS)]
    blocks = GPUS // GPUS_PER_BLOCK
    chunks = [(np.zeros(0), np.zeros(0))] * blocks
    for path in log.glob("worker_main.train_*.parquet"):
        chunks[int(path.stem.rsplit("_", 1)[1])] = _intervals(
            pq.read_table(path, columns=["wall_start_ms", "total_time_ms"]))
    training_end = max(end.max() for _, end in chunks if len(end))

    inf_idle, train_idle = np.zeros(GPUS), np.zeros(GPUS)
    for block in range(blocks):
        gpus = range(block * GPUS_PER_BLOCK, (block + 1) * GPUS_PER_BLOCK)
        handover = max(inference[gpu][1].max() for gpu in gpus)
        start, end = chunks[block]
        # The trainer takes a block once both engines are empty, which the
        # simulator sees a tick or two before the last single-token iteration's
        # end is logged (~2 ms). Clipping the chunks at the handover counts that
        # overlap once, as inference -- the union. More than that would mean the
        # split point is wrong.
        assert not len(start) or start.min() >= handover - 10.0, (cell, block, handover - start.min())
        start = np.maximum(start, handover)
        trained = _union_length(start, end) if len(start) else 0.0
        for gpu in gpus:
            inf_idle[gpu] = handover - _union_length(*inference[gpu])
            train_idle[gpu] = (training_end - handover) - trained
    return inf_idle / 1000.0 * K, train_idle / 1000.0 * K


def idle_figure(sweep: Path, out: Path) -> None:
    """Idle GPU-seconds up to the last gradient, split at each block's handover."""
    inf = np.zeros((ROLLOUTS, len(BS)))
    train = np.zeros_like(inf)
    for j, b in enumerate(BS):
        for r in range(ROLLOUTS):
            i, t = idle_gpu_seconds(sweep / f"r{r}_{b}")
            inf[r, j], train[r, j] = i.sum(), t.sum()
    total = inf + train
    labels = [str(b) for b in BS]
    # One scale for all three, so the two parts read against their sum.
    vmax = np.percentile(total, 98)
    fig, ax = plt.subplots(3, 1, figsize=(0.42 * len(BS) + 2.5, 30))
    panels = (
        (inf, "inference idle GPU-seconds",
         "before the pair transitions to training: gaps between its own iterations"),
        (train, "training idle GPU-seconds",
         "after the transition, up to training_end: gaps between the block's chunks"),
        (total, "total idle GPU-seconds", "inference + training idle"),
    )
    for a, (values, name, sub) in zip(ax, panels):
        im = _grid(a, values, labels, name, sub, True, 0.0, vmax, "{:.0f}", center=0.0, cmap="Oranges")
        a.set_title(name, fontsize=11, pad=24)
        a.set_xlabel("migration threshold B")
        for label, b in zip(a.get_xticklabels(), BS):
            label.set_fontweight("bold" if b in MEASURED_BS else "normal")
        low = np.vstack([values, values.mean(0)]).argmin(1)
        for i, j in enumerate(low):
            a.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", lw=1.6))
        fig.colorbar(im, ax=a, fraction=0.03, pad=0.02).set_label("idle GPU-s")
    fig.suptitle(
        f"slime B sweep, simulated (x{K}) - idle GPU time until the last gradient\n"
        "darker = more idle (one scale for all panels); box = least idle B per row; "
        "bold B = measured arm",
        fontsize=11,
        y=0.91,
    )
    fig.savefig(out, dpi=140, bbox_inches="tight")

    print(f"\n{'B':>5} {'inf idle':>9} {'train idle':>11} {'total':>8}   "
          f"(GPU-s, means over {ROLLOUTS} rollouts)")
    for j, b in enumerate(BS):
        print(f"{b:>5} {inf[:, j].mean():9.1f} {train[:, j].mean():11.1f} {total[:, j].mean():8.1f}")
    print(f"least idle mean B = {BS[total.mean(0).argmin()]} ({total.mean(0).min():.1f} GPU-s)")


# `UnifiedStage` discriminants (simulator/src/common/request_stage.rs).
PENDING, SUSPENDED = 0, 4


def _peak(events: list[tuple[float, int]]) -> int:
    """Peak concurrency of +1/-1 events; a leave sorts before an enter at a tie."""
    events.sort()
    return int(np.cumsum([step for _, step in events]).max()) if events else 0


def engine_queues(cell: Path) -> np.ndarray:
    """Peak queue depth on the worst engine of a cell: [total, migrated, retracted].

    Rebuilt from the stage timeline: a request sits in an engine's queue from a
    `Pending` stamp until its next transition. Its FIRST pending stay is the t=0
    dispatch -- 128 samples per engine, identical at every B, so it would own the
    peak of every cell; it is left out. A later stay is either a migration
    arrival (the previous stamp is `Suspended`, on the source engine) or a KV
    retraction (the previous stamp is prefill/decode on this same engine).
    Each column is its own peak, so total can be less than the parts' sum.
    """
    table = pq.read_table(cell / "raw" / "request_slo.parquet",
                          columns=["stage_times_ms", "stage_codes", "stage_worker_ids"])
    queued = [[[] for _ in range(GPUS)] for _ in range(3)]
    for times, codes, workers in zip(*(table[c].to_pylist() for c in table.column_names)):
        for k in range(1, len(codes)):
            if codes[k] != PENDING:
                continue
            leave = times[k + 1] if k + 1 < len(times) else np.inf
            span = [(times[k], 1), (leave, -1)]
            queued[0][workers[k]] += span
            queued[1 if codes[k - 1] == SUSPENDED else 2][workers[k]] += span
    return np.array([max(map(_peak, kind)) for kind in queued])


def queue_figure(sweep: Path, out: Path) -> None:
    """Peak per-engine queueing after the dispatch, by cause."""
    peaks = np.zeros((ROLLOUTS, len(BS), 3))
    for j, b in enumerate(BS):
        for r in range(ROLLOUTS):
            peaks[r, j] = engine_queues(sweep / f"r{r}_{b}")
    labels = [str(b) for b in BS]
    vmax = np.percentile(peaks[:, :, 0], 98)
    fig, ax = plt.subplots(3, 1, figsize=(0.42 * len(BS) + 2.5, 30))
    panels = (
        ("peak queue depth on one engine", "pending after the t=0 dispatch, any cause"),
        ("... of which migrated in", "requests handed over by a released block, waiting to re-prefill"),
        ("... of which retracted", "decode evicted for KV on the same engine, waiting to re-prefill"),
    )
    for k, (a, (name, sub)) in enumerate(zip(ax, panels)):
        im = _grid(a, peaks[:, :, k], labels, name, sub, True, 0.0, vmax, "{:.0f}", center=0.0,
                   cmap="Purples")
        a.set_title(name, fontsize=11, pad=24)
        a.set_xlabel("migration threshold B")
        for label, b in zip(a.get_xticklabels(), BS):
            label.set_fontweight("bold" if b in MEASURED_BS else "normal")
        fig.colorbar(im, ax=a, fraction=0.03, pad=0.02).set_label("requests")
    fig.suptitle(
        "slime B sweep, simulated - worst per-engine queue during the run, excluding the t=0 dispatch\n"
        "cell = peak over the 8 engines (requests); one scale for all panels; bold B = measured arm",
        fontsize=11,
        y=0.91,
    )
    fig.savefig(out, dpi=140, bbox_inches="tight")

    print(f"\n{'B':>5} {'queue':>7} {'migrated':>9} {'retracted':>10}   "
          f"(peak on one engine, means over {ROLLOUTS} rollouts)")
    for j, b in enumerate(BS):
        print(f"{b:>5} " + " ".join(f"{peaks[:, j, k].mean():{w}.1f}" for k, w in enumerate((7, 9, 10))))


def nomig_figure(sweep: Path, out: Path) -> None:
    """Straggler idle with no migration: max(inference time) x 8 - sum of active time.

    Every engine keeps the 128 samples the trace gave it, so the rollout lasts as
    long as its slowest engine and everyone else waits. Per engine, the idle is
    `makespan - active`, with active the engine's summed iteration time; the
    total column is their sum, i.e. `8 x makespan - sum(active)`.
    """
    idle = np.zeros((ROLLOUTS, GPUS))
    makespans = np.zeros(ROLLOUTS)
    for r in range(ROLLOUTS):
        cell = sweep / f"r{r}"
        makespan = makespans[r] = windows(cell / "raw" / "request_slo.parquet").max()
        active = np.array([
            np.array(pq.read_table(cell / "raw" / "cost_log" / f"worker_main_{gpu}.parquet",
                                   columns=["total_time_ms"])["total_time_ms"]).sum()
            for gpu in range(GPUS)
        ]) / 1000.0 * K
        idle[r] = makespan - active
    total = idle.sum(1, keepdims=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 10), gridspec_kw={"width_ratios": [GPUS, 1.4]})
    im = _grid(ax[0], idle, [f"e{g}" for g in range(GPUS)], "idle per engine (s)",
               "makespan - that engine's inference active time", True, 0.0,
               np.percentile(idle, 98), "{:.0f}", center=0.0, cmap="Oranges")
    ax[0].set_title("idle per engine (s)", fontsize=11, pad=24)
    ax[0].set_xlabel("engine (the trace's group_index % 8)")
    fig.colorbar(im, ax=ax[0], fraction=0.05, pad=0.02).set_label("s")
    im = _grid(ax[1], total, ["total"], "8 x max - sum", "GPU-s", True, 0.0,
               np.percentile(total, 98), "{:.0f}", center=0.0, cmap="Oranges")
    ax[1].set_title("8 x max - sum", fontsize=11, pad=24)
    ax[1].set_yticklabels([])  # the rows are the left panel's
    fig.colorbar(im, ax=ax[1], fraction=0.3, pad=0.08).set_label("GPU-s")
    fig.suptitle(
        f"slime rollouts, simulated with no migration (x{K}) - straggler idle\n"
        "max(inference time) x 8 - per-engine inference active time; darker = more idle",
        fontsize=11,
    )
    fig.savefig(out, dpi=140, bbox_inches="tight")

    share = total[:, 0] / (GPUS * makespans) * 100.0
    print(f"{'rollout':>7} {'makespan':>9} {'idle GPU-s':>11} {'idle %':>7}")
    for r in range(ROLLOUTS):
        print(f"{r:>7} {makespans[r]:9.1f} {total[r, 0]:11.1f} {share[r]:7.1f}")
    print(f"{'mean':>7} {makespans.mean():9.1f} {total.mean():11.1f} {share.mean():7.1f}")


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
    global BS
    sweep = Path(sys.argv[1])
    root = Path(__file__).resolve().parents[2]
    if (sweep / "r0").is_dir():
        # A rollout-only sweep: the no-migration control, which has no B axis.
        tag = f"_{sys.argv[2]}" if len(sys.argv) > 2 else ""
        nomig_figure(sweep, root / f"nomig_idle{tag}.png")
        return
    BS = sorted(int(p.name.split("_")[1]) for p in sweep.glob("r0_*") if p.is_dir())
    if not set(BS) <= set(MEASURED_BS):
        # An optional second argument tags the figures, so a variant sweep (a
        # different trigger, say) does not overwrite the baseline's.
        tag = f"_{sys.argv[2]}" if len(sys.argv) > 2 else ""
        # A tagged grid is a counterfactual: replay the tuners on it rather than
        # reading the measured run's trajectory off it.
        grid_speedup_figure(sweep, root / f"b_sweep_speedup_grid{tag}.png", closed_loop=bool(tag))
        idle_figure(sweep, root / f"b_sweep_idle_grid{tag}.png")
        queue_figure(sweep, root / f"b_sweep_queue_grid{tag}.png")
        return
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
