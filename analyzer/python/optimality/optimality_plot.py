"""Render the optimality sub-optimality waterfall.

The Rust `optimality` subject emits, per aggregation **level** (cluster / pool /
worker / iteration), six telescoping buckets that sum to that level's Real GPU·s:
`idle | imbalance | batching | communication | hardware_gap | hardware_optimal`.
It also emits a per-kernel array (the balanced Real of each location split into
batching / communication / hardware_gap / hardware_optimal).

The overview emits four figures with the same horizontal stacked-bar shape aligned to the optimal
floor (the green `hardware_optimal` segment sits at the left, waste grows
rightward to Real). Each of the two views ships in an absolute (GPU·s) and a
normalized (every bar scaled to its own Real = 100%) variant:

* `optimality_waterfall.png` / `_normalized.png` — one bar per level
  (cluster / pool / worker; the Busy-anchored `iteration` level is omitted here).
* `optimality_kernels.png`   / `_normalized.png` — one bar per kernel (top-N + `other`).

It also emits one `optimality_workers/<worker>_kernel_ladder.png` per worker.
Each ladder has one bar for every R0–R5 rung. Kernel colors persist between bars;
translucent ribbons track the same kernel as its attributable time shrinks.
R0/R1 append aggregate idle/imbalance chunks because those gaps have no
per-kernel attribution.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path

from common.layout import load_payload, plot_output_path, resolve_artifact
from common.style import plt, save_plot
from matplotlib.patches import Patch, Polygon

PAYLOAD = "optimality_waterfall.json"

# Semantic bucket colors: green = irreducible optimal, warmer/greyer = recoverable
# waste. Keys match the Rust `bucket_keys`.
BUCKET_COLOR = {
    "hardware_optimal": "#54A24B",  # green — irreducible compute floor
    "hardware_gap": "#4C78A8",  # blue — profiled↔hardware maturity
    "communication": "#E45756",  # red — network
    "batching": "#F58518",  # orange — small-batch loss
    "imbalance": "#B279A2",  # purple — straggler
    "idle": "#98A2B3",  # grey — scheduler idle
}
BUCKET_LABEL = {
    "idle": "idle (scheduler)",
    "imbalance": "imbalance (straggler)",
    "batching": "batching (small batch)",
    "communication": "communication (network)",
    "hardware_gap": "hardware gap (kernel maturity)",
    "hardware_optimal": "hardware-optimal (irreducible)",
}
# Kernel-level bars carry only the four attributable-to-a-leaf buckets.
KERNEL_BUCKETS = ["batching", "communication", "hardware_gap", "hardware_optimal"]
# Muted categorical colors for kernel identity. The assignment is importance-
# ordered across the run (largest R2 kernels get the strongest early colors), so
# charts avoid the arbitrary neon extremes of a continuous rainbow colormap.
KERNEL_PALETTE = [
    "#355F8A",  # deep blue
    "#D17A22",  # burnt orange
    "#3F7D5B",  # forest
    "#B24C4A",  # brick
    "#6D5A8D",  # muted violet
    "#2F7F7B",  # teal
    "#8A6848",  # umber
    "#9C5F78",  # berry
    "#5F6B76",  # slate
    "#B58B2A",  # ochre
    "#6F8FB3",  # soft blue
    "#E09A5A",  # soft orange
    "#78A083",  # sage
    "#CB7774",  # soft red
    "#9583AE",  # lavender
    "#67A5A1",  # soft teal
    "#A58868",  # sand brown
    "#B98399",  # dusty rose
    "#89949D",  # cool grey
    "#C6A957",  # muted gold
]
LADDER_SPECIAL_COLOR = {
    "imbalance": "#C58AAE",  # dusty pink
    "idle": "#B8C0CC",  # cool light grey
}
RUNG_LABEL = {
    "real": "R0 Real",
    "busy": "R1 Busy",
    "balanced": "R2 Balanced",
    "per_config_best": "R3 Per-config best",
    "ignore_network": "R4 Ignore network",
    "hardware_limit": "R5 Hardware limit",
}


def render(log_dir: Path) -> list[Callable[[], Path]]:
    if not resolve_artifact(log_dir, PAYLOAD).is_file():
        return []
    payload = load_payload(log_dir, PAYLOAD)
    if not payload.get("available", False):
        reason = payload.get("meta", {}).get("reason", "unavailable")
        print(f"[optimality] nothing to render: {reason}")
        return []
    jobs: list[Callable[[], Path]] = []
    if payload.get("levels"):
        # Absolute (GPU·s) + normalized (each bar's Real = 100%, so the bucket
        # *shares* are comparable across levels of very different absolute size).
        jobs.append(
            partial(
                _render_levels,
                payload,
                plot_output_path(log_dir, "optimality_waterfall.png"),
                run_label=log_dir.name,
            )
        )
        jobs.append(
            partial(
                _render_levels,
                payload,
                plot_output_path(log_dir, "optimality_waterfall_normalized.png"),
                run_label=log_dir.name,
                normalize=True,
            )
        )
    if payload.get("kernels"):
        jobs.append(
            partial(
                _render_kernels,
                payload,
                plot_output_path(log_dir, "optimality_kernels.png"),
                run_label=log_dir.name,
            )
        )
        jobs.append(
            partial(
                _render_kernels,
                payload,
                plot_output_path(log_dir, "optimality_kernels_normalized.png"),
                run_label=log_dir.name,
                normalize=True,
            )
        )
    worker_ladders = payload.get("worker_kernel_ladders", [])
    if worker_ladders:
        kernel_colors = _kernel_color_map(worker_ladders)
        for ladder in worker_ladders:
            safe_key = _safe_filename_component(str(ladder.get("key", "worker")))
            jobs.append(
                partial(
                    _render_worker_kernel_ladder,
                    ladder,
                    plot_output_path(
                        log_dir,
                        f"optimality_workers/{safe_key}_kernel_ladder.png",
                    ),
                    run_label=log_dir.name,
                    kernel_colors=kernel_colors,
                )
            )
    return jobs


def _render_levels(
    payload: dict, out_path: Path, *, run_label: str, normalize: bool = False
) -> Path:
    # The iteration level is anchored to Busy (idle = 0 by construction), not Real,
    # so it does not belong on this Real-anchored cluster/pool/worker waterfall.
    levels = [row for row in payload["levels"] if row.get("level") != "iteration"]
    # Rust bucket_keys are top-of-bar order (idle → optimal); draw left→right from
    # the optimal floor so every bar is anchored to optimal.
    top_down = payload.get("bucket_keys", list(BUCKET_LABEL))
    draw_order = list(reversed(top_down))

    n = len(levels)
    fig, ax = plt.subplots(figsize=(11.0, max(3.2, 0.52 * n + 1.8)))
    # Normalized: every bar is scaled to its own Real (100%), so bucket shares are
    # comparable across levels; absolute: bars are in GPU·s on a shared axis.
    axis_max = (
        100.0 if normalize else max((float(row.get("total", 0.0)) for row in levels), default=0.0)
    )
    for i, row in enumerate(levels):
        total = float(row.get("total", 0.0))
        scale = (100.0 / total) if (normalize and total > 0) else 1.0
        left = 0.0
        for key in draw_order:
            value = float(row["buckets"].get(key, 0.0)) * scale
            if value <= 0.0:
                continue
            ax.barh(i, value, left=left, color=BUCKET_COLOR[key], edgecolor="white", linewidth=0.6)
            left += value
        ratio = float(row.get("optimality_ratio", 0.0))
        ax.text(
            left + axis_max * 0.01,
            i,
            f"{ratio * 100:.0f}% opt",
            va="center",
            ha="left",
            fontsize=8,
            color="#475467",
        )

    ax.set_yticks(range(n))
    ax.set_yticklabels([_level_tick(row) for row in levels])
    ax.invert_yaxis()  # cluster (row 0) on top
    ax.set_xlim(0, axis_max * 1.12 if axis_max > 0 else 1.0)
    ax.set_xlabel(
        "share of Real GPU·s (%)" if normalize else "GPU·seconds (waste above the optimal floor)"
    )
    ax.grid(True, axis="x")
    _bucket_legend(ax, top_down)
    if normalize:
        ax.set_title(f"{run_label}\nOptimality waterfall (normalized) — each bar = its Real (100%)")
    else:
        headline = float(payload.get("optimality_ratio", 0.0)) * 100
        ax.set_title(
            f"{run_label}\nOptimality waterfall — {headline:.0f}% of GPU·s is hardware-optimal work"
        )
    save_plot(fig, out_path)
    return out_path


def _render_kernels(
    payload: dict, out_path: Path, *, run_label: str, normalize: bool = False
) -> Path:
    kernels = payload["kernels"]
    draw_order = list(reversed(KERNEL_BUCKETS))  # optimal floor first

    n = len(kernels)
    fig, ax = plt.subplots(figsize=(11.0, max(3.2, 0.42 * n + 1.8)))
    # Normalized: each kernel bar is scaled to its own Real (100%), exposing the
    # batching/hardware-gap *share* even for tiny kernels; absolute: GPU·s.
    axis_max = (
        100.0 if normalize else max((float(k.get("real", 0.0)) for k in kernels), default=0.0)
    )
    for i, kern in enumerate(kernels):
        real = float(kern.get("real", 0.0))
        scale = (100.0 / real) if (normalize and real > 0) else 1.0
        left = 0.0
        for key in draw_order:
            value = float(kern["buckets"].get(key, 0.0)) * scale
            if value <= 0.0:
                continue
            ax.barh(i, value, left=left, color=BUCKET_COLOR[key], edgecolor="white", linewidth=0.6)
            left += value

    ax.set_yticks(range(n))
    ax.set_yticklabels([_kernel_tick(k) for k in kernels], fontsize=8)
    ax.invert_yaxis()  # largest (row 0) on top
    ax.set_xlim(0, axis_max * (1.12 if normalize else 1.05) if axis_max > 0 else 1.0)
    ax.set_xlabel(
        "share of balanced Real (%)"
        if normalize
        else "GPU·seconds (balanced real, split by recoverable source)"
    )
    ax.grid(True, axis="x")
    _bucket_legend(ax, KERNEL_BUCKETS)
    suffix = " (normalized)" if normalize else ""
    ax.set_title(
        f"{run_label}\nPer-kernel optimality{suffix} — batching / communication / hardware headroom"
    )
    save_plot(fig, out_path)
    return out_path


def _render_worker_kernel_ladder(
    ladder: dict,
    out_path: Path,
    *,
    run_label: str,
    kernel_colors: dict[str, object],
) -> Path:
    """Draw one worker's R0→R5 stacked-kernel alluvial ladder."""
    rung_keys = list(RUNG_LABEL)
    kernels = ladder.get("kernels", [])
    special = ladder.get("special_chunks", {})
    bar_height = 0.52

    # Every rung uses the same kernel order. R0/R1 deliberately reuse the R2
    # kernel baseline; their otherwise-unattributable gaps are explicit chunks.
    kernel_source = {
        "real": "balanced",
        "busy": "balanced",
        "balanced": "balanced",
        "per_config_best": "per_config_best",
        "ignore_network": "ignore_network",
        "hardware_limit": "hardware_limit",
    }
    intervals: list[dict[str, tuple[float, float]]] = []
    totals: list[float] = []
    for rung in rung_keys:
        left = 0.0
        row_intervals: dict[str, tuple[float, float]] = {}
        source = kernel_source[rung]
        for kernel in kernels:
            name = str(kernel.get("name", ""))
            value = max(0.0, float(kernel.get("rungs", {}).get(source, 0.0)))
            row_intervals[name] = (left, left + value)
            left += value
        if rung in ("real", "busy"):
            left += max(0.0, float(special.get("imbalance", 0.0)))
        if rung == "real":
            left += max(0.0, float(special.get("idle", 0.0)))
        intervals.append(row_intervals)
        totals.append(left)

    fig_height = max(5.6, 4.8 + 0.13 * len(kernels))
    fig, ax = plt.subplots(figsize=(14.0, fig_height))

    # Ribbons sit behind the bars. A disappearing comm kernel simply tapers to
    # zero at R4; zero-width endpoints are intentionally not drawn.
    for row in range(len(rung_keys) - 1):
        for kernel in kernels:
            name = str(kernel.get("name", ""))
            left0, right0 = intervals[row][name]
            left1, right1 = intervals[row + 1][name]
            if right0 <= left0 or right1 <= left1:
                continue
            color = kernel_colors[name]
            ax.add_patch(
                Polygon(
                    [
                        (left0, row + bar_height / 2),
                        (right0, row + bar_height / 2),
                        (right1, row + 1 - bar_height / 2),
                        (left1, row + 1 - bar_height / 2),
                    ],
                    closed=True,
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.13,
                    zorder=1,
                )
            )
    for row, rung in enumerate(rung_keys):
        source = kernel_source[rung]
        left = 0.0
        for kernel in kernels:
            name = str(kernel.get("name", ""))
            value = max(0.0, float(kernel.get("rungs", {}).get(source, 0.0)))
            if value > 0.0:
                ax.barh(
                    row,
                    value,
                    left=left,
                    height=bar_height,
                    color=kernel_colors[name],
                    edgecolor="white",
                    linewidth=0.55,
                    zorder=3,
                )
            left += value
        if rung in ("real", "busy"):
            imbalance = max(0.0, float(special.get("imbalance", 0.0)))
            if imbalance > 0.0:
                ax.barh(
                    row,
                    imbalance,
                    left=left,
                    height=bar_height,
                    color=LADDER_SPECIAL_COLOR["imbalance"],
                    edgecolor="white",
                    linewidth=0.55,
                    zorder=3,
                )
                left += imbalance
        if rung == "real":
            idle = max(0.0, float(special.get("idle", 0.0)))
            if idle > 0.0:
                ax.barh(
                    row,
                    idle,
                    left=left,
                    height=bar_height,
                    color=LADDER_SPECIAL_COLOR["idle"],
                    edgecolor="white",
                    linewidth=0.55,
                    zorder=3,
                )

    axis_max = max(totals, default=0.0)
    label_pad = axis_max * 0.012
    for row, total in enumerate(totals):
        ax.text(
            total + label_pad,
            row,
            f"{total:,.1f}",
            va="center",
            ha="left",
            fontsize=8,
            color="#475467",
        )
    ax.set_yticks(range(len(rung_keys)))
    ax.set_yticklabels([RUNG_LABEL[key] for key in rung_keys])
    ax.invert_yaxis()
    ax.set_xlim(0.0, axis_max * 1.09 if axis_max > 0 else 1.0)
    ax.set_xlabel("GPU·seconds")
    ax.grid(True, axis="x")
    ax.set_title(
        f"{run_label} · worker {ladder.get('label', ladder.get('key', ''))}\n"
        "Kernel rung ladder — colors and ribbons track the same kernel; "
        "idle and imbalance are aggregate"
    )

    kernel_handles = [
        Patch(
            facecolor=kernel_colors[str(kernel.get("name", ""))],
            edgecolor="white",
            label=_short_kernel_name(str(kernel.get("name", ""))),
        )
        for kernel in kernels
    ]
    special_handles = [
        Patch(
            facecolor=LADDER_SPECIAL_COLOR["imbalance"],
            edgecolor="white",
            label="imbalance (aggregate R1−R2)",
        ),
        Patch(
            facecolor=LADDER_SPECIAL_COLOR["idle"],
            edgecolor="white",
            label="idle (aggregate R0−R1)",
        ),
    ]
    ax.legend(
        handles=kernel_handles + special_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
        fontsize=7,
    )
    save_plot(fig, out_path)
    return out_path


def _kernel_color_map(worker_ladders: list[dict]) -> dict[str, str]:
    """Stable run-wide colors, ordered by each location's largest R2 share."""
    importance: dict[str, float] = {}
    for ladder in worker_ladders:
        for kernel in ladder.get("kernels", []):
            name = str(kernel.get("name", ""))
            balanced = float(kernel.get("rungs", {}).get("balanced", 0.0))
            importance[name] = max(importance.get(name, 0.0), balanced)
    names = sorted(importance, key=lambda name: (-importance[name], name))
    return {name: KERNEL_PALETTE[index % len(KERNEL_PALETTE)] for index, name in enumerate(names)}


def _safe_filename_component(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return safe.strip("._") or "worker"


def _short_kernel_name(name: str) -> str:
    return name if len(name) <= 46 else f"…{name[-45:]}"


def _level_tick(row: dict) -> str:
    label = row.get("label", row.get("key", ""))
    total = float(row.get("total", 0.0))
    return f"{label}\n{total:,.0f} GPU·s"


def _kernel_tick(kern: dict) -> str:
    name = kern.get("name", "")
    short = name if len(name) <= 34 else f"…{name[-33:]}"
    return f"{short}\n{float(kern.get('real', 0.0)):,.0f} GPU·s"


def _bucket_legend(ax, keys_top_down: list[str]) -> None:
    """Legend in top-of-bar order (waste first, optimal last) so it reads like the
    stacked bar from Real down to the floor."""
    handles = [
        Patch(facecolor=BUCKET_COLOR[key], edgecolor="white", label=BUCKET_LABEL[key])
        for key in keys_top_down
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14 - 0.02 * (len(handles) > 4)),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
