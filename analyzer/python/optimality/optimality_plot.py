"""Render the optimality sub-optimality waterfall.

The Rust `optimality` subject emits, per aggregation **level** (cluster / pool /
worker / iteration), six telescoping buckets that sum to that level's Real GPU·s:
`idle | imbalance | batching | communication | hardware_gap | hardware_optimal`.
It also emits a per-kernel array (the balanced Real of each location split into
batching / communication / hardware_gap / hardware_optimal).

Four figures, all the SAME horizontal stacked-bar shape aligned to the optimal
floor (the green `hardware_optimal` segment sits at the left, waste grows
rightward to Real). Each of the two views ships in an absolute (GPU·s) and a
normalized (every bar scaled to its own Real = 100%) variant:

* `optimality_waterfall.png` / `_normalized.png` — one bar per level
  (cluster / pool / worker; the Busy-anchored `iteration` level is omitted here).
* `optimality_kernels.png`   / `_normalized.png` — one bar per kernel (top-N + `other`).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path

from common.layout import load_payload, plot_output_path, resolve_artifact
from common.style import plt, save_plot
from matplotlib.patches import Patch

PAYLOAD = "optimality_waterfall.json"

# Semantic bucket colors: green = irreducible optimal, warmer/greyer = recoverable
# waste. Keys match the Rust `bucket_keys`.
BUCKET_COLOR = {
    "hardware_optimal": "#54A24B",  # green — irreducible compute floor
    "hardware_gap": "#4C78A8",      # blue — profiled↔hardware maturity
    "communication": "#E45756",     # red — network
    "batching": "#F58518",          # orange — small-batch loss
    "imbalance": "#B279A2",         # purple — straggler
    "idle": "#98A2B3",              # grey — scheduler idle
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
    axis_max = 100.0 if normalize else max(
        (float(row.get("total", 0.0)) for row in levels), default=0.0
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
    axis_max = 100.0 if normalize else max(
        (float(k.get("real", 0.0)) for k in kernels), default=0.0
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
        "share of balanced Real (%)" if normalize else "GPU·seconds (balanced real, split by recoverable source)"
    )
    ax.grid(True, axis="x")
    _bucket_legend(ax, KERNEL_BUCKETS)
    suffix = " (normalized)" if normalize else ""
    ax.set_title(
        f"{run_label}\nPer-kernel optimality{suffix} — batching / communication / hardware headroom"
    )
    save_plot(fig, out_path)
    return out_path


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
