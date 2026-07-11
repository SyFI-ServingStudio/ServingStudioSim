"""Render iteration totals and kernel-granular measured/simulated stacks."""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import partial
from pathlib import Path

from common.figure import add_legend
from common.layout import load_payload, plot_output_path, resolve_artifact
from common.style import ACCENT, CURVE, MARKER, plt, save_plot
from matplotlib.patches import ConnectionPatch, Patch

PAYLOAD = "alignment_iteration_series.json"


def render(log_dir: Path) -> list[Callable[[], Path]]:
    if not resolve_artifact(log_dir, PAYLOAD).is_file():
        return []
    payload = load_payload(log_dir, PAYLOAD)
    iterations = payload.get("iterations") or []
    if not iterations:
        reason = payload.get("meta", {}).get("reason", "no paired iterations")
        print(f"[alignment_iteration] nothing to render: {reason}")
        return []
    jobs: list[Callable[[], Path]] = [
        partial(
            _render_overall,
            iterations,
            plot_output_path(log_dir, "alignment_iteration_overview.png"),
            run_label=log_dir.name,
        )
    ]
    for breakdown in payload.get("breakdowns") or []:
        iteration_id = breakdown["iteration_id"]
        jobs.append(
            partial(
                _render_breakdown,
                breakdown,
                plot_output_path(log_dir, f"alignment_iteration_{iteration_id}_breakdown.png"),
                run_label=log_dir.name,
            )
        )
    return jobs


def _render_overall(rows: list[dict], out_path: Path, *, run_label: str) -> Path:
    iteration_ids = [row["iteration_id"] for row in rows]
    measured = [row["measured_ms"] for row in rows]
    simulated = [row["simulated_ms"] for row in rows]
    relative = [row.get("relative_diff_pct", math.nan) for row in rows]
    cumulative = [row.get("cumulative_relative_diff_pct", math.nan) for row in rows]

    fig, axes = plt.subplots(3, 1, figsize=(11.0, 9.0), sharex=True)
    axes[0].plot(iteration_ids, measured, color=CURVE, marker="o", markersize=3, label="Measured")
    axes[0].plot(
        iteration_ids,
        simulated,
        color=ACCENT,
        marker="o",
        markersize=3,
        label="Simulated",
    )
    axes[0].set_ylabel("iteration time (ms)")
    add_legend(axes[0])

    axes[1].plot(iteration_ids, relative, color=MARKER, marker="o", markersize=3)
    axes[1].axhline(0.0, color="#667085", linewidth=1.0, linestyle="--")
    axes[1].set_ylabel("relative diff (%)")

    axes[2].plot(iteration_ids, cumulative, color="#54A24B", marker="o", markersize=3)
    axes[2].axhline(0.0, color="#667085", linewidth=1.0, linestyle="--")
    axes[2].set_ylabel("cumulative diff (%)")
    axes[2].set_xlabel("measured iteration id")

    for ax in axes:
        ax.grid(True)
    fig.suptitle(
        f"{run_label}\nIteration-level measured vs simulated",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    save_plot(fig, out_path)
    return out_path


def _render_breakdown(row: dict, out_path: Path, *, run_label: str) -> Path:
    measured_rows = [
        item
        for item in row.get("measured_kernels") or []
        if float(item.get("duration_ms", 0.0)) > 1e-12
    ]
    measured = _aggregate_measured_for_plot(measured_rows)
    simulated = [
        item
        for item in row.get("simulated_kernels") or []
        if float(item.get("folded_ms", 0.0)) > 1e-12
    ]
    key_rows = len(measured) + len(simulated)
    fig_height = max(6.8, 4.7 + 0.16 * math.ceil(key_rows / 2))
    fig, ax = plt.subplots(figsize=(16.0, fig_height))

    mapped_ids = list(
        dict.fromkeys(item["operation"] for item in measured + simulated if item.get("operation"))
    )
    cmap = plt.get_cmap("tab20")
    mapped_colors = {key: cmap(index % cmap.N) for index, key in enumerate(mapped_ids)}
    measured_colors = [mapped_colors.get(item.get("operation"), "#98A2B3") for item in measured]
    simulated_colors = [mapped_colors.get(item.get("operation"), "#D0D5DD") for item in simulated]
    measured_handles, measured_centers = _draw_stack(
        ax,
        measured,
        y=1.0,
        prefix="M",
        value_key="duration_ms",
        colors=measured_colors,
        label=_measured_key,
    )
    simulated_handles, simulated_centers = _draw_stack(
        ax,
        simulated,
        y=0.0,
        prefix="S",
        value_key="folded_ms",
        colors=simulated_colors,
        label=_simulated_key,
    )
    _label_measured_phases(ax, measured)
    for mapping_id in measured_centers.keys() & simulated_centers.keys():
        ax.add_artist(
            ConnectionPatch(
                xyA=(measured_centers[mapping_id], 0.72),
                xyB=(simulated_centers[mapping_id], 0.28),
                coordsA="data",
                coordsB="data",
                axesA=ax,
                axesB=ax,
                arrowstyle="->",
                color=mapped_colors[mapping_id],
                linewidth=1.0,
                alpha=0.7,
                zorder=5,
            )
        )

    measured_sum = float(row.get("measured_kernel_sum_ms", 0.0))
    simulated_sum = float(row.get("simulated_leaf_workload_ms", 0.0))
    ax.set_yticks([1.0, 0.0])
    ax.set_yticklabels(
        [
            f"Nsight measured\n{measured_sum:.3f} ms launch sum",
            f"Timing-predict\n{simulated_sum:.3f} ms folded leaves",
        ]
    )
    ax.set_ylim(-0.65, 1.65)
    ax.set_xlabel("kernel duration / folded leaf workload (ms)")
    ax.set_title(
        f"{run_label}\nIteration {row['iteration_id']} · "
        f"{row.get('stage', '')} per-kernel stacked breakdown"
    )
    ax.grid(True, axis="x")
    ax.text(
        0.0,
        -0.28,
        "Measured = sum of CUDA launch durations; simulated = leaf time × exact CostTree Scale. "
        "They are workload stacks, not overlap-aware wall-clock totals.",
        transform=ax.transAxes,
        fontsize=9,
        color="#475467",
        ha="left",
        va="top",
    )
    handles = measured_handles + simulated_handles
    if handles:
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.37),
            ncol=2,
            frameon=False,
            fontsize=8,
            handlelength=1.2,
            columnspacing=1.4,
        )
    fig.subplots_adjust(
        left=0.14,
        right=0.98,
        top=0.88,
        bottom=min(0.72, 0.30 + key_rows * 0.008),
    )
    save_plot(fig, out_path)
    return out_path


def _draw_stack(
    ax,
    rows: list[dict],
    *,
    y: float,
    prefix: str,
    value_key: str,
    colors: list,
    label: Callable[[str, dict], str],
) -> tuple[list[Patch], dict[str, float]]:
    """Draw one stack and return centers for exact mapping-group arrows."""
    left = 0.0
    total = sum(float(item[value_key]) for item in rows)
    handles = []
    mapped_centers = {}
    for index, (item, color) in enumerate(zip(rows, colors, strict=True), start=1):
        segment_id = f"{prefix}{index}"
        value = float(item[value_key])
        ax.barh(
            y,
            value,
            left=left,
            height=0.58,
            color=color,
            edgecolor="white",
            linewidth=0.7,
        )
        if total > 0.0 and value / total >= 0.025:
            text = segment_id if value / total < 0.09 else f"{segment_id}\n{value:.3f}"
            ax.text(
                left + value / 2,
                y,
                text,
                ha="center",
                va="center",
                fontsize=7,
                color="#101828",
                clip_on=True,
            )
        handles.append(Patch(facecolor=color, edgecolor="white", label=label(segment_id, item)))
        if item.get("operation"):
            mapped_centers[item["operation"]] = left + value / 2
        left += value
    return handles, mapped_centers


def _aggregate_measured_for_plot(rows: list[dict]) -> list[dict]:
    """Keep row-level payloads, but render one segment per mapping group.

    Unmapped rows are grouped by implementation name. This preserves all timing
    while keeping the figure human-sized and making many-real-to-one-sim arrows
    explicit.
    """
    aggregates: dict[tuple[str, str, str], dict] = {}
    order: list[tuple[str, str, str]] = []
    for row in rows:
        phase = row["phase"]
        operation = row.get("operation")
        key = (phase, "mapped", operation) if operation else (phase, "unmapped", row["name"])
        if key not in aggregates:
            order.append(key)
            aggregates[key] = {
                "phase": phase,
                "name": operation or row["name"],
                "operation": operation,
                "calls": 0,
                "duration_ms": 0.0,
                "row_count": 0,
            }
        aggregate = aggregates[key]
        aggregate["calls"] += int(row.get("calls", 1))
        aggregate["duration_ms"] += float(row["duration_ms"])
        aggregate["row_count"] += 1
    return [aggregates[key] for key in order]


def _label_measured_phases(ax, measured: list[dict]) -> None:
    """Annotate contiguous phase spans without splitting the measured stack."""
    left = 0.0
    spans: list[tuple[str, float, float]] = []
    for item in measured:
        value = float(item["duration_ms"])
        phase = item["phase"]
        if spans and spans[-1][0] == phase:
            old_phase, start, _ = spans[-1]
            spans[-1] = (old_phase, start, left + value)
        else:
            spans.append((phase, left, left + value))
        left += value
    for index, (phase, start, end) in enumerate(spans):
        if index:
            ax.vlines(start, 0.67, 1.33, color="#344054", linewidth=1.0, linestyle=":")
        ax.text(
            (start + end) / 2,
            1.36,
            phase,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#344054",
            clip_on=False,
        )


def _stack_colors(count: int, palette: str) -> list:
    cmap = plt.get_cmap(palette)
    return [cmap(index % cmap.N) for index in range(count)]


def _measured_key(segment_id: str, item: dict) -> str:
    operation = item.get("operation") or "unmapped"
    return (
        f"{segment_id} [{item['phase']}] {_short_kernel_name(item['name'])} · "
        f"{item['calls']} launch(es) · "
        f"{float(item['duration_ms']):.3f} ms · {operation}"
    )


def _simulated_key(segment_id: str, item: dict) -> str:
    operation = item.get("operation") or "unmapped"
    return (
        f"{segment_id} {_short_kernel_name(item['name'])} [{item['kind']}] · "
        f"×{item['multiplicity']} · {float(item['folded_ms']):.3f} ms · {operation}"
    )


def _short_kernel_name(name: str, limit: int = 70) -> str:
    compact = " ".join(name.split())
    return compact if len(compact) <= limit else f"{compact[: limit - 1]}…"
