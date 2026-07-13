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
MAX_BREAKDOWN_PLOTS = 128
BREAKDOWN_PLOTS_PER_DIR = 32
# An 18-inch diagnostic canvas is 2700 px wide at 150 DPI. That fixed width
# contains the two-column legend without an expensive tight-bbox redraw. The
# subject-level overview keeps the shared 300-DPI PNG default.
BREAKDOWN_DPI = 150
BREAKDOWN_PNG_OPTIONS = {"compress_level": 1}


def render(log_dir: Path) -> list[Callable[[], Path]]:
    payload_path = resolve_artifact(log_dir, PAYLOAD)
    if not payload_path.is_file():
        return []
    payload = load_payload(log_dir, PAYLOAD)
    iterations = payload.get("iterations") or []
    if not iterations:
        reason = payload.get("meta", {}).get("reason", "no paired iterations")
        print(f"[alignment_iteration] nothing to render: {reason}")
        return []
    jobs: list[Callable[[], Path]] = [
        partial(
            _render_kernel_overall,
            iterations,
            plot_output_path(log_dir, "alignment_iteration_overview.png"),
            run_label=log_dir.name,
        ),
        partial(
            _render_gpu_cycle_overall,
            iterations,
            plot_output_path(log_dir, "alignment_iteration_gpu_cycle_overview.png"),
            run_label=log_dir.name,
            gpu_time_multiplier=float(payload["meta"]["gpu_time_multiplier"]),
        ),
    ]
    sampled_breakdowns = _evenly_sample_breakdowns(payload.get("breakdowns") or [])
    breakdown_specs: list[tuple[dict, Path]] = []
    for offset in range(0, len(sampled_breakdowns), BREAKDOWN_PLOTS_PER_DIR):
        breakdown_group = sampled_breakdowns[offset : offset + BREAKDOWN_PLOTS_PER_DIR]
        first_iteration_id = breakdown_group[0]["iteration_id"]
        last_iteration_id = breakdown_group[-1]["iteration_id"]
        group_dir = f"iter_{first_iteration_id}_to_{last_iteration_id}"
        for breakdown in breakdown_group:
            iteration_id = breakdown["iteration_id"]
            out_path = plot_output_path(
                log_dir,
                f"{group_dir}/iter_{iteration_id}_breakdown.png",
            )
            breakdown_specs.append((breakdown, out_path))

    desired_outputs = {out_path for _, out_path in breakdown_specs}
    _remove_stale_breakdown_outputs(log_dir, desired_outputs)
    for breakdown, out_path in breakdown_specs:
        # Rendering sampled diagnostics is intentionally resumable. A newer
        # payload or renderer invalidates the image; otherwise an interrupted
        # rerun keeps the completed work.
        if _output_is_current(out_path, payload_path, Path(__file__)):
            continue
        jobs.append(
            partial(
                _render_breakdown,
                breakdown,
                out_path,
                run_label=log_dir.name,
            )
        )
    return jobs


def _evenly_sample_breakdowns(
    breakdowns: list[dict],
    limit: int = MAX_BREAKDOWN_PLOTS,
) -> list[dict]:
    """Keep at most ``limit`` rows, evenly spanning raw iteration order.

    Sampling affects image generation only: the Rust payload and reports retain
    every iteration.  Sorting by the raw id makes filenames, directory ranges,
    and selected endpoints deterministic even if a producer changes JSON row
    order later.
    """
    assert limit > 0
    ordered = sorted(breakdowns, key=lambda row: int(row["iteration_id"]))
    if len(ordered) <= limit:
        return ordered
    if limit == 1:
        return [ordered[0]]
    last_index = len(ordered) - 1
    return [
        ordered[round(sample_index * last_index / (limit - 1))] for sample_index in range(limit)
    ]


def _output_is_current(output_path: Path, *source_paths: Path) -> bool:
    """Whether an existing image is at least as new as every render input."""
    if not output_path.is_file():
        return False
    output_mtime_ns = output_path.stat().st_mtime_ns
    return all(output_mtime_ns >= source_path.stat().st_mtime_ns for source_path in source_paths)


def _remove_stale_breakdown_outputs(log_dir: Path, desired_pngs: set[Path]) -> None:
    """Remove generated breakdowns outside the current sample/layout policy.

    Keep an old JPG for a still-selected row until its replacement PNG is
    successfully written. This makes the format migration failure-safe while
    ensuring an old 256-row sample does not remain mixed with the new 128 rows.
    """
    plots_dir = log_dir / "plots"
    if not plots_dir.is_dir():
        return
    fallback_jpgs = {path.with_suffix(".jpg") for path in desired_pngs}
    keep = desired_pngs | fallback_jpgs
    desired_dirs = {path.parent for path in desired_pngs}
    for pattern in ("iter_*_to_*/iter_*_breakdown.png", "iter_*_to_*/iter_*_breakdown.jpg"):
        for path in plots_dir.glob(pattern):
            if path not in keep:
                path.unlink()
    for group_dir in plots_dir.glob("iter_*_to_*"):
        if group_dir not in desired_dirs and group_dir.is_dir() and not any(group_dir.iterdir()):
            group_dir.rmdir()


def _render_kernel_overall(rows: list[dict], out_path: Path, *, run_label: str) -> Path:
    return _render_overview(
        rows,
        out_path,
        run_label=run_label,
        measured_key="measured_ms",
        simulated_key="simulated_ms",
        relative_key="relative_diff_pct",
        cumulative_key="cumulative_relative_diff_pct",
        measured_label="Measured kernel busy union",
        simulated_label="Timing-predict",
        y_label="kernel time (ms)",
        title="Kernel busy union vs timing-predict",
    )


def _render_gpu_cycle_overall(
    rows: list[dict],
    out_path: Path,
    *,
    run_label: str,
    gpu_time_multiplier: float,
) -> Path:
    # The last measured iteration has no next first-kernel boundary and is
    # therefore not a GPU cycle. Rust leaves its cycle fields null explicitly.
    return _render_overview(
        rows,
        out_path,
        run_label=run_label,
        measured_key="measured_gpu_cycle_ms",
        simulated_key="simulated_gpu_cycle_ms",
        relative_key="gpu_cycle_relative_diff_pct",
        cumulative_key="gpu_cycle_cumulative_relative_diff_pct",
        measured_label="Measured GPU cycle",
        simulated_label=f"Timing-predict × {gpu_time_multiplier:.4g}",
        y_label="GPU iteration cycle (ms)",
        title="Measured GPU cycle vs scaled timing-predict",
    )


def _render_overview(
    rows: list[dict],
    out_path: Path,
    *,
    run_label: str,
    measured_key: str,
    simulated_key: str,
    relative_key: str,
    cumulative_key: str,
    measured_label: str,
    simulated_label: str,
    y_label: str,
    title: str,
) -> Path:
    paired_rows = [
        row
        for row in rows
        if row.get(measured_key) is not None and row.get(simulated_key) is not None
    ]
    iteration_ids = [row["iteration_id"] for row in paired_rows]
    measured = [row[measured_key] for row in paired_rows]
    simulated = [row[simulated_key] for row in paired_rows]
    relative = [row.get(relative_key, math.nan) for row in paired_rows]
    cumulative = [row.get(cumulative_key, math.nan) for row in paired_rows]

    fig, axes = plt.subplots(3, 1, figsize=(11.0, 9.0), sharex=True)
    axes[0].plot(
        iteration_ids,
        measured,
        color=CURVE,
        marker="o",
        markersize=3,
        label=measured_label,
    )
    axes[0].plot(
        iteration_ids,
        simulated,
        color=ACCENT,
        marker="o",
        markersize=3,
        label=simulated_label,
    )
    axes[0].set_ylabel(y_label)
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
        f"{run_label}\n{title}",
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
    fig, ax = plt.subplots(figsize=(18.0, fig_height))

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
    # Margins and the 18-inch width above contain the title, annotation, and
    # two-column legend. Avoid tight-bbox's second layout/draw; low-compression
    # PNG preserves small text and fine lines without spending CPU on file size.
    save_plot(
        fig,
        out_path,
        dpi=BREAKDOWN_DPI,
        tight=False,
        pil_kwargs=BREAKDOWN_PNG_OPTIONS,
    )
    legacy_jpg = out_path.with_suffix(".jpg")
    if legacy_jpg.is_file():
        legacy_jpg.unlink()
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
