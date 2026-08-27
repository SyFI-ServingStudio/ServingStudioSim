"""Render iteration totals and kernel-granular measured/simulated stacks."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
from functools import partial
from pathlib import Path

from common.figure import add_legend, fmt_ms
from common.layout import (
    load_payload,
    plot_output_path,
    read_sharded_records,
    resolve_artifact,
)
from common.style import ACCENT, CURVE, MARKER, plt, save_plot
from matplotlib.patches import ConnectionPatch, Patch

PAYLOAD = "alignment_iteration_series.json"
TIMELINE_PAYLOAD = "alignment_timeline.json"
MAX_BREAKDOWN_PLOTS = 128
BREAKDOWN_PLOTS_PER_DIR = 32
MAX_STREAM_ROWS = 8
MIN_STREAM_SHARE = 0.01
# An 18-inch diagnostic canvas is 2700 px wide at 150 DPI. That fixed width
# contains the two-column legend without an expensive tight-bbox redraw. The
# subject-level overview keeps the shared 300-DPI PNG default.
BREAKDOWN_DPI = 150
BREAKDOWN_PNG_OPTIONS = {"compress_level": 1}
BREAKDOWN_WIDTH_INCHES = 18.0
BREAKDOWN_LEGEND_COLUMNS = 4


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
        )
    ]
    gpu_time_multiplier = _recommended_gpu_time_multiplier(payload)
    if gpu_time_multiplier is not None:
        jobs.append(
            partial(
                _render_gpu_cycle_overall,
                iterations,
                plot_output_path(log_dir, "alignment_iteration_gpu_cycle_overview.png"),
                run_label=log_dir.name,
                gpu_time_multiplier=gpu_time_multiplier,
            )
        )
    # Sample first, read second. The breakdowns live in a byte-range-addressed
    # shard precisely so that rendering 128 of 2,040 iterations does not have to
    # parse the other 1,912.
    sampled_ids = _evenly_sample_iteration_ids(iterations)
    sampled_breakdowns = read_sharded_records(log_dir, payload["breakdown_detail"], sampled_ids)
    timeline_path = resolve_artifact(log_dir, TIMELINE_PAYLOAD)
    timeline_by_iteration: dict[int, dict] = {}
    render_sources = [payload_path, Path(__file__)]
    if timeline_path.is_file():
        timeline_payload = load_payload(log_dir, TIMELINE_PAYLOAD)
        timeline_rows = read_sharded_records(
            log_dir, timeline_payload["iteration_detail"], sampled_ids
        )
        timeline_by_iteration = {int(row["iteration_id"]): row for row in timeline_rows}
        render_sources.extend(
            [
                timeline_path,
                resolve_artifact(log_dir, timeline_payload["iteration_detail"]["file"]),
            ]
        )

    breakdown_specs: list[tuple[dict, dict | None, Path]] = []
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
            breakdown_specs.append(
                (breakdown, timeline_by_iteration.get(int(iteration_id)), out_path)
            )

    desired_outputs = {out_path for _, _, out_path in breakdown_specs}
    _remove_stale_breakdown_outputs(log_dir, desired_outputs)
    for breakdown, timeline_iteration, out_path in breakdown_specs:
        # Rendering sampled diagnostics is intentionally resumable. A newer
        # payload or renderer invalidates the image; otherwise an interrupted
        # rerun keeps the completed work.
        if _output_is_current(out_path, *render_sources):
            continue
        renderer = _render_stream_breakdown if timeline_iteration is not None else _render_breakdown
        if timeline_iteration is None:
            jobs.append(partial(renderer, breakdown, out_path, run_label=log_dir.name))
        else:
            jobs.append(
                partial(
                    renderer,
                    breakdown,
                    timeline_iteration,
                    out_path,
                    run_label=log_dir.name,
                )
            )
    return jobs


def _recommended_gpu_time_multiplier(payload: dict) -> float | None:
    value = (payload.get("meta") or {}).get("recommended_gpu_time_multiplier")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 1.0 else None


def _evenly_sample_iteration_ids(
    iterations: list[dict],
    limit: int = MAX_BREAKDOWN_PLOTS,
) -> list[int]:
    """Keep at most ``limit`` iteration ids, evenly spanning raw iteration order.

    Sampling affects image generation only: the Rust payload and reports retain
    every iteration.  Sorting by the raw id makes filenames, directory ranges,
    and selected endpoints deterministic even if a producer changes JSON row
    order later.
    """
    assert limit > 0
    ordered = sorted(int(row["iteration_id"]) for row in iterations)
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


def _stream_operation_rows(timeline_iteration: dict, device_id: int) -> dict[int, list[dict]]:
    """Assign each reduced occurrence to the selected device's real streams."""
    totals: dict[tuple[int, str, str | None], float] = defaultdict(float)
    order: list[tuple[int, str, str | None]] = []
    for kernel in timeline_iteration["measured"]["kernels"]:
        intervals = [item for item in kernel["iv"] if int(item[0]) == device_id]
        if not intervals:
            continue
        weights = [max(0, int(item[2]) - int(item[1])) for item in intervals]
        weight_sum = sum(weights)
        if weight_sum == 0:
            weights = [1] * len(intervals)
            weight_sum = len(intervals)
        occurrence_ms = float(kernel["occ_ns"]) / 1.0e6
        for interval, weight in zip(intervals, weights, strict=True):
            key = (int(interval[4]), kernel["ph"], kernel.get("op"))
            if key not in totals:
                order.append(key)
            totals[key] += occurrence_ms * weight / weight_sum

    rows_by_stream: dict[int, list[dict]] = defaultdict(list)
    for track_index, phase, operation in order:
        rows_by_stream[track_index].append(
            {
                "phase": phase,
                "operation": operation,
                "duration_ms": totals[(track_index, phase, operation)],
            }
        )
    return dict(rows_by_stream)


def _display_stream_rows(rows_by_stream: dict[int, list[dict]]) -> list[tuple[str, list[dict]]]:
    """Bound plot height while preserving every stream's additive work."""
    stream_ms = {
        track_index: sum(float(row["duration_ms"]) for row in rows)
        for track_index, rows in rows_by_stream.items()
    }
    total_ms = sum(stream_ms.values())
    ranked = sorted(stream_ms, key=lambda track_index: (-stream_ms[track_index], track_index))
    kept = {
        track_index
        for track_index in ranked[:MAX_STREAM_ROWS]
        if total_ms == 0.0 or stream_ms[track_index] / total_ms >= MIN_STREAM_SHARE
    }
    if ranked and not kept:
        kept.add(ranked[0])

    displayed = [
        (f"stream {track_index}", rows_by_stream[track_index]) for track_index in sorted(kept)
    ]
    omitted = [track_index for track_index in ranked if track_index not in kept]
    if omitted:
        aggregate: dict[tuple[str, str | None], float] = defaultdict(float)
        for track_index in omitted:
            for row in rows_by_stream[track_index]:
                aggregate[(row["phase"], row.get("operation"))] += float(row["duration_ms"])
        displayed.append(
            (
                f"other {len(omitted)} streams (aggregated)",
                [
                    {"phase": phase, "operation": operation, "duration_ms": duration_ms}
                    for (phase, operation), duration_ms in aggregate.items()
                ],
            )
        )
    return displayed


def _compact_path_rows(rows: list[dict], target_ms: float) -> list[dict]:
    """Aggregate semantic work and scale only to remove measured stream overlap."""
    totals: dict[tuple[str | None, str | None], float] = defaultdict(float)
    order: list[tuple[str | None, str | None]] = []
    for row in rows:
        key = (row.get("phase"), row.get("operation"))
        if key not in totals:
            order.append(key)
        totals[key] += float(row["duration_ms"])
    additive_ms = sum(totals.values())
    scale = target_ms / additive_ms if additive_ms > 0.0 else 0.0
    return [
        {
            "phase": phase,
            "operation": operation,
            "duration_ms": totals[(phase, operation)] * scale,
        }
        for phase, operation in order
    ]


def _operation_comparison_labels(
    semantic_names: list[str], measured_path: list[dict], simulated_path: list[dict]
) -> dict[str, str]:
    """Format one measured-vs-simulated summary for each timeline color."""
    measured_by_operation: dict[str, float] = defaultdict(float)
    simulated_by_operation: dict[str, float] = defaultdict(float)
    for row in measured_path:
        measured_by_operation[row.get("operation") or "unmapped"] += float(row["duration_ms"])
    for row in simulated_path:
        simulated_by_operation[row.get("operation") or "unmapped"] += float(row["duration_ms"])

    labels = {}
    for name in semantic_names:
        measured_ms = measured_by_operation[name]
        simulated_ms = simulated_by_operation[name]
        relative_pct = (
            (simulated_ms - measured_ms) / measured_ms * 100.0
            if measured_ms > 1.0e-12
            else math.nan
        )
        relative_label = f"{relative_pct:+.1f}%" if math.isfinite(relative_pct) else "n/a"
        labels[name] = f"{name} | {fmt_ms(measured_ms)} | {fmt_ms(simulated_ms)} | {relative_label}"
    return labels


def _render_stream_breakdown(
    breakdown: dict,
    timeline_iteration: dict,
    out_path: Path,
    *,
    run_label: str,
) -> Path:
    """Draw compact real streams plus one measured and one simulated path."""
    device_id = breakdown.get("critical_device_id")
    if device_id is None:
        device_id = timeline_iteration["measured"].get("critical_device_id")
    if device_id is None:
        work_by_device: dict[int, int] = defaultdict(int)
        for kernel in timeline_iteration["measured"]["kernels"]:
            for interval in kernel["iv"]:
                work_by_device[int(interval[0])] += max(0, int(interval[2]) - int(interval[1]))
        if not work_by_device:
            raise ValueError("multi-stream breakdown has no measured device work")
        device_id = min(work_by_device, key=lambda item: (-work_by_device[item], item))
    device_id = int(device_id)

    stream_rows = _stream_operation_rows(timeline_iteration, device_id)
    displayed_streams = _display_stream_rows(stream_rows)
    additive_rows = [row for rows in stream_rows.values() for row in rows]
    measured_ms = float(breakdown["measured_kernel_sum_ms"]) - float(
        breakdown.get("measured_concurrent_hidden_ms", 0.0) or 0.0
    )
    measured_path = _compact_path_rows(additive_rows, measured_ms)

    simulated_path = [
        {
            "phase": None,
            "operation": row.get("operation") or row["name"],
            "duration_ms": float(row["critical_path_ms"]),
        }
        for row in breakdown["simulated_kernels"]
        if float(row["critical_path_ms"]) > 1.0e-12
    ]
    semantic_names = list(
        dict.fromkeys(
            (row.get("operation") or "unmapped")
            for _label, rows in displayed_streams
            for row in rows
        )
    )
    semantic_names.extend(
        name for name in (row["operation"] for row in simulated_path) if name not in semantic_names
    )
    palette = plt.get_cmap("tab20")
    colors = {name: palette(index % palette.N) for index, name in enumerate(semantic_names)}
    comparison_labels = _operation_comparison_labels(semantic_names, measured_path, simulated_path)

    labels = [f"GPU {device_id} {label}" for label, _rows in displayed_streams]
    labels.extend(["Nsight reduced critical path", "Timing-predict critical path"])
    rows_to_draw = [rows for _label, rows in displayed_streams] + [measured_path, simulated_path]
    legend_rows = math.ceil(len(semantic_names) / BREAKDOWN_LEGEND_COLUMNS)
    legend_height_inches = 0.18 * legend_rows + 0.22
    plot_height_inches = max(3.0, 0.62 * len(labels))
    figure_height_inches = plot_height_inches + legend_height_inches + 1.30
    fig, axis = plt.subplots(figsize=(BREAKDOWN_WIDTH_INCHES, figure_height_inches))
    for y_position, rows in enumerate(rows_to_draw):
        left_ms = 0.0
        for row in rows:
            duration_ms = float(row["duration_ms"])
            semantic_name = row.get("operation") or "unmapped"
            axis.barh(
                y_position,
                duration_ms,
                left=left_ms,
                height=0.72,
                color=colors[semantic_name],
                edgecolor="white",
                linewidth=0.35,
            )
            left_ms += duration_ms

    axis.set_yticks(range(len(labels)), labels)
    axis.invert_yaxis()
    axis.set_xlabel("aggregated kernel duration (ms; each row compacted independently)")
    axis.grid(True, axis="x", alpha=0.3)
    simulated_ms = sum(float(row["duration_ms"]) for row in simulated_path)
    delta_ms = simulated_ms - measured_ms
    relative_pct = delta_ms / measured_ms * 100.0 if measured_ms > 0.0 else math.nan
    axis.set_title(
        f"{run_label}\nIteration {breakdown['iteration_id']} · {breakdown['stage']} · "
        f"critical device {device_id}\n"
        f"Nsight {measured_ms:.3f} ms · Timing-predict {simulated_ms:.3f} ms · "
        f"delta {delta_ms:+.3f} ms ({relative_pct:+.2f}%)",
        fontweight="bold",
    )
    if semantic_names:
        fig.legend(
            handles=[
                Patch(facecolor=colors[name], label=comparison_labels[name])
                for name in semantic_names
            ],
            loc="lower left",
            bbox_to_anchor=(0.02, 0.015, 0.96, 0.01),
            mode="expand",
            frameon=False,
            ncol=BREAKDOWN_LEGEND_COLUMNS,
            fontsize=8,
            handlelength=1.2,
            columnspacing=1.4,
            title="operation | measured | simulated | Δ",
            title_fontsize=8,
        )
    # The legend owns a figure-level band below the axes. Size that band in
    # physical inches so long operation inventories cannot cover the stream
    # rows, while retaining the pre-sized single-draw rendering contract.
    fig.subplots_adjust(
        left=0.17,
        right=0.98,
        top=1.0 - 0.85 / figure_height_inches,
        bottom=(legend_height_inches + 0.60) / figure_height_inches,
    )
    save_plot(
        fig,
        out_path,
        dpi=BREAKDOWN_DPI,
        tight=False,
        pil_kwargs=BREAKDOWN_PNG_OPTIONS,
    )
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
        if float(item.get("critical_path_ms", 0.0)) > 1e-12
    ]
    key_rows = len(measured) + len(simulated)
    fig_height = max(10.2, 7.3 + 0.16 * math.ceil(key_rows / 2))
    fig, (ax, cumulative_ax) = plt.subplots(
        2,
        1,
        figsize=(BREAKDOWN_WIDTH_INCHES, fig_height),
        gridspec_kw={"height_ratios": [3.5, 1.6]},
    )

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
        hidden_key="concurrent_hidden_ms",
    )
    simulated_handles, simulated_centers = _draw_stack(
        ax,
        simulated,
        y=0.0,
        prefix="S",
        value_key="critical_path_ms",
        colors=simulated_colors,
        label=_simulated_key,
    )
    _label_measured_phases(ax, measured)
    for mapping_id, measured_center, simulated_center in _mapping_center_pairs(
        measured_centers, simulated_centers
    ):
        ax.add_artist(
            ConnectionPatch(
                xyA=(measured_center, 0.72),
                xyB=(simulated_center, 0.28),
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
    simulated_sum = float(row.get("simulated_critical_path_ms", 0.0))
    # The measured lane lays every occurrence end to end, so when the framework
    # ran two CUDA streams at once the lane is longer than the GPU was busy by
    # exactly the overlap. Draw that surplus instead of letting the bar quietly
    # grow: the stack stays the like-for-like partner of a CostTree `Sum`, and
    # the shaded tail says how much of it never cost wall time. Zero, and so
    # invisible, on a single-stream capture.
    hidden_ms = float(row.get("measured_concurrent_hidden_ms", 0.0) or 0.0)
    _draw_cumulative_critical_path_error(
        cumulative_ax,
        measured,
        simulated,
        simulated_colors,
        measured_sum_ms=measured_sum,
        simulated_sum_ms=simulated_sum,
    )
    shared_path_limit_ms = max(measured_sum, simulated_sum) * 1.03
    if shared_path_limit_ms <= 1e-12:
        shared_path_limit_ms = 1.0
    ax.set_xlim(0.0, shared_path_limit_ms)
    cumulative_ax.set_xlim(0.0, shared_path_limit_ms)
    measured_label = f"Nsight measured\n{measured_sum:.3f} ms replica path"
    if hidden_ms > 1e-9:
        measured_label += (
            f"\n(−{hidden_ms:.3f} ms hatched: ran\n"
            f"concurrently → {measured_sum - hidden_ms:.3f} ms busy)"
        )
    ax.set_yticks([1.0, 0.0])
    ax.set_yticklabels(
        [
            measured_label,
            f"Timing-predict\n{simulated_sum:.3f} ms CostTree path",
        ]
    )
    ax.set_ylim(-0.65, 1.65)
    ax.set_xlabel("replica critical-path contribution (ms)")
    ax.set_title(
        f"{run_label}\nIteration {row['iteration_id']} · "
        f"{row.get('stage', '')} per-kernel replica critical-path breakdown"
    )
    ax.grid(True, axis="x")
    ax.text(
        0.0,
        -0.28,
        "Measured = per-occurrence cross-rank reduction (independent: slowest-rank duration; "
        "synchronizing collective: last-arrival→done), summed. Simulated = exact CostTree "
        "Sum/Scale/Max attribution. "
        "Collective arrival-wait is excluded here and lives in the GPU-cycle correction.",
        transform=ax.transAxes,
        fontsize=9,
        color="#475467",
        ha="left",
        va="top",
    )
    handles = measured_handles + simulated_handles
    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
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
        bottom=min(0.58, 0.24 + key_rows * 0.007),
        hspace=0.55,
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
    hidden_key: str | None = None,
) -> tuple[list[Patch], dict[str, list[float]]]:
    """Draw one stack and retain every center owned by each operation.

    One measured operation may own multiple additive simulated slots.  Keeping
    all centers prevents a later slot from silently replacing an earlier one in
    the diagnostic arrows.
    """
    left = 0.0
    total = sum(float(item[value_key]) for item in rows)
    handles = []
    mapped_centers: dict[str, list[float]] = {}
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
        # Hatch the part of THIS segment that ran while an earlier CUDA stream
        # was already busy. The lane is a sum laid end to end, not a time axis,
        # so the mark cannot say when the overlap happened — but putting it on
        # the operation that actually overlapped is the difference between "the
        # shared expert was hidden" and a meaningless lump at the end of the
        # bar. The analyzer charges each overlap to one side only, so the
        # hatched marks across the lane add up to the `-N ms` in its label.
        hidden = float(item.get(hidden_key, 0.0) or 0.0) if hidden_key else 0.0
        hidden = min(hidden, value)
        if hidden > 1e-9:
            ax.barh(
                y,
                hidden,
                left=left + value - hidden,
                height=0.58,
                facecolor="none",
                edgecolor="0.25",
                hatch="////",
                linewidth=0.0,
                zorder=4,
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
            mapped_centers.setdefault(item["operation"], []).append(left + value / 2)
        left += value
    return handles, mapped_centers


def _mapping_center_pairs(
    measured_centers: dict[str, list[float]],
    simulated_centers: dict[str, list[float]],
) -> list[tuple[str, float, float]]:
    """Return every operation-owned measured-to-simulated segment pair."""
    return [
        (operation, measured_center, simulated_center)
        for operation in measured_centers.keys() & simulated_centers.keys()
        for measured_center in measured_centers[operation]
        for simulated_center in simulated_centers[operation]
    ]


def _simulated_width_cumulative_error_steps(
    measured: list[dict], simulated: list[dict]
) -> tuple[float, list[float], list[float]]:
    """Accumulate error over variable-width simulated-slot intervals.

    When one measured operation owns several simulated slots, apportion its
    measured duration by simulated slot width.  This is a plotting convention:
    it preserves the operation total and exact iteration endpoint without
    inventing a kernel-level split.
    """
    measured_by_operation: dict[str, float] = defaultdict(float)
    unmatched_measured_ms = 0.0
    for item in measured:
        operation = item.get("operation")
        if operation:
            measured_by_operation[operation] += float(item["duration_ms"])
        else:
            unmatched_measured_ms += float(item["duration_ms"])

    simulated_by_operation: dict[str, float] = defaultdict(float)
    for item in simulated:
        operation = item.get("operation")
        if operation:
            simulated_by_operation[operation] += float(item["critical_path_ms"])

    unmatched_measured_ms += sum(
        duration_ms
        for operation, duration_ms in measured_by_operation.items()
        if simulated_by_operation.get(operation, 0.0) <= 1e-12
    )
    edges_ms = [0.0]
    cumulative_errors_ms: list[float] = []
    cumulative_error_ms = -unmatched_measured_ms
    for item in simulated:
        simulated_ms = float(item["critical_path_ms"])
        operation = item.get("operation")
        measured_share_ms = 0.0
        if operation:
            operation_simulated_ms = simulated_by_operation[operation]
            measured_share_ms = (
                measured_by_operation.get(operation, 0.0) * simulated_ms / operation_simulated_ms
            )
        cumulative_error_ms += simulated_ms - measured_share_ms
        edges_ms.append(edges_ms[-1] + simulated_ms)
        cumulative_errors_ms.append(cumulative_error_ms)
    return -unmatched_measured_ms, edges_ms, cumulative_errors_ms


def _draw_cumulative_critical_path_error(
    ax,
    measured: list[dict],
    simulated: list[dict],
    simulated_colors: list,
    *,
    measured_sum_ms: float,
    simulated_sum_ms: float,
) -> None:
    """Draw a compact cumulative-error row aligned to simulated slot widths."""
    baseline_ms, edges_ms, cumulative_errors_ms = _simulated_width_cumulative_error_steps(
        measured, simulated
    )
    if not simulated:
        return

    for index, (item, color) in enumerate(zip(simulated, simulated_colors, strict=True), start=1):
        left_ms = edges_ms[index - 1]
        right_ms = edges_ms[index]
        ax.axvspan(left_ms, right_ms, color=color, alpha=0.18, linewidth=0.0)
        if simulated_sum_ms > 0.0 and (right_ms - left_ms) / simulated_sum_ms >= 0.025:
            ax.text(
                (left_ms + right_ms) / 2.0,
                0.94,
                f"S{index}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=6.5,
                color="#344054",
            )

    interval_levels_ms = [baseline_ms, *cumulative_errors_ms[:-1]]
    ax.stairs(
        interval_levels_ms,
        edges_ms,
        baseline=None,
        color="#6941C6",
        linewidth=1.4,
    )
    ax.vlines(
        edges_ms[-1],
        interval_levels_ms[-1],
        cumulative_errors_ms[-1],
        color="#6941C6",
        linewidth=1.4,
    )
    ax.scatter(edges_ms[1:], cumulative_errors_ms, color="#6941C6", s=9, zorder=3)
    ax.axhline(0.0, color="#667085", linewidth=0.9, linestyle="--")
    ax.set_ylabel("cumulative\nsim − measured (ms)", fontsize=8)
    ax.set_xlabel("simulated cumulative critical-path contribution (ms)", fontsize=8)
    ax.grid(True, axis="y", alpha=0.35)

    total_delta_ms = simulated_sum_ms - measured_sum_ms
    relative_diff_pct = (
        total_delta_ms / measured_sum_ms * 100.0 if measured_sum_ms > 1e-12 else math.nan
    )
    relative_label = f"{relative_diff_pct:+.1f}%" if math.isfinite(relative_diff_pct) else "n/a"
    ax.text(
        edges_ms[-1],
        cumulative_errors_ms[-1],
        f" total {total_delta_ms:+.3f} ms ({relative_label})",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#344054",
    )


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
                "concurrent_hidden_ms": 0.0,
                "row_count": 0,
            }
        aggregate = aggregates[key]
        aggregate["calls"] += int(row.get("calls", 1))
        aggregate["duration_ms"] += float(row["duration_ms"])
        aggregate["concurrent_hidden_ms"] += float(row.get("concurrent_hidden_ms", 0.0) or 0.0)
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
    # A worklet compiled into two tree positions gives its slots the same name
    # in both, so without this the legend prints two identical rows and the
    # arrows into them cannot be told apart. The analyzer sends the shortest
    # label that separates them, and only when there is something to separate.
    context = item.get("context")
    name = _short_kernel_name(item["name"])
    if context:
        name = f"{name} @{context}"
    return (
        f"{segment_id} {name} [{item['kind']}] · "
        f"×{item['multiplicity']} · {float(item['critical_path_ms']):.3f} ms · {operation}"
    )


def _short_kernel_name(name: str, limit: int = 70) -> str:
    compact = " ".join(name.split())
    return compact if len(compact) <= limit else f"{compact[: limit - 1]}…"
