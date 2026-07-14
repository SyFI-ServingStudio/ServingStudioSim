"""Render overall / pool / worker kernel-time composition as stacked bars.

Overall spans the figure top. Pool summaries and their workers flow through one
to three balanced columns below it. A pool that is taller than a column wraps
into a clearly marked continuation box; the summary row appears only once.
Every row is a 0–100% horizontal stack using one global position color mapping.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
from functools import partial
from pathlib import Path

from common.figure import corner_box, fmt_count, fmt_ms
from common.layout import load_payload, plot_output_path
from common.style import CURVE, MARKER, plt, save_plot
from matplotlib.patches import Patch, Rectangle

_PAYLOAD = "kernel_time_share_composition.json"
_OUTPUT = "kernel_time_share_composition.png"


def render(log_dir: Path) -> list[Callable[[], Path]]:
    payload = load_payload(log_dir, _PAYLOAD)
    if not payload.get("available") or not payload.get("overall", {}).get("segments"):
        reason = payload.get("meta", {}).get("reason", f"no composition in {_PAYLOAD}")
        print(f"[kernel_time_share_plot] nothing to render: {reason}")
        return []
    return [
        partial(
            _render,
            payload,
            plot_output_path(log_dir, _OUTPUT),
            run_label=Path(payload.get("meta", {}).get("log_dir", str(log_dir))).name,
        )
    ]


def _render(payload: dict, out_path: Path, *, run_label: str) -> Path:
    positions = [row for row in payload.get("positions", []) if row["overall_share_pct"] > 1e-10]
    position_names = [row["name"] for row in positions]
    short_labels = _short_unique_suffixes(position_names)
    colors = _position_colors(position_names)

    pools = sorted(payload.get("pools") or [], key=lambda row: row["pool_tag"])
    workers_by_pool: dict[str, list[dict]] = defaultdict(list)
    for worker in payload.get("workers") or []:
        workers_by_pool[worker["pool_tag"]].append(worker)
    for workers in workers_by_pool.values():
        workers.sort(key=lambda row: int(row["worker_id"]))

    total_detail_rows = sum(1 + len(workers_by_pool[pool["pool_tag"]]) for pool in pools)
    requested_columns = _column_count(total_detail_rows)
    columns = _flow_pool_blocks(pools, workers_by_pool, requested_columns)
    num_columns = max(1, len(columns))
    max_rows = max(sum(len(block["rows"]) for block in column) for column in columns)

    legend_columns = min(4, max(3, num_columns + 2))
    legend_rows = math.ceil(len(positions) / legend_columns)
    figure_width = 6.2 * num_columns
    figure_height = max(6.4, 3.5 + 0.54 * max_rows + 0.28 * legend_rows)
    fig = plt.figure(figsize=(figure_width, figure_height))
    grid = fig.add_gridspec(
        2,
        num_columns,
        height_ratios=[1.0, max(2.3, 0.58 * max_rows)],
        hspace=0.38,
        wspace=0.27,
    )

    overall_ax = fig.add_subplot(grid[0, :])
    _draw_stack(
        overall_ax,
        payload["overall"],
        y=0.0,
        colors=colors,
        short_labels=short_labels,
        label_threshold=4.0,
        show_names=True,
        strong_outline=True,
    )
    overall_ax.set_yticks([0.0])
    overall_ax.set_yticklabels([f"OVERALL\n{fmt_ms(payload['overall']['kernel_time_ms'])}"])
    overall_ax.get_yticklabels()[0].set_fontweight("bold")
    overall_ax.set_ylim(-0.62, 1.25)
    _style_percent_axis(overall_ax, show_xlabel=False)
    overall_ax.add_patch(
        Rectangle(
            (0.0, -0.48),
            100.0,
            0.96,
            facecolor=CURVE,
            edgecolor=CURVE,
            alpha=0.055,
            linewidth=1.4,
            zorder=-1,
            clip_on=False,
        )
    )
    meta = payload.get("meta", {})
    sample_note = (
        f"exact · {fmt_count(meta.get('raw_rows', 0))} rows"
        if meta.get("exact")
        else f"estimated · {fmt_count(meta.get('sampled_rows', 0))} / "
        f"{fmt_count(meta.get('raw_rows', 0))} rows"
    )
    num_pools = int(meta.get("num_pools", 0))
    num_workers = int(meta.get("num_workers", 0))
    corner_box(
        overall_ax,
        [
            sample_note,
            f"{fmt_count(num_pools)} {'pool' if num_pools == 1 else 'pools'} · "
            f"{fmt_count(num_workers)} {'worker' if num_workers == 1 else 'workers'}",
        ],
        loc="upper right",
    )

    detail_axes = [fig.add_subplot(grid[1, column]) for column in range(num_columns)]
    for column_index, (ax, blocks) in enumerate(zip(detail_axes, columns, strict=True)):
        _draw_detail_column(
            ax,
            blocks,
            colors=colors,
            short_labels=short_labels,
            column_index=column_index,
            num_columns=num_columns,
        )

    handles = [
        Patch(
            facecolor=colors[position["name"]],
            edgecolor="white",
            label=f"{short_labels[position['name']]}  {position['overall_share_pct']:.1f}%",
        )
        for position in positions
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=legend_columns,
        frameon=False,
        title="CostTree leaf position · overall share",
        fontsize=8,
        title_fontsize=9,
        handlelength=1.2,
        columnspacing=1.5,
    )
    fig.suptitle(
        f"{run_label}\nKernel-time composition by CostTree position",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    bottom = min(0.34, 0.07 + 0.035 * legend_rows)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.88, bottom=bottom)
    save_plot(fig, out_path)
    return out_path


def _draw_detail_column(
    ax,
    blocks: list[dict],
    *,
    colors: dict[str, object],
    short_labels: dict[str, str],
    column_index: int,
    num_columns: int,
) -> None:
    cursor = 0.0
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    pool_tick_indices: set[int] = set()
    for block_index, block in enumerate(blocks):
        if block_index:
            cursor += 0.72
        rows = block["rows"]
        first_y = -cursor
        row_y: list[float] = []
        for row in rows:
            y = -cursor
            row_y.append(y)
            is_pool = row["level"] == "pool"
            _draw_stack(
                ax,
                row["data"],
                y=y,
                colors=colors,
                short_labels=short_labels,
                label_threshold=7.0 if is_pool else 12.0,
                show_names=is_pool,
                strong_outline=is_pool,
            )
            tick_positions.append(y)
            if is_pool:
                pool_tick_indices.add(len(tick_labels))
                tick_labels.append(
                    f"POOL · {block['pool_tag']}\n{fmt_ms(row['data']['kernel_time_ms'])}"
                )
            else:
                tick_labels.append(
                    f"worker {row['data']['worker_id']}\n"
                    f"{fmt_ms(row['data']['kernel_time_ms'])}"
                )
            cursor += 1.0
        last_y = row_y[-1]
        ax.add_patch(
            Rectangle(
                (0.0, last_y - 0.47),
                100.0,
                first_y - last_y + 0.94,
                facecolor=CURVE,
                edgecolor=CURVE,
                alpha=0.035,
                linewidth=1.0,
                linestyle="--" if block["continued"] else "-",
                zorder=-1,
                clip_on=False,
            )
        )
        if block["continued"]:
            ax.text(
                0.0,
                first_y + 0.55,
                f"{block['pool_tag']} · continued",
                ha="left",
                va="bottom",
                fontsize=8,
                color=MARKER,
                fontweight="semibold",
            )

    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=8)
    for index, label in enumerate(ax.get_yticklabels()):
        if index in pool_tick_indices:
            label.set_fontweight("bold")
            label.set_color(CURVE)
    ax.set_ylim(-cursor + 0.28, 0.7)
    _style_percent_axis(ax, show_xlabel=True)
    title = "Pools & workers" if column_index == 0 else f"continued · {column_index + 1}/{num_columns}"
    ax.set_title(title, fontsize=11, fontweight="semibold")


def _draw_stack(
    ax,
    row: dict,
    *,
    y: float,
    colors: dict[str, object],
    short_labels: dict[str, str],
    label_threshold: float,
    show_names: bool,
    strong_outline: bool,
) -> None:
    left = 0.0
    for segment in row.get("segments") or []:
        width = float(segment["share_pct"])
        if width <= 1e-10:
            continue
        position = segment["position"]
        ax.barh(
            y,
            width,
            left=left,
            height=0.58,
            color=colors[position],
            edgecolor="white",
            linewidth=1.0 if strong_outline else 0.65,
            zorder=2,
        )
        if width >= label_threshold:
            short_label = short_labels[position]
            # A percentage still fits a narrow segment where a long semantic
            # suffix would collide with its neighbors. Name only segments whose
            # width is proportional to the actual label length.
            name_fits = width >= max(label_threshold, 0.9 * len(short_label))
            text = (
                f"{short_label}\n{width:.1f}%"
                if show_names and name_fits
                else f"{width:.0f}%"
            )
            ax.text(
                left + width / 2.0,
                y,
                text,
                ha="center",
                va="center",
                fontsize=6.5,
                color="#101828",
                fontweight="semibold" if strong_outline else "normal",
                clip_on=True,
                zorder=3,
            )
        left += width


def _style_percent_axis(ax, *, show_xlabel: bool) -> None:
    ax.set_xlim(0.0, 100.0)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("kernel time share (%)" if show_xlabel else "")
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)


def _column_count(total_rows: int) -> int:
    if total_rows <= 10:
        return 1
    if total_rows <= 24:
        return 2
    return 3


def _flow_pool_blocks(
    pools: list[dict],
    workers_by_pool: dict[str, list[dict]],
    requested_columns: int,
) -> list[list[dict]]:
    """Flow pool blocks through balanced columns, splitting oversized pools.

    Row capacity is based on actual bars (pool summary + workers), not number of
    pools. A continuation chunk carries only workers and gets a dashed box, so
    the summary value is never duplicated.
    """
    total_rows = sum(1 + len(workers_by_pool[pool["pool_tag"]]) for pool in pools)
    num_columns = min(max(1, requested_columns), max(1, total_rows))
    target_rows = math.ceil(total_rows / num_columns)
    columns: list[list[dict]] = [[]]
    used_rows = 0

    for pool in pools:
        pool_tag = pool["pool_tag"]
        pending = [{"level": "pool", "data": pool}]
        pending.extend({"level": "worker", "data": worker} for worker in workers_by_pool[pool_tag])
        continued = False
        while pending:
            remaining = target_rows - used_rows
            # Do not orphan a new pool summary at the foot of a column when a
            # fresh column is still available.
            if (
                not continued
                and used_rows > 0
                and remaining < min(3, len(pending))
                and len(columns) < num_columns
            ):
                columns.append([])
                used_rows = 0
                remaining = target_rows
            if remaining <= 0 and len(columns) < num_columns:
                columns.append([])
                used_rows = 0
                remaining = target_rows
            take = len(pending) if len(columns) == num_columns else min(len(pending), remaining)
            take = max(1, take)
            columns[-1].append(
                {
                    "pool_tag": pool_tag,
                    "continued": continued,
                    "rows": pending[:take],
                }
            )
            pending = pending[take:]
            used_rows += take
            continued = True
            if pending and len(columns) < num_columns:
                columns.append([])
                used_rows = 0

    return [column for column in columns if column]


def _short_unique_suffixes(names: list[str]) -> dict[str, str]:
    parts = {name: name.split(".") for name in names}
    labels: dict[str, str] = {}
    for name, path in parts.items():
        for depth in range(1, len(path) + 1):
            candidate = ".".join(path[-depth:])
            if sum(other == candidate or other.endswith(f".{candidate}") for other in names) == 1:
                labels[name] = candidate
                break
        else:
            labels[name] = name
    return labels


def _position_colors(names: list[str]) -> dict[str, object]:
    # Three built-in qualitative maps provide 60 stable categorical colors
    # without adding subject-specific palette constants or changing shared style.
    palette = [
        color
        for cmap_name in ("tab20", "tab20b", "tab20c")
        for color in plt.get_cmap(cmap_name).colors
    ]
    return {name: palette[index % len(palette)] for index, name in enumerate(names)}
