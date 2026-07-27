"""Render scalar sweep metrics from `sweep_metrics_grid.json`.

One axis becomes grouped line plots. Two axes become one heatmap per scalar.
Additional axes facet those heatmaps in manifest/domain order; missing cells stay
blank. This module arranges Rust-projected values but never reads run artifacts.
"""

from __future__ import annotations

import itertools
import json
import math
from collections import defaultdict
from collections.abc import Callable
from functools import partial
from pathlib import Path

from common.layout import load_payload, plot_output_path
from common.style import CURVE, MARKER, plt, save_plot
from matplotlib.colors import LinearSegmentedColormap

_LINE_COLORS = [CURVE, "#F58518", "#54A24B", "#B279A2"]
# Keep a monotonic sequential scale while avoiding both near-white and very dark
# endpoints. The narrower lightness range makes dense annotated grids calmer.
_BASE_BLUES = plt.get_cmap("Blues")
_HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "sweep_blues",
    [_BASE_BLUES(position) for position in (0.12, 0.28, 0.44, 0.60, 0.72)],
)


def render(log_dir: Path) -> list[Callable[[], Path]]:
    payload = load_payload(log_dir, "sweep_metrics_grid.json")
    axes = payload.get("axes") or []
    rows = payload.get("runs") or []
    metrics = payload.get("metrics") or []
    if not axes or not rows or not metrics:
        print("[sweep] nothing to render: empty axes, runs, or metrics")
        return []

    experiment_label = Path(payload.get("meta", {}).get("experiment_dir", str(log_dir))).name
    if len(axes) == 1:
        grouped = defaultdict(list)
        for metric in metrics:
            grouped[metric["group"]].append(metric)
        return [
            partial(
                _render_lines,
                payload,
                group_metrics,
                plot_output_path(log_dir, f"sweep_{group_name}.png"),
                experiment_label=experiment_label,
            )
            for group_name, group_metrics in grouped.items()
        ]

    return [
        partial(
            _render_heatmaps,
            payload,
            metric,
            plot_output_path(log_dir, f"sweep_{metric['key']}.png"),
            experiment_label=experiment_label,
        )
        for metric in metrics
    ]


def _render_lines(
    payload: dict,
    metrics: list[dict],
    out_path: Path,
    *,
    experiment_label: str,
) -> Path:
    axis = payload["axes"][0]
    domain = payload["domains"][axis]
    rows_by_value = {_value_key(row["coordinates"][axis]): row for row in payload["runs"]}
    numeric_axis = all(_is_number(value) for value in domain)
    x_values = [float(value) for value in domain] if numeric_axis else list(range(len(domain)))
    fig, plot_axis = plt.subplots(figsize=(9.0, 4.8))
    for index, metric in enumerate(metrics):
        values = [
            _display_value(
                rows_by_value.get(_value_key(value), {}).get("metrics", {}).get(metric["key"]),
                metric["unit"],
            )
            for value in domain
        ]
        plot_axis.plot(
            x_values,
            values,
            color=_LINE_COLORS[index % len(_LINE_COLORS)],
            marker="o",
            linewidth=2.0,
            label=metric["label"],
        )
    if not numeric_axis:
        plot_axis.set_xticks(x_values, [_domain_label(payload, axis, value) for value in domain])
    plot_axis.set_title(f"{experiment_label}\n{metrics[0]['group'].replace('_', ' ').title()}")
    plot_axis.set_xlabel(axis)
    plot_axis.set_ylabel(_shared_unit(metrics))
    plot_axis.grid(True)
    plot_axis.legend(frameon=False)
    save_plot(fig, out_path)
    return out_path


def _render_heatmaps(
    payload: dict,
    metric: dict,
    out_path: Path,
    *,
    experiment_label: str,
) -> Path:
    x_axis, y_axis = payload["axes"][:2]
    facet_axes = payload["axes"][2:]
    facet_domains = [payload["domains"][axis] for axis in facet_axes]
    facet_coordinates = list(itertools.product(*facet_domains)) if facet_axes else [()]
    panel_count = len(facet_coordinates)
    column_count = max(1, math.ceil(math.sqrt(panel_count)))
    row_count = math.ceil(panel_count / column_count)
    fig, plot_axes = plt.subplots(
        row_count,
        column_count,
        figsize=(max(7.2, 4.8 * column_count), max(5.2, 4.1 * row_count)),
        squeeze=False,
    )
    image = None
    for panel_index, facet_values in enumerate(facet_coordinates):
        plot_axis = plot_axes[panel_index // column_count][panel_index % column_count]
        matrix = _metric_matrix(payload, metric, dict(zip(facet_axes, facet_values)))
        image = plot_axis.imshow(matrix, aspect="auto", origin="lower", cmap=_HEATMAP_CMAP)
        x_domain = payload["domains"][x_axis]
        y_domain = payload["domains"][y_axis]
        plot_axis.set_xticks(
            range(len(x_domain)),
            [_domain_label(payload, x_axis, value) for value in x_domain],
            rotation=35,
            ha="right",
        )
        plot_axis.set_yticks(
            range(len(y_domain)),
            [_domain_label(payload, y_axis, value) for value in y_domain],
        )
        plot_axis.set_xlabel(x_axis)
        plot_axis.set_ylabel(y_axis)
        if facet_axes:
            plot_axis.set_title(_facet_title(facet_axes, facet_values))
        _annotate_matrix(plot_axis, matrix, metric["unit"])

    for panel_index in range(panel_count, row_count * column_count):
        plot_axes[panel_index // column_count][panel_index % column_count].set_visible(False)
    # Reserve stable margins explicitly. Matplotlib's constrained layout can
    # clip a two-line suptitle when a long vertical colorbar label is present.
    fig.subplots_adjust(left=0.1, right=0.79, top=0.84, bottom=0.16, wspace=0.34, hspace=0.48)
    if image is not None:
        colorbar_axis = fig.add_axes((0.84, 0.18, 0.022, 0.58))
        fig.colorbar(
            image,
            cax=colorbar_axis,
            label=_unit_label(metric["unit"]),
        )
    fig.suptitle(
        f"{experiment_label} · {metric['label']}",
        fontsize=13,
        fontweight="bold",
        y=0.95,
    )
    save_plot(fig, out_path, tight=False)
    return out_path


def _metric_matrix(payload: dict, metric: dict, facet: dict) -> list[list[float]]:
    x_axis, y_axis = payload["axes"][:2]
    x_domain = payload["domains"][x_axis]
    y_domain = payload["domains"][y_axis]
    matrix = [[math.nan for _ in x_domain] for _ in y_domain]
    x_positions = {_value_key(value): index for index, value in enumerate(x_domain)}
    y_positions = {_value_key(value): index for index, value in enumerate(y_domain)}
    for row in payload["runs"]:
        coordinates = row["coordinates"]
        if any(_value_key(coordinates[axis]) != _value_key(value) for axis, value in facet.items()):
            continue
        value = _display_value(row.get("metrics", {}).get(metric["key"]), metric["unit"])
        if value is None:
            continue
        matrix[y_positions[_value_key(coordinates[y_axis])]][
            x_positions[_value_key(coordinates[x_axis])]
        ] = value
    return matrix


def _annotate_matrix(plot_axis, matrix: list[list[float]], unit: str) -> None:
    finite_values = [value for row in matrix for value in row if not math.isnan(value)]
    minimum = min(finite_values) if finite_values else 0.0
    maximum = max(finite_values) if finite_values else 0.0
    for y_index, row in enumerate(matrix):
        for x_index, value in enumerate(row):
            if math.isnan(value):
                plot_axis.text(x_index, y_index, "—", ha="center", va="center", color=MARKER)
                continue
            normalized = 0.5 if maximum == minimum else (value - minimum) / (maximum - minimum)
            red, green, blue, _ = _HEATMAP_CMAP(normalized)
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            color = "#111827" if luminance >= 0.55 else "white"
            plot_axis.text(
                x_index,
                y_index,
                _short_value(value, unit),
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )


def _domain_label(payload: dict, axis: str, value) -> str:
    key = _value_key(value)
    for row in payload["runs"]:
        if _value_key(row["coordinates"][axis]) == key:
            label = row.get("labels", {}).get(axis)
            if label:
                return label
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _facet_title(axes: list[str], values: tuple) -> str:
    if not axes:
        return "all runs"
    return " · ".join(
        f"{axis}={json.dumps(value, ensure_ascii=False, separators=(',', ':'))}"
        for axis, value in zip(axes, values)
    )


def _value_key(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _display_value(value, unit: str):
    if not _is_number(value):
        return None
    return float(value) * 100.0 if unit == "%" else float(value)


def _shared_unit(metrics: list[dict]) -> str:
    units = {metric["unit"] for metric in metrics}
    return _unit_label(next(iter(units))) if len(units) == 1 else "value"


def _unit_label(unit: str) -> str:
    return {
        "ms/token": "milliseconds / token",
        "ms": "milliseconds",
        "tok/s": "tokens / s",
        "%": "percent",
        "req/s": "requests / s",
        "requests": "requests",
    }.get(unit, unit)


def _short_value(value: float, unit: str) -> str:
    if unit in ("tok/s", "requests") and abs(value) >= 1000:
        return f"{value / 1000:.1f}k"
    if unit == "%":
        return f"{value:.1f}%"
    if abs(value) >= 100:
        return f"{value:.0f}"
    return f"{value:.2f}"
