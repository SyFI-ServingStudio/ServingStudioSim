"""Render per-pool KV-cache occupancy from the Rust `kv_occupancy_series.json`.

One figure (`kv_occupancy.png`). The focus metric is `active` (resident/current
KV) drawn per shard: each worker's own series is a faint **shadow** step line, and
the across-shard **min / max / mean** stand out on top — a bold mean line and a
shaded min–max band. A wide band means shard imbalance. `retained_prefix` is the
measured prefix-cache component of active KV and is drawn as an amber area/line.
`projected` (the admission gate's future estimate) and `promised`
(reserved-not-resident) are drawn as pool-mean context lines (dashed / dotted).
When `run_meta` carries a capacity the y-axis is percent of capacity with a dotted
100% line; otherwise it falls back to raw tokens. `projected` is intentionally
allowed above 100% — that's the signal the admitted horizon outruns the pool.

Reads only the payload — no parquet. `render` returns render *jobs*; `__main__`
runs them.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path

import matplotlib.colors as mcolors
from common.figure import add_legend, corner_box, finalize, fmt_count, fmt_value, new_axes
from common.style import CURVE, MARKER

# Color encodes the METRIC, not the shard: `active` is one hue (its shards are shades
# of it), retained prefix, `projected`, and `promised` each their own hue. So the
# eye reads a metric by color and a shard by lightness within the active hue.
_ACTIVE_HUE = CURVE  # blue — resident/current KV
_RETAINED_PREFIX_COLOR = "#D97706"  # amber — retained prefix-cache component
_PROJECTED_COLOR = "#C1121F"  # dark red — admission-gate future estimate
_PROMISED_COLOR = "#6D28D9"  # violet — reserved-not-resident


def _tint(color: str, amount: float) -> tuple[float, float, float]:
    """Blend `color` toward white by `amount` (0 = color, 1 = white) — lighter shades
    of one hue for the per-shard lines."""
    red, green, blue = mcolors.to_rgb(color)
    return (
        red + (1 - red) * amount,
        green + (1 - green) * amount,
        blue + (1 - blue) * amount,
    )


def _shade(color: str, amount: float) -> tuple[float, float, float]:
    """Blend `color` toward black by `amount` — the darker, standout mean line."""
    red, green, blue = mcolors.to_rgb(color)
    return (red * (1 - amount), green * (1 - amount), blue * (1 - amount))


def _shard_tints(worker_count: int) -> list[tuple[float, float, float]]:
    """One active-hue lightness shade per shard, ordered light to mid."""
    if worker_count <= 1:
        return [_tint(_ACTIVE_HUE, 0.35)]
    return [
        _tint(_ACTIVE_HUE, 0.60 - 0.45 * (worker_index / (worker_count - 1)))
        for worker_index in range(worker_count)
    ]


def render(log_dir: Path) -> list[Callable[[], Path]]:
    from common.layout import load_payload, plot_output_path

    payload = load_payload(log_dir, "kv_occupancy_series.json")
    if not payload.get("series") or not payload.get("t_start_ms"):
        reason = payload.get("meta", {}).get("reason", "no series in kv_occupancy_series.json")
        print(f"[kv_occupancy_plot] nothing to render: {reason}")
        return []
    meta = payload.get("meta", {})
    run_label = Path(meta.get("log_dir", str(log_dir))).name
    return [
        partial(
            _render, payload, plot_output_path(log_dir, "kv_occupancy.png"), run_label=run_label
        ),
    ]


def _edges_s(start_times_ms: list[float], end_times_ms: list[float]) -> list[float]:
    """Bin edges in seconds: contiguous bins → n+1 edges (one flat `stairs` step
    per bin)."""
    return [time_ms / 1000.0 for time_ms in (*start_times_ms, end_times_ms[-1])]


def _has_capacity(payload: dict) -> bool:
    """True when every series carries a positive capacity — the precondition for a
    percent axis (mixing percent and raw tokens on one axis would mislabel)."""
    series = payload.get("series", [])
    return bool(series) and all(
        (series_definition.get("capacity_tokens") or 0) > 0 for series_definition in series
    )


def _render(payload: dict, out_path: Path, *, run_label: str = "") -> Path:
    """Draw the per-shard KV occupancy plot (shadows + standout min/max/mean)."""
    fig, ax = new_axes(figsize=(9.5, 5.0))
    edges = _edges_s(payload["t_start_ms"], payload["t_end_ms"])
    starts = edges[:-1]  # left edge per bin, for the stepped min–max band fill
    pct_mode = _has_capacity(payload)

    def scale(tokens: list[float], capacity: float | None) -> list[float]:
        # Percent of capacity when known, else raw tokens.
        if pct_mode and capacity:
            return [token_count / capacity * 100.0 for token_count in tokens]
        return list(tokens)

    peak_active_mean = 0.0
    peak_active_max = 0.0
    peak_retained_prefix = 0.0
    peak_projected = 0.0
    y_axis_top = 0.0
    has_retained_prefix_breakdown = bool(
        payload.get("meta", {}).get("has_retained_prefix_breakdown", False)
    )
    for series_definition in payload["series"]:
        capacity = series_definition.get("capacity_tokens")
        series_label = series_definition.get("label", series_definition.get("key", ""))
        active_series = series_definition.get("active") or {}
        active_mean = scale(active_series.get("mean") or [], capacity)
        active_minimum = scale(active_series.get("min") or [], capacity)
        active_maximum = scale(active_series.get("max") or [], capacity)
        if not active_mean:
            continue
        worker_series = series_definition.get("workers") or []
        has_multiple_workers = len(worker_series) > 1
        label_prefix = f"{series_label} · " if len(payload["series"]) > 1 else ""

        # (1) Retained prefix is a component of active, so render its pool mean as
        # an area under the active curves rather than as another total.
        retained_prefix_mean = scale(
            (series_definition.get("retained_prefix") or {}).get("mean") or [], capacity
        )
        if has_retained_prefix_breakdown and retained_prefix_mean:
            ax.fill_between(
                starts,
                0.0,
                retained_prefix_mean,
                step="post",
                color=_RETAINED_PREFIX_COLOR,
                alpha=0.16,
                linewidth=0,
            )
            ax.stairs(
                retained_prefix_mean,
                edges,
                color=_RETAINED_PREFIX_COLOR,
                linewidth=1.7,
                baseline=None,
                label=f"{label_prefix}retained prefix",
            )
            peak_retained_prefix = max(peak_retained_prefix, max(retained_prefix_mean))

        # (2) Per-shard active lines — shades of ONE hue (the active hue), light→mid,
        # so a shard reads by lightness while staying clearly "active".
        tints = _shard_tints(len(worker_series))
        for worker_definition, tint in zip(worker_series, tints):
            worker_active = scale(worker_definition.get("active_tokens") or [], capacity)
            if not worker_active:
                continue
            ax.stairs(
                worker_active,
                edges,
                color=tint,
                linewidth=1.1,
                baseline=None,
                label=f"{label_prefix}active · w{worker_definition.get('worker_id')}",
            )

        # (3) Standout across-shard aggregate, same hue: light-blue min–max band +
        # bold dark-blue mean.
        if has_multiple_workers:
            ax.fill_between(
                starts,
                active_minimum,
                active_maximum,
                step="post",
                color=_tint(_ACTIVE_HUE, 0.55),
                alpha=0.45,
                linewidth=0,
                label=f"{label_prefix}active · min–max",
            )
        ax.stairs(
            active_mean,
            edges,
            color=_shade(_ACTIVE_HUE, 0.25),
            linewidth=2.6,
            baseline=None,
            label=(
                f"{label_prefix}active · mean" if has_multiple_workers else f"{label_prefix}active"
            ),
        )

        # (4) Context: projected (future estimate) + promised (reserved), pool-mean,
        # each its own hue so the three metrics read apart by color.
        projected_mean = scale(
            (series_definition.get("projected") or {}).get("mean") or [], capacity
        )
        promised_mean = scale((series_definition.get("promised") or {}).get("mean") or [], capacity)
        if projected_mean:
            ax.stairs(
                projected_mean,
                edges,
                color=_PROJECTED_COLOR,
                linewidth=1.6,
                linestyle="--",
                alpha=0.9,
                baseline=None,
                label=f"{label_prefix}projected",
            )
            peak_projected = max(peak_projected, max(projected_mean))
        if any(value > 0 for value in promised_mean):
            ax.stairs(
                promised_mean,
                edges,
                color=_PROMISED_COLOR,
                linewidth=1.4,
                linestyle=":",
                alpha=0.85,
                baseline=None,
                label=f"{label_prefix}promised",
            )

        peak_active_mean = max(peak_active_mean, max(active_mean))
        peak_active_max = max(peak_active_max, max(active_maximum))
        y_axis_top = max(y_axis_top, max(active_maximum), max(projected_mean or [0.0]))

    if pct_mode:
        # Dotted full-capacity reference; headroom above 100% so an over-capacity
        # projected curve stays visible.
        ax.axhline(100.0, color=MARKER, linestyle=":", linewidth=1.0, alpha=0.7)
        ax.set_ylim(bottom=0.0, top=max(105.0, y_axis_top * 1.05))
        ax.set_ylabel("KV occupancy (% of capacity)")
        summary = [
            f"peak active (mean) = {fmt_value(peak_active_mean, '%')}",
            f"peak active (worst shard) = {fmt_value(peak_active_max, '%')}",
            f"peak projected = {fmt_value(peak_projected, '%')}",
        ]
        if has_retained_prefix_breakdown:
            summary.insert(
                2,
                f"peak retained prefix = {fmt_value(peak_retained_prefix, '%')}",
            )
    else:
        ax.set_ylim(bottom=0.0, top=max(1.0, y_axis_top * 1.05))
        ax.set_ylabel("KV occupancy (tokens)")
        summary = [
            f"peak active (mean) = {fmt_count(peak_active_mean)} tok",
            f"peak active (worst shard) = {fmt_count(peak_active_max)} tok",
            f"peak projected = {fmt_count(peak_projected)} tok",
        ]
        if has_retained_prefix_breakdown:
            summary.insert(
                2,
                f"peak retained prefix = {fmt_count(peak_retained_prefix)} tok",
            )

    ax.set_xlim(left=edges[0], right=edges[-1])
    corner_box(ax, summary, loc="upper left")
    add_legend(ax)
    finalize(
        fig,
        ax,
        out_path,
        title="KV-cache occupancy (per shard)",
        xlabel="sim time (s)",
        ylabel=ax.get_ylabel(),
        run_label=run_label,
    )
    return out_path
