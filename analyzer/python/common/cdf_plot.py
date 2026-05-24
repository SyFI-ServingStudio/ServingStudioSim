"""CDF renderer from a Rust `CdfSeries` payload entry.

A series is `{key,label,unit,n,x[],y_pct[],markers:{p50,p90,p99}}`. This module
owns only the CDF-specific drawing (cumulative curve + percentile vlines); figure
setup, title, labels, the corner summary box, grid, and save come from
`common.figure`. New plot types follow the same shape: draw data here, delegate
the rest.
"""

from __future__ import annotations

from pathlib import Path

from common.figure import corner_box, finalize, fmt_count, fmt_value, new_axes
from common.style import ACCENT, CURVE, MARKER

_PERCENTILES = (("p50", "p50"), ("p90", "p90"), ("p99", "p99"))


def render_cdf(series: dict, out_path: Path, *, run_label: str = "") -> Path:
    """Render one CDF series to `out_path` and return it (so a render job's result
    is the written path)."""
    fig, ax = new_axes()
    label = series.get("label", series.get("key", "metric"))
    unit = series.get("unit", "ms")
    x = series.get("x") or []
    y = series.get("y_pct") or []

    if not x:
        ax.text(0.5, 0.5, f"no samples for {label}", transform=ax.transAxes,
                ha="center", va="center", color=MARKER, fontsize=12)
    else:
        ax.plot(x, y, color=CURVE, linewidth=2.0, solid_capstyle="round")
        ax.fill_between(x, y, color=CURVE, alpha=0.06)
        corner_box(ax, _percentile_lines(ax, series.get("markers") or {}, unit))
        ax.set_xlim(left=0.0, right=max(1.0, max(x) * 1.03))

    ax.set_ylim(0, 100)
    finalize(
        fig, ax, out_path,
        title=f"{label} CDF  (n={fmt_count(series.get('n', 0))})",
        xlabel=f"{label} ({unit})",
        ylabel="cumulative %",
        run_label=run_label,
    )
    return out_path


def _percentile_lines(ax, markers: dict, unit: str = "ms") -> list[str]:
    """Draw p50/p90/p99 vlines; return their labels (in the series' unit) for the
    corner box (a box, not inline text, so labels stay readable when percentiles
    cluster)."""
    lines = []
    for key, name in _PERCENTILES:
        value = markers.get(key)
        if value is None:
            continue
        ax.axvline(value, color=ACCENT, linewidth=1.0, linestyle="--", alpha=0.7, zorder=1)
        lines.append(f"{name} = {fmt_value(value, unit)}")
    return lines
