"""Reusable plot scaffolding shared by every analyzer figure.

The intent: a new plot type only writes its *data-specific* drawing (the curve,
the bars, the heatmap) and calls these for everything generic — figure creation,
the run-labeled title, axis labels, grid, the corner annotation box, value
formatting, and saving. Keep cross-cutting plot conventions here, not in each
plot module.
"""

from __future__ import annotations

from pathlib import Path

from common.style import plt, save_plot

# Corner anchors → (x, y, ha, va) in axes fraction. CDF/timeseries curves leave
# the upper-left empty; bars vary, so callers pick.
_CORNERS = {
    "upper left": (0.03, 0.97, "left", "top"),
    "upper right": (0.97, 0.97, "right", "top"),
    "lower left": (0.03, 0.03, "left", "bottom"),
    "lower right": (0.97, 0.03, "right", "bottom"),
}


def fmt_ms(value: float) -> str:
    """Human ms with auto unit: µs / ms / s. Use for any millisecond axis label
    or annotation so units read consistently across plots."""
    if value >= 1000:
        return f"{value / 1000:.2f} s"
    if value >= 1:
        return f"{value:.1f} ms"
    return f"{value * 1000:.0f} µs"


def fmt_value(value: float, unit: str = "ms") -> str:
    """Format a metric value for a label, honoring its unit. The millisecond
    family (`ms`, `ms/token`) auto-scales µs/ms/s via `fmt_ms`; `%` prints a
    percent; anything else prints the number with its unit verbatim. Use this for
    any per-metric annotation so a non-latency CDF isn't mislabeled in ms."""
    if unit in ("ms", "ms/token"):
        scaled = fmt_ms(value)
        return f"{scaled}/token" if unit == "ms/token" else scaled
    if unit == "%":
        return f"{value:.1f}%"
    return f"{value:.2f} {unit}"


def fmt_count(value: float) -> str:
    """Thousands-separated integer count (e.g. sample n)."""
    return f"{int(value):,}"


def new_axes(figsize: tuple[float, float] = (7.5, 4.5)):
    """Standard single-axes figure."""
    return plt.subplots(figsize=figsize)


def corner_box(ax, lines: list[str], *, loc: str = "upper left") -> None:
    """A rounded white annotation box of text `lines` anchored to a corner —
    the standard way to surface summary numbers (percentiles, totals) on a plot
    without colliding with the data."""
    if not lines:
        return
    x, y, ha, va = _CORNERS[loc]
    ax.text(
        x, y, "\n".join(lines),
        transform=ax.transAxes, ha=ha, va=va,
        fontsize=9, color="#111827",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
              "edgecolor": "#D0D5DD", "alpha": 0.92},
    )


def finalize(
    fig, ax, out_path: Path, *,
    title: str, xlabel: str, ylabel: str,
    run_label: str = "", grid: bool = True,
) -> None:
    """End-of-plot: run-labeled title, axis labels, grid, save+close. Every plot
    module ends with this single call."""
    ax.set_title(f"{run_label}\n{title}" if run_label else title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True)
    save_plot(fig, out_path)
