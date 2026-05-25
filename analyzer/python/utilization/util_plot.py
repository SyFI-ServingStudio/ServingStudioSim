"""Render per-pool GPU utilization from the Rust `utilization_series.json` payload.

One figure (`utilization_series.png`): the fraction of each pool's workers busy
computing, drawn as a step function over sim time (each fine bin is a constant
fraction, so steps read more honestly than a smoothed line). A dotted 100% line
marks full capacity; each pool gets a dashed run-average reference in its color.

Reads only the payload — no parquet. `render` returns a single render *job*
(a callable that draws the figure and returns its path); `__main__` runs it.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable

from common.figure import add_legend, corner_box, finalize, fmt_value, new_axes
from common.style import ACCENT, CURVE, MARKER

# Per-pool line colors, cycled by pool order. One pool today; a disaggregated
# deployment with several pools draws one line each.
_POOL_COLORS = [CURVE, ACCENT, "#54A24B", "#E45756", "#72B7B2", "#B279A2"]


def render(log_dir: Path) -> list[Callable[[], Path]]:
    from common.layout import load_payload, plot_output_path

    payload = load_payload(log_dir, "utilization_series.json")
    if not payload.get("series") or not payload.get("t_start_ms"):
        reason = payload.get("meta", {}).get("reason", "no series in utilization_series.json")
        print(f"[util_plot] nothing to render: {reason}")
        return []
    meta = payload.get("meta", {})
    gpu = meta.get("gpu_name") or "unknown GPU"
    run_label = Path(meta.get("log_dir", str(log_dir))).name
    return [
        partial(_render, payload, plot_output_path(log_dir, "utilization_series.png"),
                title=f"GPU utilization  ({gpu})", run_label=run_label),
    ]


def _edges_s(t_start: list[float], t_end: list[float]) -> list[float]:
    """Bin edges in seconds: bins are contiguous, so the n+1 edges are the starts
    plus the final end (one flat `stairs` step per bin)."""
    return [t / 1000.0 for t in (*t_start, t_end[-1])]


def _render(payload: dict, out_path: Path, *, title: str, run_label: str = "") -> Path:
    """Draw the per-pool utilization step plot (percent y-axis) to `out_path`."""
    fig, ax = new_axes(figsize=(9.0, 4.5))
    edges = _edges_s(payload["t_start_ms"], payload["t_end_ms"])
    avg = payload.get("meta", {}).get("avg", {})
    peak_pct = 0.0
    avg_pcts: list[float] = []
    for i, series in enumerate(payload["series"]):
        util = series.get("util") or []
        if not util:
            continue
        key = series.get("key", "")
        color = _POOL_COLORS[i % len(_POOL_COLORS)]
        pct = [v * 100.0 for v in util]
        ax.stairs(pct, edges, label=series.get("label", key),
                  color=color, linewidth=2.0, baseline=None)
        peak_pct = max(peak_pct, max(pct))
        a = avg.get(key)
        if a is not None:
            ax.axhline(a * 100.0, color=color, linestyle="--", linewidth=1.0, alpha=0.6)
            avg_pcts.append(a * 100.0)
    # Dotted full-capacity reference (100% of workers busy).
    ax.axhline(100.0, color=MARKER, linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_xlim(left=edges[0], right=edges[-1])
    ax.set_ylim(bottom=0.0, top=105.0)
    mean_avg = sum(avg_pcts) / len(avg_pcts) if avg_pcts else 0.0
    corner_box(ax, [f"avg = {fmt_value(mean_avg, '%')}", f"peak = {fmt_value(peak_pct, '%')}"],
               loc="upper right")
    add_legend(ax)
    finalize(
        fig, ax, out_path,
        title=title, xlabel="sim time (s)", ylabel="GPU utilization (%)", run_label=run_label,
    )
    return out_path
