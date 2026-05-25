"""Render per-batch composition from the Rust `batch_scatter.json` payload.

One figure, three stacked subplots sharing the sim-time x-axis — batch_tokens,
prefill_tokens, decode_request_count — each its own scatter on its own y-scale
(the counts live on different ranges, so overlaying them would flatten the small
one). Each point is a cost_log iteration; the points are a time-even downsample of
all iterations, and each panel's dashed line + corner box is the run-average (over
all iterations, from the report's stats).

Reads only the payload — no parquet. `render` returns a single render job. Reuses
the shared style/formatting (`common.style`, `common.figure`); the per-panel layout
is the only thing finalize (single-axes) can't do, so it's wired inline here.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable

from common.figure import corner_box, fmt_count
from common.style import ACCENT, CURVE, MARKER, plt, save_plot

# Per-field point colors (decode reuses the throughput-decode green for consistency).
_COLORS = {"batch_tokens": CURVE, "prefill_tokens": ACCENT, "decode_request_count": "#54A24B"}


def render(log_dir: Path) -> list[Callable[[], Path]]:
    from common.layout import load_payload, plot_output_path

    payload = load_payload(log_dir, "batch_scatter.json")
    if not payload.get("series") or not payload.get("time_ms"):
        reason = payload.get("meta", {}).get("reason", "no points in batch_scatter.json")
        print(f"[scatter_plot] nothing to render: {reason}")
        return []
    run_label = Path(payload.get("meta", {}).get("log_dir", str(log_dir))).name
    return [
        partial(_render, payload, plot_output_path(log_dir, "batch_scatter.png"),
                title="Per-batch composition", run_label=run_label),
    ]


def _render(payload: dict, out_path: Path, *, title: str, run_label: str = "") -> Path:
    """Draw the stacked per-field scatter (one subplot per series) to `out_path`."""
    series = [s for s in payload["series"] if s.get("values")]
    t_s = [t / 1000.0 for t in payload["time_ms"]]
    avg = payload.get("meta", {}).get("avg", {})
    n = max(len(series), 1)
    fig, axes = plt.subplots(n, 1, figsize=(9.0, 2.4 * n + 0.8), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, s in zip(axes, series):
        key = s.get("key", "")
        values = s["values"]
        color = _COLORS.get(key, MARKER)
        ax.scatter(t_s, values, s=6, alpha=0.4, linewidths=0, color=color)
        a = avg.get(key)
        if a is not None:
            ax.axhline(a, color=color, linestyle="--", linewidth=1.0, alpha=0.7)
            corner_box(ax, [f"avg = {fmt_count(a)}"], loc="upper right")
        ax.set_ylabel(s.get("label", key))
        ax.set_ylim(bottom=0.0, top=max(1.0, max(values) * 1.08))
        ax.grid(True)
    if t_s:
        axes[0].set_xlim(left=min(t_s), right=max(t_s))
    axes[-1].set_xlabel("sim time (s)")
    fig.suptitle(f"{run_label}\n{title}" if run_label else title, fontsize=16, fontweight="bold")
    save_plot(fig, out_path)
    return out_path
