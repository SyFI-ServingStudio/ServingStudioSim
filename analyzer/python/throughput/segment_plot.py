"""Render per-segment throughput from the Rust `throughput_segments.json` payload.

Two figures, both per-GPU prefill / decode / total tokens-per-second drawn as a
step function over time (each segment is a constant rate, so steps read more
honestly than a smoothed line):

- `throughput_segments.png` — fine view, one step per `request_state` snapshot
  interval (can be hundreds of steps on a long run).
- `throughput_binned.png` — coarse view, ≤10 equal-width bins (ref's trend shape)
  plus a dashed run-average reference line per series.

Reads only the payload — no parquet. `render` returns the two render *jobs*
(callables that draw a figure and return its path); `__main__` runs them in
parallel.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable

from common.figure import add_legend, corner_box, finalize, fmt_value, new_axes
from common.style import ACCENT, CURVE, MARKER

# Per-series line colors: total is the primary curve, prefill/decode the accents.
_COLORS = {"total": CURVE, "prefill": ACCENT, "decode": "#54A24B"}


def _tps(value: float) -> str:
    """Tokens-per-second label via the shared unit-aware `fmt_value` (the same path
    the CDF plot uses) — thousands-separated, no decimals."""
    return fmt_value(value, "tok/s")


def render(log_dir: Path) -> list[Callable[[], Path]]:
    from common.layout import load_payload, plot_output_path

    payload = load_payload(log_dir, "throughput_segments.json")
    if not payload.get("series") or not payload.get("t_start_ms"):
        reason = payload.get("meta", {}).get("reason", "no segments in throughput_segments.json")
        print(f"[segment_plot] nothing to render: {reason}")
        return []
    meta = payload.get("meta", {})
    avg = meta.get("avg_per_gpu", {})
    gpu = _gpu_suffix(meta)
    run_label = Path(meta.get("log_dir", str(log_dir))).name

    # Fine view = the payload's top-level segment arrays; coarse = the `coarse`
    # block (falls back to fine if a degenerate single-bin run has none). Both go
    # through one renderer — the only difference is the binned plot draws dashed
    # run-average reference lines and names the bin count.
    coarse = payload.get("coarse") or payload
    n_bins = len(coarse["series"][0].get("per_gpu") or [])
    return [
        partial(_render_view, payload, plot_output_path(log_dir, "throughput_segments.png"),
                title=f"Per-GPU throughput  {gpu}", avg=avg, avg_lines=False, run_label=run_label),
        partial(_render_view, coarse, plot_output_path(log_dir, "throughput_binned.png"),
                title=f"Per-GPU throughput · {n_bins} bins  {gpu}", avg=avg, avg_lines=True,
                run_label=run_label),
    ]


def _edges_s(t_start: list[float], t_end: list[float]) -> list[float]:
    """Segment edges in seconds: consecutive segments are contiguous, so the n+1
    edges are the starts plus the final end (one flat `stairs` step per segment)."""
    return [t / 1000.0 for t in (*t_start, t_end[-1])]


def _draw_series(ax, view: dict) -> float:
    """Step-plot the total/prefill/decode per-GPU series of `view` (a block with
    `t_start_ms`/`t_end_ms`/`series`). Returns the peak total (for y-limit)."""
    edges = _edges_s(view["t_start_ms"], view["t_end_ms"])
    peak_total = 0.0
    for series in view["series"]:
        values = series.get("per_gpu") or []
        if not values:
            continue
        key = series.get("key", "")
        ax.stairs(values, edges, label=series.get("label", key),
                  color=_COLORS.get(key, MARKER), linewidth=2.0, baseline=None)
        if key == "total":
            peak_total = max(values)
    ax.set_xlim(left=edges[0], right=edges[-1])
    return peak_total


def _gpu_suffix(meta: dict) -> str:
    return f"({meta.get('gpu_name') or 'unknown GPU'} ×{meta.get('num_gpus', 1)})"


def _render_view(
    view: dict, out_path: Path, *,
    title: str, avg: dict, avg_lines: bool, run_label: str = "",
) -> Path:
    """Draw one throughput step plot (fine or coarse) to `out_path`. `avg_lines`
    adds a dashed per-series run-average reference (the coarse/binned view)."""
    fig, ax = new_axes(figsize=(9.0, 4.5))
    peak_total = _draw_series(ax, view)
    if avg_lines:
        for key in ("total", "prefill", "decode"):
            a = avg.get(key)
            if a is not None:
                ax.axhline(a, color=_COLORS[key], linestyle="--", linewidth=1.0, alpha=0.6)
    corner_box(ax, [f"avg = {_tps(avg.get('total', 0.0))}", f"peak = {_tps(peak_total)}"],
               loc="upper right")
    ax.set_ylim(bottom=0.0, top=max(1.0, peak_total * 1.08))
    add_legend(ax)
    finalize(
        fig, ax, out_path,
        title=title, xlabel="sim time (s)", ylabel="tokens / s per GPU", run_label=run_label,
    )
    return out_path
