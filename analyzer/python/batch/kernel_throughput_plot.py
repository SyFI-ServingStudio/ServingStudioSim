"""Render achieved throughput per cost-tree location from the kernel-throughput
payload.

Two figures — achieved TFLOP/s (compute) and GB/s (memory bandwidth) — each a
horizontal bar of the per-location p50, with a p90 marker for spread. A
"location" is a cost-tree leaf name (e.g. `afd.post_attn.o_proj`); the parallel
branches of a Max node share a name, so `max(gemm, gemm)` is pooled into one bar.
Reads only the payload — no parquet. `render` returns one job per figure (a
callable that draws it and returns its path); `__main__` runs them in parallel.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable

from common.figure import corner_box, finalize, fmt_value, new_axes
from common.style import ACCENT, CURVE
from common.layout import load_payload, plot_output_path

_PAYLOAD = "kernel_throughput_locations.json"

# The two achieved-rate metrics, each `(payload key, figure label, unit)`.
_METRICS = (
    ("tflops", "achieved compute", "TFLOP/s"),
    ("gbps", "achieved memory BW", "GB/s"),
)


def render(log_dir: Path) -> list[Callable[[], Path]]:
    payload = load_payload(log_dir, _PAYLOAD)
    locations = payload.get("locations") or []
    if not locations:
        reason = payload.get("meta", {}).get("reason", f"no locations in {_PAYLOAD}")
        print(f"[kernel_throughput_plot] nothing to render: {reason}")
        return []
    meta = payload.get("meta", {})
    run_label = Path(meta.get("log_dir", str(log_dir))).name
    return [
        partial(_bar, log_dir, locations, meta, key, label, unit, run_label)
        for key, label, unit in _METRICS
    ]


def _bar(log_dir, locations, meta, key, label, unit, run_label) -> Path:
    """One horizontal bar chart: p50 achieved `key` per location (ascending), a
    p90 marker for the tail. Locations with no sample for this metric are dropped
    (a memory-bound kernel has no flops, and vice versa)."""
    rows = [
        (loc["name"], loc[key])
        for loc in locations
        if loc.get(key, {}).get("n", 0) > 0 and loc[key].get("p50") is not None
    ]
    rows.sort(key=lambda r: r[1]["p50"])
    out = plot_output_path(log_dir, f"kernel_{key}_by_location.png")

    fig, ax = new_axes(figsize=(8.0, max(3.0, 0.38 * len(rows) + 1.0)))
    y = range(len(rows))
    p50 = [r[1]["p50"] for r in rows]
    ax.barh(list(y), p50, color=CURVE, height=0.7)
    # p90 as a tail marker so a wide compute/BW spread (mixed shapes at one
    # location) is visible next to the median bar.
    p90 = [r[1].get("p90") for r in rows]
    ax.scatter(
        [v for v in p90 if v is not None],
        [i for i, v in enumerate(p90) if v is not None],
        color=ACCENT, s=18, zorder=3,
    )
    for i, v in zip(y, p50):
        ax.text(v, i, f" {fmt_value(v, unit)}", va="center", ha="left", fontsize=7)

    ax.set_yticks(list(y))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.margins(x=0.18)  # room for the value labels past the bar ends
    # Bars ascend, so the lower-right region is empty — safe for the legend box.
    corner_box(
        ax,
        [
            "bar = p50   • = p90",
            f"locations: {len(rows)}",
            f"sample: 1/{meta.get('sample_stride', '?')} iters",
        ],
        loc="lower right",
    )
    finalize(
        fig, ax, out,
        title=f"{label} p50 by cost-tree location",
        xlabel=f"{label} ({unit})",
        ylabel="",
        run_label=run_label,
    )
    return out
