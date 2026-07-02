"""Render per-pool KV-cache occupancy from the Rust `kv_occupancy_series.json`.

One figure (`kv_occupancy.png`). The focus metric is `active` (resident/current
KV) drawn per shard: each worker's own series is a faint **shadow** step line, and
the across-shard **min / max / mean** stand out on top — a bold mean line, a shaded
min–max band, and thin min/max edges. A wide band means shard imbalance. `projected`
(the admission gate's future estimate) and `promised` (reserved-not-resident) are
drawn as pool-mean context lines (dashed / dotted). When `run_meta` carries a
capacity the y-axis is percent of capacity with a dotted 100% line; otherwise it
falls back to raw tokens. `projected` is intentionally allowed above 100% — that's
the signal the admitted horizon outruns the pool.

Reads only the payload — no parquet. `render` returns render *jobs*; `__main__`
runs them.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable

import matplotlib.colors as mcolors

from common.figure import add_legend, corner_box, finalize, fmt_count, fmt_value, new_axes
from common.style import CURVE, MARKER

# Color encodes the METRIC, not the shard: `active` is one hue (its shards are shades
# of it), `projected` and `promised` each their own hue. So the eye reads a metric by
# color and a shard by lightness within the active hue.
_ACTIVE_HUE = CURVE           # blue — resident/current KV
_PROJECTED_COLOR = "#C1121F"  # dark red — admission-gate future estimate
_PROMISED_COLOR = "#6D28D9"   # violet — reserved-not-resident


def _tint(color: str, amount: float) -> tuple[float, float, float]:
    """Blend `color` toward white by `amount` (0 = color, 1 = white) — lighter shades
    of one hue for the per-shard lines."""
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def _shade(color: str, amount: float) -> tuple[float, float, float]:
    """Blend `color` toward black by `amount` — the darker, standout mean line."""
    r, g, b = mcolors.to_rgb(color)
    return (r * (1 - amount), g * (1 - amount), b * (1 - amount))


def _shard_tints(n: int) -> list[tuple[float, float, float]]:
    """`n` lightness shades of the active hue (light → mid), one per shard."""
    if n <= 1:
        return [_tint(_ACTIVE_HUE, 0.35)]
    return [_tint(_ACTIVE_HUE, 0.60 - 0.45 * (j / (n - 1))) for j in range(n)]


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
        partial(_render, payload, plot_output_path(log_dir, "kv_occupancy.png"),
                run_label=run_label),
    ]


def _edges_s(t_start: list[float], t_end: list[float]) -> list[float]:
    """Bin edges in seconds: contiguous bins → n+1 edges (one flat `stairs` step
    per bin)."""
    return [t / 1000.0 for t in (*t_start, t_end[-1])]


def _has_capacity(payload: dict) -> bool:
    """True when every series carries a positive capacity — the precondition for a
    percent axis (mixing percent and raw tokens on one axis would mislabel)."""
    series = payload.get("series", [])
    return bool(series) and all((s.get("capacity_tokens") or 0) > 0 for s in series)


def _render(payload: dict, out_path: Path, *, run_label: str = "") -> Path:
    """Draw the per-shard KV occupancy plot (shadows + standout min/max/mean)."""
    fig, ax = new_axes(figsize=(9.5, 5.0))
    edges = _edges_s(payload["t_start_ms"], payload["t_end_ms"])
    starts = edges[:-1]  # left edge per bin, for the stepped min–max band fill
    pct_mode = _has_capacity(payload)

    def scale(tokens: list[float], cap: float | None) -> list[float]:
        # Percent of capacity when known, else raw tokens.
        if pct_mode and cap:
            return [t / cap * 100.0 for t in tokens]
        return list(tokens)

    peak_active_mean = 0.0
    peak_active_max = 0.0
    peak_projected = 0.0
    top = 0.0
    for s in payload["series"]:
        cap = s.get("capacity_tokens")
        label = s.get("label", s.get("key", ""))
        active = s.get("active") or {}
        a_mean = scale(active.get("mean") or [], cap)
        a_min = scale(active.get("min") or [], cap)
        a_max = scale(active.get("max") or [], cap)
        if not a_mean:
            continue
        workers = s.get("workers") or []
        multi = len(workers) > 1
        pfx = f"{label} · " if len(payload["series"]) > 1 else ""

        # (1) Per-shard active lines — shades of ONE hue (the active hue), light→mid,
        # so a shard reads by lightness while staying clearly "active".
        tints = _shard_tints(len(workers))
        for w, tint in zip(workers, tints):
            wa = scale(w.get("active_tokens") or [], cap)
            if not wa:
                continue
            ax.stairs(wa, edges, color=tint, linewidth=1.1, baseline=None,
                      label=f"{pfx}active · w{w.get('worker_id')}")

        # (2) Standout across-shard aggregate, same hue: light-blue min–max band +
        # bold dark-blue mean.
        if multi:
            ax.fill_between(starts, a_min, a_max, step="post", color=_tint(_ACTIVE_HUE, 0.55),
                            alpha=0.45, linewidth=0, label=f"{pfx}active · min–max")
        ax.stairs(a_mean, edges, color=_shade(_ACTIVE_HUE, 0.25), linewidth=2.6, baseline=None,
                  label=f"{pfx}active" + " · mean" if multi else f"{pfx}active")

        # (3) Context: projected (future estimate) + promised (reserved), pool-mean,
        # each its own hue so the three metrics read apart by color.
        p_mean = scale((s.get("projected") or {}).get("mean") or [], cap)
        pr_mean = scale((s.get("promised") or {}).get("mean") or [], cap)
        if p_mean:
            ax.stairs(p_mean, edges, color=_PROJECTED_COLOR, linewidth=1.6, linestyle="--",
                      alpha=0.9, baseline=None, label=f"{pfx}projected")
            peak_projected = max(peak_projected, max(p_mean))
        if any(v > 0 for v in pr_mean):
            ax.stairs(pr_mean, edges, color=_PROMISED_COLOR, linewidth=1.4, linestyle=":",
                      alpha=0.85, baseline=None, label=f"{pfx}promised")

        peak_active_mean = max(peak_active_mean, max(a_mean))
        peak_active_max = max(peak_active_max, max(a_max))
        top = max(top, max(a_max), max(p_mean or [0.0]))

    if pct_mode:
        # Dotted full-capacity reference; headroom above 100% so an over-capacity
        # projected curve stays visible.
        ax.axhline(100.0, color=MARKER, linestyle=":", linewidth=1.0, alpha=0.7)
        ax.set_ylim(bottom=0.0, top=max(105.0, top * 1.05))
        ax.set_ylabel("KV occupancy (% of capacity)")
        summary = [
            f"peak active (mean) = {fmt_value(peak_active_mean, '%')}",
            f"peak active (worst shard) = {fmt_value(peak_active_max, '%')}",
            f"peak projected = {fmt_value(peak_projected, '%')}",
        ]
    else:
        ax.set_ylim(bottom=0.0, top=max(1.0, top * 1.05))
        ax.set_ylabel("KV occupancy (tokens)")
        summary = [
            f"peak active (mean) = {fmt_count(peak_active_mean)} tok",
            f"peak active (worst shard) = {fmt_count(peak_active_max)} tok",
            f"peak projected = {fmt_count(peak_projected)} tok",
        ]

    ax.set_xlim(left=edges[0], right=edges[-1])
    corner_box(ax, summary, loc="upper left")
    add_legend(ax)
    finalize(
        fig, ax, out_path,
        title="KV-cache occupancy (per shard)", xlabel="sim time (s)",
        ylabel=ax.get_ylabel(), run_label=run_label,
    )
    return out_path
