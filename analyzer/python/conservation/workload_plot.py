"""Render the workload-conservation checks from `workload_conservation_checks.json`.

One figure: a per-check Δ% bar (relative deviation of cost_log *actual* from the
request_slo *expected*). The checks span wildly different magnitudes (prefill
tokens ~1e7, causal attn work ~1e10, decode KV ~1e12), so the relative Δ% is the
only sensible shared y-scale. The tolerance band is shaded and the warn lines are
drawn; bars are colored by status (OK / WARN / FAIL). When the run conserves, all
bars sit flat inside the green band — a one-glance pass; a real accounting bug
makes its bar pop out and change color.

Reads only the payload — no parquet. Reuses `common.figure` / `common.style`
(local literals for the three status colors only, as the scatter subject does for
its series colors).
"""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from typing import Callable

from common.figure import corner_box, finalize, new_axes

# Status → bar color. Local literals (not shared style) — these three semantic
# colors are specific to this validation subject.
_STATUS_COLOR = {"OK": "#54A24B", "WARN": "#E6A23C", "FAIL": "#D62728"}


def render(log_dir: Path) -> list[Callable[[], Path]]:
    from common.layout import load_payload, plot_output_path

    payload = load_payload(log_dir, "workload_conservation_checks.json")
    checks = payload.get("checks") or []
    if not payload.get("meta", {}).get("available", False) or not checks:
        reason = payload.get("meta", {}).get("reason", "no checks in payload")
        print(f"[workload_plot] nothing to render: {reason}")
        return []
    run_label = Path(payload.get("meta", {}).get("log_dir", str(log_dir))).name
    return [
        partial(_render, payload, plot_output_path(log_dir, "workload_conservation.png"),
                run_label=run_label),
    ]


def _render(payload: dict, out_path: Path, *, run_label: str = "") -> Path:
    checks = payload["checks"]
    meta = payload.get("meta", {})
    tol = float(meta.get("tolerance_pct", 0.01))
    warn = float(meta.get("warn_pct", 5.0))

    names = [c["name"] for c in checks]
    statuses = [c.get("status", "FAIL") for c in checks]
    # Δ% may be null (expected == 0 with a nonzero delta ⇒ undefined): treat as an
    # off-scale failure spike so it is unmistakable.
    finite = [abs(c["delta_pct"]) for c in checks if c.get("delta_pct") is not None]
    ymax = max(warn * 1.3, (max(finite) * 1.15 if finite else 0.0), tol * 5.0, 0.01)

    heights = []
    for c in checks:
        dp = c.get("delta_pct")
        if dp is None:
            heights.append(math.copysign(ymax, c.get("delta", 1.0) or 1.0))
        else:
            heights.append(max(-ymax, min(ymax, dp)))  # clamp into view
    colors = [_STATUS_COLOR.get(s, _STATUS_COLOR["FAIL"]) for s in statuses]

    fig, ax = new_axes(figsize=(9.0, 5.0))
    x = list(range(len(checks)))
    ax.bar(x, heights, color=colors, width=0.62, zorder=3)
    # Tolerance band (OK region) + warn bounds.
    ax.axhspan(-tol, tol, color="#54A24B", alpha=0.15, zorder=1, label=f"±{tol:g}% OK band")
    for sign in (-1, 1):
        ax.axhline(sign * warn, color="#E6A23C", linestyle="--", linewidth=1.0, alpha=0.8, zorder=2)
    ax.axhline(0.0, color="#6B7280", linewidth=0.8, zorder=2)
    ax.set_ylim(-ymax * 1.08, ymax * 1.08)

    # Annotate clamped / off-scale bars with their true Δ% (or n/a).
    for xi, c, h in zip(x, checks, heights):
        dp = c.get("delta_pct")
        clamped = dp is None or abs(dp) > ymax
        if clamped:
            txt = "n/a%" if dp is None else f"{dp:+.2g}%"
            va = "bottom" if h >= 0 else "top"
            ax.text(xi, h, txt, ha="center", va=va, fontsize=8, color="#111827")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)

    n_ok = sum(s == "OK" for s in statuses)
    summary = "ALL CONSERVED" if meta.get("all_ok") else "MISMATCH — see report"
    corner_box(ax, [summary, f"{n_ok}/{len(checks)} checks OK"], loc="upper left")

    finalize(
        fig, ax, out_path,
        title="Workload conservation (cost_log actual vs request_slo expected)",
        xlabel="", ylabel="Δ% = (actual − expected) / expected",
        run_label=run_label, grid=True,
    )
    return out_path
