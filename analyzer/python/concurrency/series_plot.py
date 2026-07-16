"""Render bounded in-flight request concurrency from analyzer JSON."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable

from common.figure import corner_box, finalize, fmt_value, new_axes
from common.style import CURVE


def render(log_dir: Path) -> list[Callable[[], Path]]:
    from common.layout import load_payload, plot_output_path

    payload = load_payload(log_dir, "concurrency_series.json")
    if not payload.get("t_ms") or not payload.get("active"):
        reason = payload.get("meta", {}).get("reason", "no concurrency series")
        print(f"[concurrency_plot] nothing to render: {reason}")
        return []
    run_label = Path(payload.get("meta", {}).get("log_dir", str(log_dir))).name
    return [
        partial(
            _render,
            payload,
            plot_output_path(log_dir, "concurrency_series.png"),
            run_label=run_label,
        )
    ]


def _render(payload: dict, out_path: Path, *, run_label: str) -> Path:
    t_s = [value / 1000.0 for value in payload["t_ms"]]
    active = payload["active"]
    peak = payload["peak"]
    mean = sum(active) / len(active)

    fig, ax = new_axes(figsize=(9.0, 4.5))
    ax.plot(t_s, active, color=CURVE, linewidth=2.0)
    ax.fill_between(t_s, active, color=CURVE, alpha=0.16)
    ax.set_xlim(left=0.0, right=t_s[-1])
    ax.set_ylim(bottom=0.0, top=max(1.0, peak * 1.08))
    corner_box(
        ax,
        [
            f"mean = {fmt_value(mean, 'requests')}",
            f"peak = {fmt_value(peak, 'requests')}",
        ],
        loc="upper left",
    )
    finalize(
        fig,
        ax,
        out_path,
        title="In-flight request concurrency",
        xlabel="sim time (s)",
        ylabel="active requests",
        run_label=run_label,
    )
    return out_path
