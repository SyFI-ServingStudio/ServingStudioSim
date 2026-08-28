"""Render completion throughput and raw measured-vs-sim latency CDFs."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path

from common.cdf_plot import render_cdf_comparison
from common.figure import add_legend, corner_box, finalize, fmt_value, new_axes
from common.layout import load_payload, plot_output_path, resolve_artifact
from common.style import ACCENT, CURVE

PAYLOAD = "alignment_e2e_series.json"


def render(log_dir: Path) -> list[Callable[[], Path]]:
    if not resolve_artifact(log_dir, PAYLOAD).is_file():
        return []
    payload = load_payload(log_dir, PAYLOAD)
    throughput = payload.get("throughput") or {}
    throughput_summary = payload.get("throughput_summary") or {}
    if not throughput.get("t_start_ms"):
        reason = payload.get("meta", {}).get("reason", "no E2E throughput bins")
        print(f"[alignment_e2e] nothing to render: {reason}")
        return []
    jobs: list[Callable[[], Path]] = [
        partial(
            _render_throughput,
            throughput,
            plot_output_path(log_dir, "alignment_e2e_completion_throughput.png"),
            summary=throughput_summary,
            run_label=log_dir.name,
        )
    ]
    for comparison in payload.get("latency_cdf_comparisons") or []:
        jobs.append(
            partial(
                render_cdf_comparison,
                comparison,
                plot_output_path(
                    log_dir,
                    f"alignment_{comparison['key']}_cdf_comparison.png",
                ),
                run_label=log_dir.name,
            )
        )
    return jobs


def _render_throughput(
    series: dict,
    out_path: Path,
    *,
    summary: dict,
    run_label: str,
) -> Path:
    starts = series["t_start_ms"]
    ends = series["t_end_ms"]
    edges_s = [value / 1000.0 for value in (*starts, ends[-1])]
    fig, ax = new_axes(figsize=(10.0, 4.8))
    ax.stairs(
        series["measured_output_tps"],
        edges_s,
        label="Measured",
        color=CURVE,
        linewidth=2.0,
        baseline=None,
    )
    ax.stairs(
        series["simulated_output_tps"],
        edges_s,
        label="Simulated",
        color=ACCENT,
        linewidth=2.0,
        linestyle="--",
        baseline=None,
    )
    ax.set_xlim(edges_s[0], edges_s[-1])
    ax.set_ylim(bottom=0.0)
    summary_fields = (
        ("measured_client_completion_tps", "client avg"),
        ("simulated_completion_tps", "sim avg"),
    )
    summary_lines = [
        f"{label} = {fmt_value(summary[key], 'tok/s')}"
        for key, label in summary_fields
        if key in summary
    ]
    if summary_lines:
        corner_box(ax, summary_lines, loc="upper right")
    add_legend(ax)
    finalize(
        fig,
        ax,
        out_path,
        title="Output-token completion throughput",
        xlabel="time from run start (s)",
        ylabel="completed output tokens / s",
        run_label=run_label,
    )
    return out_path
