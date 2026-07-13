"""Render measured-vs-sim scheduler workload by recorded iteration id."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path

from common.figure import add_legend, corner_box, finalize, fmt_count, new_axes
from common.layout import load_payload, plot_output_path, resolve_artifact
from common.style import ACCENT, CURVE

PAYLOAD = "alignment_workload_series.json"

_PLOTS = (
    (
        "prefill_tokens",
        "Prefill tokens per scheduled iteration",
        "prefill tokens",
        "alignment_prefill_tokens_over_time.png",
    ),
    (
        "decode_batch_size",
        "Decode batch size per scheduled iteration",
        "decode requests",
        "alignment_decode_batch_size_over_time.png",
    ),
    (
        "scheduled_kv_tokens",
        "Scheduled KV workload per iteration",
        "KV tokens touched",
        "alignment_scheduled_kv_workload_over_time.png",
    ),
    (
        "iteration_cycle_ms",
        "Actual iteration cycle: vLLM vs VibeSim",
        "iteration cycle (ms)",
        "alignment_iteration_time_by_iteration.png",
    ),
)


def render(log_dir: Path) -> list[Callable[[], Path]]:
    if not resolve_artifact(log_dir, PAYLOAD).is_file():
        return []
    payload = load_payload(log_dir, PAYLOAD)
    measured = payload.get("measured") or {}
    simulated = payload.get("simulated") or {}
    if not measured.get("iteration_id") or not simulated.get("iteration_id"):
        reason = payload.get("meta", {}).get("reason", "no workload iteration series")
        print(f"[alignment_workload] nothing to render: {reason}")
        return []
    return [
        partial(
            _render_series,
            measured,
            simulated,
            key,
            title,
            ylabel,
            plot_output_path(log_dir, filename),
            run_label=log_dir.name,
        )
        for key, title, ylabel, filename in _PLOTS
    ]


def _render_series(
    measured: dict,
    simulated: dict,
    key: str,
    title: str,
    ylabel: str,
    out_path: Path,
    *,
    run_label: str,
) -> Path:
    measured_points = [
        (iteration_id, value)
        for iteration_id, value in zip(measured["iteration_id"], measured[key], strict=True)
        if value is not None
    ]
    simulated_points = [
        (iteration_id, value)
        for iteration_id, value in zip(simulated["iteration_id"], simulated[key], strict=True)
        if value is not None
    ]
    measured_iteration_ids, measured_values = zip(*measured_points, strict=True)
    simulated_iteration_ids, simulated_values = zip(*simulated_points, strict=True)
    measured_label = "vLLM GPU cycle" if key == "iteration_cycle_ms" else "vLLM measured"
    simulated_label = (
        "VibeSim actual cycle" if key == "iteration_cycle_ms" else "VibeSim"
    )

    fig, ax = new_axes(figsize=(10.0, 4.8))
    ax.step(
        measured_iteration_ids,
        measured_values,
        where="post",
        label=measured_label,
        color=CURVE,
        linewidth=1.8,
    )
    ax.step(
        simulated_iteration_ids,
        simulated_values,
        where="post",
        label=simulated_label,
        color=ACCENT,
        linewidth=1.8,
        linestyle="--",
    )
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    corner_box(
        ax,
        [
            f"measured iterations: {fmt_count(len(measured_values))}",
            f"simulated iterations: {fmt_count(len(simulated_values))}",
        ],
        loc="upper right",
    )
    add_legend(ax, loc="upper left")
    finalize(
        fig,
        ax,
        out_path,
        title=title,
        xlabel="iteration id",
        ylabel=ylabel,
        run_label=run_label,
    )
    return out_path
