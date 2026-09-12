"""Render measured-vs-sim scheduler workload by iteration id or elapsed time."""

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
        "Actual iteration cycle: vLLM vs ServingStudioSim",
        "iteration cycle (ms)",
        "alignment_iteration_time_by_iteration.png",
    ),
)

_DECODE_TIME_PLOT = (
    "decode_batch_size",
    "Decode batch size over elapsed time",
    "decode requests",
    "alignment_decode_batch_size_over_elapsed_time.png",
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
    iteration_jobs = [
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
    key, title, ylabel, filename = _DECODE_TIME_PLOT
    # Rust independently normalizes each side's time_ms to its first observed
    # iteration. Keep that contract here; absolute host/GPU clocks are unrelated.
    elapsed_time_job = partial(
        _render_series,
        measured,
        simulated,
        key,
        title,
        ylabel,
        plot_output_path(log_dir, filename),
        x_key="time_ms",
        x_scale=1e-3,
        xlabel="elapsed time (s)",
        run_label=log_dir.name,
    )
    return [*iteration_jobs, elapsed_time_job]


def _render_series(
    measured: dict,
    simulated: dict,
    key: str,
    title: str,
    ylabel: str,
    out_path: Path,
    *,
    run_label: str,
    x_key: str = "iteration_id",
    x_scale: float = 1.0,
    xlabel: str = "iteration id",
) -> Path:
    measured_points = [
        (x_value * x_scale, value)
        for x_value, value in zip(measured[x_key], measured[key], strict=True)
        if x_value is not None and value is not None
    ]
    simulated_points = [
        (x_value * x_scale, value)
        for x_value, value in zip(simulated[x_key], simulated[key], strict=True)
        if x_value is not None and value is not None
    ]
    measured_x, measured_values = zip(*measured_points, strict=True)
    simulated_x, simulated_values = zip(*simulated_points, strict=True)
    measured_label = "vLLM GPU cycle" if key == "iteration_cycle_ms" else "vLLM measured"
    simulated_label = (
        "ServingStudioSim actual cycle" if key == "iteration_cycle_ms" else "ServingStudioSim"
    )

    fig, ax = new_axes(figsize=(10.0, 4.8))
    ax.step(
        measured_x,
        measured_values,
        where="post",
        label=measured_label,
        color=CURVE,
        linewidth=1.8,
    )
    ax.step(
        simulated_x,
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
        xlabel=xlabel,
        ylabel=ylabel,
        run_label=run_label,
    )
    return out_path
