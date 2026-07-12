"""Matplotlib rendering for the ``measure`` trend diagnostic.

Two figures, lifted from the experiment-local diagnostics with the vLLM-band and
prior-DB reference overlays removed:

- ``plot_runtime_trend``     — per-launch CUPTI duration scatter + rolling median.
- ``plot_runtime_telemetry`` — 3 panels: runtime / power+SM-clock / util+temp,
  with the telemetry traces shifted back by the measured runtime lag.

matplotlib is imported lazily (Agg backend) so importing this module costs
nothing on a CPU-only host and needs no display.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

_SCATTER_COLOR = "#4C78A8"
_MEDIAN_COLOR = "#1F4E79"
_POWER_COLOR = "#E15759"
_CLOCK_COLOR = "#4E79A7"
_UTIL_COLOR = "#59A14F"
_TEMP_COLOR = "#F28E2B"


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _title(summary: dict[str, Any]) -> str:
    label = summary.get("label") or "kernel"
    shape = summary.get("shape")
    duration = _duration_s(summary)
    suffix = f" {shape}" if shape else ""
    return f"{label}{suffix} — per-launch CUPTI duration, one continuous {duration:g}s capture"


def _duration_s(summary: dict[str, Any]) -> float:
    metadata = summary.get("metadata") or {}
    assert isinstance(metadata, dict)
    return float(metadata.get("requested_duration_s", 0.0) or 0.0)


def plot_runtime_trend(
    path: Path,
    samples: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    from profiling.profilers.trend import rolling_median

    plt = _pyplot()
    starts = np.asarray([sample["start_s"] for sample in samples], dtype=float)
    durations = np.asarray([sample["duration_ms"] for sample in samples], dtype=float)

    figure, axis = plt.subplots(figsize=(15, 8))
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    figure.suptitle(_title(summary), fontsize=16, fontweight="bold")

    axis.scatter(starts, durations, s=4, alpha=0.16, color=_SCATTER_COLOR, label="every launch")
    centers, medians = rolling_median(durations)
    axis.plot(
        starts[centers],
        medians,
        color=_MEDIAN_COLOR,
        linewidth=2.2,
        label="rolling median",
    )

    runtime = summary["runtime_ms"]
    assert isinstance(runtime, dict)
    axis.text(
        0.012,
        0.025,
        "\n".join(
            [
                f"launches: {len(samples):,}",
                f"mean / median: {float(runtime['mean']):.6f} / {float(runtime['median']):.6f} ms",
                f"first 1 s / last 1 s mean: "
                f"{float(runtime['first_1s_mean']):.6f} / {float(runtime['last_1s_mean']):.6f} ms",
                f"linear slope: {float(runtime['linear_slope_ms_per_s']):+.6f} ms/s",
            ]
        ),
        transform=axis.transAxes,
        fontsize=10,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "alpha": 0.88},
    )
    axis.set_xlabel("GPU timeline since first captured launch (s)")
    axis.set_ylabel("CUPTI kernel duration per launch (ms)")
    axis.set_xlim(0.0, max(float(starts[-1]), _duration_s(summary)))
    axis.grid(True, alpha=0.22)
    axis.legend(loc="upper left", fontsize=9)

    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(path, dpi=160, facecolor="white")
    plt.close(figure)


def _lag_s(summary: dict[str, Any], metric_key: str) -> float:
    """Measured steady-state runtime lag for a telemetry metric, or 0.0."""

    telemetry = summary.get("telemetry")
    if not isinstance(telemetry, dict):
        return 0.0
    metrics = telemetry.get("metrics")
    if not isinstance(metrics, dict):
        return 0.0
    metric = metrics.get(metric_key)
    if not isinstance(metric, dict):
        return 0.0
    lag = metric.get("steady_state_strongest_lagged_runtime_correlation")
    if not isinstance(lag, dict):
        return 0.0
    return float(lag.get("telemetry_lag_s", 0.0) or 0.0)


def plot_runtime_telemetry(
    path: Path,
    samples: list[dict[str, Any]],
    aligned: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    from profiling.profilers.trend import rolling_median

    plt = _pyplot()
    runtime_times = np.asarray([sample["start_s"] for sample in samples], dtype=float)
    runtime_values = np.asarray([sample["duration_ms"] for sample in samples], dtype=float)
    telemetry_times = np.asarray([row["time_s"] for row in aligned], dtype=float)

    power_lag_s = _lag_s(summary, "power_w")
    clock_lag_s = _lag_s(summary, "sm_clock_mhz")

    figure, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
    figure.patch.set_facecolor("white")
    for axis in axes:
        axis.set_facecolor("white")
        axis.grid(True, alpha=0.2)

    runtime_axis = axes[0]
    runtime_axis.scatter(runtime_times, runtime_values, s=3, alpha=0.13, color=_SCATTER_COLOR)
    centers, medians = rolling_median(runtime_values)
    runtime_axis.plot(runtime_times[centers], medians, color=_MEDIAN_COLOR, linewidth=2.0)
    runtime_axis.set_ylabel("kernel duration (ms)")

    power_axis = axes[1]
    power = _column(aligned, "power_w")
    sm_clock = _column(aligned, "sm_clock_mhz")
    power_axis.plot(
        telemetry_times - power_lag_s,
        power,
        color=_POWER_COLOR,
        linewidth=1.6,
        label=f"power (shifted -{power_lag_s:.2f} s)",
    )
    power_limits = [
        float(row["power_limit_w"]) for row in aligned if row.get("power_limit_w") is not None
    ]
    if power_limits:
        power_axis.axhline(
            float(np.median(power_limits)),
            color=_POWER_COLOR,
            linestyle=":",
            alpha=0.8,
            label="power limit",
        )
    power_axis.set_ylabel("Power (W)", color=_POWER_COLOR)
    clock_axis = power_axis.twinx()
    clock_axis.plot(
        telemetry_times - clock_lag_s,
        sm_clock,
        color=_CLOCK_COLOR,
        linewidth=1.5,
        label=f"SM clock (shifted -{clock_lag_s:.2f} s)",
    )
    clock_axis.set_ylabel("SM clock (MHz)", color=_CLOCK_COLOR)
    lines = power_axis.get_lines() + clock_axis.get_lines()
    power_axis.legend(lines, [line.get_label() for line in lines], loc="upper right")

    state_axis = axes[2]
    state_axis.plot(
        telemetry_times, _column(aligned, "gpu_util_pct"), color=_UTIL_COLOR, linewidth=1.5,
        label="GPU utilization",
    )
    state_axis.set_ylabel("GPU utilization (%)", color=_UTIL_COLOR)
    temperature_axis = state_axis.twinx()
    temperature_axis.plot(
        telemetry_times, _column(aligned, "temperature_c"), color=_TEMP_COLOR, linewidth=1.5,
        label="temperature",
    )
    temperature_axis.set_ylabel("Temperature (°C)", color=_TEMP_COLOR)
    lines = state_axis.get_lines() + temperature_axis.get_lines()
    state_axis.legend(lines, [line.get_label() for line in lines], loc="upper right")
    state_axis.set_xlabel("Time since capture start (s)")

    clock_corr = _corr(summary, "sm_clock_mhz")
    subtitle = "" if clock_corr is None else f"; corr(runtime, shifted SM clock)={clock_corr:+.3f}"
    figure.suptitle(
        f"{summary.get('label') or 'kernel'} — CUPTI runtime with synchronized NVML telemetry"
        f"\ntelemetry shifted back by measured lag{subtitle}",
        fontsize=16,
        fontweight="bold",
    )
    axes[-1].set_xlim(0.0, max(float(runtime_times[-1]), _duration_s(summary)))
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(path, dpi=160, facecolor="white")
    plt.close(figure)


def _column(aligned: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray(
        [np.nan if row.get(key) is None else float(row[key]) for row in aligned],  # type: ignore[arg-type]
        dtype=float,
    )


def _corr(summary: dict[str, Any], metric_key: str) -> float | None:
    telemetry = summary.get("telemetry")
    if not isinstance(telemetry, dict):
        return None
    metrics = telemetry.get("metrics")
    if not isinstance(metrics, dict):
        return None
    metric = metrics.get(metric_key)
    if not isinstance(metric, dict):
        return None
    lag = metric.get("steady_state_strongest_lagged_runtime_correlation")
    if not isinstance(lag, dict):
        return None
    value = lag.get("correlation")
    return None if value is None else float(value)
