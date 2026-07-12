"""Sustained per-launch duration trend + telemetry capture for ``measure``.

The op-agnostic core lifted from ``gemm_10s_cupti_trend.py`` /
``gemm_10s_cupti_telemetry.py``, minus the experiment-specific vLLM band and
prior-DB reference overlays. ``run_measure_capture`` is the single orchestrator
the ``Timer.cupti`` hook calls: it drives a sustained CUPTI window over an
arbitrary kernel callable, samples NVML telemetry concurrently, writes CSV /
``summary.json`` / two PNGs into the context's output dir, and returns a
representative kernel time so the runner still completes normally.

``torch`` and the CUPTI extension are imported lazily inside
``run_measure_capture`` so this module stays importable on a CPU-only host;
``numpy`` (CPU-only) is used at module scope for the runtime summary.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from profiling.profilers.measure_context import MeasureContext

_MEASUREMENT = "one continuous CUPTI activity window; per-launch kernel-only duration"
_ROLLING_WINDOW = 201

_RUNTIME_CSV_FIELDS = ["sample_index", "start_s", "duration_ms", "kernel_count", "kernel_names"]


def run_measure_capture(fn: Callable[[], object], context: MeasureContext) -> float:
    """Run one sustained trend+telemetry capture and emit artifacts.

    Returns the per-launch median duration (ms) so the intercepted
    ``Timer.cupti`` call hands the runner a valid time and finishes normally.
    """

    import torch

    from profiling.profilers import telemetry as telemetry_mod
    from profiling.profilers import trend_plot
    from profiling.profilers.cupti_kernel_profiler import CuptiKernelProfiler

    device = torch.device("cuda:0")
    profiler = CuptiKernelProfiler(device=device)

    output_dir = Path(context.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sampler = None
    capture_hook = None
    interval_s = None
    if context.telemetry:
        interval_s = 1.0 / context.telemetry_hz
        sampler = telemetry_mod.NvmlTelemetrySampler(interval_s)
        capture_hook = sampler

    series, capture_metadata = profiler.profile_series_for_duration(
        fn,
        duration_s=context.duration_s,
        clear_l2_before_run=context.clear_l2,
        clear_l2_between_launches=context.clear_l2,
        capture_hook=capture_hook,
    )

    metadata = {
        **capture_metadata,
        "label": context.label,
        "shape": context.shape,
        "telemetry_interval_s": interval_s,
        "telemetry_sample_count": len(sampler.samples) if sampler is not None else 0,
    }
    summary = summarize(series, metadata)

    artifacts: list[Path] = []
    _write_series_csv(output_dir / "runtimes.csv", series)
    artifacts.append(output_dir / "runtimes.csv")

    aligned: list[dict[str, object]] | None = None
    if sampler is not None and len(sampler.samples) >= 2:
        aligned = telemetry_mod.align_telemetry(series, sampler.samples)
        summary["telemetry"] = telemetry_mod.summarize_telemetry(aligned)
        _write_aligned_csv(output_dir / "telemetry.csv", aligned)
        artifacts.append(output_dir / "telemetry.csv")

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifacts.append(output_dir / "summary.json")

    trend_plot.plot_runtime_trend(output_dir / "runtime_trend.png", series, summary)
    artifacts.append(output_dir / "runtime_trend.png")
    if aligned is not None:
        trend_plot.plot_runtime_telemetry(
            output_dir / "runtime_telemetry.png", series, aligned, summary
        )
        artifacts.append(output_dir / "runtime_telemetry.png")

    median_ms = float(summary["runtime_ms"]["median"])  # type: ignore[index]
    context.time_ms = median_ms
    context.artifacts = [str(path) for path in artifacts]
    return median_ms


def summarize(
    samples: list[dict[str, object]],
    metadata: dict[str, object],
) -> dict[str, object]:
    """Runtime statistics over the per-launch series (op- and experiment-agnostic)."""

    starts = np.asarray([sample["start_s"] for sample in samples], dtype=float)
    durations = np.asarray([sample["duration_ms"] for sample in samples], dtype=float)

    first_mask = starts < 1.0
    last_mask = starts >= max(float(starts[-1]) - 1.0, 0.0)
    first_1s_mean = float(np.mean(durations[first_mask])) if first_mask.any() else float(np.mean(durations))
    last_1s_mean = float(np.mean(durations[last_mask])) if last_mask.any() else float(np.mean(durations))
    slope_ms_per_s = (
        float(np.polyfit(starts, durations, deg=1)[0]) if np.std(starts) > 0 else 0.0
    )

    one_second_bins = []
    for second in range(int(np.ceil(starts[-1])) if starts[-1] > 0 else 1):
        mask = (starts >= second) & (starts < second + 1)
        if not np.any(mask):
            continue
        values = durations[mask]
        one_second_bins.append(
            {
                "second": second,
                "count": int(values.size),
                "mean_ms": float(np.mean(values)),
                "median_ms": float(np.median(values)),
                "min_ms": float(np.min(values)),
                "max_ms": float(np.max(values)),
            }
        )

    return {
        "schema_version": 1,
        "measurement": _MEASUREMENT,
        "label": metadata.get("label"),
        "shape": metadata.get("shape"),
        "metadata": metadata,
        "runtime_ms": {
            "mean": float(np.mean(durations)),
            "median": float(np.median(durations)),
            "min": float(np.min(durations)),
            "max": float(np.max(durations)),
            "p10": float(np.percentile(durations, 10)),
            "p90": float(np.percentile(durations, 90)),
            "p99": float(np.percentile(durations, 99)),
            "first_1s_mean": first_1s_mean,
            "last_1s_mean": last_1s_mean,
            "linear_slope_ms_per_s": slope_ms_per_s,
        },
        "one_second_bins": one_second_bins,
    }


def rolling_median(values: np.ndarray, window: int = _ROLLING_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """Centered rolling median; window is clamped odd and to len(values)."""

    resolved = min(window, len(values))
    if resolved % 2 == 0:
        resolved -= 1
    resolved = max(resolved, 1)
    windows = np.lib.stride_tricks.sliding_window_view(values, resolved)
    centers = np.arange(resolved // 2, resolved // 2 + len(windows))
    return centers, np.median(windows, axis=1)


def _write_series_csv(path: Path, samples: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_RUNTIME_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(samples)


def _write_aligned_csv(path: Path, aligned: list[dict[str, object]]) -> None:
    fieldnames = list(aligned[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aligned)
