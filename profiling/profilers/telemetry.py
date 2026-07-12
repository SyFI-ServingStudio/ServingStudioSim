"""NVML GPU telemetry sampling + alignment for the ``measure`` trend diagnostic.

Lifted and generalized from the experiment-local ``gemm_10s_cupti_telemetry.py``.
An ``NvmlTelemetrySampler`` polls the running GPU (power / SM-clock / mem-clock /
util / temp / pstate / throttle) on a background host thread while a single CUPTI
capture window runs; ``align_telemetry`` then bins per-launch runtimes onto the
telemetry timeline, and ``summarize_telemetry`` reports per-metric ranges and
(lagged) correlations against runtime.

Telemetry is best-effort: if ``pynvml`` or a physical NVML handle is unavailable
the sampler stays ``enabled=False`` and collects nothing, so the runtime trend is
still emitted. ``torch`` / ``pynvml`` are imported lazily so this module is safe
to import on a CPU-only host; ``numpy`` (a CPU-only core dependency) is imported
at module scope for the analysis helpers.
"""

from __future__ import annotations

import threading
import time
from statistics import fmean
from typing import Any

import numpy as np


def _safe_nvml(pynvml: Any, call: Any, *args: Any) -> Any | None:
    try:
        return call(*args)
    except pynvml.NVMLError:
        return None


class NvmlTelemetrySampler:
    """Context manager that samples the current CUDA device on a host thread.

    ``__enter__`` resolves the physical NVML handle for Torch's *current* device
    (reusing ``energy._nvml_handle_for_cuda_device`` so it survives a
    ``CUDA_VISIBLE_DEVICES`` remap), stamps a ``perf_counter`` origin, and starts
    polling; ``__exit__`` stops and joins. Sample ``time_s`` is relative to the
    origin, which the caller enters just before the formal CUPTI capture so the
    telemetry and launch timelines start together.
    """

    def __init__(self, interval_s: float) -> None:
        if interval_s <= 0:
            raise ValueError("telemetry interval must be positive")
        self.interval_s = interval_s
        self.samples: list[dict[str, Any]] = []
        self.enabled = False
        self.error: BaseException | None = None
        self._pynvml: Any | None = None
        self._handle: object | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._origin_s: float | None = None

    def __enter__(self) -> NvmlTelemetrySampler:
        try:
            import pynvml
            import torch

            from profiling.profilers.energy import _nvml_handle_for_cuda_device
        except ImportError as exc:  # pragma: no cover - depends on runtime env
            self.error = exc
            return self

        try:
            pynvml.nvmlInit()
            device_idx = torch.cuda.current_device()
            handle = _nvml_handle_for_cuda_device(pynvml, torch, device_idx)
        except Exception as exc:  # noqa: BLE001 - telemetry is best-effort
            self.error = exc
            return self

        if handle is None:
            return self

        self._pynvml = pynvml
        self._handle = handle
        self.enabled = True
        self._origin_s = time.perf_counter()
        self._thread = threading.Thread(
            target=self._run,
            name="vibesim-nvml-telemetry",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        del exc_info
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.5, self.interval_s * 2))
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001 - shutdown must never raise on exit
                pass

    def _run(self) -> None:
        assert self._origin_s is not None
        next_sample_s = self._origin_s
        while not self._stop_event.is_set():
            now_s = time.perf_counter()
            if now_s < next_sample_s:
                self._stop_event.wait(next_sample_s - now_s)
                continue
            self.samples.append(self._sample(now_s - self._origin_s))
            next_sample_s += self.interval_s

    def _sample(self, elapsed_s: float) -> dict[str, Any]:
        pynvml: Any = self._pynvml
        handle: Any = self._handle
        utilization = _safe_nvml(pynvml, pynvml.nvmlDeviceGetUtilizationRates, handle)
        power_mw = _safe_nvml(pynvml, pynvml.nvmlDeviceGetPowerUsage, handle)
        power_limit_mw = _safe_nvml(pynvml, pynvml.nvmlDeviceGetPowerManagementLimit, handle)
        return {
            "time_s": elapsed_s,
            "power_w": None if power_mw is None else float(power_mw) / 1000.0,
            "power_limit_w": None if power_limit_mw is None else float(power_limit_mw) / 1000.0,
            "sm_clock_mhz": _safe_nvml(
                pynvml, pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_SM
            ),
            "memory_clock_mhz": _safe_nvml(
                pynvml, pynvml.nvmlDeviceGetClockInfo, handle, pynvml.NVML_CLOCK_MEM
            ),
            "gpu_util_pct": None if utilization is None else int(utilization.gpu),
            "memory_util_pct": None if utilization is None else int(utilization.memory),
            "temperature_c": _safe_nvml(
                pynvml, pynvml.nvmlDeviceGetTemperature, handle, pynvml.NVML_TEMPERATURE_GPU
            ),
            "pstate": _safe_nvml(pynvml, pynvml.nvmlDeviceGetPerformanceState, handle),
            "clock_throttle_reasons": _safe_nvml(
                pynvml, pynvml.nvmlDeviceGetCurrentClocksThrottleReasons, handle
            ),
        }


def align_telemetry(
    runtimes: list[dict[str, Any]],
    telemetry: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Bin each per-launch runtime onto the nearest telemetry sample interval.

    Assignment uses the midpoints between successive telemetry timestamps, so a
    launch is attributed to the telemetry sample whose window it falls in. Each
    returned row is one telemetry sample enriched with the runtime count / mean /
    median / p90 that landed in it.
    """

    runtime_times = np.asarray([row["start_s"] for row in runtimes], dtype=float)
    runtime_values = np.asarray([row["duration_ms"] for row in runtimes], dtype=float)
    telemetry_times = np.asarray([row["time_s"] for row in telemetry], dtype=float)
    if len(telemetry_times) < 2:
        raise RuntimeError("need at least two telemetry samples to align")
    midpoints = (telemetry_times[:-1] + telemetry_times[1:]) / 2.0
    assignments = np.searchsorted(midpoints, runtime_times)

    aligned = []
    for telemetry_index, telemetry_row in enumerate(telemetry):
        values = runtime_values[assignments == telemetry_index]
        row = dict(telemetry_row)
        row.update(
            {
                "runtime_count": int(values.size),
                "runtime_mean_ms": None if values.size == 0 else float(np.mean(values)),
                "runtime_median_ms": None if values.size == 0 else float(np.median(values)),
                "runtime_p90_ms": None if values.size == 0 else float(np.percentile(values, 90)),
            }
        )
        aligned.append(row)
    return aligned


def _numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def _correlation(rows: list[dict[str, Any]], telemetry_key: str) -> float | None:
    pairs = [
        (float(row["runtime_mean_ms"]), float(row[telemetry_key]))
        for row in rows
        if row.get("runtime_mean_ms") is not None and row.get(telemetry_key) is not None
    ]
    if len(pairs) < 2:
        return None
    runtime, telemetry = np.asarray(pairs, dtype=float).T
    if np.std(runtime) == 0 or np.std(telemetry) == 0:
        return None
    return float(np.corrcoef(runtime, telemetry)[0, 1])


def _strongest_lagged_correlation(
    rows: list[dict[str, Any]],
    telemetry_key: str,
    max_lag_s: float = 0.5,
    window_start_s: float = 0.0,
) -> dict[str, float] | None:
    valid = [
        row
        for row in rows
        if float(row["time_s"]) >= window_start_s
        and row.get("runtime_mean_ms") is not None
        and row.get(telemetry_key) is not None
    ]
    if len(valid) < 3:
        return None
    runtime = np.asarray([row["runtime_mean_ms"] for row in valid], dtype=float)
    telemetry = np.asarray([row[telemetry_key] for row in valid], dtype=float)
    times = np.asarray([row["time_s"] for row in valid], dtype=float)
    interval_s = float(np.median(np.diff(times)))
    if interval_s <= 0:
        return None
    max_lag_samples = max(int(round(max_lag_s / interval_s)), 0)
    candidates = []
    for lag_samples in range(-max_lag_samples, max_lag_samples + 1):
        runtime_start = max(0, -lag_samples)
        runtime_end = len(runtime) - max(0, lag_samples)
        telemetry_start = max(0, lag_samples)
        telemetry_end = len(telemetry) - max(0, -lag_samples)
        runtime_slice = runtime[runtime_start:runtime_end]
        telemetry_slice = telemetry[telemetry_start:telemetry_end]
        if np.std(runtime_slice) == 0 or np.std(telemetry_slice) == 0:
            continue
        candidates.append(
            (
                lag_samples * interval_s,
                float(np.corrcoef(runtime_slice, telemetry_slice)[0, 1]),
            )
        )
    if not candidates:
        return None
    lag_s, correlation = max(candidates, key=lambda candidate: abs(candidate[1]))
    return {"telemetry_lag_s": lag_s, "correlation": correlation}


def _decode_throttle_reasons(reason: int) -> list[str]:
    try:
        import pynvml
    except ImportError:  # pragma: no cover - only hit without the NVML runtime
        return [str(reason)]
    if reason == int(pynvml.nvmlClocksThrottleReasonNone):
        return ["None"]
    names = []
    for name in (
        "GpuIdle",
        "ApplicationsClocksSetting",
        "SwPowerCap",
        "HwSlowdown",
        "SyncBoost",
        "SwThermalSlowdown",
        "HwThermalSlowdown",
        "HwPowerBrakeSlowdown",
        "DisplayClockSetting",
    ):
        bit = int(getattr(pynvml, f"nvmlClocksThrottleReason{name}"))
        if reason & bit:
            names.append(name)
    return names


def summarize_telemetry(aligned: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-metric range + runtime correlations over the aligned telemetry rows.

    ``steady_state_start_s`` is the first sample where measured power reaches the
    management limit; lagged correlations are reported both over the whole run and
    from that steady-state onset (power/clock feedback lags the workload).
    """

    power_limited_rows = [
        row
        for row in aligned
        if row.get("power_w") is not None
        and row.get("power_limit_w") is not None
        and float(row["power_w"]) >= float(row["power_limit_w"])
    ]
    steady_state_start_s = float(power_limited_rows[0]["time_s"]) if power_limited_rows else 0.0

    metrics: dict[str, Any] = {}
    for key in ("power_w", "sm_clock_mhz", "memory_clock_mhz", "gpu_util_pct", "temperature_c"):
        values = _numeric_values(aligned, key)
        metrics[key] = (
            None
            if not values
            else {
                "min": min(values),
                "mean": fmean(values),
                "max": max(values),
                "runtime_correlation": _correlation(aligned, key),
                "strongest_lagged_runtime_correlation": _strongest_lagged_correlation(aligned, key),
                "steady_state_strongest_lagged_runtime_correlation": _strongest_lagged_correlation(
                    aligned, key, window_start_s=steady_state_start_s
                ),
            }
        )

    observed_reasons = sorted(
        {
            int(row["clock_throttle_reasons"])
            for row in aligned
            if row.get("clock_throttle_reasons") is not None
        }
    )
    return {
        "sample_count": len(aligned),
        "steady_state_start_s": steady_state_start_s,
        "metrics": metrics,
        "lag_semantics": "positive telemetry_lag_s correlates runtime(t) with telemetry(t + lag)",
        "observed_pstates": sorted(
            {int(row["pstate"]) for row in aligned if row.get("pstate") is not None}
        ),
        "observed_clock_throttle_reasons": observed_reasons,
        "decoded_clock_throttle_reasons": {
            str(reason): _decode_throttle_reasons(reason) for reason in observed_reasons
        },
    }
