"""Timing helpers for L1a runners."""

from __future__ import annotations

import importlib
import time
import warnings
from collections.abc import Callable
from statistics import median
from typing import Any, cast

from profiling.runners.exceptions import ProfilerNotImplemented

_AGGREGATE_RUNS = 3
_AGGREGATE_WARNING_RATIO = 1.1


class Timer:
    """L1 runner timing primitives.

    Agent note: these methods intentionally mirror the four legacy timing
    families documented in L1. Do not alias one timer to another; runner authors
    choose the method because each one measures a different boundary.

    Stage-1 outlier handling is aggregate-level: each method compares three
    independent rep-average measurements, warns if their max/min ratio is at
    least 1.1x, and returns their median. Keep that shape aligned with the L1
    docs before changing it.
    """

    @staticmethod
    def do_bench(fn: Callable[[], object], *, warmup: int, rep: int) -> float:
        """Return average runtime in milliseconds using Triton's benchmarker."""

        _validate_reps("Timer.do_bench", warmup=warmup, rep=rep)
        triton = _import_optional_module("triton", "Timer.do_bench")

        def measure_once() -> float:
            return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep))

        return _median_aggregate_time(measure_once)

    @staticmethod
    def cuda_event(fn: Callable[[], object], *, warmup: int, rep: int) -> float:
        """Return average runtime in milliseconds using CUDA events.

        This matches the legacy NCCL/attention event-timing path: warm up,
        synchronize once, record a CUDA event pair around the measured loop, and
        divide the elapsed GPU time by the repetition count.
        """

        _validate_reps("Timer.cuda_event", warmup=warmup, rep=rep)
        torch = _import_torch("Timer.cuda_event")
        if not torch.cuda.is_available():
            raise ProfilerNotImplemented("Timer.cuda_event requires CUDA")
        for _ in range(warmup):
            fn()

        torch.cuda.synchronize()

        def measure_once() -> float:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(rep):
                fn()
            end.record()
            end.synchronize()
            return float(start.elapsed_time(end)) / rep

        return _median_aggregate_time(measure_once)

    @staticmethod
    def wall_clock(fn: Callable[[], object], *, warmup: int, rep: int) -> float:
        """Return average end-to-end runtime in milliseconds."""

        _validate_reps("Timer.wall_clock", warmup=warmup, rep=rep)
        for _ in range(warmup):
            fn()

        def measure_once() -> float:
            _try_cuda_synchronize()
            start_s = time.perf_counter()
            for _ in range(rep):
                fn()
            _try_cuda_synchronize()
            return (time.perf_counter() - start_s) * 1000.0 / rep

        return _median_aggregate_time(measure_once)

    @staticmethod
    def cupti(
        fn: Callable[[], object],
        *,
        warmup: int,
        rep: int,
        kernel_name: str | None = None,
    ) -> float:
        """Return average kernel-only runtime in milliseconds using CUPTI."""

        _validate_reps("Timer.cupti", warmup=warmup, rep=rep)
        profile_kernel = _load_cupti_profile_kernel()

        def measure_once() -> float:
            summary: Any = profile_kernel(
                fn,
                num_warmup=warmup,
                num_iter=rep,
                kernel_name_contains=kernel_name,
            )
            return float(summary.mean_ms)

        return _median_aggregate_time(measure_once)


def _median_aggregate_time(measure_once: Callable[[], float]) -> float:
    measurements_ms = [measure_once() for _ in range(_AGGREGATE_RUNS)]
    _warn_if_aggregate_runs_diverge(measurements_ms)
    return float(median(measurements_ms))


def _warn_if_aggregate_runs_diverge(measurements_ms: list[float]) -> None:
    positive_measurements_ms = [value for value in measurements_ms if value > 0]
    if len(positive_measurements_ms) < 2:
        return

    min_ms = min(positive_measurements_ms)
    max_ms = max(positive_measurements_ms)
    ratio = max_ms / min_ms
    if ratio < _AGGREGATE_WARNING_RATIO:
        return

    warnings.warn(
        "Timer aggregate runs differ by "
        f"{ratio:.2f}x; values_ms={_format_measurements(measurements_ms)}; "
        f"returning median of {_AGGREGATE_RUNS} runs",
        RuntimeWarning,
        stacklevel=3,
    )


def _format_measurements(measurements_ms: list[float]) -> str:
    return "[" + ", ".join(f"{value:.6g}" for value in measurements_ms) + "]"


def _validate_reps(timer_name: str, *, warmup: int, rep: int) -> None:
    if warmup < 0:
        raise ValueError(f"{timer_name} warmup must be >= 0")
    if rep <= 0:
        raise ValueError(f"{timer_name} rep must be >= 1")


def _import_optional_module(module_name: str, timer_name: str) -> Any:
    try:
        return cast(Any, importlib.import_module(module_name))
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{timer_name} requires {module_name}") from exc


def _import_torch(timer_name: str) -> Any:
    return _import_optional_module("torch", timer_name)


def _try_cuda_synchronize() -> None:
    try:
        torch = cast(Any, importlib.import_module("torch"))
    except ImportError:
        return
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.synchronize()


def _load_cupti_profile_kernel() -> Callable[..., Any]:
    try:
        cupti_module = cast(
            Any,
            importlib.import_module("profiling.profilers.cupti_kernel_profiler"),
        )
    except ImportError:
        try:
            cupti_module = cast(Any, importlib.import_module("cupti_kernel_profiler"))
        except ImportError as exc:
            raise ProfilerNotImplemented(
                "Timer.cupti requires the reusable CUPTI profiler module"
            ) from exc
    return cast(Callable[..., Any], cupti_module.profile_kernel)
