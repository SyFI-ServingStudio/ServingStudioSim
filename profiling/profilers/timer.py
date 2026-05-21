"""Timing helpers for L1a runners."""

from __future__ import annotations

import importlib
import time
import warnings
from collections.abc import Callable
from statistics import median
from typing import Any, cast

from profiling.profilers._duration import (
    DEFAULT_MIN_DURATION_MS,
    DEFAULT_MIN_REP,
    iters_for_duration,
    warn_if_multi_gpu_duration_mode,
)
from profiling.runners.exceptions import ProfilerNotImplemented

_AGGREGATE_RUNS = 3
_AGGREGATE_WARNING_RATIO = 1.1
_ESTIMATE_WARMUP = 10
_ESTIMATE_ITERS = 10
_MIN_PER_ITER_MS = 1e-4

# Adaptive CUPTI sampling defaults (see Timer.cupti).
_CUPTI_BATCH = 10
_CUPTI_MIN_ITER = 20
_CUPTI_MAX_ITER = 500
_CUPTI_TOL = 0.01


class Timer:
    """L1 runner timing primitives.

    Agent note: these methods intentionally mirror the four legacy timing
    families documented in L1. Do not alias one timer to another; runner authors
    choose the method because each one measures a different boundary.

    Stage-1 outlier handling is aggregate-level: each method compares three
    independent rep-average measurements, warns if their max/min ratio is at
    least 1.1x, and returns their median. Keep that shape aligned with the L1
    docs before changing it.

    Loop sizing is time-centric by default: run for at least ``min_duration_ms``
    (default 500), but never fewer than ``min_rep`` iterations (default 3) so an
    ultra-slow kernel whose single iteration already exceeds the budget still
    gets enough reps for the median. The loop size is ``max(time_iters, min_rep)``
    -- the longer run wins. We time a short burst (~10 iters) to estimate per-iter
    cost (mirroring ``Energy.perf``), since the time budget is in wall-clock ms.

    ``rep`` is an optional escape hatch: pass it to fix an exact iteration count
    instead. It is mutually exclusive with ``min_duration_ms`` / ``min_rep``
    (passing both raises). Because the time-centric size is derived from a
    per-process estimate it is non-deterministic across ranks, so multi-GPU /
    collective runs MUST pass ``rep`` for a fixed, identical loop on every rank.

    ``do_bench`` is duration-native -- Triton's ``rep`` is itself a ms budget that
    sizes its own inner loop -- so for it the resolved iteration count is
    converted back to a ms budget (``iters * per_iter_ms``) before handing off.

    ``cupti`` is the exception: it records each launch's true kernel duration, so
    instead of a time budget it samples adaptively and stops once the mean
    converges (see ``Timer.cupti``). It takes ``min_rep`` / ``max_rep`` / ``tol``
    rather than ``min_duration_ms``, and needs no warmup.
    """

    @staticmethod
    def do_bench(
        fn: Callable[[], object],
        *,
        warmup: int,
        rep: int | None = None,
        min_duration_ms: int | None = None,
        min_rep: int | None = None,
    ) -> float:
        """Return average runtime in milliseconds using Triton's benchmarker.

        Time-centric by default; pass ``rep`` for an exact count (see class docstring).
        """

        _validate_warmup("Timer.do_bench", warmup)
        rep = _resolve_loop_size(
            fn,
            rep=rep,
            min_duration_ms=min_duration_ms,
            min_rep=min_rep,
            timer_name="Timer.do_bench",
            duration_native=True,
        )
        triton = _import_optional_module("triton", "Timer.do_bench")

        def measure_once() -> float:
            return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep))

        return _median_aggregate_time(measure_once)

    @staticmethod
    def cuda_event(
        fn: Callable[[], object],
        *,
        warmup: int,
        rep: int | None = None,
        min_duration_ms: int | None = None,
        min_rep: int | None = None,
    ) -> float:
        """Return average runtime in milliseconds using CUDA events.

        This matches the legacy NCCL/attention event-timing path: warm up,
        synchronize once, record a CUDA event pair around the measured loop, and
        divide the elapsed GPU time by the repetition count.

        Time-centric by default; pass ``rep`` for an exact count (see class docstring).
        """

        _validate_warmup("Timer.cuda_event", warmup)
        rep = _resolve_loop_size(
            fn,
            rep=rep,
            min_duration_ms=min_duration_ms,
            min_rep=min_rep,
            timer_name="Timer.cuda_event",
            duration_native=False,
        )
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
    def wall_clock(
        fn: Callable[[], object],
        *,
        warmup: int,
        rep: int | None = None,
        min_duration_ms: int | None = None,
        min_rep: int | None = None,
    ) -> float:
        """Return average end-to-end runtime in milliseconds.

        Time-centric by default; pass ``rep`` for an exact count (see class docstring).
        """

        _validate_warmup("Timer.wall_clock", warmup)
        rep = _resolve_loop_size(
            fn,
            rep=rep,
            min_duration_ms=min_duration_ms,
            min_rep=min_rep,
            timer_name="Timer.wall_clock",
            duration_native=False,
        )
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
        warmup: int = 0,
        rep: int | None = None,
        min_rep: int | None = None,
        max_rep: int | None = None,
        tol: float | None = None,
        kernel_name: str | None = None,
    ) -> float:
        """Return average kernel-only runtime in milliseconds using CUPTI.

        Unlike the wall-clock timers, CUPTI records each launch's true kernel
        duration, so the default path samples adaptively (batches of
        ``_CUPTI_BATCH``) and stops once the mean's relative standard error drops
        below ``tol`` (default 1%), bounded by ``[min_rep, max_rep]`` samples and
        needing no warmup. Pass ``rep`` for a fixed sample count instead -- a
        deterministic escape hatch, mutually exclusive with the convergence knobs.
        """

        _validate_warmup("Timer.cupti", warmup)
        if rep is not None and (
            min_rep is not None or max_rep is not None or tol is not None
        ):
            raise ValueError(
                "Timer.cupti: rep is mutually exclusive with min_rep / max_rep / tol"
            )
        cupti = _load_cupti_module()

        if rep is not None:
            if rep <= 0:
                raise ValueError("Timer.cupti rep must be >= 1")

            def measure_once() -> float:
                summary: Any = cupti.profile_kernel(
                    fn,
                    num_warmup=warmup,
                    num_iter=rep,
                    kernel_name_contains=kernel_name,
                )
                return float(summary.mean_ms)

            return _median_aggregate_time(measure_once)

        warn_if_multi_gpu_duration_mode("Timer.cupti")
        summary = cupti.profile_kernel_until_converged(
            fn,
            num_warmup=warmup,
            batch=_CUPTI_BATCH,
            min_iter=_CUPTI_MIN_ITER if min_rep is None else min_rep,
            max_iter=_CUPTI_MAX_ITER if max_rep is None else max_rep,
            tol=_CUPTI_TOL if tol is None else tol,
            kernel_name_contains=kernel_name,
        )
        return float(summary.mean_ms)


def _resolve_loop_size(
    fn: Callable[[], object],
    *,
    rep: int | None,
    min_duration_ms: int | None,
    min_rep: int | None,
    timer_name: str,
    duration_native: bool,
) -> int:
    """Resolve the loop size, time-centric by default with ``rep`` as an override.

    Mirrors ``_median_aggregate_time``: a thin wrapper the four timers share so
    the loop-sizing policy lives in one place. Default path: estimate per-iter
    cost and run ``max(iters_for_duration(min_duration_ms), min_rep)`` iterations
    -- run for the budget, but never fewer than ``min_rep`` reps. ``rep`` instead
    fixes an exact iteration count and is mutually exclusive with the time knobs.

    The return unit follows the benchmarker: an iteration count for the loop-based
    timers, or a ms budget for ``duration_native`` timers (``do_bench``, whose
    Triton ``rep`` is itself a ms budget), reached by converting iters back to ms.
    """

    if rep is not None and (min_duration_ms is not None or min_rep is not None):
        raise ValueError(
            f"{timer_name}: rep is mutually exclusive with min_duration_ms / min_rep"
        )

    if rep is not None:
        if rep <= 0:
            raise ValueError(f"{timer_name} rep must be >= 1")
        # Exact override. do_bench still needs ms, so convert the count.
        if not duration_native:
            return rep
        return max(round(rep * _estimate_per_iter_ms(fn)), 1)

    duration = DEFAULT_MIN_DURATION_MS if min_duration_ms is None else min_duration_ms
    floor = DEFAULT_MIN_REP if min_rep is None else min_rep
    if duration <= 0:
        raise ValueError(f"{timer_name} min_duration_ms must be >= 1")
    if floor < 1:
        raise ValueError(f"{timer_name} min_rep must be >= 1")

    warn_if_multi_gpu_duration_mode(timer_name)
    per_iter_ms = _estimate_per_iter_ms(fn)
    iters = max(iters_for_duration(duration, per_iter_ms), floor)
    if not duration_native:
        return iters
    return max(round(iters * per_iter_ms), 1)


def _estimate_per_iter_ms(fn: Callable[[], object]) -> float:
    for _ in range(_ESTIMATE_WARMUP):
        fn()
    _try_cuda_synchronize()
    start_s = time.perf_counter()
    for _ in range(_ESTIMATE_ITERS):
        fn()
    _try_cuda_synchronize()
    elapsed_ms = (time.perf_counter() - start_s) * 1000.0
    return max(elapsed_ms / _ESTIMATE_ITERS, _MIN_PER_ITER_MS)


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


def _validate_warmup(timer_name: str, warmup: int) -> None:
    if warmup < 0:
        raise ValueError(f"{timer_name} warmup must be >= 0")


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


def _load_cupti_module() -> Any:
    try:
        return cast(
            Any,
            importlib.import_module("profiling.profilers.cupti_kernel_profiler"),
        )
    except ImportError:
        try:
            return cast(Any, importlib.import_module("cupti_kernel_profiler"))
        except ImportError as exc:
            raise ProfilerNotImplemented(
                "Timer.cupti requires the reusable CUPTI profiler module"
            ) from exc
