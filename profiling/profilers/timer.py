"""Timing helpers for L1a runners."""

from __future__ import annotations

import importlib
import json
import os
import time
import warnings
from collections.abc import Callable, Sequence
from statistics import median
from typing import Any, cast

from profiling.profilers._duration import (
    DEFAULT_MIN_DURATION_MS,
    DEFAULT_MIN_REP,
    iters_for_duration,
    warn_if_multi_gpu_duration_mode,
)
from profiling.profilers.measure_context import get_measure_context
from profiling.runners.exceptions import ProfilerNotImplemented

_AGGREGATE_RUNS = 3
_AGGREGATE_WARNING_RATIO = 1.1
_ESTIMATE_WARMUP = 10
_ESTIMATE_ITERS = 10
_MIN_PER_ITER_MS = 1e-4

# Duration-sized CUPTI defaults (see Timer.cupti). Ten real launches estimate
# the formal count; one uninterrupted capture then records the full active-time
# budget so capture restarts do not reset the workload's power/clock state.
# The maximum is a real execution cap for ultra-short kernels, not an overflow
# condition: reaching it trades active-window length for bounded profiling cost.
_CUPTI_ESTIMATE_ITERS = 10
_CUPTI_MIN_ITER = 10
# 10k, not 100k. The cap binds only for kernels shorter than budget/cap, so at
# 100k it bound below ~0.02 ms -- and there it dominated everything: a 0.0016 ms
# elementwise spec spent 100 s of wall clock on 100k cold-L2 launches, because
# the per-launch 64 MiB L2 displacement costs far more than the kernel. Those
# kernels are also the ones a short window already measures precisely (0.03%
# spread across a whole 33k-launch window, and every prefix of it).
_CUPTI_MAX_ITER = 10_000
_CUPTI_MIN_DURATION_MS = 2_000

# Adaptive budget. A kernel whose full-clock power demand stays under the board
# cap never downclocks, and its per-launch time is final within tens of ms: over
# a 20 s window such kernels moved 0.02-0.31% while the die warmed 25 C, because
# the SM clock is the only path from temperature to timing and it stays pinned.
# One that does cross the cap is 8-19% off at cold and needs seconds. The two are
# separated by their own launch durations -- no NVML needed -- so the probe below
# is the measurement for the first class and a discarded prelude for the second.
_CUPTI_PROBE_DURATION_MS = 100
# The probe gets its own, much tighter launch cap. Every launch costs ~0.05 ms
# of wall clock no matter how short the kernel is -- the per-launch 64 MiB L2
# displacement and the CUPTI record dominate -- so for a microsecond kernel the
# cost is set by the launch COUNT, not by the budget. Sharing _CUPTI_MAX_ITER
# made the probe cost exactly what the full budget costs there (both capped at
# the same number), which is why such kernels saw no saving at all. 1000
# launches reproduce the full window's mean to 0.03% (measured: the first 1676
# of a 33529-launch elementwise window read 0.002986 vs 0.002987 for all of it).
_CUPTI_PROBE_MAX_ITER = 1_000
# 2%, not 1%. Measured over 60 captures that paid the full budget: grouped by
# the drift the probe reported, the full window then moved the answer by 0.56%
# (drift 1-2%, 7 captures), 4.91% (2-5%), 5.88% (5-10%) and 7.30% (>=10%). The
# signal only predicts a real change above 2%; below it the 2000 ms window cost
# 9.2 s to confirm a number the probe already had.
_CUPTI_DRIFT_TOLERANCE = 0.02
# Eight, so that a probe sized at the _CUPTI_MIN_ITER floor still gets a
# verdict. At 20 it did not, and "cannot tell" means pay the full budget: 26 of
# the 60 full-budget captures in that same run were slow kernels (median 13 ms)
# forced down this path, 53.8 s of measurement that moved the answer by 0.10%.
# Worse, those kernels ended up SLOWER than a plain fixed budget, because they
# paid the probe and then the full window.
_CUPTI_DRIFT_MIN_SAMPLES = 8
# Below this, split the probe in half rather than in quarters. A ten-launch
# probe has two launches per quarter and five per half, and for a kernel this
# slow the half carries the same before/after meaning -- the window is hundreds
# of ms either way -- while using every sample instead of a fifth of them.
_CUPTI_DRIFT_QUARTER_MIN_SAMPLES = 16

# Pins every adaptive capture to one fixed budget, exactly as if each runner had
# passed ``min_duration_ms``. It exists so the adaptive budget can be measured
# against a long fixed one on the same cards and the same revision -- the only
# way to tell a method change apart from run-to-run noise -- and it doubles as
# the escape hatch if the drift verdict ever misfires on a new kernel. Unset in
# normal use; a caller's explicit ``min_duration_ms`` still wins over it.
CUPTI_BUDGET_ENV = "VIBESIM_PROFILE_CUPTI_BUDGET_MS"

# Append one JSON line per adaptive capture to this path, recording what the
# drift verdict saw and what it decided. The budget decision is otherwise
# invisible -- two runs of one spec can take different branches and only differ
# in wall clock -- and a wrong verdict costs accuracy, not just time. Unset in
# normal use; workers append concurrently, so each record is one short line.
CUPTI_TRACE_ENV = "VIBESIM_PROFILE_CUPTI_TRACE"


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

    ``cupti`` uses a GPU-active-time budget rather than wall clock. Ten CUPTI
    samples estimate the count required for the budget, then one uninterrupted
    capture records exactly that many launches. The budget is adaptive: a 100 ms
    probe runs first and is kept as the answer when the kernel's own per-launch
    durations show no drift, and only a kernel that does drift pays the full
    2000 ms -- sized from the probe's closing rate, so the drifting path costs
    one estimate, not two. The probe also carries a tighter launch cap than the
    full budget, because below a few tens of microseconds a capture's cost is
    its launch count rather than its budget. Pass ``min_duration_ms`` to fix the
    budget instead. By default each launch is preceded by a read-only 64 MiB
    L2-displacement reduction; pass ``clear_l2=False`` only for an explicit
    warm-cache probe.
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

        time_ms = _median_aggregate_time(measure_once)
        _compare_against_cupti(fn, time_ms, rep)
        return time_ms

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
        min_duration_ms: int | None = None,
        min_rep: int | None = None,
        max_rep: int | None = None,
        kernel_name: str | None = None,
        clear_l2: bool = True,
        interval_union: bool = False,
    ) -> float:
        """Return average kernel-only runtime in milliseconds using CUPTI.

        Unlike the wall-clock timers, CUPTI records each launch's true kernel
        duration. With ``interval_union=True``, overlapping physical kernels
        count once, so a multi-stream fused callable is priced by GPU-active
        union instead of summed residency. The default path measures ten
        launches, computes ``ceil(budget_ms / estimate_ms)``, and records that
        many launches in one uninterrupted formal capture. The budget is a
        100 ms probe, kept when the probe's launches show no drift and replaced
        by a fresh 2000 ms capture when they do -- that second capture is sized
        from the probe's own closing launch rate rather than a second estimate,
        so it starts from the board state the probe left behind. An explicit
        ``min_duration_ms`` skips the probe and is used as given. With
        ``clear_l2=True`` (default),
        a read-only 64 MiB reduction displaces L2 before every logical launch;
        its CUPTI records are excluded from the returned callable time.
        ``min_rep`` is a launch-count floor and ``max_rep`` caps the formal
        launch count when the active-time estimate asks for more repetitions.
        Pass ``rep`` for the fixed-count median-of-three path instead.

        When a ``MeasureContext`` is active (only ever set by ``python -m profiling
        measure``), this call is diverted to a sustained trend+telemetry capture
        that writes CSV / summary / plots into the context's output dir and returns
        the per-launch median so the runner still completes. The guard is inert for
        every normal ``run`` / ``query`` / simulator call.
        """

        measure_context = get_measure_context()
        if measure_context is not None and not measure_context.consumed:
            measure_context.consumed = True
            from profiling.profilers.trend import run_measure_capture

            return run_measure_capture(fn, measure_context)

        _validate_warmup("Timer.cupti", warmup)
        if rep is not None and (
            min_duration_ms is not None or min_rep is not None or max_rep is not None
        ):
            raise ValueError(
                "Timer.cupti: rep is mutually exclusive with min_duration_ms / min_rep / max_rep"
            )
        cupti = _load_cupti_module()
        duration_kwargs = {"interval_union": True} if interval_union else {}

        if rep is not None:
            if rep <= 0:
                raise ValueError("Timer.cupti rep must be >= 1")

            def measure_once() -> float:
                summary: Any = cupti.profile_kernel(
                    fn,
                    num_warmup=warmup,
                    num_iter=rep,
                    clear_l2_before_run=clear_l2,
                    clear_l2_between_launches=clear_l2,
                    kernel_name_contains=kernel_name,
                    **duration_kwargs,
                )
                return float(summary.mean_ms)

            return _median_aggregate_time(measure_once)

        warn_if_multi_gpu_duration_mode("Timer.cupti")
        if min_duration_ms is None:
            min_duration_ms = _budget_override()
        if min_duration_ms is not None and min_duration_ms < 0:
            raise ValueError("Timer.cupti min_duration_ms must be >= 0")

        iter_floor = _CUPTI_MIN_ITER if min_rep is None else min_rep
        iter_cap = _CUPTI_MAX_ITER if max_rep is None else max_rep
        # An explicit max_rep is a ceiling on every capture, probe included --
        # but never below the floor, which an explicit min_rep can raise past
        # the probe's own cap. `profile_kernel_for_duration` rejects
        # max_iter < min_iter, so without the clamp a caller asking for
        # min_rep=2000 would get a ValueError instead of a measurement.
        probe_cap = max(iter_floor, min(iter_cap, _CUPTI_PROBE_MAX_ITER))

        def capture_for_duration(duration_ms: int, cap: int) -> Any:
            return cupti.profile_kernel_for_duration(
                fn,
                num_warmup=warmup,
                estimate_iter=_CUPTI_ESTIMATE_ITERS,
                min_duration_ms=duration_ms,
                min_iter=iter_floor,
                max_iter=cap,
                clear_l2_before_run=clear_l2,
                clear_l2_between_launches=clear_l2,
                kernel_name_contains=kernel_name,
                **duration_kwargs,
            )

        def capture_for_count(num_iter: int) -> Any:
            # Same entry point as the budget-sized capture, with the count
            # supplied instead of estimated -- NOT ``profile_kernel``, which
            # opens one CUPTI window per launch. Those restarts let the board
            # recover between launches and read a power-capped kernel up to 13%
            # faster than it sustains (measured on batched_gemm and
            # single_gemm, reproducible to 0.2-0.4%).
            return cupti.profile_kernel_for_duration(
                fn,
                num_warmup=warmup,
                estimate_iter=_CUPTI_ESTIMATE_ITERS,
                min_duration_ms=_CUPTI_MIN_DURATION_MS,
                min_iter=iter_floor,
                max_iter=iter_cap,
                launch_count=num_iter,
                clear_l2_before_run=clear_l2,
                clear_l2_between_launches=clear_l2,
                kernel_name_contains=kernel_name,
                **duration_kwargs,
            )

        if min_duration_ms is not None:
            # An explicit budget is an instruction, not a starting point.
            fixed_started_at = time.perf_counter()
            fixed = capture_for_duration(min_duration_ms, iter_cap)
            fixed_launches = getattr(fixed, "per_iter_ms", None)
            # Traced too, even though no verdict was made: the launch count a
            # budget actually bought is the thing you need to compare two
            # budgets, and for a microsecond kernel it saturates at iter_cap
            # long before the budget is spent.
            _emit_trace(
                {
                    "kernel_name": kernel_name,
                    "budget_ms": min_duration_ms,
                    "launches": len(fixed_launches) if fixed_launches is not None else None,
                },
                reps=None,
                mean_ms=float(fixed.mean_ms),
                full_wall_s=time.perf_counter() - fixed_started_at,
            )
            return float(fixed.mean_ms)

        probe_started_at = time.perf_counter()
        probe = capture_for_duration(_CUPTI_PROBE_DURATION_MS, probe_cap)
        probe_wall_s = time.perf_counter() - probe_started_at
        probe_launches = getattr(probe, "per_iter_ms", None)
        drift_ratio = _drift_ratio(probe_launches)
        drifted = drift_ratio is None or drift_ratio > _CUPTI_DRIFT_TOLERANCE
        trace = {
            "kernel_name": kernel_name,
            "probe_launches": len(probe_launches) if probe_launches is not None else None,
            "probe_mean_ms": float(probe.mean_ms),
            "probe_wall_s": round(probe_wall_s, 4),
            "drift_ratio": drift_ratio,
            "drifted": drifted,
        }
        if not drifted:
            _emit_trace(trace, reps=None, mean_ms=float(probe.mean_ms), full_wall_s=0.0)
            return float(probe.mean_ms)
        # Size the full capture from the probe's own launches rather than paying
        # a second ten-launch estimate: the probe already timed this kernel more
        # thoroughly than an estimator would. The rate comes from the probe's
        # tail, which ran at the downclocked speed the full window will spend
        # nearly all of its time at; sizing off the head would under-count.
        #
        # That full window therefore starts from an already-throttled board
        # instead of the cold one a lone 2000 ms capture used to see. Measured
        # on H100-class hardware that shifts a throttled kernel's recorded time
        # by +0.2-0.4% (bf16_fused_moe 3.1131 -> 3.1242 ms; nvfp4 nt=65536
        # 3.6361 -> 3.6442 ms) and a flat one not at all -- inside the +/-1%
        # throughput-regression band, and toward the sustained value rather than
        # away from it.
        reps = _reps_from_probe(probe_launches, iter_floor, iter_cap)
        full_started_at = time.perf_counter()
        if reps is None:
            # The probe reported no usable per-launch series, so there is
            # nothing to size from; fall back to estimating the count again.
            mean_ms = float(capture_for_duration(_CUPTI_MIN_DURATION_MS, iter_cap).mean_ms)
        else:
            mean_ms = float(capture_for_count(reps).mean_ms)
        _emit_trace(trace, reps, mean_ms, time.perf_counter() - full_started_at)
        return mean_ms


def _drifted(per_iter_ms: Sequence[float] | None) -> bool:
    """Did the kernel slow down inside the probe window?

    ``True`` also covers "cannot tell", so the caller spends the full budget
    whenever the probe is not evidence of stability.
    """

    ratio = _drift_ratio(per_iter_ms)
    return ratio is None or ratio > _CUPTI_DRIFT_TOLERANCE


def _drift_ratio(per_iter_ms: Sequence[float] | None) -> float | None:
    """Relative gap between the probe's first and last quarter of launches.

    A kernel that stays under the board's power cap holds full clock and reads
    within a few tenths of a percent across a whole window; one that crosses the
    cap downclocks and reads several percent apart, so the two classes are
    separated by more than an order of magnitude and the threshold sits in the
    gap rather than on either side of it.

    ``None`` means the probe cannot answer -- too few launches to split, or a
    nonsensical head -- which is not the same as a measured zero drift. Keep
    that floor as low as the launch floor allows: a refusal costs the full
    budget, so a verdict this test declines to give is the most expensive
    answer it can produce.
    """

    if per_iter_ms is None or len(per_iter_ms) < _CUPTI_DRIFT_MIN_SAMPLES:
        return None
    split = (
        len(per_iter_ms) // 4
        if len(per_iter_ms) >= _CUPTI_DRIFT_QUARTER_MIN_SAMPLES
        else len(per_iter_ms) // 2
    )
    head = median(per_iter_ms[:split])
    tail = median(per_iter_ms[-split:])
    if head <= 0:
        return None
    return abs(tail - head) / head


def _emit_trace(
    trace: dict[str, Any],
    reps: int | None,
    mean_ms: float,
    full_wall_s: float,
) -> None:
    """Append one budget decision to ``CUPTI_TRACE_ENV``, when that is set.

    Opened per record in append mode: several worker processes share the path,
    and a short line written by one ``write`` is the cheapest thing that stays
    intact across them. Failure to write is swallowed -- a diagnostic must never
    be the reason a measurement run dies.
    """

    path = os.environ.get(CUPTI_TRACE_ENV)
    if not path:
        return
    record = trace | {
        "full_reps": reps,
        "mean_ms": mean_ms,
        "full_wall_s": round(full_wall_s, 4),
    }
    try:
        with open(path, "a", encoding="utf-8") as sink:
            sink.write(json.dumps(record) + "\n")
    except OSError:
        pass


def _budget_override() -> int | None:
    """The fixed CUPTI budget ``CUPTI_BUDGET_ENV`` asks for, if it asks for one.

    Raises on a value that is not a non-negative integer rather than falling
    back to the adaptive default: this knob's whole purpose is to make a run's
    budget unambiguous, so a typo that silently restored the default would
    produce a comparison that quietly measures nothing.
    """

    raw = os.environ.get(CUPTI_BUDGET_ENV)
    if raw is None or not raw.strip():
        return None
    try:
        budget_ms = int(raw)
    except ValueError:
        raise ValueError(
            f"{CUPTI_BUDGET_ENV} must be an integer number of ms, got {raw!r}"
        ) from None
    if budget_ms < 0:
        raise ValueError(f"{CUPTI_BUDGET_ENV} must be >= 0, got {budget_ms}")
    return budget_ms


def _reps_from_probe(
    per_iter_ms: Sequence[float] | None,
    iter_floor: int,
    iter_cap: int,
) -> int | None:
    """Launch count covering the full budget at the probe's closing rate.

    Replaces the second ten-launch estimate a duration-sized capture would run:
    the probe already measured this kernel, and its last quarter measured it in
    the throttled state the full capture will hold. ``None`` means the probe
    carried no usable series and the caller must size the capture some other way.
    """

    if not per_iter_ms:
        return None
    tail = per_iter_ms[-max(len(per_iter_ms) // 4, 1) :]
    rate_ms = median(tail)
    if rate_ms <= 0:
        return None
    return min(iter_cap, max(iter_floor, iters_for_duration(_CUPTI_MIN_DURATION_MS, rate_ms)))


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
        raise ValueError(f"{timer_name}: rep is mutually exclusive with min_duration_ms / min_rep")

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


TIMER_COMPARE_ENV = "VIBESIM_TIMER_COMPARE"


def _compare_against_cupti(fn: Callable[[], object], event_ms: float, event_rep: int) -> None:
    """Diagnostic: price the same callable with CUPTI as well as CUDA events.

    `dsa_persistent_topk_decode:vllm_cuda` is the only production backend still
    on `Timer.cuda_event`, and it costs 670 s of the 3151 GPU-s a full fill
    needs -- it never sees the adaptive CUPTI budget and pays a flat
    3 x DEFAULT_MIN_DURATION_MS instead. Whether it can move to `Timer.cupti`
    is an accuracy question first: CUPTI prices kernel-only time, CUDA events
    price the whole loop including inter-kernel gaps and host dispatch, so a
    multi-kernel composite is legitimately dearer under events.

    Off unless `VIBESIM_TIMER_COMPARE` is set. Records to the timeline, which
    already crosses the container boundary, so no new plumbing is needed.
    """

    if os.environ.get(TIMER_COMPARE_ENV, "").strip().lower() in ("", "0", "false", "no", "off"):
        return
    from profiling.instrument import emit

    started = time.time()
    try:
        cupti_ms = Timer.cupti(fn, warmup=5)
    except Exception as exc:  # a diagnostic must not fail the fill
        emit("timer.compare", started, time.time(), event_ms=event_ms, error=repr(exc)[:300])
        return
    emit(
        "timer.compare",
        started,
        time.time(),
        event_ms=event_ms,
        cupti_ms=cupti_ms,
        event_rep=event_rep,
        ratio=round(event_ms / cupti_ms, 4) if cupti_ms > 0 else None,
    )


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
