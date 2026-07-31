"""Reusable CUPTI-based kernel profiler for CUDA workloads.

This is the L1-local copy of the legacy ``ref/profile/cupti_kernel_profiler.py``
helper. ``Timer.cupti`` depends on this module for kernel-only timings, while
runner files stay small and only choose the timer method.
"""

from __future__ import annotations

import os
import site
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from math import ceil, sqrt
from pathlib import Path
from statistics import fmean, median, stdev
from typing import Any, cast

try:
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, load
except ImportError as exc:  # pragma: no cover - depends on optional runtime env
    torch = None  # type: ignore[assignment]
    CUDA_HOME = None  # type: ignore[assignment]
    load = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR: ImportError | None = exc
else:
    _TORCH_IMPORT_ERROR = None


THIS_DIR = Path(__file__).resolve().parent
EXT_SOURCE = THIS_DIR / "csrc" / "cupti_activity_profiler.cpp"
BUILD_DIR = Path(os.environ.get("MOESIM_CUPTI_BUILD_DIR", "/tmp/moesim_cupti_ext"))


@dataclass(frozen=True)
class KernelRecord:
    name: str
    device_id: int
    stream_id: int
    correlation_id: int
    start_ns: int
    end_ns: int
    duration_ns: int


@dataclass(frozen=True)
class KernelProfileSummary:
    matched_kernel_names: list[str]
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    num_warmup: int
    num_iter: int
    launches_per_run: int
    clear_l2_bytes: int
    clear_l2_between_launches: bool
    per_iter_ms: list[float]
    matched_kernel_count_per_run: list[int]


@dataclass(frozen=True)
class _LaunchPattern:
    """Exact CUPTI kernel-name pattern for one callable and one L2 clear."""

    callable_kernel_names: tuple[str, ...]
    clear_kernel_names: tuple[str, ...]


def _require_torch() -> Any:
    if _TORCH_IMPORT_ERROR is not None or torch is None:
        raise RuntimeError("CUPTI profiling requires torch with CUDA extension support") from (
            _TORCH_IMPORT_ERROR
        )
    return torch


def _site_roots() -> list[Path]:
    roots = [Path(path) for path in site.getsitepackages()]
    user_site = site.getusersitepackages()
    if user_site:
        roots.append(Path(user_site))
    return [path for path in roots if path.exists()]


def _resolve_cupti_paths() -> tuple[Path, Path]:
    candidates: list[tuple[Path, Path]] = []
    for root in _site_roots():
        include_dir = root / "nvidia" / "cuda_cupti" / "include"
        lib_dir = root / "nvidia" / "cuda_cupti" / "lib"
        if (include_dir / "cupti.h").exists() and any(lib_dir.glob("libcupti.so*")):
            candidates.append((include_dir, lib_dir))

        triton_include = root / "triton" / "backends" / "nvidia" / "include"
        triton_lib = root / "triton" / "backends" / "nvidia" / "lib" / "cupti"
        if (triton_include / "cupti.h").exists() and any(triton_lib.glob("libcupti.so*")):
            candidates.append((triton_include, triton_lib))

    if not candidates:
        raise RuntimeError(
            "Could not locate CUPTI headers and libraries in the current environment"
        )
    return candidates[0]


def _load_extension() -> Any:
    _require_torch()
    if load is None:
        raise RuntimeError("CUPTI profiling requires torch.utils.cpp_extension.load")

    include_dir, lib_dir = _resolve_cupti_paths()
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    extra_includes = [str(include_dir)]
    if CUDA_HOME is not None:
        extra_includes.append(str(Path(CUDA_HOME) / "include"))

    cupti_lib = lib_dir / "libcupti.so"
    if not cupti_lib.exists():
        matches = sorted(lib_dir.glob("libcupti.so*"))
        if not matches:
            raise RuntimeError(f"Could not locate libcupti in {lib_dir}")
        cupti_lib = matches[0]

    # Agent note: keep the extension name stable so repeated Timer.cupti calls
    # reuse the same compiled module inside MOESIM_CUPTI_BUILD_DIR.
    return load(
        name="moesim_cupti_activity_profiler",
        sources=[str(EXT_SOURCE)],
        extra_include_paths=extra_includes,
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=[str(cupti_lib), f"-Wl,-rpath,{lib_dir}"],
        build_directory=str(BUILD_DIR),
        verbose=False,
        with_cuda=False,
    )


_CUPTI_EXT: Any | None = None


def _get_extension() -> Any:
    global _CUPTI_EXT
    if _CUPTI_EXT is None:
        _CUPTI_EXT = _load_extension()
    return _CUPTI_EXT


def default_l2_flush_bytes(device: int | str | Any = 0) -> int:
    torch_mod = _require_torch()
    props = torch_mod.cuda.get_device_properties(device)
    l2_size = getattr(props, "l2_cache_size", 0) or 0
    return max(64 * 1024 * 1024, 2 * int(l2_size))


def clear_l2_cache(buffer: Any) -> None:
    """Read-displace L2, then synchronize before the next measured interval.

    A reduction over a buffer larger than L2 leaves mostly clean cache lines and
    avoids the dirty writeback pressure created by ``zero_()``/memset clears.
    This is a cold-ish cache preconditioner, not a hardware invalidate.
    """

    torch_mod = _require_torch()
    buffer.sum()
    torch_mod.cuda.synchronize(buffer.device)


def relative_sem(samples: list[float]) -> float:
    """Relative standard error of the mean: ``stdev/sqrt(n)/mean``.

    The convergence target for adaptive CUPTI profiling -- how precise the mean
    estimate is, as a fraction of the mean. Returns +inf until it is well-defined
    (n >= 2 and mean > 0) so callers keep sampling.
    """

    n = len(samples)
    if n < 2:
        return float("inf")
    mean = fmean(samples)
    if mean <= 0:
        return float("inf")
    return (stdev(samples) / sqrt(n)) / mean


def _records_from_dicts(raw_records: list[dict]) -> list[KernelRecord]:
    return [KernelRecord(**record) for record in raw_records]


def match_kernel_records(
    records: list[KernelRecord],
    kernel_name_contains: str | None = None,
) -> list[KernelRecord]:
    if not records:
        raise RuntimeError("No CUPTI kernel records captured for this iteration")

    if kernel_name_contains is not None:
        matches = [record for record in records if kernel_name_contains in record.name]
        if not matches:
            names = ", ".join(sorted({record.name for record in records}))
            raise RuntimeError(
                f"No kernel matched substring {kernel_name_contains!r}. Captured kernels: {names}"
            )
        return matches

    return records


def _ordered_records(records: list[KernelRecord]) -> list[KernelRecord]:
    """Return a deterministic launch order for one synchronized capture."""

    return sorted(
        records,
        key=lambda record: (record.start_ns, record.end_ns, record.correlation_id),
    )


def _prepare_launch_pattern(
    profiler: CuptiKernelProfiler,
    fn: Callable[[], object],
    *,
    clear_l2_before_run: bool,
    clear_l2_between_launches: bool,
    kernel_name_contains: str | None,
) -> _LaunchPattern:
    """Probe one callable launch and the inter-launch L2 clear separately.

    A multi-launch capture also records the clear-buffer kernels. Remembering
    the complete ordered patterns lets the measured burst be split by position,
    rather than guessing from implementation names that may be shared by an
    unrelated target kernel.
    """

    callable_records = _ordered_records(
        profiler.capture_once(
            fn,
            launches_per_run=1,
            clear_l2_before_run=clear_l2_before_run,
            clear_l2_between_launches=False,
        )
    )
    match_kernel_records(
        callable_records,
        kernel_name_contains=kernel_name_contains,
    )

    clear_records: list[KernelRecord] = []
    if clear_l2_between_launches:
        clear_records = _ordered_records(
            profiler.capture_once(
                lambda: profiler._clear_buffer.sum(),
                launches_per_run=1,
                clear_l2_before_run=False,
                clear_l2_between_launches=False,
            )
        )
        if not clear_records:
            raise RuntimeError("No CUPTI kernel records captured for the L2 clear")

    return _LaunchPattern(
        callable_kernel_names=tuple(record.name for record in callable_records),
        clear_kernel_names=tuple(record.name for record in clear_records),
    )


def _capture_launch_unit(
    profiler: CuptiKernelProfiler,
    fn: Callable[[], object],
    *,
    launches_per_run: int,
    clear_l2_before_run: bool,
    clear_l2_between_launches: bool,
    kernel_name_contains: str | None,
    launch_pattern: _LaunchPattern | None,
) -> tuple[list[float], set[str], list[int]]:
    """Capture one unit and return one duration/count per logical callable run."""

    records = _ordered_records(
        profiler.capture_once(
            fn,
            launches_per_run=launches_per_run,
            clear_l2_before_run=clear_l2_before_run,
            clear_l2_between_launches=clear_l2_between_launches,
        )
    )
    if launches_per_run == 1:
        matched = match_kernel_records(
            records,
            kernel_name_contains=kernel_name_contains,
        )
        return (
            [sum(record.duration_ns for record in matched) / 1e6],
            {record.name for record in matched},
            [len(matched)],
        )

    if launch_pattern is None:
        raise RuntimeError("multi-launch CUPTI capture requires a launch pattern")

    callable_count = len(launch_pattern.callable_kernel_names)
    clear_count = len(launch_pattern.clear_kernel_names)
    expected_count = launches_per_run * callable_count
    if clear_l2_between_launches:
        expected_count += (launches_per_run - 1) * clear_count
    if len(records) != expected_count:
        raise RuntimeError(
            "CUPTI multi-launch record count changed: "
            f"expected={expected_count}, captured={len(records)}"
        )

    cursor = 0
    per_launch_ms: list[float] = []
    matched_kernel_names: set[str] = set()
    matched_kernel_counts: list[int] = []
    for launch_index in range(launches_per_run):
        callable_records = records[cursor : cursor + callable_count]
        cursor += callable_count
        callable_names = tuple(record.name for record in callable_records)
        if callable_names != launch_pattern.callable_kernel_names:
            raise RuntimeError(
                "CUPTI callable kernel pattern changed inside the multi-launch "
                f"capture at logical launch {launch_index}"
            )
        matched = match_kernel_records(
            callable_records,
            kernel_name_contains=kernel_name_contains,
        )
        per_launch_ms.append(sum(record.duration_ns for record in matched) / 1e6)
        matched_kernel_names.update(record.name for record in matched)
        matched_kernel_counts.append(len(matched))

        if clear_l2_between_launches and launch_index + 1 < launches_per_run:
            clear_records = records[cursor : cursor + clear_count]
            cursor += clear_count
            clear_names = tuple(record.name for record in clear_records)
            if clear_names != launch_pattern.clear_kernel_names:
                raise RuntimeError(
                    "CUPTI L2-clear kernel pattern changed inside the multi-launch "
                    f"capture after logical launch {launch_index}"
                )

    return per_launch_ms, matched_kernel_names, matched_kernel_counts


def split_launch_series(
    records: list[KernelRecord],
    *,
    launch_pattern: _LaunchPattern,
    launches_per_run: int,
    clear_l2_between_launches: bool,
    kernel_name_contains: str | None = None,
) -> list[dict[str, object]]:
    """Split one multi-launch capture into a per-launch time series.

    Companion to ``_capture_launch_unit``: same ordered-pattern splitting, but it
    keeps each launch's start/end timestamps so a duration *trend* (per-launch
    start_ns + kernel-only duration) can be plotted, not just the aggregate mean.
    Interleaved L2-clear kernels are located by ``launch_pattern`` and excluded
    from each launch's reported duration. Used by ``profilers.trend``.
    """

    ordered = _ordered_records(records)
    callable_count = len(launch_pattern.callable_kernel_names)
    clear_count = len(launch_pattern.clear_kernel_names)
    expected_count = launches_per_run * callable_count
    if clear_l2_between_launches:
        expected_count += (launches_per_run - 1) * clear_count
    if len(ordered) != expected_count:
        raise RuntimeError(
            "CUPTI launch-series record count mismatch: "
            f"expected={expected_count}, captured={len(ordered)}"
        )

    cursor = 0
    series: list[dict[str, object]] = []
    for launch_index in range(launches_per_run):
        callable_records = ordered[cursor : cursor + callable_count]
        cursor += callable_count
        callable_names = tuple(record.name for record in callable_records)
        if callable_names != launch_pattern.callable_kernel_names:
            raise RuntimeError(
                "CUPTI callable kernel pattern changed inside the launch series "
                f"at logical launch {launch_index}"
            )
        matched = match_kernel_records(
            callable_records,
            kernel_name_contains=kernel_name_contains,
        )
        series.append(
            {
                "start_ns": min(record.start_ns for record in matched),
                "end_ns": max(record.end_ns for record in matched),
                "duration_ms": sum(record.duration_ns for record in matched) / 1e6,
                "kernel_count": len(matched),
                "kernel_names": " | ".join(dict.fromkeys(record.name for record in matched)),
            }
        )

        if clear_l2_between_launches and launch_index + 1 < launches_per_run:
            clear_records = ordered[cursor : cursor + clear_count]
            cursor += clear_count
            clear_names = tuple(record.name for record in clear_records)
            if clear_names != launch_pattern.clear_kernel_names:
                raise RuntimeError(
                    "CUPTI L2-clear kernel pattern changed inside the launch series "
                    f"after logical launch {launch_index}"
                )

    return series


class CuptiKernelProfiler:
    """Reusable CUPTI profiler for CUDA callables that launch kernels."""

    def __init__(
        self,
        *,
        device: int | str | Any = "cuda",
        clear_l2_bytes: int | None = None,
    ) -> None:
        torch_mod = _require_torch()
        if not torch_mod.cuda.is_available():
            raise RuntimeError("CUDA is required for CUPTI kernel profiling")

        self.device = torch_mod.device(device)
        if self.device.type != "cuda":
            raise ValueError(f"Expected a CUDA device, got {self.device}")

        self._profiler = _get_extension().CuptiKernelActivityProfiler()
        self.clear_l2_bytes = (
            default_l2_flush_bytes(self.device) if clear_l2_bytes is None else clear_l2_bytes
        )
        self._clear_buffer = torch_mod.ones(
            self.clear_l2_bytes // 4,
            device=self.device,
            dtype=torch_mod.float32,
        )

    def clear_l2_cache(self) -> None:
        clear_l2_cache(self._clear_buffer)

    def _run_launch_burst(
        self,
        fn: Callable[[], object],
        *,
        launches_per_run: int,
        clear_l2_between_launches: bool,
    ) -> None:
        # Only fires when launches_per_run > 1 (it cools L2 between the launches
        # batched into one capture window); with the usual launches_per_run == 1
        # it is a no-op and clear_l2_before_run already gives each launch cold L2.
        # Caveat: this read-displacement reduction runs INSIDE the capture
        # window. Multi-launch parsing identifies its exact record pattern and
        # excludes it from the callable's reported kernel time.
        for launch_idx in range(launches_per_run):
            fn()
            if clear_l2_between_launches and launch_idx + 1 < launches_per_run:
                self._clear_buffer.sum()

    def capture_once(
        self,
        fn: Callable[[], object],
        *,
        launches_per_run: int = 1,
        clear_l2_before_run: bool = True,
        clear_l2_between_launches: bool = True,
    ) -> list[KernelRecord]:
        if launches_per_run <= 0:
            raise ValueError("launches_per_run must be positive")

        if clear_l2_before_run:
            self.clear_l2_cache()

        self._profiler.start()
        self._run_launch_burst(
            fn,
            launches_per_run=launches_per_run,
            clear_l2_between_launches=clear_l2_between_launches,
        )
        _require_torch().cuda.synchronize(self.device)
        return _records_from_dicts(self._profiler.stop())

    def profile(
        self,
        fn: Callable[[], object],
        *,
        num_warmup: int = 10,
        num_iter: int = 100,
        launches_per_run: int = 1,
        clear_l2_before_run: bool = True,
        clear_l2_between_launches: bool = True,
        kernel_name_contains: str | None = None,
    ) -> KernelProfileSummary:
        if launches_per_run <= 0:
            raise ValueError("launches_per_run must be positive")

        torch_mod = _require_torch()
        for _ in range(num_warmup):
            if clear_l2_before_run:
                self.clear_l2_cache()
            self._run_launch_burst(
                fn,
                launches_per_run=launches_per_run,
                clear_l2_between_launches=clear_l2_between_launches,
            )
            torch_mod.cuda.synchronize(self.device)

        per_iter_ms: list[float] = []
        matched_kernel_names: set[str] = set()
        matched_kernel_count_per_run: list[int] = []
        for _ in range(num_iter):
            records = self.capture_once(
                fn,
                launches_per_run=launches_per_run,
                clear_l2_before_run=clear_l2_before_run,
                clear_l2_between_launches=clear_l2_between_launches,
            )
            matched = match_kernel_records(records, kernel_name_contains=kernel_name_contains)
            per_iter_ms.append(sum(record.duration_ns for record in matched) / 1e6)
            matched_kernel_names.update(record.name for record in matched)
            matched_kernel_count_per_run.append(len(matched))

        return KernelProfileSummary(
            matched_kernel_names=sorted(matched_kernel_names),
            mean_ms=fmean(per_iter_ms),
            median_ms=median(per_iter_ms),
            min_ms=min(per_iter_ms),
            max_ms=max(per_iter_ms),
            num_warmup=num_warmup,
            num_iter=num_iter,
            launches_per_run=launches_per_run,
            clear_l2_bytes=self.clear_l2_bytes,
            clear_l2_between_launches=clear_l2_between_launches,
            per_iter_ms=per_iter_ms,
            matched_kernel_count_per_run=matched_kernel_count_per_run,
        )

    def profile_until_converged(
        self,
        fn: Callable[[], object],
        *,
        num_warmup: int = 0,
        batch: int = 10,
        min_duration_ms: int = 2_000,
        min_iter: int = 20,
        max_iter: int = 5_000_000,
        tol: float = 0.01,
        launches_per_run: int = 1,
        clear_l2_before_run: bool = True,
        clear_l2_between_launches: bool = True,
        kernel_name_contains: str | None = None,
    ) -> KernelProfileSummary:
        """Sample per-kernel timings until the mean estimate converges, then stop.

        CUPTI records each launch's true kernel duration. We sample in ``batch``
        bursts and stop only after the matched kernels have accumulated
        ``min_duration_ms`` of GPU active time and ``relative_sem`` (the mean's
        relative standard error) drops below ``tol``. ``max_iter`` is a hard
        safety cap: reaching it before both conditions raises rather than
        returning a measurement shorter than the requested duration.

        ``num_warmup`` defaults to 0: each captured launch is L2-flushed and
        independent, and convergence absorbs a cold first sample on its own.
        Set ``min_duration_ms=0`` only for a deliberately duration-free diagnostic.
        """

        if batch <= 0:
            raise ValueError("batch must be positive")
        if min_duration_ms < 0:
            raise ValueError("min_duration_ms must be >= 0")
        if min_iter < 2:
            raise ValueError("min_iter must be >= 2")
        if max_iter < min_iter:
            raise ValueError("max_iter must be >= min_iter")
        if max_iter < launches_per_run:
            raise ValueError("max_iter must cover at least one launches_per_run unit")

        torch_mod = _require_torch()
        for _ in range(num_warmup):
            if clear_l2_before_run:
                self.clear_l2_cache()
            self._run_launch_burst(
                fn,
                launches_per_run=launches_per_run,
                clear_l2_between_launches=clear_l2_between_launches,
            )
            torch_mod.cuda.synchronize(self.device)

        per_iter_ms: list[float] = []
        matched_kernel_names: set[str] = set()
        matched_kernel_count_per_run: list[int] = []
        launch_pattern = None
        if launches_per_run > 1:
            launch_pattern = _prepare_launch_pattern(
                self,
                fn,
                clear_l2_before_run=clear_l2_before_run,
                clear_l2_between_launches=clear_l2_between_launches,
                kernel_name_contains=kernel_name_contains,
            )
        accumulated_kernel_ms = 0.0
        while len(per_iter_ms) < max_iter:
            for _ in range(batch):
                if len(per_iter_ms) + launches_per_run > max_iter:
                    break
                (
                    unit_per_launch_ms,
                    unit_kernel_names,
                    unit_kernel_counts,
                ) = _capture_launch_unit(
                    self,
                    fn,
                    launches_per_run=launches_per_run,
                    clear_l2_before_run=clear_l2_before_run,
                    clear_l2_between_launches=clear_l2_between_launches,
                    kernel_name_contains=kernel_name_contains,
                    launch_pattern=launch_pattern,
                )
                per_iter_ms.extend(unit_per_launch_ms)
                accumulated_kernel_ms += sum(unit_per_launch_ms)
                matched_kernel_names.update(unit_kernel_names)
                matched_kernel_count_per_run.extend(unit_kernel_counts)
            duration_reached = accumulated_kernel_ms >= min_duration_ms
            converged = len(per_iter_ms) >= min_iter and relative_sem(per_iter_ms) < tol
            if duration_reached and converged:
                break

        duration_reached = accumulated_kernel_ms >= min_duration_ms
        converged = len(per_iter_ms) >= min_iter and relative_sem(per_iter_ms) < tol
        if not duration_reached or not converged:
            raise RuntimeError(
                "CUPTI reached max_iter before satisfying the measurement "
                f"contract: iterations={len(per_iter_ms)}, "
                f"accumulated_kernel_ms={accumulated_kernel_ms:.6f}, "
                f"min_duration_ms={min_duration_ms}, "
                f"relative_sem={relative_sem(per_iter_ms):.6g}, tol={tol}"
            )

        return KernelProfileSummary(
            matched_kernel_names=sorted(matched_kernel_names),
            mean_ms=fmean(per_iter_ms),
            median_ms=median(per_iter_ms),
            min_ms=min(per_iter_ms),
            max_ms=max(per_iter_ms),
            num_warmup=num_warmup,
            num_iter=len(per_iter_ms),
            launches_per_run=launches_per_run,
            clear_l2_bytes=self.clear_l2_bytes,
            clear_l2_between_launches=clear_l2_between_launches,
            per_iter_ms=per_iter_ms,
            matched_kernel_count_per_run=matched_kernel_count_per_run,
        )

    def profile_for_duration(
        self,
        fn: Callable[[], object],
        *,
        num_warmup: int = 0,
        estimate_iter: int = 10,
        min_duration_ms: int = 2_000,
        min_iter: int = 20,
        max_iter: int = 50_000,
        clear_l2_before_run: bool = True,
        clear_l2_between_launches: bool = True,
        kernel_name_contains: str | None = None,
    ) -> KernelProfileSummary:
        """Estimate a launch count, then measure it in one CUPTI window.

        The estimator records ``estimate_iter`` real callable launches and uses
        their kernel-only mean to size ``ceil(min_duration_ms / estimate_ms)``.
        The formal measurement is exactly one uninterrupted CUPTI activity
        window containing that many logical launches. This keeps the requested
        active-time budget without injecting periodic capture restarts into the
        workload's power/clock state.

        By default every logical launch starts after a read-only reduction over
        the L2-displacement buffer. The clear kernels are captured for ordering
        validation but excluded from the callable's reported duration. Explicit
        flags remain available for warm-cache diagnostics.
        """

        if num_warmup < 0:
            raise ValueError("num_warmup must be >= 0")
        if estimate_iter <= 0:
            raise ValueError("estimate_iter must be positive")
        if min_duration_ms < 0:
            raise ValueError("min_duration_ms must be >= 0")
        if min_iter <= 0:
            raise ValueError("min_iter must be positive")
        if max_iter < min_iter:
            raise ValueError("max_iter must be >= min_iter")

        torch_mod = _require_torch()
        for _ in range(num_warmup):
            self._run_launch_burst(
                fn,
                launches_per_run=1,
                clear_l2_between_launches=clear_l2_between_launches,
            )
        if num_warmup:
            torch_mod.cuda.synchronize(self.device)

        launch_pattern = _prepare_launch_pattern(
            self,
            fn,
            clear_l2_before_run=clear_l2_before_run,
            clear_l2_between_launches=clear_l2_between_launches,
            kernel_name_contains=kernel_name_contains,
        )
        estimate_ms, _, _ = _capture_launch_unit(
            self,
            fn,
            launches_per_run=estimate_iter,
            clear_l2_before_run=clear_l2_before_run,
            clear_l2_between_launches=clear_l2_between_launches,
            kernel_name_contains=kernel_name_contains,
            launch_pattern=launch_pattern,
        )
        estimate_mean_ms = fmean(estimate_ms)
        if estimate_mean_ms <= 0:
            raise RuntimeError(f"CUPTI duration estimate must be positive, got {estimate_mean_ms}")

        estimated_iter = max(ceil(min_duration_ms / estimate_mean_ms), min_iter)
        # ``max_iter`` is an execution-budget cap, not an error threshold. Very
        # short kernels can otherwise turn a modest active-time target into
        # millions of cold-L2 launches and CUPTI records. The capped sample is
        # still a valid per-launch mean; it simply stops before exhausting the
        # requested active-time budget.
        formal_iter = min(estimated_iter, max_iter)

        per_iter_ms, matched_kernel_names, matched_kernel_count_per_run = _capture_launch_unit(
            self,
            fn,
            launches_per_run=formal_iter,
            clear_l2_before_run=clear_l2_before_run,
            clear_l2_between_launches=clear_l2_between_launches,
            kernel_name_contains=kernel_name_contains,
            launch_pattern=launch_pattern,
        )
        return KernelProfileSummary(
            matched_kernel_names=sorted(matched_kernel_names),
            mean_ms=fmean(per_iter_ms),
            median_ms=median(per_iter_ms),
            min_ms=min(per_iter_ms),
            max_ms=max(per_iter_ms),
            num_warmup=num_warmup,
            num_iter=len(per_iter_ms),
            launches_per_run=formal_iter,
            clear_l2_bytes=self.clear_l2_bytes,
            clear_l2_between_launches=clear_l2_between_launches,
            per_iter_ms=per_iter_ms,
            matched_kernel_count_per_run=matched_kernel_count_per_run,
        )

    def profile_series_for_duration(
        self,
        fn: Callable[[], object],
        *,
        duration_s: float,
        estimate_iter: int = 10,
        min_iter: int = 20,
        max_iter: int = 5_000_000,
        clear_l2_before_run: bool = True,
        clear_l2_between_launches: bool = True,
        kernel_name_contains: str | None = None,
        capture_hook: Any | None = None,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        """Capture a per-launch duration *series* over one ~``duration_s`` window.

        This is the trend-diagnostic sibling of ``profile_for_duration``: it sizes
        the launch count the same way (estimate from ``estimate_iter`` real
        launches, then ``ceil(duration_ms / estimate_ms)``) and records exactly
        that many logical launches in one uninterrupted CUPTI window, but returns
        every launch's ``start_ns`` + kernel-only ``duration_ms`` instead of a
        single reduced mean. ``capture_hook`` is an optional context manager
        wrapping only the formal capture (the telemetry sampler enters here so its
        wall-clock origin lines up with the first launch).

        ``clear_l2_between_launches=False`` is the warm continuous window that
        surfaces sustained power/clock drift; the default clears L2 before every
        launch, matching the ``profile.db`` measurement character.
        """

        if duration_s <= 0:
            raise ValueError("duration_s must be positive")

        torch_mod = _require_torch()
        launch_pattern = _prepare_launch_pattern(
            self,
            fn,
            clear_l2_before_run=clear_l2_before_run,
            clear_l2_between_launches=clear_l2_between_launches,
            kernel_name_contains=kernel_name_contains,
        )
        estimate_ms, _, _ = _capture_launch_unit(
            self,
            fn,
            launches_per_run=estimate_iter,
            clear_l2_before_run=clear_l2_before_run,
            clear_l2_between_launches=clear_l2_between_launches,
            kernel_name_contains=kernel_name_contains,
            launch_pattern=launch_pattern,
        )
        estimate_mean_ms = fmean(estimate_ms)
        if estimate_mean_ms <= 0:
            raise RuntimeError(f"CUPTI duration estimate must be positive, got {estimate_mean_ms}")

        duration_ms = duration_s * 1000.0
        formal_iter = max(ceil(duration_ms / estimate_mean_ms), min_iter)
        if formal_iter > max_iter:
            raise RuntimeError(
                "CUPTI estimated launch count exceeds max_iter: "
                f"estimate_mean_ms={estimate_mean_ms:.9f}, duration_s={duration_s}, "
                f"formal_iter={formal_iter}, max_iter={max_iter}"
            )

        hook = capture_hook if capture_hook is not None else nullcontext()
        with hook:
            records = self.capture_once(
                fn,
                launches_per_run=formal_iter,
                clear_l2_before_run=clear_l2_before_run,
                clear_l2_between_launches=clear_l2_between_launches,
            )
        series = split_launch_series(
            records,
            launch_pattern=launch_pattern,
            launches_per_run=formal_iter,
            clear_l2_between_launches=clear_l2_between_launches,
            kernel_name_contains=kernel_name_contains,
        )
        first_start_ns = cast(int, series[0]["start_ns"])
        for sample_index, sample in enumerate(series):
            sample["sample_index"] = sample_index
            sample["start_s"] = (cast(int, sample["start_ns"]) - first_start_ns) / 1e9

        metadata: dict[str, object] = {
            "gpu": torch_mod.cuda.get_device_name(self.device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "requested_duration_s": duration_s,
            "estimate_mean_ms": estimate_mean_ms,
            "launch_count": formal_iter,
            "cupti_record_count": len(records),
            "callable_kernel_names": list(launch_pattern.callable_kernel_names),
            "clear_kernel_names": list(launch_pattern.clear_kernel_names),
            "clear_l2_before_run": clear_l2_before_run,
            "clear_l2_between_launches": clear_l2_between_launches,
            "clear_l2_bytes": self.clear_l2_bytes,
            "gpu_kernel_span_s": (cast(int, series[-1]["end_ns"]) - first_start_ns) / 1e9,
        }
        return series, metadata


def profile_kernel(
    fn: Callable[[], object],
    *,
    device: int | str | Any = "cuda",
    num_warmup: int = 10,
    num_iter: int = 100,
    launches_per_run: int = 1,
    clear_l2_bytes: int | None = None,
    clear_l2_before_run: bool = True,
    clear_l2_between_launches: bool = True,
    kernel_name_contains: str | None = None,
) -> KernelProfileSummary:
    profiler = CuptiKernelProfiler(device=device, clear_l2_bytes=clear_l2_bytes)
    return profiler.profile(
        fn,
        num_warmup=num_warmup,
        num_iter=num_iter,
        launches_per_run=launches_per_run,
        clear_l2_before_run=clear_l2_before_run,
        clear_l2_between_launches=clear_l2_between_launches,
        kernel_name_contains=kernel_name_contains,
    )


def profile_kernel_until_converged(
    fn: Callable[[], object],
    *,
    device: int | str | Any = "cuda",
    num_warmup: int = 0,
    batch: int = 10,
    min_duration_ms: int = 2_000,
    min_iter: int = 20,
    max_iter: int = 5_000_000,
    tol: float = 0.01,
    launches_per_run: int = 1,
    clear_l2_bytes: int | None = None,
    clear_l2_before_run: bool = True,
    clear_l2_between_launches: bool = True,
    kernel_name_contains: str | None = None,
) -> KernelProfileSummary:
    profiler = CuptiKernelProfiler(device=device, clear_l2_bytes=clear_l2_bytes)
    return profiler.profile_until_converged(
        fn,
        num_warmup=num_warmup,
        batch=batch,
        min_duration_ms=min_duration_ms,
        min_iter=min_iter,
        max_iter=max_iter,
        tol=tol,
        launches_per_run=launches_per_run,
        clear_l2_before_run=clear_l2_before_run,
        clear_l2_between_launches=clear_l2_between_launches,
        kernel_name_contains=kernel_name_contains,
    )


def profile_kernel_for_duration(
    fn: Callable[[], object],
    *,
    device: int | str | Any = "cuda",
    num_warmup: int = 0,
    estimate_iter: int = 10,
    min_duration_ms: int = 2_000,
    min_iter: int = 20,
    max_iter: int = 50_000,
    clear_l2_bytes: int | None = None,
    clear_l2_before_run: bool = True,
    clear_l2_between_launches: bool = True,
    kernel_name_contains: str | None = None,
) -> KernelProfileSummary:
    profiler = CuptiKernelProfiler(device=device, clear_l2_bytes=clear_l2_bytes)
    return profiler.profile_for_duration(
        fn,
        num_warmup=num_warmup,
        estimate_iter=estimate_iter,
        min_duration_ms=min_duration_ms,
        min_iter=min_iter,
        max_iter=max_iter,
        clear_l2_before_run=clear_l2_before_run,
        clear_l2_between_launches=clear_l2_between_launches,
        kernel_name_contains=kernel_name_contains,
    )
