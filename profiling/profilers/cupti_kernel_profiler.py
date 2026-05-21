"""Reusable CUPTI-based kernel profiler for CUDA workloads.

This is the L1-local copy of the legacy ``ref/profile/cupti_kernel_profiler.py``
helper. ``Timer.cupti`` depends on this module for kernel-only timings, while
runner files stay small and only choose the timer method.
"""

from __future__ import annotations

import os
import site
from collections.abc import Callable
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from statistics import fmean, median, stdev
from typing import Any

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
    # Launch a write kernel over a buffer larger than L2, then sync so the next
    # measured interval starts from a cold-ish cache state.
    torch_mod = _require_torch()
    buffer.zero_()
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
            default_l2_flush_bytes(self.device)
            if clear_l2_bytes is None
            else clear_l2_bytes
        )
        self._clear_buffer = torch_mod.empty(
            self.clear_l2_bytes // 4,
            device=self.device,
            dtype=torch_mod.int32,
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
        # Caveat: this flush runs INSIDE the capture window, so with
        # launches_per_run > 1 it is captured by CUPTI -- pass a kernel_name filter
        # there so the zero_() kernel is not summed into the measured time.
        for launch_idx in range(launches_per_run):
            fn()
            if clear_l2_between_launches and launch_idx + 1 < launches_per_run:
                self._clear_buffer.zero_()

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
        min_iter: int = 20,
        max_iter: int = 500,
        tol: float = 0.01,
        launches_per_run: int = 1,
        clear_l2_before_run: bool = True,
        clear_l2_between_launches: bool = True,
        kernel_name_contains: str | None = None,
    ) -> KernelProfileSummary:
        """Sample per-kernel timings until the mean estimate converges, then stop.

        CUPTI records each launch's true kernel duration, so a stable kernel
        converges in a few dozen samples -- far cheaper than a fixed huge count
        whose per-launch CUPTI overhead dominates. We sample in ``batch`` bursts
        and stop once ``relative_sem`` (the mean's relative standard error) drops
        below ``tol``, bounded by ``[min_iter, max_iter]``. ``num_warmup`` defaults
        to 0: each captured launch is L2-flushed and independent, and convergence
        absorbs a cold first sample on its own.
        """

        if batch <= 0:
            raise ValueError("batch must be positive")
        if min_iter < 2:
            raise ValueError("min_iter must be >= 2")
        if max_iter < min_iter:
            raise ValueError("max_iter must be >= min_iter")

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
        while len(per_iter_ms) < max_iter:
            for _ in range(batch):
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
                if len(per_iter_ms) >= max_iter:
                    break
            if len(per_iter_ms) >= min_iter and relative_sem(per_iter_ms) < tol:
                break

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


def profile_kernel(
    fn: Callable[[], object],
    *,
    device: int | str | Any = "cuda",
    num_warmup: int = 10,
    num_iter: int = 100,
    launches_per_run: int = 1,
    clear_l2_bytes: int | None = None,
    clear_l2_before_run: bool = True,
    clear_l2_between_launches: bool = False,
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
    min_iter: int = 20,
    max_iter: int = 500,
    tol: float = 0.01,
    launches_per_run: int = 1,
    clear_l2_bytes: int | None = None,
    clear_l2_before_run: bool = True,
    clear_l2_between_launches: bool = False,
    kernel_name_contains: str | None = None,
) -> KernelProfileSummary:
    profiler = CuptiKernelProfiler(device=device, clear_l2_bytes=clear_l2_bytes)
    return profiler.profile_until_converged(
        fn,
        num_warmup=num_warmup,
        batch=batch,
        min_iter=min_iter,
        max_iter=max_iter,
        tol=tol,
        launches_per_run=launches_per_run,
        clear_l2_before_run=clear_l2_before_run,
        clear_l2_between_launches=clear_l2_between_launches,
        kernel_name_contains=kernel_name_contains,
    )
