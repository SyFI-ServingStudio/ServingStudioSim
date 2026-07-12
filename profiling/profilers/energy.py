"""Energy helper for L1a runners.

The detailed design makes energy a first-class metric. This helper keeps that
field wired without making NVML a hard dependency for development machines.

Agent note: ``Energy.perf`` always sizes its NVML window from ``min_duration_ms``
(a per-process timing estimate), so the iteration count is non-deterministic
across ranks. Multi-GPU / collective runs must not rely on this path -- see
``warn_if_multi_gpu_duration_mode``.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from profiling.profilers._duration import (
    DEFAULT_MIN_DURATION_MS,
    DEFAULT_MIN_REP,
    iters_for_duration,
    warn_if_multi_gpu_duration_mode,
)

_TOTAL_ENERGY_SUPPORTED_BY_GPU: dict[str | int, bool] = {}
_MAX_CLEAN_RESTARTS = 3


@dataclass(frozen=True)
class _EnergyAttempt:
    energy_j: float
    executed_iters: int
    elapsed_s: float
    failed: bool = False


class Energy:
    @staticmethod
    def perf(
        fn: Callable[[], object],
        *,
        warmup: int = 10,
        min_duration_ms: int | None = None,
        min_rep: int | None = None,
        per_iter_time_ms: float | None = None,
    ) -> float:
        # Time-centric like Timer: poll for at least min_duration_ms, but never
        # fewer than min_rep iterations (the larger wins).
        min_duration_ms = (
            DEFAULT_MIN_DURATION_MS if min_duration_ms is None else min_duration_ms
        )
        min_rep = DEFAULT_MIN_REP if min_rep is None else min_rep
        warn_if_multi_gpu_duration_mode("Energy.perf")
        for _ in range(warmup):
            fn()

        try:
            import pynvml  # type: ignore[import-not-found]
            import torch
        except ImportError:
            return 0.0

        if not torch.cuda.is_available():
            return 0.0

        pynvml.nvmlInit()
        device_idx = torch.cuda.current_device()
        try:
            handle = _nvml_handle_for_cuda_device(pynvml, torch, device_idx)
        except pynvml.NVMLError:
            return 0.0
        if handle is None:
            return 0.0

        # Agent note: runners should pass the real Timer result so Energy uses
        # the same kernel timing contract as the recorded row. The internal
        # one-call estimate only exists for diagnostic callers without timing.
        if per_iter_time_ms is None:
            planned_iters = Energy._estimate_iters(
                fn,
                min_duration_ms,
                synchronize=torch.cuda.synchronize,
            )
        else:
            planned_iters = iters_for_duration(min_duration_ms, per_iter_time_ms)
        planned_iters = max(planned_iters, min_rep)
        try:
            if _supports_total_energy(pynvml, handle, device_idx):
                return _measure_by_total_energy_counter(
                    pynvml,
                    torch,
                    handle,
                    fn,
                    planned_iters,
                    min_duration_ms,
                )
            return _measure_by_power_polling(
                pynvml,
                torch,
                handle,
                fn,
                planned_iters,
                min_duration_ms,
            )
        except pynvml.NVMLError:
            return 0.0

    @staticmethod
    def _estimate_iters(
        fn: Callable[[], object],
        min_duration_ms: int,
        *,
        synchronize: Callable[[], object] | None = None,
    ) -> int:
        if synchronize is not None:
            synchronize()
        start_s = time.perf_counter()
        fn()
        if synchronize is not None:
            synchronize()
        elapsed_ms = max((time.perf_counter() - start_s) * 1000.0, 0.01)
        return iters_for_duration(min_duration_ms, elapsed_ms)


def _nvml_handle_for_cuda_device(
    pynvml: Any,
    torch: Any,
    logical_device_idx: int,
) -> object | None:
    """Resolve Torch's current CUDA device to the same physical NVML device.

    Local profiling workers commonly see physical GPU N as logical CUDA device
    0 through ``CUDA_VISIBLE_DEVICES=N``. Torch's device UUID survives that
    remapping, while an NVML index does not. If an older Torch build cannot
    expose the UUID, only use the logical index when no visibility remapping is
    present; returning no handle is safer than recording another GPU's energy.
    """

    get_device_properties = getattr(torch.cuda, "get_device_properties", None)
    if get_device_properties is not None:
        cuda_uuid = getattr(get_device_properties(logical_device_idx), "uuid", None)
        nvml_uuid = _normalize_nvml_uuid(cuda_uuid)
        if nvml_uuid is not None:
            try:
                return pynvml.nvmlDeviceGetHandleByUUID(nvml_uuid)
            except AttributeError:
                # Old pynvml without UUID lookup may still be safe in an
                # unmasked single-GPU process, handled by the fallback below.
                pass

    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        return None
    return pynvml.nvmlDeviceGetHandleByIndex(logical_device_idx)


def _normalize_nvml_uuid(cuda_uuid: object) -> str | None:
    if cuda_uuid is None:
        return None
    if isinstance(cuda_uuid, bytes):
        uuid_text = cuda_uuid.decode("utf-8", errors="strict")
    else:
        uuid_text = str(cuda_uuid)
    uuid_text = uuid_text.strip()
    if not uuid_text:
        return None
    if uuid_text.startswith(("GPU-", "MIG-")):
        return uuid_text
    return f"GPU-{uuid_text}"


def _supports_total_energy(pynvml: Any, handle: object, device_idx: int) -> bool:
    cache_key = _gpu_cache_key(pynvml, handle, device_idx)
    if cache_key in _TOTAL_ENERGY_SUPPORTED_BY_GPU:
        return _TOTAL_ENERGY_SUPPORTED_BY_GPU[cache_key]

    try:
        pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        supported = True
    except pynvml.NVMLError as exc:
        if not _is_not_supported_error(pynvml, exc):
            raise
        supported = False

    # Agent note: cache per GPU because heterogeneous nodes may mix datacenter
    # cards with consumer cards that lack NVML total-energy counters.
    _TOTAL_ENERGY_SUPPORTED_BY_GPU[cache_key] = supported
    return supported


def _measure_by_total_energy_counter(
    pynvml: Any,
    torch: Any,
    handle: object,
    fn: Callable[[], object],
    planned_iters: int,
    min_duration_ms: int,
) -> float:
    return _measure_with_clean_restarts(
        lambda attempt_iters: _measure_total_energy_counter_once(
            pynvml,
            torch,
            handle,
            fn,
            attempt_iters,
        ),
        planned_iters,
        min_duration_ms,
    )


def _measure_by_power_polling(
    pynvml: Any,
    torch: Any,
    handle: object,
    fn: Callable[[], object],
    planned_iters: int,
    min_duration_ms: int,
) -> float:
    return _measure_with_clean_restarts(
        lambda attempt_iters: _measure_power_polling_once(
            pynvml,
            torch,
            handle,
            fn,
            attempt_iters,
        ),
        planned_iters,
        min_duration_ms,
    )


def _measure_with_clean_restarts(
    measure_attempt_fn: Callable[[int], _EnergyAttempt],
    planned_iters: int,
    min_duration_ms: int,
) -> float:
    target_s = max(float(min_duration_ms) / 1000.0, 0.0)
    attempt_iters = max(planned_iters, 1)
    last_attempt: _EnergyAttempt | None = None

    for _ in range(_MAX_CLEAN_RESTARTS + 1):
        last_attempt = measure_attempt_fn(attempt_iters)
        if last_attempt.failed or target_s <= 0.0 or last_attempt.elapsed_s >= target_s:
            return _energy_per_iter(last_attempt)

        # Agent note: if Timer's recorded time overestimates the warm-cache
        # energy loop speed, discard the short NVML window and retry from a
        # fresh counter/poller boundary with the aggregate observed rate.
        observed_per_iter_ms = max(
            last_attempt.elapsed_s * 1000.0 / max(last_attempt.executed_iters, 1),
            0.001,
        )
        attempt_iters = max(
            iters_for_duration(min_duration_ms, observed_per_iter_ms),
            last_attempt.executed_iters + 1,
        )

    if last_attempt is None:
        return 0.0
    return _energy_per_iter(last_attempt)


def _measure_total_energy_counter_once(
    pynvml: Any,
    torch: Any,
    handle: object,
    fn: Callable[[], object],
    attempt_iters: int,
) -> _EnergyAttempt:
    torch.cuda.synchronize()
    e0_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    start_s = time.perf_counter()
    for _ in range(attempt_iters):
        fn()
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - start_s
    e1_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    return _EnergyAttempt(
        energy_j=max(float(e1_mj - e0_mj), 0.0) / 1000.0,
        executed_iters=attempt_iters,
        elapsed_s=elapsed_s,
    )


def _measure_power_polling_once(
    pynvml: Any,
    torch: Any,
    handle: object,
    fn: Callable[[], object],
    attempt_iters: int,
) -> _EnergyAttempt:
    torch.cuda.synchronize()
    with _NvmlPoller(pynvml, handle, poll_hz=100) as poller:
        for _ in range(attempt_iters):
            fn()
        torch.cuda.synchronize()

    return _EnergyAttempt(
        energy_j=max(poller.avg_watts * poller.elapsed_s, 0.0),
        executed_iters=attempt_iters,
        elapsed_s=poller.elapsed_s,
        failed=poller.error is not None,
    )


def _energy_per_iter(attempt: _EnergyAttempt) -> float:
    if attempt.failed:
        return 0.0
    return max(attempt.energy_j, 0.0) / max(attempt.executed_iters, 1)


def _gpu_cache_key(pynvml: Any, handle: object, device_idx: int) -> str | int:
    try:
        uuid = pynvml.nvmlDeviceGetUUID(handle)
    except AttributeError:
        return device_idx
    except pynvml.NVMLError:
        return device_idx
    if isinstance(uuid, bytes):
        return uuid.decode("utf-8", errors="replace")
    return str(uuid)


def _is_not_supported_error(pynvml: Any, exc: BaseException) -> bool:
    unsupported_value = getattr(pynvml, "NVML_ERROR_NOT_SUPPORTED", None)
    return getattr(exc, "value", None) == unsupported_value or exc.__class__.__name__.endswith(
        "NotSupported"
    )


class _NvmlPoller:
    """100Hz NVML power sampler for GPUs without cumulative energy counters."""

    def __init__(self, pynvml: Any, handle: object, *, poll_hz: int) -> None:
        self._pynvml = pynvml
        self._handle = handle
        self._period_s = 1.0 / poll_hz
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples_watts: list[float] = []
        self._start_s: float | None = None
        self._end_s: float | None = None
        self.error: BaseException | None = None

    def __enter__(self) -> _NvmlPoller:
        self._start_s = time.perf_counter()
        self._sample_once()
        self._thread = threading.Thread(
            target=self._poll_until_stopped,
            name="vibesim-nvml-power-poller",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        del exc_type, exc, traceback
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.2, self._period_s * 2))
        self._sample_once()
        self._end_s = time.perf_counter()

    @property
    def avg_watts(self) -> float:
        if not self._samples_watts:
            return 0.0
        return sum(self._samples_watts) / len(self._samples_watts)

    @property
    def elapsed_s(self) -> float:
        end_s = self._end_s if self._end_s is not None else time.perf_counter()
        start_s = self._start_s if self._start_s is not None else end_s
        return max(end_s - start_s, 0.0)

    def _poll_until_stopped(self) -> None:
        while not self._stop_event.wait(self._period_s):
            self._sample_once()

    def _sample_once(self) -> None:
        if self.error is not None:
            return
        try:
            power_mw = self._pynvml.nvmlDeviceGetPowerUsage(self._handle)
        except self._pynvml.NVMLError as exc:
            self.error = exc
            return
        self._samples_watts.append(float(power_mw) / 1000.0)
