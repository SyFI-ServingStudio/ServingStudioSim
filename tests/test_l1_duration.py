from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from profiling.profilers._duration import (
    iters_for_duration,
    warn_if_multi_gpu_duration_mode,
)
from profiling.profilers.energy import ENERGY_ENV, Energy


def test_iters_for_duration_rounds_up():
    assert iters_for_duration(1000, 0.5) == 2000
    assert iters_for_duration(1000, 3.0) == 334
    assert iters_for_duration(0, 3.0) == 1


def test_iters_for_duration_rejects_nonpositive():
    with pytest.raises(ValueError):
        iters_for_duration(1000, 0.0)


def test_warn_helper_silent_on_single_rank(monkeypatch: pytest.MonkeyPatch):
    fake_torch = SimpleNamespace(
        distributed=SimpleNamespace(
            is_available=lambda: True,
            is_initialized=lambda: True,
            get_world_size=lambda: 1,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_multi_gpu_duration_mode("ctx")  # no process group > 1 => silent


def test_warn_helper_silent_without_distributed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(distributed=None)
    )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_multi_gpu_duration_mode("ctx")


def test_warn_helper_fires_on_multi_rank(monkeypatch: pytest.MonkeyPatch):
    fake_torch = SimpleNamespace(
        distributed=SimpleNamespace(
            is_available=lambda: True,
            is_initialized=lambda: True,
            get_world_size=lambda: 8,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.warns(RuntimeWarning, match="non-deterministic across the 8 ranks"):
        warn_if_multi_gpu_duration_mode("ctx")


def test_energy_warns_multi_gpu(monkeypatch: pytest.MonkeyPatch):
    # The window is opt-in; the warning is only reachable once it is asked for.
    monkeypatch.setenv(ENERGY_ENV, "1")
    # cuda unavailable => Energy.perf bails after warning without touching NVML.
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        distributed=SimpleNamespace(
            is_available=lambda: True,
            is_initialized=lambda: True,
            get_world_size=lambda: 2,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.warns(RuntimeWarning, match="non-deterministic across the 2 ranks"):
        assert Energy.perf(lambda: None, warmup=0) == 0.0


def test_empty_energy_variable_counts_as_unset(monkeypatch: pytest.MonkeyPatch):
    """`NAME= cmd` clears a variable for one command, and `docker --env NAME=`
    produces the same. A bare membership test made that turn the expensive
    window on, which is the opposite of clearing it."""

    from profiling.profilers import energy as energy_mod

    monkeypatch.setenv(ENERGY_ENV, "")
    assert energy_mod.energy_enabled() is False
    monkeypatch.setenv(energy_mod.REQUIRE_ENERGY_ENV, "")
    assert energy_mod.require_measured_energy() is False
