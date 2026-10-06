"""CPU tests for the capture-workload engine teardown (``engine_shutdown``).

These prove, without vLLM or a GPU, the one behavior the per-rank rocpd capture
depends on: the driver shuts the engine down *synchronously with a grace window*
so each rocprofv3-traced TP worker exits cleanly and finalizes its database,
instead of being SIGKILLed by vLLM's 5 s best-effort atexit path. The object is
duck-typed, so a tiny fake engine stands in for the real ``LLM``.
"""

from __future__ import annotations

import pytest

from alignment.profiler.engine_shutdown import (
    DEFAULT_SHUTDOWN_TIMEOUT_S,
    graceful_shutdown_llm,
)


class _FakeEngineCore:
    """Records the shutdown timeout it was handed (the grace window that matters)."""

    def __init__(self, accepts_timeout: bool = True):
        self._accepts_timeout = accepts_timeout
        self.calls: list[float | None] = []

    def shutdown(self, timeout=None):
        if timeout is not None and not self._accepts_timeout:
            raise TypeError("shutdown() got an unexpected keyword argument 'timeout'")
        self.calls.append(timeout)


class _FakeEngine:
    def __init__(self, engine_core=None, direct_shutdown=False):
        if engine_core is not None:
            self.engine_core = engine_core
        if direct_shutdown:
            self._direct = _FakeEngineCore()

    def shutdown(self, timeout=None):  # only reached when engine_core is absent
        self._direct.shutdown(timeout=timeout)


class _FakeLLM:
    def __init__(self, llm_engine):
        self.llm_engine = llm_engine


def test_shuts_down_engine_core_with_grace_window():
    """The engine-core client is shut down with the generous timeout, not 5 s."""
    core = _FakeEngineCore()
    report = graceful_shutdown_llm(_FakeLLM(_FakeEngine(core)))
    assert report["ok"] is True
    assert "engine_core.shutdown(timeout=" in report["path"]
    # The grace window is actually passed through — the whole point of the fix.
    assert core.calls == [DEFAULT_SHUTDOWN_TIMEOUT_S]
    assert core.calls[0] >= 30.0  # must exceed vLLM's 5 s best-effort grace


def test_timeout_is_overridable():
    core = _FakeEngineCore()
    graceful_shutdown_llm(_FakeLLM(_FakeEngine(core)), timeout_s=45.0)
    assert core.calls == [45.0]


def test_falls_back_to_no_arg_shutdown_when_timeout_unsupported():
    """A client whose shutdown() lacks a timeout kwarg still shuts down cleanly."""
    core = _FakeEngineCore(accepts_timeout=False)
    report = graceful_shutdown_llm(_FakeLLM(_FakeEngine(core)))
    assert report["ok"] is True
    assert report["path"] == "engine_core.shutdown()"
    assert core.calls == [None]


def test_falls_back_to_engine_shutdown_without_engine_core():
    """When there is no engine_core, the engine's own shutdown is used."""
    engine = _FakeEngine(engine_core=None, direct_shutdown=True)
    report = graceful_shutdown_llm(_FakeLLM(engine))
    assert report["ok"] is True
    assert report["path"].startswith("llm_engine.shutdown(timeout=")
    assert engine._direct.calls == [DEFAULT_SHUTDOWN_TIMEOUT_S]


def test_llm_may_be_the_engine_itself():
    """Passing the engine directly (no .llm_engine wrapper) still works."""
    core = _FakeEngineCore()
    report = graceful_shutdown_llm(_FakeEngine(core))
    assert report["ok"] is True
    assert core.calls == [DEFAULT_SHUTDOWN_TIMEOUT_S]


def test_missing_hook_reports_not_ok_without_raising():
    class _Bare:
        pass

    report = graceful_shutdown_llm(_FakeLLM(_Bare()))
    assert report["ok"] is False
    assert "no engine_core.shutdown" in report["error"]


def test_none_llm_is_not_fatal():
    report = graceful_shutdown_llm(None)
    assert report["ok"] is False
    assert "None" in report["error"]


def test_shutdown_exception_is_captured_not_raised():
    class _Boom:
        def shutdown(self, timeout=None):
            raise RuntimeError("device teardown blew up")

    report = graceful_shutdown_llm(_FakeLLM(_FakeEngine(_Boom())))
    assert report["ok"] is False
    assert "device teardown blew up" in report["error"]
