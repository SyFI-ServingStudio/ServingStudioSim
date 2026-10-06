"""Synchronously tear down a vLLM offline ``LLM`` so a rocprofv3 capture finalizes.

Why this exists — the per-rank rocpd capture bug it fixes:

rocprofv3 writes each traced process's rocpd database in a *tool-finalization*
handler that runs only on a CLEAN process exit. When the capture driver wraps an
offline ``LLM(tensor_parallel_size=N)`` launch, rocprofv3 traces the whole
process tree: the driver, the ``EngineCore`` process, and its ``MultiprocExecutor``
TP worker processes (which hold every GPU kernel dispatch). If the driver just
calls ``.generate(...)`` and returns, vLLM's interpreter-exit (atexit) teardown
runs the EngineCore process-manager shutdown with its *best-effort* grace window
(``timeout=None`` -> 5.0 s in ``vllm/v1/utils.py``). ROCm teardown plus each
worker's own finalize does not finish in 5 s, so the manager calls
``kill_process_tree`` and SIGKILLs the EngineCore and every TP worker. A SIGKILLed
process never runs rocprofv3's finalize handler, so NO per-rank rocpd database is
written — the exact symptom ``locate_rocpd_per_rank`` reports as "no per-rank
rocpd databases". The single-GPU (TP1) path never hit this because its one traced
process is the driver itself, which exits cleanly.

The fix is to tear the engine down *synchronously and with a generous grace
window* while the driver is still alive, BEFORE the interpreter's atexit path
runs. vLLM's V1 ``LLMEngine`` owns an ``engine_core`` client whose
``shutdown(timeout=...)`` SIGTERMs the EngineCore and waits up to ``timeout`` for
it (and, transitively, its workers) to exit on their own. Given enough time each
worker leaves its busy loop, the worker process exits normally, rocprofv3's
finalize runs, and the per-rank database lands on disk. The client's shutdown also
detaches its finalizer, so the later atexit path becomes a no-op and never
force-kills.

This module is intentionally dependency-free: it imports neither torch nor vLLM
at module load and duck-types the ``LLM`` object, so the capture workload can call
it on the GPU host while the behavior stays unit-testable on-host with a fake
engine. The real caller is the generated ``gen_tp.py`` capture driver, which runs
with the repository on ``PYTHONPATH``.
"""

from __future__ import annotations

from typing import Any

#: Default grace, in seconds, handed to the EngineCore process-manager shutdown.
#: Generous on purpose: it must cover ROCm device teardown plus each TP worker's
#: own rocprofv3 finalize so the manager's force-kill path is never reached. The
#: capture driver may override it (e.g. via an env knob) for a slower stack.
DEFAULT_SHUTDOWN_TIMEOUT_S = 120.0


def _call_shutdown(shutdown: Any, timeout_s: float) -> str:
    """Call a ``shutdown`` callable, preferring the ``timeout=`` grace-window form.

    Returns a short tag describing which call shape succeeded, for logging. Falls
    back to a no-argument call when the bound method does not accept ``timeout``
    (older/!MP client shapes), so a stack without the timeout knob still shuts
    down gracefully — just on its own default grace.
    """
    try:
        shutdown(timeout=timeout_s)
        return "shutdown(timeout=%g)" % timeout_s
    except TypeError:
        shutdown()
        return "shutdown()"


def graceful_shutdown_llm(
    llm: Any, *, timeout_s: float = DEFAULT_SHUTDOWN_TIMEOUT_S
) -> dict[str, Any]:
    """Synchronously shut a vLLM offline ``LLM`` down with a generous grace window.

    Walks ``llm.llm_engine.engine_core`` (the V1 offline shape) to the engine-core
    client and calls its ``shutdown(timeout=timeout_s)`` so the EngineCore and its
    TP worker processes exit cleanly — letting rocprofv3 finalize each per-rank
    rocpd database — before the interpreter's atexit force-kill path can fire.
    Accepts degraded shapes: ``llm`` may itself be the engine, and the engine may
    expose ``shutdown`` directly rather than through ``engine_core``.

    Returns a report dict (``{"path", "timeout_s", "ok", ...}``) rather than
    raising: the capture has already produced its trace by the time this runs, so
    a shutdown that cannot find a hook should be logged, not fatal. ``ok`` is
    ``True`` only when a shutdown callable was actually invoked.
    """
    report: dict[str, Any] = {"path": None, "timeout_s": timeout_s, "ok": False}
    if llm is None:
        report["error"] = "llm is None"
        return report

    # V1 offline: LLM.llm_engine is the engine; .engine_core is the client that
    # owns the EngineCore process (and, under it, the MultiprocExecutor workers).
    engine = getattr(llm, "llm_engine", llm)
    engine_core = getattr(engine, "engine_core", None)

    try:
        core_shutdown = getattr(engine_core, "shutdown", None)
        if callable(core_shutdown):
            how = _call_shutdown(core_shutdown, timeout_s)
            report.update(path="engine_core.%s" % how, ok=True)
            return report

        engine_shutdown = getattr(engine, "shutdown", None)
        if callable(engine_shutdown):
            how = _call_shutdown(engine_shutdown, timeout_s)
            report.update(path="llm_engine.%s" % how, ok=True)
            return report
    except Exception as error:  # a teardown failure must not mask the capture
        report["error"] = repr(error)
        return report

    report["error"] = "no engine_core.shutdown or llm_engine.shutdown found"
    return report
