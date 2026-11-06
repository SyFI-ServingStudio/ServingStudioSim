"""Startup work in processes forked from the service.

Building and checking the members runs one simulator subprocess per item, but
the Python around each one (parsing a member's cost-tree JSON, writing a run's
trace and config) holds the GIL, so a thread pool serializes it on the main
interpreter: at 64 jobs that, not the simulator, set the startup time.
:func:`forked_map` runs each item in a worker forked from this process
instead. The worker inherits the function and the items, so neither is
pickled -- a closure over the index works as it does in a thread -- and only
each result travels back.
"""

from __future__ import annotations

import gc
import multiprocessing
import threading
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any

# The function and items a worker inherits; set in each worker by `_install`.
_WORK: tuple[Callable[[Any], Any], Sequence[Any]] | None = None


def _install(function: Callable[[Any], Any], items: Sequence[Any]) -> None:
    global _WORK
    _WORK = (function, items)


def _run(position: int) -> Any:
    assert _WORK is not None, "a forked worker runs only after _install"
    function, items = _WORK
    return function(items[position])


def forked_map(function: Callable[[Any], Any], items: Sequence[Any], jobs: int) -> Iterator[Any]:
    """``function(item)`` for each of ``items``, in order, on ``jobs`` workers
    forked from this process. A result must pickle; an exception the function
    raises is raised here, as :meth:`ThreadPoolExecutor.map` would.

    Forking copies only the calling thread, so a process with other Python
    threads alive (a server, a test client) gets a thread pool instead: same
    results, the Python serialized again. Native threads are expected and
    left alone: the profiling imports start BLAS/OpenMP pools and jemalloc's
    background thread, which a worker never uses -- it runs Python and starts
    subprocesses. They are why Python 3.12's fork DeprecationWarning shows
    under pytest."""
    if not items:
        return
    if threading.active_count() > 1:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            yield from pool.map(function, items)
        return
    # Objects that exist before the fork stay out of the workers' collections,
    # so a collection does not write to (and copy) every page they share.
    gc.freeze()
    try:
        with ProcessPoolExecutor(
            max_workers=min(jobs, len(items)),
            mp_context=multiprocessing.get_context("fork"),
            initializer=_install,
            initargs=(function, items),
        ) as pool:
            yield from pool.map(_run, range(len(items)))
    finally:
        gc.unfreeze()
