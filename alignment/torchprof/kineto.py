"""Read a vLLM torch-profiler (Kineto) chrome trace into rocpd-shaped records.

The capture boundary of the torch-profiler producer. A PyTorch Kineto chrome
trace is a JSON document (optionally gzipped) with a top-level ``traceEvents``
array; each GPU kernel dispatch is a complete event (``"ph": "X"``) with
``"cat": "kernel"``, a demangled ``"name"``, ``"ts"`` / ``"dur"`` in
MICROSECONDS, and an ``"args"`` dict carrying ``"grid"`` / ``"block"`` /
``"stream"`` / ``"correlation"`` / ``"device"``. This module turns those kernel
events into :class:`RocpdDispatch` records (timestamps converted to the
nanoseconds the rest of the pipeline uses) and any canonical
``(?:vllm|sglang)_iteration(N): <phase>`` ``user_annotation`` ranges into
:class:`RoctxRegion`, so the shared
:func:`alignment.rocpd.evidence.build_ranges_from_dispatches` attributes them
exactly as it does a rocpd capture.

Per-rank separation is inherent: each worker writes its OWN trace file
(``<prefix>_dp..._tp..._ep..._rank<N>.<ts>.pt.trace.json[.gz]``), so one file is
one rank's single device and the offline merge keys the device id off the
``rank<N>`` in the filename -- the same rule the rocpd merge uses.

Scope matches the rocpd producer: the eager (no-graph), kernel-only path. The
sentinel is identified by name and excluded from the compared set downstream;
its iteration ordinal rides in ``grid[1]`` (``grid_size_y``), recovered by the
shared sentinel reconstruction, which falls back to dispatch order when a build's
roctracer backend does not populate ``grid`` for HIP kernels.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from profiling.profilers.rocprof_kernel_profiler import RocpdDispatch, RoctxRegion

from ..nsys.parse import ITER_RE

#: Kineto event categories that are GPU kernel DISPATCHES. Only ``kernel`` is the
#: compute-kernel lane the rocpd ``kernel_dispatch`` table mirrors; host-side API
#: rows (``cuda_runtime`` / ``hip`` launches), ``cpu_op``, ``python_function`` and
#: the memcpy/memset lanes are intentionally excluded so the compared kernel set
#: matches the rocpd producer's. The sentinel is a real GPU kernel and so lands in
#: this lane.
KERNEL_CATEGORIES = frozenset({"kernel"})

#: The Kineto event category a roctx/NVTX range lands in when the profiler records
#: it (``with_stack`` / ``record_function``). Iteration markers are matched out of
#: these by :data:`alignment.nsys.parse.ITER_RE`, identical to the rocpd roctx path.
ANNOTATION_CATEGORIES = frozenset({"user_annotation"})

#: Chrome-trace complete-event phase: an event with a start and a duration.
_COMPLETE_PHASE = "X"


def _open_trace(path: Path):
    """Open a chrome trace, transparently gunzipping a ``.gz`` (by name or magic)."""
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b" or str(path).endswith(".gz"):
        return gzip.decompress(raw)
    return raw


def _us_to_ns(microseconds: float) -> int:
    """Kineto timestamps are microseconds; the pipeline is integer nanoseconds."""
    return int(round(microseconds * 1000.0))


def _grid_component(grid, index: int) -> int:
    """One component of a Kineto ``args['grid']`` ``[x, y, z]``, 0 when absent.

    A build whose roctracer backend omits ``grid`` for HIP kernels yields 0 here;
    the sentinel reconstruction then falls back to dispatch order, so a missing
    grid degrades the iteration index to the ordinal rather than failing.
    """
    if isinstance(grid, (list, tuple)) and len(grid) > index and grid[index] is not None:
        try:
            return int(grid[index])
        except (TypeError, ValueError):
            return 0
    return 0


def _dispatch_from_event(event: dict) -> RocpdDispatch:
    """One GPU ``kernel`` event -> a :class:`RocpdDispatch` in nanoseconds.

    ``pid`` is forced to 0: a per-rank trace is one process, so the containment
    join's process guard must always admit (a Kineto kernel event's ``pid`` is the
    DEVICE id, not an OS pid, and would wrongly fail an equality check). The device
    id is taken from ``args['device']`` (the authoritative field), falling back to
    the event ``pid``; within a single rank's file it is that rank's local device,
    and the offline merge overwrites it with the filename rank anyway.
    """
    args = event.get("args") or {}
    ts = float(event.get("ts", 0.0))
    dur = float(event.get("dur", 0.0))
    correlation = args.get("correlation", args.get("External id"))
    device = args.get("device", event.get("pid", 0))
    grid = args.get("grid")
    return RocpdDispatch(
        start_ns=_us_to_ns(ts),
        end_ns=_us_to_ns(ts + dur),
        name=event.get("name", "unknown"),
        stream_id=int(args.get("stream", 0) or 0),
        correlation_id=int(correlation) if correlation is not None else None,
        device_id=int(device) if device is not None else 0,
        pid=0,
        grid_size_x=_grid_component(grid, 0),
        grid_size_y=_grid_component(grid, 1),
        grid_size_z=_grid_component(grid, 2),
    )


def _region_from_event(event: dict) -> RoctxRegion:
    """One iteration-marker ``user_annotation`` event -> a :class:`RoctxRegion`."""
    ts = float(event.get("ts", 0.0))
    dur = float(event.get("dur", 0.0))
    return RoctxRegion(
        start_ns=_us_to_ns(ts),
        end_ns=_us_to_ns(ts + dur),
        tid=event.get("tid"),
        # One file is one process; leaving pid unset lets the containment join's
        # process guard admit every dispatch (single-process capture).
        pid=None,
        name=event.get("name", ""),
    )


def read_kineto_trace(path: Path) -> tuple[list[RocpdDispatch], list[RoctxRegion]]:
    """Parse a Kineto chrome trace into (GPU kernel dispatches, iteration regions).

    Returns every GPU ``kernel``-category dispatch (model kernels AND any
    ``vibesim_sentinel`` markers -- the shared segmentation separates them) in
    launch order, plus the canonical ``(?:vllm|sglang)_iteration(N)``
    ``user_annotation`` ranges the trace carries (empty on the common torch-on-ROCm
    build that does not surface roctx, in which case the sentinel path owns
    segmentation).
    """
    document = json.loads(_open_trace(Path(path)))
    events = document.get("traceEvents", [])

    dispatches: list[RocpdDispatch] = []
    regions: list[RoctxRegion] = []
    for event in events:
        if event.get("ph") != _COMPLETE_PHASE:
            continue
        category = event.get("cat")
        if category in KERNEL_CATEGORIES:
            dispatches.append(_dispatch_from_event(event))
        elif category in ANNOTATION_CATEGORIES and ITER_RE.match(event.get("name", "")):
            regions.append(_region_from_event(event))

    dispatches.sort(key=lambda dispatch: (dispatch.start_ns, dispatch.stream_id))
    regions.sort(key=lambda region: region.start_ns)
    return dispatches, regions
