"""Read an AMD rocpd capture into the same ``RangeStats`` the nsys parser builds.

This is the ROCm counterpart of ``alignment/nsys``'s capture boundary. The nsys
side attributes a CUDA kernel to an iteration by *correlation ownership* — a
runtime API call launched inside an NVTX range owns the kernel sharing its
``correlationId`` — because CUDA-graph decode kernels execute past their NVTX
range. rocprofv3 does not export that host runtime→dispatch correlation in the
offline rocpd database, and the eager (no-HIP-graph) model does not need it: a
kernel's GPU dispatch lands inside the roctx iteration range that launched it.

So the one genuinely new rule here is the **timestamp-containment ownership
join**: a dispatch belongs to the ``vllm_iteration(N): <phase>`` roctx range
whose ``[start, end)`` contains the dispatch's launch time. Everything after the
join — building the normalized files — reuses the backend-neutral nsys pieces
unchanged (``RangeStats``, ``KernelEvent``, ``kernel_category``), so the rocpd
and nsys producers emit a byte-identical schema.

Correlation ownership and HIP-graph coverage are a later chunk; this module is
the eager, graph-free path and raises rather than guessing when a capture has no
iteration markers at all.

The raw rocpd reads (full dispatch rows, roctx regions) live in the profiling
reader ``profiling.profilers.rocprof_kernel_profiler`` and are reused here so the
schema-discovery logic is not forked.
"""

from __future__ import annotations

import bisect

from profiling.profilers.rocprof_kernel_profiler import (
    RocpdDispatch,
    RoctxRegion,
    kernel_dispatch_records_from_rocpd,
    roctx_regions_from_rocpd,
)

from ..nsys.evidence import kernel_category
from ..nsys.parse import ITER_RE, KernelEvent, RangeStats, Worker

#: (iteration index, phase, region). The phase is the text after the colon in a
#: ``vllm_iteration(3): forward`` marker, matching the nsys ``_parse_label`` rule.
IterationRegion = tuple[int, str, RoctxRegion]


def iteration_regions(regions: list[RoctxRegion]) -> list[IterationRegion]:
    """Keep only roctx ranges that are canonical iteration markers, in start order.

    A capture carries roctx ranges for many things; alignment ownership keys off
    exactly the ``(?:vllm|sglang)_iteration(N): <phase>`` markers the instrumented
    fork emits, the same contract the nsys parser's ``ITER_RE`` enforces.
    """
    parsed: list[IterationRegion] = []
    for region in regions:
        match = ITER_RE.match(region.name)
        if match is not None:
            parsed.append((int(match.group(1)), match.group(2), region))
    parsed.sort(key=lambda item: item[2].start_ns)
    return parsed


def _pid_matches(region: RoctxRegion, dispatch: RocpdDispatch) -> bool:
    """A dispatch can only belong to a region its own process emitted.

    When either side did not record a pid (a reduced capture, or a synthetic
    fixture) the check cannot distinguish processes and admits the pairing — a
    single-process capture is the common case and the timestamp containment is
    then the whole decision.
    """
    if region.pid is None or dispatch.pid == 0:
        return True
    return region.pid == dispatch.pid


def owning_region(
    dispatch: RocpdDispatch,
    iter_regions: list[IterationRegion],
    region_starts: list[int],
) -> IterationRegion | None:
    """Return the tightest iteration region whose ``[start, end)`` holds the launch.

    ``iter_regions`` is sorted by start and ``region_starts`` is its start column,
    so the search begins at the last region that opened at or before the launch
    and walks outward to earlier (enclosing) regions. The first one that still
    contains the launch and shares the process is the owner; scanning outward is
    what makes a nested phase marker win over the iteration range that encloses
    it. ``None`` means the dispatch fell outside every iteration (warm-up or an
    inter-iteration gap) and is intentionally dropped.
    """
    index = bisect.bisect_right(region_starts, dispatch.start_ns) - 1
    while index >= 0:
        iteration, phase, region = iter_regions[index]
        if region.end_ns > dispatch.start_ns and _pid_matches(region, dispatch):
            return iter_regions[index]
        index -= 1
    return None


def build_ranges_from_rocpd(
    db_path: str,
    *,
    default_stage: str = "all",
) -> list[RangeStats]:
    """Attribute every kernel dispatch to its roctx iteration range.

    Reads the full dispatch rows and roctx regions from the rocpd database, keeps
    the iteration markers, and groups dispatches into one ``RangeStats`` per
    ``(device, iteration, phase)`` by timestamp containment. The populated
    ``RangeStats`` are exactly what the nsys parser hands to
    ``build_iteration_details`` / ``build_device_kernel_sequences``, so the rest
    of the pipeline is shared.

    Raises ``ValueError`` when the capture carries no iteration markers: without
    them there is no ownership to establish, which is the state of the upstream
    (uninstrumented) vLLM capture and is a real "nothing to align" signal, not
    something to paper over.
    """
    dispatches = kernel_dispatch_records_from_rocpd(db_path)
    iter_regions = iteration_regions(roctx_regions_from_rocpd(db_path))
    if not iter_regions:
        raise ValueError(
            "rocpd capture has no (vllm|sglang)_iteration(N) roctx ranges; the "
            "dispatches cannot be attributed to iterations (an uninstrumented "
            "capture, or one taken without the roctx iteration plugin)"
        )
    region_starts = [region.start_ns for _, _, region in iter_regions]

    workers: dict[int, Worker] = {}
    groups: dict[tuple[int, int, str], RangeStats] = {}
    for dispatch in dispatches:
        owner = owning_region(dispatch, iter_regions, region_starts)
        if owner is None:
            continue
        iteration, phase, region = owner
        device_id = dispatch.device_id
        worker = workers.get(device_id)
        if worker is None:
            # rocpd keys a dispatch's process by pid; the device is its agent id.
            # A synthetic global_pid keeps the nsys Worker shape without inventing
            # an nsys-style globalPid+pid thread identity the rocpd path never uses.
            worker = Worker(
                global_pid=region.pid if region.pid is not None else device_id,
                pid=dispatch.pid,
                name="rocpd-worker",
                device_id=device_id,
            )
            workers[device_id] = worker
        key = (device_id, iteration, phase)
        item = groups.get(key)
        if item is None:
            item = RangeStats(
                iteration=iteration,
                phase=phase,
                stage=default_stage,
                worker=worker,
                start=region.start_ns,
                end=region.end_ns,
                emitting_global_tid=region.tid,
            )
            groups[key] = item

        name = dispatch.name or "unknown"
        category = kernel_category(name)
        duration = dispatch.duration_ns
        item.intervals.append((dispatch.start_ns, dispatch.end_ns))
        item.kernel_count += 1
        item.sum_ns += duration
        item.category_ns[category] += duration
        item.kernel_name_ns[name] += duration
        item.kernel_name_count[name] += 1
        item.kernel_events.append(
            KernelEvent(
                start=dispatch.start_ns,
                end=dispatch.end_ns,
                name=name,
                category=category,
                stream_id=dispatch.stream_id,
                correlation_id=dispatch.correlation_id,
            )
        )

    # `partition_kernel_tracks` orders tracks by first launch, so the events a
    # range carries must be in launch order; rocpd returned them in start order
    # globally, but a per-range sort keeps that guarantee after grouping.
    for item in groups.values():
        item.kernel_events.sort(key=lambda event: (event.start, event.stream_id))
    return list(groups.values())
