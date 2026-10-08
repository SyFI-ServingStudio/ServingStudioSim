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
from ..profiler.roctx_shim import (
    DEFAULT_ENGINE_PREFIX,
    SENTINEL_GRID_Y_OFFSET,
    SENTINEL_KERNEL_NAME,
    iteration_label,
)

#: (iteration index, phase, region). The phase is the text after the colon in a
#: ``vllm_iteration(3): forward`` marker, matching the nsys ``_parse_label`` rule.
IterationRegion = tuple[int, str, RoctxRegion]

#: The phase a sentinel-reconstructed iteration carries. The Option-B sentinel
#: marks the whole forward (there is one sentinel per forward), so its range is
#: the ``forward`` phase, matching the roctx shim's ``vllm_iteration(N): forward``.
_SENTINEL_PHASE = "forward"


def is_sentinel_dispatch(dispatch: RocpdDispatch) -> bool:
    """Whether a dispatch is a ``vibesim_sentinel`` iteration-boundary marker.

    Identity is the kernel name: the Option-B shim launches a Triton kernel whose
    function name (interned verbatim in rocpd's kernel-symbol table) contains
    ``vibesim_sentinel``, which no torch/aiter kernel carries. These dispatches
    are markers, not model work, so the producer excludes them from the attributed
    and compared kernel set and uses them only to bound iteration ranges.
    """
    return bool(dispatch.name and SENTINEL_KERNEL_NAME in dispatch.name)


def sentinel_iteration_regions(
    sentinels: list[RocpdDispatch],
    dispatches: list[RocpdDispatch],
    *,
    prefix: str = DEFAULT_ENGINE_PREFIX,
) -> list[IterationRegion]:
    """Reconstruct iteration ranges from sentinel marker dispatches.

    Each sentinel bounds one forward: sentinel ``N``'s launch starts iteration
    ``N`` and the next sentinel's launch ends it (the last sentinel runs to the
    end of all GPU work). The absolute iteration index rides in the sentinel's
    y-grid dimension (``grid_size_y - SENTINEL_GRID_Y_OFFSET``); when that encoding
    is present and strictly increasing it is used, otherwise the index falls back
    to the sentinel's ordinal position in the dispatch stream (0, 1, 2, …), which
    is always correct for consecutive forwards. The returned tuples have the exact
    :data:`IterationRegion` shape the roctx path produces, so the same
    timestamp-containment ownership join attributes dispatches to them unchanged.
    """
    ordered = sorted(sentinels, key=lambda d: d.start_ns)
    decoded = [d.grid_size_y - SENTINEL_GRID_Y_OFFSET for d in ordered]
    use_decoded = all(index >= 0 for index in decoded) and all(
        earlier < later for earlier, later in zip(decoded, decoded[1:])
    )
    # The last sentinel's range runs past the end of every recorded dispatch so no
    # trailing kernel of the final iteration is dropped (+1 to keep [start, end)).
    end_bound = max(
        (d.end_ns for d in [*dispatches, *ordered]),
        default=ordered[-1].end_ns,
    ) + 1

    regions: list[IterationRegion] = []
    for ordinal, sentinel in enumerate(ordered):
        iteration = decoded[ordinal] if use_decoded else ordinal
        start = sentinel.start_ns
        end = ordered[ordinal + 1].start_ns if ordinal + 1 < len(ordered) else end_bound
        region = RoctxRegion(
            start_ns=start,
            end_ns=end,
            tid=None,
            pid=sentinel.pid or None,
            name=iteration_label(iteration, _SENTINEL_PHASE, prefix=prefix),
        )
        regions.append((iteration, _SENTINEL_PHASE, region))
    return regions


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

    Two iteration-marker sources are supported, roctx-first. When the capture
    carries ``vllm_iteration(N)`` roctx ranges (the stack records markers) those
    are authoritative. When it does not but carries ``vibesim_sentinel`` marker
    kernels (Option B — the ROCm-7.2 / rocprofv3-1.3.2 stack where the SDK MARKER
    service never registers, so ``--marker-trace`` yields nothing while
    ``--kernel-trace`` is reliable), the ranges are reconstructed from the
    sentinel dispatches. Sentinel dispatches are markers, not model work, so they
    are excluded from the attributed/compared kernel set in BOTH cases.

    Raises ``ValueError`` when the capture carries neither marker source: without
    them there is no ownership to establish, which is the state of the upstream
    (uninstrumented) vLLM capture and is a real "nothing to align" signal, not
    something to paper over.
    """
    all_dispatches = kernel_dispatch_records_from_rocpd(db_path)
    roctx = roctx_regions_from_rocpd(db_path)
    return build_ranges_from_dispatches(
        all_dispatches, roctx, default_stage=default_stage, capture_kind="rocpd"
    )


def build_ranges_from_dispatches(
    all_dispatches: list[RocpdDispatch],
    roctx: list[RoctxRegion],
    *,
    default_stage: str = "all",
    capture_kind: str = "capture",
) -> list[RangeStats]:
    """Attribute pre-read dispatches to their iteration ranges (backend-neutral).

    The backend-neutral core of :func:`build_ranges_from_rocpd`, factored out so a
    non-rocpd reader (the torch/kineto producer in ``alignment/torchprof``) can
    feed the identical sentinel/roctx iteration-segmentation and
    timestamp-containment attribution without forking it. ``all_dispatches`` carries
    the model GPU kernels *and* any ``vibesim_sentinel`` markers (both already as
    :class:`RocpdDispatch`); ``roctx`` is the capture's iteration ranges (empty when
    the backend records none, which is the common torch-on-ROCm case). The
    roctx-first, sentinel-fallback marker selection and the raise-when-neither
    contract are exactly the rocpd path's, so the two producers cannot drift.

    ``capture_kind`` only colours the "nothing to align" error text.
    """
    sentinels = [d for d in all_dispatches if is_sentinel_dispatch(d)]
    # The sentinels are markers; only the model dispatches are ever attributed.
    dispatches = [d for d in all_dispatches if not is_sentinel_dispatch(d)]

    iter_regions = iteration_regions(roctx)
    if not iter_regions and sentinels:
        iter_regions = sentinel_iteration_regions(sentinels, dispatches)
    if not iter_regions:
        raise ValueError(
            f"{capture_kind} capture has no (vllm|sglang)_iteration(N) roctx ranges "
            "and no vibesim_sentinel marker kernels; the dispatches cannot be "
            "attributed to iterations (an uninstrumented capture, or one taken "
            "without the roctx iteration plugin / Option-B sentinel emitter)"
        )
    return _attribute_dispatches_to_regions(iter_regions, dispatches, default_stage)


def _attribute_dispatches_to_regions(
    iter_regions: list[IterationRegion],
    dispatches: list[RocpdDispatch],
    default_stage: str,
) -> list[RangeStats]:
    """Group dispatches into one ``RangeStats`` per ``(device, iteration, phase)``.

    The timestamp-containment ownership join shared by the roctx and sentinel
    marker sources: each dispatch is attributed to the iteration region whose
    ``[start, end)`` holds its launch, and dispatches outside every region are
    dropped (warm-up / inter-iteration gaps).
    """
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
