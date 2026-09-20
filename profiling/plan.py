"""Collect every missing spec first, then issue whole kernels one per GPU.

Today a fill is demand-driven: the cost tree is walked node by node, and each
node that misses the cache profiles its own shapes immediately. An instrumented
tp4/ep4/spec5 fill (job 560) shows what that costs:

  - 55 separate profiling calls for 23 ``(kind, backend)`` units. ``elementwise``
    was invoked 14 times and ``single_gemm`` 11 times, because those kinds appear
    at that many cost-tree nodes.
  - Each call fans its specs across 4 GPUs and joins, so 55 calls became 208
    worker launches -- 208 cold JIT/autotune caches for 23 units of work.
  - Nothing overlaps across calls, so all four cards sat idle together 20% of the
    run (806 GPU-s) and another 284 GPU-s went into waiting for a group's
    slowest chunk.

No spec is measured twice -- the DB dedupes, and the per-call spec counts sum
exactly to the row counts. So the waste is entirely per-call fixed cost.
Measured end to end, running each unit whole on one GPU does the same 6308 rows
in 2595 GPU-s instead of 6256.

This module does the other half: collect what is missing without measuring it,
then issue the units across the available GPUs. A unit is cut only where leaving
it whole risks a long tail: before dispatch when one unit carries more than a
card's even share of the specs (``_slice_oversized``), and at the tail when there
are fewer units left than idle cards (``_take_next``). Both decide on spec counts
alone -- nothing here records or assumes what a kernel costs.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from profiling.db.kind import KernelKind
from profiling.db.registry import find_kernel_profiler_spec
from profiling.instrument import span

# A piece smaller than this is not worth its own process: the unit's fixed cost
# (process start, imports, CUDA context, the kind's JIT/autotune) is paid again
# per piece, and for `elementwise` alone that autotune is 0.702 s per call.
MIN_PIECE_SPECS = 16

# Once a unit is judged dominant, how finely to cut it: into pieces of about a
# card's even share divided by this. Only affects units that pass the dominance
# gate -- see `_slice_oversized`.
_PIECES_PER_CARD = 4


@dataclass(frozen=True)
class WorkUnit:
    """One ``(kind, backend, gpu_name)`` and every spec that needs measuring."""

    kernel_kind: KernelKind
    backend: str
    specs: tuple[dict[str, Any], ...]
    gpu_count: int
    # The cache key the miss was found under, carried through to the insert.
    # Dropping it and letting the worker's observed device name stand in is not
    # equivalent: `gpu/spec.json` accepts aliases, so a run whose `gpu:` is
    # `B200` would check key "B200" and insert under "NVIDIA B200" -- the same
    # specs missing on every build, forever. A PD deployment goes further and
    # asks for two different keys in one walk.
    gpu_name: str | None = None

    @property
    def size(self) -> int:
        """Ordering key. Spec count, because on a cold DB there is nothing else.

        It is a poor proxy for work -- across this table the per-spec cost spans
        0.120 s to 2.73 s -- but every better estimate needs history that a first
        run does not have. Ordering only decides which unit goes first, and the
        tail split (which does not use this at all) is what protects the
        makespan.
        """

        return len(self.specs)

    def sliced(self, pieces: int) -> list[WorkUnit]:
        """Split round-robin, so each piece gets a comparable mix of shapes.

        Specs arrive roughly ordered by size. Contiguous slicing would hand the
        largest shapes to the last piece and leave a short tail on an odd count
        (257 -> 128+128+1); striding gives 129+128 and mixes shapes evenly.
        """

        out = []
        for index in range(pieces):
            share = self.specs[index::pieces]
            if share:
                out.append(
                    WorkUnit(
                        self.kernel_kind,
                        self.backend,
                        tuple(share),
                        self.gpu_count,
                        self.gpu_name,
                    )
                )
        return out


class WorkCollector:
    """Accumulates missing specs per ``(kind, backend, gpu_name)`` during a dry walk."""

    def __init__(self) -> None:
        self._specs: dict[tuple[KernelKind, str, str | None], list[dict[str, Any]]] = {}
        self._seen: dict[tuple[KernelKind, str, str | None], set[str]] = {}
        self._lock = threading.Lock()

    def record(
        self,
        kernel_kind: KernelKind,
        backend: str,
        specs: list[dict[str, Any]],
        *,
        gpu_name: str | None = None,
    ) -> None:
        # Keyed by `gpu_name` too, so the dedupe below never folds together two
        # nodes that were checked against different cache keys -- a PD
        # deployment names a prefill GPU and a decode GPU in the same walk, and
        # collapsing them would leave one of the two unmeasured.
        key = (kernel_kind, backend, gpu_name)
        with self._lock:
            bucket = self._specs.setdefault(key, [])
            seen = self._seen.setdefault(key, set())
            for spec in specs:
                # Two cost-tree nodes can ask for the same shape before either is
                # measured; without this the unit would carry it twice and the
                # second copy would be a wasted measurement rather than a DB hit.
                identity = repr(sorted(spec.items(), key=lambda item: item[0]))
                if identity in seen:
                    continue
                seen.add(identity)
                bucket.append(spec)

    def units(self) -> list[WorkUnit]:
        """One unit per ``(kind, backend, gpu_name, gpu_count)``.

        ``gpu_count`` is per spec, not per kernel -- the collective kinds derive
        it from the spec's own ``num_gpus``/``ep_size`` -- so one backend can
        carry a 2-GPU spec and a 4-GPU spec at once. A unit has to be
        homogeneous in it: the unit is what gets a GPU reservation, and sizing
        that reservation from one arbitrary spec would hand a 4-GPU spec a
        2-GPU pool. ``execute_profile_batch`` classifies the same way
        internally.
        """

        units = []
        for (kernel_kind, backend, gpu_name), specs in self._specs.items():
            by_gpu_count: dict[int, list[dict[str, Any]]] = {}
            for spec in specs:
                by_gpu_count.setdefault(_gpu_count(kernel_kind, backend, spec), []).append(spec)
            for gpu_count, group in by_gpu_count.items():
                units.append(WorkUnit(kernel_kind, backend, tuple(group), gpu_count, gpu_name))
        return sorted(units, key=lambda unit: -unit.size)

    def __len__(self) -> int:
        """Total specs recorded. Comparable with the caller's own missing count."""

        with self._lock:
            return sum(len(specs) for specs in self._specs.values())

    def __bool__(self) -> bool:
        return len(self) > 0


def _gpu_count(kernel_kind: KernelKind, backend: str, spec: dict[str, Any]) -> int:
    profiler_spec = find_kernel_profiler_spec(kernel_kind, backend)
    count_fn = profiler_spec.gpu_count_fn
    return int(count_fn(spec)) if count_fn is not None else 1


_active = threading.local()


def active_collector() -> WorkCollector | None:
    return getattr(_active, "collector", None)


def set_collector(collector: WorkCollector | None) -> WorkCollector | None:
    """Install (or clear) the collector and return what was there.

    The PyO3 bridge calls ``count_missing_{kind}`` by a name it derives itself,
    with a fixed argument list, so there is no seam to thread a collector
    through. This is the same shape as the existing ``_jit_enabled`` toggle that
    Rust flips through ``enable_jit_profiling``.
    """

    previous = getattr(_active, "collector", None)
    _active.collector = collector
    return previous


@contextlib.contextmanager
def collecting(collector: WorkCollector | None = None) -> Iterator[WorkCollector]:
    """Walk the cost tree without measuring: record misses and move on.

    The walk still returns ``MissingEntry`` for anything uncached, so a caller
    must treat this as a dry run and walk again after ``issue``.
    """

    collector = collector or WorkCollector()
    previous = getattr(_active, "collector", None)
    _active.collector = collector
    try:
        yield collector
    finally:
        _active.collector = previous


@dataclass
class IssueReport:
    units: int = 0
    pieces: int = 0
    specs: int = 0
    split_units: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


def issue(
    collector: WorkCollector,
    *,
    db_path: Path,
    gpu_name: str | None,
    gpus: list[int],
) -> IssueReport:
    """Run every collected unit, one whole unit per GPU.

    Multi-GPU units run first and alone: they need every card at once, and
    interleaving them with single-GPU work would mean holding cards idle waiting
    for a quorum. There are three of them (the all-reduce kinds, 97 rows), so
    serialising them costs little and removes the only barrier in the loop.

    ``gpu_name`` is only the fallback for units recorded without one; a unit that
    carries its own cache key uses that. Both are passed to
    ``execute_profile_batch`` as the requested key, which validates it against
    what the workers actually observed before inserting anything.
    """

    from profiling.db.batch import execute_profile_batch
    from profiling.exec.local import LocalGpuPool

    report = IssueReport()
    units = collector.units()
    report.units = len(units)
    report.specs = sum(unit.size for unit in units)

    def run(unit: WorkUnit, assigned: list[int]) -> None:
        with span("unit.run", kind=unit.kernel_kind, backend=unit.backend, specs=unit.size):
            outcome = execute_profile_batch(
                unit.kernel_kind,
                [dict(spec, backend=unit.backend) for spec in unit.specs],
                pool=LocalGpuPool(gpus=assigned),
                db_path=db_path,
                gpu_name=unit.gpu_name if unit.gpu_name is not None else gpu_name,
            )
        # `execute_profile_batch` does not raise on a spec that failed to
        # measure: it logs the batch's failures and leaves `None` in its place.
        # That is right for the JIT path, where the caller's next `table.query`
        # still reports the row as missing and the command fails there. A cache
        # build has no such second look, so without this an OOM-killed worker
        # would end with "cache build complete" over an empty table.
        unmeasured = sum(1 for metrics in outcome.results if metrics is None)
        if unmeasured:
            raise RuntimeError(
                f"{unmeasured} of {unit.size} spec(s) were not measured; "
                "see the batch failure summary above for the reasons"
            )

    multi = [unit for unit in units if unit.gpu_count > 1]
    single = [unit for unit in units if unit.gpu_count == 1]

    for unit in multi:
        if unit.gpu_count > len(gpus):
            report.failures.append(
                f"{unit.kernel_kind}:{unit.backend} needs {unit.gpu_count} GPUs, have {len(gpus)}"
            )
            continue
        report.pieces += 1
        try:
            run(unit, gpus[: unit.gpu_count])
        except Exception as error:
            # Same treatment as a single-GPU unit: record it and keep going, so
            # one broken collective does not discard every unit behind it.
            report.failures.append(f"{unit.kernel_kind}:{unit.backend}: {error!r}")

    pending = _slice_oversized(single, len(gpus), report)
    free = list(gpus)
    running: dict[Any, tuple[WorkUnit, list[int]]] = {}
    with ThreadPoolExecutor(max_workers=max(len(gpus), 1)) as executor:
        while pending or running:
            while free and pending:
                piece = _take_next(pending, len(free), report)
                assigned = [free.pop()]
                running[executor.submit(run, piece, assigned)] = (piece, assigned)
                report.pieces += 1
            if not running:
                break
            done, _ = wait(list(running), return_when="FIRST_COMPLETED")
            for future in done:
                piece, assigned = running.pop(future)
                free.extend(assigned)
                error = future.exception()
                if error is not None:
                    report.failures.append(f"{piece.kernel_kind}:{piece.backend}: {error!r}")
    return report


def _slice_oversized(
    units: list[WorkUnit], gpu_count: int, report: IssueReport
) -> list[WorkUnit]:
    """Cut units large enough to monopolise a card, before dispatch.

    Insurance against a long tail. Without it the makespan has no bound beyond
    the single largest unit, which holds one card while every other card sits
    idle. Cutting that unit into P bounds it at roughly 1/P of its work plus P
    fixed costs.

    Whether to cut and how finely are separate questions, and collapsing them
    into one threshold is a trap. A threshold alone cuts a set that is already
    balanced: four equal units on four cards need no cutting at all, yet any
    threshold below their size shreds each into four, paying twelve extra fixed
    costs for nothing. Swept against the measured table that is the last row
    below -- 60 extra pieces, 1260 GPU-seconds, 265 seconds of makespan:

        rule               makespan   GPU-s   pieces   longest piece
        no cut                  705    2619       20           705 s
        every unit into 4       970    3879       80           192 s

    So the gate is dominance, not size: cut only a unit carrying more than a
    card's even share of the specs, because only such a unit can still be running
    when the others are done. A unit that passes the gate is then cut finely,
    into as many pieces as fit at a quarter of that share, capped at the card
    count.

    Spec count is the only signal used, deliberately: a cold DB has no record of
    what a kernel costs, and this keeps none. The gate therefore catches a unit
    with many specs and a slow kernel, and misses one with few specs and a slow
    kernel. On the measured table nothing is dominant -- the largest is
    `dsa_sparse_mla_attention` at 1426 specs against a 1602 share -- so nothing
    is cut and nothing is paid. The 705-second tail there is `single_gemm`, 544
    specs, which no spec-count rule can see. This insures against the run where
    one kernel's grid is an order of magnitude larger than the rest, and it is
    free on the runs where it is not.
    """

    if gpu_count < 2 or not units:
        return list(units)
    total = sum(unit.size for unit in units)
    even_share = max(MIN_PIECE_SPECS, total // gpu_count)
    target = max(MIN_PIECE_SPECS, even_share // _PIECES_PER_CARD)
    out: list[WorkUnit] = []
    for unit in units:
        if unit.size <= even_share:
            out.append(unit)
            continue
        pieces = min(gpu_count, -(-unit.size // target))
        pieces = min(pieces, max(1, unit.size // MIN_PIECE_SPECS))
        if pieces < 2:
            out.append(unit)
            continue
        sliced = unit.sliced(pieces)
        report.split_units.append(f"{unit.kernel_kind}:{unit.backend} x{len(sliced)}")
        out.extend(sliced)
    return sorted(out, key=lambda unit: -unit.size)


def _take_next(pending: list[WorkUnit], free_count: int, report: IssueReport) -> WorkUnit:
    """Pop the next piece, splitting only when the queue is about to run dry.

    Splitting is never free: every piece re-pays the unit's fixed cost (process
    start, imports, CUDA context, the kind's JIT/autotune). So this fires on an
    observation rather than a forecast -- there are fewer units left than idle
    cards, so a card idles no matter what -- and uses no per-kernel constant.

    What it does NOT fix: a long unit dispatched early. Units go out largest
    first, so in the measured table `single_gemm` (705 s) leaves on the first
    card immediately, the other nineteen finish on the remaining three by 636 s,
    and by then `pending` is empty -- there is nothing left to split. Getting
    under that 705 s needs `single_gemm` itself split at dispatch, which needs an
    estimate of its cost. Spec count is not that estimate: the largest unit by
    spec count here is `dsa_sparse_mla_attention` at 1426 specs and 208 s, while
    `single_gemm` is 544 specs and 705 s. Recording each unit's wall so the next
    run can order and split by time is the fix; the first cold run cannot have it.
    """

    if len(pending) >= free_count or pending[0].size < MIN_PIECE_SPECS * 2:
        return pending.pop(0)

    unit = pending.pop(0)
    pieces = min(free_count, max(1, unit.size // MIN_PIECE_SPECS))
    if pieces < 2:
        return unit
    sliced = unit.sliced(pieces)
    report.split_units.append(f"{unit.kernel_kind}:{unit.backend} x{len(sliced)}")
    pending[:0] = sliced[1:]
    return sliced[0]
