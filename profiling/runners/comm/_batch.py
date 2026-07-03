"""Shared list-runner orchestration for the multi-GPU comm runners.

The L1 runner contract is ``run(kwargs_list) -> list[RunnerResult]`` (1:1, in
order). For comm runners the expensive part is the rank-group launch
(``mp.spawn`` + ``init_process_group`` / ``nvshmem.core.init`` + teardown —
seconds to ~12 s), NOT the collective (microseconds). The single-spec contract
paid that launch once *per shape*; this batch path pays it **once per chunk**:
one ``launcher.run`` spawns the group, a per-rank ``_..._per_rank_batch`` fn
loops every size inside the live group, and rank 0 returns the ordered payloads.

**Whole-chunk failure semantics.** The ranks run the same sizes in lockstep, so
a per-size abort on one rank would desynchronize the collective and deadlock the
others. Per-rank fns therefore do NOT catch per-size — a failure raises out of
the whole group. This helper maps that into ``RunnerResult`` errors for the
*entire* chunk (count preserved for the downstream ``zip(strict=True)``); compute
runners keep per-item granularity via ``batched``. Per-size comm granularity
(symmetric try/finally barriers) is a possible later refinement.

Imported only inside the comm runner modules, which are themselves lazy-loaded in
the worker subprocess — so no torch/launcher import reaches the main process.
"""

from __future__ import annotations

from collections.abc import Callable

from profiling.runners.comm._launcher import MultiGpuLauncher
from profiling.runners.metrics import CommMetrics, RunnerResult

# A per-rank batch fn: ``(*, rank, world_size, specs, warmup, rep) -> list[dict] | None``.
# Rank 0 returns one timing dict per spec (in order); other ranks return None.
PerRankBatchFn = Callable[..., "list[dict] | None"]

# Framework loop counts. Not part of the coerced spec kwargs (``args_to_spec``
# emits only schema fields), so they're supplied here uniformly for every size.
_WARMUP = 50
_REP = 100


def run_comm_batch(
    launcher: MultiGpuLauncher,
    per_rank_batch_fn: PerRankBatchFn,
    kwargs_list: list[dict],
) -> list[RunnerResult]:
    """Spawn ``launcher``'s rank group once for the whole homogeneous chunk and map
    rank 0's ordered per-size payloads into ``RunnerResult``s. Any launcher
    failure, or a missing / wrong-length payload, fails the whole chunk (see the
    module docstring) rather than risk a partial, deadlock-prone result."""
    if not kwargs_list:
        return []
    try:
        payloads = launcher.run(
            per_rank_batch_fn, specs=kwargs_list, warmup=_WARMUP, rep=_REP
        )
    except Exception as exc:  # noqa: BLE001 — the whole rank group tore down; fail the chunk
        return all_error(len(kwargs_list), str(exc))
    if payloads is None or len(payloads) != len(kwargs_list):
        got = 0 if payloads is None else len(payloads)
        return all_error(
            len(kwargs_list),
            f"comm batch returned {got} rank-0 payloads for {len(kwargs_list)} specs",
        )
    return [RunnerResult(metrics=_comm_metrics(payload)) for payload in payloads]


def all_error(count: int, message: str) -> list[RunnerResult]:
    """``count`` identical error results — one per spec so the 1:1 count holds."""
    return [RunnerResult(error=message) for _ in range(count)]


def _comm_metrics(payload: dict) -> CommMetrics:
    return CommMetrics(
        time_ms=float(payload["time_ms"]),
        algbw_gbps=float(payload["algbw_gbps"]),
        busbw_gbps=float(payload["busbw_gbps"]),
        energy_j=float(payload.get("energy_j", 0.0)),
    )
