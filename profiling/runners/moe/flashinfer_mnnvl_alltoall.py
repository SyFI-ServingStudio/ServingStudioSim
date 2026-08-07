"""FlashInfer MNNVL two-sided MoE all-to-all runners (transfer / prepare).

L1a-only, list-native. The expensive setup here is not the rank-group spawn but
the **fabric-memory workspace**: ``MnnvlMemory`` maps a multicast region across
every rank of the EP group, and flashinfer caches exactly one mapping per
process (``only one moe mapping supported now``). So one spawn per chunk builds
the workspace once and loops every shape inside it.

What is measured
----------------
``moe_comm`` — the transfer itself — and ``mnnvl_moe_alltoallv_prepare_without_allgather``
for the metadata pass. ``moe_comm`` is the single call underneath both legs of
vLLM's round trip (see
``vllm/model_executor/layers/fused_moe/prepare_finalize/flashinfer_nvlink_two_sided.py``):

    dispatch  MnnvlMoe.mnnvl_moe_alltoallv        -> torch.empty  + moe_comm
    combine   MnnvlMoe.mnnvl_moe_alltoallv_combine-> torch.zeros  + moe_comm + torch.sum

The wrappers' allocations are deliberately outside the measurement. Combine's
zero fill writes ``token_count x top_k x hidden`` (805 MB per layer at 8k tokens,
comparable to the transfer beside it) and its ``torch.sum`` reads the same buffer
back; both scale with the token count, which is not one of this leaf's axes.
Leaving them inside would have forced a third dimension on the cache. The arch
prices them as two elementwise leaves instead, which is what they are.

How a shape is realised
-----------------------
The two sweep axes are ``max_send_rows`` and ``max_recv_rows`` — the busiest
sender's and the busiest receiver's row counts. Rather than search for a routing
draw that happens to produce a requested pair, the runner builds the send/recv
index tensors directly (``_transfer_matrix`` + ``_rank_plan``), which is exact
and needs no search. The index semantics come straight from ``SendRecvDispls``
in ``trtllm_alltoall.cuh``: an inclusive per-peer cumsum plus one row index per
transferred row.

Every rank derives the whole ``ep_size x ep_size`` matrix from the same spec, so
the plans agree by construction — rank ``r``'s send count to ``d`` is rank ``d``'s
recv count from ``r``, which is what keeps the collective from deadlocking.

The same rule governs the *number* of calls, which is the easier one to get
wrong: anything deciding how many transfers to launch has to be a function of
the spec, never of a measurement. See ``_iters_per_graph``.

``direction`` decides the read-side reuse, the one way the two legs genuinely
differ: dispatch gathers a token row once per distinct destination it reaches
(fan-out ``_destinations_per_token``), combine reads each expert-output row
exactly once.

Timing basis — CUDA-graph replay
--------------------------------
Measured with an eager rep loop, these calls carry a floor that has nothing to
do with the transfer: ~78 us for ``prepare`` from 1 token through 4096, against
15.6 us for the same five kernels in a vLLM trace. Eight ranks launched from
Python re-rendezvous on every call, and at decode scale that rendezvous *is* the
measurement.

vLLM does not pay it: decode replays as a captured CUDA graph, and 75 MoE layers
run back to back with the ranks already locked to each other. So the graph is
not a trick for a smaller number — it is the mode the alignment target actually
runs in, and the eager floor is an artefact this benchmark invented.

Capturing N iterations into one graph and replaying it reproduces that, and the
floor goes away (H200, 8 ranks, tokens/rank on the left):

    prepare   8 tok: eager 78.05 -> graph 15.51 us   (trace: 15.60, -0.6%)
    combine   8 tok: eager 36.22 -> graph 12.22 us   (trace: 15.12)
    dispatch  8 tok: eager 13.60 -> graph  7.96 us   (trace: 10.27)
    dispatch  8192 : eager 1679  -> graph 1676   us  (trace: 1922)

The eager/graph ratio falls monotonically to 1.0 as tokens grow (5.0x, 4.9x,
3.4x, 2.5x, 1.1x) — the signature of a fixed per-call cost, not of a transfer
that scales oddly.

What is left is a uniform 6-22% *under*-read against the trace, and that has a
physical reading rather than an excuse: back-to-back replays lock the ranks
harder than the real loop, where these collectives sit between attention and
GEMM and the ranks arrive with real jitter. The graph number is a lower bound.

CUPTI was tried and rejected: it sums kernel durations, and for a collective the
peer spin-wait is *inside* the kernel, so it measures rank skew. At 1024 tokens
it returned 20x the eager value with 2-3x run-to-run spread, and per-kernel it
was non-monotonic in token count.

``MnnvlConfig`` needs a CPU communicator with allgather/bcast/barrier. vLLM
supplies one wrapping its own CPU process group; ``_TorchDistCommBackend`` below
is the same adapter over the group the launcher already created.
"""

from __future__ import annotations

from typing import Any

from profiling.runners.comm._batch import all_error, run_comm_batch
from profiling.runners.comm._launcher import TorchMpLauncher
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import RunnerResult

# Object collectives (allgather of the fabric handles) must not ride on NCCL:
# they are CPU-side and happen while no stream work is in flight.
_LAUNCH_BACKEND = "cpu:gloo,cuda:nccl"
# 512 MB, matching vLLM's `MnnvlConfig` for the same kernel.
_FABRIC_PAGE_SIZE = 1 << 29


def profile_moe_alltoall_batch(kwargs_list: list[dict]) -> list[RunnerResult]:
    """Profile a homogeneous chunk of dispatch/combine shapes with one spawn."""
    return _run_batch(kwargs_list, _alltoall_per_rank_batch)


def profile_moe_alltoall_prepare_batch(kwargs_list: list[dict]) -> list[RunnerResult]:
    """Profile a homogeneous chunk of metadata-preparation shapes with one spawn."""
    return _run_batch(kwargs_list, _prepare_per_rank_batch)


def _run_batch(kwargs_list: list[dict], per_rank_fn) -> list[RunnerResult]:
    if not kwargs_list:
        return []
    ep_size = int(kwargs_list[0]["ep_size"])
    if ep_size < 2:
        return all_error(len(kwargs_list), "MoE all-to-all needs at least 2 ranks")
    launcher = TorchMpLauncher(ep_size, backend=_LAUNCH_BACKEND)
    return run_comm_batch(launcher, per_rank_fn, kwargs_list)


class _TorchDistCommBackend:
    """flashinfer's `CommBackend` over the launcher's process group.

    Mirrors vLLM's `CustomCommunicator`. Kept structural rather than imported so
    the profiler does not depend on a vLLM checkout being present.
    """

    def __init__(self, group):
        self._group = group

    def Get_rank(self) -> int:
        import torch.distributed as dist

        return dist.get_rank(group=self._group)

    def Get_size(self) -> int:
        import torch.distributed as dist

        return dist.get_world_size(group=self._group)

    def allgather(self, data: Any):
        import torch.distributed as dist

        gathered = [None] * self.Get_size()
        dist.all_gather_object(gathered, data, group=self._group)
        return gathered

    def bcast(self, data: Any, root: int) -> Any:
        import torch.distributed as dist

        holder = [data]
        dist.broadcast_object_list(holder, src=root, group=self._group)
        return holder[0]

    def barrier(self) -> None:
        import torch.distributed as dist

        dist.barrier(group=self._group)

    def Split(self, color: int, key: int) -> _TorchDistCommBackend:
        return self


def _mnnvl_session(rank: int, world_size: int):
    """Map the EP group's fabric workspaces and return them with the API handle."""
    try:
        import torch
        import torch.distributed as dist
        from flashinfer.comm.mapping import Mapping
        from flashinfer.comm.mnnvl import MnnvlConfig
        from flashinfer.comm.trtllm_alltoall import MnnvlMoe
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "torch and flashinfer (comm.trtllm_alltoall) are required for the "
            "MNNVL MoE all-to-all runner"
        ) from exc

    # `tp_size=world_size` is what vLLM passes: the flashinfer kernel asserts the
    # workspace spans the whole EP group, which here is every launched rank.
    mapping = Mapping(world_size, rank, world_size, tp_size=world_size)
    config = MnnvlConfig(
        comm_backend=_TorchDistCommBackend(dist.group.WORLD),
        fabric_page_size=_FABRIC_PAGE_SIZE,
        allocation_granularity=0,
    )
    try:
        workspace = MnnvlMoe.get_moe_workspaces(mapping, config)
        prepare_workspace = MnnvlMoe.get_moe_prepare_workspace(mapping, config)
    except Exception as exc:  # noqa: BLE001 — fabric memory is a host capability
        raise KernelLaunchFailed(
            f"MNNVL workspace allocation failed (fabric memory unavailable?): {exc}"
        ) from exc
    return torch, MnnvlMoe, workspace, prepare_workspace


def _release_mnnvl_session(mnnvl_moe) -> None:
    """Unmap the fabric workspaces while the process group is still up.

    Not housekeeping — without it the rank never exits. flashinfer frees the
    mapping from `MnnvlMemory.__del__`, and that path calls
    `close_mnnvl_memory`, which asks the CPU communicator for the group size and
    then `cuMemUnmap`s every peer's handle. It is a collective. The launcher's
    `_torch_mp_entry` destroys the process group in its `finally`, so leaving the
    workspaces to garbage collection runs that collective against a group that no
    longer exists: `__del__`'s `sys.is_finalizing()` guard does not help, because
    the collection happens before finalisation begins.

    Observed cost of not doing this: each finished chunk left 8 ranks spinning
    (`R` state, rank 0 asleep, 1-7 in the driver) holding their GPU memory
    forever. They are invisible in the build log — the results had already been
    recorded — and they contaminate every later measurement on the same box.

    The workspaces are flashinfer class attributes, cached one-per-process, so
    clearing them here is what drops the last reference. The barrier on both
    sides keeps any rank from unmapping while a peer is still using the region.
    """
    import gc

    import torch.distributed as dist

    dist.barrier()
    mnnvl_moe.moe_workspace_tensor = None
    mnnvl_moe.moe_prepare_workspace_tensor = None
    mnnvl_moe.moe_workspace = None
    mnnvl_moe.moe_prepare_workspace = None
    mnnvl_moe.moe_mapping = None
    gc.collect()
    dist.barrier()


def _routing(torch, tokens: int, top_k: int, slot_count: int, seed: int):
    """`top_k` distinct expert slots per token, drawn uniformly, plus weights.

    Distinct per token because a router selects without replacement.

    Only `prepare` draws a routing table at all — the transfer builds its index
    tensors directly (`_rank_plan`), which is what lets its two row axes be set
    exactly instead of searched for.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.rand(tokens, slot_count, device="cuda", generator=generator)
    expert_ids = scores.topk(top_k, dim=1).indices.to(torch.int32)
    weights = torch.rand(tokens, top_k, device="cuda", generator=generator, dtype=torch.float32)
    return expert_ids.contiguous(), weights.contiguous()


def _destinations_per_token(ep_size: int, top_k: int, slot_count: int) -> float:
    """Expected distinct destination ranks one token reaches.

    A token picks `top_k` distinct slots out of `slot_count`; a rank owns
    `slot_count // ep_size` of them and is missed only if all `top_k` picks fall
    outside its shard. For GLM-5.2 (ep 8, top_k 8, 256 slots) that is 5.25 of 8,
    which is why a rank's send row count is ~5x its token count and not 8x.

    This is the one place the fan-out is defined. The runner uses it to size the
    dispatch input buffer, and the arch uses the same formula to turn a token
    count into a send-row count, so the two sides of the leaf agree.
    """
    experts_per_rank = slot_count // ep_size
    miss = 1.0
    for i in range(top_k):
        remaining_outside = slot_count - experts_per_rank - i
        if remaining_outside <= 0:
            return float(ep_size)
        miss *= remaining_outside / (slot_count - i)
    return ep_size * (1.0 - miss)


def _transfer_matrix(max_send_rows: int, max_recv_rows: int, ep_size: int) -> list[list[int]]:
    """Rows rank `r` sends to rank `d`, as an `ep_size x ep_size` matrix.

    A (max_send, max_recv) pair does not name a unique transfer, so it needs a
    canonical realisation. The rule: **the smaller side is balanced at its
    maximum and the larger side carries all the skew.** It is forced, not
    chosen — conservation makes `sum(send) == sum(recv) == total`, so a side
    whose maximum equals `total / ep_size` has no room to be anything but flat.
    It also puts the fully balanced transfer exactly on the diagonal, and makes
    the two off-diagonal halves the input-skew and output-skew cases the axis
    probe measured.

    Within those margins the matrix is the maximum-entropy (product) plan,
    `M[r][d] ~ send[r] * recv[d] / total`, rounded so the margins hold exactly.
    Not the northwest-corner plan, which would be banded and would silently
    leave peer links idle.
    """
    if max_send_rows < 1 or max_recv_rows < 1:
        raise KernelLaunchFailed("both row counts must be positive")
    if max(max_send_rows, max_recv_rows) > ep_size * min(max_send_rows, max_recv_rows):
        raise KernelLaunchFailed(
            f"max_send_rows={max_send_rows} and max_recv_rows={max_recv_rows} cannot "
            f"coexist in an ep_size={ep_size} group: every sent row is a received "
            "row, so the two maxima are within a factor of ep_size"
        )

    if max_send_rows >= max_recv_rows:
        total = max_recv_rows * ep_size
        recv = [max_recv_rows] * ep_size
        send = _one_hot_margin(total, max_send_rows, ep_size)
    else:
        total = max_send_rows * ep_size
        send = [max_send_rows] * ep_size
        recv = _one_hot_margin(total, max_recv_rows, ep_size)
    return _product_plan(send, recv, total)


def _one_hot_margin(total: int, peak: int, ep_size: int) -> list[int]:
    """`ep_size` non-negative counts summing to `total`, whose maximum is `peak`.

    Rank 0 takes the peak; the rest split what is left as evenly as integers
    allow, with the remainder spread one row at a time so no rank accidentally
    overtakes the peak.
    """
    margin = [0] * ep_size
    margin[0] = peak
    remaining = total - peak
    if ep_size > 1:
        base, extra = divmod(remaining, ep_size - 1)
        for rank in range(1, ep_size):
            margin[rank] = base + (1 if rank <= extra else 0)
    return margin


def _product_plan(send: list[int], recv: list[int], total: int) -> list[list[int]]:
    """Integer matrix with row sums `send` and column sums `recv`.

    Floor the product plan, then hand out the residual rows to the cells with
    the largest dropped fraction that still have both a row and a column
    deficit. Such a cell always exists while any deficit remains (the row and
    column deficits sum to the same number), so this terminates exactly.
    """
    ep_size = len(send)
    plan = [[0] * ep_size for _ in range(ep_size)]
    fractions = []
    for r in range(ep_size):
        for d in range(ep_size):
            exact = send[r] * recv[d] / total
            plan[r][d] = int(exact)
            fractions.append((exact - plan[r][d], r, d))
    row_deficit = [send[r] - sum(plan[r]) for r in range(ep_size)]
    column_deficit = [recv[d] - sum(plan[r][d] for r in range(ep_size)) for d in range(ep_size)]
    fractions.sort(reverse=True)
    while any(deficit > 0 for deficit in row_deficit):
        placed = False
        for _, r, d in fractions:
            if row_deficit[r] > 0 and column_deficit[d] > 0:
                plan[r][d] += 1
                row_deficit[r] -= 1
                column_deficit[d] -= 1
                placed = True
                break
        if not placed:  # pragma: no cover — the margins guarantee a cell exists
            raise KernelLaunchFailed("transfer plan residual could not be placed")
    return plan


def _rank_plan(torch, plan: list[list[int]], rank: int, ep_size: int, input_rows: int):
    """This rank's `(send_cumsum, send_indices, recv_cumsum, recv_indices)`.

    `SendRecvDispls` wants an inclusive per-peer cumsum and, per transferred row,
    an index into the data tensor — the send side into `input`, the recv side
    into `output`.

    Send indices walk `input_rows` cyclically with a per-destination offset. When
    `input_rows` is smaller than the send total (dispatch, where one token row
    leaves for each of ~5 destinations) that reproduces the real read reuse
    while keeping the rows within one destination distinct, which is what a
    router selecting without replacement guarantees. Recv indices are contiguous:
    every arriving row lands in its own slot, in both directions.
    """
    send_counts = plan[rank]
    recv_counts = [plan[source][rank] for source in range(ep_size)]

    send_indices: list[int] = []
    for destination, count in enumerate(send_counts):
        offset = (destination * input_rows) // ep_size
        send_indices.extend((offset + i) % input_rows for i in range(count))

    recv_indices = list(range(sum(recv_counts)))

    def cumsum(counts: list[int]):
        running = 0
        out = []
        for count in counts:
            running += count
            out.append(running)
        return torch.tensor(out, dtype=torch.int32, device="cuda")

    return (
        cumsum(send_counts),
        torch.tensor(send_indices, dtype=torch.int32, device="cuda"),
        cumsum(recv_counts),
        torch.tensor(recv_indices, dtype=torch.int32, device="cuda"),
    )


# Graph sizing. One captured graph should hold enough iterations that the replay
# launch is negligible, and the whole measurement should stay short enough that a
# few hundred shapes finish in a coffee break. One call spans ~8 us at decode and
# ~13 ms at the top of the row axis, so a fixed count cannot serve both ends.
#
# The count comes from the bytes the shape moves, which the spec already states.
# It could instead come from timing the call first, and an earlier version did
# exactly that — and deadlocked: each rank timed its own eager loop, two ranks
# landed on either side of a rounding boundary, captured different iteration
# counts, and the ones that finished early spun inside the transfer waiting for
# peers that never came. Anything deciding HOW MANY collectives to launch must be
# a function of the spec, which every rank has, and not of a measurement, which
# is per-rank by nature. Deriving it from bytes removes the failure mode instead
# of synchronising around it.
#
# The budget only has to be right to an order of magnitude: the sweep spans 16
# doublings and this picks the plateau, not the value.
_GRAPH_BYTE_BUDGET = 8.0 * 1024**3
_MIN_ITERS_PER_GRAPH = 4
_MAX_ITERS_PER_GRAPH = 100
_REPLAYS = 20


def _graph_time_ms(torch, run_once, warmup: int, bytes_per_call: int) -> float:
    """Mean per-call time under CUDA-graph replay.

    See the module docstring for why a graph and not an eager loop.

    Capture needs its warmup on a side stream (torch requires the allocator to
    have seen the shapes before it starts recording), and a barrier on both
    sides: capture itself launches nothing, but the replays must start together
    or the first collective absorbs the skew.

    `bytes_per_call` comes from the spec, so every rank captures the same number
    of iterations without having to agree on anything at run time — see
    `_iters_per_graph`.

    A capture failure raises rather than falling back to eager timing. The eager
    number is not a degraded version of this one — it is 5x larger at decode
    scale — so a silent fallback would put a floor back into the curve with
    nothing in the row to say so.
    """
    import torch.distributed as dist

    iters_per_graph = _iters_per_graph(bytes_per_call)

    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(warmup):
            run_once()
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters_per_graph):
            run_once()

    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(_REPLAYS):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)

    del graph
    torch.cuda.empty_cache()
    return elapsed_ms / (_REPLAYS * iters_per_graph)


def _iters_per_graph(bytes_per_call: int) -> int:
    """How many calls to capture into one graph, from the bytes one call moves.

    Pure, and a function of the spec alone — that is the whole design constraint.
    Every rank must capture the same length or the replays launch different
    numbers of collectives and the group deadlocks, so this must not depend on
    anything measured, sampled, or rank-local.

    Small shapes get the maximum, which is where amortising the replay launch
    actually matters; large ones get few, which keeps the sweep's wall clock
    bounded (the top of the row axis moves ~800 MB per call).
    """
    if bytes_per_call <= 0:
        return _MAX_ITERS_PER_GRAPH
    affordable = int(_GRAPH_BYTE_BUDGET // bytes_per_call)
    return max(_MIN_ITERS_PER_GRAPH, min(_MAX_ITERS_PER_GRAPH, affordable))


def _bandwidths(moved_bytes: int, time_ms: float) -> tuple[float, float]:
    """algbw from the payload this rank moves; p2p-style, so busbw is the same."""
    latency_s = time_ms / 1000.0
    algbw_gbps = (moved_bytes / latency_s) / 1e9 if latency_s > 0 else 0.0
    return algbw_gbps, algbw_gbps


# The transfer's own warmup is short: it is one kernel with no autotuning, and
# every extra eager call pays the rendezvous floor for nothing.
_WARMUP_CALLS = 10


def _alltoall_per_rank_batch(
    *,
    rank: int,
    world_size: int,
    specs: list[dict],
    warmup: int,
    rep: int,
) -> list[dict] | None:
    """Runs inside each spawned rank. No per-shape try/except: the ranks must
    stay in lockstep or the transfer deadlocks (see `comm/_batch`).

    `warmup`/`rep` are the framework's eager-loop knobs and are ignored — the
    graph path sizes its capture from the spec's byte count, and `rep`'s 100
    would reserve tens of gigabytes at the top of the row axis.
    """
    torch, mnnvl_moe, workspace, _prepare_workspace = _mnnvl_session(rank, world_size)
    from flashinfer.comm.trtllm_alltoall import moe_comm

    results: list[dict] = []
    try:
        for spec in specs:
            results.append(_time_one_transfer(torch, moe_comm, workspace, spec, rank, world_size))
            # The buffers died with the call above; hand their blocks back before
            # the next shape asks for bigger ones.
            torch.cuda.empty_cache()
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
    finally:
        _release_mnnvl_session(mnnvl_moe)

    if rank != 0:
        return None
    return results


def _time_one_transfer(torch, moe_comm, workspace, spec: dict, rank: int, world_size: int) -> dict:
    """One grid cell: build the plan, allocate the two buffers, time `moe_comm`.

    A function rather than a loop body so the buffers go out of scope on return —
    at the top of the row axis they are hundreds of megabytes each, and holding a
    shape's allocation while the next one is built would double the peak.
    """
    max_send_rows = int(spec["max_send_rows"])
    max_recv_rows = int(spec["max_recv_rows"])
    top_k = int(spec["top_k"])
    slot_count = int(spec["slot_count"])
    hidden_bytes = int(spec["hidden_bytes"])
    direction = str(spec["direction"])
    if hidden_bytes % 16 != 0:
        raise KernelLaunchFailed(
            f"hidden_bytes {hidden_bytes} must be 16-byte aligned "
            "(moe_comm moves whole 16-byte vectors)"
        )
    hidden = hidden_bytes // 2

    plan = _transfer_matrix(max_send_rows, max_recv_rows, world_size)
    send_total = sum(plan[rank])
    recv_total = sum(plan[source][rank] for source in range(world_size))

    if direction == "dispatch":
        # The send buffer holds tokens, and one token row leaves for each distinct
        # rank it reaches, so it is smaller than the send total. It still has to
        # hold at least what goes to one peer, or that peer's rows could not be
        # distinct.
        fan_out = _destinations_per_token(world_size, top_k, slot_count)
        input_rows = max(1, round(send_total / fan_out), max(plan[rank]))
    else:
        # Combine reads each row of the expert output exactly once.
        input_rows = max(1, send_total)

    input_tensor = torch.randn(input_rows, hidden, dtype=torch.bfloat16, device="cuda")
    output_tensor = torch.empty(max(1, recv_total), hidden, dtype=torch.bfloat16, device="cuda")
    send_cumsum, send_indices, recv_cumsum, recv_indices = _rank_plan(
        torch, plan, rank, world_size, input_rows
    )

    def run_once():
        moe_comm(
            input_tensor,
            send_cumsum,
            send_indices,
            output_tensor,
            recv_cumsum,
            recv_indices,
            workspace,
            rank,
            world_size,
        )

    # Sizing takes the busier of the two sides at the CELL level — the spec's own
    # two maxima — never this rank's `send_total`/`recv_total`.
    #
    # Those two ARE rank-dependent, by construction: `_transfer_matrix` hands rank
    # 0 the whole skew (`sum(plan[0]) == max_send_rows`) and gives its peers less.
    # Feeding them to `_iters_per_graph` made ranks capture different iteration
    # counts, so they replayed different numbers of collectives: rank 0 ran out
    # first and sat in the barrier below while ranks 1-7 spun inside a transfer
    # whose last peer never arrived (GPU 0 at 0% util, 1-7 at 100%, forever).
    #
    # The maxima are the right budget anyway — they bound the busiest rank's
    # traffic, which is what the capture has to fit.
    graph_budget_bytes = max(max_send_rows, max_recv_rows) * hidden_bytes
    time_ms = _graph_time_ms(torch, run_once, _WARMUP_CALLS, graph_budget_bytes)
    algbw_gbps, busbw_gbps = _bandwidths(send_total * hidden_bytes, time_ms)
    return {"time_ms": time_ms, "algbw_gbps": algbw_gbps, "busbw_gbps": busbw_gbps}


def _prepare_per_rank_batch(
    *,
    rank: int,
    world_size: int,
    specs: list[dict],
    warmup: int,
    rep: int,
) -> list[dict] | None:
    """Times the metadata pass alone — the five index kernels plus the exchange.

    Still keyed by `tokens_per_rank`: this pass reads the `tokens x top_k`
    routing table and bins it over `ep_size x slot_count`, so its axis is the
    token count, not the rows the transfer later moves. Drawn uniformly, which
    under-reads a real skewed layer by ~9% at 8k tokens — 3% of the MoE comm
    budget, recorded in the alignment report rather than modelled.
    """
    torch, mnnvl_moe, _workspace, prepare_workspace = _mnnvl_session(rank, world_size)
    results: list[dict] = []
    try:
        for index, spec in enumerate(specs):
            results.append(
                _time_one_prepare(
                    torch, mnnvl_moe, prepare_workspace, spec, rank, world_size, seed=index
                )
            )
            torch.cuda.empty_cache()
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
    finally:
        _release_mnnvl_session(mnnvl_moe)

    if rank != 0:
        return None
    return results


def _time_one_prepare(
    torch, mnnvl_moe, prepare_workspace, spec: dict, rank: int, world_size: int, *, seed: int
) -> dict:
    """One grid cell of the metadata pass. A function for the same scope reason
    as `_time_one_transfer`: the routing table and the graph's captured output
    allocations both die on return."""
    tokens = int(spec["tokens_per_rank"])
    top_k = int(spec["top_k"])
    slot_count = int(spec["slot_count"])
    expert_ids, weights = _routing(torch, tokens, top_k, slot_count, seed=seed)

    def run_once():
        return mnnvl_moe.mnnvl_moe_alltoallv_prepare_without_allgather(
            expert_ids,
            weights,
            None,
            prepare_workspace,
            tokens,
            rank,
            world_size,
            slot_count,
            slot_count,
            top_k,
        )

    # Unlike the transfer, prepare allocates its outputs, so every captured
    # iteration keeps a `tokens x ep_size x top_k` expert table and an equally
    # sized scale table alive in the graph's private pool. That, not the routing
    # table it reads, is what bounds the capture.
    allocated_bytes_per_call = tokens * world_size * top_k * (4 + 4)
    time_ms = _graph_time_ms(torch, run_once, _WARMUP_CALLS, allocated_bytes_per_call)
    # The metadata pass moves the routing table, not the activations.
    index_bytes = tokens * top_k * 4
    algbw_gbps, busbw_gbps = _bandwidths(index_bytes, time_ms)
    return {"time_ms": time_ms, "algbw_gbps": algbw_gbps, "busbw_gbps": busbw_gbps}
