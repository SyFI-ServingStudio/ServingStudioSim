"""Fixed-pattern MNNVL exchange runners (list-native, one spawn per chunk).

Times the repo's fixed-pattern exchange kernels (``mnnvl_fixed_exchange``):

* ``moe_ep_quantized_all_gather`` / ``mnnvl_lamport``: the four dispatch tensors
  all-gathered by direct stores into every peer's mapped landing rows;
* ``moe_ep_reduce_scatter`` / ``mnnvl_lamport``: bf16 partials pushed to their
  owners' landing rows and summed there (fp32, rank order);
* ``moe_ep_reduce_scatter`` / ``mnnvl_pull``: the partials already sit in each
  rank's mapped region (the producer wrote them there) and the owner reads its
  rows from every rank and sums them;
* ``moe_ep_quantize_and_all_gather`` / ``mnnvl_lamport``: bf16 rows
  NVFP4-quantized inside the dispatch push (quantization + all-gather in one
  launch).

One spawn per chunk maps the workspace once, sized for the chunk's largest
per-rank count, and loops every shape inside it. Every shape is first checked
bit for bit against a Torch reference (the fused quantization against vLLM's
own NVFP4 quantization), then timed exactly like the MNNVL
all-to-all runner (``runners/moe/flashinfer_mnnvl_alltoall.py``): N calls
captured in one CUDA graph, replayed back to back, CUDA-event time per call; the
slowest rank is kept (as the PyNCCL and MNNVL all-reduce runners do).
"""

from __future__ import annotations

from functools import partial

from profiling.runners.comm._batch import all_error, run_comm_batch
from profiling.runners.comm._launcher import TorchMpLauncher
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import RunnerResult

_LAUNCH_BACKEND = "cpu:gloo,cuda:nccl"
_GRAPH_BYTE_BUDGET = 8.0 * 1024**3
_MIN_ITERS_PER_GRAPH = 4
_MAX_ITERS_PER_GRAPH = 100
_REPLAYS = 20
_WARMUP_CALLS = 20
_SENTINEL = 0xFFFFFFFF
# The fused quantization's global scale: any positive value exercises the same
# path; this one keeps N(0, 3) activations inside e4m3 block scales.
_GLOBAL_SCALE = 448.0 * 6.0 / 20.0
# Legs whose spec describes NVFP4 dispatch rows (activation_dtype, top_k); the
# others move plain bf16 rows (dtype).
_NVFP4_LEGS = frozenset({"dispatch", "quant_dispatch"})
# Legs whose workspace holds plain bf16 rows in the dispatch landing.
_BF16_ROW_LEGS: frozenset[str] = frozenset()


def profile_moe_ep_quantized_all_gather_batch(kwargs_list: list[dict]) -> list[RunnerResult]:
    return _run(kwargs_list, "dispatch")


def profile_moe_ep_reduce_scatter_batch(kwargs_list: list[dict]) -> list[RunnerResult]:
    return _run(kwargs_list, "combine")


def profile_moe_ep_reduce_scatter_pull_batch(kwargs_list: list[dict]) -> list[RunnerResult]:
    return _run(kwargs_list, "pull")


def profile_moe_ep_quantize_and_all_gather_batch(kwargs_list: list[dict]) -> list[RunnerResult]:
    return _run(kwargs_list, "quant_dispatch")


def _run(kwargs_list: list[dict], leg: str) -> list[RunnerResult]:
    if not kwargs_list:
        return []
    for spec in kwargs_list:
        error = spec_error(spec, leg)
        if error:
            return all_error(len(kwargs_list), error)
    launcher = TorchMpLauncher(int(kwargs_list[0]["num_gpus"]), backend=_LAUNCH_BACKEND)
    return run_comm_batch(launcher, partial(_per_rank, leg=leg), kwargs_list)


def spec_error(spec: dict, leg: str) -> str | None:
    num_gpus = int(spec["num_gpus"])
    tokens = [int(t) for t in spec["per_rank_tokens"]]
    if not 2 <= num_gpus <= 8 or len(tokens) != num_gpus:
        return f"the fixed exchange runs 2..8 ranks with one count each, got {spec}"
    if spec.get("fabric", "nvlink") != "nvlink":
        return "the fixed exchange needs NVLink-mapped (MNNVL) memory"
    if sum(tokens) == 0:
        return "no tokens to exchange"
    hidden = int(spec["hidden_size"])
    if leg in _NVFP4_LEGS:
        if str(spec["activation_dtype"]) != "nvfp4_e2m1":
            return "the fixed exchange gathers NVFP4 activations only"
        if hidden % 256 or int(spec["top_k"]) % 4:
            return "hidden_size must be a multiple of 256 and top_k of 4 (16-byte planes)"
    else:
        if str(spec["dtype"]) != "bf16":
            return "the fixed exchange moves bf16 rows only here"
        if hidden % 8:
            return "hidden_size must be a multiple of 8"
    return None


# --------------------------------------------------------------------------- per-rank body
def _session(rank: int, world: int, specs: list[dict], bf16_rows: bool):
    try:
        import torch

        from profiling.runners.comm.mnnvl_fixed_exchange import DispatchRow, FixedExchange
    except ImportError as exc:
        raise ProfilerNotImplemented(f"fixed exchange needs torch and flashinfer: {exc}") from exc
    max_rows = max(int(t) for spec in specs for t in spec["per_rank_tokens"])
    hidden = int(specs[0]["hidden_size"])
    top_k = int(specs[0].get("top_k", 8))
    if any(int(s["hidden_size"]) != hidden or int(s.get("top_k", top_k)) != top_k for s in specs):
        raise KernelLaunchFailed("one chunk must share hidden_size and top_k")
    try:
        row = DispatchRow.bf16(hidden) if bf16_rows else None
        exchange = FixedExchange(rank, world, max_rows, hidden, top_k, row=row)
    except Exception as exc:  # noqa: BLE001 — fabric memory is a host capability
        raise KernelLaunchFailed(f"MNNVL workspace allocation failed: {exc}") from exc
    return torch, exchange


def _per_rank(*, rank, world_size, specs, warmup, rep, leg) -> list[dict] | None:
    time_leg = _TIME_LEG[leg]
    torch, exchange = _session(rank, world_size, specs, leg in _BF16_ROW_LEGS)
    results = []
    try:
        for spec in specs:
            counts = [int(t) for t in spec["per_rank_tokens"]]
            exchange.set_counts(counts)
            results.append(time_leg(torch, exchange, counts, rank, world_size))
            torch.cuda.empty_cache()
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
    finally:
        exchange.close()
    return results if rank == 0 else None


# ---------------------------------------------------------------------------------- timing
def _iters_per_graph(bytes_per_call: int) -> int:
    affordable = int(_GRAPH_BYTE_BUDGET // max(1, bytes_per_call))
    return max(_MIN_ITERS_PER_GRAPH, min(_MAX_ITERS_PER_GRAPH, affordable))


def _graph_time_ms(torch, run_once, bytes_per_call: int) -> float:
    """Mean per-call time under CUDA-graph replay, slowest rank (see module doc)."""
    import torch.distributed as dist

    iters = _iters_per_graph(bytes_per_call)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(_WARMUP_CALLS):
            run_once()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
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
    elapsed = start.elapsed_time(end) / (_REPLAYS * iters)
    del graph
    slowest = torch.tensor([elapsed], dtype=torch.float64, device="cuda")
    dist.all_reduce(slowest, op=dist.ReduceOp.MAX)
    return float(slowest.item())


def _metrics(time_ms: float, received_bytes: int, sent_bytes: int) -> dict:
    """algbw: the busiest direction's bytes on this rank; busbw: the same (point to point)."""
    seconds = time_ms / 1e3
    moved = max(received_bytes, sent_bytes)
    gbps = moved / seconds / 1e9 if seconds > 0 else 0.0
    return {"time_ms": time_ms, "algbw_gbps": gbps, "busbw_gbps": gbps}


def _busiest_gather_bytes(counts, row_bytes):
    total = sum(counts)
    received = max(total - c for c in counts) * row_bytes
    sent = max(c * (len(counts) - 1) for c in counts) * row_bytes
    return received, sent


def _busiest_scatter_bytes(counts, hidden):
    received, sent = _busiest_gather_bytes(counts, 2 * hidden)
    return sent, received  # the transpose of the gather


def _expect_equal(torch, name, got, want) -> None:
    if not torch.equal(got, want):
        raise KernelLaunchFailed(f"{name} differs from the reference")


# ------------------------------------------------------------- NVFP4 dispatch (all-gather)
def _words(torch, t):
    return t.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF


def _canonical(torch, payload, scales, weights, ids):
    """The gathered planes every rank must hold (the kernel's lossless wire rewrites)."""
    p = _words(torch, payload)
    magnitude = p & 0x77777777
    nonzero = (magnitude | (magnitude >> 1) | (magnitude >> 2)) & 0x11111111
    p = p & (0x77777777 | (nonzero << 3))
    s = _words(torch, scales)
    s = s.masked_fill(s == _SENTINEL, 0x7FFFFFFF)
    w = _words(torch, weights)
    w = w.masked_fill(w == _SENTINEL, 0x7FFFFFFF)
    return p, s, w, _words(torch, ids)


def _dispatch_inputs(torch, rank, rows, hidden, top_k):
    g = torch.Generator(device="cuda").manual_seed(1000 + rank)
    payload = torch.randint(
        0, 256, (rows, hidden // 2), generator=g, device="cuda", dtype=torch.int32
    ).to(torch.uint8)
    scales = torch.randint(
        0, 0x7F, (rows, hidden // 16), generator=g, device="cuda", dtype=torch.int32
    ).to(torch.uint8)
    weights = torch.rand((rows, top_k), generator=g, device="cuda", dtype=torch.float32)
    ids = torch.randint(0, 1 << 16, (rows, top_k), generator=g, device="cuda", dtype=torch.int32)
    if rows:
        payload.view(torch.int32)[0, 0] = int(0x88888888 - (1 << 32))  # eight -0 nibbles
        payload.view(torch.int32)[rows - 1, -1] = -1  # eight -6 nibbles = the sentinel
        weights.view(torch.int32)[rows - 1, 0] = -1  # a NaN word equal to the sentinel
    return payload, scales, weights, ids


def _out_planes(torch, exchange, total):
    row = exchange.row
    return [
        torch.empty((total, b), dtype=torch.uint8, device="cuda")
        for b in (row.payload, row.scales, row.weights, row.ids)
    ]


def _time_dispatch(torch, exchange, counts, rank, world):
    hidden, top_k = exchange.hidden_size, exchange.top_k
    total = sum(counts)
    inputs = _dispatch_inputs(torch, rank, counts[rank], hidden, top_k)
    out = _out_planes(torch, exchange, total)
    exchange.dispatch(*inputs, out)
    torch.cuda.synchronize()
    want = [
        torch.cat(parts)
        for parts in zip(
            *(
                _canonical(torch, *_dispatch_inputs(torch, r, counts[r], hidden, top_k))
                for r in range(world)
            )
        )
    ]
    for name, got, ref in zip(("payload", "scales", "weights", "ids"), out, want):
        _expect_equal(torch, f"dispatch {name}", _words(torch, got), ref)
    row_bytes = exchange.row.total
    time_ms = _graph_time_ms(torch, lambda: exchange.dispatch(*inputs, out), total * row_bytes)
    return _metrics(time_ms, *_busiest_gather_bytes(counts, row_bytes))


def _quant_inputs(torch, rank, rows, hidden, top_k):
    g = torch.Generator(device="cuda").manual_seed(2000 + rank)
    x = (torch.randn((rows, hidden), generator=g, device="cuda") * 3).to(torch.bfloat16)
    if rows:
        x[0, :32] = 0
        x[rows - 1, 16:48] = -1e-9
    weights = torch.rand((rows, top_k), generator=g, device="cuda", dtype=torch.float32)
    ids = torch.randint(0, 1 << 16, (rows, top_k), generator=g, device="cuda", dtype=torch.int32)
    return x, weights, ids


def _vllm_nvfp4(torch, x, global_scale, hidden):
    from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input

    q, s = moe_kernel_quantize_input(
        x,
        global_scale,
        quant_dtype="nvfp4",
        per_act_token_quant=False,
        block_shape=None,
        is_scale_swizzled=False,
    )
    rows = x.shape[0]
    return q.view(torch.uint8).reshape(rows, hidden // 2), s.view(torch.uint8).reshape(
        rows, hidden // 16
    )


def _time_quant_dispatch(torch, exchange, counts, rank, world):
    hidden, top_k = exchange.hidden_size, exchange.top_k
    total = sum(counts)
    global_scale = torch.tensor([_GLOBAL_SCALE], dtype=torch.float32, device="cuda")
    x, weights, ids = _quant_inputs(torch, rank, counts[rank], hidden, top_k)
    out = _out_planes(torch, exchange, total)
    exchange.quant_dispatch(x, global_scale, weights, ids, out)
    torch.cuda.synchronize()
    parts = []
    for r in range(world):
        xr, wr, idr = _quant_inputs(torch, r, counts[r], hidden, top_k)
        if counts[r]:
            q, s = _vllm_nvfp4(torch, xr, global_scale, hidden)
        else:
            q = torch.empty((0, hidden // 2), dtype=torch.uint8, device="cuda")
            s = torch.empty((0, hidden // 16), dtype=torch.uint8, device="cuda")
        parts.append(_canonical(torch, q, s, wr, idr))
    want = [torch.cat(p) for p in zip(*parts)]
    for name, got, ref in zip(("payload", "scales", "weights", "ids"), out, want):
        _expect_equal(
            torch, f"quant dispatch {name} (vs vLLM NVFP4 quant)", _words(torch, got), ref
        )
    row_bytes = exchange.row.total
    time_ms = _graph_time_ms(
        torch,
        lambda: exchange.quant_dispatch(x, global_scale, weights, ids, out),
        total * row_bytes,
    )
    return _metrics(time_ms, *_busiest_gather_bytes(counts, row_bytes))


# ------------------------------------------------------------- bf16 combine (reduce-scatter)
def _partials(torch, rank, total, hidden):
    g = torch.Generator(device="cuda").manual_seed(3000 + rank)
    return (torch.randn((total, hidden), generator=g, device="cuda") * 4).to(torch.bfloat16)


def _combine_reference(torch, counts, rank, world, hidden):
    total = sum(counts)
    first = sum(counts[:rank])
    acc = torch.zeros((counts[rank], hidden), dtype=torch.float32, device="cuda")
    for r in range(world):
        acc += _partials(torch, r, total, hidden)[first : first + counts[rank]].float()
    return acc.to(torch.bfloat16)


def _time_combine(torch, exchange, counts, rank, world):
    hidden = exchange.hidden_size
    total = sum(counts)
    partials = _partials(torch, rank, total, hidden)
    out = torch.empty((counts[rank], hidden), dtype=torch.bfloat16, device="cuda")
    exchange.combine(partials, out)
    torch.cuda.synchronize()
    _expect_equal(torch, "combine", out, _combine_reference(torch, counts, rank, world, hidden))
    time_ms = _graph_time_ms(torch, lambda: exchange.combine(partials, out), total * hidden * 2)
    return _metrics(time_ms, *_busiest_scatter_bytes(counts, hidden))


def _time_pull(torch, exchange, counts, rank, world):
    import torch.distributed as dist

    hidden = exchange.hidden_size
    total = sum(counts)
    partials = _partials(torch, rank, total, hidden)
    # Both sets hold the same partials, so graph replays of either round parity read them.
    for set_index in (0, 1):
        exchange.partials(set_index, total).copy_(partials)
    torch.cuda.synchronize()
    dist.barrier()
    out = torch.empty((counts[rank], hidden), dtype=torch.bfloat16, device="cuda")
    exchange.pull_combine(out)
    torch.cuda.synchronize()
    _expect_equal(
        torch, "pull combine", out, _combine_reference(torch, counts, rank, world, hidden)
    )
    time_ms = _graph_time_ms(torch, lambda: exchange.pull_combine(out), total * hidden * 2)
    dist.barrier()
    return _metrics(time_ms, *_busiest_scatter_bytes(counts, hidden))


# Leg name -> ``(torch, exchange, counts, rank, world) -> metrics dict``.
_TIME_LEG = {
    "dispatch": _time_dispatch,
    "quant_dispatch": _time_quant_dispatch,
    "combine": _time_combine,
    "pull": _time_pull,
}
