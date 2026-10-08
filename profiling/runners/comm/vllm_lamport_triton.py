"""vLLM Inkling Lamport RS + short-conv + AG + residual + RMSNorm runner.

Times the production public call
``vllm.models.inkling.nvidia.ops.lamport.LamportRSConv.rs_sconv_ag_add_norm``
unmodified: four PDL-chained Triton launches (publish input, wait + reduce +
conv-state insert, short conv + residual + publish, wait + gather + RMSNorm)
coupled by cross-GPU Lamport spin-waits over torch symmetric memory. One row is
one whole call.

``LamportRSConv`` takes its TP group from ``vllm.distributed.get_tp_group()``,
so each rank initializes vLLM's real distributed state
(``VllmTpGroupLauncher``) instead of stubbing the group.

Inputs follow the production producer contract. The partials are TP-partial
``[T, H]`` sublayer outputs (plus the MoE sink partial when ``has_shared``);
the conv-state cache is Inkling's paged per-layer SWA cache
``[num_blocks, H_rank, block_size, D]`` whose layout the runner derives from the
Inkling config (``sconv_swa_attn.py``): ``ws = hidden / num_kv_heads``,
``H_rank = (hidden / tp) / ws``, ``D = next_pow2(2·head_dim + 2·ws)``, and the
stream offset ``2·head_dim`` (attention sublayer) or ``2·head_dim + ws`` (MLP
sublayer). The default token layout is decode-like: one token per request at
a random position, its three prior taps pre-populated in scattered physical
blocks. ``PREFILL_LAYOUT`` (one request, consecutive positions, taps from the
same call) exists for the sensitivity check only.

Timing mirrors the FlashInfer fused all-reduce runner: the call runs inside a
CUDA graph (``_CALLS_PER_GRAPH`` back-to-back calls, a multiple of the three
Lamport buffer generations, as production's 84 calls per graph are). After a
host barrier aligns the ranks, wall-clock replay time per call is taken, the
median of three trials per rank, and the slowest rank is kept. Every rank runs
the same calls, so the spin-waits see balanced arrival; serving-time cross-rank
skew is not folded in.

The ``served`` regime (backend ``vllm_lamport_triton_served``) times the call
the way a decode step runs it instead of back to back. Each call follows a
cuBLAS GEMM that writes its input partial (the o_proj / routed-output shape
``[T, hidden/tp] x [hidden/tp, hidden]``), so its first launch chains to a GEMM
under PDL, not to the previous call's gather. Every call owns its own
"layer" (input, residual, conv-state cache, conv/norm weights and GEMM weight,
one per decoder layer up to 42, as production's per-layer conv caches), so
everything not written by the immediate predecessor is cold in L2 when the call
starts, except the residual: each call without a sink partial takes the
previous call's residual output, as an attention call takes the residual the
previous layer's MoE call just wrote. A ``has_shared`` (MoE) call keeps its
own cold residual, since the routed experts stream far more than L2 between
it and the attention call that wrote it. With ``has_shared`` the sink partial
is produced on a side stream and joined by an event before the call, as vLLM's
aux-stream sinks are on a decode step; the two-parent join leaves the first
launch no PDL edge. The row is the marginal cost: replay time of
[predecessors, call] x N minus that of [predecessors] x N, per call, on the
slowest rank.
"""

from __future__ import annotations

import functools
import math
import statistics
import sys
import time
from collections import defaultdict

from profiling.db.args import DType
from profiling.runners.comm._batch import all_error, run_comm_batch
from profiling.runners.comm._launcher import VllmTpGroupLauncher
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import RunnerResult

# ``LamportRSConv.__init__`` accepts exactly these TP world sizes (lamport.py).
SUPPORTED_TP_SIZES: frozenset[int] = frozenset({2, 4, 8})
# ``_sconv_publish_kernel`` statically asserts a window of 4.
SUPPORTED_CONV_KERNEL_SIZE = 4
# ``lamport._MAX_TOKENS``: the symmetric buffers hold at most this many tokens.
MAX_TOKENS = 16384

# Inkling config constants that fix the conv-state cache layout
# (Inkling-Small: num_key_value_heads 8, head_dim 128).
INKLING_NUM_KV_HEADS = 8
INKLING_HEAD_DIM = 128
INKLING_RMS_NORM_EPS = 1e-6

DECODE_LAYOUT = "decode"
PREFILL_LAYOUT = "prefill"

BALANCED_REGIME = "balanced"
SERVED_REGIME = "served"
# One "layer" per call in the served regime: Inkling-Small has 42 decoder
# layers, each with its own conv-state cache. A multiple of the three Lamport
# buffer generations.
_SERVED_MAX_SLOTS = 42
# Device bytes the served regime may keep across its per-layer slots; large T
# keeps fewer (the call itself then streams far more than L2 holds).
_SERVED_SLOT_BUDGET_BYTES = 12 << 30
# The served regime's sink stand-in: [T, _SINK_K] x [_SINK_K, hidden] on the
# side stream, shorter than the main-stream predecessor so the main stream
# arrives last, as the routed call does against the aux-stream sinks.
_SINK_K = 256
_SERVED_TRIALS = 5

_CALLS_PER_GRAPH = 30
_TIMED_TRIALS = 3
_MIN_TIMED_SECONDS = 0.02
# Decode positions are drawn from [K-1, _MAX_DECODE_POSITION).
_MAX_DECODE_POSITION = 4096
# The prefill layout's first token position: a chunk continuing a cached prefix.
_PREFILL_START_POSITION = 1021
_SEED = 20261008
_RTOL = 1.6e-2
_ATOL = 1e-2


def profile_rs_sconv_ag_residual_rms_norm_batch(
    kwargs_list: list[dict],
) -> list[RunnerResult]:
    return run_lamport_batch(kwargs_list, layout=DECODE_LAYOUT)


def profile_rs_sconv_ag_residual_rms_norm_served_batch(
    kwargs_list: list[dict],
) -> list[RunnerResult]:
    return run_lamport_batch(kwargs_list, layout=DECODE_LAYOUT, regime=SERVED_REGIME)


def run_lamport_batch(
    kwargs_list: list[dict], *, layout: str, regime: str = BALANCED_REGIME
) -> list[RunnerResult]:
    """Spawn one vLLM TP group for the chunk and profile every spec in order."""
    if not kwargs_list:
        return []
    num_gpus = int(kwargs_list[0]["num_gpus"])
    if num_gpus not in SUPPORTED_TP_SIZES:
        sizes = "/".join(str(size) for size in sorted(SUPPORTED_TP_SIZES))
        return all_error(len(kwargs_list), f"Lamport RS+sconv supports TP {sizes}, got {num_gpus}")
    try:
        for spec in kwargs_list:
            _check_spec(spec, num_gpus)
    except ValueError as exc:
        return all_error(len(kwargs_list), str(exc))
    return run_comm_batch(
        VllmTpGroupLauncher(num_gpus),
        functools.partial(_per_rank_batch, layout=layout, regime=regime),
        kwargs_list,
    )


def _check_spec(spec: dict, num_gpus: int) -> None:
    if int(spec["num_gpus"]) != num_gpus:
        raise ValueError("a Lamport chunk must share one num_gpus")
    tokens = int(spec["num_tokens"])
    if not 1 <= tokens <= MAX_TOKENS:
        raise ValueError(f"num_tokens must be in [1, {MAX_TOKENS}], got {tokens}")
    if int(spec["conv_kernel_size"]) != SUPPORTED_CONV_KERNEL_SIZE:
        raise ValueError(
            f"Lamport short conv requires conv_kernel_size {SUPPORTED_CONV_KERNEL_SIZE}, "
            f"got {spec['conv_kernel_size']}"
        )
    if DType.from_value(spec["dtype"]) is not DType.BF16:
        raise ValueError(f"Lamport RS+sconv is bf16 only, got {spec['dtype']}")
    hidden = int(spec["hidden_dim"])
    shard = hidden // num_gpus
    if hidden % (2 * num_gpus) or hidden % INKLING_NUM_KV_HEADS:
        raise ValueError(f"hidden_dim {hidden} does not shard over TP {num_gpus}")
    if shard % (hidden // INKLING_NUM_KV_HEADS):
        raise ValueError(
            f"TP {num_gpus} exceeds the {INKLING_NUM_KV_HEADS} conv-state heads "
            "(sconv_swa_attn requires tp <= num_kv_heads)"
        )


def _imports():
    try:
        import torch
        import torch.distributed as dist
        from vllm.distributed import get_tp_group
        from vllm.models.inkling.nvidia.ops.lamport import LamportRSConv
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "the vLLM checkout with the Inkling model (vllm_upstream_fork_env) is required"
        ) from exc
    return torch, dist, get_tp_group, LamportRSConv


def _per_rank_batch(
    *,
    rank: int,
    world_size: int,
    specs: list[dict],
    warmup: int,
    rep: int,
    layout: str,
    regime: str = BALANCED_REGIME,
) -> list[dict] | None:
    torch, dist, get_tp_group, lamport_cls = _imports()
    tp_group = get_tp_group()
    if tp_group.world_size != world_size or tp_group.rank_in_group != rank:
        raise KernelLaunchFailed("vLLM TP group does not match the spawned ranks")

    by_hidden: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    for index, spec in enumerate(specs):
        by_hidden[int(spec["hidden_dim"])].append((index, spec))

    results: list[dict | None] = [None] * len(specs)
    for hidden, indexed in by_hidden.items():
        max_tokens = max(int(spec["num_tokens"]) for _, spec in indexed)
        state = lamport_cls(hidden, SUPPORTED_CONV_KERNEL_SIZE, max_tokens=max_tokens)
        for index, spec in indexed:
            results[index] = _profile_one(
                torch=torch,
                dist=dist,
                tp_group=tp_group,
                state=state,
                spec=spec,
                layout=layout,
                warmup=warmup,
                rep=rep,
                regime=regime,
            )
        del state
        torch.cuda.synchronize()
        tp_group.barrier()

    if rank != 0:
        return None
    return list(results)  # type: ignore[arg-type]


def _profile_one(
    *, torch, dist, tp_group, state, spec, layout, warmup, rep, regime=BALANCED_REGIME
) -> dict:
    rank, world = tp_group.rank_in_group, tp_group.world_size
    tokens = int(spec["num_tokens"])
    hidden = int(spec["hidden_dim"])
    has_shared = bool(spec["has_shared"])
    case = build_case(
        torch,
        rank=rank,
        world_size=world,
        tokens=tokens,
        hidden=hidden,
        has_shared=has_shared,
        layout=layout,
        device=torch.device("cuda", torch.cuda.current_device()),
    )

    def launch():
        return state.rs_sconv_ag_add_norm(
            case["input"],
            case["residual"],
            case["conv_weight"],
            case["norm_weight"],
            INKLING_RMS_NORM_EPS,
            case["cache"],
            case["positions"],
            case["block_table"],
            case["seq_idx"],
            case["slot_mapping"],
            case["off_s"],
            case["ws"],
            case["block_size"],
            shared_tensor=case["shared"],
        )

    errors = validate_case(torch, case, launch, rank=rank, world_size=world)
    failed = torch.tensor(
        0.0 if errors["ok"] else 1.0, dtype=torch.float32, device=case["residual"].device
    )
    dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=tp_group.device_group)
    if failed.item() != 0:
        raise KernelLaunchFailed(
            f"Lamport output disagrees with the torch reference at T={tokens} "
            f"has_shared={has_shared} (rank {rank}: {errors})"
        )
    if rank == 0:
        print(
            f"[lamport] T={tokens} has_shared={has_shared} layout={layout} reference "
            f"max_abs normed={errors['normed_abs']:.3g} residual={errors['residual_abs']:.3g} "
            f"cache={errors['cache_abs']:.3g}",
            file=sys.stderr,
            flush=True,
        )

    if regime == SERVED_REGIME:
        cases = served_slot_cases(
            torch,
            first=case,
            rank=rank,
            world_size=world,
            tokens=tokens,
            hidden=hidden,
            has_shared=has_shared,
            layout=layout,
        )
        time_ms = time_served(
            torch=torch,
            dist=dist,
            tp_group=tp_group,
            state=state,
            cases=cases,
            side_stream_shared=has_shared,
            chain_residual=not has_shared,
            warmup=warmup,
        )
    else:
        time_ms = _time_cuda_graph(
            torch=torch,
            dist=dist,
            tp_group=tp_group,
            launch=launch,
            device=case["residual"].device,
            warmup=warmup,
            rep=rep,
        )
    message_bytes = tokens * hidden * case["residual"].element_size()
    latency_s = time_ms / 1000.0
    algbw = message_bytes / latency_s / 1e9 if latency_s > 0 else 0.0
    return {
        "time_ms": time_ms,
        "algbw_gbps": algbw,
        "busbw_gbps": algbw * 2 * (world - 1) / world,
    }


def conv_cache_layout(hidden: int, world_size: int, has_shared: bool) -> dict[str, int]:
    """Inkling's per-rank conv-state cache geometry (``sconv_swa_attn.py``)."""
    ws = hidden // INKLING_NUM_KV_HEADS
    heads_per_rank = (hidden // world_size) // ws
    head_size = 1 << (2 * INKLING_HEAD_DIM + 2 * ws - 1).bit_length()
    # MoE (has_shared) calls are MLP-sublayer calls; the others are mostly
    # attention-sublayer calls. The offset only moves the stream within a page.
    off_s = 2 * INKLING_HEAD_DIM + (ws if has_shared else 0)
    return {"ws": ws, "heads_per_rank": heads_per_rank, "head_size": head_size, "off_s": off_s}


def build_case(
    torch,
    *,
    rank: int,
    world_size: int,
    tokens: int,
    hidden: int,
    has_shared: bool,
    layout: str,
    device,
) -> dict:
    """Allocate one call's inputs; identical on every rank except rank-owned shards."""
    kernel = SUPPORTED_CONV_KERNEL_SIZE
    block_size = kernel
    shard = hidden // world_size
    layout_info = conv_cache_layout(hidden, world_size, has_shared)
    ws, off_s = layout_info["ws"], layout_info["off_s"]
    bf16 = torch.bfloat16
    gen = torch.Generator(device=device).manual_seed(_SEED + 7919 * tokens + int(has_shared))

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen, device=device) * scale).to(bf16)

    partials = randn(world_size, tokens, hidden, scale=0.1)
    shared = randn(world_size, tokens, hidden, scale=0.1) if has_shared else None
    residual = randn(tokens, hidden)
    conv_weight = randn(hidden, kernel, scale=0.3)
    norm_weight = (1.0 + 0.1 * torch.randn(hidden, generator=gen, device=device)).to(bf16)

    if layout == DECODE_LAYOUT:
        positions = torch.randint(
            kernel - 1, _MAX_DECODE_POSITION, (tokens,), generator=gen, device=device
        )
        seq_idx = torch.arange(tokens, device=device, dtype=torch.int32)
        max_blocks = _MAX_DECODE_POSITION // block_size + 1
        num_blocks = 2 * tokens
        physical = torch.randperm(num_blocks, generator=gen, device=device).view(tokens, 2)
        block_table = torch.zeros(tokens, max_blocks, dtype=torch.int32, device=device)
        rows = torch.arange(tokens, device=device)
        first = (positions - (kernel - 1)) // block_size
        last = positions // block_size
        block_table[rows, first] = physical[:, 0].to(torch.int32)
        block_table[rows, last] = physical[:, 1].to(torch.int32)
    elif layout == PREFILL_LAYOUT:
        start = _PREFILL_START_POSITION
        positions = start + torch.arange(tokens, device=device)
        seq_idx = torch.zeros(tokens, device=device, dtype=torch.int32)
        first = (start - (kernel - 1)) // block_size
        last = (start + tokens - 1) // block_size
        num_blocks = last - first + 1
        block_table = torch.zeros(1, last + 1, dtype=torch.int32, device=device)
        block_table[0, first:] = torch.randperm(num_blocks, generator=gen, device=device).to(
            torch.int32
        )
    else:
        raise ValueError(f"unknown token layout {layout!r}")
    seq64 = seq_idx.to(torch.int64)
    slot_mapping = (
        block_table[seq64, positions // block_size].to(torch.int64) * block_size
        + positions % block_size
    )

    # The cached conv input held before the call, for every rank's channels.
    history = randn(num_blocks, block_size, hidden, scale=0.4)
    prior = []
    for tap in range(kernel - 1):
        source = (positions - (kernel - 1) + tap).clamp(min=0)
        block = block_table[seq64, source // block_size].to(torch.int64)
        prior.append(history[block, source % block_size])
    prior_taps = torch.stack(prior, dim=1)

    # This rank's paged cache: its channel shard of the history in its stream;
    # the other streams hold NaN so a misplaced read or write is detected.
    cache = torch.full(
        (num_blocks, layout_info["heads_per_rank"], block_size, layout_info["head_size"]),
        float("nan"),
        dtype=bf16,
        device=device,
    )
    own = history[:, :, rank * shard : (rank + 1) * shard]
    own = own.view(num_blocks, block_size, layout_info["heads_per_rank"], ws).permute(0, 2, 1, 3)
    cache[..., off_s : off_s + ws] = own

    return {
        "partials": partials,
        "shared_all": shared,
        "input": partials[rank].contiguous(),
        "shared": None if shared is None else shared[rank].contiguous(),
        "residual": residual,
        "conv_weight_full": conv_weight,
        "conv_weight": conv_weight[rank * shard : (rank + 1) * shard].contiguous(),
        "norm_weight": norm_weight,
        "prior_taps": prior_taps,
        "cache": cache,
        "positions": positions,
        "block_table": block_table,
        "seq_idx": seq_idx,
        "slot_mapping": slot_mapping,
        "off_s": off_s,
        "ws": ws,
        "block_size": block_size,
        "rank": rank,
        "shard": shard,
    }


def validate_case(torch, case: dict, launch, *, rank: int, world_size: int) -> dict:
    """Run one call and compare outputs and the updated conv state to torch."""
    from profiling.runners.comm.rs_sconv_ag_residual_rms_norm_reference import (
        rs_sconv_ag_residual_rms_norm_reference,
    )

    cache_before = case["cache"].clone()
    normed, residual_out = launch()
    torch.cuda.synchronize()
    ref_normed, ref_residual, ref_x = rs_sconv_ag_residual_rms_norm_reference(
        case["partials"],
        case["shared_all"],
        case["residual"],
        case["conv_weight_full"],
        case["norm_weight"],
        INKLING_RMS_NORM_EPS,
        case["prior_taps"],
        case["positions"],
        case["seq_idx"],
    )

    shard, ws, off_s, block_size = case["shard"], case["ws"], case["off_s"], case["block_size"]
    cache = case["cache"]
    slots = case["slot_mapping"]
    written = cache[slots // block_size, :, slots % block_size, off_s : off_s + ws]
    written = written.reshape(slots.numel(), shard)
    ref_written = ref_x[:, rank * shard : (rank + 1) * shard]

    expected_cache = cache_before.clone()
    expected_cache[slots // block_size, :, slots % block_size, off_s : off_s + ws] = (
        ref_written.reshape(slots.numel(), -1, ws).to(cache.dtype)
    )
    untouched = expected_cache.clone()
    untouched[slots // block_size, :, slots % block_size, off_s : off_s + ws] = 0
    actual_untouched = cache.clone()
    actual_untouched[slots // block_size, :, slots % block_size, off_s : off_s + ws] = 0
    rest_ok = bool(torch.equal(untouched.nan_to_num(7.0), actual_untouched.nan_to_num(7.0)))

    out = {}
    ok = rest_ok
    for name, actual, expected in (
        ("normed", normed, ref_normed),
        ("residual", residual_out, ref_residual),
        ("cache", written, ref_written),
    ):
        diff = (actual.float() - expected.float()).abs()
        bound = _ATOL + _RTOL * expected.float().abs()
        out[f"{name}_abs"] = float(diff.max().item())
        out[f"{name}_rel"] = float((diff / expected.float().abs().clamp(min=1e-3)).max().item())
        ok = ok and bool(torch.isfinite(actual).all()) and bool((diff <= bound).all())
    out["rest_of_cache_unchanged"] = rest_ok
    out["ok"] = ok
    return out


def _time_cuda_graph(*, torch, dist, tp_group, launch, device, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()

    # Production replays the call inside a CUDA graph (84 calls per forward).
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(_CALLS_PER_GRAPH):
            launch()
    torch.cuda.synchronize()

    tp_group.barrier()
    start = time.perf_counter()
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    estimate = (time.perf_counter() - start) / 2
    # Every rank must replay the same number of calls, or a spin-wait never ends.
    wanted = max(math.ceil(rep / _CALLS_PER_GRAPH), math.ceil(_MIN_TIMED_SECONDS / estimate))
    replays = torch.tensor(float(wanted), dtype=torch.float32, device=device)
    dist.all_reduce(replays, op=dist.ReduceOp.MAX, group=tp_group.device_group)
    replays = int(replays.item())

    trials = []
    for _ in range(_TIMED_TRIALS):
        torch.cuda.synchronize()
        tp_group.barrier()
        start = time.perf_counter()
        for _ in range(replays):
            graph.replay()
        torch.cuda.synchronize()
        trials.append((time.perf_counter() - start) / (replays * _CALLS_PER_GRAPH))
    local_ms = torch.tensor(statistics.median(trials) * 1000.0, dtype=torch.float32, device=device)
    dist.all_reduce(local_ms, op=dist.ReduceOp.MAX, group=tp_group.device_group)
    del graph
    return float(local_ms.item())


# Tensors one served-regime slot keeps; the rest of ``build_case`` is reference data.
_LAUNCH_KEYS = (
    "input",
    "shared",
    "residual",
    "conv_weight",
    "norm_weight",
    "cache",
    "positions",
    "block_table",
    "seq_idx",
    "slot_mapping",
    "off_s",
    "ws",
    "block_size",
)


def served_slot_count(tokens: int, hidden: int, world_size: int, has_shared: bool) -> int:
    """Per-layer slots for the served regime: up to 42, within the byte budget."""
    layout = conv_cache_layout(hidden, world_size, has_shared)
    cache_bytes = 2 * tokens * layout["heads_per_rank"] * SUPPORTED_CONV_KERNEL_SIZE
    cache_bytes *= layout["head_size"] * 2
    rows = 3 + int(has_shared)  # input, residual, (shared), one output row
    gemm_weight_bytes = (hidden // world_size) * hidden * 2
    per_slot = cache_bytes + rows * tokens * hidden * 2 + gemm_weight_bytes
    slots = min(_SERVED_MAX_SLOTS, _SERVED_SLOT_BUDGET_BYTES // per_slot)
    return max(3, slots - slots % 3)


def served_slot_cases(
    torch, *, first, rank, world_size, tokens, hidden, has_shared, layout, slots=None
) -> list[dict]:
    """``first`` plus fresh per-layer slots, each with its own buffers and GEMM weights.

    Every slot also carries the predecessor operands: ``pred_a`` (shared by all
    slots) times its own ``pred_w`` writes ``input``; with ``has_shared`` the
    side-stream ``sink_a x sink_w`` writes ``shared``.
    """
    if slots is None:
        slots = served_slot_count(tokens, hidden, world_size, has_shared)
    device = first["residual"].device
    bf16 = torch.bfloat16
    gen = torch.Generator(device=device).manual_seed(_SEED + 31 * tokens + rank)
    k = hidden // world_size
    pred_a = (torch.randn(tokens, k, generator=gen, device=device) * 0.05).to(bf16)
    sink_a = (torch.randn(tokens, _SINK_K, generator=gen, device=device) * 0.05).to(bf16)
    cases = []
    for slot in range(slots):
        full = (
            first
            if slot == 0
            else build_case(
                torch,
                rank=rank,
                world_size=world_size,
                tokens=tokens,
                hidden=hidden,
                has_shared=has_shared,
                layout=layout,
                device=device,
            )
        )
        kept = {key: full[key] for key in _LAUNCH_KEYS}
        del full
        kept["pred_a"] = pred_a
        kept["pred_w"] = (torch.randn(k, hidden, generator=gen, device=device) * 0.05).to(bf16)
        if has_shared:
            kept["sink_a"] = sink_a
            kept["sink_w"] = (torch.randn(_SINK_K, hidden, generator=gen, device=device) * 0.05).to(
                bf16
            )
        cases.append(kept)
    torch.cuda.synchronize()
    return cases


def time_served(
    *,
    torch,
    dist,
    tp_group,
    state,
    cases,
    side_stream_shared,
    warmup,
    chain_residual=False,
    min_calls=30,
) -> float:
    """Marginal per-call ms of the Lamport call after its served predecessors.

    Graph A replays, per slot, the predecessor GEMM and, for a slot with a sink
    partial, the sink GEMM (forked onto a side stream and joined when
    ``side_stream_shared``, else serial on the main stream); graph B adds the
    Lamport call after each. ``len(cases)`` must be a multiple of 3 (buffer
    generations). With ``chain_residual`` each call takes the previous call's
    residual output, as consecutive sublayers do in serving. Trials alternate A
    and B; each rank keeps its median, and the slowest rank's B minus the
    slowest rank's A is the marginal cost.
    """
    if len(cases) % 3:
        raise ValueError("served slots must be a multiple of the 3 Lamport generations")
    rounds = max(1, math.ceil(min_calls / len(cases)))
    side = torch.cuda.Stream()
    device = cases[0]["residual"].device

    def predecessors(case):
        if case["shared"] is None:
            torch.matmul(case["pred_a"], case["pred_w"], out=case["input"])
        elif side_stream_shared:
            main = torch.cuda.current_stream()
            side.wait_stream(main)
            with torch.cuda.stream(side):
                torch.matmul(case["sink_a"], case["sink_w"], out=case["shared"])
            torch.matmul(case["pred_a"], case["pred_w"], out=case["input"])
            main.wait_stream(side)
        else:
            torch.matmul(case["sink_a"], case["sink_w"], out=case["shared"])
            torch.matmul(case["pred_a"], case["pred_w"], out=case["input"])

    def call(case, residual=None):
        _, residual_out = state.rs_sconv_ag_add_norm(
            case["input"],
            case["residual"] if residual is None else residual,
            case["conv_weight"],
            case["norm_weight"],
            INKLING_RMS_NORM_EPS,
            case["cache"],
            case["positions"],
            case["block_table"],
            case["seq_idx"],
            case["slot_mapping"],
            case["off_s"],
            case["ws"],
            case["block_size"],
            shared_tensor=case["shared"],
        )
        return residual_out if chain_residual else None

    def calls_in_order():
        residual = None
        for case in cases:
            predecessors(case)
            residual = call(case, residual)

    for _ in range(max(1, warmup // len(cases))):
        calls_in_order()
    torch.cuda.synchronize()

    graph_a = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_a):
        for _ in range(rounds):
            for case in cases:
                predecessors(case)
    graph_b = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_b):
        for _ in range(rounds):
            calls_in_order()
    torch.cuda.synchronize()
    calls = rounds * len(cases)

    def replay_seconds(graph, replays):
        torch.cuda.synchronize()
        tp_group.barrier()
        start = time.perf_counter()
        for _ in range(replays):
            graph.replay()
        torch.cuda.synchronize()
        return time.perf_counter() - start

    estimate = replay_seconds(graph_b, 2) / 2
    wanted = torch.tensor(
        float(max(2, math.ceil(_MIN_TIMED_SECONDS / estimate))), dtype=torch.float32, device=device
    )
    dist.all_reduce(wanted, op=dist.ReduceOp.MAX, group=tp_group.device_group)
    replays = int(wanted.item())

    trials_a, trials_b = [], []
    for _ in range(_SERVED_TRIALS):
        trials_a.append(replay_seconds(graph_a, replays) / (replays * calls))
        trials_b.append(replay_seconds(graph_b, replays) / (replays * calls))
    medians = torch.tensor(
        [statistics.median(trials_a) * 1000.0, statistics.median(trials_b) * 1000.0],
        dtype=torch.float32,
        device=device,
    )
    dist.all_reduce(medians, op=dist.ReduceOp.MAX, group=tp_group.device_group)
    del graph_a, graph_b
    pred_ms, total_ms = (float(v) for v in medians.tolist())
    return total_ms - pred_ms
