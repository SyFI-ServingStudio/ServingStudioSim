"""Profile Inkling's fused QKVR prep (``fused_qkvr_prep``) from the alignment fork.

``vllm_triton`` times one call of
``vllm.models.inkling.nvidia.ops.qkvr_prep.fused_qkvr_prep`` with the arguments
``InklingAttention.forward`` passes (``attention.py``): the per-rank ``qkvr``
GEMM output, the K/V short-conv weights, the q/k RMSNorm weights, the
``rel_logits_proj`` matrix, the layer's conv-state cache and its LBNHC paged
KV cache, and ``tau`` on global layers only.

The call switches implementation inside itself. Below 128 tokens it is one
``_qkvr_qkv_kernel`` launch. From 128 tokens it forks: ``_kv_kernel`` runs on
vLLM's global ``aux_stream()`` while ``_q_kernel`` and then one of two
rel-projection kernels run on the current stream
(``_rel_proj_throughput_kernel`` once ``8 * num_tokens`` reaches 8192 rows for
``rel_extent`` 512 or 2048 rows otherwise; ``_rel_proj_low_latency_kernel``
below), and the current stream joins the side stream. The L1 unit is the whole
call, priced by its wall span, never the sum of its kernels.

Production runs this call inside a CUDA graph (FULL graphs for decode, the
breakable graph for mixed batches up to the 2048-token capture size), so the
runner captures one call into a CUDA graph and times graph replays: CUPTI
records each replay's kernels and ``wall_span`` prices the replay from its
first kernel start to its last kernel end, so the two streams' overlap counts
once and the in-graph gaps between kernels count too. Eager launches would
instead price the Python launch gaps of the three Triton calls.

Inputs follow the layer's producer contract: one request per token, each with
a random earlier context so all ``W - 1`` conv taps come from the conv-state
cache, as in decode (see ``timing_layout``). Outside the timed region the call's
outputs are checked against the independent Torch reference in
``inkling_qkvr_prep_reference``: q, rel logits, the K/V rows written to the
attention cache, the raw K/V written to the conv-state cache, and that every
other cache slot is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention.inkling_qkvr_prep_reference import (
    PrepShape,
    compare,
    reference_conv,
    reference_prep,
)
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

KIND = "inkling_qkvr_prep"
_BACKEND = f"{KIND}:vllm_triton"
_MAX_TOKENS = 65_536
# max_model_len of the measured deployment; positions stay below it.
_MAX_POSITION = 8192
_EPS = 1e-6
# Per-head packed width of the conv-state cache: K | V | attn-out | mlp-out,
# padded to a power of two (sconv_swa_attn.py). 2048 for Inkling-Small at TP4;
# it only sets a stride.
_CONV_HEAD_WIDTH = 2048
_SEED = 0x1A7C
_KV_POISON = 77.0
# bf16 outputs; both sides round at the same points, so differences are
# reduction-order ulps. 3 ulps at the reference's largest magnitude.
_MAX_REL_ERR = 3 * 2.0**-8


@dataclass(frozen=True)
class Layout:
    """Requests of one forward: ``(context_len, new_tokens)`` each, plus pad rows.

    ``context_len`` None draws a random earlier context (see ``build_inputs``).
    """

    requests: tuple[tuple[int | None, int], ...]
    num_pad: int = 0

    @property
    def num_tokens(self) -> int:
        return sum(n for _, n in self.requests) + self.num_pad


def timing_layout(num_tokens: int) -> Layout:
    """The batch the timed call sees: ``num_tokens`` decode tokens.

    Every token is its own request with an earlier context, so all ``W - 1``
    taps are read from the conv-state cache, as in production decode. A
    chunked-prefill batch reads most taps from ``qkvr`` instead and runs 5-20%
    faster on B200 (numbers in the kernel module); the schema keeps no
    tap-source axis, so rows are an upper bound for prefill-heavy batches.
    """

    return Layout(requests=((None, 1),) * num_tokens)


def _validate_args(
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    d_rel: int,
    rel_extent: int,
    apply_log_scaling: bool,
    sconv_kernel_size: int,
    conv_block_size: int,
    page_size: int,
    dtype: object,
) -> None:
    if type(num_tokens) is not int or not 1 <= num_tokens <= _MAX_TOKENS:
        raise ValueError(f"num_tokens must be an integer in [1, {_MAX_TOKENS}]")
    for name, value in (
        ("num_q_heads", num_q_heads),
        ("num_kv_heads", num_kv_heads),
        ("rel_extent", rel_extent),
        ("sconv_kernel_size", sconv_kernel_size),
        ("conv_block_size", conv_block_size),
        ("page_size", page_size),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if num_q_heads % num_kv_heads:
        raise ValueError("num_q_heads must be a multiple of num_kv_heads")
    if not isinstance(apply_log_scaling, bool):
        raise ValueError("apply_log_scaling must be a bool")
    # fused_qkvr_prep asserts both (qkvr_prep.py:820-821).
    if head_dim != 128:
        raise ProfilerNotImplemented(f"{_BACKEND} supports head_dim 128")
    if d_rel != 16:
        raise ProfilerNotImplemented(f"{_BACKEND} supports d_rel 16")
    # The conv-state cache is bf16 only (sconv_swa_attn.py).
    if str(dtype) != "bf16":
        raise ProfilerNotImplemented(f"{_BACKEND} supports bf16")
    if 2 * head_dim > _CONV_HEAD_WIDTH:
        raise ProfilerNotImplemented(f"{_BACKEND} packs K and V into {_CONV_HEAD_WIDTH} columns")


@dataclass
class PrepInputs:
    shape: PrepShape
    qkvr: Any
    k_weight: Any
    v_weight: Any
    q_norm_weight: Any
    k_norm_weight: Any
    rel_proj: Any
    conv_cache: Any  # logical [blocks, H, N, D], LBNHC storage
    kv_storage: Any  # [pages, page_size, H, 2 * head_dim]
    key_cache: Any
    value_cache: Any
    positions: Any
    conv_block_table: Any
    seq_idx: Any
    conv_slot_mapping: Any
    query_start: Any
    attention_slot_mapping: Any
    tau: Any | None
    conv_block_size: int
    # Reference bookkeeping (not passed to the call).
    first_new_position: Any
    first_row: Any


def build_inputs(
    torch: Any,
    device: Any,
    shape: PrepShape,
    layout: Layout,
    *,
    apply_log_scaling: bool,
    conv_block_size: int,
    page_size: int,
) -> PrepInputs:
    """Seeded inputs, conv-state cache and paged KV cache for one layer call.

    A request given as ``(None, n)`` gets a random context in
    ``[W - 1, _MAX_POSITION - n]``. Each request owns distinct conv blocks for
    the positions it reads or writes and distinct attention pages for the
    positions it writes; block 0 / page 0 is the null entry everything else
    in the block table points to.
    """

    gen = torch.Generator().manual_seed(_SEED)
    width = shape.sconv_kernel_size
    requests = []
    for context, new in layout.requests:
        if context is None:
            high = max(width, _MAX_POSITION - new)
            context = int(torch.randint(width - 1, high, (1,), generator=gen).item())
        requests.append((context, new))
    real = sum(n for _, n in requests)
    tokens = real + layout.num_pad

    positions, seq_idx, query_start, first_pos, conv_slots, attn_slots = ([] for _ in range(6))
    max_blocks = max(-(-(c + n) // conv_block_size) for c, n in requests)
    block_table = torch.zeros((len(requests), max_blocks), dtype=torch.int32)
    next_block, next_page, row = 1, 1, 0
    for r, (context, new) in enumerate(requests):
        first_block = max(0, context - (width - 1)) // conv_block_size
        last_block = (context + new - 1) // conv_block_size
        for logical in range(first_block, last_block + 1):
            block_table[r, logical] = next_block
            next_block += 1
        page_of: dict[int, int] = {}
        for i in range(new):
            pos = context + i
            positions.append(pos)
            seq_idx.append(r)
            query_start.append(row)
            first_pos.append(context)
            conv_slots.append(
                int(block_table[r, pos // conv_block_size]) * conv_block_size
                + pos % conv_block_size
            )
            if pos // page_size not in page_of:
                page_of[pos // page_size] = next_page
                next_page += 1
            attn_slots.append(page_of[pos // page_size] * page_size + pos % page_size)
        row += new
    for _ in range(layout.num_pad):
        # CUDA-graph padding rows: no conv or attention slot.
        positions.append(0)
        seq_idx.append(0)
        query_start.append(0)
        first_pos.append(0)
        conv_slots.append(-1)
        attn_slots.append(-1)
    first_row = torch.tensor(query_start, dtype=torch.int64)

    qkvr = torch.randn((tokens, shape.row_width), generator=gen).to(torch.bfloat16)
    k_weight = (0.3 * torch.randn((shape.kv_width, width), generator=gen)).to(torch.bfloat16)
    v_weight = (0.3 * torch.randn((shape.kv_width, width), generator=gen)).to(torch.bfloat16)
    q_norm_weight = (1 + 0.1 * torch.randn(shape.head_dim, generator=gen)).to(torch.bfloat16)
    k_norm_weight = (1 + 0.1 * torch.randn(shape.head_dim, generator=gen)).to(torch.bfloat16)
    rel_proj = (0.25 * torch.randn((shape.d_rel, shape.rel_extent), generator=gen)).to(
        torch.bfloat16
    )
    # Conv-state cache in LBNHC storage ([block, N, H, D]) viewed as [block, H, N, D].
    # Only the K and V streams are read; the attn-out / mlp-out columns stay zero.
    conv_storage = torch.zeros(
        (next_block, conv_block_size, shape.num_kv_heads, _CONV_HEAD_WIDTH), dtype=torch.bfloat16
    )
    conv_storage[..., : 2 * shape.head_dim] = torch.randn(
        (next_block, conv_block_size, shape.num_kv_heads, 2 * shape.head_dim), generator=gen
    ).to(torch.bfloat16)
    kv_storage = torch.full(
        (next_page, page_size, shape.num_kv_heads, 2 * shape.head_dim),
        _KV_POISON,
        dtype=torch.bfloat16,
    )
    # tau's values do not change the work; draw them away from 1.0 so the
    # correctness check covers the multiply on global layers.
    tau = (1.0 + 0.3 * torch.rand(tokens, generator=gen)) if apply_log_scaling else None

    to = lambda t: None if t is None else t.to(device)  # noqa: E731
    kv_storage = to(kv_storage)
    return PrepInputs(
        shape=shape,
        qkvr=to(qkvr),
        k_weight=to(k_weight),
        v_weight=to(v_weight),
        q_norm_weight=to(q_norm_weight),
        k_norm_weight=to(k_norm_weight),
        rel_proj=to(rel_proj),
        conv_cache=to(conv_storage).transpose(1, 2),
        kv_storage=kv_storage,
        key_cache=kv_storage[..., : shape.head_dim],
        value_cache=kv_storage[..., shape.head_dim :],
        positions=to(torch.tensor(positions, dtype=torch.int64)),
        conv_block_table=to(block_table),
        seq_idx=to(torch.tensor(seq_idx, dtype=torch.int32)),
        conv_slot_mapping=to(torch.tensor(conv_slots, dtype=torch.int64)),
        query_start=to(torch.tensor(query_start, dtype=torch.int32)),
        attention_slot_mapping=to(torch.tensor(attn_slots, dtype=torch.int64)),
        tau=to(tau),
        conv_block_size=conv_block_size,
        first_new_position=to(torch.tensor(first_pos, dtype=torch.int64)),
        first_row=to(first_row),
    )


def call_prep(fused_qkvr_prep: Any, x: PrepInputs) -> tuple[Any, Any]:
    """One ``fused_qkvr_prep`` call with the layer's argument order."""
    s = x.shape
    return fused_qkvr_prep(
        x.qkvr,
        x.k_weight,
        x.v_weight,
        x.q_norm_weight,
        x.k_norm_weight,
        x.rel_proj,
        _EPS,
        s.num_q_heads,
        s.num_kv_heads,
        s.head_dim,
        s.d_rel,
        x.conv_cache,
        x.key_cache,
        x.value_cache,
        x.positions,
        x.conv_block_table,
        x.seq_idx,
        x.conv_slot_mapping,
        x.query_start,
        x.attention_slot_mapping,
        0,  # off_k: K is the first packed stream
        s.head_dim,  # off_v
        x.conv_block_size,
        x.tau,
    )


def check_outputs(torch: Any, x: PrepInputs, conv_before: Any, q_out: Any, rel_out: Any) -> dict:
    """Compare one call's outputs and cache writes with the Torch reference.

    ``conv_before`` is a copy of the conv-state cache taken before the call
    (the cached taps the reference reads). Raises ``KernelLaunchFailed`` on a
    mismatch; returns the error report.
    """

    s = x.shape
    valid = x.conv_slot_mapping >= 0
    cb = x.conv_block_size
    hd = s.head_dim

    def cached_tap(pos: Any, rows: Any) -> tuple[Any, Any]:
        req = x.seq_idx[rows].long()
        block = x.conv_block_table[req, pos // cb].long()
        cell = conv_before[block, :, pos % cb]  # [n, H, D]
        return cell[..., :hd].reshape(len(rows), -1), cell[..., hd : 2 * hd].reshape(len(rows), -1)

    k_conv, v_conv = reference_conv(
        torch,
        s,
        x.qkvr,
        x.k_weight,
        x.v_weight,
        x.positions,
        x.first_new_position,
        x.first_row,
        cached_tap,
    )
    ref = reference_prep(
        torch, s, x.qkvr, k_conv, v_conv, x.q_norm_weight, x.k_norm_weight, x.rel_proj, x.tau, _EPS
    )
    report: dict[str, Any] = {
        "q": compare(torch, q_out[valid], ref["q"][valid]),
        "rel_logits": compare(torch, rel_out[valid], ref["rel"][valid]),
    }

    slots = x.attention_slot_mapping[valid]
    page, off = slots // x.key_cache.shape[1], slots % x.key_cache.shape[1]
    report["k_cache"] = compare(torch, x.key_cache[page, off], ref["k_cache_rows"][valid])
    report["v_cache"] = compare(torch, x.value_cache[page, off], ref["v_cache_rows"][valid])

    conv_slots = x.conv_slot_mapping[valid]
    blk, cell = conv_slots // cb, conv_slots % cb
    written = x.conv_cache[blk, :, cell]  # [n, H, D]
    q_w, kv_w = s.q_width, s.kv_width
    raw_k = x.qkvr[valid, q_w : q_w + kv_w].view(-1, s.num_kv_heads, hd)
    raw_v = x.qkvr[valid, q_w + kv_w : q_w + 2 * kv_w].view(-1, s.num_kv_heads, hd)
    report["conv_cache_raw_kv_exact"] = bool(
        torch.equal(written[..., :hd], raw_k) and torch.equal(written[..., hd : 2 * hd], raw_v)
    )

    conv_mask = torch.ones(x.conv_cache.shape[0], cb, dtype=torch.bool, device=x.qkvr.device)
    conv_mask[blk, cell] = False
    untouched_conv = x.conv_cache.transpose(1, 2)[conv_mask]
    report["conv_cache_others_untouched"] = bool(
        torch.equal(untouched_conv, conv_before.transpose(1, 2)[conv_mask])
        and torch.equal(written[..., 2 * hd :], conv_before[blk, :, cell][..., 2 * hd :])
    )
    kv_mask = torch.ones(x.kv_storage.shape[:2], dtype=torch.bool, device=x.qkvr.device)
    kv_mask[page, off] = False
    report["kv_cache_others_untouched"] = bool((x.kv_storage[kv_mask] == _KV_POISON).all().item())

    worst = max(report[k]["max_rel"] for k in ("q", "rel_logits", "k_cache", "v_cache"))
    report["max_rel"] = worst
    report["max_abs"] = max(report[k]["max_abs"] for k in ("q", "rel_logits", "k_cache", "v_cache"))
    if (
        worst > _MAX_REL_ERR
        or not report["conv_cache_raw_kv_exact"]
        or not report["conv_cache_others_untouched"]
        or not report["kv_cache_others_untouched"]
    ):
        raise KernelLaunchFailed(f"{_BACKEND} disagrees with the Torch reference: {report}")
    return report


def capture_graph(torch: Any, launch: Any) -> tuple[Any, Any]:
    """Warm ``launch`` on a side stream, then capture one call into a CUDA graph."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            launch()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = launch()
    torch.cuda.synchronize()
    return graph, outputs


def logical_bytes(shape: PrepShape, num_tokens: int) -> int:
    """Bytes one call must move (bf16): read the qkvr row and rel_proj; write q,
    rel logits, the attention-cache K/V and the conv-cache raw K/V. Cached conv
    taps and weights other than rel_proj are not counted."""
    per_token = (
        shape.row_width  # qkvr read
        + shape.q_width  # q write
        + shape.num_q_heads * shape.rel_extent  # rel logits write
        + 2 * shape.kv_width  # attention cache K/V write
        + 2 * shape.kv_width  # conv cache raw K/V write
    )
    return 2 * (per_token * num_tokens + shape.d_rel * shape.rel_extent)


def profile_inkling_qkvr_prep_vllm_triton(
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    d_rel: int,
    rel_extent: int,
    apply_log_scaling: bool,
    sconv_kernel_size: int,
    conv_block_size: int,
    page_size: int,
    dtype: object,
) -> ComputeMetrics:
    """Time one ``fused_qkvr_prep`` call (CUDA-graph replay, wall span)."""
    _validate_args(
        num_tokens,
        num_q_heads,
        num_kv_heads,
        head_dim,
        d_rel,
        rel_extent,
        apply_log_scaling,
        sconv_kernel_size,
        conv_block_size,
        page_size,
        dtype,
    )
    try:
        import torch
        from vllm.models.inkling.nvidia.ops.qkvr_prep import fused_qkvr_prep
    except ImportError as exc:
        raise ProfilerNotImplemented(
            f"{_BACKEND} requires the alignment vLLM fork with Inkling (vllm_upstream_fork_env)"
        ) from exc
    shape = PrepShape(num_q_heads, num_kv_heads, head_dim, d_rel, rel_extent, sconv_kernel_size)
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        x = build_inputs(
            torch,
            device,
            shape,
            timing_layout(num_tokens),
            apply_log_scaling=apply_log_scaling,
            conv_block_size=conv_block_size,
            page_size=page_size,
        )
        conv_before = x.conv_cache.clone()
        q_out, rel_out = call_prep(fused_qkvr_prep, x)
        torch.cuda.synchronize(device)
        check_outputs(torch, x, conv_before, q_out, rel_out)

        graph, (graph_q, graph_rel) = capture_graph(torch, lambda: call_prep(fused_qkvr_prep, x))
        graph.replay()
        torch.cuda.synchronize(device)
        if not (torch.equal(graph_q, q_out) and torch.equal(graph_rel, rel_out)):
            raise KernelLaunchFailed(f"{_BACKEND} CUDA-graph replay differs from the eager call")

        time_ms = Timer.cupti(graph.replay, wall_span=True)
        energy_j = Energy.perf(graph.replay, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except (KernelLaunchFailed, ProfilerNotImplemented):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {exc}") from exc
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(logical_bytes(shape, num_tokens) / (time_ms / 1000.0) / 1e9),
        energy_j=float(energy_j),
    )


__all__ = [
    "Layout",
    "PrepInputs",
    "build_inputs",
    "call_prep",
    "capture_graph",
    "check_outputs",
    "logical_bytes",
    "profile_inkling_qkvr_prep_vllm_triton",
    "timing_layout",
]
