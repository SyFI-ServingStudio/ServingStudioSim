"""TileLang runner for the KDA RecoverSSM commit kind (backend ``tilelang``).

This runner only allocates tensors, times one call, and returns metrics. DB
writes, subprocess selection and registry routing live in L1b.

``tilelang`` is this repository's own commit kernel, not one any engine
ships. It does what one ``KDARecoverSSMCommitContext.commit`` call of the vLLM
fork does for the same cache pages (every layer of one KV-cache group, no
prefix-cache alignment) with the fold written in closed form.

Verify already stored each window row's correction ``u_t = beta_t (v_t -
S_{t-1} k_t)`` and its raw key and gate, so folding the first ``n`` accepted
rows into the checkpoint needs no recurrence:

    S_n = S_0 * exp(G_n) + sum_t u_t (khat_t * exp(G_n - G_t))^T

with ``G_t`` the per-channel cumulative log gate through row ``t`` and
``khat_t`` the L2-normalised key. Three launches:

1. a Triton prep forms, per (request, layer, head), the decayed keys
   ``khat_t * exp(G_n - G_t)`` (zero past the accepted rows) and the total
   decay ``exp(G_n)`` into scratch, and clamps the accepted length as the
   fork's plan kernel does (``min(max(num_accepted, 0), query_len,
   spec_query_len)``, zero on a NULL slot);
2. a TileLang fold: one program owns a ``[BLOCK_V, K]`` state tile, scales it
   by the total decay and adds the accepted rows' outer products as an
   unrolled fp32 FMA chain before a single store, so the state is read and
   written once whatever the accepted length;
3. the conv window compaction, exactly as the fork's.

The update is a rank-n product with n at most the window, so the fold is
mostly a state stream (an fp32 pool within 6% of a plain copy; a bf16 pool
1.1-1.6x one, the FMA chain only partly hidden at eight rows). A tensor-core
product would pin the tile to the mma accumulator layout, whose two adjacent
elements per thread turn a bf16 state's loads and stores into 4-byte
accesses; a free-layout tile lets them vectorise to 32 bytes per thread, as
the state streaming of NVlabs/kda's TIRx KDA forward does. The chain runs to
the accepted length rounded up to a power of two (rows past it are zero), and
the program is held to a register budget that keeps more of them resident to
cover the state's latency.

Each layer's page is reached through a per-layer base-address table, as the
fork's context does, so the pages keep their interleaved vLLM layout. The
fork's fold instead loops over the accepted rows, recomputing each row's norm
and gate in every value block, so its time grows with the accepted length.
The state is fp32 or bf16; the update runs in fp32 either way.
"""

# No ``from __future__ import annotations``: TileLang reads the fold's
# parameter annotations (shapes built from closure values) at parse time.

from dataclasses import dataclass
from functools import cache
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention.kda_recoverssm_vllm_triton import (
    LOWER_BOUND,
    KdaRecoverSSMShape,
    LayerPage,
    build_commit_pages,
    check_commit,
    commit_bytes,
    commit_flops,
    validate_commit_args,
)
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "kda_recoverssm_commit:tilelang"
# The fold owns whole state rows, so each (request, layer, head) runs its prep
# math once. Threads and resident programs per SM per state dtype: B200 jobs
# 5665 and 5670 (logs/20261011_9_kda_commit_nvkda) found a bf16 pool's fold
# fastest at 128 threads held to 3 programs per SM, and an fp32 pool's, which
# is within 6% of a plain copy of its state, at 256 threads held to 2.
FOLD_BLOCK_V = 128
FOLD_LAUNCH = {"bfloat16": (128, 3), "float32": (256, 2)}
# One warp per (request, layer, head) prep record ran 1.9-2.4x faster than
# four in B200 job 5665; job 5671 found 1024-channel conv blocks fastest.
PREP_WARPS = 1
CONV_BLOCK_D = 1024

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _kda_commit_prep_kernel(
        kg_ref_ptr,
        kg_addrs_ptr,
        a_log_ptr,
        dt_bias_ptr,
        num_accepted_ptr,
        state_indices_ptr,
        query_start_loc_ptr,
        decayed_key_ptr,
        total_decay_ptr,
        commit_len_ptr,
        lower_bound,
        stride_kg_block,
        stride_kg_head,
        stride_kg_pos,
        NUM_LAYER_HEADS: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        K: tl.constexpr,
        SPEC: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_lh = tl.program_id(1)
        layer = pid_lh // NUM_HEADS
        head = pid_lh % NUM_HEADS

        slot = tl.load(state_indices_ptr + pid_b).to(tl.int64)
        query_len = tl.load(query_start_loc_ptr + pid_b + 1) - tl.load(query_start_loc_ptr + pid_b)
        accepted = tl.maximum(tl.load(num_accepted_ptr + pid_b), 0)
        n = tl.minimum(tl.minimum(accepted, query_len), SPEC)
        n = tl.where(slot > 0, n, 0)
        if pid_lh == 0:
            tl.store(commit_len_ptr + pid_b, n)
        if n == 0:
            return

        kg_ptr = (
            tl.load(kg_addrs_ptr + layer).to(tl.pointer_type(kg_ref_ptr.dtype.element_ty))
            + slot * stride_kg_block
            + head * stride_kg_head
        )
        offs_t = tl.arange(0, BLOCK_T)
        offs_k = tl.arange(0, K)
        rows = offs_t < n
        k = tl.load(
            kg_ptr + offs_t[:, None] * stride_kg_pos + offs_k[None, :],
            mask=rows[:, None],
            other=0.0,
        ).to(tl.float32)
        raw_g = tl.load(
            kg_ptr + offs_t[:, None] * stride_kg_pos + (K + offs_k)[None, :],
            mask=rows[:, None],
            other=0.0,
        ).to(tl.float32)
        khat = k * tl.rsqrt(tl.sum(k * k, axis=1) + 1e-6)[:, None]
        a = tl.exp(tl.load(a_log_ptr + pid_lh).to(tl.float32))
        dt_bias = tl.load(dt_bias_ptr + pid_lh * K + offs_k).to(tl.float32)
        gate = lower_bound * tl.sigmoid(a * (raw_g + dt_bias[None, :]))
        gate = tl.where(rows[:, None], gate, 0.0)
        through = tl.cumsum(gate, axis=0)
        total = tl.sum(gate, axis=0)
        # Row t's update decays by every later accepted row's gate.
        decayed = tl.where(rows[:, None], khat * tl.exp(total[None, :] - through), 0.0)
        record = pid_b * NUM_LAYER_HEADS + pid_lh
        tl.store(
            decayed_key_ptr + record * BLOCK_T * K + offs_t[:, None] * K + offs_k[None, :],
            decayed,
        )
        tl.store(total_decay_ptr + record * K + offs_k, tl.exp(total))

    @triton.jit
    def _kda_commit_conv_kernel(
        conv_ref_ptr,
        conv_addrs_ptr,
        commit_len_ptr,
        state_indices_ptr,
        conv_dim,
        history,
        stride_conv_block,
        stride_conv_token,
        BLOCK_D: tl.constexpr,
        BLOCK_HISTORY: tl.constexpr,
    ):
        pid_d = tl.program_id(0)
        pid_b = tl.program_id(1)
        layer = tl.program_id(2)
        n = tl.load(commit_len_ptr + pid_b)
        if n == 0:
            return
        slot = tl.load(state_indices_ptr + pid_b).to(tl.int64)
        conv_ptr = (
            tl.load(conv_addrs_ptr + layer).to(tl.pointer_type(conv_ref_ptr.dtype.element_ty))
            + slot * stride_conv_block
        )
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        offs_h = tl.arange(0, BLOCK_HISTORY)
        mask = (offs_d[:, None] < conv_dim) & (offs_h[None, :] < history)
        values = tl.load(
            conv_ptr + (n - 1 + offs_h[None, :]) * stride_conv_token + offs_d[:, None],
            mask=mask,
        )
        out_ptrs = conv_ptr + offs_h[None, :] * stride_conv_token + offs_d[:, None]
        tl.store(out_ptrs, values, mask=mask)

except ImportError:  # pragma: no cover - triton ships with torch on CUDA hosts
    triton = None


def _decay_and_fold(
    state_tile: Any, decay_s: Any, correction_s: Any, key_s: Any, v: Any, k: Any, rows: int
) -> Any:
    """``S[v, k] exp(G_n[k]) + sum_t u_t[v] key_t[k]`` over ``rows`` window
    rows, in fp32. Plain Python, so the chain unrolls at trace time instead of
    becoming a TIR loop."""
    acc = state_tile[v, k] * decay_s[k]
    for t in range(rows):
        acc = acc + correction_s[t, v] * key_s[t, k]
    return acc


@cache
def _fold_kernel(
    num_layers: int,
    num_heads: int,
    dim: int,
    spec: int,
    block_t: int,
    state_dtype: str,
    page_bytes: int,
) -> Any:
    """The TileLang fold for one page layout; the batch is a runtime size."""
    import tilelang
    import tilelang.language as T

    state_bytes = 4 if state_dtype == "float32" else 2
    batch = T.dynamic("batch")
    layer_heads = num_layers * num_heads
    threads, min_programs = FOLD_LAUNCH[state_dtype]
    # Chain lengths for accepted lengths up to 1, 2, 4 and the whole window.
    short, mid, long = (min(rows, block_t) for rows in (1, 2, 4))

    @tilelang.jit(out_idx=[])
    def build() -> Any:
        @T.prim_func
        def fold(
            state_addrs: T.Tensor((num_layers,), "int64"),
            correction_addrs: T.Tensor((num_layers,), "int64"),
            state_indices: T.Tensor((batch,), "int32"),
            commit_len: T.Tensor((batch,), "int32"),
            decayed_key: T.Tensor((batch, layer_heads, block_t, dim), "float32"),
            total_decay: T.Tensor((batch, layer_heads, dim), "float32"),
        ) -> None:
            with T.Kernel(dim // FOLD_BLOCK_V, batch, layer_heads, threads=threads) as (
                bv,
                b,
                lh,
            ):
                T.annotate_min_blocks_per_sm(min_programs)
                state_tile = T.alloc_fragment((FOLD_BLOCK_V, dim), "float32")
                correction_s = T.alloc_shared((block_t, FOLD_BLOCK_V), "float32")
                key_s = T.alloc_shared((block_t, dim), "float32")
                decay_s = T.alloc_shared((dim,), "float32")
                n = commit_len[b]
                if n > 0:
                    layer = lh // num_heads
                    head = lh % num_heads
                    slot = T.cast(state_indices[b], "int64")
                    state = T.make_tensor_from_addr(
                        state_addrs[layer] + slot * page_bytes + head * (dim * dim * state_bytes),
                        (dim, dim),
                        state_dtype,
                    )
                    correction = T.make_tensor_from_addr(
                        correction_addrs[layer] + slot * page_bytes + head * (spec * dim * 4),
                        (spec, dim),
                        "float32",
                    )
                    T.copy(state[bv * FOLD_BLOCK_V, 0], state_tile)
                    for t, v in T.Parallel(block_t, FOLD_BLOCK_V):
                        correction_s[t, v] = T.if_then_else(
                            t < n, correction[t, bv * FOLD_BLOCK_V + v], 0.0
                        )
                    T.copy(decayed_key[b, lh, 0, 0], key_s)
                    T.copy(total_decay[b, lh, 0], decay_s)
                    if n <= 1:
                        for v, k in T.Parallel(FOLD_BLOCK_V, dim):
                            state_tile[v, k] = _decay_and_fold(
                                state_tile, decay_s, correction_s, key_s, v, k, short
                            )
                    elif n <= 2:
                        for v, k in T.Parallel(FOLD_BLOCK_V, dim):
                            state_tile[v, k] = _decay_and_fold(
                                state_tile, decay_s, correction_s, key_s, v, k, mid
                            )
                    elif n <= 4:
                        for v, k in T.Parallel(FOLD_BLOCK_V, dim):
                            state_tile[v, k] = _decay_and_fold(
                                state_tile, decay_s, correction_s, key_s, v, k, long
                            )
                    else:
                        for v, k in T.Parallel(FOLD_BLOCK_V, dim):
                            state_tile[v, k] = _decay_and_fold(
                                state_tile, decay_s, correction_s, key_s, v, k, block_t
                            )
                    T.copy(state_tile, state[bv * FOLD_BLOCK_V, 0])

        return fold

    return build()


@dataclass(frozen=True)
class TileLangCommitPlan:
    """Per-group tables and scratch built once, as
    ``KDARecoverSSMCommitContext.create`` builds its own: each layer page's
    base address, the stacked gate parameters, and the prep's outputs."""

    pages: tuple[LayerPage, ...]
    state_addrs: Any
    correction_addrs: Any
    kg_addrs: Any
    conv_addrs: Any
    a_log: Any  # [L*H] fp32
    dt_bias: Any  # [L*H*K] fp32
    decayed_key: Any  # [B, L*H, T, K] fp32
    total_decay: Any  # [B, L*H, K] fp32
    commit_len: Any  # [B] int32
    block_t: int
    fold: Any
    shape: KdaRecoverSSMShape


def build_plan(
    torch: Any, pages: tuple[LayerPage, ...], shape: KdaRecoverSSMShape, max_batch: int
) -> TileLangCommitPlan:
    first = pages[0]
    for page in pages:
        for name in ("checkpoint", "correction", "kg", "conv"):
            view, ref = getattr(page, name), getattr(first, name)
            if view.stride() != ref.stride() or view.dtype != ref.dtype:
                raise ValueError(f"{_BACKEND} layers need one {name} layout")
    heads, dim, spec = shape.num_heads, shape.head_dim, shape.spec_query_len
    state, correction = first.checkpoint, first.correction
    if state.stride()[1:] != (dim * dim, dim, 1) or correction.stride()[1:] != (spec * dim, dim, 1):
        raise ValueError(f"{_BACKEND} needs dense per-head state and correction records")
    if first.kg.stride(-1) != 1 or first.conv.stride(-1) != 1:
        raise ValueError(f"{_BACKEND} needs unit-stride key/gate and conv channels")
    page_bytes = state.stride(0) * state.element_size()
    if correction.stride(0) * correction.element_size() != page_bytes:
        raise ValueError(f"{_BACKEND} needs the state and correction in one page")
    device = state.device

    def addrs(name: str) -> Any:
        return torch.tensor(
            [getattr(page, name).data_ptr() for page in pages], dtype=torch.int64, device=device
        )

    block_t = triton.next_power_of_2(spec)
    layer_heads = shape.num_layers * heads
    state_dtype = {torch.float32: "float32", torch.bfloat16: "bfloat16"}[state.dtype]
    return TileLangCommitPlan(
        pages=pages,
        state_addrs=addrs("checkpoint"),
        correction_addrs=addrs("correction"),
        kg_addrs=addrs("kg"),
        conv_addrs=addrs("conv"),
        a_log=torch.cat([page.a_log for page in pages]).float().contiguous(),
        dt_bias=torch.cat([page.dt_bias for page in pages]).float().contiguous(),
        decayed_key=torch.empty(
            max_batch, layer_heads, block_t, dim, dtype=torch.float32, device=device
        ),
        total_decay=torch.empty(max_batch, layer_heads, dim, dtype=torch.float32, device=device),
        commit_len=torch.empty(max_batch, dtype=torch.int32, device=device),
        block_t=block_t,
        fold=_fold_kernel(shape.num_layers, heads, dim, spec, block_t, state_dtype, page_bytes),
        shape=shape,
    )


def commit(
    plan: TileLangCommitPlan, num_accepted: Any, state_indices: Any, query_start_loc: Any
) -> None:
    """Fold each request's accepted rows into every layer's checkpoint and
    compact every layer's conv window, in place."""
    shape = plan.shape
    first = plan.pages[0]
    batch = state_indices.shape[0]
    heads, dim = shape.num_heads, shape.head_dim
    layer_heads = shape.num_layers * heads
    decayed_key = plan.decayed_key[:batch]
    total_decay = plan.total_decay[:batch]
    commit_len = plan.commit_len[:batch]
    _kda_commit_prep_kernel[(batch, layer_heads)](
        first.kg,
        plan.kg_addrs,
        plan.a_log,
        plan.dt_bias,
        num_accepted,
        state_indices,
        query_start_loc,
        decayed_key,
        total_decay,
        commit_len,
        LOWER_BOUND,
        first.kg.stride(0),
        first.kg.stride(1),
        first.kg.stride(2),
        NUM_LAYER_HEADS=layer_heads,
        NUM_HEADS=heads,
        K=dim,
        SPEC=shape.spec_query_len,
        BLOCK_T=plan.block_t,
        num_warps=PREP_WARPS,
    )
    plan.fold(
        plan.state_addrs,
        plan.correction_addrs,
        state_indices,
        commit_len,
        decayed_key,
        total_decay,
    )
    _kda_commit_conv_kernel[(triton.cdiv(shape.conv_dim, CONV_BLOCK_D), batch, shape.num_layers)](
        first.conv,
        plan.conv_addrs,
        commit_len,
        state_indices,
        shape.conv_dim,
        shape.conv_history_len,
        first.conv.stride(0),
        first.conv.stride(1),
        BLOCK_D=CONV_BLOCK_D,
        BLOCK_HISTORY=triton.next_power_of_2(shape.conv_history_len),
        num_warps=4,
    )


def profile_kda_recoverssm_commit_tilelang(
    batch_size: int,
    commit_len: int,
    num_heads: int,
    head_dim: int,
    spec_query_len: int,
    num_layers: int,
    dtype: DType | str,
    state_dtype: DType | str,
) -> ComputeMetrics:
    """Profile one closed-form KDA RecoverSSM commit over ``num_layers`` layers."""
    shape = validate_commit_args(
        batch_size,
        commit_len,
        num_heads,
        head_dim,
        spec_query_len,
        num_layers,
        dtype,
        state_dtype,
    )
    if shape.head_dim % FOLD_BLOCK_V or shape.head_dim & (shape.head_dim - 1):
        raise ValueError(
            f"{_BACKEND} needs a power-of-two head_dim divisible by {FOLD_BLOCK_V}, "
            f"got {shape.head_dim}"
        )
    if triton is None:
        raise ProfilerNotImplemented(f"triton is required for {_BACKEND}")
    try:
        import tilelang  # noqa: F401
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch and TileLang are required for {_BACKEND}") from exc
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        operands = build_commit_pages(torch, shape, device=device)
        plan = build_plan(torch, operands.pages, shape, max_batch=shape.batch_size)

        def kernel() -> None:
            commit(
                plan,
                operands.num_accepted_tokens,
                operands.state_indices,
                operands.query_start_loc,
            )

        check_commit(torch, operands, shape, invoke=kernel)
        time_ms = Timer.cupti(kernel, warmup=5, kernel_name=None)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {exc}") from exc
    elapsed = time_ms / 1000.0
    flops, nbytes = commit_flops(shape), commit_bytes(shape)
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=flops / elapsed / 1e12 if elapsed > 0 else 0.0,
        memory_bandwidth_gbps=nbytes / elapsed / 1e9 if elapsed > 0 else 0.0,
        energy_j=float(energy_j),
    )


__all__ = [
    "TileLangCommitPlan",
    "build_plan",
    "commit",
    "profile_kda_recoverssm_commit_tilelang",
]
