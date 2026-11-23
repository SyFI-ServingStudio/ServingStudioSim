"""DeepInfer's small prefill KV kernels -- the ``tilelang`` backends.

Each runner times the DeepInfer TileLang wrapper exactly as DeepSeek-V4.1's
prefill calls it (``deepinfer/model_executor/models/v4_modules/attn.py``,
``index_attn.py``, ``compress.py``; ``layers/layernorm.py``), after checking
the kernel against DeepInfer's own reference:

- ``fused_norm_rope_cast``: ``fused_norm_rope_hadamard_cast_store_kvcache``,
  checked by DeepInfer's ``test_fused_norm_rope_hadamard_cast_store_kvcache``
  (bit-exact against the unfused reference) on the same case.
- ``rms_norm_mxfp8``: ``norm`` with ``get_fp8_cast_params()``, checked by
  DeepInfer's ``test_norm`` at the same hidden size and output form.
- ``paged_kv_cache_load`` / ``paged_kv_cache_store``: ``load_paged_kv_cache``
  / ``store_kv_cache``, checked bit-exact against ``*_ref``.
- ``kv_index_cat``, ``expand_swa_index``, ``candidate_compact_lens``,
  ``candidate_index_restore``: checked equal to their ``*_ref``.
- ``compress_state_prefill``: ``compress_and_update_state_prefill``, checked
  by DeepInfer's ``_test_compress_state_prefill`` at the same dim, ratio and
  state size.

The ``elementwise`` copy that DeepInfer issues as ``kv_swa_cur.copy_(kv_swa)``
is a separate backend (``profiling.runners.elementwise.torch``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

BACKEND = "tilelang"
_OPS = "deepinfer.model_executor.layers.tilelang_ops"

# cast_format -> (DeepInfer cache format name, CastArgs fields).
# CastArgs.from_cache_format gives the nvfp4 / mxfp4 rows; mxfp8 is the SWA cast
# attn.py builds by hand (e4m3, 32 per UE8M0 scale, round_sf).
CAST_FORMATS = ("mxfp8", "nvfp4", "mxfp4")
CACHE_FORMATS = {
    "mxfp8": "V41_FORMAT",
    "nvfp4": "V41_FP4_FORMAT",
    "mxfp4": "FP4_INDEX_KV_FORMAT",
}
FUSED_OUTPUTS = ("cast_back", "quantized", "cache", "cache_and_norm")
LOAD_FORMATS = ("mxfp8", "nvfp4", "mxfp4")
# DeepInfer's own correctness size for the fused transform test.
FUSED_CHECK_TOKENS = 2333
CHECK_ROWS = 4096


def _need_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"torch is required for {BACKEND}") from exc
    return torch


def _import(module: str) -> Any:
    import importlib

    try:
        return importlib.import_module(f"{_OPS}.{module}")
    except ImportError as exc:
        raise ProfilerNotImplemented(
            f"{BACKEND} needs DeepInfer's TileLang ops ({_OPS}.{module}, deepinfer_env)"
        ) from exc


def _measure(
    name: str,
    check: Callable[[], None],
    build: Callable[[], Callable[[], Any]],
    bytes_accessed: int,
    *,
    kernel_name: str | None = None,
) -> ComputeMetrics:
    """Check, build the timed callable, JIT it once, then time with CUPTI."""
    try:
        try:
            check()
        except AssertionError as exc:
            raise KernelLaunchFailed(
                f"{BACKEND} {name} differs from DeepInfer's reference: {exc}"
            ) from exc
        run_once = build()
        run_once()  # TileLang JIT, outside timing.
        time_ms = Timer.cupti(run_once, warmup=3, kernel_name=kernel_name)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    except (ProfilerNotImplemented, KernelLaunchFailed):
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError(f"{BACKEND} {name} ran out of memory") from exc
        raise KernelLaunchFailed(f"{BACKEND} {name} failed: {exc}") from exc
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=bytes_accessed / elapsed_s / 1e9,
        energy_j=float(energy_j),
    )


def _positive(name: str, **values: int) -> None:
    bad = {k: v for k, v in values.items() if int(v) < 1}
    if bad:
        raise ValueError(f"{BACKEND} {name} needs positive sizes, got {bad}")


def _bf16(name: str, dtype: DType | str) -> None:
    if DType.from_value(dtype) is not DType.BF16:
        raise ValueError(f"{BACKEND} {name} requires dtype=bf16, got {dtype!r}")


# ---------------------------------------------------------------- fused transform


def fused_case_fields(
    num_heads: int,
    head_dim: int,
    rope_dim: int,
    norm: bool,
    cast_format: str,
    output: str,
    block_size: int,
) -> dict[str, Any]:
    """``_FusedTransformCase`` fields of one production call (no torch import)."""
    if cast_format not in CAST_FORMATS:
        raise ValueError(f"cast_format must be one of {CAST_FORMATS}, got {cast_format!r}")
    if output not in FUSED_OUTPUTS:
        raise ValueError(f"output must be one of {FUSED_OUTPUTS}, got {output!r}")
    if not 0 < rope_dim <= head_dim:
        raise ValueError(f"need 0 < rope_dim <= head_dim, got {rope_dim}, {head_dim}")
    to_cache = output in ("cache", "cache_and_norm")
    if to_cache != (block_size > 0):
        raise ValueError(
            f"block_size must be positive exactly for cache outputs, got {block_size} for {output}"
        )
    fmt, group, sf_format, round_sf = {
        "mxfp8": ("e4m3", 32, "ue8m0", True),
        "nvfp4": ("e2m1", 16, "e4m3", False),
        "mxfp4": ("e2m1", 32, "ue8m0", True),
    }[cast_format]
    return dict(
        num_heads=num_heads,
        head_dim=head_dim,
        rotary_dim=rope_dim,
        do_norm=bool(norm),
        do_rope=True,
        rope_offset=head_dim - rope_dim,
        # DeepSeek-V4.1 RoPE is interleaved (mla_attn.py asserts cfg.rope_interleaved).
        is_neox=False,
        cast_fmt=fmt,
        num_per_channels=group,
        sf_format=sf_format,
        round_sf=round_sf,
        cast_back=output == "cast_back",
        block_size=block_size,
        out_stage="norm" if output == "cache_and_norm" else "",
    )


def fused_bytes(
    num_tokens: int, num_heads: int, head_dim: int, cast_format: str, output: str
) -> int:
    rows = num_tokens * num_heads
    total = 2 * rows * head_dim
    if output in ("cache", "cache_and_norm"):
        per_token = {"mxfp8": 528, "nvfp4": 288, "mxfp4": 68}[cast_format] * num_heads
        total += num_tokens * per_token + (2 * rows * head_dim if output == "cache_and_norm" else 0)
    elif output == "quantized":
        bits = 8 if cast_format == "mxfp8" else 4
        group = 16 if cast_format == "nvfp4" else 32
        total += rows * head_dim * bits // 8 + rows * head_dim // group
    else:
        total += 2 * rows * head_dim
    return total


def profile_fused_norm_rope_cast_tilelang(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    rope_dim: int,
    norm: bool,
    cast_format: str,
    output: str,
    block_size: int,
    dtype: DType | str,
) -> ComputeMetrics:
    _bf16("fused_norm_rope_cast", dtype)
    _positive("fused_norm_rope_cast", num_tokens=num_tokens, num_heads=num_heads, head_dim=head_dim)
    fields = fused_case_fields(num_heads, head_dim, rope_dim, norm, cast_format, output, block_size)
    torch = _need_torch()
    ft = _import("fused_transform")
    cc = _import("cache_common")
    if block_size > 0:
        fields["cache_format"] = getattr(cc, CACHE_FORMATS[cast_format])
    case = ft._FusedTransformCase(**fields)

    def check() -> None:
        ft.test_fused_norm_rope_hadamard_cast_store_kvcache(
            case, min(num_tokens, FUSED_CHECK_TOKENS)
        )

    def build() -> Callable[[], Any]:
        torch.manual_seed(0)
        # Production cache stores are contiguous appends: slots arange (slot 0 = -1).
        args = ft._make_case(case, num_tokens, preallocate=True)
        fn = ft.fused_norm_rope_hadamard_cast_store_kvcache
        return lambda: fn(*args)

    return _measure(
        "fused_norm_rope_cast",
        check,
        build,
        fused_bytes(num_tokens, num_heads, head_dim, cast_format, output),
    )


# ---------------------------------------------------------------- RMSNorm + MXFP8


def profile_rms_norm_mxfp8_tilelang(
    num_tokens: int, hidden: int, return_bf16: bool, dtype: DType | str
) -> ComputeMetrics:
    _bf16("rms_norm_mxfp8", dtype)
    _positive("rms_norm_mxfp8", num_tokens=num_tokens, hidden=hidden)
    torch = _need_torch()
    nm = _import("norm")
    linear = _import_linear()
    cast = linear.get_fp8_cast_params()
    layout = (
        cast["num_per_channels"],
        cast["use_tma_aligned_col_major_sf"],
        cast["round_sf"],
        cast["use_packed_ue8m0"],
    )

    def check() -> None:
        nm.test_norm(hidden, False, (layout, 0, bool(return_bf16), False))

    def build() -> Callable[[], Any]:
        torch.manual_seed(0)
        x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
        weight = torch.empty(hidden, dtype=torch.bfloat16, device="cuda").normal_(1.0, 0.1)
        return lambda: nm.norm(x, weight, eps=1e-6, return_bf16=bool(return_bf16), **cast)

    out = num_tokens * hidden * (1 + (2 if return_bf16 else 0)) + num_tokens * hidden // 32
    return _measure("rms_norm_mxfp8", check, build, 2 * num_tokens * hidden + out)


def _import_linear() -> Any:
    import importlib

    try:
        return importlib.import_module("deepinfer.model_executor.layers.linear")
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{BACKEND} needs DeepInfer (deepinfer_env)") from exc


# ---------------------------------------------------------------- paged cache load / store


def load_segments(
    num_tokens: int, block_size: int, segment_tokens: int, num_blocks: int
) -> list[tuple[int, int, int, int]]:
    """``[src_block, src_begin, src_end, dst_offset]`` rows: consecutive
    ``segment_tokens`` runs from distinct pages (pure Python, no torch)."""
    if not 0 < segment_tokens <= block_size:
        raise ValueError(
            f"need 0 < segment_tokens <= block_size, got {segment_tokens}, {block_size}"
        )
    spare = block_size - segment_tokens
    rows = []
    dst = 0
    i = 0
    while dst < num_tokens:
        length = min(segment_tokens, num_tokens - dst)
        begin = i % (spare + 1)
        # Spread pages across the pool: a stride coprime with the pool size.
        block = (i * 7919) % num_blocks
        rows.append((block, begin, begin + length, dst))
        dst += length
        i += 1
    return rows


def _cache_with_values(torch: Any, layout: Any, num_blocks: int, block_size: int) -> Any:
    """A cache whose every page holds valid encoded tokens (tiled from a small one)."""
    cc = _import("cache_common")
    sk = _import("store_kv_cache")
    seed_blocks = min(num_blocks, 16)
    seed = cc.alloc_kv_cache(layout, seed_blocks, block_size, device="cuda")
    src = torch.randn(
        seed_blocks * block_size, layout.num_elements, dtype=torch.bfloat16, device="cuda"
    )
    sk.store_kv_cache_ref(
        src,
        seed,
        torch.arange(seed_blocks * block_size, dtype=torch.int32, device="cuda"),
        layout=layout,
    )
    cache = cc.alloc_kv_cache(layout, num_blocks, block_size, device="cuda")
    for start in range(0, num_blocks, seed_blocks):
        stop = min(num_blocks, start + seed_blocks)
        cache[start:stop].copy_(seed[: stop - start])
    return cache


def _parts_equal(torch: Any, got: Any, want: Any) -> None:
    for a, b in zip(got, want, strict=True):
        assert (a is None) == (b is None)
        if a is not None:
            assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def profile_paged_kv_cache_load_tilelang(
    num_tokens: int, block_size: int, segment_tokens: int, cache_format: str
) -> ComputeMetrics:
    _positive(
        "paged_kv_cache_load",
        num_tokens=num_tokens,
        block_size=block_size,
        segment_tokens=segment_tokens,
    )
    if cache_format not in LOAD_FORMATS:
        raise ValueError(f"cache_format must be one of {LOAD_FORMATS}, got {cache_format!r}")
    torch = _need_torch()
    lp = _import("load_paged_kv_cache")
    layout = getattr(_import("cache_common"), CACHE_FORMATS[cache_format])
    num_segments = -(-num_tokens // segment_tokens)
    num_blocks = num_segments + 1 if num_segments % 7919 else num_segments + 2
    state: dict[str, Any] = {}

    def check() -> None:
        cache = _cache_with_values(torch, layout, num_blocks, block_size)
        rows = load_segments(num_tokens, block_size, segment_tokens, num_blocks)
        segs = torch.tensor(rows, dtype=torch.int32, device="cuda")
        state.update(cache=cache, segs=segs)
        k = min(len(rows), max(1, CHECK_ROWS // segment_tokens))
        part = segs[:k]
        total = int(rows[k - 1][3] + rows[k - 1][2] - rows[k - 1][1])
        _parts_equal(
            torch,
            lp.load_paged_kv_cache(cache, part, total, layout=layout),
            lp.load_paged_kv_cache_ref(cache, part, total, layout=layout),
        )

    def build() -> Callable[[], Any]:
        cache, segs = state["cache"], state["segs"]
        return lambda: lp.load_paged_kv_cache(cache, segs, num_tokens, layout=layout)

    out_bytes = num_tokens * (2 * layout.num_elements if layout.do_cast else layout.bytes_per_token)
    return _measure(
        "paged_kv_cache_load", check, build, num_tokens * layout.bytes_per_token + out_bytes
    )


def store_slots(num_tokens: int, block_size: int) -> list[int]:
    """One sequence's prefill SWA slot map: only its last ``block_size`` rows
    are stored (ring position ``pos % block_size``), the rest are -1
    (``prepare_tensor_input_v4.cpp`` ``slot_mapping_store_swa``)."""
    first = max(0, num_tokens - block_size)
    return [-1] * first + [pos % block_size for pos in range(first, num_tokens)]


def profile_paged_kv_cache_store_tilelang(
    num_tokens: int, block_size: int, cache_format: str, dtype: DType | str
) -> ComputeMetrics:
    _bf16("paged_kv_cache_store", dtype)
    _positive("paged_kv_cache_store", num_tokens=num_tokens, block_size=block_size)
    if cache_format not in LOAD_FORMATS:
        raise ValueError(f"cache_format must be one of {LOAD_FORMATS}, got {cache_format!r}")
    torch = _need_torch()
    sk = _import("store_kv_cache")
    cc = _import("cache_common")
    layout = getattr(cc, CACHE_FORMATS[cache_format])
    state: dict[str, Any] = {}

    def check() -> None:
        torch.manual_seed(0)
        kv = torch.randn(num_tokens, layout.num_elements, dtype=torch.bfloat16, device="cuda")
        slots = torch.tensor(store_slots(num_tokens, block_size), dtype=torch.int32, device="cuda")
        cache = cc.alloc_kv_cache(layout, 2, block_size, device="cuda")
        cache.zero_()
        ref = cc.alloc_kv_cache(layout, 2, block_size, device="cuda")
        ref.zero_()
        sk.store_kv_cache(kv, cache, slots, layout=layout)
        sk.store_kv_cache_ref(kv, ref, slots, layout=layout)
        assert torch.equal(cache, ref)
        state.update(kv=kv, slots=slots, cache=cache)

    def build() -> Callable[[], Any]:
        kv, slots, cache = state["kv"], state["slots"], state["cache"]
        return lambda: sk.store_kv_cache(kv, cache, slots, layout=layout)

    stored = min(num_tokens, block_size)
    return _measure(
        "paged_kv_cache_store",
        check,
        build,
        4 * num_tokens + stored * (2 * layout.num_elements + layout.bytes_per_token),
    )


# ---------------------------------------------------------------- index assembly


def profile_kv_index_cat_tilelang(num_rows: int, window_size: int, topk: int) -> ComputeMetrics:
    _positive("kv_index_cat", num_rows=num_rows, window_size=window_size, topk=topk)
    torch = _need_torch()
    kc = _import("kv_index_cat")
    state: dict[str, Any] = {}

    def check() -> None:
        torch.manual_seed(0)
        kv_index = torch.randint(
            -1, 1 << 20, (num_rows, window_size), dtype=torch.int32, device="cuda"
        )
        ca_index = torch.randint(-1, 1 << 20, (num_rows, topk), dtype=torch.int32, device="cuda")
        state.update(a=kv_index, b=ca_index)
        assert torch.equal(
            kc.kv_index_cat(kv_index, ca_index, 666, 23333),
            kc.kv_index_cat_ref(kv_index, ca_index, 666, 23333),
        )

    def build() -> Callable[[], Any]:
        return lambda: kc.kv_index_cat(state["a"], state["b"], 666, 23333)

    return _measure("kv_index_cat", check, build, 8 * num_rows * (window_size + topk))


def profile_expand_swa_index_tilelang(num_rows: int, window_size: int) -> ComputeMetrics:
    _positive("expand_swa_index", num_rows=num_rows, window_size=window_size)
    torch = _need_torch()
    si = _import("swa_index")
    state: dict[str, Any] = {}

    def check() -> None:
        pivots_cpu = si._make_pivots(num_rows, window_size)
        state["pivots"] = pivots_cpu.to("cuda")
        rows = slice(0, min(num_rows, CHECK_ROWS))
        got = si.expand_swa_index(state["pivots"][rows], window_size)
        assert torch.equal(got.cpu(), si.expand_swa_index_ref(pivots_cpu[rows], window_size))

    def build() -> Callable[[], Any]:
        return lambda: si.expand_swa_index(state["pivots"], window_size)

    return _measure("expand_swa_index", check, build, 16 * num_rows + 4 * num_rows * window_size)


# ---------------------------------------------------------------- compressor state


def compressed_rows(num_tokens: int, compress_ratio: int) -> int:
    """Compressed tokens one fresh request of ``num_tokens`` emits (``_make_compressed_meta``)."""
    return max(1, num_tokens // compress_ratio)


def profile_compress_state_prefill_tilelang(
    num_tokens: int, head_dim: int, compress_ratio: int, state_size: int
) -> ComputeMetrics:
    _positive(
        "compress_state_prefill",
        num_tokens=num_tokens,
        head_dim=head_dim,
        compress_ratio=compress_ratio,
        state_size=state_size,
    )
    if compress_ratio not in (1, 2) or state_size < compress_ratio:
        raise ValueError(
            "need compress_ratio in (1, 2) and state_size >= ratio, "
            f"got {compress_ratio}, {state_size}"
        )
    torch = _need_torch()
    cp = _import("compress")

    def check() -> None:
        torch.set_default_device("cuda")
        try:
            torch.manual_seed(0)
            cp._test_compress_state_prefill(
                num_seq=4, dim=head_dim, state_cache_size=state_size, compress_ratio=compress_ratio
            )
        finally:
            torch.set_default_device("cpu")

    def build() -> Callable[[], Any]:
        # One fresh request: seqlen_q == seqlen_k == num_tokens.
        seqlen = torch.tensor([num_tokens])
        cu_q = cp._get_cuseqlen(seqlen).cuda()
        cu_c, mapping = cp._make_compressed_meta(seqlen, seqlen, compress_ratio)
        score = torch.randn(num_tokens, head_dim, dtype=torch.float32, device="cuda")
        latent = torch.randn(num_tokens, head_dim, dtype=torch.float32, device="cuda")
        state_cache = torch.randn(2, state_size, 2, head_dim, dtype=torch.float32, device="cuda")
        block_idx = torch.zeros(1, dtype=torch.int32, device="cuda")
        args = (
            score,
            latent,
            state_cache,
            block_idx,
            mapping.cuda(),
            cu_q,
            cu_q,
            cu_c.cuda(),
            compress_ratio,
        )
        return lambda: cp.compress_and_update_state_prefill(*args)

    n_c = compressed_rows(num_tokens, compress_ratio)
    stored = min(num_tokens, state_size)
    return _measure(
        "compress_state_prefill",
        check,
        build,
        8 * n_c * compress_ratio * head_dim + 16 * stored * head_dim + 2 * n_c * head_dim,
    )


# ---------------------------------------------------------------- candidate blocks


def _candidate_blocks(torch: Any, num_rows: int, topk_blocks: int) -> Any:
    stride = -(-topk_blocks // 8) * 8
    return torch.randint(-1, 2 * topk_blocks, (num_rows, stride), dtype=torch.int32, device="cuda")[
        :, :topk_blocks
    ]


def profile_candidate_compact_lens_tilelang(
    num_rows: int, topk_blocks: int, block_size: int
) -> ComputeMetrics:
    _positive("candidate_compact_lens", num_rows=num_rows, topk_blocks=topk_blocks)
    if block_size not in (8, 16):
        raise ValueError(f"block_size must be 8 or 16, got {block_size}")
    torch = _need_torch()
    cb = _import("candidate_blocks")
    state: dict[str, Any] = {}

    def check() -> None:
        torch.manual_seed(0)
        blocks = _candidate_blocks(torch, num_rows, topk_blocks)
        lens = torch.randint(
            0, 2 * topk_blocks * block_size + 1, (num_rows,), dtype=torch.int32, device="cuda"
        )
        state.update(blocks=blocks, lens=lens)
        assert torch.equal(
            cb.get_candidate_compact_lens(blocks, lens, block_size),
            cb.get_candidate_compact_lens_ref(blocks, lens, block_size),
        )

    def build() -> Callable[[], Any]:
        return lambda: cb.get_candidate_compact_lens(state["blocks"], state["lens"], block_size)

    return _measure("candidate_compact_lens", check, build, 4 * num_rows * (topk_blocks + 2))


def profile_candidate_index_restore_tilelang(
    num_rows: int, topk_tokens: int, topk_blocks: int, block_size: int
) -> ComputeMetrics:
    _positive(
        "candidate_index_restore",
        num_rows=num_rows,
        topk_tokens=topk_tokens,
        topk_blocks=topk_blocks,
    )
    if block_size not in (8, 16):
        raise ValueError(f"block_size must be 8 or 16, got {block_size}")
    torch = _need_torch()
    cb = _import("candidate_blocks")
    state: dict[str, Any] = {}

    def check() -> None:
        torch.manual_seed(0)
        blocks = _candidate_blocks(torch, num_rows, topk_blocks)
        stride = -(-topk_tokens // 8) * 8
        compact = torch.randint(
            -1, topk_blocks * block_size, (num_rows, stride), dtype=torch.int32, device="cuda"
        )[:, :topk_tokens]
        offsets = torch.randint(0, 1 << 20, (num_rows,), dtype=torch.int32, device="cuda")
        got, want = compact.clone(), compact.clone()
        cb.restore_candidate_indices_inplace(got, blocks, block_size, offsets)
        cb.restore_candidate_indices_inplace_ref(want, blocks, block_size, offsets)
        assert torch.equal(got, want)
        state.update(blocks=blocks, compact=compact, work=compact.clone(), offsets=offsets)

    def build() -> Callable[[], Any]:
        work, compact = state["work"], state["compact"]

        def run_once() -> None:
            # In place: refresh the compact indices (a memcpy, not a kernel record)
            # so every timed launch restores the same input.
            work.copy_(compact)
            cb.restore_candidate_indices_inplace(
                work, state["blocks"], block_size, state["offsets"]
            )

        return run_once

    return _measure(
        "candidate_index_restore",
        check,
        build,
        8 * num_rows * topk_tokens + 4 * num_rows * (topk_blocks + 1),
        kernel_name="restore_candidate",
    )


__all__ = [
    "compressed_rows",
    "fused_bytes",
    "fused_case_fields",
    "load_segments",
    "profile_candidate_compact_lens_tilelang",
    "profile_candidate_index_restore_tilelang",
    "profile_compress_state_prefill_tilelang",
    "profile_expand_swa_index_tilelang",
    "profile_fused_norm_rope_cast_tilelang",
    "profile_kv_index_cat_tilelang",
    "profile_paged_kv_cache_load_tilelang",
    "profile_paged_kv_cache_store_tilelang",
    "profile_rms_norm_mxfp8_tilelang",
    "store_slots",
]
