"""Profile DeepSeek V4.1 mega attention (one layer's decode or prefill segment).

Both backends build the same paged batch from ``query_context_pairs``: a
128-token MXFP8 sliding-window cache, and for ``compress_ratio`` 1 or 2 a
compressed cache holding ``context // ratio`` rows per request, with up to
``index_topk`` compressed rows selected per query token.

- ``flashmla_mega`` times the fork's public segment methods on
  ``DeepseekV4MegaAttnAttention``. Decode times ``_forward_decode_mega``, one
  ``fused_norm_rope_attn_rope_cast_decode`` launch; the global top-k remap it
  normally calls first is replaced by precomputed slot ids. Prefill times
  ``_forward_prefill_mega``: for each chunk of up to four requests, the NVFP4
  compressed gather (ratio > 0), the MXFP8 SWA gather,
  ``combine_topk_swa_indices`` and one ``fused_norm_rope_attn_rope_cast_fwd``.
  Outside the timed region, the output is checked against the Torch reference.
- ``torch`` times the independent Torch composite from
  ``compressed_sparse_mla_rope_cast_reference``: cache decode, RoPE, sparse attention
  with sink, inverse RoPE and FP8 cast.

Top-k selections are stratified: exactly ``min((pos + 1) // ratio, index_topk)``
distinct causal compressed rows per token, spread evenly over the available
range, so the indexer's count contract holds without a real indexer.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

KIND = "compressed_sparse_mla_rope_cast"
_GPU_NAME = "NVIDIA B200"
_MODES = ("decode", "prefill")
_HEAD_DIM = 512
_ROPE_DIM = 64
_WINDOW = 128
_INDEX_TOPK = 512
_PREFILL_CHUNK_SIZE = 4
_SUPPORTED_HEADS = (64, 128)
# Production page sizes: the SWA cache is always 32-token pages, and the
# compressed cache splits the 128-token vLLM block by the ratio.
_SWA_BLOCK = 32
_KV_BLOCK = 128
_RECORD_BYTES = {"mxfp8": 528, "nvfp4": 288}
_MAX_DECODE_ROWS = 2048
_MAX_REQUESTS = 256
_CHECK_ROWS = 48
_REL_TOL = 0.06


@dataclass(frozen=True)
class _Shape:
    mode: str
    pairs: tuple[tuple[int, int], ...]
    ratio: int
    num_heads: int
    max_model_len: int
    max_num_batched_tokens: int
    extra_format: str | None

    @property
    def num_tokens(self) -> int:
        return sum(query for query, _ in self.pairs)

    @property
    def compressed_block(self) -> int:
        return _KV_BLOCK // self.ratio if self.ratio else 0


def _validate_args(
    mode: str,
    query_context_pairs: tuple[tuple[int, int], ...],
    compress_ratio: int,
    window_size: int,
    index_topk: int,
    num_heads: int,
    head_dim: int,
    rope_dim: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    prefill_chunk_size: int,
    q_dtype: object,
    swa_cache_format: str,
    compressed_cache_format: str,
    output_dtype: object,
) -> _Shape:
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    if not query_context_pairs or len(query_context_pairs) > _MAX_REQUESTS:
        raise ProfilerNotImplemented(f"{KIND} supports 1..{_MAX_REQUESTS} requests")
    for pair in query_context_pairs:
        if not isinstance(pair, tuple) or len(pair) != 2 or any(type(v) is not int for v in pair):
            raise TypeError("query_context_pairs must contain integer (query, context) pairs")
        query, context = pair
        if query <= 0 or context < query:
            raise ValueError("each pair must satisfy 0 < query <= context")
    if type(max_model_len) is not int or not 1 <= max_model_len <= 1_048_576:
        raise ValueError("max_model_len must be an integer in [1, 1048576]")
    if max(context for _, context in query_context_pairs) > max_model_len:
        raise ValueError("context length exceeds max_model_len")
    total = sum(query for query, _ in query_context_pairs)
    if type(max_num_batched_tokens) is not int or not total <= max_num_batched_tokens <= 32768:
        raise ValueError("max_num_batched_tokens must cover all query tokens and be <= 32768")
    if mode == "decode" and total > _MAX_DECODE_ROWS:
        raise ProfilerNotImplemented(f"{KIND} decode supports at most {_MAX_DECODE_ROWS} rows")
    if compress_ratio not in (0, 1, 2):
        raise ProfilerNotImplemented(f"{KIND} supports compress_ratio 0 (SWA only), 1 and 2")
    model = (window_size, index_topk, head_dim, rope_dim, prefill_chunk_size)
    expected = (_WINDOW, _INDEX_TOPK, _HEAD_DIM, _ROPE_DIM, _PREFILL_CHUNK_SIZE)
    if model != expected:
        raise ProfilerNotImplemented(
            f"{KIND} supports (window, index_topk, head_dim, rope_dim, prefill_chunk_size)="
            f"{expected}, got {model}"
        )
    if num_heads not in _SUPPORTED_HEADS:
        raise ProfilerNotImplemented(
            f"{KIND} takes the kernel's padded head count {_SUPPORTED_HEADS}, got {num_heads}"
        )
    if (str(q_dtype), str(output_dtype)) != ("bf16", "fp8_e4m3"):
        raise ProfilerNotImplemented(f"{KIND} supports bf16 q and fp8_e4m3 output")
    if swa_cache_format != "mxfp8":
        raise ProfilerNotImplemented(f"{KIND} supports the mxfp8 (528 B) SWA cache")
    if compress_ratio == 0:
        if compressed_cache_format != "none":
            raise ValueError("compress_ratio=0 layers have compressed_cache_format='none'")
        extra_format = None
    else:
        if compressed_cache_format not in _RECORD_BYTES:
            raise ProfilerNotImplemented(
                f"{KIND} compressed cache must be one of {sorted(_RECORD_BYTES)}"
            )
        extra_format = compressed_cache_format
    return _Shape(
        mode,
        query_context_pairs,
        compress_ratio,
        num_heads,
        max_model_len,
        max_num_batched_tokens,
        extra_format,
    )


@dataclass
class _Workload:
    """One synthetic batch; every tensor lives on the profiling device."""

    q: Any  # [T, H, 512] bf16, standard layout
    positions: Any  # [T] int64
    cos_sin: Any  # [P, 64] fp32
    sink: Any  # [H] fp32
    swa_cache: Any  # [blocks, 32, 528] uint8
    swa_block_table: Any  # [R, max_blocks] int32
    swa_slots: Any  # [T, 128] int32 physical slots, -1 unused
    swa_lens: Any  # [T] int32
    extra_cache: Any | None  # [blocks, 128 // ratio, bytes] uint8
    extra_block_table: Any | None  # [R, max_blocks] int32
    extra_slots: Any | None  # [T, 512] int32 physical slots, -1 unused
    extra_lens: Any | None  # [T] int32
    topk_local: Any  # [T, 512] int32 request-local compressed rows, -1 unused
    seq_lens: Any  # [R] int32 (context incl. query)
    query_lens: Any  # [R] int32


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _cos_sin(torch: Any, max_pos: int, device: Any) -> Any:
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, _ROPE_DIM, 2, device=device, dtype=torch.float32) / _ROPE_DIM)
    )
    freqs = torch.outer(torch.arange(max_pos, device=device, dtype=torch.float32), inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous()


def _paged_table(
    torch: Any,
    generator: Any,
    ranges: list[tuple[int, int]],
    block: int,
    device: Any,
) -> tuple[Any, int, list[Any]]:
    """Block table for per-request live row ranges; block 0 is a shared dummy.

    Blocks outside a request's live range point at the dummy block, which no
    kernel reads. Live blocks get a shuffled physical order, as a paged
    allocator would hand them out.
    """
    width = max(1, max(_cdiv(end, block) for _, end in ranges))
    live = [list(range(start // block, _cdiv(end, block))) for start, end in ranges]
    count = sum(len(blocks) for blocks in live)
    physical = (torch.randperm(count, generator=generator) + 1).tolist()
    table = torch.zeros((len(ranges), width), dtype=torch.int32)
    cursor = 0
    slots = []
    for request, blocks in enumerate(live):
        ids = physical[cursor : cursor + len(blocks)]
        cursor += len(blocks)
        if blocks:
            table[request, blocks[0] : blocks[-1] + 1] = torch.tensor(ids, dtype=torch.int32)
        start, end = ranges[request]
        rows = torch.arange(start, end, dtype=torch.int64)
        slots.append(table[request, rows // block].long() * block + rows % block)
    return table.to(device), count + 1, slots


def _slot_ids(torch: Any, table: Any, request: Any, local: Any, block: int) -> Any:
    """Map request-local rows (``-1`` unused) to physical slot ids."""
    valid = local >= 0
    safe = local.clamp(min=0).long()
    physical = table[request[:, None], safe // block].long() * block + safe % block
    return torch.where(valid, physical, -1).to(torch.int32)


def _stratified_topk(torch: Any, generator: Any, available: Any, device: Any) -> Any:
    """``min(available, 512)`` distinct rows in ``[0, available)`` per token, ascending."""
    count = available.clamp(max=_INDEX_TOPK)
    columns = torch.arange(_INDEX_TOPK, dtype=torch.int64)
    denominator = count.clamp(min=1)[:, None]
    lower = columns[None, :] * available[:, None] // denominator
    upper = (columns[None, :] + 1) * available[:, None] // denominator
    jitter = torch.rand((available.shape[0], _INDEX_TOPK), generator=generator)
    picked = lower + (jitter * (upper - lower).clamp(min=1)).long()
    picked = torch.minimum(picked, (upper - 1).clamp(min=0))
    return torch.where(columns[None, :] < count[:, None], picked, -1).to(torch.int32).to(device)


def _build_workload(torch: Any, shape: _Shape, device: Any, writer: Any) -> _Workload:
    """Allocate and fill one batch.

    ``writer(cache, role, fmt, slots, rows, positions, cos_sin)`` stores bf16 rows,
    where ``role`` is ``"swa"`` or ``"compressed"``.
    """
    generator = torch.Generator().manual_seed(0x5EED41)
    pairs = shape.pairs
    num_heads = shape.num_heads
    max_context = max(context for _, context in pairs)
    cos_sin = _cos_sin(torch, max_context, device)

    request = torch.cat(
        [torch.full((query,), index, dtype=torch.int64) for index, (query, _) in enumerate(pairs)]
    )
    positions = torch.cat(
        [torch.arange(context - query, context, dtype=torch.int64) for query, context in pairs]
    )

    # SWA: live rows are the window below the first query token through the last.
    swa_ranges = [(max(0, context - query - (_WINDOW - 1)), context) for query, context in pairs]
    swa_table, swa_blocks, swa_live = _paged_table(torch, generator, swa_ranges, _SWA_BLOCK, device)
    swa_cache = torch.zeros(
        (swa_blocks, _SWA_BLOCK, _RECORD_BYTES["mxfp8"]), dtype=torch.uint8, device=device
    )
    swa_start = (positions - (_WINDOW - 1)).clamp(min=0)
    offsets = torch.arange(_WINDOW, dtype=torch.int64)
    swa_lens = positions - swa_start + 1
    swa_local = torch.where(offsets[None, :] < swa_lens[:, None], swa_start[:, None] + offsets, -1)
    swa_table_cpu = swa_table.cpu()
    swa_slots = _slot_ids(torch, swa_table_cpu, request, swa_local, _SWA_BLOCK).to(device)

    extra_cache = extra_table = extra_slots = extra_lens = None
    if shape.ratio:
        block = shape.compressed_block
        compressed = [context // shape.ratio for _, context in pairs]
        extra_table, extra_blocks, extra_live = _paged_table(
            torch, generator, [(0, rows) for rows in compressed], block, device
        )
        extra_cache = torch.zeros(
            (extra_blocks, block, _RECORD_BYTES[shape.extra_format]),
            dtype=torch.uint8,
            device=device,
        )
        available = (positions + 1) // shape.ratio
        topk_local = _stratified_topk(torch, generator, available, device)
        extra_slots = _slot_ids(torch, extra_table.cpu(), request, topk_local.cpu(), block).to(
            device
        )
        extra_lens = available.clamp(max=_INDEX_TOPK).to(torch.int32).to(device)
    else:
        topk_local = torch.full(
            (positions.shape[0], _INDEX_TOPK), -1, dtype=torch.int32, device=device
        )

    def fill(cache: Any, role: str, fmt: str, live: list[Any]) -> None:
        slots = torch.cat(live)
        for start in range(0, slots.numel(), 65536):
            chunk = slots[start : start + 65536]
            rows = torch.randn((chunk.numel(), _HEAD_DIM), generator=generator)
            writer(
                cache,
                role,
                fmt,
                chunk.to(device),
                rows.to(device=device, dtype=torch.bfloat16),
                (torch.arange(start, start + chunk.numel()) % max_context).to(device),
                cos_sin,
            )

    fill(swa_cache, "swa", "mxfp8", swa_live)
    if extra_cache is not None:
        fill(extra_cache, "compressed", shape.extra_format, extra_live)

    q = torch.randn((positions.shape[0], num_heads, _HEAD_DIM), generator=generator)
    sink = torch.randn((num_heads,), generator=generator)
    return _Workload(
        q=q.to(device=device, dtype=torch.bfloat16),
        positions=positions.to(device),
        cos_sin=cos_sin,
        sink=sink.to(device),
        swa_cache=swa_cache,
        swa_block_table=swa_table,
        swa_slots=swa_slots,
        swa_lens=swa_lens.to(torch.int32).to(device),
        extra_cache=extra_cache,
        extra_block_table=extra_table,
        extra_slots=extra_slots,
        extra_lens=extra_lens,
        topk_local=topk_local,
        seq_lens=torch.tensor([c for _, c in pairs], dtype=torch.int32, device=device),
        query_lens=torch.tensor([q for q, _ in pairs], dtype=torch.int32, device=device),
    )


def _require_b200(torch: Any) -> Any:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(f"{KIND} requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    name = str(torch.cuda.get_device_name(device))
    if name != _GPU_NAME or torch.cuda.get_device_capability(device)[0] != 10:
        raise ProfilerNotImplemented(f"{KIND} requires {_GPU_NAME} (SM100), got {name}")
    return device


def _logical_work(shape: _Shape, work: _Workload) -> tuple[float, float]:
    """FLOPs and bytes of the attention math plus, for prefill, the KV gathers.

    Scores and values each take ``2 * H * 512`` FLOPs per selected key. Decode
    bytes: Q, the selected quantized records, index words and the FP8 output
    with scales. Prefill bytes add the gathers: each gathered record is read
    and written back as BF16, then attention re-reads a BF16 row per selected key.
    """
    rows = shape.num_tokens
    heads = shape.num_heads
    swa_keys = float(work.swa_lens.sum().item())
    extra_keys = float(work.extra_lens.sum().item()) if work.extra_lens is not None else 0.0
    flops = 4.0 * heads * _HEAD_DIM * (swa_keys + extra_keys)
    q_bytes = rows * heads * _HEAD_DIM * 2
    out_bytes = rows * heads * (_HEAD_DIM + _HEAD_DIM // 32)
    extra_record = _RECORD_BYTES[shape.extra_format] if shape.extra_format else 0
    if shape.mode == "decode":
        index_bytes = rows * 4 * (_WINDOW + (_INDEX_TOPK if shape.ratio else 0))
        key_bytes = swa_keys * _RECORD_BYTES["mxfp8"] + extra_keys * extra_record
        return flops, q_bytes + out_bytes + index_bytes + key_bytes
    gathered_swa = sum(q + min(c - q, _WINDOW - 1) for q, c in shape.pairs)
    gathered_extra = sum(c // shape.ratio for _, c in shape.pairs) if shape.ratio else 0
    gather_bytes = gathered_swa * (_RECORD_BYTES["mxfp8"] + 1024) + gathered_extra * (
        extra_record + 1024
    )
    attention_bytes = (swa_keys + extra_keys) * (1024 + 4)
    return flops, q_bytes + out_bytes + gather_bytes + attention_bytes


def _metrics(shape: _Shape, work: _Workload, time_ms: float, energy_j: float) -> ComputeMetrics:
    flops, logical_bytes = _logical_work(shape, work)
    seconds = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=float(flops / seconds / 1e12),
        memory_bandwidth_gbps=float(logical_bytes / seconds / 1e9),
        energy_j=float(energy_j),
    )


def _reference(torch: Any, shape: _Shape, work: _Workload, rows: Any | None = None) -> Any:
    from profiling.runners.attention.compressed_sparse_mla_rope_cast_reference import (
        compressed_sparse_mla_rope_cast_reference,
    )

    select = slice(None) if rows is None else rows
    return compressed_sparse_mla_rope_cast_reference(
        work.q[select],
        work.positions[select],
        work.cos_sin,
        work.sink,
        _HEAD_DIM**-0.5,
        work.swa_cache,
        work.swa_slots[select],
        work.extra_cache,
        None if work.extra_slots is None else work.extra_slots[select],
        shape.extra_format or "nvfp4",
    )


def _torch_writer(
    cache: Any, _role: str, fmt: str, slots: Any, rows: Any, _positions: Any, _cs: Any
) -> None:
    from profiling.runners.attention.compressed_sparse_mla_rope_cast_reference import write_records

    write_records(cache, slots, rows, fmt)


def _profile(shape: _Shape, backend: str, build_and_check: Any) -> ComputeMetrics:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ProfilerNotImplemented("PyTorch is unavailable") from exc
    try:
        device = _require_b200(torch)
        work, run = build_and_check(torch, device)
        torch.cuda.synchronize(device)
        time_ms = Timer.cupti(run, kernel_name=None)
        energy_j = Energy.perf(run, per_iter_time_ms=time_ms)
        return _metrics(shape, work, time_ms, energy_j)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{KIND}:{backend} ran out of CUDA memory") from exc
    except (KernelLaunchFailed, ProfilerNotImplemented, TypeError, ValueError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{KIND}:{backend} failed: {exc}") from exc


def profile_compressed_sparse_mla_rope_cast_torch(**kwargs: Any) -> ComputeMetrics:
    """Time the Torch semantic composite, including the reference FP8 cast."""
    shape = _validate_args(**kwargs)

    def build(torch: Any, device: Any) -> tuple[_Workload, Any]:
        from profiling.runners.attention.compressed_sparse_mla_rope_cast_reference import (
            quantize_output,
        )

        work = _build_workload(torch, shape, device, _torch_writer)

        def run() -> Any:
            return quantize_output(_reference(torch, shape, work))

        return work, run

    return _profile(shape, "torch", build)


# ---- flashmla_mega ---------------------------------------------------------


def _production_writer(
    cache: Any, role: str, fmt: str, slots: Any, rows: Any, positions: Any, cs: Any
) -> None:
    """Fill caches with the fork's own writers: the MXFP8 SWA insert, the compressor's."""
    if role == "swa":
        from vllm.models.deepseek_v41.common.ops import quantize_and_insert_k_cache

        quantize_and_insert_k_cache(
            rows,
            cache.view(cache.shape[0], -1),
            slots,
            block_size=cache.shape[1],
            bytes_per_token=_RECORD_BYTES["mxfp8"],
        )
        return
    from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import rope_quant_insert

    rope_quant_insert(rows.contiguous(), positions, cs, cache, slots, 1)


def _check_rows(torch: Any, num_tokens: int) -> Any:
    if num_tokens <= _CHECK_ROWS:
        return torch.arange(num_tokens)
    return torch.linspace(0, num_tokens - 1, _CHECK_ROWS).round().long().unique()


def check_against_reference(torch: Any, shape: _Shape, work: _Workload, out: Any) -> dict:
    """Compare the kernel's FP8 output to the Torch reference on sampled rows.

    The kernel output is dequantized from its fused layout. Tolerances: each
    UE8M0 exponent within 1 of the reference's, and each (row, head)
    relative L2 error within ``_REL_TOL``. The FP8 cast alone gives about
    0.03 (a 3-bit mantissa).
    """
    from profiling.runners.attention.compressed_sparse_mla_rope_cast_reference import (
        dequantize_output,
        output_from_fused_layout,
        quantize_output,
    )

    rows = _check_rows(torch, shape.num_tokens).to(work.q.device)
    values, exponents = output_from_fused_layout(out.data[rows], out.scale[rows])
    got = dequantize_output(values, exponents)
    expected = _reference(torch, shape, work, rows)
    _ref_values, ref_exponents = quantize_output(expected)
    exponent_gap = int((exponents.int() - ref_exponents.int()).abs().max().item())
    rel = (got - expected).norm(dim=-1) / expected.norm(dim=-1).clamp(min=1e-6)
    report = {
        "rows_checked": int(rows.numel()),
        "max_rel_l2": float(rel.max().item()),
        "mean_rel_l2": float(rel.mean().item()),
        "max_abs": float((got - expected).abs().max().item()),
        "max_exponent_gap": exponent_gap,
    }
    if not torch.isfinite(got).all() or exponent_gap > 1 or report["max_rel_l2"] > _REL_TOL:
        raise KernelLaunchFailed(f"{KIND}:flashmla_mega mismatches the Torch reference: {report}")
    return report


def build_flashmla_mega(torch: Any, shape: _Shape, device: Any) -> tuple[_Workload, Any, Any]:
    """Build the workload and a closure running the production segment method."""
    try:
        from vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn import (
            DeepseekV4MegaAttnAttention,
            alloc_mega_attn_output,
            is_flashmla_mega_attn_supported,
        )
        from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata
        from vllm.v1.worker.workspace import (
            init_workspace_manager,
            is_workspace_manager_initialized,
        )
    except ImportError as exc:
        raise ProfilerNotImplemented(
            f"{KIND}:flashmla_mega requires the upstream-rebased vLLM fork (vllm_upstream_fork_env)"
        ) from exc
    supported, reason = is_flashmla_mega_attn_supported()
    if not supported:
        raise ProfilerNotImplemented(f"{KIND}:flashmla_mega unavailable: {reason}")

    work = _build_workload(torch, shape, device, _production_writer)
    num_tokens = shape.num_tokens
    out = alloc_mega_attn_output(num_tokens, shape.num_heads // 8, device)
    q_fused = _q_fused(work.q)
    positions32 = work.positions.to(torch.int32)
    layer = types.SimpleNamespace(
        compress_ratio=shape.ratio,
        PREFILL_CHUNK_SIZE=_PREFILL_CHUNK_SIZE,
        window_size=_WINDOW,
        max_num_batched_tokens=shape.max_num_batched_tokens,
        topk_indices_buffer=work.topk_local,
        swa_cache_layer=types.SimpleNamespace(kv_cache=work.swa_cache),
        scale=_HEAD_DIM**-0.5,
        attn_sink=work.sink,
        rotary_emb=types.SimpleNamespace(cos_sin_cache=work.cos_sin),
        n_wv_group=shape.num_heads // 8,
        _compressed_kv_cache=lambda: work.extra_cache,
    )
    flashmla_metadata = (
        types.SimpleNamespace(block_table=work.extra_block_table, block_size=_KV_BLOCK)
        if shape.ratio
        else None
    )

    if shape.mode == "decode":
        extra = (
            (work.extra_cache.unsqueeze(-2), work.extra_slots, work.extra_lens)
            if shape.ratio
            else (None, None, None)
        )
        # The remap kernel (compute_global_topk_indices_and_lens) is its own
        # Sim slot; hand the method the slot ids it would have produced.
        layer._decode_compressed_kv_and_topk = lambda *_args: extra
        swa_metadata = types.SimpleNamespace(
            decode_swa_indices=work.swa_slots, decode_swa_lens=work.swa_lens
        )

        def run() -> None:
            DeepseekV4MegaAttnAttention._forward_decode_mega(
                layer, q_fused, positions32, flashmla_metadata, swa_metadata, out
            )

        return work, run, out

    if not is_workspace_manager_initialized():
        init_workspace_manager(device)
    num_prefills = len(shape.pairs)
    query_start = torch.zeros(num_prefills + 1, dtype=torch.int32)
    query_start[1:] = torch.cumsum(work.query_lens.cpu(), 0)
    seq_lens_cpu = work.seq_lens.cpu()
    query_lens_cpu = work.query_lens.cpu()
    prefix = seq_lens_cpu - query_lens_cpu
    gather_lens = query_lens_cpu + prefix.clamp(max=_WINDOW - 1)
    swa_metadata = types.SimpleNamespace(
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=num_prefills,
        num_prefill_tokens=num_tokens,
        prefill_seq_lens=work.seq_lens,
        prefill_seq_lens_cpu=seq_lens_cpu,
        prefill_query_lens_cpu=query_lens_cpu,
        prefill_gather_lens=gather_lens.to(torch.int32).to(device),
        prefill_max_model_len=shape.max_model_len,
        prefill_window_size=_WINDOW,
        prefill_max_num_batched_tokens=shape.max_num_batched_tokens,
        query_start_loc_cpu=query_start,
        query_start_loc=query_start.to(device),
        block_table=work.swa_block_table,
        block_size=_SWA_BLOCK,
    )
    swa_metadata.get_prefill_chunk_plan = types.MethodType(
        DeepseekSparseSWAMetadata.get_prefill_chunk_plan, swa_metadata
    )

    def run() -> None:
        DeepseekV4MegaAttnAttention._forward_prefill_mega(
            layer, q_fused, positions32, flashmla_metadata, swa_metadata, out, 0
        )

    return work, run, out


def _q_fused(q: Any) -> Any:
    from profiling.runners.attention.compressed_sparse_mla_rope_cast_reference import (
        q_to_fused_layout,
    )

    return q_to_fused_layout(q).contiguous()


def profile_compressed_sparse_mla_rope_cast_flashmla_mega(**kwargs: Any) -> ComputeMetrics:
    """Time the fork's mega-attention decode or prefill segment on a B200."""
    shape = _validate_args(**kwargs)

    def build(torch: Any, device: Any) -> tuple[_Workload, Any]:
        work, run, out = build_flashmla_mega(torch, shape, device)
        run()
        torch.cuda.synchronize(device)
        check_against_reference(torch, shape, work, out)
        return work, run

    return _profile(shape, "flashmla_mega", build)


__all__ = [
    "profile_compressed_sparse_mla_rope_cast_flashmla_mega",
    "profile_compressed_sparse_mla_rope_cast_torch",
]
