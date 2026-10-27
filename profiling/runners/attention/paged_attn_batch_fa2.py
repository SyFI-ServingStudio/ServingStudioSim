"""Profile FlashInfer's FA2 paged batch attention as vLLM calls it off TRTLLM.

On a GPU without TRTLLM attention (sm_8x such as the L40S), vLLM's FlashInfer
backend (``vllm/v1/attention/backends/flashinfer.py``) runs both halves of a
step through FlashInfer's native wrappers, with a BF16 query over the FP8
cache (``get_q_data_type``: "Architectures with only fa2 (e.g. SM89, SM120)
cannot consume FP8 queries"):

- the decode requests (``paged_attn_batch_decode``) through
  ``BatchDecodeWithPagedKVCacheWrapper(use_tensor_cores=True)``, which plans
  and runs FlashInfer's batch *prefill* kernel with a 16-row query tile, then
  merges the split-KV partial states with
  ``PersistentVariableLengthMergeStatesKernel``. A decode-only step replays
  vLLM's FULL CUDA graph of that wrapper (``use_cuda_graph=True``), whose plan
  always splits the KV and always merges;
- the prefill requests (``paged_attn_batch_prefill``) through
  ``BatchPrefillWithPagedKVCacheWrapper``: one causal batch prefill kernel, and
  a merge only when the plan splits the KV (few query tiles).

Each call is timed with CUPTI as the busy union of every launch it makes;
planning stays outside the timed closure, as in vLLM's metadata builder.

Cache layout. vLLM's hybrid allocator gives this model 1568-token cache
blocks, but FlashInfer only takes 16-, 32- or 64-token kernel pages off
Blackwell (``get_supported_kernel_block_sizes``), so vLLM splits each block into
49 pages of 32 tokens (``select_common_block_size``: the largest supported size
dividing 1568). Under the ``LBNHC`` layout a layer's view is
``[pages, 32, kv_heads, 2 * head_dim]`` (NHD), contiguous within the layer,
and the backend passes the two halves of the last dimension as K and V. The
runner allocates the same pages, keeps the 49 pages of a block consecutive,
shuffles the blocks across a pool sized to the batch, and keeps block 0 as the
null block.

Scales. ``wrapper.run`` gets ``q_scale``, ``k_scale`` and ``v_scale`` as host
floats; the checkpoint carries no KV scales, so all three are 1.0 and the FP8
values are the dequantized values the reference reads (a ``v_scale`` other
than 1 would add an output multiply, which the capture does not show).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention.paged_attn_batch_trtllm_gen import (
    _DecodeShape,
    _Geometry,
    _PrefillShape,
    _spread,
)
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

BACKEND = "flashinfer_fa2"
_DECODE = "paged_attn_batch_decode"
_PREFILL = "paged_attn_batch_prefill"
# FlashInfer kernel page: the largest of its fa2 sizes (16, 32, 64) dividing
# vLLM's 1568-token hybrid block.
_PAGE_SIZE = 32
# Kernel pages of one vLLM cache block, kept consecutive in the pool.
_BLOCK_PAGES = 49
# vLLM's VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE default, and its per-token
# floor for the prefill temporaries (FLASHINFER_PREFILL_WORKSPACE_BYTES_PER_ELEM)
# at the default 8192-token chunk.
_WORKSPACE_FLOOR_BYTES = 394 * 1024 * 1024
_WORKSPACE_BYTES_PER_ELEM = 16
_WORKSPACE_MIN_TOKENS = 8192
# One layer's KV per call: past it the batch does not fit beside the model on
# a 48 GB card.
_MAX_KV_BYTES = 8 << 30
_MAX_REQUESTS = 1024
_MAX_QUERY_TOKENS = 65536
_DTYPES = ("bf16", "fp8_e4m3", "bf16")
# Correctness check against the float32 reference: BF16 queries, FP8 values,
# BF16 probabilities in the PV product.
_CHECK_MAX_REQUESTS = 16
_CHECK_ROWS_PER_REQUEST = 8
_ABS_TOL = 0.03
_REL_L2_TOL = 0.03


def _positive(name: str, value: object) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _validate_geometry(
    kind: str,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_dtype: object,
    kv_dtype: object,
    o_dtype: object,
) -> _Geometry:
    for name, value in (
        ("num_qo_heads", num_qo_heads),
        ("num_kv_heads", num_kv_heads),
        ("head_dim", head_dim),
    ):
        _positive(name, value)
    if num_qo_heads % num_kv_heads:
        raise ValueError("num_qo_heads must be a multiple of num_kv_heads")
    if (str(q_dtype), str(kv_dtype), str(o_dtype)) != _DTYPES:
        raise ProfilerNotImplemented(
            f"{kind}:{BACKEND} supports bf16 queries over an fp8_e4m3 cache with bf16 output"
        )
    if head_dim not in (64, 128, 256):
        raise ProfilerNotImplemented(f"{kind}:{BACKEND} supports head_dim 64, 128 or 256")
    return _Geometry(num_qo_heads, num_kv_heads, head_dim)


def validate_decode_args(
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_dtype: object,
    kv_dtype: object,
    o_dtype: object,
    batch_size: int,
    max_kv_len: int,
    total_kv_len: int,
) -> _DecodeShape:
    geometry = _validate_geometry(
        _DECODE, num_qo_heads, num_kv_heads, head_dim, q_dtype, kv_dtype, o_dtype
    )
    for name, value in (
        ("batch_size", batch_size),
        ("max_kv_len", max_kv_len),
        ("total_kv_len", total_kv_len),
    ):
        _positive(name, value)
    if not max_kv_len + batch_size - 1 <= total_kv_len <= batch_size * max_kv_len:
        raise ValueError(
            "total_kv_len must lie in [max_kv_len + batch_size - 1, batch_size * max_kv_len]"
        )
    if batch_size > _MAX_REQUESTS:
        raise ProfilerNotImplemented(f"{_DECODE} supports batch_size <= {_MAX_REQUESTS}")
    if total_kv_len * geometry.kv_bytes_per_token > _MAX_KV_BYTES:
        raise ProfilerNotImplemented(
            f"{_DECODE}:{BACKEND} caps the batch's KV at {_MAX_KV_BYTES} bytes"
        )
    return _DecodeShape(geometry, batch_size, max_kv_len, total_kv_len)


def validate_prefill_args(
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_dtype: object,
    kv_dtype: object,
    o_dtype: object,
    requests: tuple[tuple[int, int], ...],
) -> _PrefillShape:
    geometry = _validate_geometry(
        _PREFILL, num_qo_heads, num_kv_heads, head_dim, q_dtype, kv_dtype, o_dtype
    )
    if not requests:
        raise ValueError("requests must hold at least one (prefix, new tokens) pair")
    for pair in requests:
        if not isinstance(pair, tuple) or len(pair) != 2 or any(type(v) is not int for v in pair):
            raise TypeError("requests must contain integer (prefix, new tokens) pairs")
        prefix, new = pair
        if prefix < 0 or new <= 0:
            raise ValueError("each request needs a non-negative prefix and new tokens")
    shape = _PrefillShape(geometry, tuple(requests))
    if shape.num_seqs > _MAX_REQUESTS or shape.query_tokens > _MAX_QUERY_TOKENS:
        raise ProfilerNotImplemented(
            f"{_PREFILL} supports up to {_MAX_REQUESTS} requests and {_MAX_QUERY_TOKENS} new tokens"
        )
    if sum(shape.lengths) * geometry.kv_bytes_per_token > _MAX_KV_BYTES:
        raise ProfilerNotImplemented(
            f"{_PREFILL}:{BACKEND} caps the batch's KV at {_MAX_KV_BYTES} bytes"
        )
    return shape


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def workspace_bytes(geometry: _Geometry, query_tokens: int) -> int:
    """vLLM's FlashInfer workspace (FlashInferMetadataBuilder._get_workspace_buffer).

    vLLM sizes it for max_num_batched_tokens (8192 by default); a call with more
    query tokens than that needs at least its own.
    """
    tokens = max(_WORKSPACE_MIN_TOKENS, query_tokens)
    est = tokens * geometry.num_qo_heads * geometry.head_dim * _WORKSPACE_BYTES_PER_ELEM
    return max(_WORKSPACE_FLOOR_BYTES, est)


@dataclass
class PageTable:
    """FlashInfer's CSR page table plus the padded per-request rows the check reads."""

    page_counts: list[int]
    indptr: Any  # [requests + 1] int32, CPU
    indices: Any  # [total pages] int32, CPU
    last_page_len: Any  # [requests] int32, CPU
    block_table: Any  # [requests, max pages] int32, CPU (null page past the end)
    num_pages: int


def page_table(torch: Any, lengths: list[int], generator: Any) -> PageTable:
    """Pages for each request, whole 49-page blocks shuffled across the pool.

    Block 0 is the null block. A request's pages run through its blocks in
    order, so the pages inside one block stay consecutive as vLLM's view of a
    1568-token block makes them.
    """
    page_counts = [_cdiv(length, _PAGE_SIZE) for length in lengths]
    block_counts = [_cdiv(count, _BLOCK_PAGES) for count in page_counts]
    num_blocks = sum(block_counts) + 1
    order = (torch.randperm(num_blocks - 1, generator=generator) + 1).tolist()
    rows, cursor = [], 0
    for count, blocks in zip(page_counts, block_counts):
        pages = [
            block * _BLOCK_PAGES + page
            for block in order[cursor : cursor + blocks]
            for page in range(_BLOCK_PAGES)
        ]
        rows.append(pages[:count])
        cursor += blocks
    indptr = [0]
    for count in page_counts:
        indptr.append(indptr[-1] + count)
    width = max(1, max(page_counts))
    block_table = torch.zeros((len(lengths), width), dtype=torch.int32)
    for request, pages in enumerate(rows):
        block_table[request, : len(pages)] = torch.tensor(pages, dtype=torch.int32)
    last = [length - (count - 1) * _PAGE_SIZE for length, count in zip(lengths, page_counts)]
    return PageTable(
        page_counts=page_counts,
        indptr=torch.tensor(indptr, dtype=torch.int32),
        indices=torch.tensor([p for pages in rows for p in pages], dtype=torch.int32),
        last_page_len=torch.tensor(last, dtype=torch.int32),
        block_table=block_table,
        num_pages=num_blocks * _BLOCK_PAGES,
    )


@dataclass
class _Workload:
    q: Any  # [tokens, Hq, D] bf16
    raw_cache: Any  # [pages, page, Hkv, 2D] fp8 (NHD, K then V per head)
    kv_cache: tuple  # K and V views [pages, page, Hkv, D]
    table: PageTable
    lengths: list[int]
    out: Any  # [tokens, Hq, D] bf16


def _fill_fp8(torch: Any, raw: Any, generator: Any) -> None:
    """Random N(0, 0.5) values, cast to FP8 a slice at a time (no full-size bf16 copy)."""
    step = max(1, (64 << 20) // max(1, raw[0].numel()))
    for start in range(0, raw.shape[0], step):
        chunk = raw[start : start + step]
        values = torch.empty(chunk.shape, dtype=torch.bfloat16, device=raw.device)
        values.normal_(std=0.5, generator=generator)
        chunk.copy_(values.to(torch.float8_e4m3fn))


def _build_workload(
    torch: Any, geometry: _Geometry, lengths: list[int], tokens: int, device: Any
) -> _Workload:
    host = torch.Generator().manual_seed(0xFA2)
    table = page_table(torch, lengths, host)
    hkv, hq, dim = geometry.num_kv_heads, geometry.num_qo_heads, geometry.head_dim
    raw_cache = torch.empty(
        (table.num_pages, _PAGE_SIZE, hkv, 2 * dim), dtype=torch.float8_e4m3fn, device=device
    )
    cuda = torch.Generator(device=device).manual_seed(0xFA2)
    _fill_fp8(torch, raw_cache, cuda)
    q = (torch.randn((tokens, hq, dim), generator=host) * 0.5).to(
        device=device, dtype=torch.bfloat16
    )
    return _Workload(
        q=q,
        raw_cache=raw_cache,
        kv_cache=(raw_cache[..., :dim], raw_cache[..., dim:]),
        table=table,
        lengths=list(lengths),
        out=torch.empty((tokens, hq, dim), dtype=torch.bfloat16, device=device),
    )


def _import_flashinfer(kind: str) -> Any:
    try:
        import flashinfer
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{kind}:{BACKEND} requires FlashInfer (vllm_env)") from exc
    return flashinfer


def build_decode(torch: Any, shape: _DecodeShape, device: Any) -> tuple[_Workload, Any]:
    """Workload plus a closure running vLLM's planned decode wrapper.

    The wrapper is the one vLLM's builder keeps per CUDA-graph batch size
    (``_get_decode_wrapper(use_cudagraph=True)``) and plans as
    ``fast_plan_decode``'s first call does.
    """
    flashinfer = _import_flashinfer(_DECODE)
    geometry = shape.geometry
    lengths = shape.lengths
    work = _build_workload(torch, geometry, lengths, shape.batch_size, device)
    table = work.table
    batch = shape.batch_size
    workspace = torch.zeros(workspace_bytes(geometry, batch), dtype=torch.uint8, device=device)
    indptr_buf = table.indptr.to(device)
    indices_buf = table.indices.to(device)
    last_buf = table.last_page_len.to(device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        use_cuda_graph=True,
        paged_kv_indptr_buffer=indptr_buf,
        paged_kv_indices_buffer=indices_buf,
        paged_kv_last_page_len_buffer=last_buf,
        use_tensor_cores=True,
        backend="fa2",
    )
    wrapper.plan(
        indptr=table.indptr,
        indices=indices_buf,
        last_page_len=table.last_page_len,
        num_qo_heads=geometry.num_qo_heads,
        num_kv_heads=geometry.num_kv_heads,
        head_dim=geometry.head_dim,
        page_size=_PAGE_SIZE,
        pos_encoding_mode="NONE",
        window_left=-1,
        logits_soft_cap=None,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.bfloat16,
        sm_scale=geometry.head_dim**-0.5,
        non_blocking=True,
        block_tables=None,
        seq_lens=torch.tensor(lengths, dtype=torch.int32),
        fixed_split_size=-1,
        disable_split_kv=False,
    )

    def run() -> Any:
        # FlashInferImpl.forward's native decode call (flashinfer.py).
        return wrapper.run(
            work.q,
            work.kv_cache,
            q_scale=1.0,
            k_scale=1.0,
            v_scale=1.0,
            out=work.out,
            kv_cache_sf=None,
            sinks=None,
        )

    return work, run


def build_prefill(torch: Any, shape: _PrefillShape, device: Any) -> tuple[_Workload, Any]:
    """Workload plus a closure running vLLM's planned causal prefill wrapper."""
    flashinfer = _import_flashinfer(_PREFILL)
    geometry = shape.geometry
    work = _build_workload(torch, geometry, shape.lengths, shape.query_tokens, device)
    table = work.table
    workspace = torch.zeros(
        workspace_bytes(geometry, shape.query_tokens), dtype=torch.uint8, device=device
    )
    qo_indptr = torch.zeros(shape.num_seqs + 1, dtype=torch.int32)
    qo_indptr[1:] = torch.cumsum(torch.tensor(shape.appends), 0)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    wrapper.plan(
        qo_indptr=qo_indptr,
        paged_kv_indptr=table.indptr,
        paged_kv_indices=table.indices.to(device),
        paged_kv_last_page_len=table.last_page_len,
        seq_lens=torch.tensor(shape.lengths, dtype=torch.int32),
        num_qo_heads=geometry.num_qo_heads,
        num_kv_heads=geometry.num_kv_heads,
        head_dim_qk=geometry.head_dim,
        page_size=_PAGE_SIZE,
        causal=True,
        sm_scale=geometry.head_dim**-0.5,
        window_left=-1,
        logits_soft_cap=None,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.bfloat16,
        fixed_split_size=-1,
        disable_split_kv=False,
    )

    def run() -> Any:
        # FlashInferImpl.forward's native prefill call (flashinfer.py).
        return wrapper.run(
            work.q,
            work.kv_cache,
            q_scale=1.0,
            k_scale=1.0,
            v_scale=1.0,
            out=work.out,
            kv_cache_sf=None,
        )

    return work, run


def _check_rows(torch: Any, kind: str, geometry: _Geometry, work: _Workload, picks: list) -> dict:
    """Compare output rows with the float32 reference.

    ``picks`` holds ``(request, output rows, positions)``: a request's query
    rows to check and the absolute position of each in its request.
    """
    from profiling.runners.attention.fa4_varlen_attention_reference import (
        gather_request_kv,
        varlen_attention_reference,
    )

    key_cache, value_cache = work.kv_cache  # already [pages, page, heads, dim]
    max_abs, max_rel, rows_checked = 0.0, 0.0, 0
    for request, rows, positions in picks:
        keys, values = gather_request_kv(
            key_cache, value_cache, work.table.block_table[request], work.lengths[request]
        )
        rows, positions = rows.to(work.q.device), positions.to(work.q.device)
        expected = varlen_attention_reference(
            work.q[rows].float(),
            positions,
            keys,
            values,
            softmax_scale=geometry.head_dim**-0.5,
            window=0,
        )
        got = work.out[rows].float()
        if not torch.isfinite(got).all():
            raise KernelLaunchFailed(f"{kind}:{BACKEND} produced non-finite output")
        max_abs = max(max_abs, float((got - expected).abs().max().item()))
        rel = (got - expected).norm(dim=-1) / expected.norm(dim=-1).clamp(min=1e-6)
        max_rel = max(max_rel, float(rel.max().item()))
        rows_checked += int(rows.numel())
    report = {"rows_checked": rows_checked, "max_abs": max_abs, "max_rel_l2": max_rel}
    if max_abs > _ABS_TOL or max_rel > _REL_L2_TOL:
        raise KernelLaunchFailed(f"{kind}:{BACKEND} mismatches the PyTorch reference: {report}")
    return report


def check_decode(torch: Any, shape: _DecodeShape, work: _Workload) -> dict:
    lengths = shape.lengths
    picks = [
        (r, torch.tensor([r]), torch.tensor([lengths[r] - 1]))
        for r in _spread(torch, len(lengths), _CHECK_MAX_REQUESTS)
    ]
    return _check_rows(torch, _DECODE, shape.geometry, work, picks)


def check_prefill(torch: Any, shape: _PrefillShape, work: _Workload) -> dict:
    appends, starts = shape.appends, [0]
    for append in appends:
        starts.append(starts[-1] + append)
    picks = []
    for r in _spread(torch, shape.num_seqs, _CHECK_MAX_REQUESTS):
        local = torch.tensor(_spread(torch, appends[r], _CHECK_ROWS_PER_REQUEST))
        picks.append((r, starts[r] + local, shape.prefixes[r] + local))
    return _check_rows(torch, _PREFILL, shape.geometry, work, picks)


def _metrics(
    geometry: _Geometry,
    interactions: int,
    tokens: int,
    kv_tokens: int,
    time_ms: float,
    energy_j: float,
) -> ComputeMetrics:
    seconds = time_ms / 1000.0
    flops = 4.0 * geometry.num_qo_heads * geometry.head_dim * interactions
    # BF16 query in, BF16 output out, every request's FP8 K and V once.
    io_bytes = tokens * geometry.num_qo_heads * geometry.head_dim * (2 + 2) + (
        kv_tokens * geometry.kv_bytes_per_token
    )
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=float(flops / seconds / 1e12),
        memory_bandwidth_gbps=float(io_bytes / seconds / 1e9),
        energy_j=float(energy_j),
    )


def _time(torch: Any, kind: str, build: Any, check: Any, shape: Any) -> tuple[float, float]:
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        work, run = build(torch, shape, device)
        run()  # JIT module load + first launch, outside timing
        torch.cuda.synchronize(device)
        check(torch, shape, work)
        time_ms = Timer.cupti(run, kernel_name=None, interval_union=True)
        energy_j = Energy.perf(run, per_iter_time_ms=time_ms)
        return time_ms, energy_j
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{kind}:{BACKEND} ran out of CUDA memory") from exc
    except (KernelLaunchFailed, ProfilerNotImplemented, TypeError, ValueError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{kind}:{BACKEND} failed: {exc}") from exc


def _torch(kind: str) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ProfilerNotImplemented(f"{kind}: PyTorch is unavailable") from exc
    return torch


def profile_paged_attn_batch_decode(**kwargs: Any) -> ComputeMetrics:
    """Time the FA2 decode wrapper's run over the canonical decode batch."""
    shape = validate_decode_args(**kwargs)
    torch = _torch(_DECODE)
    time_ms, energy_j = _time(torch, _DECODE, build_decode, check_decode, shape)
    return _metrics(
        shape.geometry,
        shape.total_kv_len,
        shape.batch_size,
        shape.total_kv_len,
        time_ms,
        energy_j,
    )


def profile_paged_attn_batch_prefill(**kwargs: Any) -> ComputeMetrics:
    """Time the FA2 causal prefill wrapper's run over the given prefill batch."""
    shape = validate_prefill_args(**kwargs)
    torch = _torch(_PREFILL)
    time_ms, energy_j = _time(torch, _PREFILL, build_prefill, check_prefill, shape)
    return _metrics(
        shape.geometry,
        shape.interactions,
        shape.query_tokens,
        sum(shape.lengths),
        time_ms,
        energy_j,
    )
