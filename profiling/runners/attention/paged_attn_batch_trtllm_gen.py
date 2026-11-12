"""Profile FlashInfer's trtllm-gen paged batch attention as vLLM calls it.

Two calls, one per kind, made with the arguments ``FlashInferImpl.forward``
passes on a TRTLLM-attention deployment (``vllm/v1/attention/backends/
flashinfer.py``): ``trtllm_batch_decode_with_kv_cache`` for a step's decode
requests (``paged_attn_batch_decode``) and ``trtllm_batch_context_with_kv_cache``
for its prefill requests (``paged_attn_batch_prefill``). Each is timed with
CUPTI as the busy union of every launch it makes.

Cache layout. vLLM's FlashInfer cache is ``[pages, kv_heads, page_size,
2 * head_dim]`` per layer under HND and the backend passes the two halves of the
last dimension as the K and V views. The runner allocates the same raw pages,
shuffles them across a pool sized to the batch and keeps physical page 0 as the
null page that empty rows point at. Pages hold ``_PAGE_SIZE`` tokens: the
kernel time measured the same for 16, 32 and 64.

Scales. vLLM folds the query and key scales into ``bmm1_scale = scale * q_scale
* k_scale`` and passes ``bmm2_scale = v_scale``; the runner's scales are 1, so
the FP8 values are the dequantized values the reference reads.

Decode bound. ``max_seq_len`` only sizes the split of each request's KV range
(at most ``ceil(max_seq_len / 512)`` CTAs, bounded by the SMs). vLLM's FULL CUDA
graph for a decode-only step was captured with ``max_seq_len = max_model_len``;
the runner passes ``max(max_kv_len, 512 * SMs)``, which saturates the split the
same way for any ``max_model_len`` that does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

BACKEND = "flashinfer_trtllm_gen"
_DECODE = "paged_attn_batch_decode"
_PREFILL = "paged_attn_batch_prefill"
_PAGE_SIZE = 32
_NULL_PAGE = 0
# vLLM's VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE default.
_WORKSPACE_BYTES = 394 * 1024 * 1024
# Tokens of KV range one decode CTA covers at the most (2 * mStepKv).
_SPLIT_TOKENS = 512
_MAX_KV_BYTES = 32 << 30
_MAX_REQUESTS = 1024
_MAX_QUERY_TOKENS = 65536
# Correctness check: requests and query rows per request compared against the
# float32 reference. The kernel rounds the softmax probabilities to FP8 before
# the PV product, which leaves a few percent relative error.
_CHECK_MAX_REQUESTS = 16
_CHECK_ROWS_PER_REQUEST = 8
_ABS_TOL = 0.05
_REL_L2_TOL = 0.06


@dataclass(frozen=True)
class _Geometry:
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int

    @property
    def kv_bytes_per_token(self) -> int:
        return 2 * self.num_kv_heads * self.head_dim  # FP8 K and V


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
    if (str(q_dtype), str(kv_dtype), str(o_dtype)) != ("fp8_e4m3", "fp8_e4m3", "bf16"):
        raise ProfilerNotImplemented(
            f"{kind}:{BACKEND} supports fp8_e4m3 queries over an fp8_e4m3 cache with bf16 output"
        )
    return _Geometry(num_qo_heads, num_kv_heads, head_dim)


@dataclass(frozen=True)
class _DecodeShape:
    geometry: _Geometry
    batch_size: int
    max_kv_len: int
    total_kv_len: int

    @property
    def lengths(self) -> list[int]:
        """Canonical batch: one request at max_kv_len, the rest sharing the remainder.

        A decode request reads at least its own token, so every share is >= 1.
        """
        tails = self.batch_size - 1
        if not tails:
            return [self.max_kv_len]
        rest = self.total_kv_len - self.max_kv_len
        base, extra = divmod(rest, tails)
        return [self.max_kv_len] + [base + (1 if i < extra else 0) for i in range(tails)]


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
        raise ProfilerNotImplemented(f"{_DECODE} caps the batch's KV at {_MAX_KV_BYTES} bytes")
    return _DecodeShape(geometry, batch_size, max_kv_len, total_kv_len)


@dataclass(frozen=True)
class _PrefillShape:
    geometry: _Geometry
    requests: tuple[tuple[int, int], ...]  # (cached prefix, new tokens)

    @property
    def num_seqs(self) -> int:
        return len(self.requests)

    @property
    def appends(self) -> list[int]:
        return [new for _, new in self.requests]

    @property
    def prefixes(self) -> list[int]:
        return [prefix for prefix, _ in self.requests]

    @property
    def query_tokens(self) -> int:
        return sum(self.appends)

    @property
    def lengths(self) -> list[int]:
        return [prefix + new for prefix, new in self.requests]

    @property
    def interactions(self) -> int:
        """Σ over query tokens of the keys each one attends to."""
        return sum(a * p + a * (a + 1) // 2 for p, a in self.requests)


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
        raise ProfilerNotImplemented(f"{_PREFILL} caps the batch's KV at {_MAX_KV_BYTES} bytes")
    return shape


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


@dataclass
class _Workload:
    q: Any  # [tokens, Hq, D] fp8
    raw_cache: Any  # [pages, Hkv, page, 2D] fp8 (HND)
    kv_cache: tuple  # K and V views [pages, Hkv, page, D]
    block_table: Any  # [requests, max pages] int32
    seq_lens: Any  # [requests] int32
    page_counts: list[int]
    out: Any  # [tokens, Hq, D] bf16


_WORKSPACE: dict[Any, Any] = {}


def _workspace(torch: Any, device: Any) -> Any:
    """vLLM's process-wide zeroed FlashInfer workspace, allocated once."""
    if device not in _WORKSPACE:
        _WORKSPACE[device] = torch.zeros(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    return _WORKSPACE[device]


def _build_workload(
    torch: Any, geometry: _Geometry, lengths: list[int], tokens: int, device: Any
) -> _Workload:
    generator = torch.Generator().manual_seed(0x7A77)
    page_counts = [_cdiv(length, _PAGE_SIZE) for length in lengths]
    num_pages = sum(page_counts) + 1
    physical = (torch.randperm(num_pages - 1, generator=generator) + 1).to(torch.int32)
    block_table = torch.full(
        (len(lengths), max(1, max(page_counts))), _NULL_PAGE, dtype=torch.int32
    )
    cursor = 0
    for request, count in enumerate(page_counts):
        block_table[request, :count] = physical[cursor : cursor + count]
        cursor += count

    hkv, hq, dim = geometry.num_kv_heads, geometry.num_qo_heads, geometry.head_dim
    raw_cache = torch.empty(
        (num_pages, hkv, _PAGE_SIZE, 2 * dim), dtype=torch.bfloat16, device=device
    )
    raw_cache = raw_cache.normal_(std=0.5).to(torch.float8_e4m3fn)
    kv_cache = (raw_cache[..., :dim], raw_cache[..., dim:])
    q = (torch.randn((tokens, hq, dim), generator=generator) * 0.5).to(
        device=device, dtype=torch.float8_e4m3fn
    )
    return _Workload(
        q=q,
        raw_cache=raw_cache,
        kv_cache=kv_cache,
        block_table=block_table.to(device),
        seq_lens=torch.tensor(lengths, dtype=torch.int32, device=device),
        page_counts=page_counts,
        out=torch.empty((tokens, hq, dim), dtype=torch.bfloat16, device=device),
    )


def _check_rows(
    torch: Any, kind: str, geometry: _Geometry, work: _Workload, picks: list[tuple]
) -> dict:
    """Compare output rows with the float32 reference.

    ``picks`` holds ``(request, output rows, positions)``: a request's query
    rows to check and the absolute position of each in its request.
    """
    from profiling.runners.attention.fa4_varlen_attention_reference import (
        gather_request_kv,
        varlen_attention_reference,
    )

    # The reference reads [pages, page, heads, dim] views.
    key_cache, value_cache = (half.transpose(1, 2) for half in work.kv_cache)
    max_abs, max_rel, rows_checked = 0.0, 0.0, 0
    lengths = work.seq_lens.tolist()
    for request, rows, positions in picks:
        keys, values = gather_request_kv(
            key_cache, value_cache, work.block_table[request], lengths[request]
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


def _spread(torch: Any, count: int, keep: int) -> list[int]:
    if count <= keep:
        return list(range(count))
    return sorted(set(torch.linspace(0, count - 1, keep).round().long().tolist()))


def _import_flashinfer(kind: str) -> Any:
    try:
        import flashinfer.decode as decode
        import flashinfer.prefill as prefill
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{kind}:{BACKEND} requires FlashInfer (vllm_env)") from exc
    return decode, prefill


def build_decode(torch: Any, shape: _DecodeShape, device: Any) -> tuple[_Workload, Any]:
    """Workload plus a closure making vLLM's trtllm_batch_decode_with_kv_cache call."""
    decode, _ = _import_flashinfer(_DECODE)
    geometry = shape.geometry
    work = _build_workload(torch, geometry, shape.lengths, shape.batch_size, device)
    workspace = _workspace(torch, device)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    max_seq_len = max(shape.max_kv_len, _SPLIT_TOKENS * sms)
    bmm1_scale = geometry.head_dim**-0.5

    def run() -> Any:
        # FlashInferImpl.forward's decode call (flashinfer.py): one query token
        # per request, no sinks, full attention, no counter buffer.
        return decode.trtllm_batch_decode_with_kv_cache(
            query=work.q,
            kv_cache=work.kv_cache,
            workspace_buffer=workspace,
            block_tables=work.block_table,
            seq_lens=work.seq_lens,
            max_seq_len=max_seq_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=1.0,
            window_left=-1,
            sinks=None,
            o_sf_scale=None,
            out=work.out,
            kv_layout="HND",
            backend="trtllm-gen",
            q_len_per_req=1,
            max_q_len=None,
            cum_seq_lens_q=None,
            kv_cache_sf=None,
            lse=None,
            return_lse=False,
        )

    return work, run


def build_prefill(torch: Any, shape: _PrefillShape, device: Any) -> tuple[_Workload, Any]:
    """Workload plus a closure making vLLM's trtllm_batch_context_with_kv_cache call."""
    _, prefill = _import_flashinfer(_PREFILL)
    geometry = shape.geometry
    work = _build_workload(torch, geometry, shape.lengths, shape.query_tokens, device)
    workspace = _workspace(torch, device)
    appends = shape.appends
    cum_q = torch.zeros(shape.num_seqs + 1, dtype=torch.int32)
    cum_q[1:] = torch.cumsum(torch.tensor(appends), 0)
    # vLLM's cum_seq_lens_kv counts pages, not tokens (flashinfer.py build()).
    cum_kv = torch.zeros(shape.num_seqs + 1, dtype=torch.int32)
    cum_kv[1:] = torch.cumsum(torch.tensor(work.page_counts), 0)
    cum_q, cum_kv = cum_q.to(device), cum_kv.to(device)
    max_q_len, max_kv_len = max(appends), max(shape.lengths)
    bmm1_scale = geometry.head_dim**-0.5

    def run() -> Any:
        # FlashInferImpl.forward's prefill call (flashinfer.py): FP8 query over
        # the FP8 cache directly, no sinks, full causal attention.
        return prefill.trtllm_batch_context_with_kv_cache(
            query=work.q,
            kv_cache=work.kv_cache,
            workspace_buffer=workspace,
            block_tables=work.block_table,
            seq_lens=work.seq_lens,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=1.0,
            batch_size=shape.num_seqs,
            cum_seq_lens_q=cum_q,
            cum_seq_lens_kv=cum_kv,
            window_left=-1,
            sinks=None,
            o_sf_scale=None,
            out=work.out,
            kv_cache_sf=None,
        )

    return work, run


def check_decode(torch: Any, shape: _DecodeShape, work: _Workload) -> dict:
    lengths = shape.lengths
    nonempty = [r for r, length in enumerate(lengths) if length]
    picks = [
        (r, torch.tensor([r]), torch.tensor([lengths[r] - 1]))
        for r in (nonempty[i] for i in _spread(torch, len(nonempty), _CHECK_MAX_REQUESTS))
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


def _kv_bytes(geometry: _Geometry, lengths: list[int]) -> int:
    return sum(lengths) * geometry.kv_bytes_per_token


def _metrics(
    geometry: _Geometry,
    interactions: int,
    tokens: int,
    kv_bytes: int,
    time_ms: float,
    energy_j: float,
) -> ComputeMetrics:
    seconds = time_ms / 1000.0
    flops = 4.0 * geometry.num_qo_heads * geometry.head_dim * interactions
    # FP8 query in, bf16 output out, every request's K and V once.
    io_bytes = tokens * geometry.num_qo_heads * geometry.head_dim * (1 + 2) + kv_bytes
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=float(flops / seconds / 1e12),
        memory_bandwidth_gbps=float(io_bytes / seconds / 1e9),
        energy_j=float(energy_j),
    )


def call_launches(run: Any) -> list[str]:
    """Kernel names one call launches, in order (an untimed CUPTI capture)."""
    from profiling.profilers.cupti_kernel_profiler import profile_kernel

    summary = profile_kernel(
        run, num_warmup=0, num_iter=1, clear_l2_before_run=False, clear_l2_between_launches=False
    )
    return list(summary.matched_kernel_names)


def _time(torch: Any, kind: str, build: Any, check: Any, shape: Any) -> tuple[float, float]:
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        work, run = build(torch, shape, device)
        run()  # cubin load + first launch, outside timing
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
    """Time trtllm_batch_decode_with_kv_cache over the canonical decode batch."""
    shape = validate_decode_args(**kwargs)
    torch = _torch(_DECODE)
    time_ms, energy_j = _time(torch, _DECODE, build_decode, check_decode, shape)
    geometry = shape.geometry
    return _metrics(
        geometry,
        shape.total_kv_len,
        shape.batch_size,
        _kv_bytes(geometry, shape.lengths),
        time_ms,
        energy_j,
    )


def profile_paged_attn_batch_prefill(**kwargs: Any) -> ComputeMetrics:
    """Time trtllm_batch_context_with_kv_cache over the canonical prefill batch."""
    shape = validate_prefill_args(**kwargs)
    torch = _torch(_PREFILL)
    time_ms, energy_j = _time(torch, _PREFILL, build_prefill, check_prefill, shape)
    geometry = shape.geometry
    return _metrics(
        geometry,
        shape.interactions,
        shape.query_tokens,
        _kv_bytes(geometry, shape.lengths),
        time_ms,
        energy_j,
    )
