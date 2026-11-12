"""Profile relative-position attention: the fork's tml FA4 call.

Builds one varlen batch from ``query_context_pairs`` over a paged LBNHC KV cache,
picks ``num_splits`` and the bucketed ``max_seqlen_q`` exactly as
``InklingAttention._attention`` does, and times
``InklingFA4RelAttentionKernel.kernel`` with CUPTI summing every launch.

Cache layout. vLLM allocates this attention cache as LBNHC: per layer a
page is ``[page_size, kv_heads, 2 * head_dim]`` with K and V of one (token, head)
side by side, and ``_split_kv_cache`` hands FA4 the two strided halves
(token stride ``2 * kv_heads * head_dim``). The runner allocates the same raw
pages and splits them the same way. Pages are shuffled across a pool sized to
the batch (page 0 is a never-read dummy); the page table is
``ceil(max_model_len / page_size)`` wide, like vLLM's ``max_num_blocks_per_req``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

KIND = "rel_bias_attention"
_MAX_REQUESTS = 1024  # past this tml_fa4 skips the cu-blocks path (another launch set)
_MAX_TOKENS = 32768
_SUPPORTED_HEAD_DIMS = (128,)
_SUPPORTED_PAGE_SIZES = (16, 32, 64, 128)
_KV_LAYOUTS = ("lbnhc",)
_Q_STD = 4.0
# Correctness check: query rows sampled per request and tolerances against the
# float32 reference. bf16 inputs and output give about 1e-2 absolute error.
_CHECK_ROWS_PER_REQUEST = 16
_CHECK_MAX_REQUESTS = 64
_ABS_TOL = 0.05
_REL_L2_TOL = 0.02


@dataclass(frozen=True)
class _Shape:
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    is_local: bool
    rel_extent: int
    page_size: int
    max_model_len: int
    pairs: tuple[tuple[int, int], ...]

    @property
    def num_tokens(self) -> int:
        return sum(query for query, _ in self.pairs)

    @property
    def max_query_len(self) -> int:
        return max(query for query, _ in self.pairs)

    @property
    def max_kv_len(self) -> int:
        # attention.py `_max_kv_len`: local layers never see more than the window.
        return self.rel_extent if self.is_local else self.max_model_len

    @property
    def window_size(self) -> tuple[int, int]:
        return (self.rel_extent - 1, 0) if self.is_local else (-1, -1)


def _validate_args(
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    is_local: bool,
    rel_extent: int,
    page_size: int,
    kv_layout: str,
    q_dtype: object,
    kv_dtype: object,
    max_model_len: int,
    query_context_pairs: tuple[tuple[int, int], ...],
) -> _Shape:
    for name, value in (
        ("num_qo_heads", num_qo_heads),
        ("num_kv_heads", num_kv_heads),
        ("rel_extent", rel_extent),
        ("max_model_len", max_model_len),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if type(is_local) is not bool:
        raise TypeError(f"is_local must be a bool, got {is_local!r}")
    if num_qo_heads % num_kv_heads:
        raise ValueError("num_qo_heads must be a multiple of num_kv_heads")
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        raise ProfilerNotImplemented(f"{KIND} supports head_dim {_SUPPORTED_HEAD_DIMS}")
    if page_size not in _SUPPORTED_PAGE_SIZES:
        raise ProfilerNotImplemented(f"{KIND} supports page_size {_SUPPORTED_PAGE_SIZES}")
    if kv_layout not in _KV_LAYOUTS:
        raise ProfilerNotImplemented(f"{KIND} supports kv_layout {_KV_LAYOUTS}")
    if (str(q_dtype), str(kv_dtype)) != ("bf16", "bf16"):
        raise ProfilerNotImplemented(f"{KIND} supports bf16 queries and bf16 KV cache")
    if not query_context_pairs or len(query_context_pairs) > _MAX_REQUESTS:
        raise ProfilerNotImplemented(f"{KIND} supports 1..{_MAX_REQUESTS} requests")
    for pair in query_context_pairs:
        if not isinstance(pair, tuple) or len(pair) != 2 or any(type(v) is not int for v in pair):
            raise TypeError("query_context_pairs must contain integer (query, context) pairs")
        query, context = pair
        if query <= 0 or context < query:
            raise ValueError("each pair must satisfy 0 < query <= context")
        if context > max_model_len:
            raise ValueError("context length exceeds max_model_len")
    if sum(query for query, _ in query_context_pairs) > _MAX_TOKENS:
        raise ProfilerNotImplemented(f"{KIND} supports at most {_MAX_TOKENS} query tokens")
    return _Shape(
        num_qo_heads,
        num_kv_heads,
        head_dim,
        is_local,
        rel_extent,
        page_size,
        max_model_len,
        tuple(query_context_pairs),
    )


@dataclass
class _Workload:
    q: Any  # [T, H, D] bf16
    raw_cache: Any  # [pages, page_size, Hkv, 2D] bf16 (LBNHC)
    key_cache: Any  # [pages, page_size, Hkv, D] strided view
    value_cache: Any  # [pages, page_size, Hkv, D] strided view
    block_table: Any  # [R, ceil(max_model_len / page)] int32
    cache_seqlens: Any  # [R] int32
    cu_seqlens_q: Any  # [R + 1] int32
    rel_logits: Any  # [T, H, E] bf16
    out: Any  # [T, H, D] bf16


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _build_workload(torch: Any, shape: _Shape, device: Any) -> _Workload:
    generator = torch.Generator().manual_seed(0x1A7E1)
    pairs = shape.pairs
    page = shape.page_size
    width = _cdiv(shape.max_model_len, page)
    pages_per_request = [_cdiv(context, page) for _, context in pairs]
    num_pages = sum(pages_per_request) + 1
    physical = (torch.randperm(num_pages - 1, generator=generator) + 1).to(torch.int32)
    block_table = torch.zeros((len(pairs), width), dtype=torch.int32)
    cursor = 0
    for request, count in enumerate(pages_per_request):
        block_table[request, :count] = physical[cursor : cursor + count]
        cursor += count

    hkv, dim = shape.num_kv_heads, shape.head_dim
    raw_cache = torch.empty((num_pages, page, hkv, 2 * dim), dtype=torch.bfloat16, device=device)
    raw_cache.normal_()
    key_cache, value_cache = raw_cache.split(dim, dim=-1)

    tokens = shape.num_tokens
    q = torch.randn((tokens, shape.num_qo_heads, dim), generator=generator) * _Q_STD
    rel = torch.randn((tokens, shape.num_qo_heads, shape.rel_extent), generator=generator)
    cu = torch.zeros(len(pairs) + 1, dtype=torch.int32)
    cu[1:] = torch.cumsum(torch.tensor([query for query, _ in pairs]), 0)
    return _Workload(
        q=q.to(device=device, dtype=torch.bfloat16),
        raw_cache=raw_cache,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table.to(device),
        cache_seqlens=torch.tensor([c for _, c in pairs], dtype=torch.int32, device=device),
        cu_seqlens_q=cu.to(device),
        rel_logits=rel.to(device=device, dtype=torch.bfloat16),
        out=torch.empty((tokens, shape.num_qo_heads, dim), dtype=torch.bfloat16, device=device),
    )


def visible_keys(shape: _Shape) -> int:
    """Σ over query tokens of the keys each one attends to."""
    total = 0
    for query, context in shape.pairs:
        for position in range(context - query, context):
            seen = position + 1
            total += min(seen, shape.rel_extent) if shape.is_local else seen
    return total


def logical_bytes(shape: _Shape) -> float:
    """Q, O and rel_logits once, plus each request's visible K/V tokens once."""
    elem = 2
    tokens = shape.num_tokens
    per_token = shape.num_qo_heads * (2 * shape.head_dim + shape.rel_extent) * elem
    kv_tokens = 0
    for query, context in shape.pairs:
        first = max(0, context - query - (shape.rel_extent - 1)) if shape.is_local else 0
        kv_tokens += context - first
    kv_bytes = kv_tokens * shape.num_kv_heads * 2 * shape.head_dim * elem
    return float(tokens * per_token + kv_bytes)


def _metrics(shape: _Shape, time_ms: float, energy_j: float) -> ComputeMetrics:
    seconds = time_ms / 1000.0
    flops = 4.0 * shape.num_qo_heads * shape.head_dim * visible_keys(shape)
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=float(flops / seconds / 1e12),
        memory_bandwidth_gbps=float(logical_bytes(shape) / seconds / 1e9),
        energy_j=float(energy_j),
    )


def check_rows(torch: Any, query: int) -> Any:
    """Query rows of one request to compare: all if few, else spread plus the last."""
    if query <= _CHECK_ROWS_PER_REQUEST:
        return torch.arange(query)
    return torch.linspace(0, query - 1, _CHECK_ROWS_PER_REQUEST).round().long().unique()


def check_against_reference(torch: Any, shape: _Shape, work: _Workload) -> dict:
    """Compare sampled output rows with the float32 PyTorch reference."""
    from profiling.runners.attention.rel_bias_attention_reference import (
        gather_request_kv,
        rel_attention_reference,
    )

    requests = range(len(shape.pairs))
    if len(shape.pairs) > _CHECK_MAX_REQUESTS:
        # Every prefill request plus evenly spread decodes.
        prefills = [r for r, (q, _) in enumerate(shape.pairs) if q > 1]
        spread = torch.linspace(0, len(shape.pairs) - 1, _CHECK_MAX_REQUESTS).long().tolist()
        requests = sorted(set(prefills) | set(spread))
    max_abs, max_rel, rows_checked = 0.0, 0.0, 0
    starts = work.cu_seqlens_q.tolist()
    for request in requests:
        query, context = shape.pairs[request]
        local_rows = check_rows(torch, query).to(work.q.device)
        rows = starts[request] + local_rows
        keys, values = gather_request_kv(
            work.key_cache, work.value_cache, work.block_table[request], context
        )
        expected = rel_attention_reference(
            work.q[rows],
            context - query + local_rows,
            keys,
            values,
            work.rel_logits[rows],
            softmax_scale=1.0 / shape.head_dim,
            is_local=shape.is_local,
        )
        got = work.out[rows].float()
        if not torch.isfinite(got).all():
            raise KernelLaunchFailed(f"{KIND}:tml_fa4 produced non-finite output")
        max_abs = max(max_abs, float((got - expected).abs().max().item()))
        rel = (got - expected).norm(dim=-1) / expected.norm(dim=-1).clamp(min=1e-6)
        max_rel = max(max_rel, float(rel.max().item()))
        rows_checked += int(rows.numel())
    report = {"rows_checked": rows_checked, "max_abs": max_abs, "max_rel_l2": max_rel}
    if max_abs > _ABS_TOL or max_rel > _REL_L2_TOL:
        raise KernelLaunchFailed(f"{KIND}:tml_fa4 mismatches the PyTorch reference: {report}")
    return report


def build_tml_fa4(torch: Any, shape: _Shape, device: Any) -> tuple[_Workload, Any, int]:
    """Workload plus a closure making the production call; also returns num_splits."""
    try:
        from vllm.models.inkling.nvidia.ops.fa4_rel_attention import (
            InklingFA4RelAttentionKernel,
            bucket_max_seqlen_q,
            inkling_fa4_num_splits,
        )
    except ImportError as exc:
        raise ProfilerNotImplemented(
            f"{KIND}:tml_fa4 requires the alignment vLLM fork (vllm_env)"
        ) from exc

    work = _build_workload(torch, shape, device)
    max_seqlen_q = bucket_max_seqlen_q(shape.max_query_len)
    num_splits = inkling_fa4_num_splits(
        is_local=shape.is_local,
        batch_size=len(shape.pairs),
        max_query_len=max_seqlen_q,
        num_heads=shape.num_qo_heads,
        num_kv_heads=shape.num_kv_heads,
        max_kv_len=shape.max_kv_len,
    )

    def run() -> Any:
        return InklingFA4RelAttentionKernel.kernel(
            work.q,
            work.key_cache,
            work.value_cache,
            block_table=work.block_table,
            cache_seqlens=work.cache_seqlens,
            cu_seqlens_q=work.cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=1.0 / shape.head_dim,
            causal=True,
            window_size=shape.window_size,
            rel_extent=shape.rel_extent,
            rel_logits=work.rel_logits,
            num_splits=num_splits,
            out=work.out,
        )

    return work, run, num_splits


def profile_rel_bias_attention_tml_fa4(**kwargs: Any) -> ComputeMetrics:
    """Time the fork's InklingFA4RelAttentionKernel.kernel over one varlen batch."""
    shape = _validate_args(**kwargs)
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ProfilerNotImplemented("PyTorch is unavailable") from exc
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        work, run, _num_splits = build_tml_fa4(torch, shape, device)
        run()  # compile + first launch, outside timing
        torch.cuda.synchronize(device)
        check_against_reference(torch, shape, work)
        time_ms = Timer.cupti(run, kernel_name=None)
        energy_j = Energy.perf(run, per_iter_time_ms=time_ms)
        return _metrics(shape, time_ms, energy_j)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{KIND}:tml_fa4 ran out of CUDA memory") from exc
    except (KernelLaunchFailed, ProfilerNotImplemented, TypeError, ValueError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{KIND}:tml_fa4 failed: {exc}") from exc
