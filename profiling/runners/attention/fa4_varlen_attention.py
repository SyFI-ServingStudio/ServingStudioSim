"""Profile FA4 varlen attention: vLLM's ``flash_attn_varlen_func(fa_version=4)``.

Builds one varlen batch from ``query_context_pairs`` and makes the call
``FlashAttentionDiffKVImpl.forward`` makes (``flash_attn_diffkv.py``), timing it
with CUPTI as the busy union of every launch: the split scheduler, the forward
kernel and the combine, whichever FA4 issues for the batch.

Cache layout. vLLM's attention cache is ``[pages, page_size, kv_heads,
head_dim_qk + head_dim_v]`` per layer under LBNHC, and the backend reads its
two last-dimension slices as K and V (``split(head_size, dim=-1)``), then
``canonicalize_singleton_dim_strides`` fixes the stride of a size-1 head dim.
The runner allocates the same raw pages and makes the same views. Pages are
shuffled across a pool sized to the batch; physical page 0 is the null page
(vLLM's ``null_block``), which padded rows and the pages wholly before a
sliding window point at, as the sliding-window manager leaves them. The page
table is ``ceil(max_model_len / page_size)`` wide, like vLLM's.

Step shape. With ``cuda_graph`` a decode-only step replays the FULL CUDA
graph captured at ``padded_requests``: FA4's
host-side split choice was made then, from ``max_seqlen_q = 1``,
``max_seqlen_k = max_model_len`` and the padded batch
(``gpu_model_runner._build_attention_metadata`` with ``for_cudagraph_capture``).
The runner reproduces that call: the padding requests have empty queries
(``query_start_loc`` repeats the token count), ``seq_lens`` 0 and the null page.
Every other step runs eagerly over the real requests, longest query and longest
context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

KIND = "fa4_varlen_attention"
BACKEND = "vllm_fa4"
_MAX_REQUESTS = 4096
_MAX_TOKENS = 32768
_MAX_HEAD_DIM = 256
_SUPPORTED_PAGE_SIZES = (16, 32, 64, 128)
_KV_LAYOUTS = ("lbnhc",)
_NULL_PAGE = 0
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
    head_dim_qk: int
    head_dim_v: int
    window: int
    sinks: bool
    page_size: int
    max_model_len: int
    cuda_graph: bool
    pairs: tuple[tuple[int, int], ...]
    padded_requests: int

    @property
    def num_tokens(self) -> int:
        return sum(query for query, _ in self.pairs)

    @property
    def replays_graph(self) -> bool:
        """A decode-only step the FULL CUDA graph covers (Rust `replays_graph`)."""
        return self.cuda_graph and all(q == 1 for q, _ in self.pairs)

    @property
    def launched_requests(self) -> int:
        """Requests in the call: the capture size on a graph step, else the real batch."""
        if self.replays_graph:
            return max(self.padded_requests, len(self.pairs))
        return len(self.pairs)

    @property
    def max_seqlen_q(self) -> int:
        return 1 if self.replays_graph else max(query for query, _ in self.pairs)

    @property
    def max_seqlen_k(self) -> int:
        if self.replays_graph:
            return self.max_model_len
        return max(context for _, context in self.pairs)

    @property
    def window_size(self) -> list[int]:
        # FlashAttentionImpl: sliding_window w -> (w - 1, 0); none -> (-1, -1).
        return [self.window - 1, 0] if self.window else [-1, -1]

    def first_visible_key(self, query: int, context: int) -> int:
        """First key any query of the request sees."""
        if not self.window:
            return 0
        return max(0, context - query - (self.window - 1))


def _positive(name: str, value: object) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _validate_args(
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim_qk: int,
    head_dim_v: int,
    window: int,
    sinks: bool,
    page_size: int,
    kv_layout: str,
    q_dtype: object,
    kv_dtype: object,
    max_model_len: int,
    cuda_graph: bool,
    query_context_pairs: tuple[tuple[int, int], ...],
    padded_requests: int = 0,
) -> _Shape:
    for name, value in (
        ("num_qo_heads", num_qo_heads),
        ("num_kv_heads", num_kv_heads),
        ("head_dim_qk", head_dim_qk),
        ("head_dim_v", head_dim_v),
        ("max_model_len", max_model_len),
    ):
        _positive(name, value)
    for name, value in (
        ("window", window),
        ("padded_requests", padded_requests),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    for name, flag in (("sinks", sinks), ("cuda_graph", cuda_graph)):
        if type(flag) is not bool:
            raise TypeError(f"{name} must be a bool, got {flag!r}")
    if num_qo_heads % num_kv_heads:
        raise ValueError("num_qo_heads must be a multiple of num_kv_heads")
    for name, dim in (("head_dim_qk", head_dim_qk), ("head_dim_v", head_dim_v)):
        if dim % 16 or dim > _MAX_HEAD_DIM:
            raise ProfilerNotImplemented(
                f"{KIND} supports {name} a multiple of 16 up to {_MAX_HEAD_DIM}"
            )
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
    if padded_requests > _MAX_REQUESTS:
        raise ProfilerNotImplemented(f"{KIND} supports padded_requests <= {_MAX_REQUESTS}")
    return _Shape(
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_v,
        window,
        sinks,
        page_size,
        max_model_len,
        cuda_graph,
        tuple(query_context_pairs),
        padded_requests,
    )


@dataclass
class _Workload:
    q: Any  # [T, H, Dqk] bf16
    raw_cache: Any  # [pages, page_size, Hkv, Dqk + Dv] bf16 (LBNHC)
    key_cache: Any  # [pages, page_size, Hkv, Dqk] view
    value_cache: Any  # [pages, page_size, Hkv, Dv] view
    block_table: Any  # [B, ceil(max_model_len / page)] int32 (B = launched requests)
    seqused_k: Any  # [B] int32
    cu_seqlens_q: Any  # [B + 1] int32
    sinks: Any  # [H] bf16 or None
    out: Any  # [T, H, Dv] bf16


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _build_workload(torch: Any, shape: _Shape, device: Any) -> _Workload:
    generator = torch.Generator().manual_seed(0xFA4)
    page = shape.page_size
    width = _cdiv(shape.max_model_len, page)
    # Only the pages a request can read hold data; the rest stay the null page.
    spans = [
        (shape.first_visible_key(query, context) // page, _cdiv(context, page))
        for query, context in shape.pairs
    ]
    num_pages = sum(end - start for start, end in spans) + 1
    physical = (torch.randperm(num_pages - 1, generator=generator) + 1).to(torch.int32)
    launched = shape.launched_requests
    block_table = torch.full((launched, width), _NULL_PAGE, dtype=torch.int32)
    cursor = 0
    for request, (start, end) in enumerate(spans):
        block_table[request, start:end] = physical[cursor : cursor + end - start]
        cursor += end - start

    hkv, dqk, dv = shape.num_kv_heads, shape.head_dim_qk, shape.head_dim_v
    raw_cache = torch.empty((num_pages, page, hkv, dqk + dv), dtype=torch.bfloat16, device=device)
    raw_cache.normal_()
    key_cache, value_cache = raw_cache.split(dqk, dim=-1)

    tokens = shape.num_tokens
    queries = [query for query, _ in shape.pairs] + [0] * (launched - len(shape.pairs))
    contexts = [context for _, context in shape.pairs] + [0] * (launched - len(shape.pairs))
    cu = torch.zeros(launched + 1, dtype=torch.int32)
    cu[1:] = torch.cumsum(torch.tensor(queries), 0)
    q = torch.randn((tokens, shape.num_qo_heads, dqk), generator=generator)
    sinks = None
    if shape.sinks:
        sinks = torch.randn(shape.num_qo_heads, generator=generator).to(
            device=device, dtype=torch.bfloat16
        )
    return _Workload(
        q=q.to(device=device, dtype=torch.bfloat16),
        raw_cache=raw_cache,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table.to(device),
        seqused_k=torch.tensor(contexts, dtype=torch.int32, device=device),
        cu_seqlens_q=cu.to(device),
        sinks=sinks,
        out=torch.empty((tokens, shape.num_qo_heads, dv), dtype=torch.bfloat16, device=device),
    )


def visible_keys(shape: _Shape) -> int:
    """Σ over query tokens of the keys each one attends to."""
    total = 0
    for query, context in shape.pairs:
        for position in range(context - query, context):
            seen = position + 1
            total += min(seen, shape.window) if shape.window else seen
    return total


def logical_bytes(shape: _Shape) -> float:
    """Q and O once, plus each request's visible K/V tokens once."""
    elem = 2
    per_token = shape.num_qo_heads * (shape.head_dim_qk + shape.head_dim_v) * elem
    kv_tokens = sum(
        context - shape.first_visible_key(query, context) for query, context in shape.pairs
    )
    kv_bytes = kv_tokens * shape.num_kv_heads * (shape.head_dim_qk + shape.head_dim_v) * elem
    return float(shape.num_tokens * per_token + kv_bytes)


def _metrics(shape: _Shape, time_ms: float, energy_j: float) -> ComputeMetrics:
    seconds = time_ms / 1000.0
    flops = 2.0 * shape.num_qo_heads * (shape.head_dim_qk + shape.head_dim_v) * visible_keys(shape)
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
    from profiling.runners.attention.fa4_varlen_attention_reference import (
        gather_request_kv,
        varlen_attention_reference,
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
        expected = varlen_attention_reference(
            work.q[rows],
            context - query + local_rows,
            keys,
            values,
            softmax_scale=shape.head_dim_qk**-0.5,
            window=shape.window,
            sinks=work.sinks,
        )
        got = work.out[rows].float()
        if not torch.isfinite(got).all():
            raise KernelLaunchFailed(f"{KIND}:{BACKEND} produced non-finite output")
        max_abs = max(max_abs, float((got - expected).abs().max().item()))
        rel = (got - expected).norm(dim=-1) / expected.norm(dim=-1).clamp(min=1e-6)
        max_rel = max(max_rel, float(rel.max().item()))
        rows_checked += int(rows.numel())
    report = {"rows_checked": rows_checked, "max_abs": max_abs, "max_rel_l2": max_rel}
    if max_abs > _ABS_TOL or max_rel > _REL_L2_TOL:
        raise KernelLaunchFailed(f"{KIND}:{BACKEND} mismatches the PyTorch reference: {report}")
    return report


def build_vllm_fa4(torch: Any, shape: _Shape, device: Any) -> tuple[_Workload, Any]:
    """Workload plus a closure making the backend's flash_attn_varlen_func call."""
    try:
        from vllm.utils.torch_utils import canonicalize_singleton_dim_strides
        from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func
    except ImportError as exc:
        raise ProfilerNotImplemented(
            f"{KIND}:{BACKEND} requires vLLM's vllm_flash_attn (vllm_env)"
        ) from exc

    work = _build_workload(torch, shape, device)
    # The backend's fix for a size-1 head dim's stride (TMA alignment).
    work.key_cache = canonicalize_singleton_dim_strides(work.key_cache)
    work.value_cache = canonicalize_singleton_dim_strides(work.value_cache)
    max_seqlen_q, max_seqlen_k = shape.max_seqlen_q, shape.max_seqlen_k
    window_size = shape.window_size
    softmax_scale = shape.head_dim_qk**-0.5

    def run() -> Any:
        # FlashAttentionDiffKVImpl.forward's call; on bf16 FA4 drops the
        # descales and vLLM's max_num_splits is 0.
        return flash_attn_varlen_func(
            q=work.q,
            k=work.key_cache,
            v=work.value_cache,
            out=work.out,
            cu_seqlens_q=work.cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=work.seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=window_size,
            block_table=work.block_table,
            softcap=0.0,
            fa_version=4,
            num_splits=0,
            s_aux=work.sinks,
        )

    return work, run


def call_launches(run: Any) -> list[str]:
    """Kernel names one call launches, in order (an untimed CUPTI capture)."""
    from profiling.profilers.cupti_kernel_profiler import profile_kernel

    summary = profile_kernel(
        run, num_warmup=0, num_iter=1, clear_l2_before_run=False, clear_l2_between_launches=False
    )
    return list(summary.matched_kernel_names)


def profile_fa4_varlen_attention_vllm_fa4(**kwargs: Any) -> ComputeMetrics:
    """Time vLLM's FA4 flash_attn_varlen_func over one varlen batch."""
    shape = _validate_args(**kwargs)
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ProfilerNotImplemented("PyTorch is unavailable") from exc
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        work, run = build_vllm_fa4(torch, shape, device)
        run()  # compile + first launch, outside timing
        torch.cuda.synchronize(device)
        check_against_reference(torch, shape, work)
        time_ms = Timer.cupti(run, kernel_name=None, interval_union=True)
        energy_j = Energy.perf(run, per_iter_time_ms=time_ms)
        return _metrics(shape, time_ms, energy_j)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{KIND}:{BACKEND} ran out of CUDA memory") from exc
    except (KernelLaunchFailed, ProfilerNotImplemented, TypeError, ValueError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{KIND}:{BACKEND} failed: {exc}") from exc
