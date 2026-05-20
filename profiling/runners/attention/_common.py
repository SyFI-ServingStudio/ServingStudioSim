"""Shared mechanics for FlashInfer attention runners (L1a).

This module owns only the op-agnostic, *caller-free* pieces, so each per-kind
runner module stays self-contained and the kernel *caller* (which wrapper,
``plan()``, the ``run()`` closure, ``causal``) is visible in those files rather
than buried here:

- dtype / backend-name helpers (incl. fp8),
- input construction (tensor alloc + CSR indptr + per-head fp8 quant),
- ``measure(...)`` — the do_bench + energy + ComputeMetrics boilerplate.

prefill + rect share one caller in ``flashinfer_attn_prefill_and_rect.py`` (they differ only by
``causal``); decode keeps its own (different wrapper) in ``flashinfer_decode.py``.
Mirrors the measurement of
``ref/profile/attention/flashinfer_profiler.py`` (Timer.do_bench, FlashInfer-style
``attention_flops``), minus all DB / sweep / CLI machinery (L1b owns those).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.metrics import ComputeMetrics

# FlashInfer recommended workspace (ref DEFAULT_WORKSPACE_SIZE_MB = 2048).
WORKSPACE_BYTES = 2048 * 1024 * 1024
# e4m3fn representable range, used by per-head symmetric quant.
_FP8_MIN, _FP8_MAX = -448.0, 448.0


def to_torch_dtype(dt: DType | str) -> Any:
    """DType -> torch dtype, including fp8 (DType.torch() rejects fp8)."""
    import torch

    dt = DType.from_value(dt)
    if dt is DType.FP8_E4M3:
        return torch.float8_e4m3fn
    if dt is DType.FP8_E5M2:
        return torch.float8_e5m2
    return dt.torch()


def flashinfer_backend_name(backend: str) -> str:
    """MLSim backend string -> FlashInfer wrapper ``backend`` kwarg."""
    return "trtllm-gen" if backend == "trt" else backend


def is_fp8(dt: DType | str) -> bool:
    return DType.from_value(dt) in (DType.FP8_E4M3, DType.FP8_E5M2)


def randn(*size: int, dtype: Any, device: str = "cuda") -> Any:
    """torch.randn that tolerates fp8 (generate in fp16, then cast)."""
    import torch

    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return torch.randn(*size, dtype=torch.float16, device=device).to(dtype)
    return torch.randn(*size, dtype=dtype, device=device)


def per_head_symmetric_quant(x: Any) -> tuple[Any, Any]:
    """Per-head symmetric fp8_e4m3 quant. Mirrors the ref helper.

    ``x`` is (tokens, heads, head_dim) or (pages, page_size, heads, head_dim);
    returns (quantized fp8 tensor, per-head fp32 scales).
    """
    import torch

    if x.dim() == 3:
        x_max = x.abs().amax(dim=(0, 2)).to(torch.float32)
        scale = torch.clamp(x_max / _FP8_MAX, min=1e-6)
        scale_b = scale.view(1, -1, 1)
    elif x.dim() == 4:
        x_max = x.abs().amax(dim=(0, 1, 3)).to(torch.float32)
        scale = torch.clamp(x_max / _FP8_MAX, min=1e-6)
        scale_b = scale.view(1, 1, -1, 1)
    else:
        raise ValueError(f"unsupported tensor shape for quant: {tuple(x.shape)}")
    q = torch.clamp(x / scale_b, min=_FP8_MIN, max=_FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale


def make_workspace() -> Any:
    import torch

    return torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")


def attention_flops(
    *,
    q_len: int,
    kv_len: int,
    num_qo_heads: int,
    head_dim: int,
    causal: bool,
    batch_size: int = 1,
) -> int:
    """FlashInfer-style attention FLOPs (symmetric head_dim_qk == head_dim_vo).

    Causal uses the triangular/append factor ``q_len * (2*kv_len - q_len)``;
    non-causal uses the dense rectangle ``2 * q_len * kv_len``. Each of QK and PV
    contributes ``head_dim``, hence the ``2 * head_dim``.
    """
    if min(q_len, kv_len, num_qo_heads, head_dim, batch_size) <= 0:
        return 0
    work = q_len * (2 * kv_len - q_len) if causal else 2 * q_len * kv_len
    return batch_size * work * num_qo_heads * (2 * head_dim)


@dataclass(frozen=True)
class RaggedInputs:
    """Tensors + CSR indptr for a single-request ragged attention call.

    ``scales`` is None for non-fp8; for fp8 it is ``(s_q, s_k, s_v)`` per-head
    scales the caller forwards to ``wrapper.run``.
    """

    q: Any
    k: Any
    v: Any
    qo_indptr: Any
    kv_indptr: Any
    scales: tuple[Any, Any, Any] | None
    bytes_accessed: int


def build_ragged_inputs(
    *,
    q_len: int,
    kv_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_dtype: DType | str,
    kv_dtype: DType | str,
    o_dtype: DType | str,
) -> RaggedInputs:
    """Allocate q/k/v + CSR indptr for one ragged request (batch_size = 1).

    Symmetric head dims. For fp8 the q/k/v are generated in fp16 then per-head
    quantized; the caller passes the returned scales to ``run`` and sets the
    fp8 ``*_data_type`` plan kwargs.
    """
    import torch

    uses_fp8 = is_fp8(q_dtype) or is_fp8(kv_dtype)
    qo_indptr = torch.tensor([0, q_len], dtype=torch.int32, device="cuda")
    kv_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device="cuda")

    if uses_fp8:
        if not (is_fp8(q_dtype) and is_fp8(kv_dtype)) or is_fp8(o_dtype):
            raise ValueError(
                "ragged fp8 attention requires q_dtype == kv_dtype == fp8 and a "
                "non-fp8 o_dtype"
            )
        q_base = torch.randn(q_len, num_qo_heads, head_dim, dtype=torch.float16, device="cuda")
        k_base = torch.randn(kv_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda")
        v_base = torch.randn(kv_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda")
        q, s_q = per_head_symmetric_quant(q_base)
        k, s_k = per_head_symmetric_quant(k_base)
        v, s_v = per_head_symmetric_quant(v_base)
        scales: tuple[Any, Any, Any] | None = (s_q, s_k, s_v)
    else:
        q = randn(q_len, num_qo_heads, head_dim, dtype=to_torch_dtype(q_dtype))
        k = randn(kv_len, num_kv_heads, head_dim, dtype=to_torch_dtype(kv_dtype))
        v = randn(kv_len, num_kv_heads, head_dim, dtype=to_torch_dtype(kv_dtype))
        scales = None

    o_elem = torch.tensor([], dtype=to_torch_dtype(o_dtype)).element_size()
    bytes_accessed = int(
        q.numel() * q.element_size()
        + k.numel() * k.element_size()
        + v.numel() * v.element_size()
        + q_len * num_qo_heads * head_dim * o_elem
    )
    return RaggedInputs(q, k, v, qo_indptr, kv_indptr, scales, bytes_accessed)


@dataclass(frozen=True)
class PagedDecodeInputs:
    """Paged-KV tensors + CSR page tables for a batched single-token decode.

    ``q`` is one query token per request: ``(batch_size, num_qo_heads, head_dim)``.
    ``k_cache``/``v_cache`` are paged: ``(total_pages, page_size, num_kv_heads,
    head_dim)``. ``scales`` is None for non-fp8; for fp8-KV it is the scalar
    ``(k_scale, v_scale)`` (per-head mean) the caller forwards to ``wrapper.run``.
    """

    q: Any
    k_cache: Any
    v_cache: Any
    kv_indptr: Any
    kv_indices: Any
    kv_last_page_len: Any
    scales: tuple[float, float] | None
    bytes_accessed: int


def build_paged_decode_inputs(
    *,
    batch_size: int,
    seq_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_dtype: DType | str,
    kv_dtype: DType | str,
    o_dtype: DType | str,
    page_size: int = 16,
) -> PagedDecodeInputs:
    """Allocate paged KV + CSR page tables for ``batch_size`` decode requests.

    Each request holds ``seq_len`` cached tokens (the per-request mean kv length)
    and emits one query token. Mirrors the ref decode setup (``PAGE_SIZE=16``).
    For fp8-KV the cache is generated in fp16 then per-head quantized; the caller
    forwards the returned scalar scales to ``run`` and sets ``kv_data_type``.
    """
    import torch

    uses_fp8 = is_fp8(kv_dtype)
    if uses_fp8 and is_fp8(o_dtype):
        raise ValueError("decode fp8 requires a non-fp8 o_dtype")

    num_pages_per_seq = (seq_len + page_size - 1) // page_size
    total_pages = batch_size * num_pages_per_seq
    kv_base_dtype = torch.float16 if uses_fp8 else to_torch_dtype(kv_dtype)

    k_base = torch.randn(
        total_pages, page_size, num_kv_heads, head_dim, dtype=kv_base_dtype, device="cuda"
    )
    v_base = torch.randn(
        total_pages, page_size, num_kv_heads, head_dim, dtype=kv_base_dtype, device="cuda"
    )
    q = randn(batch_size, num_qo_heads, head_dim, dtype=to_torch_dtype(q_dtype))

    kv_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * num_pages_per_seq
    )
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device="cuda")
    last_page_len = seq_len % page_size or page_size
    kv_last_page_len = torch.full((batch_size,), last_page_len, dtype=torch.int32, device="cuda")

    if uses_fp8:
        k_cache, s_k = per_head_symmetric_quant(k_base)
        v_cache, s_v = per_head_symmetric_quant(v_base)
        scales: tuple[float, float] | None = (float(s_k.mean().item()), float(s_v.mean().item()))
    else:
        k_cache, v_cache = k_base, v_base
        scales = None

    o_elem = torch.tensor([], dtype=to_torch_dtype(o_dtype)).element_size()
    bytes_accessed = int(
        q.numel() * q.element_size()
        + k_cache.numel() * k_cache.element_size()
        + v_cache.numel() * v_cache.element_size()
        + batch_size * num_qo_heads * head_dim * o_elem
    )
    return PagedDecodeInputs(
        q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len, scales, bytes_accessed
    )


def measure(
    benchmark_fn: Callable[[], object],
    *,
    flops: int,
    bytes_accessed: int,
    warmup: int = 100,
    rep: int = 1000,
) -> ComputeMetrics:
    """do_bench the closure, sample energy, assemble ComputeMetrics.

    Timer.do_bench matches the ref's measurement for all attention paths.
    """
    time_ms = Timer.do_bench(benchmark_fn, warmup=warmup, rep=rep)
    energy_j = Energy.perf(
        benchmark_fn,
        warmup=min(warmup, 5),
        min_duration_ms=1000,
        per_iter_time_ms=time_ms,
    )
    tflops = (flops / (time_ms / 1000.0)) / 1e12 if time_ms > 0 and flops > 0 else 0.0
    bandwidth_gbps = (bytes_accessed / (time_ms / 1000.0)) / 1e9 if time_ms > 0 else 0.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=float(tflops),
        memory_bandwidth_gbps=float(bandwidth_gbps),
        energy_j=float(energy_j),
    )
