"""Profile vLLM's FLASH_ATTN backend (FlashAttention 3) paged batch attention on Hopper.

On SM90 vLLM's attention backend priority starts with ``FLASH_ATTN``; for Qwen3.8's
16 gated-GQA layers it runs FA3 (``FlashAttnFwdSm90`` in the 2026-11-03 Hopper
alignment captures) over the FP8 cache. ``FlashAttentionImpl.forward``
(``vllm/v1/attention/backends/flash_attn.py``) makes one
``flash_attn_varlen_func(..., fa_version=3)`` call per step: FP8 query (quantized
in the attention layer), the FP8 K/V caches as separate ``[blocks, block_size,
kv_heads, head_dim]`` views, ``cu_seqlens_q`` / ``seqused_k`` / ``block_table``,
per-(sequence, kv-head) descales, causal, and AOT ``scheduler_metadata``
(``get_scheduler_metadata``) when CUDA graphs are on. This runner makes that call for
a decode-only batch (``paged_attn_batch_decode``) and a prefill-only batch
(``paged_attn_batch_prefill``); a mixed step is the two rows summed.

Block size: the hybrid Qwen3.8 deployment logs "Setting attention block size to
1568 tokens" (attention page >= mamba page), so pages hold ``_PAGE_SIZE`` = 1568
tokens. Descales are 1, so FP8 values equal the dequantized values the reference
reads. Batch shapes, limits and metrics are the trtllm-gen runner's
(``paged_attn_batch_trtllm_gen``), so rows of the two backends are comparable.
Runs in ``vllm_env``.
"""

from __future__ import annotations

from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention import paged_attn_batch_trtllm_gen as trt
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

BACKEND = "vllm_fa3"
_PAGE_SIZE = 1568
_REL_L2_TOL = 0.06
_CHECK_MAX_REQUESTS = 8
_CHECK_ROWS = 4


def _import_fa(kind: str) -> tuple[Any, Any]:
    try:
        from vllm.vllm_flash_attn import flash_attn_varlen_func, get_scheduler_metadata
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{kind}:{BACKEND} requires vLLM's flash-attn (vllm_env)") from exc
    return flash_attn_varlen_func, get_scheduler_metadata


def _build(torch: Any, geometry: Any, lengths: list[int], appends: list[int], device: Any) -> dict:
    """Paged FP8 K/V caches (NHD), a shuffled block table, FP8 queries for `appends` tokens per request."""
    g = torch.Generator(device="cpu").manual_seed(0)
    hkv, hq, d = geometry.num_kv_heads, geometry.num_qo_heads, geometry.head_dim
    pages = [trt._cdiv(length, _PAGE_SIZE) for length in lengths]
    num_blocks = sum(pages) + 1
    physical = (torch.randperm(num_blocks - 1, generator=g) + 1).to(torch.int32)
    block_table = torch.zeros((len(lengths), max(1, max(pages))), dtype=torch.int32)
    at = 0
    for r, count in enumerate(pages):
        block_table[r, :count] = physical[at:at + count]; at += count
    fp8 = torch.float8_e4m3fn
    gd = torch.Generator(device=device).manual_seed(1)

    def fill(shape: tuple) -> Any:
        """FP8 N(0, 0.25) values drawn on the GPU, a few blocks at a time (a 32 GiB cache is never
        materialized in BF16)."""
        out = torch.empty(shape, dtype=fp8, device=device)
        step = max(1, (256 << 20) // max(1, out[0].numel()))
        for at in range(0, shape[0], step):
            part = out[at:at + step]
            part.copy_(torch.randn(part.shape, generator=gd, device=device, dtype=torch.bfloat16) * 0.5)
        return out

    k_cache = fill((num_blocks, _PAGE_SIZE, hkv, d))
    v_cache = fill((num_blocks, _PAGE_SIZE, hkv, d))
    tokens = sum(appends)
    q = fill((tokens, hq, d))
    cu_q = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    cu_q[1:] = torch.cumsum(torch.tensor(appends, dtype=torch.int32), 0)
    return {
        "q": q, "k": k_cache, "v": v_cache, "out": torch.empty((tokens, hq, d), dtype=torch.bfloat16, device=device),
        "block_table": block_table.to(device), "seqused_k": torch.tensor(lengths, dtype=torch.int32, device=device),
        "cu_q": cu_q.to(device), "lengths": lengths, "appends": appends, "max_q": max(appends), "max_k": max(lengths),
    }


def _closure(torch: Any, kind: str, geometry: Any, w: dict) -> Any:
    fa, sched = _import_fa(kind)
    hq, hkv, d = geometry.num_qo_heads, geometry.num_kv_heads, geometry.head_dim
    n = len(w["lengths"])
    ones = torch.ones((n, hkv), dtype=torch.float32, device=w["q"].device)
    meta = sched(
        batch_size=n, max_seqlen_q=w["max_q"], max_seqlen_k=w["max_k"], num_heads_q=hq, num_heads_kv=hkv,
        headdim=d, cache_seqlens=w["seqused_k"], qkv_dtype=torch.float8_e4m3fn, cu_seqlens_q=w["cu_q"],
        page_size=_PAGE_SIZE, causal=True, window_size=(-1, -1), num_splits=0,
    )

    def run() -> Any:
        return fa(
            q=w["q"], k=w["k"], v=w["v"], out=w["out"], cu_seqlens_q=w["cu_q"], max_seqlen_q=w["max_q"],
            seqused_k=w["seqused_k"], max_seqlen_k=w["max_k"], softmax_scale=d**-0.5, causal=True,
            window_size=[-1, -1], block_table=w["block_table"], scheduler_metadata=meta, fa_version=3,
            q_descale=ones, k_descale=ones, v_descale=ones, num_splits=0,
        )

    return run


def _check(torch: Any, kind: str, geometry: Any, w: dict) -> None:
    """fp32 reference for a few query rows of a few requests."""
    hq, hkv, d = geometry.num_qo_heads, geometry.num_kv_heads, geometry.head_dim
    group = hq // hkv
    errs = []
    starts = [0]
    for a in w["appends"]:
        starts.append(starts[-1] + a)
    for r in trt._spread(torch, len(w["lengths"]), _CHECK_MAX_REQUESTS):
        length, append = w["lengths"][r], w["appends"][r]
        blocks = w["block_table"][r, : trt._cdiv(length, _PAGE_SIZE)].long()
        k = w["k"][blocks].reshape(-1, hkv, d)[:length].float()
        v = w["v"][blocks].reshape(-1, hkv, d)[:length].float()
        for i in trt._spread(torch, append, _CHECK_ROWS):
            pos = length - append + i            # absolute position of this query token
            # Query heads grouped per kv head, so the 1M-token K/V is never repeated.
            q = w["q"][starts[r] + i].float().view(hkv, group, d)
            s = torch.einsum("hgd,thd->hgt", q, k[: pos + 1]) * d**-0.5
            ref = torch.einsum("hgt,thd->hgd", s.softmax(-1), v[: pos + 1]).reshape(hq, d)
            got = w["out"][starts[r] + i].float()
            errs.append(((got - ref).norm() / ref.norm().clamp_min(1e-6)).item())
    if errs and max(errs) > _REL_L2_TOL:
        raise KernelLaunchFailed(f"{kind}:{BACKEND} output mismatch: max relative L2 {max(errs):.3e} > {_REL_L2_TOL}")


def _time(kind: str, geometry: Any, lengths: list[int], appends: list[int]) -> tuple[float, float]:
    torch = trt._torch(kind)
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        w = _build(torch, geometry, lengths, appends, device)
        run = _closure(torch, kind, geometry, w)
        run()
        torch.cuda.synchronize(device)
        _check(torch, kind, geometry, w)
        time_ms = Timer.cupti(run, kernel_name=None, interval_union=True)
        energy_j = Energy.perf(run, per_iter_time_ms=time_ms)
        return time_ms, energy_j
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{kind}:{BACKEND} ran out of CUDA memory") from exc
    except (KernelLaunchFailed, ProfilerNotImplemented, TypeError, ValueError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{kind}:{BACKEND} failed: {exc}") from exc


def profile_paged_attn_batch_decode(**kwargs: Any) -> ComputeMetrics:
    shape = trt.validate_decode_args(**kwargs)
    time_ms, energy_j = _time(trt._DECODE, shape.geometry, shape.lengths, [1] * shape.batch_size)
    return trt._metrics(shape.geometry, shape.total_kv_len, shape.batch_size,
                        trt._kv_bytes(shape.geometry, shape.lengths), time_ms, energy_j)


def profile_paged_attn_batch_prefill(**kwargs: Any) -> ComputeMetrics:
    shape = trt.validate_prefill_args(**kwargs)
    time_ms, energy_j = _time(trt._PREFILL, shape.geometry, shape.lengths, shape.appends)
    return trt._metrics(shape.geometry, shape.interactions, shape.query_tokens,
                        trt._kv_bytes(shape.geometry, shape.lengths), time_ms, energy_j)


__all__ = ["profile_paged_attn_batch_decode", "profile_paged_attn_batch_prefill"]
