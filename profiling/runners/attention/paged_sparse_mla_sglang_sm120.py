"""DeepSeek-V4-family sparse MLA as SGLang runs it on SM12x: the
``sglang_flashinfer_sm120`` backend of ``paged_sparse_mla``.

SGLang v0.5.23 on SM12x selects the V4 KV layout for every DSv4-family pool
(``select_dsv4_kv_layout``, ``srt/mem_cache/deepseek_v4_memory_pool.py:200``: the
V4.1 layouts need SM100/SM103), so both caches hold 584-byte rows (448 fp8 nope,
64 bf16 rope, 7 scales + 1 pad) in 256-token sliding-window pages and
``page_size // compress_ratio``-slot compressed pages. Per layer
``DeepseekV4AttnBackend`` (``srt/layers/attention/deepseek_v4_backend.py:4335``)
calls ``flash_mla_with_kvcache_sm120(q, k_cache=swa, indices=swa_page_indices,
topk_length=swa_topk_lengths, attn_sink, extra_k_cache, extra_indices_in_kvcache,
extra_topk_length)``; with the default ``SGLANG_SM120_FLASHMLA_BACKEND=flashinfer``:

- rows <= 64: ``_flash_mla_flashinfer`` -- ``_page_mark_kernel`` +
  ``_page_split_kernel`` (re-split the touched 256-token SWA pages into 64-token
  pages), FlashInfer ``sparse_mla_decode_dsv4_kernel`` (split-K over 64-key
  blocks) and ``sparse_mla_decode_dsv4_merge_kernel``;
- rows > 64: ``_flash_mla_sm120_prefill`` -- the same page split, then the paged
  prefill kernel (``sparse_mla_prefill_mg_dual_kernel``), on the unpadded heads.

Decode pads the query heads to 64 unless the installed FlashInfer's DSV4 decode
dispatch covers the per-rank head count (``DeepseekV4Attention._kernel_num_heads``,
``srt/models/deepseek_v4.py:1215``); the runner applies the same rule. The caches
are written through SGLang's own ``fused_store_cache`` so the byte layout is the
production one. The reference is SGLang's own PyTorch path
(``_sm120_sparse_decode_fwd``) on a few rows. Runs in ``sglang_env``.
"""

from __future__ import annotations

import os
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

V4_ROW_BYTES = 584  # 448 fp8 nope + 64 bf16 rope + 7 scales + 1 pad
REL_FROBENIUS_TOL = 2e-2
_CHECK_ROWS = 8
_INDEX_ALIGN = 64


def _visible(pos_plus_one: int, ratio: int, cap: int) -> int:
    return min(cap, pos_plus_one // ratio) if ratio > 0 else 0


def _validate(**a: Any) -> None:
    for name in ("num_requests", "rows_per_request", "num_heads", "swa_window", "page_size"):
        if int(a[name]) < 1:
            raise ValueError(f"paged_sparse_mla: {name} must be positive, got {a[name]}")
    if int(a["context_before"]) < 0:
        raise ValueError("paged_sparse_mla: context_before must be >= 0")
    if a["kv_layout"] != "v4":
        raise ValueError(f"sglang_flashinfer_sm120 serves the V4 cache layout on SM12x, got {a['kv_layout']!r}")
    if (int(a["head_dim"]), int(a["value_dim"])) != (512, 512):
        raise ValueError("the V4 cache layout is 448 nope + 64 rope = 512 for keys and values")
    for name in ("swa_index_capacity", "extra_index_capacity"):
        if int(a[name]) % _INDEX_ALIGN:
            raise ValueError(f"paged_sparse_mla: {name} must be a multiple of {_INDEX_ALIGN}")
    if int(a["swa_index_capacity"]) < int(a["swa_window"]):
        raise ValueError("paged_sparse_mla: swa_index_capacity must cover swa_window")
    if int(a["compress_ratio"]) not in (0, 1, 2, 4, 128):
        raise ValueError(f"paged_sparse_mla: compress_ratio {a['compress_ratio']} is not a DSv4 ratio")
    if int(a["page_size"]) % 64:
        raise ValueError("paged_sparse_mla: page_size must be a multiple of 64 (FlashInfer page re-split)")
    if (DType.from_value(a["q_dtype"]), DType.from_value(a["cache_dtype"]), DType.from_value(a["output_dtype"])) != (
        DType.BF16,
        DType.FP8_E4M3,
        DType.BF16,
    ):
        raise ValueError("sglang_flashinfer_sm120 takes bf16 queries, an fp8_e4m3 cache and bf16 output")


def _load_runtime() -> tuple[Any, ...]:
    # flash_mla_sm120 reads the backend choice at import.
    os.environ.setdefault("SGLANG_SM120_FLASHMLA_BACKEND", "flashinfer")
    try:
        import torch
        from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache
        from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
        from sglang.kernels.ops.attention.flash_mla_sm120 import (
            SM120_DECODE_MAX_TOKENS,
            _sm120_sparse_decode_fwd,
            flash_mla_with_kvcache_sm120,
            flashinfer_dsv4_decode_supports_num_heads,
        )
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "sglang_flashinfer_sm120 requires the SGLang environment (sglang_env) with FlashInfer's SM120 sparse MLA"
        ) from exc
    return (
        torch,
        fused_store_cache,
        KVLayout,
        SM120_DECODE_MAX_TOKENS,
        _sm120_sparse_decode_fwd,
        flash_mla_with_kvcache_sm120,
        flashinfer_dsv4_decode_supports_num_heads,
    )


def _write_cache(torch: Any, store: Any, layout: Any, slots: Any, page_size: int, num_pages: int, g: Any) -> Any:
    """A [num_pages, page_bytes] uint8 cache with bf16 rows quantized into ``slots``."""
    page_bytes = layout.page_bytes(page_size)
    cache = torch.zeros(num_pages, page_bytes, dtype=torch.uint8, device="cuda")
    if slots.numel():
        rows = torch.randn(slots.numel(), 512, device="cuda", dtype=torch.bfloat16, generator=g)
        store(input=rows, cache=cache, indices=slots, page_size=page_size, type="flashmla", layout=layout)
    return cache


def profile_paged_sparse_mla_sglang_sm120(
    num_requests: int,
    rows_per_request: int,
    context_before: int,
    num_heads: int,
    head_dim: int,
    value_dim: int,
    swa_window: int,
    swa_index_capacity: int,
    compress_ratio: int,
    extra_index_capacity: int,
    page_size: int,
    kv_layout: str,
    q_dtype: DType | str,
    cache_dtype: DType | str,
    output_dtype: DType | str,
) -> ComputeMetrics:
    _validate(**locals())
    b, r, c0 = int(num_requests), int(rows_per_request), int(context_before)
    ratio, h_real, wnd = int(compress_ratio), int(num_heads), int(swa_window)
    swa_cap, extra_cap, ps = int(swa_index_capacity), int(extra_index_capacity), int(page_size)
    (torch, store, KVLayout, decode_max, reference, attention, decode_supports) = _load_runtime()
    rows = b * r
    try:
        g = torch.Generator(device="cuda").manual_seed(0)
        dev = "cuda"
        layout = KVLayout.V4
        pos1 = torch.arange(c0 + 1, c0 + r + 1, device=dev)  # pos + 1 per row of a request

        # Sliding-window cache: each request keeps its last (window + rows) tokens
        # contiguous from a page boundary, as the paged SWA allocator lays them out.
        span = wnd + r
        pages_per_req = -(-span // ps)
        swa_pages = b * pages_per_req + 1
        swa_base = (torch.arange(b, device=dev) * pages_per_req * ps).view(b, 1)
        swa_slots = (swa_base + torch.arange(span, device=dev).view(1, span)).reshape(-1).to(torch.int64)
        swa_cache = _write_cache(torch, store, layout, swa_slots, ps, swa_pages, g)
        swa_len = torch.clamp(pos1, max=wnd)  # [r]
        # Row i sees window tokens ending at its own position (offset wnd - 1 + i in the span).
        j = torch.arange(swa_cap, device=dev).view(1, swa_cap)
        end = (wnd - 1 + torch.arange(r, device=dev)).view(r, 1)
        local = end - (swa_len.view(r, 1) - 1) + j
        swa_idx_req = torch.where(j < swa_len.view(r, 1), local, torch.full_like(local, -1))
        swa_idx = torch.where(
            swa_idx_req.view(1, r, swa_cap) >= 0,
            swa_idx_req.view(1, r, swa_cap) + swa_base.view(b, 1, 1),
            -1,
        ).reshape(rows, swa_cap).to(torch.int32)
        swa_lens = swa_len.repeat(b).to(torch.int32)

        extra_cache = extra_idx = extra_lens = None
        n_extra_valid = torch.zeros(r, dtype=torch.long, device=dev)
        if ratio > 0:
            eps = ps // ratio if ratio <= ps else 1
            visible_total = (c0 + r) // ratio
            pages_per_req_e = max(1, -(-visible_total // eps))
            extra_pages = b * pages_per_req_e + 1
            n_extra_valid = torch.clamp(pos1 // ratio, max=extra_cap)  # [r]
            vis = (pos1 // ratio).view(r, 1).clamp(min=1)
            # Selected keys spread evenly over each row's visible compressed positions.
            kk = torch.arange(extra_cap, device=dev).view(1, extra_cap)
            spread = (kk * vis) // n_extra_valid.view(r, 1).clamp(min=1)
            base_e = (torch.arange(b, device=dev) * pages_per_req_e * eps).view(b, 1, 1)
            idx = torch.where(kk.view(1, 1, extra_cap) < n_extra_valid.view(1, r, 1), spread.view(1, r, extra_cap) + base_e, -1)
            extra_idx = idx.reshape(rows, extra_cap).to(torch.int32)
            extra_lens = n_extra_valid.repeat(b).clamp(min=1).to(torch.int32)
            touched = extra_idx[extra_idx >= 0].unique().to(torch.int64)
            extra_cache = _write_cache(torch, store, layout, touched, eps, extra_pages, g)

        # Kernel head count, as DeepseekV4Attention._kernel_num_heads picks it.
        if rows > decode_max or decode_supports(h_real, rows):
            h_kernel = h_real
        else:
            h_kernel = 64 if h_real <= 64 else h_real
        q = torch.zeros(rows, 1, h_kernel, 512, device=dev, dtype=torch.bfloat16)
        q[:, :, :h_real] = torch.randn(rows, 1, h_real, 512, device=dev, dtype=torch.bfloat16, generator=g)
        sink = torch.zeros(h_kernel, device=dev, dtype=torch.float32)
        sink[:h_real] = torch.randn(h_real, device=dev, generator=g)
        # The backend's view: the first page_size * row_bytes of each padded page.
        swa_view = swa_cache[:, : ps * V4_ROW_BYTES].view(swa_pages, ps, 1, V4_ROW_BYTES)
        extra_view = None
        if extra_cache is not None:
            eps = ps // ratio if ratio <= ps else 1
            extra_view = extra_cache[:, : eps * V4_ROW_BYTES].view(extra_cache.shape[0], eps, 1, V4_ROW_BYTES)
        scale = 512**-0.5

        def call() -> Any:
            return attention(
                q=q,
                k_cache=swa_view,
                head_dim_v=512,
                softmax_scale=scale,
                indices=swa_idx.view(rows, 1, swa_cap),
                topk_length=swa_lens,
                attn_sink=sink,
                extra_k_cache=extra_view,
                extra_indices_in_kvcache=None if extra_idx is None else extra_idx.view(rows, 1, extra_cap),
                extra_topk_length=extra_lens,
            )[0]

        out = call()
        torch.cuda.synchronize()
        sample = torch.linspace(0, rows - 1, min(rows, _CHECK_ROWS), device=dev).round().long().unique()
        ref, _ = reference(
            q[sample],
            swa_view,
            swa_idx[sample].view(-1, 1, swa_cap),
            swa_lens[sample],
            sink,
            512,
            scale,
            extra_view,
            None if extra_idx is None else extra_idx[sample].view(-1, 1, extra_cap),
            None if extra_lens is None else extra_lens[sample],
        )
        got = out.reshape(rows, 1, h_kernel, 512)[sample][:, :, :h_real].float()
        want = ref[:, :, :h_real].float()
        err = ((got - want).norm() / want.norm().clamp_min(1e-30)).item()
        if not err <= REL_FROBENIUS_TOL:
            raise KernelLaunchFailed(
                f"sglang_flashinfer_sm120 output mismatch: relative Frobenius error {err:.3e} > {REL_FROBENIUS_TOL:.0e}"
            )
        del out, ref
        time_ms = Timer.cupti(call, warmup=3)
        energy_j = Energy.perf(call, per_iter_time_ms=time_ms)
    except (ProfilerNotImplemented, KernelLaunchFailed):
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError("sglang_flashinfer_sm120 ran out of memory") from exc
        raise KernelLaunchFailed(f"sglang_flashinfer_sm120 failed: {exc}") from exc

    keys = b * int((swa_len.sum() + n_extra_valid.sum()).item())
    elapsed_s = time_ms / 1000.0
    bytes_accessed = 2 * rows * h_real * 512 + keys * V4_ROW_BYTES + 2 * rows * h_real * 512
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=2 * h_real * (512 + 512) * keys / elapsed_s / 1e12,
        memory_bandwidth_gbps=bytes_accessed / elapsed_s / 1e9,
        energy_j=float(energy_j),
    )


__all__ = ["profile_paged_sparse_mla_sglang_sm120"]
