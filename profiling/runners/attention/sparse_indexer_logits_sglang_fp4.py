"""DeepSeek-V4.1 indexer logits as SGLang v0.5.23 computes them on SM12x: the
``sglang_triton_fp4`` backend of ``sparse_indexer_logits_decode`` and the
``sglang_torch_fp4`` backend of ``sparse_indexer_logits_prefill``.

The V4.1 index keys live in an FP4 pool (``DeepSeekV4IndexerPool``): per slot 64
payload bytes (128 e2m1 values) and 4 UE8M0 scales (one per 32 values), in
64-slot pages laid out ``[64 * 64 payload | 64 * 4 scales]`` (``DSV41_INDEX_PAGE_SIZE``
is 64 unless DeepGEMM's paged sparse MQA logits exist, which they do not on SM12x;
``srt/mem_cache/deepseek_v4_memory_pool.py:532``). With ``is_sm100_or_newer()``
false on SM12x (the SM120 indexer fix), ``make_candidate_indexer`` returns the
dense-blocks backend (``srt/layers/attention/dsv4/v41_indexer/dense_blocks.py``):

- **decode** -- ``decode_scores`` (``v41_indexer/scoring.py:477``, non-paged branch
  because ``paged = is_sm90``) calls ``fp4_index_logits_decode``
  (``kernels/ops/attention/dsv4/fp4_indexer.py:490``): one Triton launch,
  ``_fp4_index_logits_kernel``, over a ``[rows, lmax]`` grid where ``lmax`` is the
  graph-stable capacity and tiles past each row's visible length exit early.
- **prefill** -- ``prefill_requests`` (``scoring.py:356``): per request, dequantize
  the visible index K to bf16 (``get_index_k_dequant``, pool.py:769), pad it to a
  multiple of 64 rows once it reaches 1024, then score query-row chunks sized to a
  1 GiB budget with ``einsum('bhd,nd->bhn')``, ReLU, head-weight multiply and head
  sum (``DeepseekV41Indexer.scores``, ``dsv41_sparse.py:285``) and mask past each
  row's length.

The top-k that follows is the separate ``logits_topk`` kind. The query projection
(``wq_b``, block-FP8) and the RoPE / head-weight GEMMs are separate kinds too. Runs
in ``sglang_env``.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

HEAD_DIM = 128
PAGE = 64  # DSV41_INDEX_PAGE_SIZE on SM12x
SLOT_BYTES = 64 + 4  # e2m1 payload + 4 ue8m0 scales
REL_TOL = 3e-2
_CHECK_ROWS = 4
_SCORE_BUDGET_BYTES = 1 << 30  # scoring.py:329 _TORCH_SCORE_BUDGET_BYTES


def _random_table(torch: Any, num_pages: int, g: Any) -> Any:
    table = torch.randint(0, 256, (num_pages, PAGE * SLOT_BYTES), device="cuda", dtype=torch.uint8, generator=g)
    # UE8M0 exponents near 1.0 (127 = 2^0) keep the scores finite.
    scales = torch.randint(120, 128, (num_pages, PAGE * 4), device="cuda", dtype=torch.uint8, generator=g)
    table[:, PAGE * 64 :] = scales
    return table


def _dequant(torch: Any, table: Any, slots: Any) -> Any:
    """bf16 [n, 128] index K at ``slots``, exactly as get_index_k_dequant reads it."""
    from sglang.srt.layers.quantization.fp8 import DSV4_DEQUANT_FP4_TABLE

    slots = slots.to(torch.int64)
    page, off = (slots // PAGE).unsqueeze(-1), slots % PAGE
    payload_cols = (off * 64).unsqueeze(-1) + torch.arange(64, device=table.device)
    scale_cols = (PAGE * 64 + off * 4).unsqueeze(-1) + torch.arange(4, device=table.device)
    u = table[page, payload_cols].view(torch.uint8)
    codes = torch.stack([u & 0x0F, (u >> 4) & 0x0F], dim=-1)
    vals = DSV4_DEQUANT_FP4_TABLE.to(table.device)[codes.long()].flatten(1)
    exps = table[page, scale_cols].to(torch.int32) & 0xFF
    scales = torch.exp2(exps.float() - 127).repeat_interleave(32, dim=-1)
    return (vals * scales).to(torch.bfloat16)


def _reference_scores(torch: Any, q: Any, k: Any, w: Any) -> Any:
    s = torch.einsum("bhd,nd->bhn", q.double(), k.double())
    return (s.relu() * w.double().unsqueeze(-1)).sum(dim=1)


def _check(name: str, got: Any, want: Any) -> None:
    import torch

    finite = torch.isfinite(want)
    g, w = got[finite].double(), want[finite]
    err = ((g - w).norm() / w.norm().clamp_min(1e-30)).item()
    if not err <= REL_TOL:
        raise KernelLaunchFailed(f"{name} output mismatch: relative Frobenius error {err:.3e} > {REL_TOL:.0e}")


def _metrics(time_ms: float, energy_j: float, rows: int, keys: int, heads: int, key_bytes: float) -> ComputeMetrics:
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=2 * rows * heads * HEAD_DIM * keys / elapsed_s / 1e12,
        memory_bandwidth_gbps=(key_bytes + 4 * rows * keys) / elapsed_s / 1e9,
        energy_j=float(energy_j),
    )


def _wrap(name: str, exc: Exception) -> Exception:
    if exc.__class__.__name__ == "OutOfMemoryError":
        return OOMError(f"{name} ran out of memory")
    return KernelLaunchFailed(f"{name} failed: {exc}")


def profile_sparse_indexer_logits_decode_sglang_triton_fp4(
    batch_size: int,
    num_keys: int,
    logits_width: int,
    next_n: int,
    num_heads: int,
    head_dim: int,
    dtype: DType | str,
) -> ComputeMetrics:
    name = "sglang_triton_fp4"
    if DType.from_value(dtype) is not DType.MXFP4_E2M1:
        raise ValueError(f"{name}: the V4.1 index keys are mxfp4_e2m1 (e2m1 + ue8m0 per 32), got {dtype!r}")
    if int(head_dim) != HEAD_DIM:
        raise ValueError(f"{name}: index head_dim is {HEAD_DIM}")
    b, n, width, nxt, h = int(batch_size), int(num_keys), int(logits_width), int(next_n), int(num_heads)
    if min(b, nxt, h) < 1 or n < 1 or width < n:
        raise ValueError(f"{name}: need positive sizes and logits_width >= num_keys")
    try:
        import torch
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import fp4_index_logits_decode
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{name} requires the SGLang environment (sglang_env)") from exc
    rows = b * nxt
    try:
        g = torch.Generator(device="cuda").manual_seed(0)
        pages_per_req = -(-width // PAGE)
        table = _random_table(torch, b * pages_per_req + 1, g)
        # Each request's compressed positions on randomly ordered pages of its own.
        perm = torch.stack([torch.randperm(pages_per_req, device="cuda", generator=g) for _ in range(b)])
        pages = perm + (torch.arange(b, device="cuda") * pages_per_req).view(b, 1)
        j = torch.arange(width, device="cuda")
        req_slots = pages[:, j // PAGE] * PAGE + (j % PAGE)  # [b, width]
        slots = req_slots.repeat_interleave(nxt, dim=0).to(torch.int64).contiguous()
        lens = torch.full((rows,), n, device="cuda", dtype=torch.int64)
        q = torch.randn(rows, h, HEAD_DIM, device="cuda", dtype=torch.bfloat16, generator=g) * 0.1
        w = (torch.rand(rows, h, device="cuda", generator=g) * 0.1).to(torch.bfloat16)

        def call() -> Any:
            return fp4_index_logits_decode(q, w, slots, lens, table, PAGE)

        out = call()
        torch.cuda.synchronize()
        k = _dequant(torch, table, slots[0, :n])
        for row in torch.linspace(0, rows - 1, min(rows, _CHECK_ROWS)).round().long().unique().tolist():
            k_row = k if row // nxt == 0 else _dequant(torch, table, slots[row, :n])
            _check(name, out[row, :n], _reference_scores(torch, q[row : row + 1], k_row, w[row : row + 1])[0])
        time_ms = Timer.cupti(call, warmup=3)
        energy_j = Energy.perf(call, per_iter_time_ms=time_ms)
    except (ProfilerNotImplemented, KernelLaunchFailed):
        raise
    except Exception as exc:
        raise _wrap(name, exc) from exc
    return _metrics(time_ms, energy_j, rows, n, h, b * n * SLOT_BYTES)


def profile_sparse_indexer_logits_prefill_sglang_torch_fp4(
    num_queries: int,
    prefix_len: int,
    num_heads: int,
    head_dim: int,
    compress_ratio: int,
    dtype: DType | str,
) -> ComputeMetrics:
    name = "sglang_torch_fp4"
    if DType.from_value(dtype) is not DType.MXFP4_E2M1:
        raise ValueError(f"{name}: the V4.1 index keys are mxfp4_e2m1 (e2m1 + ue8m0 per 32), got {dtype!r}")
    if int(head_dim) != HEAD_DIM:
        raise ValueError(f"{name}: index head_dim is {HEAD_DIM}")
    t, p, h, ratio = int(num_queries), int(prefix_len), int(num_heads), int(compress_ratio)
    if min(t, h, ratio) < 1 or p < 0:
        raise ValueError(f"{name}: need positive num_queries, num_heads and compress_ratio")
    lc = (p + t) // ratio
    if lc < 1:
        raise ValueError(f"{name}: no compressed key is visible yet (prefix_len + num_queries < compress_ratio)")
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{name} requires torch") from exc
    try:
        g = torch.Generator(device="cuda").manual_seed(0)
        num_pages = -(-lc // PAGE) + 1
        table = _random_table(torch, num_pages, g)
        slots = torch.randperm(num_pages * PAGE, device="cuda", generator=g)[:lc].sort().values
        q = torch.randn(t, h, HEAD_DIM, device="cuda", dtype=torch.bfloat16, generator=g) * 0.1
        w = (torch.rand(t, h, device="cuda", generator=g) * 0.1).to(torch.bfloat16)
        pos = torch.arange(p, p + t, device="cuda")
        lens = (pos + 1) // ratio
        j = torch.arange(lc, device="cuda")
        rows_per_chunk = max(1, _SCORE_BUDGET_BYTES // (h * max(lc, 1) * 2))

        def call() -> list:
            index_k = _dequant(torch, table, slots)
            if lc >= 1024:
                width = (lc + 63) // 64 * 64
                if width != lc:
                    index_k = torch.nn.functional.pad(index_k, (0, 0, 0, width - lc))
            chunks = []
            for start in range(0, t, rows_per_chunk):
                rs = slice(start, start + rows_per_chunk)
                s = torch.einsum("bhd,nd->bhn", q[rs], index_k)
                s = (s.relu() * w[rs].unsqueeze(-1)).sum(dim=1).float()[:, :lc]
                chunks.append(s.masked_fill(j[None, :] >= lens[rs, None], -torch.inf))
            return chunks

        out = torch.cat(call(), dim=0)
        torch.cuda.synchronize()
        k = _dequant(torch, table, slots)
        for row in torch.linspace(0, t - 1, min(t, _CHECK_ROWS)).round().long().unique().tolist():
            want = _reference_scores(torch, q[row : row + 1], k, w[row : row + 1])[0]
            want = want.masked_fill(j >= lens[row], float("-inf"))
            _check(name, out[row], want)
        del out
        time_ms = Timer.cupti(call, warmup=2)
        energy_j = Energy.perf(call, per_iter_time_ms=time_ms)
    except (ProfilerNotImplemented, KernelLaunchFailed):
        raise
    except Exception as exc:
        raise _wrap(name, exc) from exc
    return _metrics(time_ms, energy_j, t, lc, h, lc * SLOT_BYTES + 2 * lc * HEAD_DIM)


__all__ = [
    "profile_sparse_indexer_logits_decode_sglang_triton_fp4",
    "profile_sparse_indexer_logits_prefill_sglang_torch_fp4",
]
