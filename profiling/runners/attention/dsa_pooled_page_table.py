"""Runners for the pooled DSA indexer back: top-k pools to the final page table.

Two kinds share one vendored CUDA source (``dsa_indexer_back_vendor/back.cu``,
read-only, see its header) and one operand builder:

- ``dsa_pooled_topk_page_table`` backend ``cuda_row_scan_compact``: the
  source's ``e2a`` entry at 512 threads. One CTA per query row reads the row's
  selected pool ids, gathers one block-table entry per pool and the tail
  tokens' entries, and writes the compacted page table row and its valid count
  (the fork's buffer fill, pool-id cast, seq-len + 1, pool expand, buffer copy
  and index remap in one launch). Bit-exact to the reference.
- ``dsa_pooled_logits_page_table`` backend ``cuda_threshold_filter_radix``: the
  source's ``topk6`` entry at stage 4. One CTA per row picks the top-512 pools
  from the row's indexer logits (threshold from the first batch, filter into
  per-lane buffers, 11-bit radix select on the survivors, exact fallback over
  the global row) and writes the same page table row in its epilogue. The
  selected set equals the reference top-k except on tied scores; the check
  compares the selected values (stage 3 of the same kernel) and the page table
  of that selection.

Operands follow the decode verify layout: ``batch_size`` requests of
``next_n`` rows; row ``j`` of request ``b`` holds ``seq = 4 * context_len -
(next_n - 1 - j) - (b mod 4)`` tokens (clipped at 1), so its logit row has
``seq // kpool`` complete pools and the tails vary. Scores are drawn as the
decode top-k kind draws them (``context_mode``); the page-table kind's pool ids
are the reference top-k of those scores. The block table is a seeded
permutation of pages, ``ceil(max_model_len / block_size)`` entries per
request. Timing is CUPTI with L2 displaced before every launch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

PAGE_TABLE_BACKEND = "dsa_pooled_topk_page_table:cuda_row_scan_compact"
LOGITS_BACKEND = "dsa_pooled_logits_page_table:cuda_threshold_filter_radix"
_VENDOR = Path(__file__).resolve().parent / "dsa_indexer_back_vendor"
_SOURCE = _VENDOR / "back.cu"
_EXT_NAME = "dsa_indexer_back_vendor"
# Compile-time constants of the vendored source (KPOOL, SELK, PT_W; post_row
# hard-codes 64-token blocks).
_KPOOL, _TOP_K, _WIDTH, _BLOCK = 4, 512, 2176, 64
_E2A_THREADS = 512  # the measured variant (lane E jobs 22140/22141)
_TOPK6_STAGE_SELECT, _TOPK6_STAGE_FUSED, _TOPK6_PF = 3, 4, 1
_CONTEXT_MODES = ("uniform_measured", "uniform_random")
_CHECK_ROWS = 64  # rows compared against the per-row torch.topk oracle
_ext: Any = None


def _extension() -> Any:
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load

        _ext = load(
            name=_EXT_NAME,
            sources=[str(_SOURCE)],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "-std=c++20", "-lineinfo"],
            verbose=False,
        )
    return _ext


@dataclass(frozen=True)
class Shape:
    batch_size: int
    context_len: int
    next_n: int
    max_model_len: int
    top_k: int
    kpool: int
    page_table_width: int
    block_size: int

    @property
    def rows(self) -> int:
        return self.batch_size * self.next_n

    @property
    def max_blocks(self) -> int:
        return -(-self.max_model_len // self.block_size)


def validate_shape(
    *,
    batch_size: int,
    context_len: int,
    next_n: int,
    max_model_len: int,
    top_k: int,
    kpool: int,
    page_table_width: int,
    block_size: int,
    index_dtype: str,
) -> Shape:
    for name, value in (("batch_size", batch_size), ("context_len", context_len), ("next_n", next_n),
                        ("max_model_len", max_model_len)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive int, got {value!r}")
    if (top_k, kpool, page_table_width, block_size) != (_TOP_K, _KPOOL, _WIDTH, _BLOCK):
        raise ValueError(
            f"the vendored kernel is compiled for top_k {_TOP_K}, kpool {_KPOOL}, page_table_width "
            f"{_WIDTH} and {_BLOCK}-token blocks"
        )
    if index_dtype != "int32":
        raise ValueError("index_dtype must be int32")
    if kpool * context_len > max_model_len:
        raise ValueError("context_len pools exceed max_model_len tokens")
    if kpool * context_len < next_n + 3:
        raise ValueError("context_len must cover the request's next_n rows")
    return Shape(batch_size, context_len, next_n, max_model_len, top_k, kpool, page_table_width, block_size)


@dataclass
class Operands:
    seq: Any  # [rows] int32 tokens per row
    lengths: Any  # [rows] int32 complete pools per row
    req: Any  # [rows] int32
    block_table: Any  # [batch, max_blocks] int32
    logits: Any  # [rows, row_stride] fp32 view
    pool: Any  # [rows, top_k] int32 reference top-k pool ids (-1 past a short row)
    out: Any  # [rows, width] int32
    valid: Any  # [rows] int32


def build_operands(torch: Any, shape: Shape, *, row_stride: int, context_mode: str, device: Any) -> Operands:
    from profiling.runners.attention.dsa_persistent_topk_decode import _fill_measured_scores

    rows = shape.rows
    b = torch.arange(shape.batch_size, device=device, dtype=torch.int64).repeat_interleave(shape.next_n)
    j = torch.arange(shape.next_n, device=device, dtype=torch.int64).repeat(shape.batch_size)
    seq = (shape.kpool * shape.context_len - (shape.next_n - 1 - j) - (b % 4)).clamp_(min=1)
    lengths = torch.div(seq, shape.kpool, rounding_mode="floor")
    logits = torch.empty((rows, row_stride), dtype=torch.float32, device=device)
    if context_mode == "uniform_measured":
        _fill_measured_scores(torch, logits, device=device)
    else:
        logits.normal_(generator=torch.Generator(device=device).manual_seed(0))
    generator = torch.Generator(device=device).manual_seed(1)
    total = shape.batch_size * shape.max_blocks
    # Page ids below 2^25 keep every slot (page * 64 + offset) inside int32.
    block_table = (torch.randperm(total, device=device, generator=generator) % (2**25)).to(torch.int32)
    block_table = block_table.view(shape.batch_size, shape.max_blocks).contiguous()
    # Reference top-k pools, -1 past short rows (sorted by score so a short row's -inf picks trail).
    pool = torch.full((rows, shape.top_k), -1, dtype=torch.int32, device=device)
    for r0 in range(0, rows, 64):
        r1 = min(rows, r0 + 64)
        span = int(lengths[r0:r1].max())
        k = min(shape.top_k, span)
        if k == 0:
            continue
        chunk = logits[r0:r1, :span].clone()
        col = torch.arange(span, device=device)
        chunk[col[None, :] >= lengths[r0:r1, None]] = float("-inf")
        idx = torch.topk(chunk, k, dim=1, sorted=True).indices.to(torch.int32)
        keep = torch.arange(k, device=device)[None, :] < lengths[r0:r1, None]
        pool[r0:r1, :k] = torch.where(keep, idx, torch.full_like(idx, -1))
        del chunk
    return Operands(
        seq=seq.to(torch.int32).contiguous(),
        lengths=lengths.to(torch.int32).contiguous(),
        req=b.to(torch.int32).contiguous(),
        block_table=block_table,
        logits=logits,
        pool=pool,
        out=torch.empty((rows, shape.page_table_width), dtype=torch.int32, device=device),
        valid=torch.empty((rows,), dtype=torch.int32, device=device),
    )


def e2a_call(ext: Any, op: Operands, shape: Shape) -> Any:
    args = (op.pool, op.seq, op.req, op.block_table, shape.block_size, op.out, op.valid, _E2A_THREADS)
    return lambda: ext.e2a(*args)


def topk6_call(torch: Any, ext: Any, op: Operands, shape: Shape, *, stage: int = _TOPK6_STAGE_FUSED,
               pool_out: Any = None) -> Any:
    device = op.seq.device
    pool_out = pool_out if pool_out is not None else torch.empty((shape.rows, shape.top_k), dtype=torch.int32,
                                                                 device=device)
    dummy = torch.zeros(max(shape.rows, 1), dtype=torch.float32, device=device)
    none_i32 = torch.empty(0, dtype=torch.int32, device=device)
    none_i64 = torch.empty(0, dtype=torch.int64, device=device)
    args = (op.logits, op.lengths, stage, _TOPK6_PF, pool_out, dummy, none_i32, none_i64, op.seq, op.req,
            op.block_table, shape.block_size, op.out, op.valid)
    return lambda: ext.topk6(*args)


def _reference(torch: Any, op: Operands, pool: Any, shape: Shape) -> tuple[Any, Any]:
    from profiling.runners.attention.dsa_pooled_page_table_reference import pooled_page_table_reference

    return pooled_page_table_reference(
        pool, op.seq, op.req, op.block_table, kpool=shape.kpool, block_size=shape.block_size,
        block_stride=shape.block_size, page_table_width=shape.page_table_width,
    )


def check_page_table(torch: Any, ext: Any, op: Operands, shape: Shape) -> None:
    """E2a: bit-exact page table and valid counts against the Torch reference."""
    op.out.fill_(12345)
    op.valid.fill_(12345)
    e2a_call(ext, op, shape)()
    torch.cuda.synchronize()
    ref_out, ref_valid = _reference(torch, op, op.pool, shape)
    if not torch.equal(op.valid, ref_valid):
        raise KernelLaunchFailed(f"{PAGE_TABLE_BACKEND} valid counts differ from the reference")
    if not torch.equal(op.out, ref_out):
        bad = int((op.out != ref_out).any(1).sum())
        raise KernelLaunchFailed(f"{PAGE_TABLE_BACKEND} page table differs from the reference on {bad} rows")


def check_logits_page_table(torch: Any, ext: Any, op: Operands, shape: Shape) -> dict[str, int]:
    """Fused top-k: selected values equal torch.topk's per row; the page table of the selection is exact.

    Ties may pick other pools than the reference; the values selected cannot differ. The page table is
    compared sorted per row against the reference page table of the kernel's own selection (stage 3 of the
    same kernel), and the valid counts exactly.
    """
    from profiling.runners.attention.dsa_pooled_page_table_reference import pooled_topk_reference_values

    sel = torch.full((shape.rows, shape.top_k), -7, dtype=torch.int32, device=op.seq.device)
    topk6_call(torch, ext, op, shape, stage=_TOPK6_STAGE_SELECT, pool_out=sel)()
    op.out.fill_(12345)
    op.valid.fill_(12345)
    topk6_call(torch, ext, op, shape)()
    torch.cuda.synchronize()
    rows = list(range(0, shape.rows, max(1, shape.rows // _CHECK_ROWS)))
    ref_values = pooled_topk_reference_values(op.logits[rows], op.lengths[rows], shape.top_k)
    for row, values in zip(rows, ref_values, strict=True):
        k = values.numel()
        picked = sel[row]
        if bool((picked[k:] != -1).any()) or int(picked[:k].unique().numel()) != k:
            raise KernelLaunchFailed(f"{LOGITS_BACKEND} row {row}: malformed selection")
        if bool((picked[:k] < 0).any()) or bool((picked[:k] >= op.lengths[row]).any()):
            raise KernelLaunchFailed(f"{LOGITS_BACKEND} row {row}: selection out of range")
        got = torch.sort(op.logits[row, picked[:k].long()]).values
        if not torch.equal(got, values):
            raise KernelLaunchFailed(f"{LOGITS_BACKEND} row {row}: selected values differ from torch.topk")
    ref_out, ref_valid = _reference(torch, op, sel, shape)
    if not torch.equal(op.valid, ref_valid):
        raise KernelLaunchFailed(f"{LOGITS_BACKEND} valid counts differ from the reference")
    if not torch.equal(torch.sort(op.out, dim=1).values, torch.sort(ref_out, dim=1).values):
        raise KernelLaunchFailed(f"{LOGITS_BACKEND} page table rows differ from the selection's page table")
    same_set = torch.equal(torch.sort(sel, dim=1).values, torch.sort(op.pool, dim=1).values)
    return {"same_set_as_reference_topk": int(same_set)}


def page_table_bytes(shape: Shape) -> int:
    """Pool ids, seq len and request id in; one block-table entry per pool and tail token; row + count out."""
    rows = shape.rows
    return rows * (4 * shape.top_k + 8 + 4 * (shape.top_k + shape.kpool - 1) + 4 * shape.page_table_width + 4)


def logits_bytes(op: Operands) -> int:
    return 4 * int(op.lengths.to(dtype=op.lengths.dtype).sum())


def _import_torch(backend: str) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ProfilerNotImplemented(f"{backend} requires PyTorch") from exc
    return torch


def _metrics(time_ms: float, energy_j: float, logical_bytes: int) -> ComputeMetrics:
    return ComputeMetrics(
        time_ms=float(time_ms),
        energy_j=float(energy_j),
        tflops=0.0,
        memory_bandwidth_gbps=logical_bytes / (time_ms / 1000.0) / 1e9,
    )


def profile_dsa_pooled_topk_page_table_cuda_row_scan_compact(
    batch_size: int,
    context_len: int,
    next_n: int,
    max_model_len: int,
    top_k: int,
    kpool: int,
    page_table_width: int,
    block_size: int,
    index_dtype: str,
) -> ComputeMetrics:
    """Profile the one-launch pooled top-k to page table kernel (vendored ``e2a``, 512 threads)."""
    shape = validate_shape(
        batch_size=batch_size, context_len=context_len, next_n=next_n, max_model_len=max_model_len,
        top_k=top_k, kpool=kpool, page_table_width=page_table_width, block_size=block_size,
        index_dtype=index_dtype,
    )
    torch = _import_torch(PAGE_TABLE_BACKEND)
    try:
        ext = _extension()
        device = torch.device("cuda", torch.cuda.current_device())
        row_stride = shape.context_len + (-shape.context_len) % 4
        op = build_operands(torch, shape, row_stride=row_stride, context_mode="uniform_measured", device=device)
        op.logits = None  # the selection is built; the kernel never reads scores
        torch.cuda.empty_cache()
        check_page_table(torch, ext, op, shape)
        kernel = e2a_call(ext, op, shape)
        time_ms = Timer.cupti(kernel, warmup=5)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{PAGE_TABLE_BACKEND} ran out of CUDA memory") from exc
    except (ProfilerNotImplemented, ValueError, KernelLaunchFailed):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{PAGE_TABLE_BACKEND} failed: {exc}") from exc
    return _metrics(time_ms, energy_j, page_table_bytes(shape))


def profile_dsa_pooled_logits_page_table_cuda_threshold_filter_radix(
    batch_size: int,
    context_len: int,
    next_n: int,
    max_model_len: int,
    logits_row_stride: int,
    top_k: int,
    kpool: int,
    page_table_width: int,
    block_size: int,
    logits_dtype: DType | str,
    index_dtype: str,
    context_mode: str,
) -> ComputeMetrics:
    """Profile the fused pooled top-k + page table kernel (vendored ``topk6``, stage 4)."""
    shape = validate_shape(
        batch_size=batch_size, context_len=context_len, next_n=next_n, max_model_len=max_model_len,
        top_k=top_k, kpool=kpool, page_table_width=page_table_width, block_size=block_size,
        index_dtype=index_dtype,
    )
    if str(getattr(logits_dtype, "value", logits_dtype)) != "fp32":
        raise ValueError("logits_dtype must be fp32")
    if context_mode not in _CONTEXT_MODES:
        raise ValueError(f"context_mode must be one of {_CONTEXT_MODES}")
    if type(logits_row_stride) is not int or logits_row_stride < context_len or logits_row_stride % 4:
        raise ValueError("logits_row_stride must be a multiple of 4 covering context_len")
    torch = _import_torch(LOGITS_BACKEND)
    try:
        ext = _extension()
        device = torch.device("cuda", torch.cuda.current_device())
        op = build_operands(torch, shape, row_stride=logits_row_stride, context_mode=context_mode, device=device)
        check_logits_page_table(torch, ext, op, shape)
        kernel = topk6_call(torch, ext, op, shape)
        time_ms = Timer.cupti(kernel, warmup=5)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{LOGITS_BACKEND} ran out of CUDA memory") from exc
    except (ProfilerNotImplemented, ValueError, KernelLaunchFailed):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{LOGITS_BACKEND} failed: {exc}") from exc
    return _metrics(time_ms, energy_j, logits_bytes(op) + page_table_bytes(shape))


__all__ = [
    "LOGITS_BACKEND",
    "PAGE_TABLE_BACKEND",
    "build_operands",
    "e2a_call",
    "profile_dsa_pooled_logits_page_table_cuda_threshold_filter_radix",
    "profile_dsa_pooled_topk_page_table_cuda_row_scan_compact",
    "topk6_call",
    "validate_shape",
]
