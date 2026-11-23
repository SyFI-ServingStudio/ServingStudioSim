"""DeepInfer's candidate-restricted decode indexer logits on SM100 (paged).

Two backends live here, one per DeepGEMM callable of the sparse branch of
``MQA._topk_decode`` (``deepinfer/model_executor/models/v4_modules/index_attn.py``):

* ``sparse_mqa_logits_decode_metadata`` / ``deepgemm_sm100`` -- the schedule,
  ``deep_gemm.get_paged_sparse_mqa_logits_metadata(klen, block_tables,
  query_indices, page_size, candidate_blocks, q_dtype, candidate_block_size)``:
  one ``sm100_sparse_mqa_logits_metadata`` launch. ``_get_decode_metadata``
  caches it per candidate group for the step, so it runs once per group, not
  per consumer layer.
* ``sparse_mqa_logits_decode`` / ``deepgemm_sm100`` -- the scoring,
  ``deep_gemm.fp8_fp4_paged_sparse_mqa_logits((q_fp4, q_sf), kv_cache,
  weights, metadata, candidate_blocks, candidate_block_size)``: one
  ``sm100_paged_sparse_mqa_logits`` launch per consumer layer, bf16 logits of
  width ``candidate_blocks * candidate_block_size`` per row.

Rows, pages and the FP4 cache are DeepInfer's decode layout
(:mod:`deepinfer_paged_indexer_common`): ``next_n`` consecutive rows per
request, each row its request's block table and context length. Candidate
blocks are sequence-local and ascending (DeepSelect ``sorted_index=True``),
the last valid block always among them (``candidate_block_max`` scores it
+inf), -1 past a short row's valid blocks; the rows of a request share a
per-request block ranking perturbed per row, as the prefill runner
(``sparse_mqa_logits_prefill_deepgemm``) builds them.

Check (scoring): rows of the first and last request against fp32 math on the
dequantized operands over the same candidate keys.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention import deepinfer_indexer_common as common
from profiling.runners.attention import deepinfer_paged_indexer_common as paged
from profiling.runners.attention.sparse_mqa_logits_prefill_deepgemm import ROW_NOISE
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

BACKEND = "deepgemm_sm100"
TOL = 1e-3
# Score-matrix elements generated per slice when ranking candidate blocks.
_RANK_SLICE = 1 << 27


def validate(
    batch_size: int,
    context_len: int,
    next_n: int,
    block_size: int,
    candidate_blocks: int,
    candidate_block_size: int,
    dtype: DType | str,
) -> tuple[int, int, int, int, int, int]:
    if DType.from_value(dtype) is not DType.MXFP4_E2M1:
        raise ValueError(f"{BACKEND} paged sparse MQA logits take dtype=mxfp4_e2m1, got {dtype!r}")
    values = tuple(
        int(v)
        for v in (
            batch_size,
            context_len,
            next_n,
            block_size,
            candidate_blocks,
            candidate_block_size,
        )
    )
    if min(values) < 1:
        raise ValueError(f"paged sparse MQA logits need positive sizes, got {values}")
    # attention.hpp host asserts of get_paged_sparse_mqa_logits_metadata and
    # fp8_fp4_paged_sparse_mqa_logits.
    if values[4] % 4 or values[4] > 4096 or values[5] not in (8, 16):
        raise ValueError(
            "candidate_blocks must be a multiple of 4 up to 4096 and block size 8 or 16, "
            f"got {values[4:]}"
        )
    if values[3] % values[5]:
        raise ValueError(f"page size {values[3]} must hold whole candidate blocks of {values[5]}")
    # Page stride (block_size * 68 bytes for head_dim 128) must be a multiple of 128.
    if paged.page_bytes(values[3], 128) % 128:
        raise ValueError(f"page size {values[3]} gives a page stride not 128-byte aligned")
    return values  # type: ignore[return-value]


def row_valid_blocks(context_len: int, candidate_block_size: int) -> int:
    return -(-context_len // candidate_block_size)


def candidate_tokens(
    batch_size: int, context_len: int, next_n: int, candidate_blocks: int, candidate_block_size: int
) -> int:
    """Valid candidate keys summed over rows (the keys the kernel scores)."""
    valid = row_valid_blocks(context_len, candidate_block_size)
    per_row = min(context_len, min(valid, candidate_blocks) * candidate_block_size)
    return batch_size * next_n * per_row


def build_candidates(
    torch: Any, batch_size: int, next_n: int, context_len: int, num_blocks: int, block_size: int
) -> Any:
    """int32 [rows, num_blocks] ascending sequence-local candidate blocks, -1 padded."""
    width = row_valid_blocks(context_len, block_size)
    take = min(num_blocks, width)
    out = torch.full((batch_size * next_n, num_blocks), -1, dtype=torch.int32, device="cuda")
    requests_per_slice = max(1, _RANK_SLICE // max(1, next_n * width))
    generator = torch.Generator(device="cuda").manual_seed(1)
    for start in range(0, batch_size, requests_per_slice):
        stop = min(batch_size, start + requests_per_slice)
        n = stop - start
        base = torch.randn(n, 1, width, device="cuda", generator=generator)
        score = base + ROW_NOISE * torch.randn(n, next_n, width, device="cuda", generator=generator)
        score[..., width - 1] = float("inf")
        top = score.topk(take, dim=-1).indices.sort(dim=-1).values
        out[start * next_n : stop * next_n, :take] = top.view(n * next_n, take).to(torch.int32)
    return out


class _Operands:
    def __init__(
        self, torch: Any, args: tuple[int, int, int, int, int, int], num_heads: int, head_dim: int
    ):
        batch_size, context_len, next_n, block_size, num_blocks, cblock = args
        self.base = paged.build_paged_fp4(
            torch,
            batch_size=batch_size,
            next_n=next_n,
            context_len=context_len,
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=block_size,
        )
        self.klen = self.base.context_lens.view(-1).contiguous()
        self.blocks = build_candidates(torch, batch_size, next_n, context_len, num_blocks, cblock)
        self.num_blocks, self.cblock = num_blocks, cblock

    def metadata(self, deep_gemm: Any) -> Any:
        b = self.base
        return deep_gemm.get_paged_sparse_mqa_logits_metadata(
            self.klen,
            b.block_table,
            b.indices,
            b.block_size,
            self.blocks,
            b.q_fp.dtype,
            self.cblock,
        )

    def score(self, deep_gemm: Any, metadata: Any) -> Any:
        b = self.base
        return deep_gemm.fp8_fp4_paged_sparse_mqa_logits(
            (b.q_fp, b.q_sf), b.cache, b.weights, metadata, self.num_blocks, self.cblock
        )


def _imports() -> tuple[Any, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"torch is required for {BACKEND}") from exc
    try:
        import deep_gemm
    except ImportError as exc:
        raise ProfilerNotImplemented(
            f"{BACKEND} needs DeepInfer's DeepGEMM (deepinfer_env)"
        ) from exc
    return torch, deep_gemm


def _check_logits(torch: Any, ops: _Operands, logits: Any) -> None:
    b = ops.base
    offsets = torch.arange(ops.cblock, device="cuda")
    for request, local in ((0, 0), (0, b.next_n - 1), (b.batch_size - 1, b.next_n - 1)):
        keys = b.request_keys(torch, request)
        row = request * b.next_n + local
        blocks = ops.blocks[row]
        blocks = blocks[blocks >= 0].long()
        tokens = (blocks.unsqueeze(1) * ops.cblock + offsets).flatten()
        valid = tokens < b.context_len
        want = common.reference_logits_row(torch, b.q_ref[row], keys[tokens[valid]], b.weights[row])
        got = logits[row, : tokens.numel()][valid].float()
        diff = common.calc_diff(got, want)
        if not diff <= TOL:
            raise KernelLaunchFailed(
                f"{BACKEND} paged sparse logits row {row}: diff {diff:.3e} > {TOL:.0e}"
            )


def _failed(exc: Exception, what: str) -> Exception:
    if exc.__class__.__name__ == "OutOfMemoryError":
        return OOMError(f"{BACKEND} {what} ran out of memory")
    return KernelLaunchFailed(f"{BACKEND} {what} failed: {exc}")


def profile_sparse_mqa_logits_decode_deepgemm(
    batch_size: int,
    context_len: int,
    next_n: int,
    block_size: int,
    candidate_blocks: int,
    candidate_block_size: int,
    num_heads: int,
    head_dim: int,
    dtype: DType | str,
) -> ComputeMetrics:
    args = validate(
        batch_size, context_len, next_n, block_size, candidate_blocks, candidate_block_size, dtype
    )
    num_heads, head_dim = int(num_heads), int(head_dim)
    if head_dim != 128 or num_heads % 4 or num_heads > 32:
        raise ValueError(
            "paged sparse MQA logits need head_dim 128 and heads % 4 == 0 up to 32, "
            f"got {num_heads}, {head_dim}"
        )
    torch, deep_gemm = _imports()
    try:
        ops = _Operands(torch, args, num_heads, head_dim)
        metadata = ops.metadata(deep_gemm)

        def run_once() -> Any:
            return ops.score(deep_gemm, metadata)

        logits = run_once()  # JIT compile, outside timing.
        torch.cuda.synchronize()
        _check_logits(torch, ops, logits)
        del logits

        time_ms = Timer.cupti(run_once, warmup=3)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    except (ProfilerNotImplemented, KernelLaunchFailed):
        raise
    except Exception as exc:
        raise _failed(exc, "sparse_mqa_logits_decode") from exc

    rows = args[0] * args[2]
    tokens = candidate_tokens(args[0], args[1], args[2], args[4], args[5])
    flops = 2 * tokens * num_heads * head_dim
    bytes_accessed = (
        rows * num_heads * (head_dim // 2 + 4)
        + tokens * (head_dim // 2 + 4)
        + 2 * rows * num_heads
        + 2 * rows * args[4] * args[5]
    )
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=flops / elapsed_s / 1e12,
        memory_bandwidth_gbps=bytes_accessed / elapsed_s / 1e9,
        energy_j=float(energy_j),
    )


def profile_sparse_mqa_logits_decode_metadata_deepgemm(
    batch_size: int,
    context_len: int,
    next_n: int,
    block_size: int,
    candidate_blocks: int,
    candidate_block_size: int,
    dtype: DType | str,
) -> ComputeMetrics:
    args = validate(
        batch_size, context_len, next_n, block_size, candidate_blocks, candidate_block_size, dtype
    )
    torch, deep_gemm = _imports()
    try:
        ops = _Operands(torch, args, 32, 128)

        def run_once() -> Any:
            return ops.metadata(deep_gemm)

        metadata = run_once()  # JIT compile, outside timing.
        logits = ops.score(deep_gemm, metadata)
        torch.cuda.synchronize()
        _check_logits(torch, ops, logits)
        del logits

        time_ms = Timer.cupti(run_once, warmup=3)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    except (ProfilerNotImplemented, KernelLaunchFailed):
        raise
    except Exception as exc:
        raise _failed(exc, "sparse_mqa_logits_decode_metadata") from exc

    rows = args[0] * args[2]
    pages = paged.pages_per_request(args[1], args[3])
    # Candidate blocks, block-table rows, lengths and request indices.
    bytes_accessed = 4 * rows * (args[4] + pages + 2)
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=bytes_accessed / elapsed_s / 1e9,
        energy_j=float(energy_j),
    )


__all__ = [
    "build_candidates",
    "candidate_tokens",
    "profile_sparse_mqa_logits_decode_deepgemm",
    "profile_sparse_mqa_logits_decode_metadata_deepgemm",
    "row_valid_blocks",
    "validate",
]
