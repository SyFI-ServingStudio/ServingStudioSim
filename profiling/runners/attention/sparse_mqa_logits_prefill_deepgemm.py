"""DeepInfer's candidate-restricted prefill indexer logits on SM100.

Two backends live here, one per DeepGEMM callable of the sparse branch of
``MQA._compute_prefill_p_sum`` (``v4_modules/index_attn.py``):

* ``sparse_mqa_logits_metadata`` / ``deepgemm_sm100`` -- the schedule,
  ``deep_gemm.get_sparse_mqa_logits_metadata(ks, ke, num_kv_tokens,
  absolute_candidate_blocks, q_dtype, candidate_block_size,
  use_unaligned_ks=True)`` (one ``sm100_sparse_mqa_logits_metadata`` launch).
  ``_get_sparse_prefill_metadata`` caches it per query chunk for the whole
  candidate group, so it launches once per chunk, not per layer.
* ``sparse_mqa_logits_prefill`` / ``deepgemm_sm100`` -- the scoring,
  ``deep_gemm.fp8_fp4_sparse_mqa_logits((q, q_sf), (k, k_sf), weights,
  metadata, candidate_blocks, candidate_block_size, use_unaligned_ks=True)``
  (one ``sm100_sparse_mqa_logits`` launch per consumer layer and chunk). It
  writes bf16 logits for the ``candidate_blocks * candidate_block_size``
  candidate keys of each row.

Candidate blocks: each row keeps ``min(candidate_blocks, valid blocks)``
distinct sequence-local blocks in ascending order (DeepSelect
``sorted_index=True``), its last valid block always among them (the source
layer's ``candidate_block_max`` scores it +inf). Rows of one request share
most of their blocks: the selection is the top of a per-request block score
plus per-row noise (DeepGEMM's own sparse test keeps 80% of a row's blocks
from the previous row). DeepInfer anchors the blocks at ``ks //
candidate_block_size``; one request starts at key 0.

Check (scoring backend): four rows of the sparse logits against an fp32
transcription on the dequantized operands over the same candidate keys.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention import deepinfer_indexer_common as common
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

BACKEND = "deepgemm_sm100"
TOL = 1e-3
# Per-row noise relative to the shared per-request block score.
ROW_NOISE = 0.35


def validate(
    num_queries: int,
    context_len: int,
    compress_ratio: int,
    candidate_blocks: int,
    candidate_block_size: int,
    dtype: DType | str,
) -> tuple[int, int, int, int, int]:
    if DType.from_value(dtype) is not DType.MXFP4_E2M1:
        raise ValueError(f"{BACKEND} sparse MQA logits take dtype=mxfp4_e2m1, got {dtype!r}")
    values = (
        int(num_queries),
        int(context_len),
        int(compress_ratio),
        int(candidate_blocks),
        int(candidate_block_size),
    )
    if min(values) < 1 or values[0] > values[1]:
        raise ValueError(f"need positive sizes and num_queries <= context_len, got {values}")
    # fp8_fp4_sparse_mqa_logits host asserts (csrc/apis/attention.hpp).
    if values[3] % 4 or values[3] > 4096 or values[4] not in (8, 16):
        raise ValueError(
            f"candidate_blocks must be a multiple of 4 up to 4096 and block size 8 or 16, got {values[3:]}"
        )
    return values


def row_valid_blocks(num_queries: int, context_len: int, compress_ratio: int, block_size: int) -> list[int]:
    return [
        -(-keys // block_size)
        for keys in common.tail_row_keys(num_queries, context_len, compress_ratio)
    ]


def candidate_tokens(
    num_queries: int,
    context_len: int,
    compress_ratio: int,
    candidate_blocks: int,
    candidate_block_size: int,
) -> int:
    """Valid candidate keys summed over rows (the tokens the kernel scores)."""
    keys = common.tail_row_keys(num_queries, context_len, compress_ratio)
    blocks = row_valid_blocks(num_queries, context_len, compress_ratio, candidate_block_size)
    return sum(
        min(k, min(b, candidate_blocks) * candidate_block_size) for k, b in zip(keys, blocks)
    )


def build_candidates(
    torch: Any, num_queries: int, context_len: int, compress_ratio: int, num_blocks: int, block_size: int
) -> Any:
    """int32 [num_queries, num_blocks] sorted sequence-local candidate blocks, -1 padded."""
    valid = torch.tensor(
        row_valid_blocks(num_queries, context_len, compress_ratio, block_size),
        dtype=torch.int64,
        device="cuda",
    )
    width = int(valid.max().item())
    out = torch.full((num_queries, num_blocks), -1, dtype=torch.int32, device="cuda")
    base = torch.randn(width, device="cuda")
    cols = torch.arange(width, device="cuda")
    step = max(1, (1 << 27) // max(width, 1))
    for start in range(0, num_queries, step):
        stop = min(num_queries, start + step)
        score = base + ROW_NOISE * torch.randn(stop - start, width, device="cuda")
        rows_valid = valid[start:stop].unsqueeze(1)
        score.masked_fill_(cols >= rows_valid, float("-inf"))
        score.scatter_(1, rows_valid - 1, float("inf"))
        take = min(num_blocks, width)
        top = score.topk(take, dim=1).indices
        keep = torch.arange(take, device="cuda") < rows_valid.clamp(max=take)
        top = torch.where(keep, top, torch.full_like(top, 1 << 30)).sort(dim=1).values
        top = torch.where(top == (1 << 30), torch.full_like(top, -1), top)
        out[start:stop, :take] = top.to(torch.int32)
    return out


class _Operands:
    def __init__(self, torch: Any, args: tuple[int, int, int, int, int], num_heads: int, head_dim: int):
        num_queries, context_len, ratio, num_blocks, block_size = args
        torch.manual_seed(0)
        self.num_keys = common.num_request_keys(context_len, ratio)
        self.q_fp, self.q_sf, self.q_ref = common.fp4_queries(torch, num_queries, num_heads, head_dim)
        self.k_fp, self.k_sf = common.fp4_keys(torch, self.num_keys, head_dim)
        self.weights = common.bf16_weights(torch, num_queries, num_heads)
        self.ks, self.ke = common.build_tail_spans(torch, num_queries, context_len, ratio)
        local = build_candidates(torch, num_queries, context_len, ratio, num_blocks, block_size)
        # index_attn.py `_get_sparse_prefill_metadata`: anchor at floor(ks / block_size).
        base = torch.div(self.ks, block_size, rounding_mode="floor")
        self.blocks = (local + base.unsqueeze(1)).contiguous()
        self.local = local
        self.num_blocks, self.block_size = num_blocks, block_size

    def metadata(self, torch: Any, deep_gemm: Any) -> Any:
        return deep_gemm.get_sparse_mqa_logits_metadata(
            self.ks,
            self.ke,
            self.num_keys,
            self.blocks,
            self.q_fp.dtype,
            self.block_size,
            use_unaligned_ks=True,
        )


def _imports() -> tuple[Any, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"torch is required for {BACKEND}") from exc
    try:
        import deep_gemm
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{BACKEND} needs DeepInfer's DeepGEMM (deepinfer_env)") from exc
    return torch, deep_gemm


def _check_logits(torch: Any, ops: _Operands, logits: Any) -> None:
    k_ref = common.fp4_dequant(torch, ops.k_fp, ops.k_sf)
    ends = ops.ke.tolist()
    num_queries = len(ends)
    offsets = torch.arange(ops.block_size, device="cuda")
    for row in sorted({0, num_queries // 3, 2 * num_queries // 3, num_queries - 1}):
        blocks = ops.local[row]
        blocks = blocks[blocks >= 0].long()
        tokens = (blocks.unsqueeze(1) * ops.block_size + offsets).flatten()
        valid = tokens < ends[row]
        want = common.reference_logits_row(torch, ops.q_ref[row], k_ref[tokens[valid]], ops.weights[row])
        got = logits[row, : tokens.numel()][valid].float()
        diff = common.calc_diff(got, want)
        if not diff <= TOL:
            raise KernelLaunchFailed(f"{BACKEND} sparse logits row {row}: diff {diff:.3e} > {TOL:.0e}")


def _failed(exc: Exception, what: str) -> Exception:
    if exc.__class__.__name__ == "OutOfMemoryError":
        return OOMError(f"{BACKEND} {what} ran out of memory")
    return KernelLaunchFailed(f"{BACKEND} {what} failed: {exc}")


def profile_sparse_mqa_logits_prefill_deepgemm(
    num_queries: int,
    context_len: int,
    compress_ratio: int,
    candidate_blocks: int,
    candidate_block_size: int,
    num_heads: int,
    head_dim: int,
    dtype: DType | str,
) -> ComputeMetrics:
    args = validate(num_queries, context_len, compress_ratio, candidate_blocks, candidate_block_size, dtype)
    num_heads, head_dim = int(num_heads), int(head_dim)
    if head_dim != 128 or num_heads % 4 or num_heads > 64:
        raise ValueError(f"sparse MQA logits need head_dim 128 and heads % 4 == 0, got {num_heads}, {head_dim}")
    torch, deep_gemm = _imports()
    try:
        ops = _Operands(torch, args, num_heads, head_dim)
        metadata = ops.metadata(torch, deep_gemm)

        def run_once() -> Any:
            return deep_gemm.fp8_fp4_sparse_mqa_logits(
                (ops.q_fp, ops.q_sf),
                (ops.k_fp, ops.k_sf),
                ops.weights,
                metadata,
                ops.num_blocks,
                ops.block_size,
                use_unaligned_ks=True,
            )

        logits = run_once()  # JIT compile, outside timing.
        torch.cuda.synchronize()
        _check_logits(torch, ops, logits)
        del logits

        time_ms = Timer.cupti(run_once, warmup=3)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    except (ProfilerNotImplemented, KernelLaunchFailed):
        raise
    except Exception as exc:
        raise _failed(exc, "sparse_mqa_logits_prefill") from exc

    tokens = candidate_tokens(*args)
    flops = 2 * tokens * num_heads * head_dim
    bytes_accessed = (
        args[0] * num_heads * (head_dim // 2 + 4)
        + tokens * (head_dim // 2 + 4)
        + 2 * args[0] * num_heads
        + 2 * args[0] * args[3] * args[4]
    )
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=flops / elapsed_s / 1e12,
        memory_bandwidth_gbps=bytes_accessed / elapsed_s / 1e9,
        energy_j=float(energy_j),
    )


def profile_sparse_mqa_logits_metadata_deepgemm(
    num_queries: int,
    context_len: int,
    compress_ratio: int,
    candidate_blocks: int,
    candidate_block_size: int,
    dtype: DType | str,
) -> ComputeMetrics:
    args = validate(num_queries, context_len, compress_ratio, candidate_blocks, candidate_block_size, dtype)
    torch, deep_gemm = _imports()
    try:
        ops = _Operands(torch, args, 32, 128)

        def run_once() -> Any:
            return ops.metadata(torch, deep_gemm)

        metadata = run_once()  # JIT compile, outside timing.
        # The schedule is only meaningful through the kernel that reads it:
        # one scoring launch on it must reproduce the reference logits.
        logits = deep_gemm.fp8_fp4_sparse_mqa_logits(
            (ops.q_fp, ops.q_sf),
            (ops.k_fp, ops.k_sf),
            ops.weights,
            metadata,
            ops.num_blocks,
            ops.block_size,
            use_unaligned_ks=True,
        )
        torch.cuda.synchronize()
        _check_logits(torch, ops, logits)
        del logits

        time_ms = Timer.cupti(run_once, warmup=3)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    except (ProfilerNotImplemented, KernelLaunchFailed):
        raise
    except Exception as exc:
        raise _failed(exc, "sparse_mqa_logits_metadata") from exc

    # Reads the candidate blocks and the row bounds.
    bytes_accessed = 4 * args[0] * (args[3] + 2)
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
    "profile_sparse_mqa_logits_metadata_deepgemm",
    "profile_sparse_mqa_logits_prefill_deepgemm",
    "row_valid_blocks",
    "validate",
]
