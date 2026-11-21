"""Shared operand builders for DeepInfer's prefill sparse-indexer runners.

DeepInfer (``deepinfer/model_executor/models/v4_modules/index_attn.py``)
scores one query chunk per call. Every row's key window comes from
``prepare_cp_mirror`` (``tilelang_ops/cuda/prepare_cp_mirror.py``
``_write_ks_ke``): ``ks`` is the request's offset into the concatenated
compressed keys and ``ke = ks + (sk - sq + i + 1) // ratio`` for the
request's ``i``-th new row. One profiled call is one request whose
``num_queries`` rows are the tail of a ``context_len``-token context, so row
``i`` sees ``(context_len - num_queries + i + 1) // compress_ratio`` keys.

Index Q and K are FP4 E2M1 with one UE8M0 scale per 32 elements, packed four
to an int32 (DeepInfer's ``FP4_INDEX_KV_FORMAT``); DeepGEMM's own
``per_token_cast_to_fp4(..., use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)``
writes that layout. Weights and logits are bf16.

Pure-Python helpers here have no torch import so CPU tests can use them.
"""

from __future__ import annotations

from typing import Any

# DeepInfer's FP4_INDEX_KV_FORMAT: 32 elements per UE8M0 scale.
FP4_GRAN_K = 32


def tail_row_keys(num_queries: int, context_len: int, compress_ratio: int) -> list[int]:
    """Keys row ``i`` of one request's causal-tail chunk sees."""
    if not 0 < num_queries <= context_len:
        raise ValueError(f"need 0 < num_queries <= context_len, got {num_queries}, {context_len}")
    if compress_ratio < 1:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    first = context_len - num_queries + 1
    return [(first + i) // compress_ratio for i in range(num_queries)]


def num_request_keys(context_len: int, compress_ratio: int) -> int:
    """Compressed keys one request contributes to the concatenated key tensor."""
    return max(1, context_len // compress_ratio)


def build_tail_spans(torch: Any, num_queries: int, context_len: int, compress_ratio: int) -> tuple[Any, Any]:
    """int32 ``(ks, ke)`` on the GPU for one request's tail chunk."""
    keys = tail_row_keys(num_queries, context_len, compress_ratio)
    ke = torch.tensor(keys, dtype=torch.int32, device="cuda")
    ks = torch.zeros_like(ke)
    return ks, ke


def fp4_cast(torch: Any, x: Any) -> tuple[Any, Any]:
    """[N, D] bf16 -> (int8 packed E2M1 [N, D/2], int32 packed UE8M0 [N])."""
    from deep_gemm.utils import per_token_cast_to_fp4

    fp, sf = per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=FP4_GRAN_K, use_packed_ue8m0=True)
    return fp, sf.view(x.shape[0])


def fp4_dequant(torch: Any, fp: Any, sf: Any) -> Any:
    """Inverse of :func:`fp4_cast` in fp32, for the numerical checks."""
    from deep_gemm.utils import cast_back_from_fp4

    return cast_back_from_fp4(fp, sf.view(fp.shape[0], -1), gran_k=FP4_GRAN_K, use_packed_ue8m0=True)


def fp4_queries(torch: Any, num_queries: int, num_heads: int, head_dim: int) -> tuple[Any, Any, Any]:
    """Random index Q in the kernel layout, plus its fp32 dequantized copy."""
    q = torch.randn(num_queries * num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    fp, sf = fp4_cast(torch, q)
    simulated = fp4_dequant(torch, fp, sf).view(num_queries, num_heads, head_dim)
    return (
        fp.view(num_queries, num_heads, head_dim // 2),
        sf.view(num_queries, num_heads),
        simulated,
    )


def fp4_keys(torch: Any, num_keys: int, head_dim: int) -> tuple[Any, Any]:
    """Random index K in the kernel layout (built in slices to bound the bf16 temp)."""
    fp = torch.empty(num_keys, head_dim // 2, device="cuda", dtype=torch.int8)
    sf = torch.empty(num_keys, device="cuda", dtype=torch.int32)
    step = 1 << 20
    for start in range(0, num_keys, step):
        stop = min(num_keys, start + step)
        x = torch.randn(stop - start, head_dim, device="cuda", dtype=torch.bfloat16)
        fp[start:stop], sf[start:stop] = fp4_cast(torch, x)
    return fp, sf


def bf16_weights(torch: Any, num_queries: int, num_heads: int) -> Any:
    """Per-row, per-head bf16 weights; DeepInfer's ``_align_mqa_weights`` 16-byte rows."""
    stride = (num_heads * 2 + 15) // 16 * 16 // 2
    storage = torch.randn(num_queries, stride, device="cuda", dtype=torch.bfloat16)
    return storage[:, :num_heads]


def reference_logits_row(torch: Any, q_row: Any, k: Any, w_row: Any) -> Any:
    """fp32 ``sum_h w[h] * relu(q[h] . k)`` for one row against keys ``k`` [n, D]."""
    return (torch.relu(q_row.float() @ k.float().t()) * w_row.float().unsqueeze(1)).sum(0)


def calc_diff(x: Any, y: Any) -> float:
    """DeepGEMM's ``1 - cos-similarity`` style difference (deep_gemm.testing)."""
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    return float(1 - 2 * (x * y).sum() / denominator)


__all__ = [
    "FP4_GRAN_K",
    "bf16_weights",
    "build_tail_spans",
    "calc_diff",
    "fp4_cast",
    "fp4_dequant",
    "fp4_keys",
    "fp4_queries",
    "num_request_keys",
    "reference_logits_row",
    "tail_row_keys",
]
