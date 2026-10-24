"""TensorFold's speculative-tree verify attention as ``tree_verify_attention``.

This runner only allocates tensors, times one public call, and returns
metrics. DB writes, subprocess selection and registry routing live in L1b.

It times ``tensorfold.cuda.kernels.attention.attention(q, k_nodes, v_nodes,
offs, plan, scale=)`` (TensorFold perf-study 6469730a), called the way
``families/qwen3_5/cuda/forward.py`` ``multi_tree_forward`` calls it for one
softmax-attention layer:

- ``q`` is ``(W, num_qo_heads, head_dim)`` bf16, the streams' window rows back
  to back (already normed and roped); ``k_nodes`` / ``v_nodes`` are the window
  rows' own keys and values ``(W, num_kv_heads, head_dim)``.
- Each stream has its own contiguous bf16 cache ``(rows, num_kv_heads,
  head_dim)`` holding ``committed`` keys; ``offs`` is ``offsets(caches)`` as a
  device ``(S, 2)`` int64 tensor.
- ``plan`` is ``plan(parents, committed, num_qo_heads // num_kv_heads)``.
  ``multi_tree_forward`` builds it once a forward (``plan_host`` plus one copy
  and the ``_paths`` launch) and shares it across all attention layers, so it
  is built before timing.
- ``scale = head_dim ** -0.5``.

The timed closure is the three Triton launches of one call, ``_shared`` (none
when no stream has a whole committed chunk), ``_tail`` and ``_merge``; CUPTI
sums every launch (``kernel_name=None``). The partial buffers and the output
are allocated inside the call, as in production; allocation launches nothing.
Each stream's window is a chain (parents ``r - 1``): the tree's shape only
masks ``_tail`` loads and measured within noise of bushy trees.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

from profiling.db.args import DType
from profiling.runners.attention import _common
from profiling.runners.attention.tree_verify_attention_reference import (
    tree_verify_attention_reference,
)
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics

_MODULE = "tensorfold.cuda.kernels.attention"
_LABEL = "tree_verify_attention:tensorfold_triton"
_HEAD_DIMS = (128, 256)
# ``attention.MAX_NODES``: a stream's window takes 1..128 rows.
_MAX_ROWS = 128
# Guard streams for the oracle, whatever the profiled shape: an empty cache
# under a bushy 5-row tree; 16 rows of a best-first tree over 8229 keys (four
# whole 2048-key groups x 16 rows = 64, so _shared folds groups, and the tail
# chunk is partial); one row over 1100 keys (two unfolded chunks, partial tail).
_GUARD = (
    ((-1, 0, 0, 1, 1), 0),
    ((-1, 0, 0, 1, 1, 2, 2, 3, 0, 5, 4, 6, 8, 3, 9, 12), 8229),
    ((-1,), 1100),
)
# Triton rounds P to bf16 for the PV product and stores bf16 output; on randn
# inputs the RMSE ratio against the fp32 oracle measured 1.5e-3 on GB10.
RMSE_RATIO_TOL = 1e-2


def validate_args(
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_dtype: DType | str,
    kv_dtype: DType | str,
    o_dtype: DType | str,
    query_context_pairs: Sequence[Sequence[int]],
) -> tuple[tuple[int, int], ...]:
    """``attention``'s contract (bf16, head_dim 128/256) and the plan's (1..128 rows a stream)."""

    for name, dtype in (("q_dtype", q_dtype), ("kv_dtype", kv_dtype), ("o_dtype", o_dtype)):
        if DType.from_value(dtype) is not DType.BF16:
            raise ProfilerNotImplemented(f"{_LABEL} is bf16 only; got {name}={dtype}")
    if head_dim not in _HEAD_DIMS:
        raise ProfilerNotImplemented(f"{_LABEL} supports head_dim in {_HEAD_DIMS}, got {head_dim}")
    if num_kv_heads <= 0 or num_qo_heads <= 0 or num_qo_heads % num_kv_heads:
        raise ValueError(f"{_LABEL} needs num_qo_heads a positive multiple of num_kv_heads")
    if not query_context_pairs:
        raise ValueError(f"{_LABEL} needs at least one stream")
    pairs = []
    for pair in query_context_pairs:
        if len(pair) != 2 or any(type(v) is not int for v in pair):
            raise TypeError("query_context_pairs must hold integer (rows, committed) pairs")
        rows, committed = pair
        if not 1 <= rows <= _MAX_ROWS:
            raise ValueError(f"{_LABEL}: a stream's window takes 1..{_MAX_ROWS} rows, got {rows}")
        if committed < 0:
            raise ValueError(f"{_LABEL}: committed keys must be >= 0, got {committed}")
        pairs.append((rows, committed))
    return tuple(pairs)


def chain(rows: int) -> list[int]:
    return [-1, *range(rows - 1)]


def build_operands(
    torch: Any,
    offsets: Any,
    *,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    committed: Sequence[int],
    width: int,
) -> tuple[Any, Any, Any, list[tuple[Any, Any]], Any]:
    """Window rows, node keys/values, one cache per stream and the device offsets."""

    def randn(*shape: int) -> Any:
        return torch.randn(*shape, dtype=torch.bfloat16, device="cuda")

    q = randn(width, num_qo_heads, head_dim)
    k_nodes = randn(width, num_kv_heads, head_dim)
    v_nodes = randn(width, num_kv_heads, head_dim)
    # A stream with no committed keys still needs a valid aligned offset.
    caches = [
        (randn(max(p, 1), num_kv_heads, head_dim), randn(max(p, 1), num_kv_heads, head_dim))
        for p in committed
    ]
    offs = torch.tensor(offsets(caches, q.device), dtype=torch.int64, device=q.device)
    return q, k_nodes, v_nodes, caches, offs.view(len(caches), 2)


def check_correctness(
    torch: Any, module: Any, *, num_qo_heads: int, num_kv_heads: int, head_dim: int
) -> float:
    """One call on the guard streams against the oracle; return its RMSE ratio."""

    parents = [list(p) for p, _ in _GUARD]
    committed = [c for _, c in _GUARD]
    q, k_nodes, v_nodes, caches, offs = build_operands(
        torch,
        module.offsets,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        committed=committed,
        width=sum(len(p) for p in parents),
    )
    scale = head_dim**-0.5
    plan = module.plan(parents, committed, num_qo_heads // num_kv_heads, q.device)
    out = module.attention(q, k_nodes, v_nodes, offs, plan, scale=scale)
    torch.cuda.synchronize()
    if out.shape != q.shape or out.dtype != torch.bfloat16:
        raise AssertionError(f"tensorfold attention returned {tuple(out.shape)} {out.dtype}")
    if not torch.isfinite(out).all():
        raise AssertionError("tensorfold attention returned non-finite values")
    expected = tree_verify_attention_reference(
        torch, q, k_nodes, v_nodes, caches, parents, committed, scale
    )
    err = float(
        (out.float() - expected).pow(2).mean().sqrt()
        / expected.pow(2).mean().sqrt().clamp_min(1e-12)
    )
    if not err < RMSE_RATIO_TOL:
        raise AssertionError(
            f"tensorfold tree attention mismatch vs fp32 SDPA oracle: rmse ratio {err:.2e} "
            f"(tol {RMSE_RATIO_TOL:.0e})"
        )
    return err


def work(
    pairs: Sequence[tuple[int, int]], *, num_qo_heads: int, num_kv_heads: int, head_dim: int
) -> tuple[int, int]:
    """FLOPs and logical bytes of one call over chains (row ``r`` sees ``p + r + 1`` keys)."""

    keys_seen = sum(w * p + w * (w + 1) // 2 for w, p in pairs)
    flops = 4 * num_qo_heads * head_dim * keys_seen
    width = sum(w for w, _ in pairs)
    elem = 2  # bf16
    q_and_o = 2 * width * num_qo_heads * head_dim * elem
    nodes = 2 * width * num_kv_heads * head_dim * elem
    committed = 2 * sum(p for _, p in pairs) * num_kv_heads * head_dim * elem
    return flops, q_and_o + nodes + committed


def profile_tree_verify_attention_tensorfold_triton(
    *,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_dtype: DType | str,
    kv_dtype: DType | str,
    o_dtype: DType | str,
    query_context_pairs: Sequence[Sequence[int]],
) -> ComputeMetrics:
    num_qo_heads, num_kv_heads, head_dim = int(num_qo_heads), int(num_kv_heads), int(head_dim)
    pairs = validate_args(
        num_qo_heads, num_kv_heads, head_dim, q_dtype, kv_dtype, o_dtype, query_context_pairs
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_LABEL}") from exc
    try:
        module = importlib.import_module(_MODULE)
        module.plan, module.attention, module.offsets  # noqa: B018
    except (ImportError, AttributeError) as exc:
        raise ProfilerNotImplemented(f"{_LABEL} needs {_MODULE} (tensorfold_env)") from exc

    dims = dict(num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads, head_dim=head_dim)
    try:
        check_correctness(torch, module, **dims)
        committed = [p for _, p in pairs]
        q, k_nodes, v_nodes, _caches, offs = build_operands(
            torch, module.offsets, **dims, committed=committed, width=sum(w for w, _ in pairs)
        )
        plan = module.plan(
            [chain(w) for w, _ in pairs], committed, num_qo_heads // num_kv_heads, q.device
        )
        scale = head_dim**-0.5
        torch.cuda.synchronize()

        def kernel() -> Any:
            return module.attention(q, k_nodes, v_nodes, offs, plan, scale=scale)

        flops, bytes_accessed = work(pairs, **dims)
        return _common.measure(kernel, flops=flops, bytes_accessed=bytes_accessed)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_LABEL} ran out of CUDA memory") from exc
    except (ProfilerNotImplemented, ValueError, TypeError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_LABEL} failed: {exc}") from exc
