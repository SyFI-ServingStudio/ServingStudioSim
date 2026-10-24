"""TensorFold prefill chain as a ``gdn_chunk_delta_rule`` backend.

This runner only allocates tensors, times the public call, and returns metrics.
DB writes, subprocess selection and registry routing live in L1b.

It times ``tensorfold.cuda.kernels.gdn.chain(q, k, v, g, beta, state, final)``
(TensorFold perf-study 6469730a), called the way
``families/qwen3_5/cuda/prefill.py`` calls it for a prefill step:

- ``q``/``k`` are ``(W, num_key_heads, 128)`` bf16, ``v`` is
  ``(W, num_heads, 128)`` bf16, ``g`` and ``beta`` are ``(W, num_heads)`` fp32.
  Triton ``_gdn_pre`` produces them in a separate launch: conv + SiLU, q and k
  L2-normalized (q also carries ``key_head_dim ** -0.5``), ``g`` the already
  exponentiated decay, ``beta`` already through the sigmoid. That launch is not
  part of this kind, the same boundary as the FlashInfer backend's.
- ``state`` is the stream's fp32 ``(num_heads, value_head_dim, key_head_dim)``
  recurrent state, read only; ``final`` is a caller-allocated state-shaped fp32
  tensor the call writes the last row's state into. The per-token output
  ``(W, num_heads, 128)`` bf16 is allocated inside the call (no launch) and goes
  to the separate ``_gated_norm``.

One call is one ``chain_kernel`` launch over one sequence: a block per
(value head, 128 value rows) walks the rows serially in fp32. A batched prefill
step loops the call once per stream (prefill.py ~288-298), so the kind's
``num_tokens // max_sequence_length`` full sequences plus one remainder are
realized as that many back-to-back calls, each from its own state, and the time
is the CUPTI sum over every launch (``kernel_name=None``). The ``torch.cat`` of
the per-stream outputs that follows in production is caller glue, not timed.
The extension JIT-compiles on first call, which the correctness check absorbs
before timing.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.runners.attention import _common
from profiling.runners.attention._gdn_common import (
    canonical_sequence_boundaries,
    chunk_delta_rule_logical_bytes,
    chunk_delta_rule_semantic_flops,
    exact_int,
)
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics

_LABEL = "gdn_chunk_delta_rule:tensorfold_chain"
_MODULE = "tensorfold.cuda.kernels.gdn"
_CALLABLE = "chain"
# gdn.cpp `prefill`/`check_rows` and gdn_prefill.cu hard-code DK = DV = 128.
_HEAD_DIM = 128
# Guard shape for the oracle: crosses the kernel's 32-row staging boundary
# twice and splits into two sequences, so the second call starts from a
# different state. Small enough for a sequential fp32 Torch scan.
_GUARD_SEQUENCES = (77, 40)
# Output is stored bf16 (one rounding of an fp32 value): the measured RMSE
# ratio against the fp32 oracle is ~2e-3. The final state stays fp32.
OUTPUT_RMSE_TOL = 1e-2
STATE_RMSE_TOL = 1e-4


@dataclass(frozen=True)
class _Args:
    num_tokens: int
    max_sequence_length: int
    num_key_heads: int
    num_heads: int
    key_head_dim: int
    value_head_dim: int


def validate_args(
    num_tokens: int,
    max_sequence_length: int,
    num_key_heads: int,
    num_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: DType | str,
) -> _Args:
    """The public callable's contract: bf16 q/k/v, 128-wide heads, grouped value heads."""
    args = _Args(
        num_tokens=exact_int("num_tokens", num_tokens),
        max_sequence_length=exact_int("max_sequence_length", max_sequence_length),
        num_key_heads=exact_int("num_key_heads", num_key_heads),
        num_heads=exact_int("num_heads", num_heads),
        key_head_dim=exact_int("key_head_dim", key_head_dim),
        value_head_dim=exact_int("value_head_dim", value_head_dim),
    )
    if DType.from_value(dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_LABEL} measures bf16 q/k/v only, got {dtype}")
    if (args.key_head_dim, args.value_head_dim) != (_HEAD_DIM, _HEAD_DIM):
        raise ProfilerNotImplemented(
            f"{_LABEL} supports key_head_dim = value_head_dim = {_HEAD_DIM}, got "
            f"{args.key_head_dim}/{args.value_head_dim}"
        )
    if args.max_sequence_length < 1 or args.num_tokens < args.max_sequence_length:
        raise ValueError("num_tokens must be >= max_sequence_length >= 1")
    if args.num_key_heads < 1 or args.num_heads % args.num_key_heads != 0:
        raise ValueError("num_heads must be a positive multiple of num_key_heads")
    return args


def build_operands(torch: Any, *, rows: int, num_key_heads: int, num_heads: int) -> tuple[Any, ...]:
    """``_gdn_pre``'s outputs for ``rows`` rows, with its value ranges and dtypes."""
    gen = torch.Generator(device="cuda").manual_seed(0)

    def unit(shape: tuple[int, ...]) -> Any:
        x = torch.randn(*shape, generator=gen, device="cuda", dtype=torch.float32)
        return x / x.norm(dim=-1, keepdim=True)

    q = (unit((rows, num_key_heads, _HEAD_DIM)) * _HEAD_DIM**-0.5).to(torch.bfloat16)
    k = unit((rows, num_key_heads, _HEAD_DIM)).to(torch.bfloat16)
    v = torch.randn(rows, num_heads, _HEAD_DIM, generator=gen, device="cuda").to(torch.bfloat16)
    # g = exp(-exp(A_log) * softplus(a + dt_bias)) lies in (0, 1); beta = sigmoid(b).
    g = torch.exp(-torch.rand(rows, num_heads, generator=gen, device="cuda") * 0.5)
    beta = torch.sigmoid(torch.randn(rows, num_heads, generator=gen, device="cuda"))
    return q, k, v, g, beta


def new_state(torch: Any, num_heads: int, *, seed: int) -> Any:
    """A nonzero fp32 ``(num_heads, value_head_dim, key_head_dim)`` carried state."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return 0.1 * torch.randn(
        num_heads, _HEAD_DIM, _HEAD_DIM, generator=gen, device="cuda", dtype=torch.float32
    )


def reference(torch: Any, q: Any, k: Any, v: Any, g: Any, beta: Any, state: Any) -> tuple[Any, Any]:
    """Independent sequential fp32 gated delta rule over one sequence.

    With S the ``(value, key)`` state of head h and key head h // (num_heads /
    num_key_heads), each row t does S <- g S; S <- S + beta (v - S k) k^T;
    o = S q. Returns the per-row output (W, num_heads, 128) and the final state.
    """
    group = v.shape[1] // q.shape[1]
    qf = q.float().repeat_interleave(group, dim=1)
    kf = k.float().repeat_interleave(group, dim=1)
    vf, gf, bf = v.float(), g.float(), beta.float()
    s = state.clone()
    outs = []
    for t in range(q.shape[0]):
        s = s * gf[t][:, None, None]
        memory = torch.einsum("hvk,hk->hv", s, kf[t])
        delta = (vf[t] - memory) * bf[t][:, None]
        s = s + delta[:, :, None] * kf[t][:, None, :]
        outs.append(torch.einsum("hvk,hk->hv", s, qf[t]))
    return torch.stack(outs), s


def _rmse_ratio(torch: Any, got: Any, expected: Any) -> float:
    return float(
        (got.float() - expected).pow(2).mean().sqrt()
        / expected.pow(2).mean().sqrt().clamp_min(1e-12)
    )


def check_correctness(
    torch: Any, chain: Any, *, num_key_heads: int, num_heads: int
) -> tuple[float, float]:
    """Chain two guard sequences as production does and compare with the oracle.

    The second call starts from the first call's ``final``, as the prompt-cache
    cut in prefill.py does. Returns the worst (output, state) RMSE ratios.
    """
    rows = sum(_GUARD_SEQUENCES)
    q, k, v, g, beta = build_operands(
        torch, rows=rows, num_key_heads=num_key_heads, num_heads=num_heads
    )
    state = new_state(torch, num_heads, seed=1)
    expected_state = state
    worst_out = worst_state = 0.0
    lo = 0
    for length in _GUARD_SEQUENCES:
        hi = lo + length
        operands = (q[lo:hi], k[lo:hi], v[lo:hi], g[lo:hi], beta[lo:hi])
        final = torch.empty_like(state)
        out = chain(*operands, state, final)
        torch.cuda.synchronize()
        if tuple(out.shape) != (length, num_heads, _HEAD_DIM) or out.dtype != torch.bfloat16:
            raise AssertionError(f"tensorfold chain returned {tuple(out.shape)} {out.dtype}")
        expected_out, expected_state = reference(torch, *operands, expected_state)
        if not (torch.isfinite(out).all() and torch.isfinite(final).all()):
            raise AssertionError("tensorfold chain returned non-finite values")
        worst_out = max(worst_out, _rmse_ratio(torch, out, expected_out))
        worst_state = max(worst_state, _rmse_ratio(torch, final, expected_state))
        state, lo = final, hi
    if not (worst_out < OUTPUT_RMSE_TOL and worst_state < STATE_RMSE_TOL):
        raise AssertionError(
            f"tensorfold chain mismatch vs fp32 delta-rule oracle: output rmse ratio "
            f"{worst_out:.2e} (tol {OUTPUT_RMSE_TOL:.0e}), state {worst_state:.2e} "
            f"(tol {STATE_RMSE_TOL:.0e})"
        )
    return worst_out, worst_state


def profile_gdn_chunk_delta_rule_tensorfold_chain(
    *,
    num_tokens: int,
    max_sequence_length: int,
    num_key_heads: int,
    num_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: DType | str,
) -> ComputeMetrics:
    """Profile TensorFold's prefill chain as a prefill step calls it, one call a sequence."""
    args = validate_args(
        num_tokens,
        max_sequence_length,
        num_key_heads,
        num_heads,
        key_head_dim,
        value_head_dim,
        dtype,
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_LABEL}") from exc
    try:
        chain = getattr(importlib.import_module(_MODULE), _CALLABLE)
    except (ImportError, AttributeError) as exc:
        raise ProfilerNotImplemented(
            f"{_LABEL} needs {_MODULE}.{_CALLABLE} (tensorfold_env)"
        ) from exc

    try:
        check_correctness(torch, chain, num_key_heads=args.num_key_heads, num_heads=args.num_heads)
        q, k, v, g, beta = build_operands(
            torch, rows=args.num_tokens, num_key_heads=args.num_key_heads, num_heads=args.num_heads
        )
        boundaries = canonical_sequence_boundaries(args.num_tokens, args.max_sequence_length)
        calls = []
        for index, (lo, hi) in enumerate(zip(boundaries, boundaries[1:])):
            state = new_state(torch, args.num_heads, seed=2 + index)
            calls.append(
                (
                    q[lo:hi],
                    k[lo:hi],
                    v[lo:hi],
                    g[lo:hi],
                    beta[lo:hi],
                    state,
                    torch.empty_like(state),
                )
            )

        # Each call reads its own state and overwrites its own `final`, so
        # repeated launches need no reset.
        def kernel() -> Any:
            return [chain(*call) for call in calls]

        return _common.measure(
            kernel,
            flops=int(chunk_delta_rule_semantic_flops(args)),
            bytes_accessed=int(chunk_delta_rule_logical_bytes(args)),
        )
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_LABEL} ran out of CUDA memory") from exc
    except (ProfilerNotImplemented, ValueError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_LABEL} failed: {exc}") from exc
