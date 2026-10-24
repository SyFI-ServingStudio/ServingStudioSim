"""TensorFold's GDN commit replay as ``gdn_state_replay``.

This runner only allocates tensors, times one public call, and returns
metrics. DB writes, subprocess selection and registry routing live in L1b.

It times ``tensorfold.cuda.kernels.gdn.replay(table, layers, streams, rows,
counts, k0, v0, in_place=True)`` (TensorFold perf-study 6469730a), called the
way ``families/qwen3_5/cuda/forward.py`` ``_commit`` calls it after a verify
round:

- per layer, the verify window's saved ``k`` ``(W, num_qk_heads, 128)`` bf16
  (L2-normalized), ``v`` ``(W, num_value_heads, 128)`` bf16, ``g`` (decay,
  in (0, 1)) and ``beta`` (in (0, 1)) ``(W, num_value_heads)`` fp32, as
  ``glue.gdn_pre`` writes them;
- per stream and layer, a committed fp32 ``(num_value_heads, 128, 128)`` state;
- ``table`` is ``replay_table(...)`` copied to the device, ``rows`` a
  ``(streams, P)`` int32 tensor of each stream's accepted window rows and
  ``counts`` ``(streams,)`` int32, all built before timing as ``_commit`` and
  ``path_indices`` build them.

The timed closure is the call's one ``replay_kernel`` launch (``in_place``
allocates nothing; CUPTI sums every launch, ``kernel_name=None``). The kind's
``num_tokens`` rows split as evenly as possible over ``batch_size`` streams;
each stream's window holds ``max(16, rows)`` rows (a DFlash2 tree has 16, a copy
chain its own length) and its path is the window's first rows. Ragged splits
and paths that skip rows measured within noise of this layout.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.runners.attention import _common
from profiling.runners.attention._gdn_common import exact_int
from profiling.runners.attention.gdn_state_replay_reference import gdn_state_replay_reference
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics

_MODULE = "tensorfold.cuda.kernels.gdn"
_LABEL = "gdn_state_replay:tensorfold_cuda"
# gdn.cu hard-codes DK = 128 (32 lanes x 4) and gdn.py replay_table takes
# (Hv, Dv, 128) states; only 128-wide value heads are measured.
_HEAD_DIM = 128
# A DFlash2 verify window holds 16 rows a stream; longer paths are copy chains.
_TREE_ROWS = 16
# A window holds at most ``attention.MAX_NODES`` = 128 rows (single-stream copy chains).
_MAX_ROWS_PER_STREAM = 128
# Guard batch for the oracle: three streams with ragged paths that skip rows
# (a tree's root-to-leaf path), over two layers, from nonzero states.
_GUARD_PATHS = ((0, 2, 5, 11), (0,), (0, 1, 3, 4, 8, 9, 15))
_GUARD_LAYERS = 2
# Each step is fp32 (--fmad=false); against the fp32 Torch oracle the state
# RMSE ratio is round-off only.
STATE_RMSE_TOL = 1e-4


@dataclass(frozen=True)
class _Args:
    num_layers: int
    batch_size: int
    num_qk_heads: int
    num_value_heads: int
    num_tokens: int


def validate_args(
    num_layers: int,
    batch_size: int,
    num_qk_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: DType | str,
    state_dtype: DType | str,
    num_tokens: int,
) -> _Args:
    """``replay``'s contract: bf16 keys/values, fp32 state, 128-wide heads, 1..128 rows a stream."""

    args = _Args(
        num_layers=exact_int("num_layers", num_layers),
        batch_size=exact_int("batch_size", batch_size),
        num_qk_heads=exact_int("num_qk_heads", num_qk_heads),
        num_value_heads=exact_int("num_value_heads", num_value_heads),
        num_tokens=exact_int("num_tokens", num_tokens),
    )
    if DType.from_value(dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_LABEL} measures bf16 keys and values only, got {dtype}")
    if DType.from_value(state_dtype) is not DType.FP32:
        raise ProfilerNotImplemented(f"{_LABEL} keeps fp32 states only, got {state_dtype}")
    if (exact_int("key_head_dim", key_head_dim), exact_int("value_head_dim", value_head_dim)) != (
        _HEAD_DIM,
        _HEAD_DIM,
    ):
        raise ProfilerNotImplemented(
            f"{_LABEL} supports key_head_dim = value_head_dim = {_HEAD_DIM}, got "
            f"{key_head_dim}/{value_head_dim}"
        )
    if args.num_layers < 1 or args.batch_size < 1:
        raise ValueError("num_layers and batch_size must be >= 1")
    if args.num_qk_heads < 1 or args.num_value_heads % args.num_qk_heads:
        raise ValueError("num_value_heads must be a positive multiple of num_qk_heads")
    if not args.batch_size <= args.num_tokens <= _MAX_ROWS_PER_STREAM * args.batch_size:
        raise ValueError(
            f"{_LABEL}: num_tokens must give every stream 1..{_MAX_ROWS_PER_STREAM} rows, got "
            f"{args.num_tokens} over {args.batch_size} streams"
        )
    return args


def even_split(total: int, streams: int) -> list[int]:
    """``total`` rows over ``streams`` streams, the first ``total % streams`` one longer."""

    base, extra = divmod(total, streams)
    return [base + 1 if s < extra else base for s in range(streams)]


def build_operands(
    torch: Any,
    gen: Any,
    *,
    layers: int,
    streams: int,
    width: int,
    num_qk_heads: int,
    num_value_heads: int,
) -> tuple[list[Any], list[Any], list[Any], list[Any], list[list[Any]]]:
    """Per-layer saved window rows (``gdn_pre``'s dtypes and ranges) and per-stream states."""

    ks, vs, gs, betas = [], [], [], []
    for _ in range(layers):
        k = torch.randn(width, num_qk_heads, _HEAD_DIM, generator=gen, device="cuda")
        ks.append((k / k.norm(dim=-1, keepdim=True)).to(torch.bfloat16))
        vs.append(
            torch.randn(width, num_value_heads, _HEAD_DIM, generator=gen, device="cuda").to(
                torch.bfloat16
            )
        )
        # g = exp(-exp(A_log) * softplus(a + dt_bias)) in (0, 1); beta = sigmoid(b).
        gs.append(
            torch.exp(-torch.rand(width, num_value_heads, generator=gen, device="cuda") * 0.5)
        )
        betas.append(
            torch.sigmoid(torch.randn(width, num_value_heads, generator=gen, device="cuda"))
        )
    states = [
        [
            0.1 * torch.randn(num_value_heads, _HEAD_DIM, _HEAD_DIM, generator=gen, device="cuda")
            for _ in range(layers)
        ]
        for _ in range(streams)
    ]
    return ks, vs, gs, betas, states


def device_paths(torch: Any, paths: Sequence[Sequence[int]]) -> tuple[Any, Any]:
    """``rows`` ``(streams, P)`` int32 (zero-padded) and ``counts`` ``(streams,)`` int32."""

    width = max(len(p) for p in paths)
    rows = torch.tensor(
        [list(p) + [0] * (width - len(p)) for p in paths], dtype=torch.int32, device="cuda"
    )
    counts = torch.tensor([len(p) for p in paths], dtype=torch.int32, device="cuda")
    return rows, counts


def _rmse_ratio(torch: Any, got: Any, expected: Any) -> float:
    return float(
        (got.float() - expected).pow(2).mean().sqrt()
        / expected.pow(2).mean().sqrt().clamp_min(1e-12)
    )


def check_correctness(torch: Any, module: Any, *, num_qk_heads: int, num_value_heads: int) -> float:
    """Replay the guard batch in place and check it; return the worst state RMSE ratio."""

    gen = torch.Generator(device="cuda").manual_seed(1)
    streams = len(_GUARD_PATHS)
    ks, vs, gs, betas, states = build_operands(
        torch,
        gen,
        layers=_GUARD_LAYERS,
        streams=streams,
        width=streams * _TREE_ROWS,
        num_qk_heads=num_qk_heads,
        num_value_heads=num_value_heads,
    )
    # Window rows: stream s owns rows [16 s, 16 s + 16), as multi_tree_forward packs them.
    paths = [[s * _TREE_ROWS + r for r in path] for s, path in enumerate(_GUARD_PATHS)]
    expected = [
        [
            gdn_state_replay_reference(torch, ks[i], vs[i], gs[i], betas[i], states[s][i], paths[s])
            for i in range(_GUARD_LAYERS)
        ]
        for s in range(streams)
    ]
    table = module.to_device(
        module.replay_table(ks, vs, gs, betas, states), torch.int64, torch.device("cuda")
    )
    rows, counts = device_paths(torch, paths)
    module.replay(table, _GUARD_LAYERS, streams, rows, counts, ks[0], vs[0], in_place=True)
    torch.cuda.synchronize()
    worst = 0.0
    for s in range(streams):
        for i in range(_GUARD_LAYERS):
            got = states[s][i]
            if not torch.isfinite(got).all():
                raise AssertionError("tensorfold replay wrote non-finite states")
            worst = max(worst, _rmse_ratio(torch, got, expected[s][i]))
    if not worst < STATE_RMSE_TOL:
        raise AssertionError(
            f"tensorfold replay mismatch vs fp32 delta-rule oracle: state rmse ratio {worst:.2e} "
            f"(tol {STATE_RMSE_TOL:.0e})"
        )
    return worst


def work(args: _Args) -> tuple[int, int]:
    """FLOPs and logical bytes of one call (the kind's DOC formula)."""

    e1 = args.num_value_heads * _HEAD_DIM * _HEAD_DIM
    flops = 6 * args.num_layers * args.num_tokens * e1
    state_bytes = 2 * args.batch_size * e1 * 4
    row_bytes = args.num_tokens * (
        (args.num_qk_heads + args.num_value_heads) * _HEAD_DIM * 2 + 2 * args.num_value_heads * 4
    )
    return flops, args.num_layers * (state_bytes + row_bytes)


def measure_counts(
    torch: Any,
    module: Any,
    *,
    counts: Sequence[int],
    num_layers: int,
    num_qk_heads: int,
    num_value_heads: int,
) -> ComputeMetrics:
    """Time one in-place replay of ``counts[s]`` accepted rows for each stream ``s``.

    The profile row's even split and the cache-fidelity tool's ragged probes
    both measure through here, so they share one timing boundary.
    """

    windows = [max(_TREE_ROWS, c) for c in counts]
    gen = torch.Generator(device="cuda").manual_seed(0)
    ks, vs, gs, betas, states = build_operands(
        torch,
        gen,
        layers=num_layers,
        streams=len(counts),
        width=sum(windows),
        num_qk_heads=num_qk_heads,
        num_value_heads=num_value_heads,
    )
    starts = [sum(windows[:s]) for s in range(len(counts))]
    rows, counts_t = device_paths(
        torch, [[start + r for r in range(c)] for start, c in zip(starts, counts)]
    )
    table = module.to_device(
        module.replay_table(ks, vs, gs, betas, states), torch.int64, torch.device("cuda")
    )
    torch.cuda.synchronize()

    # In place, as multi.py commits: every call folds the same rows into the
    # states again; the update is contractive, so they stay finite.
    def kernel() -> Any:
        return module.replay(
            table, num_layers, len(counts), rows, counts_t, ks[0], vs[0], in_place=True
        )

    flops, bytes_accessed = work(
        _Args(
            num_layers=num_layers,
            batch_size=len(counts),
            num_qk_heads=num_qk_heads,
            num_value_heads=num_value_heads,
            num_tokens=sum(counts),
        )
    )
    return _common.measure(kernel, flops=flops, bytes_accessed=bytes_accessed)


def profile_gdn_state_replay_tensorfold_cuda(
    *,
    num_layers: int,
    batch_size: int,
    num_qk_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: DType | str,
    state_dtype: DType | str,
    num_tokens: int,
) -> ComputeMetrics:
    args = validate_args(
        num_layers,
        batch_size,
        num_qk_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim,
        dtype,
        state_dtype,
        num_tokens,
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_LABEL}") from exc
    try:
        module = importlib.import_module(_MODULE)
        module.replay, module.replay_table, module.to_device  # noqa: B018
    except (ImportError, AttributeError) as exc:
        raise ProfilerNotImplemented(f"{_LABEL} needs {_MODULE} (tensorfold_env)") from exc

    heads = dict(num_qk_heads=args.num_qk_heads, num_value_heads=args.num_value_heads)
    try:
        check_correctness(torch, module, **heads)
        return measure_counts(
            torch,
            module,
            counts=even_split(args.num_tokens, args.batch_size),
            num_layers=args.num_layers,
            **heads,
        )
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_LABEL} ran out of CUDA memory") from exc
    except (ProfilerNotImplemented, ValueError, TypeError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_LABEL} failed: {exc}") from exc
