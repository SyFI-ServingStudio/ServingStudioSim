"""TensorFold ``tree_kernel`` as the ``gdn_tree_verify_recurrence`` backend ``tensorfold_cuda``.

This runner only allocates tensors, times the public call, and returns metrics.

It times ``tensorfold.cuda.kernels.gdn.tree(q, k, v, g, beta, plan, table=table)``
(TensorFold perf-study 6469730a), called the way ``multi_tree_forward``
(``families/qwen3_5/cuda/forward.py``) calls it for one GDN layer:

- ``q``/``k`` ``(W, num_qk_heads, 128)`` bf16, ``v`` ``(W, num_value_heads, 128)``
  bf16, ``g``/``beta`` ``(W, num_value_heads)`` fp32, with ``_gdn_pre``'s ranges.
- ``table`` holds one device pointer per stream to its fp32 ``(Hv, Dv, Dk)``
  committed state; the call reads the states and never writes them, so repeated
  launches need no reset.
- ``plan`` is ``gdn.plan`` of the streams' parents lists, built before timing (the
  forward builds it once for every layer).

One call is one ``tree_kernel`` launch; the output ``(W, Hv, Dv)`` bf16 is
allocated inside the call without a launch. The extension JIT-compiles on first
call, which the correctness check absorbs before timing.

Timing reproduces the cache state the forward leaves. ``_gdn_pre`` writes q, k, v,
g and beta in the launch right before ``tree_kernel``, so they sit in L2, while the
committed states (3 MiB a stream) were last touched a whole forward earlier and come
from DRAM. Every block re-reads its rows' q/k/v/g/beta node by node on its serial
path, so whether they hit L2 moves the time by up to ~40%. Each timed call therefore
runs after Timer.cupti's L2 flush, first rewrites the five inputs from saved copies
(the producer's writes), then calls ``gdn.tree``; only ``tree_kernel`` is counted.
On GB10 this matches the first GDN layers of captured verify rounds within ~3%
(1-16 streams), where a fully cold L2 priced them 7-60% high.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention._gdn_common import exact_int
from profiling.runners.attention.gdn_tree_verify_recurrence_reference import (
    SLOT_CLASSES,
    canonical_tree,
    gdn_tree_verify_recurrence_reference,
    min_rows_for_class,
    slot_class_for_parents,
    slot_class_of_count,
)
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics

_LABEL = "gdn_tree_verify_recurrence:tensorfold_cuda"
_MODULE = "tensorfold.cuda.kernels.gdn"
_KERNEL_NAME = "tree_kernel"
# gdn.cu hard-codes DK = 128; the production value heads are 128 wide too.
_HEAD_DIM = 128
# gdn.cpp `tree`: trees of at most 1024 rows a stream (chains any length).
_MAX_TREE_ROWS = 1024
# Output is stored bf16 (one rounding of an fp32 value); the fp32 recurrence itself
# matches the oracle to ~1e-6.
OUTPUT_RMSE_TOL = 1e-2


@dataclass(frozen=True)
class Args:
    batch_size: int
    num_tokens: int
    num_qk_heads: int
    num_value_heads: int
    slot_class: str


def _even(num_tokens: int, batch_size: int) -> list[int]:
    base, extra = divmod(num_tokens, batch_size)
    return [base + (s < extra) for s in range(batch_size)]


def stream_rows(num_tokens: int, batch_size: int, min_rows: int = 1) -> list[int]:
    """Rows per stream, as even as possible with one stream of at least ``min_rows``.

    The first ``num_tokens % batch_size`` streams take one row more. When that leaves
    every stream under ``min_rows`` (the class's seed tree), stream 0 takes
    ``min_rows`` and the others split the rest evenly: a launch is priced by its
    total rows (GB10: ragged and uniform launches of equal rows within ~5%), so the
    lopsided launch stands for every launch of that class and total.
    """
    rows = _even(num_tokens, batch_size)
    if rows[0] >= min_rows or batch_size == 1:
        return rows
    return [min_rows] + _even(num_tokens - min_rows, batch_size - 1)


def validate_args(
    *,
    batch_size: int,
    num_tokens: int,
    num_qk_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: DType | str,
    state_dtype: DType | str,
    slot_class: str,
) -> Args:
    """The public callable's contract plus the canonical trees' reach."""
    args = Args(
        batch_size=exact_int("batch_size", batch_size),
        num_tokens=exact_int("num_tokens", num_tokens),
        num_qk_heads=exact_int("num_qk_heads", num_qk_heads),
        num_value_heads=exact_int("num_value_heads", num_value_heads),
        slot_class=str(slot_class),
    )
    if DType.from_value(dtype) is not DType.BF16 or DType.from_value(state_dtype) is not DType.FP32:
        raise ProfilerNotImplemented(
            f"{_LABEL} measures bf16 activations with fp32 states, got {dtype}/{state_dtype}"
        )
    if (exact_int("key_head_dim", key_head_dim), exact_int("value_head_dim", value_head_dim)) != (
        _HEAD_DIM,
        _HEAD_DIM,
    ):
        raise ProfilerNotImplemented(
            f"{_LABEL} supports key_head_dim = value_head_dim = {_HEAD_DIM}, got "
            f"{key_head_dim}/{value_head_dim}"
        )
    if args.num_qk_heads < 1 or args.num_value_heads % args.num_qk_heads != 0:
        raise ValueError("num_value_heads must be a positive multiple of num_qk_heads")
    if args.batch_size < 1 or args.num_tokens < args.batch_size:
        raise ValueError("every stream needs at least one row: num_tokens >= batch_size >= 1")
    if args.slot_class not in SLOT_CLASSES:
        raise ValueError(f"slot_class must be one of {SLOT_CLASSES}, got {args.slot_class!r}")
    try:
        need = min_rows_for_class(args.slot_class)
    except ValueError as exc:
        raise ProfilerNotImplemented(
            f"{_LABEL} has no canonical tree for {args.slot_class} (beyond 16-row windows)"
        ) from exc
    if args.num_tokens - need < args.batch_size - 1:
        raise ValueError(
            f"{args.slot_class} needs a stream of at least {need} rows and one row for each "
            f"other stream; {args.num_tokens} rows over {args.batch_size} streams fall short"
        )
    most = stream_rows(args.num_tokens, args.batch_size, need)[0]
    if args.slot_class != "chain" and most > _MAX_TREE_ROWS:
        raise ValueError(f"TensorFold runs trees of at most {_MAX_TREE_ROWS} rows a stream")
    return args


def launch_trees(args: Args) -> list[list[int]]:
    """Each stream's canonical tree; a stream too short for the class runs a chain."""
    need = min_rows_for_class(args.slot_class)
    return [
        canonical_tree(args.slot_class if rows >= need else "chain", rows)
        for rows in stream_rows(args.num_tokens, args.batch_size, need)
    ]


def semantic_flops(args: Args) -> int:
    return 7 * args.num_tokens * args.num_value_heads * _HEAD_DIM * _HEAD_DIM


def logical_bytes(args: Args) -> int:
    per_row = 2 * (2 * args.num_qk_heads * _HEAD_DIM + args.num_value_heads * _HEAD_DIM)
    per_row += 2 * 4 * args.num_value_heads  # g, beta
    per_row += 2 * args.num_value_heads * _HEAD_DIM  # output
    states = args.batch_size * args.num_value_heads * _HEAD_DIM * _HEAD_DIM * 4
    return args.num_tokens * per_row + states


def build_operands(
    torch: Any, *, rows: int, streams: int, num_qk_heads: int, num_value_heads: int, seed: int = 0
) -> tuple[Any, ...]:
    """``_gdn_pre``'s outputs for ``rows`` rows and one committed state per stream."""
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def unit(shape: tuple[int, ...]) -> Any:
        x = torch.randn(*shape, generator=gen, device="cuda", dtype=torch.float32)
        return x / x.norm(dim=-1, keepdim=True)

    q = (unit((rows, num_qk_heads, _HEAD_DIM)) * _HEAD_DIM**-0.5).to(torch.bfloat16)
    k = unit((rows, num_qk_heads, _HEAD_DIM)).to(torch.bfloat16)
    v = torch.randn(rows, num_value_heads, _HEAD_DIM, generator=gen, device="cuda").to(
        torch.bfloat16
    )
    # g = exp(-exp(A_log) * softplus(a + dt_bias)) lies in (0, 1); beta = sigmoid(b).
    g = torch.exp(-torch.rand(rows, num_value_heads, generator=gen, device="cuda") * 0.5)
    beta = torch.sigmoid(torch.randn(rows, num_value_heads, generator=gen, device="cuda"))
    states = [
        0.1
        * torch.randn(
            num_value_heads, _HEAD_DIM, _HEAD_DIM, generator=gen, device="cuda", dtype=torch.float32
        )
        for _ in range(streams)
    ]
    return q, k, v, g, beta, states


def _call(gdn: Any, torch: Any, trees: list[list[int]], operands: tuple[Any, ...]) -> Any:
    """Plan and pointer table (setup), then a closure of the production call."""
    q, k, v, g, beta, states = operands
    plan = gdn.plan(trees, q.device)
    table = gdn.to_device(gdn.pointers(states), torch.int64, q.device)
    torch.cuda.synchronize()
    if slot_class_of_count(plan.slots) != slot_class_for_parents(trees):
        raise AssertionError(f"slot rule disagrees with TensorFold's plan ({plan.slots} slots)")

    def kernel() -> Any:
        return gdn.tree(q, k, v, g, beta, plan, table=table)

    return kernel


def _rmse_ratio(got: Any, expected: Any) -> float:
    return float(
        (got.float() - expected).pow(2).mean().sqrt()
        / expected.pow(2).mean().sqrt().clamp_min(1e-12)
    )


def check_correctness(torch: Any, gdn: Any, args: Args) -> float:
    """A small launch of the requested class against the fp32 oracle.

    Three ragged streams: the class's canonical tree plus three chain rows, a single
    row, and the class's tree at its fewest rows, so the launch runs the same
    variant as the timed one and crosses stream boundaries in the table and plan.
    """
    need = min_rows_for_class(args.slot_class)
    trees = [
        canonical_tree(args.slot_class, need + 3),
        [-1],
        canonical_tree(args.slot_class, need),
    ]
    if slot_class_for_parents(trees) != args.slot_class:
        raise AssertionError(f"guard trees are not in {args.slot_class}")
    operands = build_operands(
        torch,
        rows=sum(map(len, trees)),
        streams=len(trees),
        num_qk_heads=args.num_qk_heads,
        num_value_heads=args.num_value_heads,
        seed=1,
    )
    snapshot = [s.clone() for s in operands[5]]
    out = _call(gdn, torch, trees, operands)()
    torch.cuda.synchronize()
    q = operands[0]
    if (
        tuple(out.shape) != (q.shape[0], args.num_value_heads, _HEAD_DIM)
        or out.dtype != torch.bfloat16
    ):
        raise AssertionError(f"tensorfold tree returned {tuple(out.shape)} {out.dtype}")
    if not torch.isfinite(out).all():
        raise AssertionError("tensorfold tree returned non-finite values")
    if any(not torch.equal(a, b) for a, b in zip(operands[5], snapshot, strict=True)):
        raise AssertionError("tensorfold tree wrote a committed state")
    expected = gdn_tree_verify_recurrence_reference(torch, *operands[:5], operands[5], trees)
    error = _rmse_ratio(out, expected)
    if not error < OUTPUT_RMSE_TOL:
        raise AssertionError(
            f"tensorfold tree mismatch vs fp32 delta-rule oracle: output rmse ratio "
            f"{error:.2e} (tol {OUTPUT_RMSE_TOL:.0e})"
        )
    return error


def profile_gdn_tree_verify_recurrence_tensorfold_cuda(
    *,
    batch_size: int,
    num_tokens: int,
    num_qk_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: DType | str,
    state_dtype: DType | str,
    slot_class: str,
) -> ComputeMetrics:
    """Profile one ``tree_kernel`` launch over ``batch_size`` streams' canonical trees."""
    args = validate_args(
        batch_size=batch_size,
        num_tokens=num_tokens,
        num_qk_heads=num_qk_heads,
        num_value_heads=num_value_heads,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        dtype=dtype,
        state_dtype=state_dtype,
        slot_class=slot_class,
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_LABEL}") from exc
    try:
        gdn = importlib.import_module(_MODULE)
        if not callable(getattr(gdn, "tree", None)):
            raise AttributeError("tree")
    except (ImportError, AttributeError) as exc:
        raise ProfilerNotImplemented(f"{_LABEL} needs {_MODULE}.tree (tensorfold_env)") from exc

    try:
        check_correctness(torch, gdn, args)
        trees = launch_trees(args)
        if slot_class_for_parents(trees) != args.slot_class:
            raise AssertionError(f"launch trees are not in {args.slot_class}")
        operands = build_operands(
            torch,
            rows=args.num_tokens,
            streams=args.batch_size,
            num_qk_heads=args.num_qk_heads,
            num_value_heads=args.num_value_heads,
        )
        tree = _call(gdn, torch, trees, operands)
        if not torch.isfinite(tree()).all():
            raise AssertionError("tensorfold tree returned non-finite values")
        inputs = operands[:5]
        produced = [t.clone() for t in inputs]

        # _gdn_pre's writes, then the tree: inputs L2-hot, states cold after the flush.
        def kernel() -> Any:
            for dst, src in zip(inputs, produced, strict=True):
                dst.copy_(src)
            return tree()

        torch.cuda.synchronize()
        time_ms = Timer.cupti(kernel, kernel_name=_KERNEL_NAME)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_LABEL} ran out of CUDA memory") from exc
    except (ProfilerNotImplemented, ValueError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_LABEL} failed: {exc}") from exc

    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        energy_j=float(energy_j),
        tflops=semantic_flops(args) / elapsed_s / 1e12 if time_ms > 0 else 0.0,
        memory_bandwidth_gbps=logical_bytes(args) / elapsed_s / 1e9 if time_ms > 0 else 0.0,
    )
