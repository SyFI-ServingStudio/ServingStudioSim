"""TensorFold's own byte-pass realizations of the ``elementwise`` contract.

This file is L1a-only: it allocates tensors, times one statement, and returns
metrics. DB writes, subprocess selection and registry routing live in L1b.

TensorFold (perf-study 6469730a) moves bytes around its CUDA kernels with eager
PyTorch statements in its own Torch (2.14.1+cu130), plus one tiny Triton plan
kernel. Each function below times one such statement, keyed like every
``elementwise`` row by the total ``input_size_bytes`` and ``output_size_bytes``;
the ratio between them is fixed by the statement, and a ratio the statement
cannot produce is refused. Every launch of the statement is counted
(``kernel_name=None``) and L2 is flushed before each call (``Timer.cupti``'s
default), because the operands are produced long before the copy in serving.

- ``profile_elementwise_cast``: ``x.float()`` of bf16 (input:output 1:2,
  ``unrolled_elementwise_kernel``) or ``x.to(torch.bfloat16)`` of fp32 (2:1,
  ``vectorized_elementwise_kernel``). Sampling's and the drafter's logits casts,
  the int32 -> int64 index casts (same unrolled cast kernel, same bytes).
- ``profile_elementwise_fp32_add``: ``(x.float() + y.float()).to(torch.bfloat16)``
  of two bf16 tensors (2:1, four launches): the DFlash2 tap at five target layers
  (``forward.py``, ``prefill.py``).
- ``profile_elementwise_foreach_copy``: ``torch._foreach_copy_(dst, src)`` (1:1,
  one ``multi_tensor_apply_kernel``): the commit's conv-window and KV-row copies
  (``forward.py`` ``_commit``). Time follows bytes, not the tensor count, from
  1 KiB tensors up (GB10 A/B: 32, 256 and 768 tensors of the same 23 MiB within
  2%), so the bytes are split into 32 KiB tensors.
- ``profile_elementwise_cat_strided``: ``torch.cat`` along the last dim of a
  contiguous ``[rows, 98304]`` block and a ``[rows, 288]`` column slice of a
  ``[rows, 320]`` tile (1:1): the DFlash2 drafter's vocabulary-span logits
  (``dflash2.py`` ``launch_blocks``). The strided input sends Torch to the
  unvectorized ``CatArrayBatchedCopy``, 1.3-1.8x slower than a contiguous cat of
  the same bytes on GB10. Bytes must be whole 98592-element bf16 rows.
- ``profile_elementwise_rotary``: the drafter's ``_rotary(start, rows)``
  (input 0, output the fp32 cos and sin tables, ``rows x 64`` each): arange,
  broadcast multiply, cos, sin.
- ``profile_elementwise_paths``: the tree-attention plan's Triton ``_paths``
  (input the int32 parent of each verify row, output its int32 root path of
  ``MAX_NODES`` entries plus its depth), one launch a forward.
"""

from __future__ import annotations

import importlib
import types
from collections.abc import Callable
from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BF16 = 2
_FP32 = 4
# The commit copies 61,440-byte conv windows and n x 2,048-byte KV rows.
_FOREACH_TENSOR_BYTES = 32 << 10
# DFlash2's NVFP4 head spans: [0, 98304) whole, and [248032, 248320) as columns
# 32..320 of the 64-row tiles 3875..3880 (dflash2.py `_sub_parts`).
_HEAD_SPAN = 98304
_TAIL_TILE = 320
_TAIL_SPAN = 288
_HEAD_ROW = _HEAD_SPAN + _TAIL_SPAN
# The drafter's rotary half-width: head_dim 128 (Qwen3.8-27B-DFlash2 config).
_ROTARY_HALF = 64
_ROPE_THETA = 10_000_000.0

_DFLASH2 = "tensorfold.families.qwen3_5.cuda.dflash2"
_ATTENTION = "tensorfold.cuda.kernels.attention"


def _sizes(input_size_bytes: int, output_size_bytes: int) -> tuple[int, int]:
    for name, value in (
        ("input_size_bytes", input_size_bytes),
        ("output_size_bytes", output_size_bytes),
    ):
        if isinstance(value, bool) or int(value) != value or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    if output_size_bytes <= 0:
        raise ValueError("output_size_bytes must be > 0")
    return int(input_size_bytes), int(output_size_bytes)


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "torch is required for the TensorFold elementwise runners"
        ) from exc
    return torch


def _measure(run_once: Callable[[], Any], bytes_accessed: int) -> ComputeMetrics:
    time_ms = Timer.cupti(run_once, warmup=5, kernel_name=None)
    energy_j = Energy.perf(run_once, warmup=5, per_iter_time_ms=time_ms)
    seconds = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(bytes_accessed / seconds / 1e9) if time_ms > 0 else 0.0,
        energy_j=float(energy_j),
    )


def _guarded(label: str, body: Callable[[], ComputeMetrics]) -> ComputeMetrics:
    torch = _torch()
    try:
        return body()
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{label} ran out of CUDA memory") from exc
    except (ProfilerNotImplemented, ValueError, AssertionError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{label} failed: {exc}") from exc


def cast_direction(input_size_bytes: int, output_size_bytes: int) -> str:
    """``"to_fp32"`` for bf16 -> fp32 (1:2), ``"to_bf16"`` for fp32 -> bf16 (2:1)."""
    i, o = _sizes(input_size_bytes, output_size_bytes)
    if 2 * i == o and i % _BF16 == 0:
        return "to_fp32"
    if i == 2 * o and o % _BF16 == 0:
        return "to_bf16"
    raise ValueError(
        "a cast reads half (bf16 -> fp32) or twice (fp32 -> bf16) its output bytes, "
        f"got input {i} and output {o}"
    )


def profile_elementwise_cast(input_size_bytes: int, output_size_bytes: int) -> ComputeMetrics:
    direction = cast_direction(input_size_bytes, output_size_bytes)
    torch = _torch()

    def body() -> ComputeMetrics:
        if direction == "to_fp32":
            src = torch.randn(input_size_bytes // _BF16, dtype=torch.bfloat16, device="cuda")

            def run_once():
                return src.float()
        else:
            src = torch.randn(input_size_bytes // _FP32, dtype=torch.float32, device="cuda")

            def run_once():
                return src.to(torch.bfloat16)

        got = run_once()
        torch.cuda.synchronize()
        if got.numel() != src.numel() or not torch.equal(
            got.to(src.dtype), src.to(got.dtype).to(src.dtype)
        ):
            raise AssertionError("the cast wrote the wrong values")
        return _measure(run_once, input_size_bytes + output_size_bytes)

    return _guarded("elementwise:tensorfold_torch_cast", body)


def fp32_add_elements(input_size_bytes: int, output_size_bytes: int) -> int:
    i, o = _sizes(input_size_bytes, output_size_bytes)
    if i != 2 * o or o % _BF16:
        raise ValueError(
            f"the fp32 add reads two bf16 inputs of the output's size, got input {i} and output {o}"
        )
    return o // _BF16


def profile_elementwise_fp32_add(input_size_bytes: int, output_size_bytes: int) -> ComputeMetrics:
    elements = fp32_add_elements(input_size_bytes, output_size_bytes)
    torch = _torch()

    def body() -> ComputeMetrics:
        x = torch.randn(elements, dtype=torch.bfloat16, device="cuda")
        y = torch.randn(elements, dtype=torch.bfloat16, device="cuda")

        def run_once():
            return (x.float() + y.float()).to(torch.bfloat16)

        got = run_once()
        torch.cuda.synchronize()
        # The exact sum rounded to fp32, then to bf16: the production roundings.
        want = (x.double() + y.double()).float().to(torch.bfloat16)
        if not torch.equal(got, want):
            raise AssertionError("the fp32 add wrote the wrong values")
        # The four launches move 2N + 4N (casts), 8N + 4N (add) and 4N + 2N bytes
        # for N elements; GB/s is the logical in/out bytes like every row.
        return _measure(run_once, input_size_bytes + output_size_bytes)

    return _guarded("elementwise:tensorfold_torch_fp32_add", body)


def foreach_tensor_bytes(input_size_bytes: int, output_size_bytes: int) -> list[int]:
    i, o = _sizes(input_size_bytes, output_size_bytes)
    if i != o or o % _BF16:
        raise ValueError(
            f"_foreach_copy_ copies bf16 tensors: input {i} must equal output {o}, even"
        )
    full, rest = divmod(o, _FOREACH_TENSOR_BYTES)
    return [_FOREACH_TENSOR_BYTES] * full + ([rest] if rest else [])


def profile_elementwise_foreach_copy(
    input_size_bytes: int, output_size_bytes: int
) -> ComputeMetrics:
    sizes = foreach_tensor_bytes(input_size_bytes, output_size_bytes)
    torch = _torch()

    def body() -> ComputeMetrics:
        src = [torch.randn(n // _BF16, dtype=torch.bfloat16, device="cuda") for n in sizes]
        dst = [torch.empty_like(t) for t in src]

        def run_once():
            torch._foreach_copy_(dst, src)

        run_once()
        torch.cuda.synchronize()
        if not all(torch.equal(d, s) for d, s in zip(dst, src)):
            raise AssertionError("_foreach_copy_ wrote the wrong values")
        return _measure(run_once, input_size_bytes + output_size_bytes)

    return _guarded("elementwise:tensorfold_torch_foreach_copy", body)


def cat_strided_rows(input_size_bytes: int, output_size_bytes: int) -> int:
    i, o = _sizes(input_size_bytes, output_size_bytes)
    row_bytes = _BF16 * _HEAD_ROW
    if i != o or o % row_bytes:
        raise ValueError(
            f"the drafter-head cat copies whole {_HEAD_ROW}-element bf16 rows: input {i} must "
            f"equal output {o}, a multiple of {row_bytes}"
        )
    return o // row_bytes


def profile_elementwise_cat_strided(
    input_size_bytes: int, output_size_bytes: int
) -> ComputeMetrics:
    rows = cat_strided_rows(input_size_bytes, output_size_bytes)
    torch = _torch()

    def body() -> ComputeMetrics:
        head = torch.randn(rows, _HEAD_SPAN, dtype=torch.bfloat16, device="cuda")
        tile = torch.randn(rows, _TAIL_TILE, dtype=torch.bfloat16, device="cuda")
        parts = [head[:, 0:_HEAD_SPAN], tile[:, _TAIL_TILE - _TAIL_SPAN : _TAIL_TILE]]
        # A single row's slice is contiguous, so Torch vectorizes that cat as it
        # would in production; from two rows on the tail span is strided.

        def run_once():
            return torch.cat(parts, dim=1)

        got = run_once()
        torch.cuda.synchronize()
        if not (
            torch.equal(got[:, :_HEAD_SPAN], parts[0])
            and torch.equal(got[:, _HEAD_SPAN:], parts[1])
        ):
            raise AssertionError("the drafter-head cat wrote the wrong values")
        return _measure(run_once, input_size_bytes + output_size_bytes)

    return _guarded("elementwise:tensorfold_torch_cat_strided", body)


def rotary_rows(input_size_bytes: int, output_size_bytes: int) -> int:
    i, o = _sizes(input_size_bytes, output_size_bytes)
    row_bytes = 2 * _FP32 * _ROTARY_HALF
    if i != 0 or o % row_bytes:
        raise ValueError(
            f"the rotary tables read nothing and write {row_bytes} bytes a row (fp32 cos and sin "
            f"of {_ROTARY_HALF}), got input {i} and output {o}"
        )
    return o // row_bytes


def profile_elementwise_rotary(input_size_bytes: int, output_size_bytes: int) -> ComputeMetrics:
    rows = rotary_rows(input_size_bytes, output_size_bytes)
    torch = _torch()
    try:
        drafter = importlib.import_module(_DFLASH2).DFlash2
    except (ImportError, AttributeError) as exc:
        raise ProfilerNotImplemented(
            f"tensorfold_torch_rotary needs {_DFLASH2}.DFlash2 (tensorfold_env)"
        ) from exc

    def body() -> ComputeMetrics:
        # The two attributes `_rotary` reads, as DFlash2.__init__ builds them.
        owner = types.SimpleNamespace(device=torch.device("cuda"))
        owner.inv_freq = 1.0 / _ROPE_THETA ** (
            torch.arange(_ROTARY_HALF, device="cuda", dtype=torch.float32) * 2 / (2 * _ROTARY_HALF)
        )
        start = 4096

        def run_once():
            return drafter._rotary(owner, start, rows)

        cos, sin = run_once()
        torch.cuda.synchronize()
        phase = (
            torch.arange(start, start + rows, dtype=torch.float64, device="cuda")[:, None]
            * owner.inv_freq.double()
        )
        if (
            cos.shape != (rows, _ROTARY_HALF)
            or (cos.double() - phase.cos()).abs().max() > 1e-2
            or (sin.double() - phase.sin()).abs().max() > 1e-2
        ):
            raise AssertionError("_rotary returned the wrong tables")
        return _measure(run_once, output_size_bytes)

    return _guarded("elementwise:tensorfold_torch_rotary", body)


def paths_rows(input_size_bytes: int, output_size_bytes: int, max_nodes: int) -> int:
    i, o = _sizes(input_size_bytes, output_size_bytes)
    if i % 4 or i == 0 or o != i * (max_nodes + 1):
        raise ValueError(
            f"_paths reads one int32 parent a row and writes {max_nodes} path entries "
            f"plus a depth, "
            f"got input {i} and output {o}"
        )
    return i // 4


def profile_elementwise_paths(input_size_bytes: int, output_size_bytes: int) -> ComputeMetrics:
    torch = _torch()
    try:
        attention = importlib.import_module(_ATTENTION)
        paths_kernel, max_nodes = attention._paths, attention.MAX_NODES
    except (ImportError, AttributeError) as exc:
        raise ProfilerNotImplemented(
            f"tensorfold_triton_paths needs {_ATTENTION}._paths (tensorfold_env)"
        ) from exc
    rows = paths_rows(input_size_bytes, output_size_bytes, max_nodes)

    def body() -> ComputeMetrics:
        # Verify windows of 16 rows a stream (the deployment's), each a chain:
        # a row's parent is the row before it, a window's root has none.
        window = 16
        parents = torch.tensor(
            [-1 if r % window == 0 else r - 1 for r in range(rows)],
            dtype=torch.int32,
            device="cuda",
        )
        paths = torch.empty((rows, max_nodes), dtype=torch.int32, device="cuda")
        depths = torch.empty((rows,), dtype=torch.int32, device="cuda")

        def run_once():
            # `from_packed`'s launch line (cuda/kernels/attention.py).
            paths_kernel[(rows,)](parents, paths, depths, MAXD=max_nodes, num_warps=1)

        run_once()
        torch.cuda.synchronize()
        want = torch.tensor([r % window + 1 for r in range(rows)], dtype=torch.int32, device="cuda")
        if not torch.equal(depths, want):
            raise AssertionError("_paths wrote the wrong depths")
        return _measure(run_once, input_size_bytes + output_size_bytes)

    return _guarded("elementwise:tensorfold_triton_paths", body)


__all__ = [
    "profile_elementwise_cast",
    "profile_elementwise_cat_strided",
    "profile_elementwise_foreach_copy",
    "profile_elementwise_fp32_add",
    "profile_elementwise_paths",
    "profile_elementwise_rotary",
]
