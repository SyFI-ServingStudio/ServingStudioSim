"""Sliding-window K/V append: rewrite each stream's window with its new rows.

All Python-side per-kernel knowledge for ``sliding_window_kv_append`` lives
here: the wire string ``KIND``, the ``SlidingWindowKvAppendArgs`` schema, and the
``register(...)`` call that wires the kind into ``profiling.db.registry``.

Source: TensorFold (perf-study 6469730a) adding committed target taps to the
DFlash2 drafter's context. ``families/qwen3_5/cuda/dflash2.py``
``add_taps_streams`` projects every stream's new rows once, then for each drafter
layer calls ``draft_attention.append(new, olds, sizes, window)`` twice (keys,
then values). One ``_append`` Triton launch over a (stream, kv head, 64-row
block) grid writes, for every stream, a FRESH contiguous ``(num_kv_heads, keep,
head_dim)`` tensor holding the last ``keep = min(window, context_len +
num_tokens)`` rows of [old context | its new rows]. The old context is copied,
not extended in place, so the cost follows the rows kept, not the rows added.

Cost on GB10 (8 KV heads x 128, bf16, window 2047), one call:

- set by streams and kept rows alone: at 8 streams and 2,047 kept rows it is
  0.26 ms whether 1, 16, 256 or 2,048 rows were new (and with or without an old
  context);
- a bandwidth-bound copy: about 240-260 GB/s of read + write past a few
  hundred kept rows a stream, about 2 us when only a few rows are kept.

Reuse verdict (new kind):

- ``kv_cache_append``: writes ``num_tokens`` new rows into paged slots; its
  cost follows the new rows. This call rewrites each stream's whole window into
  a new tensor, several streams per launch; 1 new row costs as much as 2,048.
- ``indexed_row_copy`` / ``elementwise``: one tensor's rows or bytes, no
  per-stream window clamp; the launch shape (streams x heads x row blocks of
  the longest window) is not a byte count.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import CUPTI_METHOD, BackendDoc, KernelDoc, arg
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND = "sliding_window_kv_append"


@dataclass(frozen=True)
class SlidingWindowKvAppendArgs(KernelArgs):
    num_kv_heads: int = arg(unit="heads", doc="KV heads of the one tensor appended (keys or values).")
    head_dim: int = arg(unit="elements", doc="Elements in each head.")
    dtype: DType = arg(doc="Element type of the context and the new rows.")
    window: int = arg(unit="tokens", doc="Rows a stream's context keeps at most.")
    batch_size: int = arg(unit="streams", doc="Streams appended in the one launch.")
    context_len: int = arg(
        unit="tokens", doc="Rows of each stream's context before the append (0 = no context yet)."
    )
    num_tokens: int = arg(unit="tokens", doc="New rows appended to each stream.")


DOC = KernelDoc(
    title="Sliding-window KV append",
    summary=(
        "Append new key or value rows to each stream's sliding-window context, "
        "rewriting the kept window into a fresh tensor, all streams in one launch."
    ),
    description=(
        "A drafter that attends a sliding window of committed context keeps one "
        "contiguous K and one V tensor per stream and layer. After each round the "
        "newly committed rows are appended: every stream gets a new tensor holding "
        "the last min(window, context_len + num_tokens) rows of its old context "
        "followed by its new rows. Each call handles one tensor (keys or values) for "
        "every stream; the measurement gives every stream the same context_len and "
        "num_tokens."
    ),
    category="Attention",
    subcategory="MHA / GQA",
    formula=(
        "out_s = [context_s ; new_s][-keep:], keep = min(window, context_len + num_tokens)",
        "FLOPs = 0",
        "GB/s = 2 · batch_size · keep · num_kv_heads · head_dim · bytes(dtype) / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "The time is the call's one _append launch. Old contexts and new rows are "
        "random and allocated before timing; the call allocates each stream's output "
        "and copies its small pointer/size table to the device, as production does. "
        "Before timing, every stream's output is checked bit for bit against a "
        "PyTorch concatenate-and-slice reference."
    ),
    caveats=(
        "Every stream has the same context_len and num_tokens; the cost follows the "
        "kept rows, so a ragged batch is priced by its mean kept rows.",
        "Only the kept rows are copied; the new rows that fall out of the window are not read.",
    ),
    reference="profiling.runners.attention.sliding_window_kv_append_reference",
)


# draft_attention._append is Triton (plain bf16 loads and stores, no MMA) launched
# from tensorfold.families.qwen3_5.cuda.draft_attention without TensorFold's CUDA
# extension loader. TensorFold's drafter needs bf16 (sm_80); append() casts
# nothing, and the kernel's pointer table is typed bfloat16, so only bf16 rows are
# measured. tl.arange(0, D) needs a power-of-two head_dim.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="tensorfold_triton",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.BF16}),
            min_compute_capability=(8, 0),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.sliding_window_kv_append_tensorfold",
            function_name="profile_sliding_window_kv_append_tensorfold_triton",
        ),
        table_name=KIND,
        args_schema=SlidingWindowKvAppendArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="tensorfold_env",
        doc=BackendDoc(
            summary=(
                "TensorFold's Triton _append (families/qwen3_5/cuda/draft_attention.py "
                "append), one launch for every stream, as DFlash2 add_taps_streams calls it."
            ),
            url="https://github.com/efeslab/TensorFold",
        ),
    )
)

__all__ = ["KIND", "SlidingWindowKvAppendArgs"]
