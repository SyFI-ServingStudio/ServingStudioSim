"""Element-wise kernel kind.

All Python-side per-kernel knowledge for ``elementwise`` lives here: the wire
string ``KIND``, the ``ElementwiseArgs`` schema, and the ``register(...)`` call
that wires this kernel into ``profiling.db.registry``.

Wire string: ``"elementwise"`` — matches Rust ``KernelSpec::KIND`` in
``simulator/src/timing/kernels/elementwise.rs`` and the Python facade stem used
by ``profiling.facade`` to generate ``get_elementwise_times`` /
``count_missing_elementwise``.

Model (mirrors ``ref/profile/elementwise/elementwise_triton.py``): a generic
byte-level fan-in elementwise/reduce keyed by *total* ``input_size_bytes`` ->
``output_size_bytes`` (dtype-agnostic, ``uint8``). It covers MoE activation
(2N->N), local MoE reduce (xN->N), copy (N->N), and zero-fill (0->N). The
profiler keys directly by total bytes because the simulator already resolves
token batch shapes.

Shape split: the args here are the TOTAL byte sizes (the DB key and runner
kwargs). On the Rust side those totals are produced by folding a per-token byte
rate (static config) with ``num_tokens`` (the runtime sweep axis,
``Cache1DLinear``) — see the Rust ``enumerate``. So ``num_tokens`` and the
per-token rates never appear in this wire schema.

Importing this module has a side effect: it appends a ``KernelProfilerSpec``
row to the registry. The runner module
``profiling.runners.elementwise.triton`` is referenced lazily via ``RunnerRef``
so the main process never eager-imports torch/triton/cuda.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import KernelArgs
from profiling.db.doc import CUPTI_METHOD, BackendDoc, KernelDoc, arg
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND: str = "elementwise"


@dataclass(frozen=True)
class ElementwiseArgs(KernelArgs):
    input_size_bytes: int = arg(unit="bytes", doc="Total input bytes for the operation.")
    output_size_bytes: int = arg(unit="bytes", doc="Total output bytes for the operation.")


DOC = KernelDoc(
    title="Elementwise byte pass",
    summary=("Read a whole multiple of the output size, or nothing, and write the output."),
    description=(
        "The simulator prices small memory-bound steps by their bytes alone: a "
        "MoE activation reads 2N bytes and writes N, a reduction over k inputs "
        "reads kN, a copy reads N, and a zero fill reads nothing. The byte counts "
        "cover the whole batch. The triton and torch backends work on uint8 "
        "buffers and round the input-to-output ratio to a whole fan-in; the "
        "torch_strided_copy and torch_cat backends price one-input bf16 copies only. "
        "The tensorfold_* backends each time one fixed TensorFold statement, so the "
        "statement fixes the input-to-output ratio (a bf16 -> fp32 cast reads half its "
        "output, rotary tables read nothing) and other ratios are refused."
    ),
    category="Other",
    formula=(
        "fan-in = max(1, ⌊input_size_bytes / output_size_bytes + 0.5⌋), or 0 for zero-fill",
        "GB/s = (input_size_bytes + output_size_bytes) / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} Every backend runs five warm-ups before timing. Triton "
        "counts only the elementwise_fan_kernel or elementwise_zero_kernel launch; "
        "the eager backends count every launch of their PyTorch operation."
    ),
    caveats=(
        "The backends match byte counts, not values: Triton sums each fan-in, "
        "while torch uses bitwise_not for one input or amax for multiple inputs.",
        "GB/s uses the requested input bytes, although the rounded fan-in can "
        "make the buffer actually read larger or smaller.",
        "The Triton kernel picks its block size by autotuning once in each "
        "profiling process, so rows measured in different processes can use "
        "different configurations.",
    ),
    # Neither backend uses a separate PyTorch reference implementation.
    reference=None,
)


# Two realizations of the one byte contract, and the choice is load-bearing at
# small sizes where both kernels are launch-bound: `triton` for a slot the
# framework hands to a Triton kernel (including torch-compile output), `torch`
# for a slot whose framework source is eager tensor arithmetic and so lands on
# TensorIterator. Picking `triton` for a torch-realized slot under-predicted a
# measured vLLM shared-expert gate application by 71.5%. A fused CUDA op such as
# vLLM's `act_and_mul_kernel` is neither, and belongs in its own kernel kind.
#
# Two more eager realizations exist only for a one-input copy (input bytes ==
# output bytes) and copy bf16 elements: `torch_strided_copy` for a copy whose
# source is a non-contiguous view, which TensorIterator cannot vectorize and so
# runs the legacy `elementwise_kernel` at about half the unary map's rate, and
# `torch_cat` for `torch.cat`, which runs `CatArrayBatchedCopy`. Pricing a
# vLLM Qwen3.8-27B strided core-output copy with the unary map under-predicted
# it by 55% at 7.8k tokens, and its cat over-predicted by 54%.
_SUMMARIES = {
    "triton": "Triton kernels: elementwise_fan_kernel sums the fan-in uint8 "
    "inputs, elementwise_zero_kernel fills zeros.",
    "torch": "Eager PyTorch: bitwise_not for fan-in 1, amax over the inputs for "
    "more, zero_ for a fill.",
    "torch_strided_copy": "Eager PyTorch copy_ from a non-contiguous bf16 view "
    "(two planes of 256-byte rows transposed): the legacy direct_copy "
    "elementwise_kernel. Copies only.",
    "torch_cat": "Eager torch.cat of two bf16 halves: CatArrayBatchedCopy. Copies only.",
}

# Each eager realization is also registered with a `_vllm` suffix: the same
# runner in vLLM's pinned environment (its Torch build), for a model whose
# glue rows must come from the serving build.
_EAGER = (
    ("torch", "profiling.runners.elementwise.torch", "profile_elementwise"),
    (
        "torch_strided_copy",
        "profiling.runners.elementwise.torch",
        "profile_elementwise_strided_copy",
    ),
    ("torch_cat", "profiling.runners.elementwise.torch", "profile_elementwise_cat"),
)
for _backend, _runner_module, _function, _env in (
    ("triton", "profiling.runners.elementwise.triton", "profile_elementwise", None),
    *((b, m, f, None) for b, m, f in _EAGER),
    *((f"{b}_vllm", m, f, "vllm_env") for b, m, f in _EAGER),
):
    _summary = _SUMMARIES[_backend.removesuffix("_vllm")]
    if _env is not None:
        _summary += " Run in vLLM's pinned environment."
    register(
        KernelProfilerSpec(
            kernel_kind=KIND,
            backend=_backend,
            # Byte-keyed / uint8 — dtype-agnostic.
            supports=BackendSupport(compute=None),
            runner_ref=RunnerRef(
                module_name=_runner_module,
                function_name=_function,
            ),
            table_name=KIND,
            args_schema=ElementwiseArgs,
            metric_family=MetricFamily.COMPUTE,
            batch_outlier_policy=BatchOutlierPolicy(),
            doc=BackendDoc(
                summary=_summary,
            ),
            subprocess_env=_env,
        )
    )


# TensorFold (perf-study 6469730a) byte passes, all in TensorFold's own Torch
# 2.14.1+cu130 (tensorfold_env). The first three reuse the eager runners above
# because the production statement is the same Torch call; the rest each time
# one fixed TensorFold statement (profiling/runners/elementwise/tensorfold.py).
# Torch's kernels have no capability floor; `_paths` is plain Triton (sm_80).
_TENSORFOLD = (
    (
        "tensorfold_torch",
        "profiling.runners.elementwise.torch",
        "profile_elementwise",
        "Eager PyTorch byte map (bitwise_not, amax, zero_) for TensorFold's small "
        "index and position launches (arange, int adds).",
    ),
    (
        "tensorfold_torch_cat",
        "profiling.runners.elementwise.torch",
        "profile_elementwise_cat",
        "torch.cat of contiguous bf16 inputs (CatArrayBatchedCopy_vectorized): conv-state, "
        "commit, tap and sampling-row cats. GB10 A/B: 2, 8 or 48 inputs and any cat dim "
        "time the same at equal bytes.",
    ),
    (
        "tensorfold_torch_strided_copy",
        "profiling.runners.elementwise.torch",
        "profile_elementwise_strided_copy",
        "copy_ from a non-contiguous bf16 view (elementwise_kernel direct_copy): the "
        "drafter's window trims kc[:, -window:].contiguous() and reshape copies.",
    ),
    (
        "tensorfold_torch_cast",
        "profiling.runners.elementwise.tensorfold",
        "profile_elementwise_cast",
        "x.float() of bf16 (input:output 1:2, unrolled_elementwise_kernel) or "
        "x.to(torch.bfloat16) of fp32 (2:1): logits and index casts.",
    ),
    (
        "tensorfold_torch_fp32_add",
        "profiling.runners.elementwise.tensorfold",
        "profile_elementwise_fp32_add",
        "(x.float() + y.float()).to(torch.bfloat16) of two bf16 tensors (2:1, four "
        "launches): the DFlash2 tap at each tapped target layer.",
    ),
    (
        "tensorfold_torch_foreach_copy",
        "profiling.runners.elementwise.tensorfold",
        "profile_elementwise_foreach_copy",
        "torch._foreach_copy_ of bf16 tensors (one multi_tensor_apply_kernel): the "
        "commit's conv-window and KV-row copies.",
    ),
    (
        "tensorfold_torch_cat_strided",
        "profiling.runners.elementwise.tensorfold",
        "profile_elementwise_cat_strided",
        "torch.cat of the DFlash2 head's [rows, 98304] span and a strided [rows, 288] "
        "column slice (unvectorized CatArrayBatchedCopy); whole 98592-element rows only.",
    ),
    (
        "tensorfold_torch_rotary",
        "profiling.runners.elementwise.tensorfold",
        "profile_elementwise_rotary",
        "The DFlash2 drafter's _rotary(start, rows): arange, broadcast multiply, cos and "
        "sin into fp32 [rows, 64] tables (input 0, output 512 bytes a row).",
    ),
    (
        "tensorfold_triton_paths",
        "profiling.runners.elementwise.tensorfold",
        "profile_elementwise_paths",
        "The tree-attention plan's Triton _paths: one int32 parent a verify row in, its "
        "128-entry root path and depth out.",
    ),
)
for _backend, _runner_module, _function, _summary in _TENSORFOLD:
    register(
        KernelProfilerSpec(
            kernel_kind=KIND,
            backend=_backend,
            supports=BackendSupport(
                compute=None,
                min_compute_capability=(8, 0) if _backend == "tensorfold_triton_paths" else None,
            ),
            runner_ref=RunnerRef(module_name=_runner_module, function_name=_function),
            table_name=KIND,
            args_schema=ElementwiseArgs,
            metric_family=MetricFamily.COMPUTE,
            batch_outlier_policy=BatchOutlierPolicy(),
            doc=BackendDoc(summary=_summary + " Runs in TensorFold's environment."),
            subprocess_env="tensorfold_env",
        )
    )
