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
from profiling.db.doc import CUPTI_METHOD, ROCPROF_METHOD, BackendDoc, KernelDoc, arg
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
        "cover the whole batch. Both backends work on uint8 buffers and round the "
        "input-to-output ratio to a whole fan-in."
    ),
    category="Other",
    formula=(
        "fan-in = max(1, ⌊input_size_bytes / output_size_bytes + 0.5⌋), or 0 for zero-fill",
        "GB/s = (input_size_bytes + output_size_bytes) / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} Both backends run five warm-ups before timing. Triton "
        "counts only the elementwise_fan_kernel or elementwise_zero_kernel launch; "
        "torch counts every launch of its selected PyTorch operation."
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
for _backend, _runner_module in (
    ("triton", "profiling.runners.elementwise.triton"),
    ("torch", "profiling.runners.elementwise.torch"),
):
    register(
        KernelProfilerSpec(
            kernel_kind=KIND,
            backend=_backend,
            # Byte-keyed / uint8 — dtype-agnostic.
            supports=BackendSupport(compute=None),
            runner_ref=RunnerRef(
                module_name=_runner_module,
                function_name="profile_elementwise",
            ),
            table_name=KIND,
            args_schema=ElementwiseArgs,
            metric_family=MetricFamily.COMPUTE,
            batch_outlier_policy=BatchOutlierPolicy(),
            doc=BackendDoc(
                summary=(
                    "Triton kernels: elementwise_fan_kernel sums the fan-in uint8 "
                    "inputs, elementwise_zero_kernel fills zeros."
                    if _backend == "triton"
                    else "Eager PyTorch: bitwise_not for fan-in 1, amax over the "
                    "inputs for more, zero_ for a fill."
                ),
            ),
        )
    )


# Eager-PyTorch byte mover on a ROCm/HIP device, timed kernel-only via
# rocprofv3. The measured MI300X anchor for the `elementwise` byte-placeholder
# floor: the GLM-5.3-Flash arch's glue slots already priced as `elementwise`
# (embedding gather, mHC stream expand/contract, MoE input/combine copies) need
# a backend with MI300X rows, and the campaign floors un-measured composed kinds
# onto this curve. Same `torch`-realization byte contract as the CUDA `torch`
# backend, MI300X-gated (gfx942) so it never competes with the NVIDIA rows.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch_rocm",
        # Byte-keyed / uint8 — dtype-agnostic.
        supports=BackendSupport(compute=None, arch_targets=frozenset({"CDNA3"})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.elementwise.torch_rocm",
            function_name="profile_elementwise_torch_rocm",
        ),
        table_name=KIND,
        args_schema=ElementwiseArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_rocm_env",
        doc=BackendDoc(
            summary=(
                "Eager PyTorch byte mover (bitwise_not / amax / zero_) on ROCm/MI300X, "
                f"timed with rocprofv3. {ROCPROF_METHOD}"
            ),
        ),
    )
)
