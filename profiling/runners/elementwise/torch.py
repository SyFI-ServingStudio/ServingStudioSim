"""Torch TensorIterator element-wise runner.

This file is L1a-only: it allocates tensors, times one kernel, and returns
metrics. DB writes, JIT policy, subprocess selection, and registry routing all
live in L1b.

Same byte contract as ``profiling.runners.elementwise.triton``
(``input_size_bytes`` -> ``output_size_bytes``, dtype-agnostic ``uint8``), a
different *realization*. Eager PyTorch expresses this contract through
TensorIterator, so the device kernel is one of

* ``at::native::vectorized_elementwise_kernel`` for fan-in 1 (a unary map) and
  for zero-fill,
* ``at::native::reduce_kernel`` for fan-in >= 2 (a reduce along the fan axis),

which is exactly what a vLLM server launches wherever the model source writes
plain tensor arithmetic instead of calling a fused CUDA/Triton op. The two
realizations are not interchangeable at small sizes: against a measured vLLM
Qwen3.6-35B-A3B-FP8 capture the Triton curve under-predicted the torch-realized
shared-expert gate application by 71.5% and its sigmoid by 64.5%, because at a
few hundred bytes per token both kernels are pure launch overhead and torch's
is the heavier one.

Choose this backend for a slot whose framework source is eager tensor
arithmetic; keep ``triton`` for slots the framework hands to a Triton kernel
(including torch-compile output), and use a dedicated kernel kind for a fused
CUDA op such as vLLM's ``act_and_mul_kernel``.

Torch is imported lazily inside the function, matching the other torch runners:
this module is only ever imported in the profiling worker subprocess (via
``RunnerRef``), never in the main process.
"""

from __future__ import annotations

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics


def _rounded_fan_in(input_size_bytes: int, output_size_bytes: int) -> int:
    """Fan-in the triton runner would use, so both backends key the same shape."""
    return max(1, int(input_size_bytes / output_size_bytes + 0.5))


def profile_elementwise(
    input_size_bytes: int,
    output_size_bytes: int,
) -> ComputeMetrics:
    input_size_bytes = int(input_size_bytes)
    output_size_bytes = int(output_size_bytes)
    if input_size_bytes < 0:
        raise ValueError("input_size_bytes must be >= 0")
    if output_size_bytes <= 0:
        raise ValueError("output_size_bytes must be > 0")

    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented("torch is required for the torch elementwise runner") from exc

    try:
        output_tensor = torch.empty(output_size_bytes, dtype=torch.uint8, device="cuda")
        fan_in = (
            0
            if input_size_bytes == 0
            else _rounded_fan_in(input_size_bytes, output_size_bytes)
        )
        input_tensor = torch.empty(
            max(fan_in, 1) * output_size_bytes, dtype=torch.uint8, device="cuda"
        )

        if fan_in == 0:
            # Zero-fill: TensorIterator's FillFunctor, one write pass.
            def run_once():
                output_tensor.zero_()

        elif fan_in == 1:
            # Unary map. `bitwise_not` is chosen because it is a genuine
            # TensorIterator unary that stays in uint8 — no dtype promotion, so
            # the measured bytes match the requested contract exactly.
            def run_once():
                torch.bitwise_not(input_tensor, out=output_tensor)

        else:
            # Fan-in reduce. `amax` keeps the uint8 output dtype (`sum` would
            # promote and silently change the write width), and dispatches to
            # the same `reduce_kernel` a framework-side reduction lands on.
            fan_view = input_tensor.view(fan_in, output_size_bytes)

            def run_once():
                torch.amax(fan_view, dim=0, out=output_tensor)

        # No autotune to warm out of the capture window (unlike the Triton
        # runner), but torch still resolves its TensorIterator config and any
        # lazy allocator growth on the first launch, so a short warmup keeps
        # those off the measurement.
        time_ms = Timer.cupti(run_once, warmup=5, kernel_name=None)
        energy_j = Energy.perf(run_once, warmup=5, per_iter_time_ms=time_ms)

        # Bandwidth-bound byte mover (read input + write output), same
        # accounting as the triton runner so the two curves stay comparable.
        bytes_accessed = input_size_bytes + output_size_bytes
        bandwidth_gbps = (bytes_accessed / (time_ms / 1000.0)) / 1e9 if time_ms > 0 else 0.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=0.0,
            memory_bandwidth_gbps=float(bandwidth_gbps),
            energy_j=float(energy_j),
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc


def _copy_elements(input_size_bytes: int, output_size_bytes: int, realization: str) -> int:
    """BF16 element count of a one-input copy; other fan-ins have no such realization."""
    input_size_bytes = int(input_size_bytes)
    output_size_bytes = int(output_size_bytes)
    if output_size_bytes <= 0 or input_size_bytes != output_size_bytes:
        raise ValueError(f"{realization} is a copy: input_size_bytes must equal output_size_bytes > 0")
    if output_size_bytes % 2:
        raise ValueError(f"{realization} copies bf16 elements: output_size_bytes must be even")
    return output_size_bytes // 2


def _time_copy(run_once, output_size_bytes: int) -> ComputeMetrics:
    time_ms = Timer.cupti(run_once, warmup=5, kernel_name=None)
    energy_j = Energy.perf(run_once, warmup=5, per_iter_time_ms=time_ms)
    bytes_accessed = 2 * output_size_bytes
    bandwidth_gbps = (bytes_accessed / (time_ms / 1000.0)) / 1e9 if time_ms > 0 else 0.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(bandwidth_gbps),
        energy_j=float(energy_j),
    )


# Row length of the strided copy. A contiguous run of 128 bf16 elements is one
# attention head row; the probe in the Qwen3.8-27B alignment (head-major
# [H, T, 128] -> token-major [T, H, 128]) showed the two-plane transpose below
# reproduces that copy's curve at every size, so the number of planes does not
# matter, only that the source is non-contiguous with 256-byte runs.
_STRIDED_RUN_ELEMENTS = 128


def profile_elementwise_strided_copy(
    input_size_bytes: int,
    output_size_bytes: int,
) -> ComputeMetrics:
    """A copy whose source is a non-contiguous view (``dst.copy_(src)`` or
    ``src.contiguous()`` of a transposed/sliced tensor).

    TensorIterator cannot vectorize it, so the device kernel is the legacy
    ``at::native::elementwise_kernel<128, 4, gpu_kernel_impl_nocast<direct_copy>>``,
    which moves bytes at roughly half the rate of the vectorized unary map the
    ``torch`` backend times. The source is two planes of 256-byte rows transposed
    into the destination; a byte count that does not split into whole rows on both
    planes falls back to an element-stride-2 view, the same kernel.
    """
    elements = _copy_elements(input_size_bytes, output_size_bytes, "strided copy")
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented("torch is required for the strided copy runner") from exc
    try:
        run_elements = _STRIDED_RUN_ELEMENTS
        rows = elements // run_elements
        if elements % run_elements == 0 and rows % 2 == 0:
            source = torch.empty(2, rows // 2, run_elements, dtype=torch.bfloat16, device="cuda")
            source.normal_()
            source = source.transpose(0, 1)
            destination = torch.empty(rows // 2, 2, run_elements, dtype=torch.bfloat16, device="cuda")
        else:
            # Every other element of a buffer twice the size: same kernel, any count.
            pitched = torch.empty(elements, 2, dtype=torch.bfloat16, device="cuda")
            pitched.normal_()
            source = pitched[:, 0]
            destination = torch.empty(elements, dtype=torch.bfloat16, device="cuda")
        if source.is_contiguous():
            raise AssertionError("strided copy source must be non-contiguous")

        def run_once():
            destination.copy_(source)

        run_once()
        torch.cuda.synchronize()
        if not torch.equal(destination, source):
            raise AssertionError("strided copy wrote the wrong values")
        return _time_copy(run_once, int(output_size_bytes))
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc


def profile_elementwise_cat(
    input_size_bytes: int,
    output_size_bytes: int,
) -> ComputeMetrics:
    """``torch.cat`` of two inputs that together are the output bytes.

    Eager ``cat`` lands on ``CatArrayBatchedCopy`` (vectorized when aligned), a
    different kernel from the unary map, and noticeably faster per byte at large
    sizes. The bytes split into two near-equal bf16 inputs.
    """
    elements = _copy_elements(input_size_bytes, output_size_bytes, "cat")
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented("torch is required for the cat runner") from exc
    try:
        first_elements = max(elements // 2, 1)
        first = torch.empty(first_elements, dtype=torch.bfloat16, device="cuda")
        second = torch.empty(elements - first_elements, dtype=torch.bfloat16, device="cuda")
        first.normal_()
        second.normal_()
        destination = torch.empty(elements, dtype=torch.bfloat16, device="cuda")

        def run_once():
            torch.cat([first, second], out=destination)

        run_once()
        torch.cuda.synchronize()
        if not torch.equal(destination[:first_elements], first):
            raise AssertionError("cat wrote the wrong values")
        return _time_copy(run_once, int(output_size_bytes))
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
