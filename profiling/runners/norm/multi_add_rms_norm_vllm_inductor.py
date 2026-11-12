"""Profile vLLM's native summed RMSNorm (+ static FP8 quant) as Inductor compiles it.

The block is vLLM's own code: the ``native`` implementation of the
``fused_add_rms_norm`` / ``rms_norm`` IR ops (the implementation vLLM's IR
lowering inlines into the compiled graph) and ``QuantFP8.forward_native`` for
the static per-tensor quant of the next linear layer's input. A sum of more
than two addends is how vLLM's graph reaches this kernel: the preceding
norm's residual sum is consumed only here, so Inductor re-adds its two
addends instead of storing it. Chaining the IR op reproduces that kernel; the
inner norms' unused outputs are dead code.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics
from profiling.runners.norm.multi_add_rms_norm_reference import multi_add_rms_norm_reference
from profiling.runners.vllm_inductor import compile_like_vllm

_BACKEND = "multi_add_rms_norm:vllm_inductor"
# Runtime constants: neither changes the generated kernel's shape or traffic.
_EPS = 1e-6
_FP8_SCALE = 1.0 / 16.0


def _validate_args(
    num_tokens: int,
    hidden_dim: int,
    num_addends: int,
    store_sum: bool,
    store_normed: bool,
    store_fp8: bool,
    dtype: DType | str,
) -> None:
    for name, value in (
        ("num_tokens", num_tokens),
        ("hidden_dim", hidden_dim),
        ("num_addends", num_addends),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if store_sum and num_addends < 2:
        raise ValueError("store_sum needs at least two addends; one addend is its own sum")
    if not (store_normed or store_fp8):
        raise ValueError("at least one of store_normed and store_fp8 must be set")
    if DType.from_value(dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_BACKEND} requires dtype=bf16, got {dtype}")


def _build_block(store_sum: bool, store_normed: bool, store_fp8: bool, num_addends: int) -> Any:
    from vllm import ir
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

    fused_add_rms_norm = ir.ops.fused_add_rms_norm.impls["native"].impl_fn
    rms_norm = ir.ops.rms_norm.impls["native"].impl_fn
    # QuantFP8 is a CustomOp and reads the ambient config while it is built.
    with set_current_vllm_config(VllmConfig()):
        quant = QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR, compile_native=False)

    def block(*args: Any) -> tuple[Any, ...]:
        addends, weight, scale = args[:num_addends], args[num_addends], args[num_addends + 1]
        if num_addends == 1:
            total, normed = None, rms_norm(addends[0], weight, _EPS)
        else:
            total = addends[0]
            for addend in addends[1:-1]:
                total = fused_add_rms_norm(addend, total, weight, _EPS)[1]
            normed, total = fused_add_rms_norm(addends[-1], total, weight, _EPS)
        outputs = []
        if store_sum:
            outputs.append(total)
        if store_normed:
            outputs.append(normed)
        if store_fp8:
            outputs.append(quant.forward_native(normed, scale)[0])
        return tuple(outputs)

    return block


def profile_multi_add_rms_norm_vllm_inductor(
    num_tokens: int,
    hidden_dim: int,
    num_addends: int,
    store_sum: bool,
    store_normed: bool,
    store_fp8: bool,
    dtype: DType | str,
) -> ComputeMetrics:
    _validate_args(num_tokens, hidden_dim, num_addends, store_sum, store_normed, store_fp8, dtype)
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires the pinned vLLM environment") from exc
    if store_fp8 and torch.cuda.get_device_capability() < (8, 9):
        raise ProfilerNotImplemented(f"{_BACKEND}: Triton lowers FP8 E4M3 only on SM89+")
    try:
        block = _build_block(store_sum, store_normed, store_fp8, num_addends)
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires the pinned vLLM environment") from exc

    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(0)
    weight = 1.0 + 0.1 * torch.randn(
        hidden_dim, dtype=torch.bfloat16, device=device, generator=generator
    )
    scale = torch.tensor(_FP8_SCALE, dtype=torch.float32, device=device)

    def make_inputs(tokens: int) -> list[Any]:
        addends = [
            torch.randn(
                (tokens, hidden_dim), dtype=torch.bfloat16, device=device, generator=generator
            )
            for _ in range(num_addends)
        ]
        return [*addends, weight, scale]

    try:
        compiled = compile_like_vllm(
            torch, block, make_inputs, {index: 0 for index in range(num_addends)}, num_tokens
        )
        inputs = make_inputs(num_tokens)
        outputs = iter(compiled(*inputs))
        torch.cuda.synchronize()
        expected = multi_add_rms_norm_reference(torch, inputs[:num_addends], weight, scale, _EPS)
        for stored, reference, tolerance in zip(
            (store_sum, store_normed, store_fp8),
            expected,
            # The sum is rounded once; normalized rows differ by a bf16 ulp from the
            # reference's rounding order; FP8 rows by one E4M3 step (2^-3 relative).
            ((1e-2, 1e-2), (2e-2, 2e-2), (0.13, 2e-2)),
            strict=True,
        ):
            if stored:
                torch.testing.assert_close(
                    next(outputs).float(),
                    reference.float(),
                    rtol=tolerance[0],
                    atol=tolerance[1],
                )

        def launch() -> None:
            compiled(*inputs)

        time_ms = Timer.cupti(launch, kernel_name=None)
        energy_j = Energy.perf(launch, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {exc}") from exc

    element_bytes = 2
    row_bytes = hidden_dim * (
        num_addends * element_bytes
        + element_bytes * store_sum
        + element_bytes * store_normed
        + store_fp8
    )
    logical_bytes = num_tokens * row_bytes + hidden_dim * element_bytes
    seconds = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(logical_bytes / seconds / 1e9),
        energy_j=float(energy_j),
    )


__all__ = ["profile_multi_add_rms_norm_vllm_inductor"]
