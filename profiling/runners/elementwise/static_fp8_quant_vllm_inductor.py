"""Profile vLLM's native static per-tensor FP8 quant as Inductor compiles it.

The gated variant is the attention output gate in front of the output
projection: ``attn_output * torch.sigmoid(gate)`` from the model, then the
projection's ``QuantFP8.forward_native``. Inductor emits both as one kernel.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.elementwise.static_fp8_quant_reference import static_fp8_quant_reference
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics
from profiling.runners.vllm_inductor import compile_like_vllm

_BACKEND = "static_fp8_quant:vllm_inductor"
# A runtime scalar; its value never changes the kernel.
_FP8_SCALE = 1.0 / 16.0


def _build_block(gated: bool) -> Any:
    import torch
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

    # QuantFP8 is a CustomOp and reads the ambient config while it is built.
    with set_current_vllm_config(VllmConfig()):
        quant = QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR, compile_native=False)

    if gated:

        def gated_block(values: Any, gate: Any, scale: Any) -> Any:
            return quant.forward_native(values * torch.sigmoid(gate), scale)[0]

        return gated_block

    def block(values: Any, scale: Any) -> Any:
        return quant.forward_native(values, scale)[0]

    return block


def profile_static_fp8_quant_vllm_inductor(
    num_tokens: int, hidden_dim: int, gated: bool, dtype: DType | str
) -> ComputeMetrics:
    for name, value in (("num_tokens", num_tokens), ("hidden_dim", hidden_dim)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if DType.from_value(dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_BACKEND} requires dtype=bf16, got {dtype}")
    try:
        import torch

        block = _build_block(gated)
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires the pinned vLLM environment") from exc

    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(0)
    scale = torch.tensor(_FP8_SCALE, dtype=torch.float32, device=device)

    def make_inputs(tokens: int) -> list[Any]:
        rows = [
            torch.randn(
                (tokens, hidden_dim), dtype=torch.bfloat16, device=device, generator=generator
            )
            for _ in range(1 + gated)
        ]
        return [*rows, scale]

    try:
        token_dims = {0: 0, 1: 0} if gated else {0: 0}
        compiled = compile_like_vllm(torch, block, make_inputs, token_dims, num_tokens)
        inputs = make_inputs(num_tokens)
        actual = compiled(*inputs)
        torch.cuda.synchronize()
        expected = static_fp8_quant_reference(torch, inputs[0], inputs[1] if gated else None, scale)
        # One E4M3 step is 2^-3 relative: rounding order may move a value across it.
        torch.testing.assert_close(actual.float(), expected.float(), rtol=0.13, atol=2e-2)

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

    logical_bytes = (2 + 2 * gated + 1) * num_tokens * hidden_dim
    seconds = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(logical_bytes / seconds / 1e9),
        energy_j=float(energy_j),
    )


__all__ = ["profile_static_fp8_quant_vllm_inductor"]
