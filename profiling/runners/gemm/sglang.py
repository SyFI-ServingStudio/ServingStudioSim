"""Runners for SGLang's production BF16 dense-GEMM dispatches."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from pathlib import Path

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_FUSED_A_MAX_TOKENS = 16
_K3_BF16_GEMM_INITIALIZED = False


def _initialize_k3_bf16_gemm() -> None:
    """Match the scheduler's ``auto -> cutedsl`` BF16 GEMM initialization.

    Standalone L1 workers do not normally have SGLang's process-wide
    ``ServerArgs`` context. Publish a minimal valid context, then invoke the
    same public initializer as the production scheduler.
    """
    global _K3_BF16_GEMM_INITIALIZED
    if _K3_BF16_GEMM_INITIALIZED:
        return

    from sglang.srt.layers.quantization.unquant import initialize_bf16_gemm_config
    from sglang.srt.runtime_context import get_exec

    try:
        get_exec().kernel.bf16_gemm_backend
    except (AttributeError, KeyError, ValueError):
        from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 2,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-5,
            "torch_dtype": "bfloat16",
        }
        with tempfile.TemporaryDirectory(prefix="vibesim-sglang-k3-") as config_dir:
            Path(config_dir, "config.json").write_text(json.dumps(config), encoding="utf-8")
            set_global_server_args_for_scheduler(
                ServerArgs(
                    model_path=config_dir,
                    dtype="bfloat16",
                    device="cuda",
                    tp_size=1,
                    load_format="dummy",
                    bf16_gemm_backend="auto",
                    disable_cuda_graph=True,
                    disable_radix_cache=True,
                )
            )

    initialize_bf16_gemm_config()
    _K3_BF16_GEMM_INITIALIZED = True


def _validate_shape(m: int, n: int, k: int) -> None:
    for name, value in (("m", m), ("n", n), ("k", k)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")


def _require_sm100(backend: str) -> None:
    import torch

    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(f"CUDA is required for the {backend} backend")
    if tuple(torch.cuda.get_device_capability()) != (10, 0):
        raise ProfilerNotImplemented(
            "SGLang resolves `bf16_gemm_backend='auto'` to cutedsl only on SM100"
        )


def _measure(
    kernel: Callable[[], object],
    m: int,
    n: int,
    k: int,
    dtype: DType,
    input_elements: int,
) -> ComputeMetrics:
    import torch

    # CuTe-DSL and JIT initialization are not part of a serving invocation.
    kernel()
    torch.cuda.synchronize()
    time_ms = Timer.cupti(kernel, interval_union=True)
    energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    elapsed_s = time_ms / 1000.0
    flops = 2 * m * n * k
    logical_bytes = input_elements * dtype.size_bytes()
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=float(flops / elapsed_s / 1e12 if elapsed_s else 0.0),
        memory_bandwidth_gbps=float(logical_bytes / elapsed_s / 1e9 if elapsed_s else 0.0),
        energy_j=float(energy_j),
    )


def profile_single_gemm_sglang_bf16(
    m: int,
    n: int,
    k: int,
    dtype: DType | str,
) -> ComputeMetrics:
    _validate_shape(m, n, k)
    dtype = DType.from_value(dtype)
    if dtype is not DType.BF16:
        raise ValueError(f"SGLang's BF16 GEMM dispatch requires bf16, got {dtype.value}")

    try:
        import torch
        import torch.nn.functional as functional
        from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import (
            cutedsl_bf16_gemm,
            use_cutedsl_bf16_gemm,
        )
    except ImportError as exc:
        raise ProfilerNotImplemented("the SGLang environment is required") from exc

    _require_sm100("sglang_bf16_auto")
    try:
        activations = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        if use_cutedsl_bf16_gemm(m, n, k):

            def kernel():
                return cutedsl_bf16_gemm(activations, weight, None)

        else:

            def kernel():
                return functional.linear(activations, weight)

        elements = activations.numel() + weight.numel() + m * n
        return _measure(kernel, m, n, k, dtype, elements)
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc


def profile_single_gemm_sglang_fused_a(
    m: int,
    n: int,
    k: int,
    dtype: DType | str,
) -> ComputeMetrics:
    _validate_shape(m, n, k)
    dtype = DType.from_value(dtype)
    if dtype is not DType.BF16:
        raise ValueError(f"SGLang's fused-A GEMM requires bf16, got {dtype.value}")

    try:
        import torch
        import torch.nn.functional as functional
        from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import (
            cutedsl_bf16_gemm,
            use_cutedsl_bf16_gemm,
        )
        from sglang.kernels.ops.gemm.fused_a_gemm import dsv3_fused_a_gemm
    except ImportError as exc:
        raise ProfilerNotImplemented("the SGLang environment is required") from exc

    _require_sm100("sglang_fused_a_auto")
    try:
        activations = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        weight_eligible = n % 16 == 0 and k % 256 == 0
        if weight_eligible and 1 <= m <= _FUSED_A_MAX_TOKENS:
            weight_t = weight.T

            def kernel():
                return dsv3_fused_a_gemm(activations, weight_t)

        elif use_cutedsl_bf16_gemm(m, n, k):

            def kernel():
                return cutedsl_bf16_gemm(activations, weight, None)

        else:

            def kernel():
                return functional.linear(activations, weight)

        elements = activations.numel() + weight.numel() + m * n
        return _measure(kernel, m, n, k, dtype, elements)
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc


def profile_single_gemm_sglang_k3_raw_bf16(
    m: int,
    n: int,
    k: int,
    dtype: DType | str,
) -> ComputeMetrics:
    """Profile K3's eager raw BF16 module-projection path.

    K3's chunked-prefill path calls the same raw projection helper as the
    eager layer. Initialize SGLang's production BF16 dispatcher in this
    standalone worker while retaining the separate prefill cache backend.
    """
    _validate_shape(m, n, k)
    dtype = DType.from_value(dtype)
    if dtype is not DType.BF16:
        raise ValueError(f"K3 raw BF16 GEMM requires bf16, got {dtype.value}")

    try:
        import torch
        import torch.nn.functional as functional
        from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import (
            cutedsl_bf16_gemm,
            use_cutedsl_bf16_gemm,
        )
    except ImportError as exc:
        raise ProfilerNotImplemented("the SGLang environment is required") from exc

    _require_sm100("sglang_k3_raw_bf16")
    try:
        _initialize_k3_bf16_gemm()
        generator = torch.Generator(device="cuda").manual_seed(31)
        activations = torch.randn(m, k, dtype=torch.bfloat16, device="cuda", generator=generator)
        weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda", generator=generator)
        use_cutedsl = use_cutedsl_bf16_gemm(m, n, k)

        def kernel():
            if use_cutedsl:
                return cutedsl_bf16_gemm(activations, weight, None)
            return functional.linear(activations, weight)

        return _measure(
            kernel,
            m,
            n,
            k,
            dtype,
            activations.numel() + weight.numel() + m * n,
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
