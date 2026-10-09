"""BF16 NKI dense compute on one Trainium2 LNC2 unit."""

from __future__ import annotations

from profiling.db.args import DType
from profiling.profilers.neuron_timer import measure_nki
from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics


def _validate_shape(m: int, hidden: int, output: int, dtype: DType | str) -> None:
    if DType.from_value(dtype) != DType.BF16:
        raise ProfilerNotImplemented("Neuron dense runners currently support BF16 only")
    if min(m, hidden, output) < 1:
        raise ValueError("Neuron dense dimensions must be positive")
    # qkv_tkg requires H%128=0 and H/128 divisible by LNC=2.
    # Restrict both dispatch phases to the verified aligned model dimensions.
    if hidden % 256 or output % 256:
        raise ValueError("Neuron LNC2 dense feature dimensions must be multiples of 256")


def _metrics(time_ms: float, flops: int, bytes_accessed: int) -> ComputeMetrics:
    seconds = time_ms / 1000
    return ComputeMetrics(
        time_ms=time_ms,
        tflops=flops / seconds / 1e12,
        memory_bandwidth_gbps=bytes_accessed / seconds / 1e9,
        # No supported energy sampler is wired for Neuron yet.
        energy_j=0.0,
    )


def profile_single_gemm(m: int, n: int, k: int, dtype: DType | str) -> ComputeMetrics:
    """Pure matmul specialization of NxDI's public QKV production kernel."""
    _validate_shape(m, k, n, dtype)
    _validate_qkv_rows(m)
    import ml_dtypes
    import numpy as np
    from nkilib.core.qkv.qkv import qkv

    rng = np.random.default_rng(42)
    a = rng.normal(size=(1, m, k)).astype(ml_dtypes.bfloat16)
    weight = (rng.normal(size=(k, n)) / k**0.5).astype(ml_dtypes.bfloat16)
    reference = (a.astype(np.float32) @ weight.astype(np.float32)).astype(ml_dtypes.bfloat16)

    def check(output):
        # NEFF metadata omits the leading singleton batch dimension.
        np.testing.assert_allclose(
            output.reshape(reference.shape).astype(np.float32),
            reference.astype(np.float32), rtol=0.02, atol=0.02,
        )

    time_ms = measure_nki(
        qkv, {"input": a, "fused_qkv_weights": weight}, check,
        output_dtype=ml_dtypes.bfloat16,
    )
    return _metrics(time_ms, 2 * m * n * k, 2 * (m * k + k * n + m * n))


def _validate_qkv_rows(m: int) -> None:
    # The public entry selects CTE above 96 rows. Its default LNC2 configuration
    # exceeds SBUF at H=4096 even when the output width meets CTE's <=4096 cap.
    # This backend covers the verified TKG path; decoder prefill uses NxDI.
    if m > 96:
        raise ProfilerNotImplemented("NKI QKV backend supports TKG with 1..96 rows")


def dense_mlp_reference(torch, hidden, gate_weight, up_weight, down_weight):
    """Independent TKG fused-MLP math with its BF16 rounding boundary.

    TKG keeps gate/up, activation and multiplication in FP32, then rounds the
    product to BF16 before down. The runner supports only one-token Llama8B.
    Source: nkilib/core/mlp/mlp_tkg/mlp_tkg_gate_up_projection*.py, 3b542be2.
    """
    gate = hidden.float() @ gate_weight.float()
    up = hidden.float() @ up_weight.float()
    activated = torch.nn.functional.silu(gate)
    intermediate = (activated * up).to(torch.bfloat16)
    return (intermediate.float() @ down_weight.float()).to(torch.bfloat16)


def _validate_mlp_shape(m: int, hidden: int, intermediate: int, dtype: DType | str) -> None:
    _validate_shape(m, hidden, intermediate, dtype)
    if (m, hidden, intermediate) != (1, 4096, 14336):
        raise ProfilerNotImplemented(
            "NKI fused MLP is validated only for m=1, hidden=4096, intermediate=14336; "
            "full-rank m128 exceeds vendor SBUF limits in both CTE and DECODE modes"
        )


def profile_dense_mlp(
    m: int, hidden: int, intermediate: int, dtype: DType | str
) -> ComputeMetrics:
    """One fused gate/up, SiLU product and down projection; no norm or residual."""
    _validate_mlp_shape(m, hidden, intermediate, dtype)
    import ml_dtypes
    import numpy as np
    import torch
    from nkilib.core.mlp.mlp import mlp
    from nkilib.core.utils.common_types import ActFnType, NormType

    rng = np.random.default_rng(42)
    bf16 = ml_dtypes.bfloat16
    x = rng.normal(size=(1, m, hidden)).astype(bf16)
    gate = (rng.normal(size=(hidden, intermediate)) / hidden**0.5).astype(bf16)
    up = (rng.normal(size=(hidden, intermediate)) / hidden**0.5).astype(bf16)
    down = (rng.normal(size=(intermediate, hidden)) / intermediate**0.5).astype(bf16)
    tensors = [
        torch.from_numpy(value.astype(np.float32)).to(torch.bfloat16)
        for value in (x, gate, up, down)
    ]
    reference = dense_mlp_reference(torch, *tensors).float().numpy()

    def check(output):
        np.testing.assert_allclose(
            output.reshape(reference.shape).astype(np.float32), reference, rtol=0.02, atol=0.02
        )

    time_ms = measure_nki(
        mlp,
        {
            "hidden_tensor": x,
            "gate_proj_weights_tensor": gate,
            "up_proj_weights_tensor": up,
            "down_proj_weights_tensor": down,
            "normalization_type": NormType.NO_NORM,
            "activation_fn": ActFnType.SiLU,
            # The default LNC2 column path holds FP32 gate+up and a same-size
            # receive buffer in one partition: 16*I bytes before weight tiles.
            # The full Llama8B I=14336 exceeds its 200KiB SBUF limit. The public
            # transpose path distributes I across partitions while preserving
            # the same fused math and rounding. No rank/TensorParallel chunking.
            "use_tkg_gate_up_proj_column_tiling": False,
        },
        check,
        output_dtype=bf16,
    )
    # Logical GEMM FLOPs dominate; activation transcendental work is omitted.
    # Traffic counts external BF16 operands, as the intermediate stays on-chip.
    return _metrics(
        time_ms, 6 * m * hidden * intermediate,
        2 * (2 * m * hidden + 3 * hidden * intermediate),
    )
