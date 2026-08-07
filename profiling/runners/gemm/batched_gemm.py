"""Production-layout Torch runners for GLM-5.2 MLA batched GEMMs.

Each timed callable is only its unquantized ``torch.bmm(..., out=...)`` launch.
Q-absorption and V-up layout construction mirror vLLM v0.23.0 commit
``0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`` in
``mla_attention.py::{process_weights_after_loading,_v_up_proj}``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_SUPPORTED_HEAD_COUNTS = frozenset({16, 32, 64})
_SUPPORTED_DTYPE = DType.BF16
_REQUIRED_GPU = "NVIDIA H200"
_Q_HEAD_WIDTH = 256
_QK_NOPE_HEAD_DIM = 192
_KV_LORA_RANK = 512
_V_HEAD_DIM = 256
_PACKED_HEAD_WIDTH = _QK_NOPE_HEAD_DIM + _V_HEAD_DIM
_H200_PADDED_HEADS = 64


@dataclass(frozen=True)
class _QAbsorbOperands:
    q_base: Any
    lhs: Any
    packed_weight: Any
    rhs: Any
    out: Any


@dataclass(frozen=True)
class _VUpOperands:
    attention_base: Any
    attention_output: Any
    lhs: Any
    packed_weight: Any
    rhs: Any
    out_base: Any
    out: Any


def _validate_args(
    num_batches: int,
    m: int,
    n: int,
    k: int,
    dtype: DType | str,
) -> tuple[int, int, int, int, DType]:
    num_batches = int(num_batches)
    m = int(m)
    n = int(n)
    k = int(k)
    dtype = DType.from_value(dtype)

    if num_batches <= 0 or m <= 0 or n <= 0 or k <= 0:
        raise ValueError(
            "num_batches, m, n, and k must be > 0, got "
            f"num_batches={num_batches}, m={m}, n={n}, k={k}"
        )
    if num_batches not in _SUPPORTED_HEAD_COUNTS:
        raise ValueError(
            "torch_mla_q_absorb_glm52 supports num_batches in {16, 32, 64}, "
            f"got {num_batches}"
        )
    if (k, n) != (_QK_NOPE_HEAD_DIM, _KV_LORA_RANK):
        raise ValueError(
            "torch_mla_q_absorb_glm52 requires (k, n) == (192, 512), "
            f"got ({k}, {n})"
        )
    if dtype is not _SUPPORTED_DTYPE:
        raise ValueError(
            "torch_mla_q_absorb_glm52 supports only bf16, "
            f"got {dtype.value}"
        )
    return num_batches, m, n, k, dtype


def _validate_cuda_device(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(
            "CUDA is required for the torch_mla_q_absorb_glm52 backend"
        )
    gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if gpu_name != _REQUIRED_GPU:
        raise ProfilerNotImplemented(
            "torch_mla_q_absorb_glm52 is verified only on "
            f"{_REQUIRED_GPU}, got {gpu_name}"
        )


def _validate_v_up_args(
    num_batches: int,
    m: int,
    n: int,
    k: int,
    dtype: DType | str,
) -> tuple[int, int, int, int, DType]:
    num_batches = int(num_batches)
    m = int(m)
    n = int(n)
    k = int(k)
    dtype = DType.from_value(dtype)

    if num_batches <= 0 or m <= 0 or n <= 0 or k <= 0:
        raise ValueError(
            "num_batches, m, n, and k must be > 0, got "
            f"num_batches={num_batches}, m={m}, n={n}, k={k}"
        )
    if num_batches not in _SUPPORTED_HEAD_COUNTS:
        raise ValueError(
            "torch_mla_v_up_glm52 supports num_batches in {16, 32, 64}, "
            f"got {num_batches}"
        )
    if (k, n) != (_KV_LORA_RANK, _V_HEAD_DIM):
        raise ValueError(
            "torch_mla_v_up_glm52 requires (k, n) == (512, 256), "
            f"got ({k}, {n})"
        )
    if dtype is not _SUPPORTED_DTYPE:
        raise ValueError(
            "torch_mla_v_up_glm52 supports only bf16, "
            f"got {dtype.value}"
        )
    return num_batches, m, n, k, dtype


def _validate_v_up_cuda_device(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(
            "CUDA is required for the torch_mla_v_up_glm52 backend"
        )
    gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if gpu_name != _REQUIRED_GPU:
        raise ProfilerNotImplemented(
            "torch_mla_v_up_glm52 is verified only on "
            f"{_REQUIRED_GPU}, got {gpu_name}"
        )


def _build_q_absorb_operands(
    torch: Any,
    *,
    num_batches: int,
    m: int,
    torch_dtype: Any,
    device: str,
) -> _QAbsorbOperands:
    """Allocate vLLM's packed/interleaved GLM Q-absorption operands."""
    q_base = torch.randn(
        (m, num_batches, _Q_HEAD_WIDTH),
        dtype=torch_dtype,
        device=device,
    )
    lhs = q_base[..., :_QK_NOPE_HEAD_DIM].transpose(0, 1)

    packed_weight = torch.randn(
        (num_batches * _PACKED_HEAD_WIDTH, _KV_LORA_RANK),
        dtype=torch_dtype,
        device=device,
    )
    packed_weight_t = packed_weight.T
    packed_weight_view = packed_weight_t.view(
        _KV_LORA_RANK,
        num_batches,
        _PACKED_HEAD_WIDTH,
    )
    w_uk, _w_uv = packed_weight_view.split(
        [_QK_NOPE_HEAD_DIM, _V_HEAD_DIM],
        dim=-1,
    )
    rhs = w_uk.permute(1, 2, 0)
    out = torch.empty(
        (num_batches, m, _KV_LORA_RANK),
        dtype=torch_dtype,
        device=device,
    )
    return _QAbsorbOperands(
        q_base=q_base,
        lhs=lhs,
        packed_weight=packed_weight,
        rhs=rhs,
        out=out,
    )


def _build_v_up_operands(
    torch: Any,
    *,
    num_batches: int,
    m: int,
    torch_dtype: Any,
    device: str,
) -> _VUpOperands:
    """Allocate vLLM's padded/interleaved GLM V-up operands."""
    attention_base = torch.randn(
        (m, _H200_PADDED_HEADS, _KV_LORA_RANK),
        dtype=torch_dtype,
        device=device,
    )
    attention_output = attention_base[:, :num_batches, :]
    lhs = attention_output.transpose(0, 1)

    packed_weight = torch.randn(
        (num_batches * _PACKED_HEAD_WIDTH, _KV_LORA_RANK),
        dtype=torch_dtype,
        device=device,
    )
    packed_weight_t = packed_weight.T
    packed_weight_view = packed_weight_t.view(
        _KV_LORA_RANK,
        num_batches,
        _PACKED_HEAD_WIDTH,
    )
    _w_uk, w_uv = packed_weight_view.split(
        [_QK_NOPE_HEAD_DIM, _V_HEAD_DIM],
        dim=-1,
    )
    rhs = w_uv.transpose(0, 1)

    out_base = torch.empty(
        (m, num_batches * _V_HEAD_DIM),
        dtype=torch_dtype,
        device=device,
    )
    out = out_base.view(m, num_batches, _V_HEAD_DIM).transpose(0, 1)
    return _VUpOperands(
        attention_base=attention_base,
        attention_output=attention_output,
        lhs=lhs,
        packed_weight=packed_weight,
        rhs=rhs,
        out_base=out_base,
        out=out,
    )


def _logical_elements(num_batches: int, m: int, n: int, k: int) -> int:
    """Read logical LHS/RHS elements and write output elements.

    This excludes unused gaps in the packed backing storage and therefore is
    logical traffic, not a claim about physical device traffic.
    """
    return num_batches * m * k + num_batches * k * n + num_batches * m * n


def profile_mla_q_absorb_glm52(
    num_batches: int,
    m: int,
    n: int,
    k: int,
    dtype: DType | str,
) -> ComputeMetrics:
    """Profile GLM-5.2 MLA Q absorption as one exact-layout Torch BMM."""
    num_batches, m, n, k, dtype = _validate_args(
        num_batches,
        m,
        n,
        k,
        dtype,
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "torch is required for the torch_mla_q_absorb_glm52 backend"
        ) from exc

    _validate_cuda_device(torch)

    try:
        operands = _build_q_absorb_operands(
            torch,
            num_batches=num_batches,
            m=m,
            torch_dtype=dtype.torch(),
            device="cuda",
        )

        def kernel() -> None:
            torch.bmm(operands.lhs, operands.rhs, out=operands.out)

        # NvJet names vary by shape, so the callable is intentionally
        # unfiltered. On the verified Torch 2.10/H200 stack it launches one BMM.
        time_ms = Timer.cupti(kernel)
        energy_j = Energy.perf(
            kernel,
            warmup=5,
            per_iter_time_ms=time_ms,
        )

        flops = 2 * num_batches * m * n * k
        tflops = (
            (flops / (time_ms / 1000.0)) / 1e12
            if time_ms > 0.0
            else 0.0
        )
        bytes_accessed = int(
            _logical_elements(num_batches, m, n, k) * dtype.size_bytes()
        )
        bandwidth_gbps = (
            (bytes_accessed / (time_ms / 1000.0)) / 1e9
            if time_ms > 0.0
            else 0.0
        )
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=float(tflops),
            memory_bandwidth_gbps=float(bandwidth_gbps),
            energy_j=float(energy_j),
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc


def profile_mla_v_up_glm52(
    num_batches: int,
    m: int,
    n: int,
    k: int,
    dtype: DType | str,
) -> ComputeMetrics:
    """Profile GLM-5.2 MLA V-up as one exact-layout Torch BMM."""
    num_batches, m, n, k, dtype = _validate_v_up_args(
        num_batches,
        m,
        n,
        k,
        dtype,
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "torch is required for the torch_mla_v_up_glm52 backend"
        ) from exc

    _validate_v_up_cuda_device(torch)

    try:
        operands = _build_v_up_operands(
            torch,
            num_batches=num_batches,
            m=m,
            torch_dtype=dtype.torch(),
            device="cuda",
        )

        def kernel() -> None:
            torch.bmm(operands.lhs, operands.rhs, out=operands.out)

        # The upstream head-padding copy and all metadata views are outside this
        # callable. NvJet names vary; the verified Torch/H200 path launches one BMM.
        time_ms = Timer.cupti(kernel)
        energy_j = Energy.perf(
            kernel,
            warmup=5,
            per_iter_time_ms=time_ms,
        )

        flops = 2 * num_batches * m * n * k
        tflops = (
            (flops / (time_ms / 1000.0)) / 1e12
            if time_ms > 0.0
            else 0.0
        )
        bytes_accessed = int(
            _logical_elements(num_batches, m, n, k) * dtype.size_bytes()
        )
        bandwidth_gbps = (
            (bytes_accessed / (time_ms / 1000.0)) / 1e9
            if time_ms > 0.0
            else 0.0
        )
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=float(tflops),
            memory_bandwidth_gbps=float(bandwidth_gbps),
            energy_j=float(energy_j),
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
