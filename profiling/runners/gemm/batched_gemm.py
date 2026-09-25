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

# Per-rank head counts of GLM's 64 q heads at validated TP degrees
# (TP1/2/4/8). The timed op is a plain torch.bmm over the head axis, so a
# new degree only needs its per-rank count added here.
_SUPPORTED_HEAD_COUNTS = frozenset({8, 16, 32, 64})
_SUPPORTED_DTYPE = DType.BF16
_SUPPORTED_GPUS = frozenset({"NVIDIA H200", "NVIDIA B200"})
_Q_HEAD_WIDTH = 256
_QK_NOPE_HEAD_DIM = 192
_KV_LORA_RANK = 512
_V_HEAD_DIM = 256
_PACKED_HEAD_WIDTH = _QK_NOPE_HEAD_DIM + _V_HEAD_DIM
_H200_PADDED_HEADS = 64
_K3_HEAD_COUNTS = frozenset({12, 96})
_K3_ABSORB_SHAPES = frozenset({(128, 512), (512, 128), (512, 256)})


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
            "torch_mla_q_absorb_glm52 supports num_batches in "
            f"{sorted(_SUPPORTED_HEAD_COUNTS)}, got {num_batches}"
        )
    if (k, n) != (_QK_NOPE_HEAD_DIM, _KV_LORA_RANK):
        raise ValueError(f"torch_mla_q_absorb_glm52 requires (k, n) == (192, 512), got ({k}, {n})")
    if dtype is not _SUPPORTED_DTYPE:
        raise ValueError(f"torch_mla_q_absorb_glm52 supports only bf16, got {dtype.value}")
    return num_batches, m, n, k, dtype


def _validate_cuda_device(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for the torch_mla_q_absorb_glm52 backend")
    gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if gpu_name not in _SUPPORTED_GPUS:
        raise ProfilerNotImplemented(
            "torch_mla_q_absorb_glm52 is verified only on "
            f"{sorted(_SUPPORTED_GPUS)}, got {gpu_name}"
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
            "torch_mla_v_up_glm52 supports num_batches in "
            f"{sorted(_SUPPORTED_HEAD_COUNTS)}, got {num_batches}"
        )
    if (k, n) != (_KV_LORA_RANK, _V_HEAD_DIM):
        raise ValueError(f"torch_mla_v_up_glm52 requires (k, n) == (512, 256), got ({k}, {n})")
    if dtype is not _SUPPORTED_DTYPE:
        raise ValueError(f"torch_mla_v_up_glm52 supports only bf16, got {dtype.value}")
    return num_batches, m, n, k, dtype


def _validate_v_up_cuda_device(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for the torch_mla_v_up_glm52 backend")
    gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if gpu_name not in _SUPPORTED_GPUS:
        raise ProfilerNotImplemented(
            f"torch_mla_v_up_glm52 is verified only on {sorted(_SUPPORTED_GPUS)}, got {gpu_name}"
        )


def _validate_k3_absorb_args(
    num_batches: int,
    m: int,
    n: int,
    k: int,
    dtype: DType | str,
) -> tuple[int, int, int, int, DType]:
    values = (int(num_batches), int(m), int(n), int(k))
    dtype = DType.from_value(dtype)
    if min(values) <= 0:
        raise ValueError("num_batches, m, n, and k must be positive")
    if num_batches not in _K3_HEAD_COUNTS:
        raise ValueError(f"K3 absorb BMM requires heads in {sorted(_K3_HEAD_COUNTS)}")
    # Decode uses [heads, B, 128] @ [heads, 128, 512] and
    # [heads, B, 512] @ [heads, 512, 128]. Chunked-prefix MLA additionally
    # projects cached latent rows with [heads, S, 512] @ [heads, 512, 256].
    # Any positive m is valid for these shapes.
    if (k, n) not in _K3_ABSORB_SHAPES:
        raise ValueError(
            "K3 absorb BMM requires (k,n) in {(128,512),(512,128),(512,256)}"
        )
    if dtype is not DType.BF16:
        raise ValueError("K3 absorb BMM requires dtype=bf16")
    return *values, dtype


def profile_sglang_k3_absorb(
    num_batches: int,
    m: int,
    n: int,
    k: int,
    dtype: DType | str,
) -> ComputeMetrics:
    num_batches, m, n, k, dtype = _validate_k3_absorb_args(num_batches, m, n, k, dtype)
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented("torch is required for the K3 absorb BMM") from exc
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for the K3 absorb BMM")
    if str(torch.cuda.get_device_name(torch.cuda.current_device())) != "NVIDIA B200":
        raise ProfilerNotImplemented("K3 absorb BMM is verified only on NVIDIA B200")
    lhs = torch.randn((num_batches, m, k), dtype=torch.bfloat16, device="cuda")
    rhs = torch.randn((num_batches, k, n), dtype=torch.bfloat16, device="cuda")
    output = torch.empty((num_batches, m, n), dtype=torch.bfloat16, device="cuda")

    def kernel() -> None:
        torch.bmm(lhs, rhs, out=output)

    try:
        time_ms = Timer.cupti(kernel)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
    seconds = time_ms / 1000.0
    flops = 2 * num_batches * m * n * k
    bytes_accessed = (num_batches * (m * k + k * n + m * n)) * dtype.size_bytes()
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=flops / seconds / 1e12 if seconds else 0.0,
        memory_bandwidth_gbps=bytes_accessed / seconds / 1e9 if seconds else 0.0,
        energy_j=float(energy_j),
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
        tflops = (flops / (time_ms / 1000.0)) / 1e12 if time_ms > 0.0 else 0.0
        bytes_accessed = int(_logical_elements(num_batches, m, n, k) * dtype.size_bytes())
        bandwidth_gbps = (bytes_accessed / (time_ms / 1000.0)) / 1e9 if time_ms > 0.0 else 0.0
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
        tflops = (flops / (time_ms / 1000.0)) / 1e12 if time_ms > 0.0 else 0.0
        bytes_accessed = int(_logical_elements(num_batches, m, n, k) * dtype.size_bytes())
        bandwidth_gbps = (bytes_accessed / (time_ms / 1000.0)) / 1e9 if time_ms > 0.0 else 0.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=float(tflops),
            memory_bandwidth_gbps=float(bandwidth_gbps),
            energy_j=float(energy_j),
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
