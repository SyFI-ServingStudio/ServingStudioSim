"""Production-layout Torch runners for MLA batched GEMMs.

Each timed callable is only its unquantized ``torch.bmm(..., out=...)`` launch.
Q-absorption and V-up layout construction mirror vLLM v0.23.0 commit
``0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`` in
``mla_attention.py::{process_weights_after_loading,_v_up_proj}``.

``torch_mla_q_absorb_no_rope`` and ``torch_mla_v_up_unpadded`` reuse the same
construction for an MLA layout with qk_nope 256 and no RoPE part, v_head 256,
kv_lora 512, and an unpadded attention output (the alignment fork's
``mla_attention.py:1170-1252,993-1008,1300-1332``; ``FlashInferMLASparseImpl``
sets no ``q_pad_num_heads``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

# The timed op is a plain torch.bmm over the head axis, so any positive
# per-rank head count launches. The padded V-up layout is the exception: it
# slices its LHS out of an attention output allocated with a 64-head stride
# (_PADDED_HEADS), so that layout holds at most 64 heads.
_SUPPORTED_DTYPE = DType.BF16
_Q_HEAD_WIDTH = 256
_QK_NOPE_HEAD_DIM = 192
_KV_LORA_RANK = 512
_V_HEAD_DIM = 256
_PACKED_HEAD_WIDTH = _QK_NOPE_HEAD_DIM + _V_HEAD_DIM
_PADDED_HEADS = 64


@dataclass(frozen=True)
class _MlaLayout:
    q_head_width: int
    qk_nope_head_dim: int
    v_head_dim: int
    kv_lora_rank: int
    # Attention-output head stride for V-up; None means unpadded (num_batches).
    padded_heads: int | None


_NO_ROPE_UNPADDED_LAYOUT = _MlaLayout(
    q_head_width=256,
    qk_nope_head_dim=256,
    v_head_dim=256,
    kv_lora_rank=512,
    padded_heads=None,
)


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
    if (k, n) != (_QK_NOPE_HEAD_DIM, _KV_LORA_RANK):
        raise ValueError(f"torch_mla_q_absorb requires (k, n) == (192, 512), got ({k}, {n})")
    if dtype is not _SUPPORTED_DTYPE:
        raise ValueError(f"torch_mla_q_absorb supports only bf16, got {dtype.value}")
    return num_batches, m, n, k, dtype


def _validate_cuda_device(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for the torch_mla_q_absorb backend")


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
    if num_batches > _PADDED_HEADS:
        raise ValueError(
            f"torch_mla_v_up pads the attention output to {_PADDED_HEADS} heads, "
            f"so num_batches must be <= {_PADDED_HEADS}, got {num_batches}"
        )
    if (k, n) != (_KV_LORA_RANK, _V_HEAD_DIM):
        raise ValueError(f"torch_mla_v_up requires (k, n) == (512, 256), got ({k}, {n})")
    if dtype is not _SUPPORTED_DTYPE:
        raise ValueError(f"torch_mla_v_up supports only bf16, got {dtype.value}")
    return num_batches, m, n, k, dtype


def _validate_v_up_cuda_device(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for the torch_mla_v_up backend")


def _build_q_absorb_operands(
    torch: Any,
    *,
    num_batches: int,
    m: int,
    torch_dtype: Any,
    device: str,
) -> _QAbsorbOperands:
    """Allocate vLLM's packed/interleaved Q-absorption operands."""
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
    """Allocate vLLM's padded/interleaved V-up operands."""
    attention_base = torch.randn(
        (m, _PADDED_HEADS, _KV_LORA_RANK),
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


def profile_mla_q_absorb(
    num_batches: int,
    m: int,
    n: int,
    k: int,
    dtype: DType | str,
) -> ComputeMetrics:
    """Profile MLA Q absorption as one exact-layout Torch BMM."""
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
            "torch is required for the torch_mla_q_absorb backend"
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
        # unfiltered. On the Torch 2.10/H200 stack it was observed to launch one BMM.
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


def profile_mla_v_up(
    num_batches: int,
    m: int,
    n: int,
    k: int,
    dtype: DType | str,
) -> ComputeMetrics:
    """Profile MLA V-up as one exact-layout Torch BMM."""
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
        raise ProfilerNotImplemented("torch is required for the torch_mla_v_up backend") from exc

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
        # callable. NvJet names vary; the Torch/H200 path was observed to launch one BMM.
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


def _build_layout_q_absorb_operands(
    torch: Any, layout: _MlaLayout, *, num_batches: int, m: int, torch_dtype: Any, device: str
) -> _QAbsorbOperands:
    """Q absorption: (N, B, P) x W_UK_T (N, P, L) -> (N, B, L) over kv_b_proj views."""
    packed_width = layout.qk_nope_head_dim + layout.v_head_dim
    q_base = torch.randn((m, num_batches, layout.q_head_width), dtype=torch_dtype, device=device)
    lhs = q_base[..., : layout.qk_nope_head_dim].transpose(0, 1)
    packed_weight = torch.randn(
        (num_batches * packed_width, layout.kv_lora_rank), dtype=torch_dtype, device=device
    )
    packed_view = packed_weight.T.view(layout.kv_lora_rank, num_batches, packed_width)
    w_uk, _w_uv = packed_view.split([layout.qk_nope_head_dim, layout.v_head_dim], dim=-1)
    rhs = w_uk.permute(1, 2, 0)
    out = torch.empty((num_batches, m, layout.kv_lora_rank), dtype=torch_dtype, device=device)
    return _QAbsorbOperands(q_base=q_base, lhs=lhs, packed_weight=packed_weight, rhs=rhs, out=out)


def _build_layout_v_up_operands(
    torch: Any, layout: _MlaLayout, *, num_batches: int, m: int, torch_dtype: Any, device: str
) -> _VUpOperands:
    """V-up: (N, B, L) x W_UV (N, L, V) -> (N, B, V), written into a (B, N*V) output."""
    packed_width = layout.qk_nope_head_dim + layout.v_head_dim
    heads = layout.padded_heads or num_batches
    attention_base = torch.randn((m, heads, layout.kv_lora_rank), dtype=torch_dtype, device=device)
    attention_output = attention_base[:, :num_batches, :]
    lhs = attention_output.transpose(0, 1)
    packed_weight = torch.randn(
        (num_batches * packed_width, layout.kv_lora_rank), dtype=torch_dtype, device=device
    )
    packed_view = packed_weight.T.view(layout.kv_lora_rank, num_batches, packed_width)
    _w_uk, w_uv = packed_view.split([layout.qk_nope_head_dim, layout.v_head_dim], dim=-1)
    rhs = w_uv.transpose(0, 1)
    out_base = torch.empty((m, num_batches * layout.v_head_dim), dtype=torch_dtype, device=device)
    out = out_base.view(m, num_batches, layout.v_head_dim).transpose(0, 1)
    return _VUpOperands(
        attention_base=attention_base,
        attention_output=attention_output,
        lhs=lhs,
        packed_weight=packed_weight,
        rhs=rhs,
        out_base=out_base,
        out=out,
    )


def _validate_layout_args(
    backend: str,
    expected_kn: tuple[int, int],
    num_batches: int,
    m: int,
    n: int,
    k: int,
    dtype: DType | str,
) -> tuple[int, int, int, int, DType]:
    num_batches, m, n, k = int(num_batches), int(m), int(n), int(k)
    dtype = DType.from_value(dtype)
    if num_batches <= 0 or m <= 0 or n <= 0 or k <= 0:
        raise ValueError(
            "num_batches, m, n, and k must be > 0, got "
            f"num_batches={num_batches}, m={m}, n={n}, k={k}"
        )
    if (k, n) != expected_kn:
        raise ValueError(f"{backend} requires (k, n) == {expected_kn}, got ({k}, {n})")
    if dtype is not _SUPPORTED_DTYPE:
        raise ValueError(f"{backend} supports only bf16, got {dtype.value}")
    return num_batches, m, n, k, dtype


def _profile_layout_bmm(
    backend: str,
    build: Any,
    layout: _MlaLayout,
    num_batches: int,
    m: int,
    n: int,
    k: int,
    dtype: DType,
) -> ComputeMetrics:
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"torch is required for the {backend} backend") from exc
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(f"CUDA is required for the {backend} backend")
    try:
        operands = build(
            torch, layout, num_batches=num_batches, m=m, torch_dtype=dtype.torch(), device="cuda"
        )

        def kernel() -> None:
            torch.bmm(operands.lhs, operands.rhs, out=operands.out)

        # Untimed semantic check on the exact strided operands.
        kernel()
        torch.cuda.synchronize()
        expected = torch.bmm(operands.lhs.float(), operands.rhs.float())
        torch.testing.assert_close(operands.out.float(), expected, rtol=2e-2, atol=2e-1)

        time_ms = Timer.cupti(kernel)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
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


def profile_mla_q_absorb_no_rope(
    num_batches: int,
    m: int,
    n: int,
    k: int,
    dtype: DType | str,
) -> ComputeMetrics:
    """Profile MLA Q absorption with no RoPE part ([N,B,256] x [N,256,512])."""
    backend = "torch_mla_q_absorb_no_rope"
    layout = _NO_ROPE_UNPADDED_LAYOUT
    args = _validate_layout_args(
        backend, (layout.qk_nope_head_dim, layout.kv_lora_rank), num_batches, m, n, k, dtype
    )
    return _profile_layout_bmm(backend, _build_layout_q_absorb_operands, layout, *args)


def profile_mla_v_up_unpadded(
    num_batches: int,
    m: int,
    n: int,
    k: int,
    dtype: DType | str,
) -> ComputeMetrics:
    """Profile MLA V-up into an unpadded head axis ([N,B,512] x [N,512,256])."""
    backend = "torch_mla_v_up_unpadded"
    layout = _NO_ROPE_UNPADDED_LAYOUT
    args = _validate_layout_args(
        backend, (layout.kv_lora_rank, layout.v_head_dim), num_batches, m, n, k, dtype
    )
    return _profile_layout_bmm(backend, _build_layout_v_up_operands, layout, *args)
