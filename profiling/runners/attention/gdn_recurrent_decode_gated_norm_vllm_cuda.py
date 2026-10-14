"""vLLM's fused CUDA verify recurrence with the gated output norm.

Times ``vllm._custom_ops.fused_gdn_decode_post_conv_mtp`` exactly as
``_forward_core_decode_spec_post_conv_fused_norm`` calls it on a pure MTP
verify step: the post-convolution ``mixed_qkv`` and the output gate are strided
views of one qkvz projection, ``a``/``b`` views of the ba projection, the FP32
state pool holds one checkpoint slot per token, and the gate is sigmoid.
Operand construction, the reference check and its restoration stay outside
CUPTI; only ``gdn_decode_post_conv_mtp_kernel`` is timed.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention._gdn_common import load_required_callable, tokens_per_request
from profiling.runners.attention._gdn_verify import (
    VerifyRecurrenceOperands,
    build_verify_recurrence_operands,
)
from profiling.runners.attention.gdn_recurrent_decode_torch import (
    _logical_bytes,
    _semantic_flops,
)
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "gdn_recurrent_decode_gated_norm:vllm_cuda"
_CALLABLE_MODULE = "vllm._custom_ops"
_CALLABLE_NAME = "fused_gdn_decode_post_conv_mtp"
_KERNEL_NAME = "gdn_decode_post_conv_mtp_kernel"
# Kernel limits (fused_gdn_decode_kernel.cu): kDimK = kDimV = 128,
# kMaxMtpTokens = 8, ValueHeadsPerKeyHead instantiated for 1, 2, 3, 4, 8.
_HEAD_DIM = 128
_MAX_TOKENS_PER_REQUEST = 8
_HEAD_RATIOS = frozenset({1, 2, 3, 4, 8})
_NORM_EPS = 1e-6
_GATE_ACTIVATION = "sigmoid"
_OUTPUT_ATOL = 2e-2
_OUTPUT_RTOL = 2e-2
_STATE_ATOL = 2e-5
_STATE_RTOL = 2e-4


@dataclass(frozen=True)
class _ValidatedArgs:
    batch_size: int
    per_request: int
    num_qk_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int
    dtype: DType
    state_dtype: DType


def _validate_args(
    batch_size: int,
    num_qk_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: DType | str,
    state_dtype: DType | str,
    num_tokens: int,
) -> _ValidatedArgs:
    batch_size = int(batch_size)
    if batch_size <= 0 or int(num_qk_heads) <= 0 or int(num_value_heads) <= 0:
        raise ValueError("batch_size, num_qk_heads and num_value_heads must be > 0")
    args = _ValidatedArgs(
        batch_size=batch_size,
        per_request=tokens_per_request(batch_size, int(num_tokens)),
        num_qk_heads=int(num_qk_heads),
        num_value_heads=int(num_value_heads),
        key_head_dim=int(key_head_dim),
        value_head_dim=int(value_head_dim),
        dtype=DType.from_value(dtype),
        state_dtype=DType.from_value(state_dtype),
    )
    if args.per_request > _MAX_TOKENS_PER_REQUEST:
        raise ValueError(
            f"{_BACKEND} handles at most {_MAX_TOKENS_PER_REQUEST} tokens per request, "
            f"got {args.per_request}"
        )
    if args.key_head_dim != _HEAD_DIM or args.value_head_dim != _HEAD_DIM:
        raise ValueError(f"{_BACKEND} requires key_head_dim = value_head_dim = {_HEAD_DIM}")
    if (
        args.num_value_heads % args.num_qk_heads != 0
        or args.num_value_heads // args.num_qk_heads not in _HEAD_RATIOS
    ):
        raise ValueError(f"{_BACKEND} requires num_value_heads / num_qk_heads in {_HEAD_RATIOS}")
    # BackendSupport admits only BF16 activations; the state dtype is not a
    # capability axis, so the runner keeps its own check.
    if args.state_dtype is not DType.FP32:
        raise ValueError(f"{_BACKEND} is profiled with state_dtype=fp32")
    return args


def _build_operands(torch: Any, args: _ValidatedArgs, *, device: Any) -> VerifyRecurrenceOperands:
    return build_verify_recurrence_operands(
        torch,
        batch_size=args.batch_size,
        per_request=args.per_request,
        num_qk_heads=args.num_qk_heads,
        num_value_heads=args.num_value_heads,
        key_head_dim=args.key_head_dim,
        value_head_dim=args.value_head_dim,
        dtype=args.dtype.torch(),
        state_dtype=args.state_dtype.torch(),
        device=device,
    )


def _invoke(fused_callable: Any, operands: VerifyRecurrenceOperands, out: Any) -> Any:
    return fused_callable(
        mixed_qkv=operands.mixed_qkv,
        a=operands.a,
        b=operands.b,
        A_log=operands.A_log,
        dt_bias=operands.dt_bias,
        state_indices=operands.state_indices,
        cu_seqlens=operands.cu_seqlens,
        num_accepted_tokens=operands.num_accepted,
        state=operands.state,
        output_gate=operands.output_gate,
        norm_weight=operands.norm_weight,
        out=out,
        scale=operands.key_head_dim**-0.5,
        norm_eps=_NORM_EPS,
        output_gate_activation=_GATE_ACTIVATION,
    )


def _check_correctness(
    torch: Any, fused_callable: Any, operands: VerifyRecurrenceOperands, out: Any
) -> None:
    """Compare output and the whole checkpoint pool with the reference, then restore."""
    from profiling.runners.attention.gdn_spec_decode_reference import (
        gdn_recurrent_spec_decode_gated_norm_reference,
    )

    state_snapshot = operands.state.clone()
    query, key, value = operands.semantic_qkv()
    expected_output, expected_state = gdn_recurrent_spec_decode_gated_norm_reference(
        query,
        key,
        value,
        operands.a.contiguous(),
        operands.b.contiguous(),
        operands.A_log,
        operands.dt_bias,
        state_snapshot.clone(),
        operands.state_indices,
        operands.num_accepted,
        operands.output_gate,
        operands.norm_weight,
        _NORM_EPS,
    )
    try:
        _invoke(fused_callable, operands, out)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            out.float(), expected_output.float(), atol=_OUTPUT_ATOL, rtol=_OUTPUT_RTOL
        )
        torch.testing.assert_close(
            operands.state, expected_state, atol=_STATE_ATOL, rtol=_STATE_RTOL
        )
    finally:
        operands.state.copy_(state_snapshot)


def profile_gdn_recurrent_decode_gated_norm_vllm_cuda(
    batch_size: int,
    num_qk_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: DType | str,
    state_dtype: DType | str,
    num_tokens: int,
) -> ComputeMetrics:
    """Profile vLLM's fused verify recurrence + gated norm."""
    args = _validate_args(
        batch_size,
        num_qk_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim,
        dtype,
        state_dtype,
        num_tokens,
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_BACKEND}") from exc

    try:
        fused_callable = load_required_callable(
            importlib.import_module,
            backend=_BACKEND,
            module_name=_CALLABLE_MODULE,
            callable_name=_CALLABLE_NAME,
        )
        device = torch.device("cuda", torch.cuda.current_device())
        operands = _build_operands(torch, args, device=device)
        out = torch.empty(
            (operands.num_tokens, args.num_value_heads, args.value_head_dim),
            dtype=args.dtype.torch(),
            device=device,
        )
        _check_correctness(torch, fused_callable, operands, out)

        def kernel() -> Any:
            return _invoke(fused_callable, operands, out)

        time_ms = Timer.cupti(kernel, kernel_name=_KERNEL_NAME)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed") from exc

    flops, logical_bytes = _metrics(args)
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        energy_j=float(energy_j),
        tflops=flops / elapsed_s / 1e12 if time_ms > 0.0 else 0.0,
        memory_bandwidth_gbps=(logical_bytes / elapsed_s / 1e9 if time_ms > 0.0 else 0.0),
    )


def _metrics(args: _ValidatedArgs) -> tuple[int, float]:
    """Recurrence counts plus the norm: 7 FLOPs and gate/output traffic per output element."""
    shape = dict(
        batch_size=args.batch_size,
        num_qk_heads=args.num_qk_heads,
        num_value_heads=args.num_value_heads,
        key_head_dim=args.key_head_dim,
        value_head_dim=args.value_head_dim,
        per_request=args.per_request,
    )
    output_elements = (
        args.batch_size * args.per_request * args.num_value_heads * args.value_head_dim
    )
    flops = _semantic_flops(**shape) + 7 * output_elements
    logical_bytes = (
        _logical_bytes(**shape, dtype=args.dtype, state_dtype=args.state_dtype)
        + (output_elements + args.value_head_dim) * args.dtype.size_bytes()
    )
    return flops, logical_bytes
