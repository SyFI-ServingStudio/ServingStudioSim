"""vLLM sigmoid-gating Triton runner for GDN recurrent decode.

vLLM runs the decode requests of a step that also carries a prefill through
``fused_sigmoid_gating_delta_rule_update`` instead of the packed decode kernel:
separate contiguous Q/K/V, one token per request described by ``cu_seqlens``,
and the same indexed FP32 state pool updated in place. The two kernels compute
the same recurrence but schedule it differently, so this backend prices that
call. With MTP the same call prices the verify requests of such a step
(``num_tokens``): q tokens per request, ``num_accepted_tokens`` and ``[B, q]``
checkpoint indices. Operand construction, the reference check and its state restoration stay
outside CUPTI; only ``fused_sigmoid_gating_delta_rule_update_kernel`` is timed.
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
from profiling.runners.attention.gdn_recurrent_decode_vllm_triton import (
    _build_operands,
    _Operands,
    _validate_args,
    _ValidatedArgs,
)
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "gdn_recurrent_decode:vllm_triton_sigmoid_gating"
_CALLABLE_MODULE = "vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating"
_CALLABLE_NAME = "fused_sigmoid_gating_delta_rule_update"
_KERNEL_NAME = "fused_sigmoid_gating_delta_rule_update_kernel"
_OUTPUT_ATOL = 1e-2
_OUTPUT_RTOL = 1e-2
_STATE_ATOL = 2e-5
_STATE_RTOL = 2e-4


@dataclass(frozen=True)
class _VarlenOperands:
    base: _Operands
    query: Any
    key: Any
    value: Any
    cu_seqlens: Any


def _load_callable() -> Any:
    return load_required_callable(
        importlib.import_module,
        backend=_BACKEND,
        module_name=_CALLABLE_MODULE,
        callable_name=_CALLABLE_NAME,
    )


def _build_varlen_operands(torch: Any, args: _ValidatedArgs, *, device: Any) -> _VarlenOperands:
    base = _build_operands(torch, args, device=device)
    # vLLM flattens the decode tokens into one varlen batch: [1, B, heads, dim].
    return _VarlenOperands(
        base=base,
        query=base.query.unsqueeze(0).contiguous(),
        key=base.key.unsqueeze(0).contiguous(),
        value=base.value.unsqueeze(0).contiguous(),
        cu_seqlens=torch.arange(args.batch_size + 1, dtype=torch.int32, device=device),
    )


def _invoke(fused_callable: Any, operands: _VarlenOperands, args: _ValidatedArgs) -> Any:
    base = operands.base
    return fused_callable(
        A_log=base.A_log,
        a=base.a,
        b=base.b,
        dt_bias=base.dt_bias,
        q=operands.query,
        k=operands.key,
        v=operands.value,
        scale=args.key_head_dim**-0.5,
        initial_state=base.initial_state,
        inplace_final_state=True,
        cu_seqlens=operands.cu_seqlens,
        ssm_state_indices=base.ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
    )


def _check_correctness(
    torch: Any, fused_callable: Any, operands: _VarlenOperands, args: _ValidatedArgs
) -> None:
    """Compare output and every valid state slot with the reference, then restore."""
    from profiling.runners.attention.gdn_recurrent_decode_reference import (
        gdn_recurrent_decode_reference,
    )

    base = operands.base
    state_snapshot = base.initial_state.clone()
    semantic_state = state_snapshot[1:].transpose(-1, -2).contiguous()
    expected_output, expected_state = gdn_recurrent_decode_reference(
        base.query,
        base.key,
        base.value,
        base.a,
        base.b,
        base.A_log,
        base.dt_bias,
        semantic_state,
    )
    try:
        output, final_state = _invoke(fused_callable, operands, args)
        torch.cuda.synchronize()
        if final_state is not base.initial_state:
            raise AssertionError("sigmoid-gating decode must update the state pool in place")
        if not torch.equal(base.initial_state[0], state_snapshot[0]):
            raise AssertionError("reserved state slot zero was mutated")
        actual_state = base.initial_state[1:].transpose(-1, -2)
        torch.testing.assert_close(
            output.reshape(expected_output.shape).float(),
            expected_output.float(),
            atol=_OUTPUT_ATOL,
            rtol=_OUTPUT_RTOL,
        )
        torch.testing.assert_close(actual_state, expected_state, atol=_STATE_ATOL, rtol=_STATE_RTOL)
    finally:
        base.initial_state.copy_(state_snapshot)


@dataclass(frozen=True)
class _VerifyCall:
    operands: VerifyRecurrenceOperands
    query: Any
    key: Any
    value: Any
    a: Any
    b: Any


def _build_verify_call(torch: Any, args: _ValidatedArgs, per_request: int, *, device: Any):
    operands = build_verify_recurrence_operands(
        torch,
        batch_size=args.batch_size,
        per_request=per_request,
        num_qk_heads=args.num_qk_heads,
        num_value_heads=args.num_value_heads,
        key_head_dim=args.key_head_dim,
        value_head_dim=args.value_head_dim,
        dtype=args.dtype.torch(),
        state_dtype=args.state_dtype.torch(),
        device=device,
    )
    query, key, value = operands.semantic_qkv()
    # The mixed step's rearrange_mixed_qkv and index_select hand the kernel
    # contiguous [1, T, H, D] q/k/v and [T, Hv] a/b.
    return _VerifyCall(
        operands=operands,
        query=query.unsqueeze(0),
        key=key.unsqueeze(0),
        value=value.unsqueeze(0),
        a=operands.a.contiguous(),
        b=operands.b.contiguous(),
    )


def _invoke_verify(fused_callable: Any, call: _VerifyCall, args: _ValidatedArgs) -> Any:
    """The spec-decode call of ``_forward_core`` in a step that also carries a prefill."""
    operands = call.operands
    return fused_callable(
        A_log=operands.A_log,
        a=call.a,
        b=call.b,
        dt_bias=operands.dt_bias,
        q=call.query,
        k=call.key,
        v=call.value,
        scale=args.key_head_dim**-0.5,
        initial_state=operands.state,
        inplace_final_state=True,
        cu_seqlens=operands.cu_seqlens,
        ssm_state_indices=operands.state_indices,
        num_accepted_tokens=operands.num_accepted,
        use_qk_l2norm_in_kernel=True,
    )


def _check_verify_correctness(
    torch: Any, fused_callable: Any, call: _VerifyCall, args: _ValidatedArgs
) -> None:
    """Compare output and the whole checkpoint pool with the reference, then restore."""
    from profiling.runners.attention.gdn_spec_decode_reference import (
        gdn_recurrent_spec_decode_reference,
    )

    operands = call.operands
    state_snapshot = operands.state.clone()
    expected_output, expected_state = gdn_recurrent_spec_decode_reference(
        call.query[0],
        call.key[0],
        call.value[0],
        call.a,
        call.b,
        operands.A_log,
        operands.dt_bias,
        state_snapshot.clone(),
        operands.state_indices,
        operands.num_accepted,
    )
    try:
        output, _ = _invoke_verify(fused_callable, call, args)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output.reshape(expected_output.shape).float(),
            expected_output.float(),
            atol=_OUTPUT_ATOL,
            rtol=_OUTPUT_RTOL,
        )
        torch.testing.assert_close(
            operands.state, expected_state, atol=_STATE_ATOL, rtol=_STATE_RTOL
        )
    finally:
        operands.state.copy_(state_snapshot)


def profile_gdn_recurrent_decode_vllm_triton_sigmoid_gating(
    batch_size: int,
    num_qk_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: DType | str,
    state_dtype: DType | str,
    num_tokens: int | None = None,
) -> ComputeMetrics:
    """Profile vLLM's sigmoid-gating recurrent update.

    One token per request, or ``num_tokens / batch_size`` verify tokens per
    request with acceptance-indexed checkpoints when ``num_tokens`` is set.
    """
    args = _validate_args(
        batch_size,
        num_qk_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim,
        dtype,
        state_dtype,
    )
    per_request = tokens_per_request(args.batch_size, num_tokens)
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_BACKEND}") from exc

    try:
        fused_callable = _load_callable()
        device = torch.device("cuda", torch.cuda.current_device())
        if per_request == 1:
            operands = _build_varlen_operands(torch, args, device=device)
            _check_correctness(torch, fused_callable, operands, args)

            def kernel() -> Any:
                return _invoke(fused_callable, operands, args)

        else:
            call = _build_verify_call(torch, args, per_request, device=device)
            _check_verify_correctness(torch, fused_callable, call, args)

            def kernel() -> Any:
                return _invoke_verify(fused_callable, call, args)

        time_ms = Timer.cupti(kernel, kernel_name=_KERNEL_NAME)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed") from exc

    flops = _semantic_flops(
        batch_size=args.batch_size,
        num_qk_heads=args.num_qk_heads,
        num_value_heads=args.num_value_heads,
        key_head_dim=args.key_head_dim,
        value_head_dim=args.value_head_dim,
        per_request=per_request,
    )
    logical_bytes = _logical_bytes(
        batch_size=args.batch_size,
        num_qk_heads=args.num_qk_heads,
        num_value_heads=args.num_value_heads,
        key_head_dim=args.key_head_dim,
        value_head_dim=args.value_head_dim,
        dtype=args.dtype,
        state_dtype=args.state_dtype,
        per_request=per_request,
    )
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        energy_j=float(energy_j),
        tflops=flops / elapsed_s / 1e12 if time_ms > 0.0 else 0.0,
        memory_bandwidth_gbps=(logical_bytes / elapsed_s / 1e9 if time_ms > 0.0 else 0.0),
    )
