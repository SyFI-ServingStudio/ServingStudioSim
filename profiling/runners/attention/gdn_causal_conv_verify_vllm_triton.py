"""vLLM's causal_conv1d_update on a speculative (MTP verify) step.

``gdn_causal_conv_decode`` with ``num_tokens = q · batch_size``, q > 1. vLLM
calls the same ``_causal_conv1d_update_kernel`` as one-token decode, but in its
varlen speculative form (``qwen_gdn_linear_attn.py`` ``_forward_core`` and
``_forward_core_decode_spec_fused_norm``): request-major tokens described by
``query_start_loc``, ``num_accepted_tokens``, the first column of the ``[B, q]``
checkpoint indices, and a history row of ``kernel_size − 2 + q`` samples. The
operands are built the same way here; the correctness check and its operand
restoration stay outside CUPTI, which counts only ``_causal_conv1d_update_kernel``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profiling.runners.attention._gdn_common import conv_state_pool

KERNEL_NAME = "_causal_conv1d_update_kernel"
_OUTPUT_ATOL = 1e-2
_OUTPUT_RTOL = 1e-2


@dataclass(frozen=True)
class VerifyOperands:
    x: Any
    weight: Any
    state: Any
    state_indices: Any
    num_accepted: Any
    query_start_loc: Any
    per_request: int


def accepted_counts(batch_size: int, per_request: int) -> tuple[int, ...]:
    """Deterministic accepted counts covering 1..q (the kernel reads from ``a − 1``)."""
    return tuple(request % per_request + 1 for request in range(batch_size))


def checkpoint_indices(batch_size: int, per_request: int) -> tuple[tuple[int, ...], ...]:
    """vLLM's ``[B, q]`` per-token checkpoint slots; slot zero stays reserved."""
    return tuple(
        tuple(request * per_request + token + 1 for token in range(per_request))
        for request in range(batch_size)
    )


def build_operands(
    torch: Any,
    *,
    batch_size: int,
    per_request: int,
    channels: int,
    kernel_size: int,
    dtype: Any,
    state_dtype: Any,
    device: Any,
    state_dim_first: bool,
) -> VerifyOperands:
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    num_tokens = batch_size * per_request
    x = torch.empty((num_tokens, channels), dtype=dtype, device=device).uniform_(
        -0.25, 0.25, generator=generator
    )
    weight = torch.empty((channels, kernel_size), dtype=dtype, device=device).uniform_(
        -0.125, 0.125, generator=generator
    )
    state = conv_state_pool(
        torch,
        (num_tokens + 1, channels, kernel_size - 2 + per_request),
        dtype=state_dtype,
        device=device,
        dim_first=state_dim_first,
    ).uniform_(-0.25, 0.25, generator=generator)
    return VerifyOperands(
        x=x,
        weight=weight,
        state=state,
        state_indices=torch.tensor(
            checkpoint_indices(batch_size, per_request), dtype=torch.int32, device=device
        ),
        num_accepted=torch.tensor(
            accepted_counts(batch_size, per_request), dtype=torch.int32, device=device
        ),
        query_start_loc=torch.arange(
            0, num_tokens + 1, per_request, dtype=torch.int32, device=device
        ),
        per_request=per_request,
    )


def invoke(fused_callable: Any, operands: VerifyOperands) -> Any:
    """The call ``_forward_core_decode_spec_fused_norm`` makes."""
    return fused_callable(
        operands.x,
        operands.state,
        operands.weight,
        None,
        "silu",
        conv_state_indices=operands.state_indices[:, 0],
        num_accepted_tokens=operands.num_accepted,
        query_start_loc=operands.query_start_loc,
        max_query_len=operands.per_request,
        validate_data=False,
    )


def check_correctness(
    torch: Any, fused_callable: Any, operands: VerifyOperands, *, synchronize: Any | None = None
) -> None:
    """Compare output and the selected history rows with the reference, then restore."""
    from profiling.runners.attention.gdn_spec_decode_reference import (
        gdn_causal_conv_spec_decode_reference,
    )

    synchronize = torch.cuda.synchronize if synchronize is None else synchronize
    x_snapshot = operands.x.clone()
    state_snapshot = operands.state.clone()
    expected_output, expected_state = gdn_causal_conv_spec_decode_reference(
        x_snapshot.clone(),
        operands.weight,
        state_snapshot.clone().contiguous(),
        operands.state_indices,
        operands.num_accepted,
    )
    try:
        returned = invoke(fused_callable, operands)
        synchronize()
        torch.testing.assert_close(
            returned.float(), expected_output.float(), atol=_OUTPUT_ATOL, rtol=_OUTPUT_RTOL
        )
        if not torch.equal(operands.state, expected_state):
            raise AssertionError("verify conv history rows must exactly match the reference")
    finally:
        operands.x.copy_(x_snapshot)
        operands.state.copy_(state_snapshot)
