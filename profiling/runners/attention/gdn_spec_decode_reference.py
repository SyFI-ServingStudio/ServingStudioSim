"""Torch semantics for Gated DeltaNet speculative (MTP verify) decode.

A verify step carries ``q`` tokens per request: the accepted token plus the
drafts. vLLM keeps one checkpoint slot per token position (``state_indices``
is ``[B, q]``); the request's history starts from the checkpoint of its last
accepted token (column ``num_accepted - 1``), and every token writes its own
checkpoint. These references take that slot-indexed pool layout directly, so
the runners can compare whole pools.

The convolution keeps one raw-history row per request (``state_indices[:, 0]``)
of width ``kernel_size - 2 + q``; the recurrence keeps FP32 ``[Hv, Dv, Dk]``
slot contents (vLLM's transposed state layout).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from profiling.runners.attention.gdn_recurrent_decode_reference import (
    gdn_recurrent_decode_reference,
)

__all__ = [
    "gdn_causal_conv_spec_decode_reference",
    "gdn_recurrent_spec_decode_reference",
    "gdn_recurrent_spec_decode_gated_norm_reference",
]


def _check_uniform(num_tokens: int, state_indices: torch.Tensor, num_accepted: torch.Tensor) -> int:
    if state_indices.ndim != 2:
        raise ValueError("state_indices must be [requests, tokens_per_request]")
    batch, per_request = state_indices.shape
    if num_tokens != batch * per_request:
        raise ValueError(
            f"{num_tokens} tokens do not fill {batch} requests of {per_request} tokens each"
        )
    if num_accepted.shape != (batch,):
        raise ValueError("num_accepted must hold one count per request")
    if torch.any(num_accepted < 1).item() or torch.any(num_accepted > per_request).item():
        raise ValueError("num_accepted must lie in [1, tokens_per_request]")
    if torch.any(state_indices <= 0).item():
        raise ValueError("state_indices must not name reserved slot zero")
    return per_request


def gdn_causal_conv_spec_decode_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    num_accepted: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Depthwise causal convolution + SiLU over ``q`` verify tokens per request.

    ``x`` is ``[B*q, C]`` (request-major), ``weight`` ``[C, W]``, ``state``
    ``[S, C, W - 2 + q]`` (dim-first), ``state_indices`` ``[B, q]`` int32 and
    ``num_accepted`` ``[B]`` int32. Request ``b`` reads history columns
    ``a-1 .. a+W-3`` of slot ``state_indices[b, 0]`` (``a`` its accepted
    count), convolves its ``q`` tokens, and rewrites that slot as history
    columns ``a .. a+W-3`` followed by its raw tokens. Returns the BF16 output
    and ``state`` (mutated in place).
    """
    per_request = _check_uniform(x.shape[0], state_indices, num_accepted)
    channels, width = weight.shape
    if state.shape[1:] != (channels, width - 2 + per_request):
        raise ValueError(f"state slots must be [{channels}, {width - 2 + per_request}]")
    output = torch.empty_like(x)
    weight_fp32 = weight.float()
    for request in range(state_indices.shape[0]):
        slot = int(state_indices[request, 0])
        accepted = int(num_accepted[request])
        tokens = x[request * per_request : (request + 1) * per_request]
        old = state[slot].clone()
        history = old[:, accepted - 1 : accepted - 1 + width - 1]
        sequence = torch.cat((history, tokens.transpose(0, 1)), dim=-1).float()
        for token in range(per_request):
            window = sequence[:, token : token + width]
            output[request * per_request + token] = F.silu(
                torch.sum(window * weight_fp32, dim=-1)
            ).to(x.dtype)
        state[slot] = torch.cat(
            (old[:, accepted : accepted + width - 2], tokens.transpose(0, 1)), dim=-1
        )
    return output, state


def gdn_recurrent_spec_decode_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    num_accepted: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gated delta-rule recurrence over ``q`` verify tokens per request.

    Q/K are ``[B*q, Hq, Dk]``, V ``[B*q, Hv, Dv]``, raw gates ``[B*q, Hv]``,
    ``state`` an FP32 slot pool ``[S, Hv, Dv, Dk]``. Each request starts from
    slot ``state_indices[b, a-1]`` and writes its state after token ``t`` to
    slot ``state_indices[b, t]``. Returns the BF16 ``[B*q, Hv, Dv]`` output and
    the pool (mutated in place).
    """
    per_request = _check_uniform(query.shape[0], state_indices, num_accepted)
    batch = state_indices.shape[0]
    rows = torch.arange(batch, device=state_indices.device)
    sources = state_indices[rows, num_accepted.to(torch.int64) - 1].to(torch.int64)
    running = state[sources].transpose(-1, -2).contiguous()
    output = torch.empty_like(value)

    def per_token(tensor: torch.Tensor, token: int) -> torch.Tensor:
        return tensor.view(batch, per_request, *tensor.shape[1:])[:, token].contiguous()

    for token in range(per_request):
        token_output, running = gdn_recurrent_decode_reference(
            per_token(query, token),
            per_token(key, token),
            per_token(value, token),
            per_token(a, token),
            per_token(b, token),
            A_log,
            dt_bias,
            running,
        )
        output.view(batch, per_request, *value.shape[1:])[:, token] = token_output
        state[state_indices[:, token].to(torch.int64)] = running.transpose(-1, -2)
    return output, state


def gdn_recurrent_spec_decode_gated_norm_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    num_accepted: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`gdn_recurrent_spec_decode_reference` followed by the output norm.

    The BF16 recurrent output of each value head is RMS-normalized in FP32,
    scaled by ``norm_weight`` and multiplied by ``sigmoid(output_gate)``
    (``output_gate`` is ``[B*q, Hv, Dv]``), then rounded to BF16.
    """
    core, state = gdn_recurrent_spec_decode_reference(
        query, key, value, a, b, A_log, dt_bias, state, state_indices, num_accepted
    )
    core_fp32 = core.float()
    rstd = torch.rsqrt(torch.mean(core_fp32 * core_fp32, dim=-1, keepdim=True) + norm_eps)
    gated = core_fp32 * rstd * norm_weight.float() * torch.sigmoid(output_gate.float())
    return gated.to(torch.bfloat16), state
