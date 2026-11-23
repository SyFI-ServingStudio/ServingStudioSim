"""Torch reference for the fused expert-parallel MoE layer (``mega_moe``).

Independent of DeepGEMM: it takes dequantized FP32 tensors in checkpoint order
(``[gate; up]`` gate/up rows, no interleave) and computes, for every token,

    y = sum_k w_k · down_e( q8( clamped_swiglu(gate_up_e(x)) · w_k ) )
        + sum_s down_s( q8( clamped_swiglu(gate_up_s(x)) ) )

where ``q8`` is the MXFP8 round trip the kernel applies to the down GEMM's
input and the shared experts take weight 1. ``expert_partial`` evaluates only
the slots one rank owns, so the EP layer is the sum of every rank's partial
over all tokens; the runner all-reduces those partials.
"""

from __future__ import annotations

from typing import Any

from profiling.runners.moe.mxfp4_fused_moe_reference import (
    clamped_swiglu,
    quantize_dequantize_mxfp8,
)


def _expert(torch: Any, x: Any, gate_up: Any, down: Any, limit: float, scale: Any) -> Any:
    gate, up = (x @ gate_up.T).chunk(2, dim=-1)
    act = clamped_swiglu(torch, gate, up, limit) * scale
    return quantize_dequantize_mxfp8(torch, act) @ down.T


def expert_partial(
    torch: Any,
    x: Any,
    topk_idx: Any,
    topk_weights: Any,
    gate_up: Any,
    down: Any,
    first_slot: int,
    limit: float,
) -> Any:
    """Contribution of slots ``[first_slot, first_slot + len(gate_up))`` to every token.

    ``x`` [T, H] fp32, ``topk_idx`` [T, K] global slot ids, ``topk_weights``
    [T, K], ``gate_up`` [E_local, 2I, H], ``down`` [E_local, H, I].
    """

    out = torch.zeros_like(x)
    for local in range(gate_up.shape[0]):
        token, choice = (topk_idx == first_slot + local).nonzero(as_tuple=True)
        if token.numel() == 0:
            continue
        scale = topk_weights[token, choice].unsqueeze(-1)
        out.index_add_(
            0, token, _expert(torch, x[token], gate_up[local], down[local], limit, scale)
        )
    return out


def shared_experts(torch: Any, x: Any, gate_up: Any, down: Any, limit: float) -> Any:
    """Shared experts packed along the intermediate axis, as the kernel takes them:
    ``gate_up`` [2·S·I, H] (``[gate; up]``), ``down`` [H, S·I]."""

    return _expert(torch, x, gate_up, down, limit, 1.0)
