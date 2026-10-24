"""Independent value reference for the gated attention's output gate."""

from typing import Any


def attn_output_gate_reference(torch: Any, out: Any, q_gate: Any, head_dim: int) -> Any:
    """``out`` (tokens, heads, head_dim), ``q_gate`` (tokens, heads * 2 * head_dim).

    Returns (tokens, heads * head_dim).
    """
    tokens, heads = out.shape[0], out.shape[1]
    gate = q_gate.reshape(tokens, heads, 2 * head_dim)[..., head_dim:]
    return (out.float() * torch.sigmoid(gate.float())).to(out.dtype).reshape(tokens, -1)


__all__ = ["attn_output_gate_reference"]
