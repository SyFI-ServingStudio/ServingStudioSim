"""Independent value reference for the fused QK RMSNorm + partial NeoX RoPE + gate split."""

from typing import Any


def qk_rms_norm_rope_gate_reference(
    torch: Any,
    q_gate: Any,
    k: Any,
    q_weight: Any,
    k_weight: Any,
    cos_sin_cache: Any,
    positions: Any,
    eps: float,
    head_dim: int,
    rotary_dim: int,
) -> tuple[Any, Any, Any]:
    """Return contiguous (q, k, gate); ``positions`` is one row per token."""
    num_tokens = q_gate.shape[0]
    half = rotary_dim // 2
    cos = cos_sin_cache[positions, :half].float()[:, None, :]
    sin = cos_sin_cache[positions, half:rotary_dim].float()[:, None, :]

    def norm_rope(heads: Any, weight: Any) -> Any:
        values = heads.float()
        normed = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
        # The kernel stores the normalized head in the input dtype before rotating it.
        normed = (normed * weight.float()).to(heads.dtype).float()
        first, second = normed[..., :half], normed[..., half:rotary_dim]
        rotated = torch.cat(
            (first * cos - second * sin, second * cos + first * sin, normed[..., rotary_dim:]),
            dim=-1,
        )
        return rotated.to(heads.dtype).reshape(num_tokens, -1)

    q_and_gate = q_gate.reshape(num_tokens, -1, 2 * head_dim)
    q_out = norm_rope(q_and_gate[..., :head_dim], q_weight)
    k_out = norm_rope(k.reshape(num_tokens, -1, head_dim), k_weight)
    gate_out = q_and_gate[..., head_dim:].reshape(num_tokens, -1)
    return q_out, k_out, gate_out


__all__ = ["qk_rms_norm_rope_gate_reference"]
