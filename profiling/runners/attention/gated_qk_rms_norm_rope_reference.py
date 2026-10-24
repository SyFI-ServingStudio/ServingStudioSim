"""Independent value reference for gated QK RMSNorm + partial NeoX RoPE (gate left in place)."""

from typing import Any


def rotary_axes(rotary_dim: int, section: tuple[int, int, int] | None) -> list[int]:
    """Position row each rotated pair reads: 0 for text, interleaved MRoPE sections otherwise.

    Pair ``i`` reads the height row when ``i % 3 == 1`` and ``i < 3 * section[1]``,
    the width row when ``i % 3 == 2`` and ``i < 3 * section[2]``, else the
    temporal row (Qwen3.5's interleaved multimodal RoPE).
    """
    half = rotary_dim // 2
    if section is None:
        return [0] * half
    return [
        1 if i % 3 == 1 and i < 3 * section[1] else 2 if i % 3 == 2 and i < 3 * section[2] else 0
        for i in range(half)
    ]


def gated_qk_rms_norm_rope_reference(
    torch: Any,
    q_gate: Any,
    k: Any,
    q_weight: Any,
    k_weight: Any,
    positions: Any,
    inv_freq: Any,
    eps: float,
    head_dim: int,
    section: tuple[int, int, int] | None = None,
) -> tuple[Any, Any]:
    """Return contiguous ``(q, k)`` as ``(tokens, heads, head_dim)``.

    ``q_gate`` is ``(tokens, heads * 2 * head_dim)`` with each head's query
    before its gate; ``positions`` is ``(tokens,)`` or, with ``section``,
    ``(3, tokens)``. Angles are fp32 ``position * inv_freq``.
    """
    num_tokens = q_gate.shape[0]
    half = inv_freq.numel()
    rotary_dim = 2 * half
    axes = torch.tensor(rotary_axes(rotary_dim, section), device=q_gate.device)
    pos = positions.float()
    if section is None:
        angle = pos[:, None] * inv_freq.float()[None, :]
    else:
        angle = pos[axes, :].T * inv_freq.float()[None, :]
    cos, sin = angle.cos()[:, None, :], angle.sin()[:, None, :]

    def norm_rope(heads: Any, weight: Any) -> Any:
        values = heads.float()
        normed = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
        normed = (normed * weight.float()).to(torch.bfloat16).float()
        first, second = normed[..., :half], normed[..., half:rotary_dim]
        rotated = torch.cat(
            (first * cos - second * sin, second * cos + first * sin, normed[..., rotary_dim:]),
            dim=-1,
        )
        return rotated.to(torch.bfloat16)

    q_rows = q_gate.reshape(num_tokens, -1, 2 * head_dim)[..., :head_dim]
    return norm_rope(q_rows, q_weight), norm_rope(k.reshape(num_tokens, -1, head_dim), k_weight)


__all__ = ["gated_qk_rms_norm_rope_reference", "rotary_axes"]
