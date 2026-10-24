"""Independent value reference for head-major QK RMSNorm + full rotate-half RoPE."""

from typing import Any


def head_major_qk_norm_rope_reference(
    torch: Any,
    qkv: Any,
    q_weight: Any,
    k_weight: Any,
    cos: Any,
    sin: Any,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[Any, Any, Any]:
    """``qkv`` (R, (H + 2 Hkv) · D); ``cos``/``sin`` (R, D / 2) fp32.

    Returns ``(q, k, v)`` as ``(H, R, D)``, ``(Hkv, R, D)``, ``(Hkv, R, D)`` bf16;
    ``q`` is None when ``num_q_heads`` is 0.
    """
    rows = qkv.shape[0]
    half = head_dim // 2
    heads = qkv.float().reshape(rows, num_q_heads + 2 * num_kv_heads, head_dim).transpose(0, 1)

    def norm_rope(x: Any, weight: Any) -> Any:
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
        x = x.to(torch.bfloat16).float()
        a, b = x[..., :half], x[..., half:]
        return torch.cat((a * cos - b * sin, b * cos + a * sin), -1).to(torch.bfloat16)

    q = norm_rope(heads[:num_q_heads], q_weight) if num_q_heads else None
    k = norm_rope(heads[num_q_heads : num_q_heads + num_kv_heads], k_weight)
    v = heads[num_q_heads + num_kv_heads :].to(torch.bfloat16)
    return q, k, v
