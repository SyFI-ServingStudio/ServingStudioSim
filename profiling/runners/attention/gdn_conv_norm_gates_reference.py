"""Torch semantics for ``gdn_conv_norm_gates``: a Gated DeltaNet layer's causal
convolution, SiLU, q/k normalization and gates over per-token windows.

Written from the model definition (Hugging Face Qwen3-Next / Qwen3.5
``torch_causal_conv1d_update`` + ``l2norm`` + the gate formulas), not from the
TensorFold kernel: each token convolves its own ``kernel_size`` rows (oldest
first, the token itself last), so a prompt's previous rows and a draft tree's
ancestors are the same contract.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["gdn_conv_norm_gates_reference"]

# Hugging Face ``l2norm``: x * rsqrt(sum(x^2) + eps).
_L2NORM_EPSILON = 1e-6
_SOFTPLUS_THRESHOLD = 20.0


def gdn_conv_norm_gates_reference(
    qkv: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    windows: torch.Tensor,
    stream_ids: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    num_qk_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(q, k, v, g, beta)`` for the rows of ``windows``.

    - ``qkv`` ``(T, C)`` bf16: the launch's packed q|k|v projection rows,
      ``C = 2·num_qk_heads·key_head_dim + num_value_heads·value_head_dim``.
    - ``conv_state`` ``(S·keep, C)`` bf16, ``keep = kernel_size − 1``: the
      ``keep`` rows before this launch of each of ``S`` streams.
    - ``conv_weight`` ``(C, kernel_size)``: depthwise filter, tap ``j`` applied
      to window row ``j`` (oldest first).
    - ``windows`` ``(R, kernel_size)`` int: row ``r``'s window, each entry an
      index into ``[stream state rows (0..keep); qkv rows (keep..)]``.
    - ``stream_ids`` ``(R,)`` int: the stream whose state a ``< keep`` entry reads.
    - ``a``, ``b`` ``(R, num_value_heads)`` bf16 raw gates; ``A_log``,
      ``dt_bias`` ``(num_value_heads,)`` fp32.

    The convolution and SiLU run in fp32 and round to bf16 (the activation
    dtype the convolution writes); q and k are L2-normalized in fp32 (q also
    scaled by ``key_head_dim ** -0.5``) and rounded to bf16; v is the bf16
    convolution output; ``g = exp(−exp(A_log)·softplus(a + dt_bias))`` (the
    decay itself, not its log) and ``beta = sigmoid(b)`` stay fp32.
    """
    if key_head_dim != value_head_dim:
        raise ValueError("the reference packs q, k and v heads of one width")
    kernel_size = conv_weight.shape[1]
    keep = kernel_size - 1
    width = 2 * num_qk_heads * key_head_dim + num_value_heads * value_head_dim
    if qkv.shape[1] != width or conv_state.shape[1] != width or conv_weight.shape[0] != width:
        raise ValueError(f"packed width must be {width}")
    rows = windows.shape[0]

    src = windows.long()
    from_state = src < keep
    state_rows = stream_ids.long()[:, None] * keep + src.clamp(max=keep - 1)
    qkv_rows = (src - keep).clamp(min=0)
    x = torch.where(
        from_state[..., None],
        conv_state[state_rows.clamp(min=0)].float(),
        qkv[qkv_rows].float(),
    )                                                       # (R, kernel_size, C)
    conv = (x * conv_weight.float().t()[None]).sum(dim=1)   # (R, C)
    act = F.silu(conv).to(torch.bfloat16)

    q_width = num_qk_heads * key_head_dim
    q_raw, k_raw, v_raw = torch.split(act, [q_width, q_width, num_value_heads * value_head_dim], dim=-1)

    def l2norm(t: torch.Tensor) -> torch.Tensor:
        t = t.reshape(rows, num_qk_heads, key_head_dim).float()
        return t * torch.rsqrt(t.square().sum(dim=-1, keepdim=True) + _L2NORM_EPSILON)

    q = (l2norm(q_raw) * key_head_dim**-0.5).to(torch.bfloat16)
    k = l2norm(k_raw).to(torch.bfloat16)
    v = v_raw.reshape(rows, num_value_heads, value_head_dim).contiguous()
    softplus = F.softplus(a.float() + dt_bias.float(), beta=1.0, threshold=_SOFTPLUS_THRESHOLD)
    g = torch.exp(-torch.exp(A_log.float()) * softplus)
    beta = torch.sigmoid(b.float())
    return q, k, v, g, beta
